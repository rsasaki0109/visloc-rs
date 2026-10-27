//! Offline mapper NFR factors recovered from versioned Basalt MargData.
//!
//! The M8a/M8b reduction and Jacobians follow Basalt's BSD-3-Clause sources at
//! commit `0f3b2b52c807f70ff4e2973ce253c73329eea7bc` (`nfr_mapper.cpp`,
//! `marg_helper.cpp`, and `utils/nfr.h`).  See `benchmarks/basalt/` for the
//! pinned fixture and provenance report.
pub mod features;
pub(crate) mod imu_factor;
pub use imu_factor::{
    imu_preintegration_relative_pose_factor, refine_pair_velocities, VelocityPair,
};
pub mod imu_ba;
pub mod localization_map;
pub mod session;
pub mod triangulation;

pub use triangulation::{
    canonical_setup_opt_bytes, canonical_setup_opt_hash, setup_opt, setup_opt_from_corners,
    triangulate_ba, triangulate_pair, MapperLandmark, MapperObservation, SetupOptCandidate,
    SetupOptInput, SetupOptRejectReason, SetupOptReport, SetupOptResult, SetupOptTrackReport,
};

pub use features::{
    compute_angles, compute_descriptors, compute_essential, compute_hash_bow,
    detect_keypoints_mapping, extract_mapper_features, hash_descriptor, match_stereo_features,
    match_temporal_ransac, match_temporal_ransac_seeded, match_temporal_stage,
    query_bow_candidates, unproject_rays, BowEntry, BowQueryCandidate, FeaturePipelineError,
    MapperImageFeatures, MapperImageId, StereoFeatureMatch, TemporalMatchStage,
    TemporalRansacResult,
};
pub use localization_map::{
    fit_opencv_fisheye, localization_camera_id, orb_descriptor_to_f32, FisheyeFit,
    LocalizationMapError, LocalizationMapExport, LocalizationMapImage,
};
pub use session::{
    NfrMapper, NfrMapperCurrentPoints, NfrMapperDetectionReport, NfrMapperError,
    NfrMapperFeatureError, NfrMapperFilterReport, NfrMapperHeadlessConfig, NfrMapperHeadlessError,
    NfrMapperHeadlessReport, NfrMapperHeadlessStage, NfrMapperImageRecord, NfrMapperIngestReport,
    NfrMapperLandmarkDb, NfrMapperLandmarkRecord, NfrMapperMatchAllReport, NfrMapperMatchData,
    NfrMapperOptimizeError, NfrMapperOptimizeReport, NfrMapperPoseRecord, NfrMapperResult,
    NfrMapperSetupOptError, NfrMapperStereoError, NfrMapperStereoReport, NfrMapperTrackReport,
};

/// The image identity used by Basalt's mapper track graph.
///
/// Upstream calls this type `TimeCamId`: a frame/timestamp alone is not an
/// image identity because every stereo frame contributes one node for each
/// camera.  This is intentionally a separate type from the M8c frontend's
/// [`MapperImageId`] so the track API cannot accidentally erase the camera
/// stream.  Use [`TimeCamId::new`] at the boundary between the two APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TimeCamId {
    pub frame_id: u64,
    pub cam_id: u16,
}

impl TimeCamId {
    pub const fn new(frame_id: u64, cam_id: u16) -> Self {
        Self { frame_id, cam_id }
    }
}

impl From<MapperImageId> for TimeCamId {
    fn from(value: MapperImageId) -> Self {
        Self::new(value.frame_id, u16::from(value.cam_id))
    }
}

impl From<TimeCamId> for MapperImageId {
    fn from(value: TimeCamId) -> Self {
        Self {
            frame_id: value.frame_id,
            cam_id: u8::try_from(value.cam_id).expect("mapper camera id must fit u8"),
        }
    }
}

/// Feature index in one mapper image.  The pinned C++ type is a signed
/// `int`; mapper feature indices are non-negative, so the Rust boundary uses
/// the existing public unsigned descriptor-ID contract.
pub type FeatureId = u64;

/// A node in the upstream track graph: `(TimeCamId, FeatureId)`.
pub type ImageFeaturePair = (TimeCamId, FeatureId);

/// Explicit image-pair key used by [`TrackBuilder`].
pub type ImagePair = (TimeCamId, TimeCamId);

/// Match data consumed by the track builder.
///
/// The upstream builder deliberately consumes only `MatchData::inliers`.
/// Raw descriptor matches, geometric models, and scores are not part of the
/// graph contract and therefore cannot silently create a track edge here.
/// The stateful [`session::NfrMapper`] keeps the full accepted-pair payload
/// (raw pairs plus `T_i_j`) alongside this projection for callers that need
/// the source `MatchData` lifecycle state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchData {
    pub inliers: Vec<(FeatureId, FeatureId)>,
}

impl MatchData {
    pub const fn new(inliers: Vec<(FeatureId, FeatureId)>) -> Self {
        Self { inliers }
    }

    pub fn from_inliers<I>(inliers: I) -> Self
    where
        I: IntoIterator<Item = (FeatureId, FeatureId)>,
    {
        Self::new(inliers.into_iter().collect())
    }
}

/// Deterministic Rust spelling of upstream `Matches`.
pub type Matches = BTreeMap<ImagePair, MatchData>;

/// One exported upstream feature track, keyed by image identity.
pub type FeatureTrack = BTreeMap<TimeCamId, FeatureId>;

/// Exported tracks retain the union-find root as their key.  In particular,
/// this map is *not* renumbered after filtering.
pub type FeatureTracks = BTreeMap<u64, FeatureTrack>;

use crate::vio::margdata::{AomBlockData, FramePoseData, MargData, MatrixData, OfObservationData};
use nalgebra::{
    DMatrix, DVector, Matrix2, Matrix3, Matrix6, Point2, Point3, UnitQuaternion, Vector2, Vector3,
    Vector6,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::AddAssign;
use visloc_core::geometry::SE3;

use crate::calibration::BasaltCalibration;
use crate::camera::DoubleSphereCamera;

const POSE_DOF: usize = 6;
const NAV_STATE_DOF: usize = 15;

/// Errors reported while reducing an upstream mixed `AbsOrderMap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MargDataProcessError {
    InvalidMatrix,
    InvalidAomOrder,
    UnknownBlockSize,
    MissingKeyframeState,
    SingularMarginalBlock,
}

/// Failure modes of the M8b covariance/factor recovery stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NfrExtractionError {
    InvalidMatrix,
    RankDeficient,
    MissingKeyframe,
    MissingPose,
    MissingAomBlock,
    NonFinite,
    SingularCovariance,
}

/// Bookkeeping from the M8a `processMargData` operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MargDataProcessReport {
    pub input_size: usize,
    pub output_size: usize,
    pub input_rank: usize,
    pub output_rank: usize,
    pub kept_columns: Vec<usize>,
    pub marginalized_columns: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelativePoseFactor {
    pub from: u64,
    pub to: u64,
    pub translation: [f64; 3],
    pub rotation: [f64; 4],
    pub information: Vec<f64>,
    pub weight: f64,
}
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RollPitchFactor {
    pub frame_id: u64,
    pub roll: f64,
    pub pitch: f64,
    pub information: [f64; 4],
    pub weight: f64,
    /// The complete upstream measurement rotation.  The old Rust scaffold
    /// retained only Euler roll/pitch, but `NfrMapper` stores
    /// `R_w_i_meas` (including yaw) and uses it in the gravity residual.
    /// Keeping the matrix is what makes the factor linearization literal.
    #[serde(default = "identity_rotation_array")]
    pub measured_rotation: [f64; 9],
}
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BaCovisibilityFactor {
    pub frame_a: u64,
    pub frame_b: u64,
    pub shared_tracks: u32,
    pub information: f64,
    pub weight: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapperFactors {
    pub provenance_version: String,
    pub relative_pose: Vec<RelativePoseFactor>,
    // Omitted when empty so default-off runs serialize byte-identically to
    // builds without the opt-in IMU factors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imu_relative_pose: Vec<RelativePoseFactor>,
    pub roll_pitch: Vec<RollPitchFactor>,
    pub ba_covisibility: Vec<BaCovisibilityFactor>,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapperConfig {
    pub enabled: bool,
    pub relative_pose_weight: f64,
    pub roll_pitch_weight: f64,
    pub ba_weight: f64,
    /// Relative-pose factors per marginalized keyframe, to the keyframes
    /// nearest in time; 0 keeps upstream's star to every keyframe in the
    /// window.
    pub relative_pose_max_neighbors: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OfflineMapperConfig {
    pub max_points: usize,
    pub max_hamming: u32,
    pub second_best_ratio: f64,
    pub bow_bits: u8,
    /// Number of HashBoW results requested for each temporal image query.
    /// This is the source `mapper_num_frames_to_match` setting; the legacy
    /// `match_window` field remains the same value for existing M8c callers.
    pub match_window: u64,
    /// Strict HashBoW score gate used by `NfrMapper::match_all`.
    pub frames_to_match_threshold: f64,
    pub min_matches: usize,
    pub ransac_threshold: f64,
    pub min_track_length: usize,
    pub min_triangulation_distance: f64,
}
impl Default for OfflineMapperConfig {
    fn default() -> Self {
        Self {
            max_points: 800,
            max_hamming: 70,
            second_best_ratio: 1.2,
            bow_bits: 16,
            match_window: 30,
            frames_to_match_threshold: 0.04,
            min_matches: 20,
            ransac_threshold: 5e-5,
            min_track_length: 5,
            min_triangulation_distance: 0.07,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub id: u64,
    pub x: f64,
    pub y: f64,
    pub descriptor: [u8; 32],
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorMatch {
    pub left: u64,
    pub right: u64,
    pub distance: u32,
}
/// Legacy synthetic-input helper retained for the existing NFR test/API.
///
/// The M8c image frontend never calls this ID-ordered truncation path; use
/// [`detect_keypoints_mapping`] for raw
/// mapper images.
pub fn detect_keypoints(mut points: Vec<Keypoint>, config: OfflineMapperConfig) -> Vec<Keypoint> {
    points.sort_by_key(|p| p.id);
    points.truncate(config.max_points);
    points
}
pub fn hamming_distance(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}
pub fn mutual_hamming_matches(
    left: &[Keypoint],
    right: &[Keypoint],
    config: OfflineMapperConfig,
) -> Vec<DescriptorMatch> {
    let left_descriptors = left
        .iter()
        .map(|point| point.descriptor)
        .collect::<Vec<_>>();
    let right_descriptors = right
        .iter()
        .map(|point| point.descriptor)
        .collect::<Vec<_>>();
    features::mutual_descriptor_matches(&left_descriptors, &right_descriptors, config)
        .into_iter()
        .map(|mut match_| {
            match_.left = left[match_.left as usize].id;
            match_.right = right[match_.right as usize].id;
            match_
        })
        .collect()
}
pub fn bow_hash16(descriptor: &[u8; 32], bits: u8) -> u16 {
    hash_descriptor(descriptor, bits.min(16)) as u16
}
pub fn bow_candidates(
    left: &[Keypoint],
    right: &[Keypoint],
    config: OfflineMapperConfig,
) -> Vec<(u64, u64)> {
    left.iter()
        .flat_map(|l| {
            right
                .iter()
                .filter(move |r| {
                    bow_hash16(&l.descriptor, config.bow_bits)
                        == bow_hash16(&r.descriptor, config.bow_bits)
                })
                .map(move |r| (l.id, r.id))
        })
        .collect()
}
/// Legacy synthetic-input gate retained for API compatibility.
///
/// It is deliberately not part of the M8c path: essential filtering there is
/// performed by [`features::match_stereo_features`], using calibrated rays
/// and the known stereo transform.  This helper has no ray/pose inputs and
/// therefore cannot claim upstream geometric parity.
pub fn essential_ransac_gate(
    matches: &[DescriptorMatch],
    config: OfflineMapperConfig,
) -> Vec<DescriptorMatch> {
    matches
        .iter()
        .copied()
        .filter(|m| ((m.left as f64 - m.right as f64).abs() * 1e-6) <= config.ransac_threshold)
        .collect()
}

/// Union/find implementation used by the literal `tracks.h` port.
///
/// The public field names intentionally mirror upstream.  `find` performs
/// path compression and `union` uses the same rank/tie rule as the pinned
/// implementation: when ranks are equal, the first root remains the parent
/// and its rank is incremented.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnionFind {
    pub m_cc_parent: Vec<u32>,
    pub m_cc_rank: Vec<u32>,
    pub m_cc_size: Vec<u32>,
}

impl UnionFind {
    pub const INVALID_INDEX: u32 = u32::MAX;

    pub const fn invalid_index() -> u32 {
        Self::INVALID_INDEX
    }

    pub fn init_sets(&mut self, num_cc: usize) {
        let num_cc = u32::try_from(num_cc).expect("track graph cannot exceed u32 nodes");
        self.m_cc_size = vec![1; usize::try_from(num_cc).expect("u32 fits usize")];
        self.m_cc_parent = (0..num_cc).collect();
        self.m_cc_rank = vec![0; usize::try_from(num_cc).expect("u32 fits usize")];
    }

    pub fn get_num_nodes(&self) -> usize {
        self.m_cc_size.len()
    }

    pub fn find(&mut self, i: u32) -> u32 {
        let index = usize::try_from(i).expect("union-find index must fit usize");
        let parent = self.m_cc_parent[index];
        if parent != i && parent != Self::INVALID_INDEX {
            let root = self.find(parent);
            self.m_cc_parent[index] = root;
        }
        self.m_cc_parent[index]
    }

    pub fn union(&mut self, i: u32, j: u32) {
        let i = self.find(i);
        let j = self.find(j);
        if i == j {
            return;
        }

        let i_index = usize::try_from(i).expect("union-find index must fit usize");
        let j_index = usize::try_from(j).expect("union-find index must fit usize");
        if self.m_cc_rank[i_index] < self.m_cc_rank[j_index] {
            self.m_cc_parent[i_index] = j;
            self.m_cc_size[j_index] += self.m_cc_size[i_index];
        } else {
            self.m_cc_parent[j_index] = i;
            self.m_cc_size[i_index] += self.m_cc_size[j_index];
            if self.m_cc_rank[i_index] == self.m_cc_rank[j_index] {
                self.m_cc_rank[i_index] += 1;
            }
        }
    }

    /// Compatibility spelling from `union_find.h`.
    #[allow(non_snake_case)]
    pub fn InitSets(&mut self, num_cc: u32) {
        self.init_sets(usize::try_from(num_cc).expect("u32 fits usize"));
    }

    /// Compatibility spelling from `union_find.h`.
    #[allow(non_snake_case)]
    pub fn GetNumNodes(&self) -> usize {
        self.get_num_nodes()
    }

    /// Compatibility spelling from `union_find.h`.
    #[allow(non_snake_case)]
    pub fn Find(&mut self, i: u32) -> u32 {
        self.find(i)
    }

    /// Compatibility spelling from `union_find.h`.
    #[allow(non_snake_case)]
    pub fn Union(&mut self, i: u32, j: u32) {
        self.union(i, j)
    }
}

/// Statistics emitted by [`TrackBuilder::filter`].  Root IDs are represented
/// as `u64` at the public Rust boundary, but are the exact zero-based UF node
/// indices used by upstream `TrackId` keys.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackFilterReport {
    pub minimum_track_length: usize,
    pub node_count: usize,
    pub component_count_before: usize,
    pub component_count_after: usize,
    pub conflict_track_ids: Vec<u64>,
    pub short_track_ids: Vec<u64>,
    pub rejected_track_ids: Vec<u64>,
    pub rejected_conflict_ids: Vec<u64>,
    pub rejected_short_ids: Vec<u64>,
    pub component_lengths: BTreeMap<u64, usize>,
    pub track_length_histogram: BTreeMap<usize, usize>,
}

impl TrackFilterReport {
    pub const fn exported_track_count(&self) -> usize {
        self.component_count_after
    }
}

/// One track with observations that retain the full upstream image identity.
///
/// `observations` is a vector for compatibility with the old synthetic API;
/// production M8d code should consume [`FeatureTracks`] from
/// [`TrackBuilder::export`], whose per-track `BTreeMap` enforces one feature
/// per `TimeCamId`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Track {
    pub id: u64,
    pub observations: Vec<(TimeCamId, FeatureId)>,
}

/// Faithful port of upstream `TrackBuilder` from `utils/tracks.h`.
///
/// Nodes are inserted from a deterministic `BTreeSet`, unions are performed
/// only for `MatchData::inliers` under explicit `(TimeCamId, TimeCamId)` keys,
/// and exported keys remain the UF roots.  No post-export renumbering or
/// camera-ID erasure is performed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TrackBuilder {
    pub map_node_to_index: BTreeMap<ImageFeaturePair, u32>,
    pub uf_tree: UnionFind,
    pub rejected_track_ids: BTreeSet<u32>,
    last_filter_report: Option<TrackFilterReport>,
}

impl TrackBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the deterministic node map and union all inlier correspondences.
    pub fn build(&mut self, map_pair_wise_matches: &Matches) {
        self.map_node_to_index.clear();
        self.rejected_track_ids.clear();
        self.last_filter_report = None;

        let mut all_features = BTreeSet::<ImageFeaturePair>::new();
        for ((image_i, image_j), match_data) in map_pair_wise_matches {
            for &(feature_i, feature_j) in &match_data.inliers {
                all_features.insert((*image_i, feature_i));
                all_features.insert((*image_j, feature_j));
            }
        }

        for (index, feature) in all_features.into_iter().enumerate() {
            let index = u32::try_from(index).expect("track graph cannot exceed u32 nodes");
            self.map_node_to_index.insert(feature, index);
        }
        self.uf_tree.init_sets(self.map_node_to_index.len());

        // BTreeMap iteration is the deterministic equivalent of the explicit
        // image-pair traversal.  MatchData::matches (when available at an
        // upstream boundary) is intentionally not consulted here: only
        // geometric inliers form track edges.
        for ((image_i, image_j), match_data) in map_pair_wise_matches {
            for &(feature_i, feature_j) in &match_data.inliers {
                let node_i = self.map_node_to_index[&(*image_i, feature_i)];
                let node_j = self.map_node_to_index[&(*image_j, feature_j)];
                self.uf_tree.union(node_i, node_j);
            }
        }
    }

    /// Compatibility spelling from `tracks.h`.
    #[allow(non_snake_case)]
    pub fn Build(&mut self, map_pair_wise_matches: &Matches) {
        self.build(map_pair_wise_matches);
    }

    /// Filter complete components that contain a duplicate image or are too
    /// short.  The rejected component's every parent entry is set to the
    /// upstream invalid sentinel; no individual observation survives.
    pub fn filter(&mut self, minimum_track_length: usize) -> TrackFilterReport {
        let node_count = self.map_node_to_index.len();
        let mut tracks = BTreeMap::<u32, BTreeSet<TimeCamId>>::new();
        let mut conflict_ids = BTreeSet::<u32>::new();

        // This traversal is in map order, matching `std::map` and forcing
        // path compression before the invalid-root sweep below.
        for (feature, &node_index) in &self.map_node_to_index {
            let track_id = self.uf_tree.find(node_index);
            if conflict_ids.contains(&track_id) {
                continue;
            }
            let image_id = feature.0;
            let image_set = tracks.entry(track_id).or_default();
            if !image_set.insert(image_id) {
                conflict_ids.insert(track_id);
            }
        }

        let component_lengths = tracks
            .iter()
            .map(|(&root, images)| (u64::from(root), images.len()))
            .collect::<BTreeMap<_, _>>();
        let component_count_before = component_lengths.len();

        let mut short_ids = BTreeSet::<u32>::new();
        for (&root, images) in &tracks {
            if images.len() < minimum_track_length {
                short_ids.insert(root);
            }
        }

        let mut rejected = conflict_ids.clone();
        rejected.extend(short_ids.iter().copied());
        self.rejected_track_ids = rejected.clone();

        // `Find` above compressed all nodes, so parent entries can be marked
        // by root exactly as in upstream's invalid-root loop.
        for parent in &mut self.uf_tree.m_cc_parent {
            if rejected.contains(parent) {
                let root = usize::try_from(*parent).expect("u32 fits usize");
                self.uf_tree.m_cc_size[root] = 1;
                *parent = UnionFind::INVALID_INDEX;
            }
        }

        let mut track_length_histogram = BTreeMap::<usize, usize>::new();
        for (&root, images) in &tracks {
            if !rejected.contains(&root) {
                *track_length_histogram.entry(images.len()).or_default() += 1;
            }
        }
        let component_count_after = track_length_histogram.values().sum();
        let conflict_track_ids = conflict_ids
            .iter()
            .map(|&root| u64::from(root))
            .collect::<Vec<_>>();
        let short_track_ids = short_ids
            .iter()
            .map(|&root| u64::from(root))
            .collect::<Vec<_>>();
        let rejected_track_ids = rejected
            .iter()
            .map(|&root| u64::from(root))
            .collect::<Vec<_>>();

        let report = TrackFilterReport {
            minimum_track_length,
            node_count,
            component_count_before,
            component_count_after,
            conflict_track_ids: conflict_track_ids.clone(),
            short_track_ids: short_track_ids.clone(),
            rejected_track_ids,
            rejected_conflict_ids: conflict_track_ids,
            rejected_short_ids: short_track_ids,
            component_lengths,
            track_length_histogram,
        };
        self.last_filter_report = Some(report.clone());
        report
    }

    /// Compatibility spelling from `tracks.h` (upstream always returns
    /// `false`; callers inspect `TrackCount`/`Export` for the result).
    #[allow(non_snake_case)]
    pub fn Filter(&mut self, minimum_track_length: usize) -> bool {
        let _ = self.filter(minimum_track_length);
        false
    }

    /// Return the number of parent IDs in the UF forest, matching upstream's
    /// `TrackCount` implementation.  In particular this intentionally does
    /// not call `Find`: before filtering, upstream reports the distinct stored
    /// parent entries rather than compressing the forest first.  After
    /// filtering every rejected node carries `InvalidIndex`, which is removed
    /// here exactly as in `tracks.h`.
    pub fn track_count(&self) -> usize {
        self.uf_tree
            .m_cc_parent
            .iter()
            .copied()
            .filter(|&parent| parent != UnionFind::INVALID_INDEX)
            .collect::<BTreeSet<_>>()
            .len()
    }

    #[allow(non_snake_case)]
    pub fn TrackCount(&self) -> usize {
        self.track_count()
    }

    pub fn node_count(&self) -> usize {
        self.map_node_to_index.len()
    }

    pub const fn last_filter_report(&self) -> Option<&TrackFilterReport> {
        self.last_filter_report.as_ref()
    }

    /// Export tracks keyed by their un-renumbered upstream UF root.
    pub fn export(&mut self) -> FeatureTracks {
        let mut tracks = FeatureTracks::new();
        for (feature, &node_index) in &self.map_node_to_index {
            let track_id = self.uf_tree.find(node_index);
            if track_id != UnionFind::INVALID_INDEX {
                tracks
                    .entry(u64::from(track_id))
                    .or_default()
                    .insert(feature.0, feature.1);
            }
        }
        tracks
    }

    /// Compatibility spelling from `tracks.h`.
    #[allow(non_snake_case)]
    pub fn Export(&mut self, tracks: &mut FeatureTracks) {
        *tracks = self.export();
    }

    /// Export as the legacy vector wrapper while retaining full `TimeCamId`
    /// observations.  Root IDs and observation order remain canonical.
    pub fn export_vec(&mut self) -> Vec<Track> {
        self.export()
            .into_iter()
            .map(|(id, observations)| Track {
                id,
                observations: observations.into_iter().collect(),
            })
            .collect()
    }
}

/// Compact, serializable summary used by the checked-in M8d oracle fixture.
/// The full canonical observations are retained so a digest mismatch can be
/// diagnosed without rerunning the upstream executable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackOracleSummary {
    pub node_count: usize,
    pub component_count_before: usize,
    pub component_count_after: usize,
    pub rejected_conflict_ids: Vec<u64>,
    pub rejected_short_ids: Vec<u64>,
    pub rejected_track_ids: Vec<u64>,
    pub exported_track_count: usize,
    pub track_length_histogram: BTreeMap<usize, usize>,
    pub canonical_observations: Vec<Track>,
    pub canonical_hash: u64,
}

/// Build tracks and return the exact values captured by an M8d oracle run.
pub fn build_track_oracle(
    map_pair_wise_matches: &Matches,
    minimum_track_length: usize,
) -> TrackOracleSummary {
    let mut builder = TrackBuilder::new();
    builder.build(map_pair_wise_matches);
    let report = builder.filter(minimum_track_length);
    let exported = builder.export();
    let canonical_observations = exported
        .iter()
        .map(|(&id, observations)| Track {
            id,
            observations: observations
                .iter()
                .map(|(&image, &feature)| (image, feature))
                .collect(),
        })
        .collect::<Vec<_>>();
    TrackOracleSummary {
        node_count: report.node_count,
        component_count_before: report.component_count_before,
        component_count_after: report.component_count_after,
        rejected_conflict_ids: report.rejected_conflict_ids,
        rejected_short_ids: report.rejected_short_ids,
        rejected_track_ids: report.rejected_track_ids,
        exported_track_count: exported.len(),
        track_length_histogram: report.track_length_histogram,
        canonical_hash: canonical_observation_hash(&exported),
        canonical_observations,
    }
}

/// Canonical observation bytes used by the M8d oracle fixture.  Records are
/// encoded in root/image/feature order and use little-endian fixed-width
/// integers, making the digest independent of hash-map iteration order.
pub fn canonical_observation_bytes(tracks: &FeatureTracks) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (&track_id, observations) in tracks {
        bytes.extend_from_slice(&track_id.to_le_bytes());
        bytes.extend_from_slice(&(observations.len() as u64).to_le_bytes());
        for (image_id, &feature_id) in observations {
            bytes.extend_from_slice(&image_id.frame_id.to_le_bytes());
            bytes.extend_from_slice(&u64::from(image_id.cam_id).to_le_bytes());
            bytes.extend_from_slice(&feature_id.to_le_bytes());
        }
    }
    bytes
}

/// FNV-1a digest of [`canonical_observation_bytes`].  The M8d fixture stores
/// this compact digest alongside the full canonical observations; consumers
/// needing SHA-256 can hash the returned bytes without changing the ordering
/// contract.
pub fn canonical_observation_hash(tracks: &FeatureTracks) -> u64 {
    canonical_observation_bytes(tracks)
        .into_iter()
        .fold(1469598103934665603u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
        })
}

/// Legacy synthetic-input helper retained solely for old NFR tests.  It does
/// not model explicit image-pair keys, camera IDs, or upstream inlier
/// semantics and is therefore non-production; M8d callers must use
/// [`TrackBuilder`].
pub fn union_find_tracks(
    frame_pairs: &[(u64, Vec<DescriptorMatch>)],
    config: OfflineMapperConfig,
) -> Vec<Track> {
    let mut sets: Vec<Vec<(TimeCamId, FeatureId)>> = Vec::new();
    for (frame, matches) in frame_pairs {
        for match_ in matches {
            let mut found = None;
            for (index, set) in sets.iter().enumerate() {
                if set.iter().any(|(_, feature_id)| {
                    *feature_id == match_.left || *feature_id == match_.right
                }) {
                    found = Some(index);
                    break;
                }
            }
            if let Some(index) = found {
                sets[index].push((TimeCamId::new(*frame, 0), match_.right));
            } else {
                sets.push(vec![
                    (TimeCamId::new(*frame, 0), match_.left),
                    (TimeCamId::new(*frame, 0), match_.right),
                ]);
            }
        }
    }
    sets.into_iter()
        .enumerate()
        .filter(|(_, observations)| observations.len() >= config.min_track_length)
        .map(|(id, observations)| Track {
            id: id as u64,
            observations,
        })
        .collect()
}
pub fn triangulation_gate(track: &Track, baseline: f64, config: OfflineMapperConfig) -> bool {
    track.observations.len() >= config.min_track_length
        && baseline >= config.min_triangulation_distance
}

#[cfg(test)]
mod track_builder_tests {
    use super::*;

    fn tc(frame_id: u64, cam_id: u16) -> TimeCamId {
        TimeCamId::new(frame_id, cam_id)
    }

    #[test]
    fn union_find_uses_rank_and_path_compression() {
        let mut uf = UnionFind::default();
        uf.init_sets(8);

        // Build two height-two trees and merge them.  Equal-rank ties must
        // retain the first root, as in union_find.h.
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(0, 2);
        uf.union(4, 5);
        uf.union(6, 7);
        uf.union(4, 6);
        uf.union(0, 4);

        assert_eq!(uf.m_cc_parent[0], 0);
        assert_eq!(uf.m_cc_rank[0], 3);
        assert_eq!(uf.find(7), 0);
        assert_eq!(uf.m_cc_parent[7], 0, "Find must compress the path");
        for index in 0..8 {
            assert_eq!(uf.find(index), 0);
        }
        assert_eq!(uf.m_cc_size[0], 8);
        assert!(uf.m_cc_parent.iter().all(|&parent| parent == 0));
    }

    #[test]
    fn builder_maps_camera_nodes_and_exports_unrenumbered_roots() {
        let image_a = tc(10, 0);
        let image_b = tc(10, 1);
        let image_c = tc(11, 0);
        let image_d = tc(12, 0);
        let mut matches = Matches::new();
        matches.insert((image_a, image_b), MatchData::from_inliers([(7, 4)]));
        matches.insert((image_b, image_c), MatchData::from_inliers([(4, 9)]));
        matches.insert((image_c, image_d), MatchData::from_inliers([(9, 2)]));

        let mut builder = TrackBuilder::new();
        builder.build(&matches);
        assert_eq!(builder.node_count(), 4);
        assert_eq!(
            builder
                .map_node_to_index
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![(image_a, 7), (image_b, 4), (image_c, 9), (image_d, 2),]
        );

        let report = builder.filter(3);
        assert_eq!(report.component_count_before, 1);
        assert_eq!(report.component_count_after, 1);
        assert!(report.rejected_track_ids.is_empty());

        let tracks = builder.export();
        assert_eq!(tracks.keys().copied().collect::<Vec<_>>(), vec![0]);
        assert_eq!(
            tracks[&0]
                .iter()
                .map(|(id, &f)| (*id, f))
                .collect::<Vec<_>>(),
            vec![(image_a, 7), (image_b, 4), (image_c, 9), (image_d, 2),]
        );
        assert_ne!(canonical_observation_hash(&tracks), 0);
    }

    #[test]
    fn filter_rejects_entire_conflicting_component_and_short_components() {
        let image_a = tc(1, 0);
        let image_b = tc(1, 1);
        let image_c = tc(2, 0);
        let image_d = tc(3, 0);
        let image_e = tc(4, 0);
        let mut matches = Matches::new();
        // The component contains image_a twice; it must be rejected as a
        // whole, including the otherwise valid b/c branch.
        matches.insert((image_a, image_b), MatchData::from_inliers([(1, 2)]));
        matches.insert((image_b, image_c), MatchData::from_inliers([(2, 3)]));
        matches.insert((image_a, image_c), MatchData::from_inliers([(2, 3)]));
        // This separate two-image component is below the length-3 gate.
        matches.insert((image_d, image_e), MatchData::from_inliers([(8, 9)]));

        let mut builder = TrackBuilder::new();
        builder.build(&matches);
        let report = builder.filter(3);
        assert_eq!(report.component_count_before, 2);
        assert_eq!(report.component_count_after, 0);
        assert_eq!(report.conflict_track_ids, vec![0]);
        // Upstream's `Filter` skips the rest of a component after detecting a
        // conflict, so the partial `tracks[root]` entry is also below the
        // minimum-length threshold.  The root remains rejected only once.
        assert_eq!(report.short_track_ids, vec![0, 4]);
        assert_eq!(report.rejected_track_ids, vec![0, 4]);
        assert!(builder.export().is_empty());
        assert_eq!(builder.track_count(), 0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalBaConfig {
    pub enabled: bool,
    /// Basalt's `config.mapper_use_lm` switch.  The legacy global-BA helper
    /// keeps this enabled by default (matching the checked-in M8e fixture),
    /// while the stateful NFR entry point can opt into the exact configured
    /// mapper branch.
    pub use_lm: bool,
    pub lambda_initial: f64,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub max_iterations: usize,
    pub huber_delta: f64,
    /// Basalt's mapper configuration at the pinned revision.
    pub observation_std_dev: f64,
    pub use_factors: bool,
}
impl Default for GlobalBaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            use_lm: true,
            lambda_initial: 1e-32,
            lambda_min: 1e-32,
            // Matches the checked-in EuRoC mapper fixture.  Basalt's generic
            // VioConfig default is 1e2, but that config explicitly widens the
            // LM ceiling to 1e3.
            lambda_max: 1e3,
            max_iterations: 10,
            huber_delta: 1.5,
            observation_std_dev: 0.25,
            use_factors: true,
        }
    }
}

/// Persistent damping state owned by the pinned `NfrMapper` object.
///
/// Upstream initializes `lambda` from `mapper_lm_lambda_min` in the
/// constructor, keeps `lambda_vee = 2`, and carries all three values across
/// separate `optimize()` calls.  Keeping that state outside a one-shot BA
/// summary is what lets the stateful session preserve the same LM schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalBaOptimizerState {
    pub lambda: f64,
    pub min_lambda: f64,
    pub max_lambda: f64,
    pub lambda_vee: f64,
}

impl GlobalBaOptimizerState {
    /// Construct the source mapper state.  `lambda_initial` is intentionally
    /// ignored here: Basalt's `NfrMapper` constructor sets the live lambda to
    /// `mapper_lm_lambda_min`, not to the VIO optimizer's initial value.
    pub const fn new(min_lambda: f64, max_lambda: f64) -> Self {
        Self {
            lambda: min_lambda,
            min_lambda,
            max_lambda,
            lambda_vee: 2.0,
        }
    }

    /// Construct the one-shot state used by the existing global-BA API.
    pub fn from_config(config: GlobalBaConfig) -> Self {
        let min_lambda = config.lambda_min;
        let max_lambda = config.lambda_max;
        Self {
            lambda: config.lambda_initial.clamp(min_lambda, max_lambda),
            min_lambda,
            max_lambda,
            lambda_vee: 2.0,
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct MapperState {
    /// Compatibility translation view.  The optimizer itself updates
    /// `se3_poses` in the full six-dimensional Basalt pose chart.
    pub poses: std::collections::BTreeMap<u64, [f64; 3]>,
    /// Full `T_w_i` estimates after the solve (translation additive, rotation
    /// left-multiplied, exactly as `PoseState::incPose`).
    pub se3_poses: BTreeMap<u64, SE3>,
    /// Full stereographic/inverse-distance mapper landmark estimates.
    pub landmarks: BTreeMap<u64, MapperLandmark>,
    pub tracks: Vec<Track>,
    pub gauge_anchor: u64,
}

/// One trial in the upstream mapper's LM inner loop.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalBaTrial {
    pub lambda: f64,
    pub f_diff: f64,
    pub after_vision_cost: f64,
    pub after_relative_cost: f64,
    pub after_roll_pitch_cost: f64,
    pub after_total_cost: f64,
    pub max_pose_increment: f64,
    pub accepted: bool,
}

/// Per-iteration trace retained for the M8e numerical parity gate.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalBaIteration {
    pub iteration: usize,
    pub vision_cost: f64,
    pub relative_cost: f64,
    pub roll_pitch_cost: f64,
    pub total_cost: f64,
    pub h_diagonal: Vec<f64>,
    /// Positive solve vector (`H x = b`).  Upstream applies `-x` to poses.
    pub pose_solve: Vec<f64>,
    /// Landmark back-substitution increments in sorted track order.
    pub landmark_increments: Vec<(u64, [f64; 3])>,
    pub trials: Vec<GlobalBaTrial>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapperSummary {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub gauge_anchor: u64,
    pub pose_count: usize,
    pub track_count: usize,
    pub disabled: bool,
    pub trace: Vec<GlobalBaIteration>,
    pub final_lambda: f64,
    pub final_state_hash: u64,
    pub trace_hash: u64,
}

const fn identity_rotation_array() -> [f64; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}
type Matrix2x3 = nalgebra::SMatrix<f64, 2, 3>;
type Matrix2x6 = nalgebra::SMatrix<f64, 2, 6>;
type Matrix4x6 = nalgebra::SMatrix<f64, 4, 6>;
type Matrix6x3 = nalgebra::SMatrix<f64, 6, 3>;

#[derive(Debug, Clone)]
struct VisionLandmarkSolve {
    id: u64,
    /// Undamped landmark block, kept so LM retries can damp it.
    hll: Matrix3<f64>,
    hll_inv: Matrix3<f64>,
    bl: Vector3<f64>,
    /// Schur cross blocks `H_pl`, one for each participating absolute pose.
    hpl: Vec<(usize, Matrix6x3)>,
}

#[derive(Debug, Clone)]
struct VisionLinearization {
    cost: f64,
    h: PoseBlockHessian,
    b: DVector<f64>,
    landmarks: Vec<VisionLandmarkSolve>,
}

#[derive(Debug, Clone, Copy, Default)]
struct CostBreakdown {
    vision: f64,
    relative: f64,
    roll_pitch: f64,
}

impl CostBreakdown {
    fn total(self) -> f64 {
        self.vision + self.relative + self.roll_pitch
    }
}

fn se3_to_pose_array(pose: &SE3) -> [f64; 7] {
    let q = pose.rotation.quaternion();
    [
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        q.w,
        q.i,
        q.j,
        q.k,
    ]
}

fn pose_from_array(pose: [f64; 7]) -> SE3 {
    let p = pose_value(pose);
    SE3::new(p.rotation, p.translation)
}

fn initial_mapper_poses(data: &MargData) -> BTreeMap<u64, SE3> {
    let mut poses = BTreeMap::new();
    // `frame_poses` and `frame_states` are separate in the versioned
    // MargData contract.  A pose-only keyframe must not disappear merely
    // because the old translation-only scaffold looked at frame_states.
    for frame in &data.frame_poses {
        poses.insert(frame.frame_id, pose_from_array(frame.pose));
    }
    for state in &data.frame_states {
        // `NfrMapper::addMargData` lets an explicit keyframe state replace a
        // same-ID pose-only record.  Unprocessed synthetic MargData may still
        // carry non-keyframe states without a pose table; retain those only
        // when no pose-only value exists, preserving the compatibility API.
        let is_keyframe = state.is_keyframe
            || data.kfs_all.contains(&state.frame_id)
            || data.keyframes.contains(&state.frame_id);
        if is_keyframe {
            poses.insert(state.frame_id, pose_from_array(state.pose));
        } else {
            poses
                .entry(state.frame_id)
                .or_insert_with(|| pose_from_array(state.pose));
        }
    }
    poses
}

fn rotation_from_factor(factor: &RollPitchFactor) -> Matrix3<f64> {
    Matrix3::from_row_slice(&factor.measured_rotation)
}

fn info6(values: &[f64]) -> Option<Matrix6<f64>> {
    (values.len() == 36)
        .then(|| Matrix6::from_row_slice(values))
        .filter(|m| m.iter().all(|v| v.is_finite()))
}

fn matrix6x6_from_arrays(values: [[f64; 6]; 6]) -> Matrix6<f64> {
    Matrix6::from_fn(|r, c| values[r][c])
}

fn vector6_from_array(values: [f64; 6]) -> Vector6<f64> {
    Vector6::from_row_slice(&values)
}

fn dynamic_matrix6(block: &Matrix6<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(6, 6, |row, col| block[(row, col)])
}

fn dynamic_vector6(vector: &Vector6<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(6, 1, |row, _| vector[row])
}

fn pose_relative_jacobians(
    host_pose: &SE3,
    host_extrinsic: &SE3,
    target_pose: &SE3,
    target_extrinsic: &SE3,
    host: TimeCamId,
    target: TimeCamId,
) -> (SE3, Matrix6<f64>, Matrix6<f64>) {
    if host == target {
        return (SE3::identity(), Matrix6::zeros(), Matrix6::zeros());
    }

    // This is `computeRelPose` from basalt/utils/ba_utils.h, written using
    // the repository's small SE3 value type.  Its tangent convention is the
    // mapper's PoseState chart: additive world translation and left SO(3).
    let tmp2 = target_extrinsic.inverse();
    let body_relative = target_pose.inverse().compose(host_pose);
    let tmp = tmp2.compose(&body_relative);
    let relative = tmp.compose(host_extrinsic);

    let mut rr_h = Matrix6::zeros();
    let rh = host_pose
        .rotation
        .inverse()
        .to_rotation_matrix()
        .into_inner();
    rr_h.fixed_view_mut::<3, 3>(0, 0).copy_from(&rh);
    rr_h.fixed_view_mut::<3, 3>(3, 3).copy_from(&rh);

    let mut rr_t = Matrix6::zeros();
    let rt = target_pose
        .rotation
        .inverse()
        .to_rotation_matrix()
        .into_inner();
    rr_t.fixed_view_mut::<3, 3>(0, 0).copy_from(&rt);
    rr_t.fixed_view_mut::<3, 3>(3, 3).copy_from(&rt);

    (relative, tmp.adjoint() * rr_h, -(tmp2.adjoint() * rr_t))
}

fn double_sphere_projection_jacobian(
    camera: &DoubleSphereCamera,
    point: &Point3<f64>,
) -> Option<(Point2<f64>, nalgebra::SMatrix<f64, 2, 4>)> {
    let projected = camera.project(point)?;
    let p = point.coords;
    let d1 = p.norm();
    if !d1.is_finite() || d1 <= f64::EPSILON {
        return None;
    }
    let zeta = camera.xi * d1 + p.z;
    let d2 = (p.x * p.x + p.y * p.y + zeta * zeta).sqrt();
    let denominator = camera.alpha * d2 + (1.0 - camera.alpha) * zeta;
    if !d2.is_finite() || !denominator.is_finite() || denominator <= f64::EPSILON {
        return None;
    }

    let dzeta = camera.xi * p / d1 + Vector3::z();
    let dd2 = Vector3::new(
        (p.x + zeta * dzeta.x) / d2,
        (p.y + zeta * dzeta.y) / d2,
        zeta * dzeta.z / d2,
    );
    let dden = camera.alpha * dd2 + (1.0 - camera.alpha) * dzeta;
    let inv_denom2 = denominator.recip().powi(2);
    let mut jac = nalgebra::SMatrix::<f64, 2, 4>::zeros();
    for col in 0..3 {
        jac[(0, col)] =
            camera.fx * ((if col == 0 { denominator } else { 0.0 }) - p.x * dden[col]) * inv_denom2;
        jac[(1, col)] =
            camera.fy * ((if col == 1 { denominator } else { 0.0 }) - p.y * dden[col]) * inv_denom2;
    }
    jac.iter()
        .all(|v| v.is_finite())
        .then_some((projected, jac))
}

/// Linearize one anchored mapper observation exactly as
/// `ScBundleAdjustmentBase::linearizeHelperStatic` does.  The returned pose
/// matrices are already converted from relative-camera coordinates to the
/// absolute body-pose blocks.
pub(crate) fn linearize_mapper_observation(
    landmark: &MapperLandmark,
    observation: &MapperObservation,
    poses: &BTreeMap<u64, SE3>,
    calibration: &BasaltCalibration,
    pose_indices: &BTreeMap<u64, usize>,
) -> Option<(Vector2<f64>, Vec<(usize, Matrix2x6)>, Matrix2x3)> {
    let host_pose = poses.get(&landmark.host.frame_id)?;
    let target_pose = poses.get(&observation.image.frame_id)?;
    let host_extrinsic = calibration.camera_to_imu(landmark.host.cam_id)?;
    let target_extrinsic = calibration.camera_to_imu(observation.image.cam_id)?;
    let camera = calibration.camera(observation.image.cam_id)?;

    let (relative, d_rel_d_h, d_rel_d_t) = pose_relative_jacobians(
        host_pose,
        host_extrinsic,
        target_pose,
        target_extrinsic,
        landmark.host,
        observation.image,
    );

    let bearing = landmark.direction.bearing();
    let rho = landmark.inverse_distance;
    if !rho.is_finite() || !bearing.iter().all(|v| v.is_finite()) {
        return None;
    }
    let p_h = nalgebra::Vector4::new(bearing.x, bearing.y, bearing.z, rho);
    let p_t_h = relative.matrix() * p_h;
    let p_t = Point3::new(p_t_h.x, p_t_h.y, p_t_h.z);
    let (projection, j_projection) = double_sphere_projection_jacobian(camera, &p_t)?;
    let residual = projection - observation.pixel;

    let mut d_point_d_xi = Matrix4x6::zeros();
    d_point_d_xi
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(Matrix3::identity() * rho));
    d_point_d_xi
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew(&p_t.coords)));
    let d_res_d_relative = j_projection * d_point_d_xi;

    let j_direction = landmark.direction.bearing_jacobian();
    let mut d_point_d_landmark = nalgebra::SMatrix::<f64, 4, 3>::zeros();
    d_point_d_landmark
        .fixed_view_mut::<3, 2>(0, 0)
        .copy_from(&(relative.rotation.to_rotation_matrix().into_inner() * j_direction));
    d_point_d_landmark
        .fixed_view_mut::<3, 1>(0, 2)
        .copy_from(&relative.translation);
    let d_res_d_landmark = j_projection * d_point_d_landmark;

    let mut pose_terms = Vec::new();
    if landmark.host != observation.image {
        let host_index = *pose_indices.get(&landmark.host.frame_id)?;
        let target_index = *pose_indices.get(&observation.image.frame_id)?;
        pose_terms.push((host_index, d_res_d_relative * d_rel_d_h));
        pose_terms.push((target_index, d_res_d_relative * d_rel_d_t));
    }
    // A stereo observation in the same frame has two terms that point to the
    // same absolute pose.  ScBundleAdjustmentBase adds both blocks; merging
    // them here avoids changing the resulting Schur arithmetic.
    if pose_terms.len() == 2 && pose_terms[0].0 == pose_terms[1].0 {
        let (index, first) = pose_terms.remove(0);
        let (_, second) = pose_terms.remove(0);
        pose_terms.push((index, first + second));
    }

    Some((residual, pose_terms, d_res_d_landmark))
}

fn linearize_vision(
    poses: &BTreeMap<u64, SE3>,
    landmarks: &BTreeMap<u64, MapperLandmark>,
    calibration: Option<&BasaltCalibration>,
    pose_indices: &BTreeMap<u64, usize>,
    pose_count: usize,
    config: GlobalBaConfig,
) -> VisionLinearization {
    let mut out = VisionLinearization {
        cost: 0.0,
        h: PoseBlockHessian::new(pose_count),
        b: DVector::zeros(pose_count * 6),
        landmarks: Vec::new(),
    };
    let Some(calibration) = calibration else {
        return out;
    };

    for (&landmark_id, landmark) in landmarks {
        let mut hll = Matrix3::zeros();
        let mut bl = Vector3::zeros();
        let mut hpl_by_pose: BTreeMap<usize, Matrix6x3> = BTreeMap::new();

        for observation in &landmark.observations {
            let Some((residual, pose_terms, j_landmark)) = linearize_mapper_observation(
                landmark,
                observation,
                poses,
                calibration,
                pose_indices,
            ) else {
                continue;
            };
            let e = residual.norm();
            if !e.is_finite() {
                continue;
            }
            let huber_weight = if e < config.huber_delta {
                1.0
            } else {
                config.huber_delta / e
            };
            let obs_weight =
                huber_weight / (config.observation_std_dev * config.observation_std_dev);
            if !obs_weight.is_finite() {
                continue;
            }
            out.cost += 0.5 * (2.0 - huber_weight) * obs_weight * residual.dot(&residual);
            hll += obs_weight * j_landmark.transpose() * j_landmark;
            bl += obs_weight * j_landmark.transpose() * residual;

            // Accumulate the complete JpᵀWJp block for this observation.
            // In particular, stereo/temporal observations touching two
            // poses contribute both off-diagonal JiᵀWJj blocks.  Omitting
            // these before Schur subtraction makes the reduced system
            // indefinite and causes CG to fall back on the first LM step.
            for &(row_pose, row_j) in &pose_terms {
                for &(col_pose, col_j) in &pose_terms {
                    let hpp = obs_weight * row_j.transpose() * col_j;
                    out.h.add_block(row_pose, col_pose, &hpp);
                }
            }
            for &(pose_index, j_pose) in &pose_terms {
                let bp = obs_weight * j_pose.transpose() * residual;
                out.b
                    .view_mut((pose_index * 6, 0), (6, 1))
                    .add_assign(&dynamic_vector6(&bp));
                let hpl = obs_weight * j_pose.transpose() * j_landmark;
                hpl_by_pose
                    .entry(pose_index)
                    .and_modify(|value| *value += hpl)
                    .or_insert(hpl);
            }
        }

        // Basalt's landmark block uses an Eigen LDLT solve for Hll^{-1}; a
        // generic adjugate inverse amplifies the near-planar inverse-distance
        // block and changes the first LM point back-substitution materially.
        let Some(hll_inv) = hll
            .clone()
            .cholesky()
            .map(|chol| chol.inverse())
            .or_else(|| hll.try_inverse())
        else {
            continue;
        };
        if !hll_inv.iter().all(|v| v.is_finite()) {
            continue;
        }
        let hpl = hpl_by_pose.into_iter().collect::<Vec<_>>();
        for &(pose_i, hpl_i) in &hpl {
            let schur_b = hpl_i * hll_inv * bl;
            out.b
                .view_mut((pose_i * 6, 0), (6, 1))
                .add_assign(&dynamic_vector6(&(-schur_b)));
            for &(pose_j, hpl_j) in &hpl {
                let schur_h = hpl_i * hll_inv * hpl_j.transpose();
                out.h.add_block(pose_i, pose_j, &(-schur_h));
            }
        }
        out.landmarks.push(VisionLandmarkSolve {
            id: landmark_id,
            hll,
            hll_inv,
            bl,
            hpl,
        });
    }
    out
}

fn linearize_factors(
    poses: &BTreeMap<u64, SE3>,
    factors: &MapperFactors,
    pose_indices: &BTreeMap<u64, usize>,
    pose_count: usize,
    config: GlobalBaConfig,
    h: &mut PoseBlockHessian,
    b: &mut DVector<f64>,
) {
    if !config.use_factors {
        return;
    }
    for factor in &factors.relative_pose {
        let (Some(&from), Some(&to), Some(pose_i), Some(pose_j)) = (
            pose_indices.get(&factor.from),
            pose_indices.get(&factor.to),
            poses.get(&factor.from),
            poses.get(&factor.to),
        ) else {
            continue;
        };
        let Some(info) = info6(&factor.information) else {
            continue;
        };
        let measured = pose_from_array([
            factor.translation[0],
            factor.translation[1],
            factor.translation[2],
            factor.rotation[0],
            factor.rotation[1],
            factor.rotation[2],
            factor.rotation[3],
        ]);
        let (residual, ji_arr, jj_arr) = rel_pose_error(
            se3_to_pose_array(&measured),
            se3_to_pose_array(pose_i),
            se3_to_pose_array(pose_j),
        );
        let ji = matrix6x6_from_arrays(ji_arr);
        let jj = matrix6x6_from_arrays(jj_arr);
        let residual = vector6_from_array(residual);
        let w = factor.weight;
        let wi = w * info;
        add_factor_pose_block(h, from, from, ji.transpose() * wi * ji, pose_count);
        add_factor_pose_block(h, from, to, ji.transpose() * wi * jj, pose_count);
        add_factor_pose_block(h, to, from, jj.transpose() * wi * ji, pose_count);
        add_factor_pose_block(h, to, to, jj.transpose() * wi * jj, pose_count);
        add_factor_pose_gradient(b, from, ji.transpose() * wi * residual, pose_count);
        add_factor_pose_gradient(b, to, jj.transpose() * wi * residual, pose_count);
    }
    for factor in &factors.imu_relative_pose {
        let (Some(&from), Some(&to), Some(pose_i), Some(pose_j)) = (
            pose_indices.get(&factor.from),
            pose_indices.get(&factor.to),
            poses.get(&factor.from),
            poses.get(&factor.to),
        ) else {
            continue;
        };
        let Some(info) = info6(&factor.information) else {
            continue;
        };
        let measured = pose_from_array([
            factor.translation[0],
            factor.translation[1],
            factor.translation[2],
            factor.rotation[0],
            factor.rotation[1],
            factor.rotation[2],
            factor.rotation[3],
        ]);
        let (residual, ji_arr, jj_arr) = rel_pose_error(
            se3_to_pose_array(&measured),
            se3_to_pose_array(pose_i),
            se3_to_pose_array(pose_j),
        );
        let ji = matrix6x6_from_arrays(ji_arr);
        let jj = matrix6x6_from_arrays(jj_arr);
        let residual = vector6_from_array(residual);
        let w = factor.weight;
        let wi = w * info;
        add_factor_pose_block(h, from, from, ji.transpose() * wi * ji, pose_count);
        add_factor_pose_block(h, from, to, ji.transpose() * wi * jj, pose_count);
        add_factor_pose_block(h, to, from, jj.transpose() * wi * ji, pose_count);
        add_factor_pose_block(h, to, to, jj.transpose() * wi * jj, pose_count);
        add_factor_pose_gradient(b, from, ji.transpose() * wi * residual, pose_count);
        add_factor_pose_gradient(b, to, jj.transpose() * wi * residual, pose_count);
    }
    for factor in &factors.roll_pitch {
        let (Some(&index), Some(pose)) = (
            pose_indices.get(&factor.frame_id),
            poses.get(&factor.frame_id),
        ) else {
            continue;
        };
        let measured = matrix3_nested_array(rotation_from_factor(factor));
        let (residual_arr, jac_arr) = roll_pitch_error(se3_to_pose_array(pose), measured);
        let residual = Vector2::from_row_slice(&residual_arr);
        let jac = Matrix2x6::from_fn(|r, c| jac_arr[r][c]);
        let info = Matrix2::from_row_slice(&factor.information);
        let wi = factor.weight * info;
        add_factor_pose_block(h, index, index, jac.transpose() * wi * jac, pose_count);
        add_factor_pose_gradient(b, index, jac.transpose() * wi * residual, pose_count);
    }
}

fn add_factor_pose_block(
    h: &mut PoseBlockHessian,
    row: usize,
    col: usize,
    block: Matrix6<f64>,
    pose_count: usize,
) {
    if row >= pose_count || col >= pose_count {
        return;
    }
    h.add_block(row, col, &block);
}

/// Block-sparse pose Hessian for global BA.
///
/// Upstream accumulates the reduced camera system in a
/// `SparseHashAccumulator`; a dense `(6N)^2` matrix made mapper memory grow
/// quadratically with keyframes.  Blocks accumulate element-wise in the same
/// order as the former dense `view_mut(..).add_assign(..)` calls, and absent
/// blocks are exact zeros, so the solve sees the same values.
#[derive(Debug, Clone)]
struct PoseBlockHessian {
    pose_count: usize,
    blocks: BTreeMap<(usize, usize), Matrix6<f64>>,
}

impl PoseBlockHessian {
    fn new(pose_count: usize) -> Self {
        Self {
            pose_count,
            blocks: BTreeMap::new(),
        }
    }

    fn dim(&self) -> usize {
        self.pose_count * 6
    }

    fn add_block(&mut self, row: usize, col: usize, block: &Matrix6<f64>) {
        *self.blocks.entry((row, col)).or_insert_with(Matrix6::zeros) += block;
    }

    fn diagonal(&self) -> Vec<f64> {
        let mut diagonal = vec![0.0; self.dim()];
        for pose in 0..self.pose_count {
            if let Some(block) = self.blocks.get(&(pose, pose)) {
                for axis in 0..6 {
                    diagonal[pose * 6 + axis] = block[(axis, axis)];
                }
            }
        }
        diagonal
    }

    fn to_dense(&self) -> DMatrix<f64> {
        let mut dense = DMatrix::zeros(self.dim(), self.dim());
        for (&(row, col), block) in &self.blocks {
            dense.view_mut((row * 6, col * 6), (6, 6)).copy_from(block);
        }
        dense
    }

    /// `(H + H^T) / 2` with per-entry operand order `(h[r,c] + h[c,r]) * 0.5`,
    /// plus the LM diagonal, exactly as [`solve_damped_system`] builds it.
    fn symmetrized_damped(&self, h_diagonal: &[f64], lambda: f64, min_lambda: f64) -> Self {
        let zero = Matrix6::zeros();
        let mut keys = self.blocks.keys().copied().collect::<BTreeSet<_>>();
        keys.extend(self.blocks.keys().map(|&(row, col)| (col, row)));
        for pose in 0..self.pose_count {
            keys.insert((pose, pose));
        }
        let mut blocks = BTreeMap::new();
        for (row, col) in keys {
            let upper = self.blocks.get(&(row, col)).unwrap_or(&zero);
            let lower = self.blocks.get(&(col, row)).unwrap_or(&zero);
            let mut block = Matrix6::from_fn(|r, c| (upper[(r, c)] + lower[(c, r)]) * 0.5);
            if row == col {
                for axis in 0..6 {
                    let index = row * 6 + axis;
                    block[(axis, axis)] += (h_diagonal[index] * lambda).max(min_lambda);
                }
            }
            blocks.insert((row, col), block);
        }
        Self {
            pose_count: self.pose_count,
            blocks,
        }
    }

    /// `self * x`, accumulating each row in ascending column order like
    /// nalgebra's dense column-major gemv (`y = A[:,j] * x[j] + y`).
    fn mul_vector(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut y = DVector::zeros(self.dim());
        for (&(row, col), block) in &self.blocks {
            for c in 0..6 {
                let value = x[col * 6 + c];
                for r in 0..6 {
                    y[row * 6 + r] += block[(r, c)] * value;
                }
            }
        }
        y
    }
}

fn add_factor_pose_gradient(
    b: &mut DVector<f64>,
    index: usize,
    value: Vector6<f64>,
    pose_count: usize,
) {
    if index >= pose_count {
        return;
    }
    b.view_mut((index * 6, 0), (6, 1))
        .add_assign(&dynamic_vector6(&value));
}

fn evaluate_vision_cost(
    poses: &BTreeMap<u64, SE3>,
    landmarks: &BTreeMap<u64, MapperLandmark>,
    calibration: Option<&BasaltCalibration>,
    config: GlobalBaConfig,
) -> f64 {
    let Some(calibration) = calibration else {
        return 0.0;
    };
    let mut cost = 0.0;
    let pose_indices = poses
        .keys()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect::<BTreeMap<_, _>>();
    for landmark in landmarks.values() {
        for observation in &landmark.observations {
            let Some((residual, _, _)) = linearize_mapper_observation(
                landmark,
                observation,
                poses,
                calibration,
                &pose_indices,
            ) else {
                continue;
            };
            let e = residual.norm();
            if !e.is_finite() {
                continue;
            }
            let hw = if e < config.huber_delta {
                1.0
            } else {
                config.huber_delta / e
            };
            let ow = hw / (config.observation_std_dev * config.observation_std_dev);
            cost += 0.5 * (2.0 - hw) * ow * residual.dot(&residual);
        }
    }
    cost
}

fn evaluate_costs(
    poses: &BTreeMap<u64, SE3>,
    landmarks: &BTreeMap<u64, MapperLandmark>,
    factors: &MapperFactors,
    calibration: Option<&BasaltCalibration>,
    config: GlobalBaConfig,
) -> CostBreakdown {
    let vision = evaluate_vision_cost(poses, landmarks, calibration, config);
    if !config.use_factors {
        return CostBreakdown {
            vision,
            ..CostBreakdown::default()
        };
    }
    let mut relative = 0.0;
    for factor in &factors.relative_pose {
        let (Some(pose_i), Some(pose_j)) = (poses.get(&factor.from), poses.get(&factor.to)) else {
            continue;
        };
        let Some(info) = info6(&factor.information) else {
            continue;
        };
        let measured = pose_from_array([
            factor.translation[0],
            factor.translation[1],
            factor.translation[2],
            factor.rotation[0],
            factor.rotation[1],
            factor.rotation[2],
            factor.rotation[3],
        ]);
        let (residual, _, _) = rel_pose_error(
            se3_to_pose_array(&measured),
            se3_to_pose_array(pose_i),
            se3_to_pose_array(pose_j),
        );
        let residual = vector6_from_array(residual);
        relative += factor.weight * (residual.transpose() * info * residual)[(0, 0)];
    }
    for factor in &factors.imu_relative_pose {
        let (Some(pose_i), Some(pose_j)) = (poses.get(&factor.from), poses.get(&factor.to)) else {
            continue;
        };
        let Some(info) = info6(&factor.information) else {
            continue;
        };
        let measured = pose_from_array([
            factor.translation[0],
            factor.translation[1],
            factor.translation[2],
            factor.rotation[0],
            factor.rotation[1],
            factor.rotation[2],
            factor.rotation[3],
        ]);
        let (residual, _, _) = rel_pose_error(
            se3_to_pose_array(&measured),
            se3_to_pose_array(pose_i),
            se3_to_pose_array(pose_j),
        );
        let residual = vector6_from_array(residual);
        relative += factor.weight * (residual.transpose() * info * residual)[(0, 0)];
    }
    let mut roll_pitch = 0.0;
    for factor in &factors.roll_pitch {
        let Some(pose) = poses.get(&factor.frame_id) else {
            continue;
        };
        let (residual, _) = roll_pitch_error(
            se3_to_pose_array(pose),
            matrix3_nested_array(rotation_from_factor(factor)),
        );
        let residual = Vector2::from_row_slice(&residual);
        let info = Matrix2::from_row_slice(&factor.information);
        roll_pitch += factor.weight * (residual.transpose() * info * residual)[(0, 0)];
    }
    CostBreakdown {
        vision,
        relative,
        roll_pitch,
    }
}

fn solve_damped_system(
    h: &DMatrix<f64>,
    b: &DVector<f64>,
    h_diagonal: &[f64],
    lambda: f64,
    min_lambda: f64,
    _gauge_anchor: Option<usize>,
) -> DVector<f64> {
    // SparseHashAccumulator stores the symmetric normal equations; dense
    // block accumulation can leave roundoff-level antisymmetric residue that
    // makes CG report a negative denominator on this gauge-singular system.
    let mut system = (h + h.transpose()) * 0.5;
    let rhs = b.clone();
    for (index, diagonal) in h_diagonal.iter().enumerate() {
        system[(index, index)] += (diagonal * lambda).max(min_lambda);
    }
    // Do not pin a pose here.  The pinned NfrMapper passes the full
    // `AbsOrderMap` to `SparseHashAccumulator::solve`; its only regularizer
    // is the diagonal LM term above.  `gauge_anchor` remains metadata for
    // callers and summaries, but is intentionally not a hidden prior.
    //
    // The pinned accumulator selects Eigen's diagonal-preconditioned
    // ConjugateGradient path (`iterative_solver = true`, tolerance 1e-4).
    // Keep that solve contract here rather than silently replacing it with a
    // dense factorization; the latter is useful only as a malformed/indefinite
    // fallback for synthetic callers.
    if let Some(solution) = diagonal_preconditioned_cg(&system, &rhs, 1e-4) {
        return solution;
    }
    if let Some(cholesky) = system.clone().cholesky() {
        return cholesky.solve(&rhs);
    }
    if let Some(solution) = system.clone().lu().solve(&rhs) {
        return solution;
    }
    system
        .qr()
        .solve(&rhs)
        .unwrap_or_else(|| DVector::zeros(b.len()))
}

/// The upstream `SparseHashAccumulator::solve` uses Eigen's default
/// diagonal preconditioner and `ConjugateGradient` with a relative tolerance
/// of `1e-4`.  This dense multiplication backend preserves those iteration and
/// stopping semantics while leaving sparse/hash accumulation out of the Rust
/// core.  Returning `None` delegates malformed or indefinite systems to the
/// defensive factorization fallbacks in [`solve_damped_system`].
/// Block-sparse counterpart of [`solve_damped_system`]: same symmetrization,
/// damping and CG contract.  Only the rare CG failure densifies for the
/// factorization fallbacks.
fn solve_damped_block_system(
    h: &PoseBlockHessian,
    b: &DVector<f64>,
    h_diagonal: &[f64],
    lambda: f64,
    min_lambda: f64,
    gauge_anchor: Option<usize>,
) -> DVector<f64> {
    let system = h.symmetrized_damped(h_diagonal, lambda, min_lambda);
    let finite = system
        .blocks
        .values()
        .all(|block| block.iter().all(|value| value.is_finite()));
    if finite && b.len() == system.dim() {
        let diagonal = system.diagonal();
        if let Some(solution) = preconditioned_cg(
            b,
            1e-4,
            |index| diagonal[index],
            |vector| system.mul_vector(vector),
        ) {
            return solution;
        }
    }
    solve_damped_system(
        &h.to_dense(),
        b,
        h_diagonal,
        lambda,
        min_lambda,
        gauge_anchor,
    )
}

fn diagonal_preconditioned_cg(
    system: &DMatrix<f64>,
    rhs: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    if system.nrows() != system.ncols() || rhs.len() != system.nrows() {
        return None;
    }
    if !system.iter().all(|value| value.is_finite()) {
        return None;
    }
    preconditioned_cg(
        rhs,
        tolerance,
        |index| system[(index, index)],
        |vector| system * vector,
    )
}

fn preconditioned_cg(
    rhs: &DVector<f64>,
    tolerance: f64,
    diagonal_at: impl Fn(usize) -> f64,
    multiply: impl Fn(&DVector<f64>) -> DVector<f64>,
) -> Option<DVector<f64>> {
    let n = rhs.len();
    if n == 0 {
        return Some(DVector::zeros(0));
    }
    if !rhs.iter().all(|value| value.is_finite()) {
        return None;
    }
    let rhs_norm2 = rhs.dot(rhs);
    if rhs_norm2 == 0.0 {
        return Some(DVector::zeros(n));
    }

    // Eigen's DiagonalPreconditioner keeps a unit inverse for zero diagonal
    // entries.  A damped mapper system normally has no zero pivots, but this
    // branch preserves the same behavior for sparse/empty synthetic blocks.
    let inv_diagonal = DVector::from_iterator(
        n,
        (0..n).map(|index| {
            let diagonal = diagonal_at(index);
            if diagonal != 0.0 && diagonal.is_finite() {
                diagonal.recip()
            } else {
                1.0
            }
        }),
    );
    let mut x = DVector::zeros(n);
    let mut residual = rhs.clone();
    let mut z = residual.component_mul(&inv_diagonal);
    let mut direction = z.clone();
    let mut rz = residual.dot(&z);
    if !rz.is_finite() || rz <= 0.0 {
        return None;
    }
    // Eigen compares squared residuals and clamps the threshold to the
    // smallest positive normal value (`considerAsZero`).
    let threshold = (tolerance * tolerance * rhs_norm2).max(f64::MIN_POSITIVE);
    let max_iterations = n.saturating_mul(2).max(1);
    for _iteration in 0..max_iterations {
        let product = multiply(&direction);
        let denominator = direction.dot(&product);
        if !denominator.is_finite() || denominator <= 0.0 {
            return None;
        }
        let alpha = rz / denominator;
        if !alpha.is_finite() {
            return None;
        }
        x += alpha * &direction;
        residual -= alpha * product;
        if !residual.iter().all(|value| value.is_finite()) {
            return None;
        }
        if residual.dot(&residual) < threshold {
            return Some(x);
        }
        z = residual.component_mul(&inv_diagonal);
        let next_rz = residual.dot(&z);
        if !next_rz.is_finite() || next_rz <= 0.0 {
            return None;
        }
        let beta = next_rz / rz;
        if !beta.is_finite() {
            return None;
        }
        direction = &z + beta * direction;
        rz = next_rz;
    }
    Some(x)
}

fn apply_pose_solve(
    poses: &mut BTreeMap<u64, SE3>,
    pose_indices: &BTreeMap<u64, usize>,
    solve: &DVector<f64>,
) {
    for (&id, &index) in pose_indices {
        let Some(pose) = poses.get_mut(&id) else {
            continue;
        };
        let offset = index * 6;
        if offset + 6 > solve.len() {
            continue;
        }
        pose.translation -= Vector3::new(solve[offset], solve[offset + 1], solve[offset + 2]);
        let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(
            -solve[offset + 3],
            -solve[offset + 4],
            -solve[offset + 5],
        ));
        pose.rotation = rotation * pose.rotation;
    }
}

/// Damped landmark inverses `(Hll + D_l)^-1`, with
/// `D_l = max(diag(Hll) * lambda, min_lambda)`, and the reduced pose system
/// built from them. The undamped reduced system subtracts
/// `Hpl Hll^-1 Hplᵀ` (and `Hpl Hll^-1 bl`), so the damped one adds back
/// `Hpl (Hll^-1 - Hl_damped^-1) Hplᵀ` for each landmark. This needs no
/// relinearization. A landmark whose damped block is not invertible keeps
/// its undamped inverse.
fn damped_landmark_system(
    h: &PoseBlockHessian,
    b: &DVector<f64>,
    solves: &[VisionLandmarkSolve],
    lambda: f64,
    min_lambda: f64,
) -> (PoseBlockHessian, DVector<f64>, Vec<Matrix3<f64>>) {
    let mut h = h.clone();
    let mut b = b.clone();
    let mut inverses = Vec::with_capacity(solves.len());
    for solve in solves {
        let mut damped = solve.hll;
        for axis in 0..3 {
            damped[(axis, axis)] += (solve.hll[(axis, axis)] * lambda).max(min_lambda);
        }
        let damped_inv = damped
            .cholesky()
            .map(|chol| chol.inverse())
            .filter(|inverse| inverse.iter().all(|value| value.is_finite()))
            .unwrap_or(solve.hll_inv);
        let correction = solve.hll_inv - damped_inv;
        for &(pose_i, hpl_i) in &solve.hpl {
            let db = hpl_i * correction * solve.bl;
            for axis in 0..6 {
                b[pose_i * 6 + axis] += db[axis];
            }
            for &(pose_j, hpl_j) in &solve.hpl {
                h.add_block(pose_i, pose_j, &(hpl_i * correction * hpl_j.transpose()));
            }
        }
        inverses.push(damped_inv);
    }
    (h, b, inverses)
}

fn back_substitute_landmarks(
    landmarks: &mut BTreeMap<u64, MapperLandmark>,
    solves: &[VisionLandmarkSolve],
    pose_indices: &BTreeMap<u64, usize>,
    pose_solve: &DVector<f64>,
) -> Vec<(u64, [f64; 3])> {
    back_substitute_landmarks_with(landmarks, solves, None, pose_indices, pose_solve)
}

/// Landmark back-substitution with an optional per-landmark inverse that
/// replaces `hll_inv` (the damped inverse on LM retries).
fn back_substitute_landmarks_with(
    landmarks: &mut BTreeMap<u64, MapperLandmark>,
    solves: &[VisionLandmarkSolve],
    damped_inverses: Option<&[Matrix3<f64>]>,
    pose_indices: &BTreeMap<u64, usize>,
    pose_solve: &DVector<f64>,
) -> Vec<(u64, [f64; 3])> {
    let mut increments = Vec::with_capacity(solves.len());
    for (index, solve) in solves.iter().enumerate() {
        let hll_inv = damped_inverses.map_or(&solve.hll_inv, |inverses| &inverses[index]);
        let mut h_l_p_x = Vector3::zeros();
        for &(pose_index, hpl) in &solve.hpl {
            let offset = pose_index * 6;
            if offset + 6 <= pose_solve.len() {
                h_l_p_x += hpl.transpose()
                    * Vector6::from_column_slice(&pose_solve.as_slice()[offset..offset + 6]);
            }
        }
        let increment = -(hll_inv * (solve.bl - h_l_p_x));
        if let Some(landmark) = landmarks.get_mut(&solve.id) {
            landmark.direction.xy.coords += increment.fixed_rows::<2>(0);
            landmark.inverse_distance = (landmark.inverse_distance + increment.z).max(0.0);
        }
        increments.push((solve.id, [increment.x, increment.y, increment.z]));
    }
    let _ = pose_indices;
    increments
}

fn make_mapper_state(
    poses: BTreeMap<u64, SE3>,
    landmarks: BTreeMap<u64, MapperLandmark>,
    anchor: u64,
) -> MapperState {
    let translations = poses
        .iter()
        .map(|(&id, pose)| (id, pose.translation.into()))
        .collect();
    let tracks = landmarks
        .iter()
        .map(|(&id, landmark)| Track {
            id,
            observations: landmark
                .observations
                .iter()
                .map(|observation| (observation.image, observation.feature_id))
                .collect(),
        })
        .collect();
    MapperState {
        poses: translations,
        se3_poses: poses,
        landmarks,
        tracks,
        gauge_anchor: anchor,
    }
}

fn canonical_float_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else if value.is_nan() {
        f64::NAN.to_bits()
    } else {
        value.to_bits()
    }
}

fn fnv_append(hash: &mut u64, bytes: impl IntoIterator<Item = u8>) {
    for byte in bytes {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(1099511628211);
    }
}

fn fnv_append_u64(hash: &mut u64, value: u64) {
    fnv_append(hash, value.to_le_bytes());
}

fn fnv_append_f64(hash: &mut u64, value: f64) {
    fnv_append_u64(hash, canonical_float_bits(value));
}

/// Canonical FNV-1a hash of the upstream mapper state, including rotations,
/// stereographic landmarks, and observation identity.  The initialization
/// helper's `second` pair is deliberately canonicalized from the sorted
/// observation map here: upstream `NfrMapper::setup_opt()` uses that pair only
/// to initialize the landmark and does not retain it in `Keypoint` state.
/// This keeps the Rust and C++ M8e state hashes over the same optimized state.
pub fn canonical_mapper_state_hash(state: &MapperState) -> u64 {
    let mut hash = 1469598103934665603u64;
    fnv_append_u64(&mut hash, state.se3_poses.len() as u64);
    for (&id, pose) in &state.se3_poses {
        fnv_append_u64(&mut hash, id);
        for value in pose.translation.iter() {
            fnv_append_f64(&mut hash, *value);
        }
        for value in pose.rotation.quaternion().coords.iter() {
            fnv_append_f64(&mut hash, *value);
        }
    }
    fnv_append_u64(&mut hash, state.landmarks.len() as u64);
    for (&id, landmark) in &state.landmarks {
        fnv_append_u64(&mut hash, id);
        fnv_append_u64(&mut hash, landmark.host.frame_id);
        fnv_append_u64(&mut hash, u64::from(landmark.host.cam_id));
        let mut observation_images = landmark
            .observations
            .iter()
            .map(|observation| observation.image)
            .collect::<Vec<_>>();
        observation_images.sort_unstable();
        let second = observation_images.get(1).copied().unwrap_or(landmark.host);
        fnv_append_u64(&mut hash, second.frame_id);
        fnv_append_u64(&mut hash, u64::from(second.cam_id));
        fnv_append_f64(&mut hash, landmark.direction.xy.x);
        fnv_append_f64(&mut hash, landmark.direction.xy.y);
        fnv_append_f64(&mut hash, landmark.inverse_distance);
        fnv_append_u64(&mut hash, landmark.observations.len() as u64);
        for observation in &landmark.observations {
            fnv_append_u64(&mut hash, observation.image.frame_id);
            fnv_append_u64(&mut hash, u64::from(observation.image.cam_id));
            fnv_append_u64(&mut hash, observation.feature_id);
            fnv_append_f64(&mut hash, observation.pixel.x);
            fnv_append_f64(&mut hash, observation.pixel.y);
        }
    }
    hash
}

/// Canonical hash of the per-iteration LM trace (costs, H diagonal, solve,
/// landmark back-substitution, and every accept/reject trial).
pub fn canonical_global_ba_trace_hash(trace: &[GlobalBaIteration]) -> u64 {
    let mut hash = 1469598103934665603u64;
    fnv_append_u64(&mut hash, trace.len() as u64);
    for iteration in trace {
        fnv_append_u64(&mut hash, iteration.iteration as u64);
        for value in [
            iteration.vision_cost,
            iteration.relative_cost,
            iteration.roll_pitch_cost,
            iteration.total_cost,
        ] {
            fnv_append_f64(&mut hash, value);
        }
        fnv_append_u64(&mut hash, iteration.h_diagonal.len() as u64);
        for value in &iteration.h_diagonal {
            fnv_append_f64(&mut hash, *value);
        }
        fnv_append_u64(&mut hash, iteration.pose_solve.len() as u64);
        for value in &iteration.pose_solve {
            fnv_append_f64(&mut hash, *value);
        }
        fnv_append_u64(&mut hash, iteration.landmark_increments.len() as u64);
        for (id, increment) in &iteration.landmark_increments {
            fnv_append_u64(&mut hash, *id);
            for value in increment {
                fnv_append_f64(&mut hash, *value);
            }
        }
        fnv_append_u64(&mut hash, iteration.trials.len() as u64);
        for trial in &iteration.trials {
            for value in [
                trial.lambda,
                trial.f_diff,
                trial.after_vision_cost,
                trial.after_relative_cost,
                trial.after_roll_pitch_cost,
                trial.after_total_cost,
                trial.max_pose_increment,
            ] {
                fnv_append_f64(&mut hash, value);
            }
            fnv_append(&mut hash, [u8::from(trial.accepted)]);
        }
    }
    hash
}

/// Full NFR mapper BA using setup-opt landmarks and a Basalt calibration.
///
/// The legacy [`global_ba`] entry point remains available for factor-only
/// callers; this is the production M8e path.  Landmark observations are
/// grouped by their host, linearized in the anchored stereographic chart,
/// Schur-reduced into the absolute six-DoF pose system, and back-substituted
/// after each accepted/rejected LM trial.
pub fn global_ba_with_landmarks(
    data: &MargData,
    factors: &MapperFactors,
    landmarks: &BTreeMap<u64, MapperLandmark>,
    calibration: &BasaltCalibration,
    config: GlobalBaConfig,
) -> (MapperState, MapperSummary) {
    global_ba_impl(data, factors, landmarks.clone(), Some(calibration), config)
}

/// Convenience form accepting the direct output of [`setup_opt`].
pub fn global_ba_with_setup_opt(
    data: &MargData,
    factors: &MapperFactors,
    setup: &SetupOptResult,
    calibration: &BasaltCalibration,
    config: GlobalBaConfig,
) -> (MapperState, MapperSummary) {
    global_ba_with_landmarks(data, factors, &setup.landmarks, calibration, config)
}

/// Full six-DoF factor BA for callers that have not yet attached mapper
/// landmarks.  Unlike the former scaffold this still optimizes translation,
/// rotation, relative-pose rows, and roll/pitch rows.
pub fn global_ba(
    data: &MargData,
    factors: &MapperFactors,
    config: GlobalBaConfig,
) -> (MapperState, MapperSummary) {
    global_ba_impl(data, factors, BTreeMap::new(), None, config)
}

fn global_ba_impl(
    data: &MargData,
    factors: &MapperFactors,
    landmarks: BTreeMap<u64, MapperLandmark>,
    calibration: Option<&BasaltCalibration>,
    config: GlobalBaConfig,
) -> (MapperState, MapperSummary) {
    let mut poses = initial_mapper_poses(data);
    let mut landmarks = landmarks;
    let mut optimizer = GlobalBaOptimizerState::from_config(config);
    let summary = global_ba_impl_in_place(
        &mut poses,
        factors,
        &mut landmarks,
        calibration,
        config,
        &mut optimizer,
    );
    let anchor = poses.keys().next().copied().unwrap_or(0);
    let state = make_mapper_state(poses, landmarks, anchor);
    (state, summary)
}

/// Run mapper BA directly on persistent pose and landmark maps.
///
/// This is the stateful counterpart to [`global_ba_with_landmarks`].  The
/// maps are mutated in place after every accepted trial (and restored from
/// backups after rejected trials), so a caller such as `NfrMapper` retains the
/// source lifetime and can carry the damping state into its next call.
pub fn global_ba_with_state(
    poses: &mut BTreeMap<u64, SE3>,
    factors: &MapperFactors,
    landmarks: &mut BTreeMap<u64, MapperLandmark>,
    calibration: &BasaltCalibration,
    config: GlobalBaConfig,
    optimizer: &mut GlobalBaOptimizerState,
) -> MapperSummary {
    global_ba_impl_in_place(
        poses,
        factors,
        landmarks,
        Some(calibration),
        config,
        optimizer,
    )
}

/// Local bundle adjustment over a subset of keyframes (a covisibility window).
///
/// `window_frames` lists the frame IDs that stay **variable**; every other pose
/// in `poses` is treated as fixed. This localizes `global_ba_impl_in_place` to
/// the window by (a) restricting `pose_indices` to the window and (b) letting
/// out-of-window observations drop out, since every observation whose pose is
/// not in the window is skipped by `linearize_mapper_observation`. Landmarks
/// are shared, so a landmark observed by both window and non-window frames is
/// optimized against its window observations only. The window's oldest frame is
/// the gauge anchor.
///
/// This is the cheap, solver-reusing local-BA step for the online mapper: it
/// keeps the same factor machinery as global BA while solving a much smaller
/// pose block. It returns the same [`MapperSummary`] shape; `poses` is updated
/// in place only for the window frames.
pub fn local_ba_with_state(
    poses: &mut BTreeMap<u64, SE3>,
    factors: &MapperFactors,
    landmarks: &mut BTreeMap<u64, MapperLandmark>,
    window_frames: &BTreeSet<u64>,
    calibration: &BasaltCalibration,
    config: GlobalBaConfig,
    optimizer: &mut GlobalBaOptimizerState,
) -> MapperSummary {
    let mut window_poses = poses
        .iter()
        .filter(|(id, _)| window_frames.contains(id))
        .map(|(&id, pose)| (id, pose.clone()))
        .collect::<BTreeMap<_, _>>();
    let summary = global_ba_impl_in_place(
        &mut window_poses,
        factors,
        landmarks,
        Some(calibration),
        config,
        optimizer,
    );
    for (id, pose) in window_poses {
        poses.insert(id, pose);
    }
    summary
}

fn global_ba_impl_in_place(
    poses: &mut BTreeMap<u64, SE3>,
    factors: &MapperFactors,
    landmarks: &mut BTreeMap<u64, MapperLandmark>,
    calibration: Option<&BasaltCalibration>,
    config: GlobalBaConfig,
    optimizer: &mut GlobalBaOptimizerState,
) -> MapperSummary {
    let pose_indices = poses
        .keys()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect::<BTreeMap<_, _>>();
    let anchor = poses.keys().next().copied().unwrap_or(0);
    let anchor_index = pose_indices.get(&anchor).copied();
    let initial_breakdown = evaluate_costs(poses, landmarks, factors, calibration, config);
    if !config.enabled {
        let track_count = landmarks.len();
        let state = make_mapper_state(poses.clone(), landmarks.clone(), anchor);
        let final_state_hash = canonical_mapper_state_hash(&state);
        let trace_hash = canonical_global_ba_trace_hash(&[]);
        return MapperSummary {
            initial_cost: initial_breakdown.total(),
            final_cost: initial_breakdown.total(),
            iterations: 0,
            gauge_anchor: anchor,
            pose_count: pose_indices.len(),
            track_count,
            disabled: true,
            trace: Vec::new(),
            final_lambda: optimizer.lambda,
            final_state_hash,
            trace_hash,
        };
    }

    let mut cost = initial_breakdown;
    let mut trace = Vec::new();
    let mut iterations = 0;
    for iteration in 0..config.max_iterations {
        let mut vision = linearize_vision(
            poses,
            landmarks,
            calibration,
            &pose_indices,
            pose_indices.len(),
            config,
        );
        linearize_factors(
            &poses,
            factors,
            &pose_indices,
            pose_indices.len(),
            config,
            &mut vision.h,
            &mut vision.b,
        );
        let h_diagonal = vision.h.diagonal();
        let damping_diagonal = h_diagonal.clone();
        let mut solve = solve_damped_block_system(
            &vision.h,
            &vision.b,
            &damping_diagonal,
            optimizer.lambda,
            optimizer.min_lambda,
            anchor_index,
        );
        let mut max_increment = solve.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let mut iteration_trace = GlobalBaIteration {
            iteration,
            vision_cost: cost.vision,
            relative_cost: cost.relative,
            roll_pitch_cost: cost.roll_pitch,
            total_cost: cost.total(),
            h_diagonal,
            pose_solve: solve.iter().copied().collect(),
            landmark_increments: Vec::new(),
            trials: Vec::new(),
        };
        // The upstream loop computes `converged` after solving but still
        // applies that small final increment once inside the LM trial.  The
        // flag only prevents the *next* outer iteration; initializing it
        // false here preserves that source ordering.
        let mut converged = false;
        let mut accepted_step = false;
        let mut trial_count = 10;
        if config.use_lm {
            while !accepted_step && trial_count > 0 && !converged {
                // Upstream `NfrMapper::optimize` solves the damped system
                // inside this loop, so every retry uses the lambda raised by
                // the previous rejection. Re-applying the first increment
                // instead made one rejection cascade through all ten trials
                // (observed: 80 rejected trials in one V2_01 final optimize).
                // The first trial keeps the solve above, so an iteration
                // accepted on its first trial is unchanged.
                // Retries also damp the landmark blocks. Otherwise every
                // retry re-applies the full undamped landmark Newton step
                // `-Hll^-1 bl`, however small the pose step, and the trial can
                // never succeed (observed on V2_01: pose increments of 1e-5
                // still raised the vision cost by 2.8e4 on every trial).
                let mut damped_inverses = None;
                if trial_count < 10 {
                    let (h, b, inverses) = damped_landmark_system(
                        &vision.h,
                        &vision.b,
                        &vision.landmarks,
                        optimizer.lambda,
                        optimizer.min_lambda,
                    );
                    solve = solve_damped_block_system(
                        &h,
                        &b,
                        &damping_diagonal,
                        optimizer.lambda,
                        optimizer.min_lambda,
                        anchor_index,
                    );
                    max_increment = solve.iter().map(|v| v.abs()).fold(0.0, f64::max);
                    damped_inverses = Some(inverses);
                }
                converged = max_increment < 1e-5;
                let backup_poses = poses.clone();
                let backup_landmarks = landmarks.clone();
                apply_pose_solve(poses, &pose_indices, &solve);
                let increments = back_substitute_landmarks_with(
                    landmarks,
                    &vision.landmarks,
                    damped_inverses.as_deref(),
                    &pose_indices,
                    &solve,
                );
                if iteration_trace.landmark_increments.is_empty() {
                    iteration_trace.landmark_increments = increments;
                }
                let after = evaluate_costs(poses, landmarks, factors, calibration, config);
                let f_diff = cost.total() - after.total();
                let accepted = f_diff >= 0.0;
                iteration_trace.trials.push(GlobalBaTrial {
                    lambda: optimizer.lambda,
                    f_diff,
                    after_vision_cost: after.vision,
                    after_relative_cost: after.relative,
                    after_roll_pitch_cost: after.roll_pitch,
                    after_total_cost: after.total(),
                    max_pose_increment: max_increment,
                    accepted,
                });
                if accepted {
                    cost = after;
                    optimizer.lambda = (optimizer.lambda / 3.0).max(optimizer.min_lambda);
                    optimizer.lambda_vee = 2.0;
                    accepted_step = true;
                } else {
                    *poses = backup_poses;
                    *landmarks = backup_landmarks;
                    optimizer.lambda =
                        (optimizer.lambda_vee * optimizer.lambda).min(optimizer.max_lambda);
                    optimizer.lambda_vee *= 2.0;
                }
                trial_count -= 1;
            }
        } else {
            // Source Gauss-Newton branch: one unconditionally applied solve,
            // with no LM trial accounting or restore.
            converged = max_increment < 1e-5;
            apply_pose_solve(poses, &pose_indices, &solve);
            let increments =
                back_substitute_landmarks(landmarks, &vision.landmarks, &pose_indices, &solve);
            iteration_trace.landmark_increments = increments;
            // The upstream GN branch does not emit an accept/reject trial,
            // but the stateful report still needs its final real cost to
            // describe the maps it just persisted.
            cost = evaluate_costs(poses, landmarks, factors, calibration, config);
        }
        iterations = iteration + 1;
        trace.push(iteration_trace);
        // `optimize()` in the pinned NFR mapper does not terminate the outer
        // loop merely because all ten LM trials were rejected; it carries the
        // raised lambda into the next linearization.  Only its small-step
        // convergence flag ends the outer iteration schedule.
        if converged {
            break;
        }
    }
    let final_cost = cost.total();
    let state = make_mapper_state(poses.clone(), landmarks.clone(), anchor);
    let final_state_hash = canonical_mapper_state_hash(&state);
    let trace_hash = canonical_global_ba_trace_hash(&trace);
    let summary = MapperSummary {
        initial_cost: initial_breakdown.total(),
        final_cost,
        iterations,
        gauge_anchor: anchor,
        pose_count: pose_indices.len(),
        track_count: state.landmarks.len(),
        disabled: false,
        trace,
        final_lambda: optimizer.lambda,
        final_state_hash,
        trace_hash,
    };
    summary
}

// ===================== Joint VI global BA (15-dof/keyframe) =====================
//
// Everything below is additive: [`global_ba_impl_in_place`] above (and every
// helper it calls: [`PoseBlockHessian`], [`linearize_factors`],
// [`linearize_vision`], [`apply_pose_solve`], [`damped_landmark_system`],
// [`back_substitute_landmarks_with`], [`solve_damped_block_system`]) is
// unmodified and still the only code path `global_ba_with_state`/
// `local_ba_with_state` use. [`global_ba_with_state_joint`] is a new, opt-in
// entry point that extends the same vision/relative-pose/roll-pitch factor
// machinery -- reused unchanged -- to a 15-dof-per-keyframe navigation state
// (pose, velocity, gyro bias, accel bias) by embedding the existing 6-dof
// pose system into the top-left 6x6 of each 15x15 nav-state block (vision
// and the existing relative-pose/roll-pitch factors only ever touch pose)
// and adding preintegrated-IMU and bias-random-walk factor blocks from
// `mapper::imu_ba` on top.

/// Block-sparse 15-dof (pose + velocity + gyro bias + accel bias) Hessian
/// for the joint VI global BA. Mirrors [`PoseBlockHessian`]'s block-sparse
/// layout (a `BTreeMap` of dense blocks; an absent block is an exact zero)
/// at the larger per-keyframe dof count, using dynamically-sized 15x15
/// blocks rather than a fixed `Matrix6`-style type to keep this addition
/// small; see [`embed_pose_hessian_into_nav`] for how the unchanged 6-dof
/// system embeds into it.
#[derive(Debug, Clone)]
struct NavBlockHessian {
    nav_count: usize,
    blocks: BTreeMap<(usize, usize), DMatrix<f64>>,
}

impl NavBlockHessian {
    fn new(nav_count: usize) -> Self {
        Self {
            nav_count,
            blocks: BTreeMap::new(),
        }
    }

    fn dim(&self) -> usize {
        self.nav_count * imu_ba::NAV_JOINT_DOF
    }

    fn add_block(&mut self, row: usize, col: usize, block: &DMatrix<f64>) {
        self.blocks
            .entry((row, col))
            .or_insert_with(|| DMatrix::zeros(imu_ba::NAV_JOINT_DOF, imu_ba::NAV_JOINT_DOF))
            .add_assign(block);
    }

    fn diagonal(&self) -> Vec<f64> {
        let d = imu_ba::NAV_JOINT_DOF;
        let mut diagonal = vec![0.0; self.dim()];
        for nav in 0..self.nav_count {
            if let Some(block) = self.blocks.get(&(nav, nav)) {
                for axis in 0..d {
                    diagonal[nav * d + axis] = block[(axis, axis)];
                }
            }
        }
        diagonal
    }

    fn to_dense(&self) -> DMatrix<f64> {
        let d = imu_ba::NAV_JOINT_DOF;
        let mut dense = DMatrix::zeros(self.dim(), self.dim());
        for (&(row, col), block) in &self.blocks {
            dense.view_mut((row * d, col * d), (d, d)).copy_from(block);
        }
        dense
    }

    /// See [`PoseBlockHessian::symmetrized_damped`]; identical construction
    /// at the 15-dof block size.
    fn symmetrized_damped(&self, h_diagonal: &[f64], lambda: f64, min_lambda: f64) -> Self {
        let d = imu_ba::NAV_JOINT_DOF;
        let zero = DMatrix::zeros(d, d);
        let mut keys = self.blocks.keys().copied().collect::<BTreeSet<_>>();
        keys.extend(self.blocks.keys().map(|&(row, col)| (col, row)));
        for nav in 0..self.nav_count {
            keys.insert((nav, nav));
        }
        let mut blocks = BTreeMap::new();
        for (row, col) in keys {
            let upper = self.blocks.get(&(row, col)).unwrap_or(&zero);
            let lower = self.blocks.get(&(col, row)).unwrap_or(&zero);
            let mut block = DMatrix::from_fn(d, d, |r, c| (upper[(r, c)] + lower[(c, r)]) * 0.5);
            if row == col {
                for axis in 0..d {
                    let index = row * d + axis;
                    block[(axis, axis)] += (h_diagonal[index] * lambda).max(min_lambda);
                }
            }
            blocks.insert((row, col), block);
        }
        Self {
            nav_count: self.nav_count,
            blocks,
        }
    }

    fn mul_vector(&self, x: &DVector<f64>) -> DVector<f64> {
        let d = imu_ba::NAV_JOINT_DOF;
        let mut y = DVector::zeros(self.dim());
        for (&(row, col), block) in &self.blocks {
            for c in 0..d {
                let value = x[col * d + c];
                for r in 0..d {
                    y[row * d + r] += block[(r, c)] * value;
                }
            }
        }
        y
    }
}

/// Embed the existing 6-dof pose Hessian/gradient (vision + relative-pose +
/// roll-pitch, unchanged) into the top-left 6x6 of each 15x15 nav-state
/// block. `pose_h`'s block indices are keyframe indices in the same
/// enumeration order as the joint solver's own `pose_indices`, so no
/// re-indexing is needed.
fn embed_pose_hessian_into_nav(
    pose_h: &PoseBlockHessian,
    pose_b: &DVector<f64>,
    nav_count: usize,
) -> (NavBlockHessian, DVector<f64>) {
    let d = imu_ba::NAV_JOINT_DOF;
    let mut nav_h = NavBlockHessian::new(nav_count);
    for (&(row, col), block6) in &pose_h.blocks {
        let mut block15 = DMatrix::zeros(d, d);
        block15.view_mut((0, 0), (6, 6)).copy_from(block6);
        nav_h.add_block(row, col, &block15);
    }
    let mut nav_b = DVector::zeros(nav_count * d);
    for pose in 0..nav_count {
        if pose * 6 + 6 <= pose_b.len() {
            for axis in 0..6 {
                nav_b[pose * d + axis] = pose_b[pose * 6 + axis];
            }
        }
    }
    (nav_h, nav_b)
}

/// Extract the pose-only (6-dof/keyframe) sub-vector of a joint 15-dof
/// solve, for feeding into the unchanged [`apply_pose_solve`] and
/// [`back_substitute_landmarks_with`].
fn pose_solve_from_nav_solve(nav_solve: &DVector<f64>, pose_count: usize) -> DVector<f64> {
    let d = imu_ba::NAV_JOINT_DOF;
    let mut pose_solve = DVector::zeros(pose_count * 6);
    for pose in 0..pose_count {
        if pose * d + 6 <= nav_solve.len() {
            for axis in 0..6 {
                pose_solve[pose * 6 + axis] = nav_solve[pose * d + axis];
            }
        }
    }
    pose_solve
}

/// Apply the velocity/gyro-bias/accel-bias (offsets 6..15) part of a joint
/// solve, in the same `param -= x` convention as [`apply_pose_solve`]'s
/// pose update (and as `mapper::imu_ba`'s embedded factor gradient itself
/// expects; see that module's doc comment).
fn apply_nav_extra_solve(
    pose_indices: &BTreeMap<u64, usize>,
    nav_solve: &DVector<f64>,
    velocities: &mut BTreeMap<u64, Vector3<f64>>,
    gyro_biases: &mut BTreeMap<u64, Vector3<f64>>,
    accel_biases: &mut BTreeMap<u64, Vector3<f64>>,
) {
    let d = imu_ba::NAV_JOINT_DOF;
    for (&id, &index) in pose_indices {
        let offset = index * d;
        if offset + d > nav_solve.len() {
            continue;
        }
        let v = velocities.entry(id).or_default();
        for axis in 0..3 {
            v[axis] -= nav_solve[offset + 6 + axis];
        }
        let bg = gyro_biases.entry(id).or_default();
        for axis in 0..3 {
            bg[axis] -= nav_solve[offset + 9 + axis];
        }
        let ba = accel_biases.entry(id).or_default();
        for axis in 0..3 {
            ba[axis] -= nav_solve[offset + 12 + axis];
        }
    }
}

fn nav_states_at(
    poses: &BTreeMap<u64, SE3>,
    velocities: &BTreeMap<u64, Vector3<f64>>,
    gyro_biases: &BTreeMap<u64, Vector3<f64>>,
    accel_biases: &BTreeMap<u64, Vector3<f64>>,
) -> BTreeMap<u64, imu_ba::NavState> {
    poses
        .iter()
        .map(|(&id, pose)| {
            (
                id,
                imu_ba::NavState::new(
                    pose.clone(),
                    velocities.get(&id).copied().unwrap_or_default(),
                    gyro_biases.get(&id).copied().unwrap_or_default(),
                    accel_biases.get(&id).copied().unwrap_or_default(),
                ),
            )
        })
        .collect()
}

fn evaluate_imu_joint_cost(
    nav_states: &BTreeMap<u64, imu_ba::NavState>,
    imu_factors: &[imu_ba::ImuPreintegratedFactor],
    bias_factors: &[imu_ba::BiasRandomWalkFactor],
    gravity_world: Vector3<f64>,
) -> f64 {
    let mut cost = 0.0;
    for factor in imu_factors {
        cost += imu_ba::imu_preintegrated_cost(nav_states, factor, gravity_world);
    }
    for factor in bias_factors {
        cost += imu_ba::bias_random_walk_cost(nav_states, factor);
    }
    cost
}

/// Linearize every preintegrated-IMU and bias-random-walk factor at the
/// current `nav_states`, accumulating `H += J^T J`, `b += J^T r` into the
/// four 15x15 blocks / two 15-vector segments touching each factor's
/// `from`/`to` keyframe (see `mapper::imu_ba::Nav15FactorLinearization`). A
/// factor whose keyframe is outside `pose_indices` (or that fails to
/// linearize, e.g. a non-finite preintegrated delta) is silently skipped,
/// matching every other factor loop in this module.
fn linearize_imu_joint_factors(
    nav_states: &BTreeMap<u64, imu_ba::NavState>,
    pose_indices: &BTreeMap<u64, usize>,
    imu_factors: &[imu_ba::ImuPreintegratedFactor],
    bias_factors: &[imu_ba::BiasRandomWalkFactor],
    gravity_world: Vector3<f64>,
    nav_count: usize,
    h: &mut NavBlockHessian,
    b: &mut DVector<f64>,
) {
    let d = imu_ba::NAV_JOINT_DOF;
    let mut accumulate = |from_id: u64, to_id: u64, lin: imu_ba::Nav15FactorLinearization| {
        let (Some(&i), Some(&j)) = (pose_indices.get(&from_id), pose_indices.get(&to_id)) else {
            return;
        };
        if i >= nav_count || j >= nav_count {
            return;
        }
        h.add_block(i, i, &lin.h_ii);
        h.add_block(i, j, &lin.h_ij);
        h.add_block(j, i, &lin.h_ji);
        h.add_block(j, j, &lin.h_jj);
        for axis in 0..d {
            b[i * d + axis] += lin.b_i[axis];
            b[j * d + axis] += lin.b_j[axis];
        }
    };
    for factor in imu_factors {
        if let Some(lin) =
            imu_ba::linearize_imu_preintegrated_factor(nav_states, factor, gravity_world)
        {
            accumulate(factor.from, factor.to, lin);
        }
    }
    for factor in bias_factors {
        if let Some(lin) = imu_ba::linearize_bias_random_walk_factor(nav_states, factor) {
            accumulate(factor.from, factor.to, lin);
        }
    }
}

fn solve_damped_nav_block_system(
    h: &NavBlockHessian,
    b: &DVector<f64>,
    h_diagonal: &[f64],
    lambda: f64,
    min_lambda: f64,
    gauge_anchor: Option<usize>,
) -> DVector<f64> {
    let system = h.symmetrized_damped(h_diagonal, lambda, min_lambda);
    let finite = system
        .blocks
        .values()
        .all(|block| block.iter().all(|value| value.is_finite()));
    if finite && b.len() == system.dim() {
        let diagonal = system.diagonal();
        if let Some(solution) = preconditioned_cg(
            b,
            1e-4,
            |index| diagonal[index],
            |vector| system.mul_vector(vector),
        ) {
            return solution;
        }
    }
    solve_damped_system(
        &h.to_dense(),
        b,
        h_diagonal,
        lambda,
        min_lambda,
        gauge_anchor,
    )
}

/// Report from [`global_ba_with_state_joint`]. Deliberately a separate,
/// smaller type from [`MapperSummary`] (no per-iteration trace/hash
/// bookkeeping): this is a new opt-in code path with no existing
/// byte-exactness contract to preserve, unlike the pose-only solver above.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointGlobalBaSummary {
    pub initial_cost: f64,
    pub final_cost: f64,
    /// `initial_cost`'s split between the vision/relative-pose/roll-pitch
    /// terms (identical to the pose-only solver's own cost) and the new
    /// preintegrated-IMU/bias-random-walk terms. Diagnostic only, to make
    /// the relative scale of the two factor families visible without a
    /// separate instrumentation pass -- `initial_pose_factor_cost +
    /// initial_imu_factor_cost == initial_cost`.
    pub initial_pose_factor_cost: f64,
    pub initial_imu_factor_cost: f64,
    pub iterations: usize,
    pub accepted_step_count: usize,
    pub rejected_trial_count: usize,
    pub pose_count: usize,
    pub track_count: usize,
    pub final_lambda: f64,
    pub gauge_anchor: u64,
}

/// Joint visual-inertial global bundle adjustment: extends the pose-only
/// (6 dof/keyframe) solve above to the full 15-dof-per-keyframe navigation
/// state (pose, velocity, gyro bias, accel bias), adding preintegrated-IMU
/// relative-state factors and bias random-walk factors between consecutive
/// keyframes (`mapper::imu_ba`) on top of the unchanged vision/relative-pose/
/// roll-pitch factors. Gravity is fixed in world (`gravity_world`); this
/// module never optimizes it, matching the VIO's own gravity-aligned world
/// frame.
///
/// `velocities`/`gyro_biases`/`accel_biases` are updated in place, seeded
/// from whatever the caller passes in (typically the VIO's own per-keyframe
/// `MargData` estimate; a missing id defaults to zero). `poses` and
/// `landmarks` are updated in place exactly as
/// `global_ba_impl_in_place`'s pose-only solve does.
#[allow(clippy::too_many_arguments)]
pub fn global_ba_with_state_joint(
    poses: &mut BTreeMap<u64, SE3>,
    velocities: &mut BTreeMap<u64, Vector3<f64>>,
    gyro_biases: &mut BTreeMap<u64, Vector3<f64>>,
    accel_biases: &mut BTreeMap<u64, Vector3<f64>>,
    factors: &MapperFactors,
    imu_factors: &[imu_ba::ImuPreintegratedFactor],
    bias_factors: &[imu_ba::BiasRandomWalkFactor],
    landmarks: &mut BTreeMap<u64, MapperLandmark>,
    calibration: &BasaltCalibration,
    config: GlobalBaConfig,
    gravity_world: Vector3<f64>,
    optimizer: &mut GlobalBaOptimizerState,
) -> JointGlobalBaSummary {
    let pose_indices = poses
        .keys()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect::<BTreeMap<_, _>>();
    let pose_count = pose_indices.len();
    let anchor = poses.keys().next().copied().unwrap_or(0);
    let anchor_index = pose_indices.get(&anchor).copied();

    let nav_states = nav_states_at(poses, velocities, gyro_biases, accel_biases);
    let initial_pose_factor_cost =
        evaluate_costs(poses, landmarks, factors, Some(calibration), config).total();
    let initial_imu_factor_cost =
        evaluate_imu_joint_cost(&nav_states, imu_factors, bias_factors, gravity_world);
    let initial_cost = initial_pose_factor_cost + initial_imu_factor_cost;

    if !config.enabled {
        return JointGlobalBaSummary {
            initial_cost,
            initial_pose_factor_cost,
            initial_imu_factor_cost,
            final_cost: initial_cost,
            iterations: 0,
            accepted_step_count: 0,
            rejected_trial_count: 0,
            pose_count,
            track_count: landmarks.len(),
            final_lambda: optimizer.lambda,
            gauge_anchor: anchor,
        };
    }

    let mut cost = initial_cost;
    let mut iterations = 0;
    let mut accepted_step_count = 0;
    let mut rejected_trial_count = 0;

    for iteration in 0..config.max_iterations {
        let nav_states = nav_states_at(poses, velocities, gyro_biases, accel_biases);
        let mut vision = linearize_vision(
            poses,
            landmarks,
            Some(calibration),
            &pose_indices,
            pose_count,
            config,
        );
        linearize_factors(
            poses,
            factors,
            &pose_indices,
            pose_count,
            config,
            &mut vision.h,
            &mut vision.b,
        );

        let mut imu_h = NavBlockHessian::new(pose_count);
        let mut imu_b = DVector::zeros(pose_count * imu_ba::NAV_JOINT_DOF);
        linearize_imu_joint_factors(
            &nav_states,
            &pose_indices,
            imu_factors,
            bias_factors,
            gravity_world,
            pose_count,
            &mut imu_h,
            &mut imu_b,
        );

        let mut converged = false;
        let mut accepted_step = false;
        let mut trial_count = 10;
        while !accepted_step && trial_count > 0 && !converged {
            let (damped_pose_h, damped_pose_b, damped_inverses) = if trial_count < 10 {
                let (h, b, inverses) = damped_landmark_system(
                    &vision.h,
                    &vision.b,
                    &vision.landmarks,
                    optimizer.lambda,
                    optimizer.min_lambda,
                );
                (h, b, Some(inverses))
            } else {
                (vision.h.clone(), vision.b.clone(), None)
            };

            let (mut nav_h, mut nav_b) =
                embed_pose_hessian_into_nav(&damped_pose_h, &damped_pose_b, pose_count);
            for (&(row, col), block) in &imu_h.blocks {
                nav_h.add_block(row, col, block);
            }
            for index in 0..nav_b.len() {
                nav_b[index] += imu_b[index];
            }

            let h_diagonal = nav_h.diagonal();
            let solve = solve_damped_nav_block_system(
                &nav_h,
                &nav_b,
                &h_diagonal,
                optimizer.lambda,
                optimizer.min_lambda,
                anchor_index,
            );
            let max_increment = solve.iter().map(|v| v.abs()).fold(0.0, f64::max);
            converged = max_increment < 1e-5;

            let backup_poses = poses.clone();
            let backup_landmarks = landmarks.clone();
            let backup_velocities = velocities.clone();
            let backup_gyro_biases = gyro_biases.clone();
            let backup_accel_biases = accel_biases.clone();

            let pose_solve = pose_solve_from_nav_solve(&solve, pose_count);
            apply_pose_solve(poses, &pose_indices, &pose_solve);
            back_substitute_landmarks_with(
                landmarks,
                &vision.landmarks,
                damped_inverses.as_deref(),
                &pose_indices,
                &pose_solve,
            );
            apply_nav_extra_solve(&pose_indices, &solve, velocities, gyro_biases, accel_biases);

            let after_nav_states = nav_states_at(poses, velocities, gyro_biases, accel_biases);
            let after = evaluate_costs(poses, landmarks, factors, Some(calibration), config)
                .total()
                + evaluate_imu_joint_cost(
                    &after_nav_states,
                    imu_factors,
                    bias_factors,
                    gravity_world,
                );
            let f_diff = cost - after;
            let accepted = f_diff >= 0.0;
            if accepted {
                cost = after;
                optimizer.lambda = (optimizer.lambda / 3.0).max(optimizer.min_lambda);
                optimizer.lambda_vee = 2.0;
                accepted_step = true;
                accepted_step_count += 1;
            } else {
                *poses = backup_poses;
                *landmarks = backup_landmarks;
                *velocities = backup_velocities;
                *gyro_biases = backup_gyro_biases;
                *accel_biases = backup_accel_biases;
                optimizer.lambda =
                    (optimizer.lambda_vee * optimizer.lambda).min(optimizer.max_lambda);
                optimizer.lambda_vee *= 2.0;
                rejected_trial_count += 1;
            }
            trial_count -= 1;
        }
        iterations = iteration + 1;
        if converged {
            break;
        }
    }

    JointGlobalBaSummary {
        initial_cost,
        initial_pose_factor_cost,
        initial_imu_factor_cost,
        final_cost: cost,
        iterations,
        accepted_step_count,
        rejected_trial_count,
        pose_count,
        track_count: landmarks.len(),
        final_lambda: optimizer.lambda,
        gauge_anchor: anchor,
    }
}

pub fn load_margdata_directory(path: &std::path::Path) -> Result<Vec<MargData>, String> {
    let mut files = std::fs::read_dir(path)
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    files.sort_by_key(|e| e.file_name());
    files
        .into_iter()
        .map(|e| {
            let path = e.path();
            let data =
                MargData::from_json(&std::fs::read_to_string(&path).map_err(|x| x.to_string())?)
                    .map_err(|x| x.to_string())?;
            data.validate_contract()
                .map_err(|error| format!("{}: {error}", path.display()))?;
            Ok(data)
        })
        .collect()
}
pub fn mapper_summary_text(summary: &MapperSummary) -> String {
    format!(
        "poses={} tracks={} cost={:.9}->{:.9} iterations={} gauge={} disabled={}",
        summary.pose_count,
        summary.track_count,
        summary.initial_cost,
        summary.final_cost,
        summary.iterations,
        summary.gauge_anchor,
        summary.disabled
    )
}
impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            relative_pose_weight: 1.,
            roll_pitch_weight: 1.,
            ba_weight: 1.,
            relative_pose_max_neighbors: 0,
        }
    }
}

fn matrix_from_data(data: &MatrixData) -> Option<DMatrix<f64>> {
    if data.data.len() != data.rows.checked_mul(data.cols)? {
        return None;
    }
    // MatrixData is emitted from nalgebra's column-major iterator.  H is
    // symmetric, but retaining this convention also makes the helper useful
    // for callers that provide a non-symmetric diagnostic matrix.
    Some(DMatrix::from_column_slice(data.rows, data.cols, &data.data))
}

fn matrix_to_data(matrix: &DMatrix<f64>) -> MatrixData {
    MatrixData {
        rows: matrix.nrows(),
        cols: matrix.ncols(),
        data: matrix.iter().copied().collect(),
    }
}

fn refresh_square_root(data: &mut MargData, h: &DMatrix<f64>, b: &DVector<f64>) {
    if data.aom_sqrt_jacobian.cols == h.ncols() {
        return;
    }
    // Rust MargData carries both the upstream absolute system and the
    // estimator's square-root snapshot.  Upstream NFR only serializes the
    // former, so after a dimension-changing mapper reduction regenerate a
    // deterministic square root that keeps validate()/round-trips honest.
    if let Some(cholesky) = h.clone().cholesky() {
        let lower = cholesky.l();
        let jacobian = lower.transpose();
        let rhs = lower
            .lu()
            .solve(b)
            .unwrap_or_else(|| DVector::zeros(h.nrows()));
        data.aom_sqrt_jacobian = matrix_to_data(&jacobian);
        data.aom_sqrt_rhs = rhs.iter().copied().collect();
    } else {
        data.aom_sqrt_jacobian = MatrixData::new(0, h.ncols(), Vec::new()).unwrap();
        data.aom_sqrt_rhs.clear();
    }
}

fn abs_system(data: &MargData) -> Result<(DMatrix<f64>, DVector<f64>), MargDataProcessError> {
    let h = if let Some(h) = data.aom_abs_h.as_ref() {
        matrix_from_data(h).ok_or(MargDataProcessError::InvalidMatrix)?
    } else {
        let j =
            matrix_from_data(&data.aom_sqrt_jacobian).ok_or(MargDataProcessError::InvalidMatrix)?;
        if j.ncols() != data.aom_sqrt_jacobian.cols {
            return Err(MargDataProcessError::InvalidMatrix);
        }
        j.transpose() * j
    };
    if h.nrows() != h.ncols() {
        return Err(MargDataProcessError::InvalidMatrix);
    }
    let b = if let Some(b) = data.aom_abs_b.as_ref() {
        DVector::from_column_slice(b)
    } else {
        let j =
            matrix_from_data(&data.aom_sqrt_jacobian).ok_or(MargDataProcessError::InvalidMatrix)?;
        if j.ncols() != h.ncols() || data.aom_sqrt_rhs.len() != j.nrows() {
            return Err(MargDataProcessError::InvalidMatrix);
        }
        j.transpose() * DVector::from_column_slice(&data.aom_sqrt_rhs)
    };
    if b.len() != h.ncols() {
        return Err(MargDataProcessError::InvalidMatrix);
    }
    if h.iter().any(|value| !value.is_finite()) || b.iter().any(|value| !value.is_finite()) {
        return Err(MargDataProcessError::InvalidMatrix);
    }
    Ok((h, b))
}

fn matrix_rank(matrix: &DMatrix<f64>) -> usize {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return 0;
    }
    let scale = matrix
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0);
    // Eigen's default FullPivHouseholderQR threshold is machine epsilon times
    // the larger matrix dimension and the matrix scale.
    matrix
        .clone()
        .svd(false, false)
        .rank(f64::EPSILON * matrix.nrows().max(matrix.ncols()) as f64 * scale)
}

/// Apply Basalt's M8a `NfrMapper::processMargData` reduction in place.
///
/// The input is the mixed pose/state absolute system.  Existing pose blocks
/// remain untouched; a keyframe state contributes its six pose columns while
/// its velocity/bias columns are Schur-eliminated; non-keyframe states are
/// fully eliminated.  The Schur operation intentionally uses a pseudoinverse
/// for the leaving block, matching `MargHelper::marginalizeHelperSqToSq`.
pub fn process_marg_data(
    data: &mut MargData,
) -> Result<MargDataProcessReport, MargDataProcessError> {
    let (mut h, mut b) = abs_system(data)?;
    let input_size = h.ncols();
    let input_rank = matrix_rank(&h);

    let blocks = if data.aom_order.is_empty() {
        let mut offset = 0usize;
        data.aom_block_order()
            .into_iter()
            .map(|(frame_id, dof)| {
                let block = AomBlockData {
                    frame_id,
                    offset,
                    dof,
                    kind: if dof == POSE_DOF { "pose" } else { "state" }.into(),
                };
                offset += dof;
                block
            })
            .collect::<Vec<_>>()
    } else {
        data.aom_order.clone()
    };
    if blocks.is_empty() && input_size != 0 {
        return Err(MargDataProcessError::InvalidAomOrder);
    }
    let mut expected_offset = 0usize;
    for block in &blocks {
        if block.offset != expected_offset
            || block.offset.checked_add(block.dof).is_none()
            || block.offset + block.dof > input_size
        {
            return Err(MargDataProcessError::InvalidAomOrder);
        }
        expected_offset += block.dof;
        if block.dof != POSE_DOF && block.dof != NAV_STATE_DOF {
            return Err(MargDataProcessError::UnknownBlockSize);
        }
    }
    if expected_offset != input_size {
        return Err(MargDataProcessError::InvalidAomOrder);
    }

    let kfs_all_ids = if !data.kfs_all.is_empty() {
        data.kfs_all.clone()
    } else if !data.keyframes.is_empty() {
        data.keyframes.clone()
    } else {
        data.frame_states
            .iter()
            .filter(|state| state.is_keyframe)
            .map(|state| state.frame_id)
            .collect::<Vec<_>>()
    };
    let kfs_all = kfs_all_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();

    let mut keep = Vec::new();
    let mut marg = Vec::new();
    let mut output_blocks = Vec::new();
    let mut output_offset = 0usize;
    let mut frame_poses = data.frame_poses.clone();
    let mut frame_states = data.frame_states.clone();

    for block in &blocks {
        match block.dof {
            POSE_DOF => {
                keep.extend(block.offset..block.offset + POSE_DOF);
                output_blocks.push(AomBlockData {
                    frame_id: block.frame_id,
                    offset: output_offset,
                    dof: POSE_DOF,
                    kind: "pose".into(),
                });
                output_offset += POSE_DOF;
            }
            NAV_STATE_DOF if kfs_all.contains(&block.frame_id) => {
                keep.extend(block.offset..block.offset + POSE_DOF);
                marg.extend(block.offset + POSE_DOF..block.offset + NAV_STATE_DOF);
                output_blocks.push(AomBlockData {
                    frame_id: block.frame_id,
                    offset: output_offset,
                    dof: POSE_DOF,
                    kind: "pose".into(),
                });
                output_offset += POSE_DOF;

                let Some(state) = frame_states.iter().find(|s| s.frame_id == block.frame_id) else {
                    return Err(MargDataProcessError::MissingKeyframeState);
                };
                let pose = FramePoseData {
                    frame_id: state.frame_id,
                    timestamp_ns: state.timestamp_ns,
                    pose: state.pose,
                    is_keyframe: state.is_keyframe,
                };
                if let Some(existing) = frame_poses
                    .iter_mut()
                    .find(|existing| existing.frame_id == pose.frame_id)
                {
                    *existing = pose;
                } else {
                    frame_poses.push(pose);
                }
                frame_states.retain(|state| state.frame_id != block.frame_id);
            }
            NAV_STATE_DOF => {
                marg.extend(block.offset..block.offset + NAV_STATE_DOF);
                frame_states.retain(|state| state.frame_id != block.frame_id);
            }
            _ => unreachable!(),
        }
    }

    if !marg.is_empty() {
        let keep_size = keep.len();
        let marg_size = marg.len();
        let mut permutation = Vec::with_capacity(input_size);
        permutation.extend_from_slice(&keep);
        permutation.extend_from_slice(&marg);
        let mut perm_h = DMatrix::zeros(input_size, input_size);
        for (new_col, &old_col) in permutation.iter().enumerate() {
            for (new_row, &old_row) in permutation.iter().enumerate() {
                perm_h[(new_row, new_col)] = h[(old_row, old_col)];
            }
        }
        let perm_b =
            DVector::from_iterator(input_size, permutation.iter().map(|&column| b[column]));
        let h_mm = perm_h.view((keep_size, keep_size), (marg_size, marg_size));
        let h_km = perm_h.view((0, keep_size), (keep_size, marg_size));
        let h_mk = perm_h.view((keep_size, 0), (marg_size, keep_size));
        let b_m = perm_b.rows(keep_size, marg_size);
        // Eigen's CompleteOrthogonalDecomposition::pseudoInverse is the
        // upstream operation.  Nalgebra's SVD pseudo-inverse uses the same
        // Moore-Penrose contract; the absolute epsilon mirrors Eigen's
        // machine-precision cutoff for this double-precision path.
        let Some(h_mm_inv) = h_mm.clone_owned().pseudo_inverse(f64::EPSILON).ok() else {
            return Err(MargDataProcessError::SingularMarginalBlock);
        };
        let reduced_h =
            perm_h.view((0, 0), (keep_size, keep_size)).clone_owned() - h_km * &h_mm_inv * h_mk;
        let reduced_b = perm_b.rows(0, keep_size).clone_owned() - h_km * h_mm_inv * b_m;
        h = reduced_h;
        b = reduced_b;
    }

    data.aom_abs_h = Some(matrix_to_data(&h));
    data.aom_abs_b = Some(b.iter().copied().collect());
    refresh_square_root(data, &h, &b);
    data.aom_order = output_blocks;
    data.frame_poses = frame_poses;
    data.frame_states = frame_states;
    data.keyframes = kfs_all_ids
        .iter()
        .copied()
        .filter(|id| data.aom_order.iter().any(|block| block.frame_id == *id))
        .collect();
    data.kfs_all = if data.kfs_all.is_empty() {
        data.keyframes.clone()
    } else {
        data.kfs_all.clone()
    };
    let output_size = h.ncols();
    let output_rank = matrix_rank(&h);
    Ok(MargDataProcessReport {
        input_size,
        output_size,
        input_rank,
        output_rank,
        kept_columns: keep,
        marginalized_columns: marg,
    })
}

/// Camel-case spelling retained as a direct port aid for source-level parity
/// with Basalt's `NfrMapper::processMargData`.
#[allow(non_snake_case)]
pub fn processMargData(data: &mut MargData) -> Result<MargDataProcessReport, MargDataProcessError> {
    process_marg_data(data)
}

#[derive(Clone, Copy)]
struct PoseValue {
    translation: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
}

fn pose_value(pose: [f64; 7]) -> PoseValue {
    PoseValue {
        translation: Vector3::new(pose[0], pose[1], pose[2]),
        rotation: UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            pose[3], pose[4], pose[5], pose[6],
        )),
    }
}

fn pose_array(pose: PoseValue) -> [f64; 7] {
    let q = pose.rotation.quaternion();
    [
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        q.w,
        q.i,
        q.j,
        q.k,
    ]
}

fn pose_inverse(pose: PoseValue) -> PoseValue {
    let rotation = pose.rotation.inverse();
    PoseValue {
        translation: rotation.transform_vector(&(-pose.translation)),
        rotation,
    }
}

fn pose_compose(a: PoseValue, b: PoseValue) -> PoseValue {
    PoseValue {
        translation: a.translation + a.rotation.transform_vector(&b.translation),
        rotation: a.rotation * b.rotation,
    }
}

fn skew(v: &Vector3<f64>) -> nalgebra::Matrix3<f64> {
    nalgebra::Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

fn right_jacobian_inverse_so3(phi: Vector3<f64>) -> nalgebra::Matrix3<f64> {
    let theta2 = phi.norm_squared();
    let omega_hat = skew(&phi);
    if theta2 < f64::EPSILON * f64::EPSILON {
        return nalgebra::Matrix3::identity()
            + 0.5 * omega_hat
            + (1.0 / 12.0) * omega_hat * omega_hat;
    }
    let theta = theta2.sqrt();
    let coefficient = 1.0 / theta2 - (1.0 + theta.cos()) / (2.0 * theta * theta.sin());
    nalgebra::Matrix3::identity() + 0.5 * omega_hat + coefficient * omega_hat * omega_hat
}

fn se3_decoupled_log(pose: PoseValue) -> DVector<f64> {
    let omega = pose.rotation.scaled_axis();
    DVector::from_column_slice(&[
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        omega.x,
        omega.y,
        omega.z,
    ])
}

fn se3_right_jacobian_inverse_decoupled(residual: &DVector<f64>) -> DMatrix<f64> {
    let mut out = DMatrix::identity(6, 6);
    let phi = Vector3::new(residual[3], residual[4], residual[5]);
    let j = right_jacobian_inverse_so3(phi);
    // Basalt's pinned Sophus::rightJacobianInvSE3Decoupled is the
    // decoupled SE(3) approximation: unlike the SO(3)-only block, its
    // translational block is exp(omega), not identity.  Keeping this block
    // is material for mapper relative-factor H and is required before the
    // pose adjoint is applied in relPoseError.
    out.fixed_view_mut::<3, 3>(0, 0).copy_from(
        &UnitQuaternion::from_scaled_axis(phi)
            .to_rotation_matrix()
            .into_inner(),
    );
    for row in 0..3 {
        for col in 0..3 {
            out[(row + 3, col + 3)] = j[(row, col)];
        }
    }
    out
}

fn pose_to_matrix(pose: PoseValue) -> nalgebra::Matrix3<f64> {
    pose.rotation.to_rotation_matrix().into_inner()
}

fn matrix3_array(matrix: Matrix3<f64>) -> [f64; 9] {
    [
        matrix[(0, 0)],
        matrix[(0, 1)],
        matrix[(0, 2)],
        matrix[(1, 0)],
        matrix[(1, 1)],
        matrix[(1, 2)],
        matrix[(2, 0)],
        matrix[(2, 1)],
        matrix[(2, 2)],
    ]
}

fn matrix3_nested_array(matrix: Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
        [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
        [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
    ]
}

/// Basalt's absolute-position residual and 3x6 local Jacobian.
pub fn abs_position_error(
    pose: [f64; 7],
    measured_position: [f64; 3],
) -> ([f64; 3], [[f64; 6]; 3]) {
    let p = pose_value(pose);
    let residual = p.translation - Vector3::from_column_slice(&measured_position);
    let mut jacobian = [[0.0; 6]; 3];
    for (row, values) in jacobian.iter_mut().enumerate() {
        values[row] = 1.0;
    }
    (residual.into(), jacobian)
}

/// Basalt's yaw residual and 1x6 local Jacobian.
pub fn yaw_error(pose: [f64; 7], yaw_direction_body: [f64; 3]) -> (f64, [f64; 6]) {
    let p = pose_value(pose);
    let yaw_dir = Vector3::from_column_slice(&yaw_direction_body);
    let tmp = p.rotation.transform_vector(&yaw_dir);
    let mut jacobian = [0.0; 6];
    jacobian[3] = -tmp.z;
    jacobian[5] = tmp.x;
    (tmp.y, jacobian)
}

/// Basalt's gravity roll/pitch residual and 2x6 local Jacobian.
pub fn roll_pitch_error(
    pose: [f64; 7],
    measured_rotation: [[f64; 3]; 3],
) -> ([f64; 2], [[f64; 6]; 2]) {
    let p = pose_value(pose);
    let r_meas = nalgebra::Matrix3::from_row_slice(&[
        measured_rotation[0][0],
        measured_rotation[0][1],
        measured_rotation[0][2],
        measured_rotation[1][0],
        measured_rotation[1][1],
        measured_rotation[1][2],
        measured_rotation[2][0],
        measured_rotation[2][1],
        measured_rotation[2][2],
    ]);
    let r = r_meas * pose_to_matrix(p).transpose();
    let gravity = r * (-Vector3::z());
    let mut jacobian = [[0.0; 6]; 2];
    jacobian[0][3] = -r[(0, 1)];
    jacobian[1][3] = -r[(1, 1)];
    jacobian[0][4] = r[(0, 0)];
    jacobian[1][4] = r[(1, 0)];
    ([gravity.x, gravity.y], jacobian)
}

/// Basalt's relative-pose residual and its two 6x6 local Jacobians.
pub fn rel_pose_error(
    measured: [f64; 7],
    pose_i: [f64; 7],
    pose_j: [f64; 7],
) -> ([f64; 6], [[f64; 6]; 6], [[f64; 6]; 6]) {
    let t_i_j = pose_value(measured);
    let t_w_i = pose_value(pose_i);
    let t_w_j = pose_value(pose_j);
    let t_j_i = pose_compose(pose_inverse(t_w_j), t_w_i);
    let residual = se3_decoupled_log(pose_compose(t_i_j, t_j_i));
    let j_log = se3_right_jacobian_inverse_decoupled(&residual);
    let r = pose_to_matrix(t_w_i).transpose();
    let mut adj_i = DMatrix::zeros(6, 6);
    let mut adj_j = DMatrix::zeros(6, 6);
    for row in 0..3 {
        for col in 0..3 {
            adj_i[(row, col)] = r[(row, col)];
            adj_i[(row + 3, col + 3)] = r[(row, col)];
            adj_j[(row, col)] = r[(row, col)];
            adj_j[(row + 3, col + 3)] = r[(row, col)];
        }
    }
    let t_i_j_current = pose_inverse(t_w_i);
    let t_i_j_current = pose_compose(t_i_j_current, t_w_j);
    let hat_t = skew(&t_i_j_current.translation) * r;
    for row in 0..3 {
        for col in 0..3 {
            adj_j[(row, col + 3)] = hat_t[(row, col)];
        }
    }
    let d_i = j_log.clone() * adj_i;
    let d_j = -j_log * adj_j;
    let mut di = [[0.0; 6]; 6];
    let mut dj = [[0.0; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            di[row][col] = d_i[(row, col)];
            dj[row][col] = d_j[(row, col)];
        }
    }
    (
        [
            residual[0],
            residual[1],
            residual[2],
            residual[3],
            residual[4],
            residual[5],
        ],
        di,
        dj,
    )
}

#[allow(non_snake_case)]
pub fn absPositionError(pose: [f64; 7], measured_position: [f64; 3]) -> ([f64; 3], [[f64; 6]; 3]) {
    abs_position_error(pose, measured_position)
}

#[allow(non_snake_case)]
pub fn yawError(pose: [f64; 7], yaw_direction_body: [f64; 3]) -> (f64, [f64; 6]) {
    yaw_error(pose, yaw_direction_body)
}

#[allow(non_snake_case)]
pub fn rollPitchError(
    pose: [f64; 7],
    measured_rotation: [[f64; 3]; 3],
) -> ([f64; 2], [[f64; 6]; 2]) {
    roll_pitch_error(pose, measured_rotation)
}

#[allow(non_snake_case)]
pub fn relPoseError(
    measured: [f64; 7],
    pose_i: [f64; 7],
    pose_j: [f64; 7],
) -> ([f64; 6], [[f64; 6]; 6], [[f64; 6]; 6]) {
    rel_pose_error(measured, pose_i, pose_j)
}

/// The pose a recovered factor measures for `frame_id`: the packet's
/// `measurement_poses` entry when it has one, else its own window estimate.
fn pose_for_id(data: &MargData, frame_id: u64) -> Option<[f64; 7]> {
    data.measurement_poses
        .iter()
        .chain(&data.frame_poses)
        .find(|pose| pose.frame_id == frame_id)
        .map(|pose| pose.pose)
        .or_else(|| {
            data.frame_states
                .iter()
                .find(|state| state.frame_id == frame_id)
                .map(|state| state.pose)
        })
}

fn row_major_values(matrix: &DMatrix<f64>) -> Vec<f64> {
    (0..matrix.nrows())
        .flat_map(|row| (0..matrix.ncols()).map(move |col| matrix[(row, col)]))
        .collect()
}

/// Recover Basalt's M8b nonlinear factors from an already processed MargData.
pub fn extract_nonlinear_factors(
    data: &MargData,
    config: MapperConfig,
) -> Result<MapperFactors, NfrExtractionError> {
    if !config.enabled {
        return Ok(MapperFactors {
            imu_relative_pose: Vec::new(),
            provenance_version: data.provenance_version.clone(),
            relative_pose: Vec::new(),
            roll_pitch: Vec::new(),
            ba_covisibility: Vec::new(),
        });
    }
    let (h, _b) = abs_system(data).map_err(|_| NfrExtractionError::InvalidMatrix)?;
    let asize = h.ncols();
    if h.nrows() != asize || asize == 0 {
        return Err(NfrExtractionError::InvalidMatrix);
    }
    // Upstream uses FullPivHouseholderQR and rejects anything short of full
    // column rank before attempting the covariance solve.
    if matrix_rank(&h) != asize {
        return Err(NfrExtractionError::RankDeficient);
    }
    let Some(cov_old) = h.qr().solve(&DMatrix::identity(asize, asize)) else {
        return Err(NfrExtractionError::RankDeficient);
    };
    if cov_old.iter().any(|x| !x.is_finite()) {
        return Err(NfrExtractionError::NonFinite);
    }
    let kf_id = data
        .kfs_to_marg
        .first()
        .copied()
        .or_else(|| data.kf_to_marg.first().map(|pair| pair.0))
        .ok_or(NfrExtractionError::MissingKeyframe)?;
    let kf_pose = pose_for_id(data, kf_id).ok_or(NfrExtractionError::MissingPose)?;
    let kf_start = data
        .aom_order
        .iter()
        .find(|block| block.frame_id == kf_id)
        .map(|block| block.offset)
        .ok_or(NfrExtractionError::MissingAomBlock)?;
    if kf_start + POSE_DOF > asize {
        return Err(NfrExtractionError::InvalidMatrix);
    }

    let kf = pose_value(kf_pose);
    let pos = kf.translation;
    let yaw_direction_body = kf.rotation.inverse().transform_vector(&Vector3::x());
    let (_position_residual, d_position) = abs_position_error(kf_pose, pos.into());
    let (_yaw_residual, d_yaw) = yaw_error(kf_pose, yaw_direction_body.into());
    let measured_rotation = pose_to_matrix(kf);
    let measured_rotation_arr = [
        [
            measured_rotation[(0, 0)],
            measured_rotation[(0, 1)],
            measured_rotation[(0, 2)],
        ],
        [
            measured_rotation[(1, 0)],
            measured_rotation[(1, 1)],
            measured_rotation[(1, 2)],
        ],
        [
            measured_rotation[(2, 0)],
            measured_rotation[(2, 1)],
            measured_rotation[(2, 2)],
        ],
    ];
    let (_rp_residual, d_rp) = roll_pitch_error(kf_pose, measured_rotation_arr);
    let mut j_rp = DMatrix::zeros(6, asize);
    for col in 0..POSE_DOF {
        j_rp[(0, kf_start + col)] = d_position[0][col];
        j_rp[(1, kf_start + col)] = d_position[1][col];
        j_rp[(2, kf_start + col)] = d_position[2][col];
        j_rp[(3, kf_start + col)] = d_yaw[col];
        j_rp[(4, kf_start + col)] = d_rp[0][col];
        j_rp[(5, kf_start + col)] = d_rp[1][col];
    }
    let cov_rp = &j_rp * &cov_old * j_rp.transpose();
    let rp_info = nearest_psd_if_indefinite(
        cov_rp
            .view((4, 4), (2, 2))
            .clone_owned()
            .try_inverse()
            .ok_or(NfrExtractionError::SingularCovariance)?,
    );
    let (roll, pitch, _) = kf.rotation.euler_angles();
    let roll_pitch = if data.used_imu {
        vec![RollPitchFactor {
            frame_id: kf_id,
            roll,
            pitch,
            information: [
                rp_info[(0, 0)],
                rp_info[(0, 1)],
                rp_info[(1, 0)],
                rp_info[(1, 1)],
            ],
            weight: config.roll_pitch_weight,
            measured_rotation: matrix3_array(measured_rotation),
        }]
    } else {
        Vec::new()
    };

    let mut kfs = if !data.kfs_all.is_empty() {
        data.kfs_all.clone()
    } else {
        data.keyframes.clone()
    };
    if config.relative_pose_max_neighbors > 0 {
        // Frame ids advance with time, so this orders by time distance.
        kfs.sort_by_key(|&other_id| (other_id.abs_diff(kf_id), other_id));
    }
    let mut relative_pose = Vec::new();
    for other_id in kfs {
        if config.relative_pose_max_neighbors > 0
            && relative_pose.len() == config.relative_pose_max_neighbors
        {
            break;
        }
        if let Some(factor) = relative_pose_factor(
            data, kf_id, kf_pose, kf_start, other_id, &cov_old, asize, &config,
        )? {
            relative_pose.push(factor);
        }
    }
    Ok(MapperFactors {
        imu_relative_pose: Vec::new(),
        provenance_version: data.provenance_version.clone(),
        relative_pose,
        roll_pitch,
        ba_covisibility: Vec::new(),
    })
}

/// Return `information` unchanged when it is positive semidefinite, otherwise
/// its nearest PSD matrix in the Frobenius norm: symmetrize, then clamp the
/// negative eigenvalues to zero.
///
/// The factor information is `(J * cov * Jᵀ)⁻¹`, with `cov` from a QR inverse
/// of the marginalization Hessian. When that Hessian is ill-conditioned, the
/// result can be slightly indefinite. An indefinite information matrix makes
/// `rᵀ W r` unbounded below, so global BA can drive the total cost negative
/// and diverge. This was observed on V2_03, where the final cost was
/// −8.6e5, with 100 rejected trials and an ATE of 0.375 m against a usual
/// 0.076 m. Healthy factors are returned bit for bit.
static PSD_PROJECTIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Number of factor information matrices projected to PSD in this process.
pub fn psd_information_projection_count() -> usize {
    PSD_PROJECTIONS.load(std::sync::atomic::Ordering::Relaxed)
}

fn nearest_psd_if_indefinite(information: DMatrix<f64>) -> DMatrix<f64> {
    // `symmetric_eigen` iterates without limit, so never feed it non-finite
    // input; the callers reject such factors separately.
    if !information.iter().all(|value| value.is_finite()) {
        return information;
    }
    let symmetric = (&information + information.transpose()) * 0.5;
    let eigen = symmetric.clone().symmetric_eigen();
    if eigen.eigenvalues.iter().all(|&value| value >= 0.0) {
        return information;
    }
    PSD_PROJECTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let clamped = eigen.eigenvalues.map(|value| value.max(0.0));
    &eigen.eigenvectors * DMatrix::from_diagonal(&clamped) * eigen.eigenvectors.transpose()
}

/// Build the relative-pose factor between the marginalization keyframe
/// `kf_id` and `other_id` from the reduced covariance `cov_old`.
///
/// Returns `Ok(None)` when `other_id` cannot contribute a factor: it is the
/// keyframe itself, has no pose, or has no block in the packet's `aom_order`.
/// The last case is the one a larger VIO window exposes -- the AOM problem is
/// a strict subset of the window (`kfs_all`), so a recent keyframe can host no
/// active landmark and have no reduced-system columns.  Skipping it keeps the
/// rest of the recovery alive instead of aborting with `MissingAomBlock`.
#[allow(clippy::too_many_arguments)]
fn relative_pose_factor(
    data: &MargData,
    kf_id: u64,
    kf_pose: [f64; 7],
    kf_start: usize,
    other_id: u64,
    cov_old: &DMatrix<f64>,
    asize: usize,
    config: &MapperConfig,
) -> Result<Option<RelativePoseFactor>, NfrExtractionError> {
    if other_id == kf_id {
        return Ok(None);
    }
    let Some(other_pose) = pose_for_id(data, other_id) else {
        return Ok(None);
    };
    let Some(other_start) = data
        .aom_order
        .iter()
        .find(|block| block.frame_id == other_id)
        .map(|block| block.offset)
    else {
        return Ok(None);
    };
    if other_start + POSE_DOF > asize {
        return Err(NfrExtractionError::InvalidMatrix);
    }
    let kf = pose_value(kf_pose);
    let pose_o = pose_value(other_pose);
    let measurement = pose_compose(pose_inverse(kf), pose_o);
    let (_residual, d_i, d_j) = rel_pose_error(pose_array(measurement), kf_pose, other_pose);
    let mut j = DMatrix::zeros(POSE_DOF, asize);
    for row in 0..POSE_DOF {
        for col in 0..POSE_DOF {
            j[(row, kf_start + col)] = d_i[row][col];
            j[(row, other_start + col)] = d_j[row][col];
        }
    }
    let covariance = &j * cov_old * j.transpose();
    let information = covariance
        .try_inverse()
        .ok_or(NfrExtractionError::SingularCovariance)?;
    if information.iter().any(|x| !x.is_finite()) {
        return Err(NfrExtractionError::NonFinite);
    }
    let information = nearest_psd_if_indefinite(information);
    let q = measurement.rotation.quaternion();
    Ok(Some(RelativePoseFactor {
        from: kf_id,
        to: other_id,
        translation: measurement.translation.into(),
        rotation: [q.w, q.i, q.j, q.k],
        information: row_major_values(&information),
        weight: config.relative_pose_weight,
    }))
}

/// Process a clone of input MargData and recover factors in one operation.
pub fn try_recover_factors(
    data: &MargData,
    config: MapperConfig,
) -> Result<MapperFactors, NfrExtractionError> {
    if !config.enabled {
        return Ok(MapperFactors {
            imu_relative_pose: Vec::new(),
            provenance_version: data.provenance_version.clone(),
            relative_pose: Vec::new(),
            roll_pitch: Vec::new(),
            ba_covisibility: Vec::new(),
        });
    }
    let mut processed = data.clone();
    process_marg_data(&mut processed).map_err(|_| NfrExtractionError::InvalidMatrix)?;
    extract_nonlinear_factors(&processed, config)
}

#[allow(non_snake_case)]
pub fn extractNonlinearFactors(
    data: &MargData,
    config: MapperConfig,
) -> Result<MapperFactors, NfrExtractionError> {
    extract_nonlinear_factors(data, config)
}

pub fn recover_factors(data: &MargData, config: MapperConfig) -> MapperFactors {
    if !config.enabled {
        return MapperFactors {
            imu_relative_pose: Vec::new(),
            provenance_version: data.provenance_version.clone(),
            relative_pose: Vec::new(),
            roll_pitch: Vec::new(),
            ba_covisibility: Vec::new(),
        };
    }
    if let Ok(factors) = try_recover_factors(data, config) {
        return factors;
    }
    // Keep the original lightweight API useful for legacy synthetic callers
    // that carry no absolute system at all.  A real (non-empty) system never
    // falls through here: rank-deficient MargData is rejected as required by
    // M8b rather than silently receiving fabricated unit factors.
    if data
        .aom_abs_h
        .as_ref()
        .is_some_and(|matrix| matrix.rows > 0 && matrix.cols > 0)
    {
        return MapperFactors {
            imu_relative_pose: Vec::new(),
            provenance_version: data.provenance_version.clone(),
            relative_pose: Vec::new(),
            roll_pitch: Vec::new(),
            ba_covisibility: Vec::new(),
        };
    }
    legacy_recover_factors(data, config)
}

fn legacy_recover_factors(data: &MargData, config: MapperConfig) -> MapperFactors {
    if !config.enabled {
        return MapperFactors {
            imu_relative_pose: Vec::new(),
            provenance_version: data.provenance_version.clone(),
            relative_pose: Vec::new(),
            roll_pitch: Vec::new(),
            ba_covisibility: Vec::new(),
        };
    }
    let mut frames = data.frame_states.clone();
    frames.sort_by_key(|f| f.frame_id);
    let mut rp = Vec::new();
    for pair in frames.windows(2) {
        let a = pair[0].pose;
        let b = pair[1].pose;
        let qa = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(a[3], a[4], a[5], a[6]));
        let qb = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(b[3], b[4], b[5], b[6]));
        let q = qa.inverse() * qb;
        let t = Vector3::new(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        rp.push(RelativePoseFactor {
            from: pair[0].frame_id,
            to: pair[1].frame_id,
            translation: t.into(),
            rotation: [q.w, q.i, q.j, q.k],
            information: vec![1.; 36],
            weight: config.relative_pose_weight,
        });
    }
    let mut roll_pitch = Vec::new();
    for f in &frames {
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            f.pose[3], f.pose[4], f.pose[5], f.pose[6],
        ));
        let (r, p, _) = q.euler_angles();
        roll_pitch.push(RollPitchFactor {
            frame_id: f.frame_id,
            roll: r,
            pitch: p,
            information: [1.; 4],
            weight: config.roll_pitch_weight,
            measured_rotation: matrix3_array(q.to_rotation_matrix().into_inner()),
        });
    }
    let mut cov = Vec::new();
    for (i, a) in frames.iter().enumerate() {
        for b in frames.iter().skip(i + 1) {
            let shared = shared_tracks(&data.of_observations, a.frame_id, b.frame_id);
            if shared > 0 {
                cov.push(BaCovisibilityFactor {
                    frame_a: a.frame_id,
                    frame_b: b.frame_id,
                    shared_tracks: shared,
                    information: shared as f64,
                    weight: config.ba_weight,
                });
            }
        }
    }
    MapperFactors {
        imu_relative_pose: Vec::new(),
        provenance_version: data.provenance_version.clone(),
        relative_pose: rp,
        roll_pitch,
        ba_covisibility: cov,
    }
}
fn shared_tracks(obs: &[OfObservationData], a: u64, b: u64) -> u32 {
    let mut x = std::collections::BTreeSet::new();
    let mut y = std::collections::BTreeSet::new();
    for o in obs {
        if o.frame_id == a {
            x.insert(o.track_id);
        }
        if o.frame_id == b {
            y.insert(o.track_id);
        }
    }
    x.intersection(&y).count() as u32
}

#[cfg(test)]
mod psd_information_tests {
    use super::*;

    #[test]
    fn psd_information_is_returned_bit_for_bit() {
        let a = DMatrix::from_fn(6, 6, |r, c| ((r * 7 + c * 3) % 11) as f64 * 0.1 - 0.4);
        let information = &a * a.transpose() + DMatrix::identity(6, 6) * 0.01;
        let out = nearest_psd_if_indefinite(information.clone());
        for (x, y) in information.iter().zip(out.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn indefinite_information_is_projected_to_psd() {
        let information = DMatrix::from_row_slice(2, 2, &[1.0, 3.0, 3.0, 1.0]);
        let out = nearest_psd_if_indefinite(information);
        let eigen = out.clone().symmetric_eigen();
        assert!(eigen.eigenvalues.iter().all(|&v| v >= -1e-12), "{eigen:?}");
        // The positive eigenpair (4, [1,1]/sqrt(2)) survives: 2 in every entry.
        for value in out.iter() {
            assert!((value - 2.0).abs() < 1e-12, "{out}");
        }
        // Residuals can no longer produce a negative cost.
        let r = nalgebra::DVector::from_row_slice(&[1.0, -1.0]);
        assert!((r.transpose() * &out * &r)[(0, 0)] >= -1e-12);
    }
}

#[cfg(test)]
mod pose_block_hessian_tests {
    use super::*;

    /// A gauge-free, SPD-dominant but asymmetric block system resembling a
    /// Schur-reduced pose graph: chain neighbours plus sparse long links.
    fn fixture(seed: u64, poses: usize) -> (PoseBlockHessian, DVector<f64>) {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let mut h = PoseBlockHessian::new(poses);
        for pose in 0..poses {
            let mut links = vec![pose];
            if pose + 1 < poses {
                links.push(pose + 1);
            }
            if pose % 5 == 0 && pose + 7 < poses {
                links.push(pose + 7);
            }
            for &other in &links {
                let a = Matrix6::from_fn(|_, _| next());
                let block = if other == pose {
                    a * a.transpose() + Matrix6::identity() * 40.0
                } else {
                    a
                };
                h.add_block(pose, other, &block);
                if other != pose {
                    // Roundoff-level antisymmetric residue, as in accumulation.
                    let residue = Matrix6::from_fn(|_, _| next() * 1e-12);
                    h.add_block(other, pose, &(block.transpose() + residue));
                }
            }
        }
        let b = DVector::from_fn(poses * 6, |_, _| next());
        (h, b)
    }

    #[test]
    fn block_product_matches_dense_gemv_bits() {
        for seed in 1..=8 {
            let (h, x) = fixture(seed, 23);
            let dense = &h.to_dense() * &x;
            let sparse = h.mul_vector(&x);
            for (a, b) in dense.iter().zip(sparse.iter()) {
                assert!(a == b, "seed={seed}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn block_damped_solve_matches_dense_solve_bits() {
        for seed in 1..=8 {
            for &lambda in &[1e-6, 1e-2, 10.0] {
                let (h, b) = fixture(seed, 31);
                let diagonal = h.diagonal();
                let dense_diagonal = h.to_dense().diagonal();
                assert_eq!(diagonal, dense_diagonal.iter().copied().collect::<Vec<_>>());
                let expected =
                    solve_damped_system(&h.to_dense(), &b, &diagonal, lambda, 1e-12, Some(0));
                let actual = solve_damped_block_system(&h, &b, &diagonal, lambda, 1e-12, Some(0));
                for (e, a) in expected.iter().zip(actual.iter()) {
                    assert!(e == a, "seed={seed} lambda={lambda}: {e} vs {a}");
                }
            }
        }
    }

    #[test]
    fn nonfinite_block_system_uses_dense_fallback_contract() {
        let (mut h, b) = fixture(3, 5);
        h.add_block(2, 2, &(Matrix6::identity() * f64::NAN));
        let diagonal = h.diagonal();
        let expected = solve_damped_system(&h.to_dense(), &b, &diagonal, 1e-3, 1e-12, Some(0));
        let actual = solve_damped_block_system(&h, &b, &diagonal, 1e-3, 1e-12, Some(0));
        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!(e.to_bits() == a.to_bits() || (e.is_nan() && a.is_nan()));
        }
    }
}

#[cfg(test)]
mod offline_tests {
    use super::*;
    fn kp(id: u64, bit: u8) -> Keypoint {
        let mut d = [0u8; 32];
        d[0] = bit;
        Keypoint {
            id,
            x: id as f64,
            y: 0.,
            descriptor: d,
        }
    }
    #[test]
    fn detector_and_hamming_ratio_gates_are_deterministic() {
        let c = OfflineMapperConfig::default();
        let a = detect_keypoints((0..801).map(|i| kp(i, 0)).collect(), c);
        assert_eq!(a.len(), 800);
        let b = vec![kp(0, 0), kp(1, 1), kp(2, 255)];
        let m = mutual_hamming_matches(&[kp(0, 0)], &b, c);
        assert_eq!(m.len(), 1);
        assert!(m[0].distance <= 70);
    }
    #[test]
    fn bow_and_ransac_gates_are_config_driven() {
        let c = OfflineMapperConfig::default();
        let a = vec![kp(1, 3)];
        let b = vec![kp(2, 3)];
        assert_eq!(bow_candidates(&a, &b, c), vec![(1, 2)]);
        let good = DescriptorMatch {
            left: 1,
            right: 1,
            distance: 1,
        };
        let bad = DescriptorMatch {
            left: 1,
            right: 100,
            distance: 1,
        };
        assert_eq!(essential_ransac_gate(&[good, bad], c).len(), 1);
    }
    #[test]
    fn union_find_min_track_and_triangulation_distance() {
        let c = OfflineMapperConfig {
            min_track_length: 3,
            ..Default::default()
        };
        let p = (0..3)
            .map(|f| {
                (
                    f,
                    vec![DescriptorMatch {
                        left: 7,
                        right: 8,
                        distance: 0,
                    }],
                )
            })
            .collect::<Vec<_>>();
        let t = union_find_tracks(&p, c);
        assert_eq!(t.len(), 1);
        assert!(triangulation_gate(&t[0], 0.07, c));
        assert!(!triangulation_gate(&t[0], 0.01, c));
    }
}
#[cfg(test)]
mod ba_tests {
    use super::*;
    use crate::vio::landmarks::StereographicDirection;
    use crate::vio::margdata::*;
    fn d() -> MargData {
        MargData {
            // This hand-built mapper test packet has no retained upstream FEJ
            // sidecars; keep it explicitly on the legacy schema rather than
            // inventing schema-4 values.
            schema_version: MARGDATA_SCHEMA_VERSION_V3,
            aom_sqrt_jacobian: MatrixData::new(0, 0, Vec::new()).unwrap(),
            aom_sqrt_rhs: Vec::new(),
            aom_abs_h: None,
            aom_abs_b: None,
            frame_poses: Vec::new(),
            frame_states: vec![
                FrameStateData {
                    frame_id: 1,
                    timestamp_ns: 1,
                    pose: [0., 0., 0., 1., 0., 0., 0.],
                    velocity: [0.; 3],
                    gyro_bias: [0.; 3],
                    accel_bias: [0.; 3],
                    linearized: false,
                    is_keyframe: true,
                    is_latest: false,
                },
                FrameStateData {
                    frame_id: 2,
                    timestamp_ns: 2,
                    pose: [0.5, 0., 0., 1., 0., 0., 0.],
                    velocity: [0.; 3],
                    gyro_bias: [0.; 3],
                    accel_bias: [0.; 3],
                    linearized: false,
                    is_keyframe: false,
                    is_latest: true,
                },
            ],
            keyframes: vec![1],
            kf_to_marg: Vec::new(),
            kfs_all: Vec::new(),
            kfs_to_marg: Vec::new(),
            aom_order: Vec::new(),
            marginalization: MarginalizationTargets::default(),
            prior: None,
            row_counts: [0; 4],
            of_observations: Vec::new(),
            of_images: Vec::new(),
            frame_poses_fej: Default::default(),
            frame_states_fej: Default::default(),
            fej_complete: false,
            measurement_poses: Vec::new(),
            used_imu: false,
            provenance_version: "basalt-x".into(),
        }
    }
    #[test]
    fn ba_cost_decreases_and_reports_upstream_gauge() {
        let d = d();
        let f = recover_factors(&d, MapperConfig::default());
        let (s, z) = global_ba(&d, &f, GlobalBaConfig::default());
        assert!(z.final_cost <= z.initial_cost);
        assert_eq!(z.gauge_anchor, 1);
        assert!(s.se3_poses.values().all(|pose| {
            pose.translation.iter().all(|value| value.is_finite())
                && pose.rotation.coords.iter().all(|value| value.is_finite())
        }));
    }
    #[test]
    fn disabled_mapper_is_noop() {
        let d = d();
        let f = recover_factors(&d, MapperConfig::default());
        let (_, s) = global_ba(
            &d,
            &f,
            GlobalBaConfig {
                enabled: false,
                ..Default::default()
            },
        );
        assert!(s.disabled);
        assert_eq!(s.initial_cost, s.final_cost);
    }

    #[test]
    fn global_ba_gauge_metadata_does_not_pin_solver_block() {
        let h = DMatrix::<f64>::identity(6, 6);
        let b = DVector::<f64>::from_element(6, 1.0);
        let solve = solve_damped_system(&h, &b, &[1.0; 6], 1.0e-32, 1.0e-32, Some(0));
        assert!(solve.iter().all(|value| (*value - 1.0).abs() < 1.0e-8));
    }

    fn synthetic_calibration() -> BasaltCalibration {
        let camera =
            crate::camera::DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap();
        BasaltCalibration {
            t_imu_cam: vec![SE3::identity()],
            cameras: vec![camera],
            resolutions: vec![(640, 480)],
            calib_accel_bias: vec![0.0; 3],
            calib_gyro_bias: vec![0.0; 3],
            imu_update_rate_hz: 200.0,
            accel_noise_std: Vector3::zeros(),
            gyro_noise_std: Vector3::zeros(),
            accel_bias_std: Vector3::zeros(),
            gyro_bias_std: Vector3::zeros(),
            t_mocap_world: SE3::identity(),
            t_imu_marker: SE3::identity(),
            mocap_time_offset_ns: 0,
            mocap_to_imu_offset_ns: 0,
            cam_time_offset_ns: 0,
        }
    }

    #[test]
    fn full_mapper_ba_schur_updates_pose_and_stereographic_landmark() {
        let mut data = d();
        for (frame_id, x) in [(3, 0.4), (4, 0.6)] {
            data.frame_states.push(FrameStateData {
                frame_id,
                timestamp_ns: frame_id as i64,
                pose: [x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                velocity: [0.0; 3],
                gyro_bias: [0.0; 3],
                accel_bias: [0.0; 3],
                linearized: false,
                is_keyframe: true,
                is_latest: false,
            });
        }
        // The observations come from a separate, metrically valid camera
        // trajectory.  The input estimates deliberately carry a translation
        // error and a perturbed inverse-distance landmark.
        let calibration = synthetic_calibration();
        let truth_poses = [
            (1, SE3::identity()),
            (
                2,
                SE3::new(UnitQuaternion::identity(), Vector3::new(0.2, 0.0, 0.0)),
            ),
            (
                3,
                SE3::new(UnitQuaternion::identity(), Vector3::new(0.4, 0.0, 0.0)),
            ),
            (
                4,
                SE3::new(UnitQuaternion::identity(), Vector3::new(0.6, 0.0, 0.0)),
            ),
        ];
        let point_world = Point3::new(0.15, 0.1, 3.0);
        let host_point = truth_poses[0].1.inverse().transform_point(&point_world);
        let direction = StereographicDirection::from_bearing(host_point.coords).unwrap();
        let mut observations = Vec::new();
        for (frame_id, pose) in truth_poses {
            let point_camera = pose.inverse().transform_point(&point_world);
            let pixel = calibration
                .camera(0)
                .unwrap()
                .project(&point_camera)
                .unwrap();
            observations.push(MapperObservation {
                image: TimeCamId::new(frame_id, 0),
                feature_id: frame_id,
                pixel,
            });
        }
        let mut landmarks = BTreeMap::new();
        landmarks.insert(
            42,
            MapperLandmark {
                track_id: 42,
                host: TimeCamId::new(1, 0),
                second: TimeCamId::new(2, 0),
                direction: StereographicDirection {
                    xy: Point2::new(direction.xy.x + 0.01, direction.xy.y - 0.005),
                },
                inverse_distance: 1.0 / 2.8,
                observations,
            },
        );
        let factors = MapperFactors {
            imu_relative_pose: Vec::new(),
            provenance_version: "synthetic".into(),
            relative_pose: Vec::new(),
            roll_pitch: Vec::new(),
            ba_covisibility: Vec::new(),
        };
        let (state, summary) = global_ba_with_landmarks(
            &data,
            &factors,
            &landmarks,
            &calibration,
            GlobalBaConfig {
                max_iterations: 4,
                lambda_initial: 1.0e-3,
                lambda_min: 1.0e-8,
                ..Default::default()
            },
        );
        assert!(summary.final_cost < summary.initial_cost);
        assert_eq!(state.landmarks.len(), 1);
        assert_eq!(state.se3_poses.len(), 4);
        assert!(state.se3_poses.values().all(|pose| {
            pose.translation.iter().all(|value| value.is_finite())
                && pose.rotation.coords.iter().all(|value| value.is_finite())
        }));
        assert!(state.landmarks[&42].inverse_distance.is_finite());
        assert!(!summary.trace.is_empty());
    }

    #[test]
    fn mapper_double_sphere_projection_jacobian_matches_finite_difference() {
        let calibration = synthetic_calibration();
        let camera = calibration.camera(0).unwrap();
        let point = Point3::new(0.31, -0.22, 1.7);
        let (_, analytic) = double_sphere_projection_jacobian(camera, &point).unwrap();
        let epsilon = 1.0e-7;
        for axis in 0..3 {
            let mut plus = point.coords;
            let mut minus = point.coords;
            plus[axis] += epsilon;
            minus[axis] -= epsilon;
            let plus = camera.project(&Point3::from(plus)).unwrap();
            let minus = camera.project(&Point3::from(minus)).unwrap();
            let numeric = (plus - minus) / (2.0 * epsilon);
            assert!((numeric.x - analytic[(0, axis)]).abs() < 1.0e-6);
            assert!((numeric.y - analytic[(1, axis)]).abs() < 1.0e-6);
        }
        assert_eq!(analytic.column(3).norm(), 0.0);
    }
}

#[cfg(test)]
mod joint_ba_tests {
    use super::*;
    use crate::imu::{ImuNoiseModel, ImuPreintegrator};
    use crate::vio::landmarks::StereographicDirection;

    fn synthetic_calibration() -> BasaltCalibration {
        let camera =
            crate::camera::DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap();
        BasaltCalibration {
            t_imu_cam: vec![SE3::identity()],
            cameras: vec![camera],
            resolutions: vec![(640, 480)],
            calib_accel_bias: vec![0.0; 3],
            calib_gyro_bias: vec![0.0; 3],
            imu_update_rate_hz: 200.0,
            accel_noise_std: Vector3::zeros(),
            gyro_noise_std: Vector3::zeros(),
            accel_bias_std: Vector3::zeros(),
            gyro_bias_std: Vector3::zeros(),
            t_mocap_world: SE3::identity(),
            t_imu_marker: SE3::identity(),
            mocap_time_offset_ns: 0,
            mocap_to_imu_offset_ns: 0,
            cam_time_offset_ns: 0,
        }
    }

    /// Three keyframes, translation-only truth motion along x, one landmark
    /// observed by all three -- the same shape of fixture as
    /// `ba_tests::full_mapper_ba_schur_updates_pose_and_stereographic_landmark`,
    /// reused here so the joint solver is exercised on a real (if small)
    /// vision problem, not just synthetic IMU factors in isolation.
    fn synthetic_scene() -> (
        BTreeMap<u64, SE3>,
        BTreeMap<u64, MapperLandmark>,
        BasaltCalibration,
    ) {
        let calibration = synthetic_calibration();
        let truth_poses = [
            (1u64, SE3::identity()),
            (
                2,
                SE3::new(UnitQuaternion::identity(), Vector3::new(0.2, 0.0, 0.0)),
            ),
            (
                3,
                SE3::new(UnitQuaternion::identity(), Vector3::new(0.4, 0.0, 0.0)),
            ),
        ];
        let point_world = Point3::new(0.15, 0.1, 3.0);
        let host_point = truth_poses[0].1.inverse().transform_point(&point_world);
        let direction = StereographicDirection::from_bearing(host_point.coords).unwrap();
        let mut observations = Vec::new();
        for (frame_id, pose) in &truth_poses {
            let point_camera = pose.inverse().transform_point(&point_world);
            let pixel = calibration
                .camera(0)
                .unwrap()
                .project(&point_camera)
                .unwrap();
            observations.push(MapperObservation {
                image: TimeCamId::new(*frame_id, 0),
                feature_id: *frame_id,
                pixel,
            });
        }
        let mut landmarks = BTreeMap::new();
        landmarks.insert(
            42,
            MapperLandmark {
                track_id: 42,
                host: TimeCamId::new(1, 0),
                second: TimeCamId::new(2, 0),
                direction: StereographicDirection {
                    xy: Point2::new(direction.xy.x + 0.01, direction.xy.y - 0.005),
                },
                inverse_distance: 1.0 / 2.8,
                observations,
            },
        );
        let poses = truth_poses.into_iter().collect::<BTreeMap<_, _>>();
        (poses, landmarks, calibration)
    }

    fn empty_factors() -> MapperFactors {
        MapperFactors {
            imu_relative_pose: Vec::new(),
            provenance_version: "synthetic".into(),
            relative_pose: Vec::new(),
            roll_pitch: Vec::new(),
            ba_covisibility: Vec::new(),
        }
    }

    #[test]
    fn joint_solver_matches_pose_only_solver_when_imu_factors_are_empty() {
        let (mut joint_poses, mut joint_landmarks, calibration) = synthetic_scene();
        let (mut pose_only_poses, mut pose_only_landmarks, _) = synthetic_scene();
        let factors = empty_factors();
        let config = GlobalBaConfig {
            max_iterations: 4,
            lambda_initial: 1.0e-3,
            lambda_min: 1.0e-8,
            ..Default::default()
        };

        let mut pose_only_state = GlobalBaOptimizerState::from_config(config);
        let pose_only_summary = global_ba_with_state(
            &mut pose_only_poses,
            &factors,
            &mut pose_only_landmarks,
            &calibration,
            config,
            &mut pose_only_state,
        );

        let mut velocities = BTreeMap::new();
        let mut gyro_biases = BTreeMap::new();
        let mut accel_biases = BTreeMap::new();
        let mut joint_state = GlobalBaOptimizerState::from_config(config);
        let joint_summary = global_ba_with_state_joint(
            &mut joint_poses,
            &mut velocities,
            &mut gyro_biases,
            &mut accel_biases,
            &factors,
            &[],
            &[],
            &mut joint_landmarks,
            &calibration,
            config,
            Vector3::new(0.0, 0.0, -9.81),
            &mut joint_state,
        );

        assert!((joint_summary.final_cost - pose_only_summary.final_cost).abs() < 1e-9);
        for (id, pose) in &pose_only_poses {
            let joint_pose = &joint_poses[id];
            assert!((joint_pose.translation - pose.translation).norm() < 1e-9);
            assert!(
                (joint_pose.rotation.to_rotation_matrix().into_inner()
                    - pose.rotation.to_rotation_matrix().into_inner())
                .norm()
                    < 1e-9
            );
        }
        assert!(
            (joint_landmarks[&42].inverse_distance - pose_only_landmarks[&42].inverse_distance)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn joint_solver_corrects_velocity_from_chained_imu_factors() {
        let (mut poses, mut landmarks, calibration) = synthetic_scene();
        let factors = empty_factors();

        // Truth motion is 0.2 (world units) per keyframe; model that as 1 s
        // spacing so truth velocity is (0.2, 0, 0). Seed every keyframe with
        // a deliberately wrong velocity guess.
        let noise = ImuNoiseModel {
            gyro_density: 0.02,
            accel_density: 0.05,
        };
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let mut imu_factors = Vec::new();
        for (from, to) in [(1u64, 2u64), (2, 3)] {
            let mut integrator = ImuPreintegrator::new(Vector3::zeros(), Vector3::zeros())
                .with_noise(noise)
                .unwrap();
            // Zero net world-frame acceleration (specific force cancels
            // gravity only): consistent with the constant-velocity truth
            // motion above.
            for _ in 0..10 {
                integrator.integrate_sample(Vector3::zeros(), -gravity, 0.1);
            }
            imu_factors.push(imu_ba::ImuPreintegratedFactor {
                from,
                to,
                delta: integrator.delta().clone(),
                weight: 1.0,
            });
        }
        // Paired bias random-walk factors, as the real online mapper always
        // uses both together: without them, per-keyframe bias is only
        // weakly constrained by the first-order bias-correction terms
        // inside the preintegration factor alone (see the equivalent note
        // and empirical finding in `mapper::imu_ba`'s own
        // `scale_error_is_corrected_by_chained_preintegration_factors`
        // test), which lets bias partially absorb the residual instead of
        // velocity converging.
        let bias_factors = [(1u64, 2u64), (2, 3)]
            .into_iter()
            .map(|(from, to)| imu_ba::BiasRandomWalkFactor {
                from,
                to,
                dt: 1.0,
                noise: crate::imu::BiasRandomWalkNoise {
                    gyro_density: 1e-5,
                    accel_density: 1e-5,
                },
                weight: 1.0,
            })
            .collect::<Vec<_>>();
        let mut velocities = BTreeMap::new();
        for id in [1u64, 2, 3] {
            velocities.insert(id, Vector3::new(0.5, -0.3, 0.2));
        }
        let mut gyro_biases = BTreeMap::new();
        let mut accel_biases = BTreeMap::new();

        let config = GlobalBaConfig {
            max_iterations: 15,
            lambda_initial: 1.0e-3,
            lambda_min: 1.0e-8,
            ..Default::default()
        };
        let mut optimizer_state = GlobalBaOptimizerState::from_config(config);
        let summary = global_ba_with_state_joint(
            &mut poses,
            &mut velocities,
            &mut gyro_biases,
            &mut accel_biases,
            &factors,
            &imu_factors,
            &bias_factors,
            &mut landmarks,
            &calibration,
            config,
            gravity,
            &mut optimizer_state,
        );

        assert!(summary.final_cost < summary.initial_cost);
        let truth_velocity = Vector3::new(0.2, 0.0, 0.0);
        let initial_error = (Vector3::new(0.5, -0.3, 0.2) - truth_velocity).norm();
        // This fixture's monocular single-landmark vision terms do not fully
        // pin scale/gauge the way `mapper::imu_ba`'s dedicated,
        // properly-endpoint-pinned synthetic-trajectory test does (see that
        // module's `scale_error_is_corrected_by_chained_preintegration_factors`,
        // which *does* converge to near-zero error): the goal here is to
        // prove the joint solver's wiring (embedding, LM accept/reject,
        // backup/restore across pose+velocity+bias) actually pulls velocity
        // toward physical consistency on a combined vision+IMU problem, not
        // to reproduce full observability with only 3 keyframes / 2 links.
        for id in [1u64, 2, 3] {
            let error = (velocities[&id] - truth_velocity).norm();
            assert!(
                error < initial_error * 0.5,
                "keyframe {id}: velocity {:?} did not move meaningfully toward truth {:?} (initial error {initial_error:e}, final error {error:e})",
                velocities[&id],
                truth_velocity
            );
        }
        assert!(poses.values().all(|pose| {
            pose.translation.iter().all(|value| value.is_finite())
                && pose.rotation.coords.iter().all(|value| value.is_finite())
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vio::margdata::*;
    fn data() -> MargData {
        MargData {
            // Synthetic factor-recovery fixture; FEJ sidecars are absent.
            schema_version: MARGDATA_SCHEMA_VERSION_V3,
            aom_sqrt_jacobian: MatrixData::new(0, 0, Vec::new()).unwrap(),
            aom_sqrt_rhs: Vec::new(),
            aom_abs_h: None,
            aom_abs_b: None,
            frame_poses: Vec::new(),
            frame_states: vec![
                FrameStateData {
                    frame_id: 1,
                    timestamp_ns: 1,
                    pose: [0., 0., 0., 1., 0., 0., 0.],
                    velocity: [0.; 3],
                    gyro_bias: [0.; 3],
                    accel_bias: [0.; 3],
                    linearized: false,
                    is_keyframe: true,
                    is_latest: false,
                },
                FrameStateData {
                    frame_id: 2,
                    timestamp_ns: 2,
                    pose: [1., 0., 0., 1., 0., 0., 0.],
                    velocity: [0.; 3],
                    gyro_bias: [0.; 3],
                    accel_bias: [0.; 3],
                    linearized: false,
                    is_keyframe: false,
                    is_latest: true,
                },
            ],
            keyframes: vec![1],
            kf_to_marg: Vec::new(),
            kfs_all: Vec::new(),
            kfs_to_marg: Vec::new(),
            aom_order: Vec::new(),
            marginalization: MarginalizationTargets::default(),
            prior: None,
            row_counts: [0; 4],
            of_observations: vec![
                OfObservationData {
                    frame_id: 1,
                    track_id: 7,
                    camera_id: 0,
                    x: 0.,
                    y: 0.,
                },
                OfObservationData {
                    frame_id: 2,
                    track_id: 7,
                    camera_id: 0,
                    x: 1.,
                    y: 0.,
                },
            ],
            of_images: Vec::new(),
            frame_poses_fej: Default::default(),
            frame_states_fej: Default::default(),
            fej_complete: false,
            measurement_poses: Vec::new(),
            used_imu: true,
            provenance_version: "basalt-0f3b2b52".into(),
        }
    }
    #[test]
    fn reconstructs_three_nfr_factor_types() {
        let f = recover_factors(&data(), MapperConfig::default());
        assert_eq!(f.relative_pose.len(), 1);
        assert_eq!(f.roll_pitch.len(), 2);
        assert_eq!(f.ba_covisibility[0].shared_tracks, 1);
        assert_eq!(f.relative_pose[0].translation, [1., 0., 0.]);
    }
    #[test]
    fn mapper_serialization_roundtrip() {
        let f = recover_factors(&data(), MapperConfig::default());
        let x = serde_json::to_string(&f).unwrap();
        assert_eq!(f, serde_json::from_str(&x).unwrap());
    }
    #[test]
    fn mapper_off_is_empty_and_does_not_mutate_margdata() {
        let d = data();
        let before = d.clone();
        let f = recover_factors(
            &d,
            MapperConfig {
                enabled: false,
                ..Default::default()
            },
        );
        assert!(
            f.relative_pose.is_empty() && f.roll_pitch.is_empty() && f.ba_covisibility.is_empty()
        );
        assert_eq!(d, before);
    }
}

#[cfg(test)]
mod m8_fixture_tests {
    use super::*;
    use crate::vio::margdata::{
        FrameStateData, MarginalizationTargets, MARGDATA_SCHEMA_VERSION, MARGDATA_SCHEMA_VERSION_V3,
    };
    use serde_json::Value;

    fn arr(value: &Value) -> Vec<f64> {
        value
            .as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_f64().expect("finite number"))
            .collect()
    }

    fn matrix_values(value: &Value) -> Vec<f64> {
        value
            .as_array()
            .expect("matrix rows")
            .iter()
            .flat_map(|row| arr(row))
            .collect()
    }

    fn fixture() -> (MargData, Value) {
        let root: Value = serde_json::from_str(include_str!(
            "../../../../benchmarks/basalt/m8a_m8b_mh01_1403636579763555584.json"
        ))
        .expect("fixture json");
        let input = &root["input"];
        let h_rows = input["abs_H"].as_array().unwrap();
        let h_cols = h_rows[0].as_array().unwrap().len();
        let mut h_col_major = Vec::with_capacity(h_cols * h_rows.len());
        for c in 0..h_cols {
            for row in h_rows {
                h_col_major.push(row[c].as_f64().unwrap());
            }
        }
        let mut frame_poses = Vec::new();
        for pose in input["frame_poses"].as_array().unwrap() {
            let q = arr(&pose["pose"]["quaternion_xyzw"]);
            let t = arr(&pose["pose"]["translation"]);
            frame_poses.push(FramePoseData {
                frame_id: pose["id"].as_u64().unwrap(),
                timestamp_ns: pose["t_ns"].as_i64().unwrap(),
                pose: [t[0], t[1], t[2], q[3], q[0], q[1], q[2]],
                is_keyframe: true,
            });
        }
        let mut frame_states = Vec::new();
        for state in input["frame_states"].as_array().unwrap() {
            let q = arr(&state["pose"]["quaternion_xyzw"]);
            let t = arr(&state["pose"]["translation"]);
            let v = arr(&state["velocity"]);
            let bg = arr(&state["bias_gyro"]);
            let ba = arr(&state["bias_accel"]);
            frame_states.push(FrameStateData {
                frame_id: state["id"].as_u64().unwrap(),
                timestamp_ns: state["t_ns"].as_i64().unwrap(),
                pose: [t[0], t[1], t[2], q[3], q[0], q[1], q[2]],
                velocity: [v[0], v[1], v[2]],
                gyro_bias: [bg[0], bg[1], bg[2]],
                accel_bias: [ba[0], ba[1], ba[2]],
                linearized: state["linearized"].as_bool().unwrap(),
                is_keyframe: state["id"] == input["kfs_all"][7],
                is_latest: false,
            });
        }
        let aom_order = input["aom"]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|block| AomBlockData {
                frame_id: block["id"].as_u64().unwrap(),
                offset: block["offset"].as_u64().unwrap() as usize,
                dof: block["size"].as_u64().unwrap() as usize,
                kind: if block["size"] == 6 { "pose" } else { "state" }.into(),
            })
            .collect();
        let data = MargData {
            // Fixture is intentionally schema 3 because it predates the FEJ
            // sidecar contract and contains no FEJ values.
            schema_version: MARGDATA_SCHEMA_VERSION_V3,
            aom_sqrt_jacobian: MatrixData::new(0, h_cols, Vec::new()).unwrap(),
            aom_sqrt_rhs: Vec::new(),
            aom_abs_h: Some(MatrixData::new(h_rows.len(), h_cols, h_col_major).unwrap()),
            aom_abs_b: Some(arr(&input["abs_b"])),
            frame_poses,
            frame_states,
            keyframes: input["kfs_all"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect(),
            kf_to_marg: Vec::new(),
            kfs_all: input["kfs_all"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect(),
            kfs_to_marg: input["kfs_to_marg"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect(),
            aom_order,
            marginalization: MarginalizationTargets::default(),
            prior: None,
            row_counts: [0; 4],
            of_observations: Vec::new(),
            of_images: Vec::new(),
            frame_poses_fej: Default::default(),
            frame_states_fej: Default::default(),
            fej_complete: false,
            measurement_poses: Vec::new(),
            used_imu: input["use_imu"].as_bool().unwrap(),
            provenance_version: "basalt-0f3b2b52".into(),
        };
        (data, root)
    }

    #[test]
    fn m8a_reduces_fixture_72_to_48_with_upstream_order_and_gram() {
        let (mut data, root) = fixture();
        let report = process_marg_data(&mut data).expect("M8a process");
        assert_eq!((report.input_size, report.output_size), (72, 48));
        assert_eq!((report.input_rank, report.output_rank), (72, 48));
        assert_eq!(report.kept_columns, (0..48).collect::<Vec<_>>());
        assert_eq!(report.marginalized_columns, (48..72).collect::<Vec<_>>());
        assert_eq!(data.aom_order.len(), 8);
        assert_eq!(data.frame_poses.len(), 8);
        assert_eq!(data.frame_states.len(), 1);
        assert!(data.validate(), "post-M8a MargData must remain valid");
        let expected = &root["output"]["abs_H"];
        let expected_rows = expected.as_array().unwrap();
        let expected_cols = expected_rows[0].as_array().unwrap().len();
        let expected_row_major = expected_rows
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect::<Vec<_>>();
        let actual = matrix_from_data(data.aom_abs_h.as_ref().unwrap()).unwrap();
        let expected =
            DMatrix::from_row_slice(expected_rows.len(), expected_cols, &expected_row_major);
        let max_error = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_error < 1e-3, "M8a Gram max error {max_error}");
        let expected_b = arr(&root["output"]["abs_b"]);
        let max_b_error = data
            .aom_abs_b
            .as_ref()
            .unwrap()
            .iter()
            .zip(expected_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_b_error < 1e-6, "M8a gradient max error {max_b_error}");
    }

    #[test]
    fn m8b_recovers_fixture_factor_counts_measurements_and_information() {
        let (mut data, root) = fixture();
        process_marg_data(&mut data).expect("M8a process");
        let factors = extract_nonlinear_factors(&data, MapperConfig::default()).expect("M8b");
        assert_eq!(factors.roll_pitch.len(), 1);
        assert_eq!(factors.relative_pose.len(), 7);
        let expected_rp = &root["output"]["factors"]["roll_pitch"][0]["information"];
        let expected_rp = matrix_values(expected_rp);
        for (actual, expected) in factors.roll_pitch[0]
            .information
            .iter()
            .zip(expected_rp.iter())
        {
            assert!(
                (actual - expected).abs() < 1e-3,
                "RP info {actual} != {expected}"
            );
        }
        let expected_rel = root["output"]["factors"]["relative_pose"]
            .as_array()
            .unwrap();
        for (actual, expected) in factors.relative_pose.iter().zip(expected_rel.iter()) {
            let translation = arr(&expected["measurement_translation"]);
            for (x, y) in actual.translation.iter().zip(translation.iter()) {
                assert!((x - y).abs() < 1e-7, "relative translation {x} != {y}");
            }
            let quaternion = arr(&expected["measurement_quaternion_xyzw"]);
            let expected_wxyz = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]];
            for (x, y) in actual.rotation.iter().zip(expected_wxyz.iter()) {
                assert!((x - y).abs() < 1e-7, "relative quaternion {x} != {y}");
            }
            let expected_info = matrix_values(&expected["information"]);
            let max_error = actual
                .information
                .iter()
                .zip(expected_info.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            assert!(
                max_error < 1e-2,
                "relative information max error {max_error}"
            );
        }
    }

    #[test]
    fn m8b_neighbor_limit_keeps_the_factors_to_the_nearest_keyframes() {
        let (mut data, _root) = fixture();
        process_marg_data(&mut data).expect("M8a process");
        let all = extract_nonlinear_factors(&data, MapperConfig::default()).expect("M8b");
        let limited = extract_nonlinear_factors(
            &data,
            MapperConfig {
                relative_pose_max_neighbors: 2,
                ..Default::default()
            },
        )
        .expect("M8b limited");
        assert_eq!(limited.roll_pitch, all.roll_pitch);
        assert_eq!(limited.relative_pose.len(), 2);
        let mut nearest: Vec<&RelativePoseFactor> = all.relative_pose.iter().collect();
        nearest.sort_by_key(|f| (f.to.abs_diff(f.from), f.to));
        assert_eq!(
            limited.relative_pose.iter().collect::<Vec<_>>(),
            nearest[..2].to_vec()
        );
    }

    #[test]
    fn m8b_measures_the_measurement_poses_when_a_packet_has_them() {
        let (mut data, _root) = fixture();
        process_marg_data(&mut data).expect("M8a process");
        let base = extract_nonlinear_factors(&data, MapperConfig::default()).expect("M8b");
        // The packet's own poses as measurement poses change nothing.
        data.measurement_poses = data.frame_poses.clone();
        assert_eq!(
            extract_nonlinear_factors(&data, MapperConfig::default()).expect("M8b same"),
            base
        );
        // Moving one non-marginalized keyframe moves only its own measurement.
        let moved = base.relative_pose[0].to;
        let mut pose = pose_for_id(&data, moved).expect("pose");
        pose[0] += 1.0;
        data.measurement_poses = vec![FramePoseData {
            frame_id: moved,
            timestamp_ns: 0,
            pose,
            is_keyframe: true,
        }];
        let shifted = extract_nonlinear_factors(&data, MapperConfig::default()).expect("M8b");
        assert_eq!(shifted.roll_pitch, base.roll_pitch);
        for (after, before) in shifted.relative_pose.iter().zip(&base.relative_pose) {
            let dx: f64 = after
                .translation
                .iter()
                .zip(before.translation)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if after.to == moved {
                assert!((dx - 1.0).abs() < 1e-9, "moved factor shifts by {dx}");
            } else {
                assert_eq!(after, before);
            }
        }
    }

    #[test]
    fn m8b_rejects_rank_deficient_absolute_system() {
        let (mut data, _root) = fixture();
        process_marg_data(&mut data).expect("M8a process");
        let h = data.aom_abs_h.as_mut().unwrap();
        let n = h.rows;
        for row in 0..n {
            for col in 0..n {
                if row == 0 || col == 0 {
                    h.data[row + col * n] = 0.0;
                }
            }
        }
        assert_eq!(
            extract_nonlinear_factors(&data, MapperConfig::default()),
            Err(NfrExtractionError::RankDeficient)
        );
    }

    #[test]
    fn try_recover_factors_processes_input_clone_without_mutating_source() {
        let (data, _root) = fixture();
        let before = data.clone();
        let factors = try_recover_factors(&data, MapperConfig::default()).expect("M8a + M8b");
        assert_eq!(factors.roll_pitch.len(), 1);
        assert_eq!(factors.relative_pose.len(), 7);
        assert_eq!(data, before);
    }

    #[test]
    fn m8b_skips_keyframes_absent_from_aom_order() {
        // A window keyframe can be missing from the packet's AOM order because
        // the AOM problem is a strict subset of the window (observed on
        // LaMAria once `vio_max_states`/`vio_max_kfs` grow past the upstream
        // defaults).  Such a keyframe has a pose but no reduced-system
        // columns, so recovery must skip it rather than abort with
        // `MissingAomBlock`.
        let (mut data, _root) = fixture();
        process_marg_data(&mut data).expect("M8a process");
        let present = data.frame_poses[0].frame_id;
        assert!(
            !data
                .aom_order
                .iter()
                .any(|block| block.frame_id == u64::MAX),
            "fixture precondition: stray id must not be in the AOM order"
        );
        let mut stray = data
            .frame_poses
            .iter()
            .find(|pose| pose.frame_id == present)
            .unwrap()
            .clone();
        stray.frame_id = u64::MAX;
        data.frame_poses.push(stray);
        data.keyframes.push(u64::MAX);
        data.kfs_all.push(u64::MAX);

        let factors = extract_nonlinear_factors(&data, MapperConfig::default())
            .expect("stray keyframe must be skipped, not abort recovery");
        assert_eq!(factors.roll_pitch.len(), 1);
        assert_eq!(
            factors.relative_pose.len(),
            7,
            "the stray keyframe must not add a relative-pose factor"
        );
    }

    /// Opt-in integration check for the queue-facing MH01 max800 stream.
    ///
    /// The directory is intentionally supplied by the caller: this keeps the
    /// normal unit-test suite independent of a large local run and exercises
    /// the loader in place without copying or rewriting any packet.
    #[test]
    fn load_margdata_directory_validates_max800_when_requested() {
        let Some(path) = std::env::var_os("VISLOC_MARGDATA_VALIDATION_DIR") else {
            return;
        };
        let records = load_margdata_directory(std::path::Path::new(&path))
            .expect("max800 MargData directory must load and validate");
        assert_eq!(records.len(), 107, "unexpected max800 packet count");
        assert!(records
            .iter()
            .all(|record| record.schema_version == MARGDATA_SCHEMA_VERSION));
        assert!(records.iter().all(|record| {
            record.is_mapper_packet()
                && !record.kfs_to_marg.is_empty()
                && !record.aom_order.is_empty()
        }));
        let packet_ids = records
            .iter()
            .flat_map(|record| record.kfs_to_marg.iter().copied())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(packet_ids.len(), records.len(), "duplicate packet target");
        assert!(records.iter().all(|record| {
            record
                .aom_order
                .windows(2)
                .all(|pair| pair[0].offset.saturating_add(pair[0].dof) == pair[1].offset)
        }));
        assert!(records
            .iter()
            .all(|record| record.validate_contract().is_ok()));
    }

    /// Opt-in validation/replay gate for a freshly emitted schema-4 stream.
    /// The directory is supplied by the caller so the normal suite remains
    /// independent of the large MH01 artifacts.
    #[test]
    fn load_margdata_directory_validates_schema4_when_requested() {
        let Some(path) = std::env::var_os("VISLOC_MARGDATA_V4_VALIDATION_DIR") else {
            return;
        };
        let records = load_margdata_directory(std::path::Path::new(&path))
            .expect("schema-4 MargData directory must load and validate");
        assert!(
            !records.is_empty(),
            "schema-4 validation directory is empty"
        );
        assert!(records.iter().all(|record| {
            record.schema_version == MARGDATA_SCHEMA_VERSION
                && record.fej_complete
                && record.frame_poses_fej.len() == record.frame_poses.len()
                && record.frame_states_fej.len() == record.frame_states.len()
                && record.validate_contract().is_ok()
        }));
    }
}
