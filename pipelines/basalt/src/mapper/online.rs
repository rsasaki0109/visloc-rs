//! Online (incremental) wrapper around the offline `NfrMapper`.
//!
//! This module is additive only: it does not modify
//! `pipelines/basalt/src/mapper/{mod.rs,session.rs,features.rs,triangulation.rs}`,
//! which remain the pinned offline/parity path (see
//! `docs/basalt_online_mapper_design.md`). `OnlineNfrMapper` wraps a plain
//! [`NfrMapper`] and calls its existing public methods (`add_marg_data`,
//! `build_tracks`, `setup_opt`, `optimize`, `filter_outliers`,
//! `trajectory_tum`, ...) plus the same lower-level pure frontend functions
//! `session.rs` itself calls (`extract_mapper_features`,
//! `match_stereo_features`, `query_bow_candidates`, `match_temporal_stage`,
//! `match_temporal_ransac[_seeded]`) -- it adds new incremental entry points,
//! it does not change any existing one's behavior.
//!
//! Design basis: `docs/basalt_online_mapper_design.md` Sec3. Three rules
//! drive every method here:
//!
//! 1. **Never retain raw pixel buffers past the packet that produced them.**
//!    Phase A measurement showed `img_data` (retained `OfImageData`) is the
//!    dominant term in the offline mapper's 3-6 GB RSS (Sec1.2 of the design
//!    doc); `ingest_packet` extracts features for every eligible image in a
//!    packet and removes that image's raw buffer from `NfrMapper::img_data`
//!    in the same call.
//! 2. **Query each new keyframe against the BoW database exactly once**,
//!    instead of re-querying every already-processed key on every call the
//!    way batch `NfrMapper::match_all` does. `match_new_keyframe` calls the
//!    identical `query_bow_candidates`/`match_temporal_stage`/
//!    `match_temporal_ransac` sequence `match_all`'s inner loop body uses,
//!    scoped to one query. Because `query_bow_candidates` itself already
//!    restricts candidates to strictly older frame IDs, and this module only
//!    ever calls it for a key once every strictly-older key is already
//!    inserted, the accepted-pair set is identical to a batch call over the
//!    same packets (`tests::incremental_matches_equal_batch` below).
//! 3. **The final trajectory comes from the same optimization code path as
//!    the offline mapper.** `optimize_pass` is literally the tail of
//!    `NfrMapper::run_headless` (`build_tracks` -> `setup_opt` -> `optimize`
//!    -> `filter_outliers` -> `optimize`), called via the same public
//!    methods `run_headless` calls; `finalize` runs it one last time so the
//!    reported trajectory's last optimization is not distinguishable from
//!    the offline path's.

use std::collections::{BTreeSet, HashSet};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use rayon::prelude::*;
use thiserror::Error;
use visloc_core::geometry::SE3;

use crate::{
    calibration::BasaltCalibration,
    contrast::LocalContrast,
    pyramid::{ImageError, RawU16Image},
    smoothing::GaussianSmoothing,
    vio::margdata::{MargData, OfImageData},
    ImuSample,
};

use crate::mapper::{
    features::{
        extract_mapper_features_with, match_stereo_features, match_temporal_ransac,
        match_temporal_ransac_seeded, match_temporal_stage, mutual_descriptor_matches,
        query_bow_candidates, BowQueryCandidate, FeaturePipelineError, MapperImageFeatures,
        MapperImageId, MAPPING_QUALITY_LEVEL,
    },
    hamming_distance,
    session::{
        NfrMapper, NfrMapperError, NfrMapperFilterReport, NfrMapperHeadlessConfig,
        NfrMapperLandmarkDb, NfrMapperMatchData, NfrMapperOptimizeReport, NfrMapperResult,
    },
    FeatureTracks, GlobalBaConfig, GlobalBaOptimizerState, ImagePair, MapperConfig, MatchData,
    OfflineMapperConfig, RelativePoseFactor, TimeCamId,
};

/// Diagnostic-only fine-grained stage trace, gated by the
/// `BASALT_ONLINE_MAPPER_TRACE` environment variable so it costs nothing
/// (beyond one `OnceLock` read) on every production call site. Added while
/// investigating a `--pipeline` mapper-thread stall (see
/// `docs/basalt_online_mapper_pipelined_stall.md` if present, or the fix
/// commit that introduced this): the per-packet counters in
/// [`OnlineIngestReport`] only ever surface *after* a whole `ingest_packet`
/// call returns, which gives no visibility into which sub-stage a hung call
/// is stuck in. Kept permanently since it is zero-cost when unset and cheap
/// to reach for the next time a live run needs the same visibility.
fn mapper_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("BASALT_ONLINE_MAPPER_TRACE").is_some())
}

/// `BASALT_ONLINE_MAPPER_LOOP_LOG`: one `[loop]` line on stderr per loop
/// candidate that reaches the pose gates in `match_new_keyframe` (accepted or
/// rejected), with its geometry against the current map. Zero-cost when unset.
fn loop_log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("BASALT_ONLINE_MAPPER_LOOP_LOG").is_some())
}

macro_rules! mapper_trace {
    ($($arg:tt)*) => {
        if mapper_trace_enabled() {
            eprintln!("[mapper-trace] {}", format!($($arg)*));
        }
    };
}

/// Errors raised by the online mapper's incremental entry points.
#[derive(Debug, Error)]
pub enum OnlineMapperError {
    #[error("online mapper requires calibration")]
    MissingCalibration,
    #[error("packet ingest failed: {0:?}")]
    Ingest(NfrMapperError),
    #[error("mapper calibration has no camera {camera_id} for image {timestamp_ns}")]
    MissingCamera { timestamp_ns: i64, camera_id: u16 },
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
    #[error("stereo essential computation failed for timestamp {timestamp_ns}: {source}")]
    Stereo {
        timestamp_ns: i64,
        #[source]
        source: FeaturePipelineError,
    },
}

/// Policy knobs for the online mapper's background optimization trigger.
///
/// A full warm-started global optimize is superlinear in pose/landmark
/// count (dense pose-pose Hessian, `mod.rs::linearize_vision`), so running
/// one at full iteration count on every `optimize_every_k` keyframes is
/// itself superlinear in total sequence length. The periodic trigger below
/// is deliberately cheap: a small `periodic_iterations` budget, a large
/// `optimize_every_k`, and only the final pass at channel close
/// ([`OnlineNfrMapper::finalize`]) runs the full unbounded
/// `headless.num_opt_iter` budget -- literally `run_headless`'s tail, not a
/// capped approximation of it.
#[derive(Debug, Clone)]
pub struct OnlineMapperConfig {
    /// Full sorted raw IMU stream, used only when `imu_preintegration_weight > 0.0`.
    /// Empty by default; no sample copies or scans when unused.
    pub imu_samples: Arc<[ImuSample]>,
    /// Scalar weight for preintegrated-IMU factors between consecutive keyframes.
    /// <= 0.0 (the default) disables retention, keyframe scans and IMU integration.
    pub imu_preintegration_weight: f64,
    /// Scalar weight for the joint visual-inertial global BA (`mapper::imu_ba`):
    /// preintegrated-IMU relative-state factors and bias random-walk factors
    /// between consecutive keyframes, solved *jointly* with pose (full
    /// 15-dof-per-keyframe navigation state), gravity fixed in world. This is
    /// a different, more expensive lever than `imu_preintegration_weight`
    /// (which folds a *frozen*-velocity IMU delta into the existing pose-only
    /// solver as a plain relative-pose factor); see
    /// `docs/vi_slam_global_consistency_plan.md` section 1.8 for why the
    /// frozen-velocity variants could not move MH_04/MH_05. `<= 0.0` (the
    /// default) disables it -- no raw-delta retention, no extra
    /// linearization, and `optimize_pass`/`spawn_background_optimize` keep
    /// calling the unmodified pose-only `NfrMapper::optimize` path.
    pub joint_vi_ba_weight: f64,
    /// Run a periodic `build_tracks -> setup_opt -> optimize ->
    /// filter -> optimize` pass (capped at `periodic_iterations` LM
    /// iterations per `optimize` call) after this many newly accepted
    /// keyframes.
    pub optimize_every_k: usize,
    /// LM iteration budget for each *periodic* optimize call (both the
    /// pre-filter and post-filter `optimize` calls in one pass). Kept small
    /// (default 4) because the periodic trigger's job is to keep the map
    /// roughly consistent between loop events, not to fully converge every
    /// time -- full convergence happens once, in `finalize`, which uses
    /// `headless.num_opt_iter` instead.
    pub periodic_iterations: usize,
    /// An accepted temporal match pair whose two frame IDs differ by more
    /// than this many keyframes is classified as a loop closure (reported
    /// via `OnlineIngestReport::accepted_loop_pair_count`) and, like every
    /// accepted pair, is immediately in `feature_matches`. It does *not*
    /// force its own optimize trigger: the rate limit in `ingest_packet`
    /// (see `last_optimize_started_at`'s field doc) picks it up whenever
    /// the next `optimize_every_k`-or-cooldown trigger fires, so a
    /// loop-rich revisited segment cannot make every single packet start
    /// its own background job.
    pub loop_gap_keyframes: u64,
    /// Whether a match pair classified as a loop (rank gap >
    /// `loop_gap_keyframes`) is inserted into `feature_matches`/`build_tracks`.
    ///
    /// Default **on** (matches the pin). Turning it off is a diagnostic /
    /// robustness control: on V2_03 with the default VIO keyframe spacing and a
    /// short `optimize_every_k`, a single loop match was measured to drive the
    /// global BA into a catastrophic pose excursion (see the keyframe-density
    /// section in `work/vi_slam_lean_margdata_lm_speedup_20260920.md`).
    pub loop_matching: bool,
    /// Maximum angular disagreement (degrees) between a loop match's two-view
    /// RANSAC rotation and the current pose-graph relative rotation before the
    /// loop match is discarded (not inserted into `feature_matches`, so it
    /// cannot corrupt `build_tracks`/BA). The two-view rotation is metric and
    /// reliable even when translation has drifted, so a large disagreement
    /// indicates a false loop (repeated structure). `180` disables the gate.
    ///
    /// This is the map-safety gate: unlike `loop_closure_max_rotation_error_deg`
    /// (which only gates the optional extra factor), this gates the loop
    /// *match itself*, which always feeds the global problem. A single
    /// ungated false loop was measured to destroy the V2_03 estimate.
    pub loop_match_max_rotation_error_deg: f64,
    /// Candidates considered per new keyframe query in
    /// `match_new_keyframe`, i.e. the `num_results` truncation passed to
    /// `query_bow_candidates` -- independent of (and by default smaller
    /// than) `OfflineMapperConfig::match_window` (30), which batch
    /// `NfrMapper::match_all` still uses unchanged. This is a genuine
    /// semantic difference from the offline/batch path, not merely a
    /// performance tweak: RANSAC (bounded by this many attempts per query)
    /// dominated match_new_keyframe's cost on a real MH_01 run (up to
    /// ~2.7s/packet at match_window=30, well above the real-time budget a
    /// live VIO stream needs from the mapper), and a query's true loop/
    /// temporal-continuation candidates are almost always within its
    /// highest-scoring few. `tests::inverted_index_candidates_equal_full_scan`
    /// and `tests::incremental_matches_equal_batch` compare against a batch
    /// call truncated to this same value, not the offline default, so
    /// exactness is still proven at whatever `k` is configured -- just not
    /// against the *offline mapper's own* candidate count.
    pub match_top_k: usize,
    /// Knobs `NfrMapper::run_headless` uses for `finalize`'s pass
    /// (`num_opt_iter`, `outlier_threshold`, `min_num_obs`,
    /// `temporal_seed`); `num_opt_iter` is unused by the periodic trigger,
    /// which uses `periodic_iterations` instead. Production runs use
    /// `temporal_seed: None`, exactly like the offline mapper's default
    /// `run_headless` call; tests use a fixed seed for determinism.
    pub headless: NfrMapperHeadlessConfig,
    /// L1 projection re-observation: for each new keyframe, project the
    /// persistent landmark map (surfaced by `merge_background_result`) into
    /// the new keyframe's image using the current pose and add descriptor
    /// matches found within `projection_search_radius_px`. This is the
    /// ORB-SLAM3-style local-map association that keeps map points alive
    /// across many keyframes; without it tracks die after a few keyframes
    /// (median 9 observations on MH_04).
    ///
    /// Default **off**: this adds matches beyond the batch `match_all`
    /// candidate set, so `tests::incremental_matches_equal_batch` (and the
    /// byte-exact online/batch contract) only hold when it is disabled.
    pub projection_rematch: bool,
    /// Pixel radius of the projection search window. A candidate landmark's
    /// projected pixel must fall within this radius of the matched keypoint.
    pub projection_search_radius_px: f64,
    /// Maximum Hamming distance accepted for a projection descriptor match.
    pub projection_max_hamming: u32,
    /// Only consider landmarks hosted by keyframes within this many
    /// keyframe-ranks of the query (a cheap covisibility proxy). Bounds the
    /// per-keyframe projection cost as the map grows.
    pub projection_host_window: u64,
    /// Incremental local mapping: after each new keyframe is matched and
    /// before projection re-observation, triangulate new landmarks from the
    /// just-accepted pairs and append re-observations to existing landmarks
    /// (`NfrMapper::local_map_new_keyframes`). This keeps the live `lmdb`
    /// populated between optimize merges so projection can link a new keyframe
    /// to older landmarks.
    ///
    /// Default **off** for the same byte-exactness reason as
    /// `projection_rematch`.
    pub incremental_local_mapping: bool,
    /// Loop-closure relative-pose factors: when a newly accepted match is
    /// classified as a loop (`keyframe_rank` gap > `loop_gap_keyframes`),
    /// recover its metric SE3 from the map landmarks observed in both
    /// keyframes and append it as a `RelativePoseFactor`. This is the
    /// constraint that kills long-horizon drift (see the full-sequence MH_04
    /// diagnosis in `work/vi_slam_lean_margdata_lm_speedup_20260920.md`).
    ///
    /// Default **off**; opt-in until the byte-exactness A/B is re-baselined.
    pub loop_closure_factors: bool,
    /// Minimum number of 3D-3D correspondences required before a loop factor is
    /// accepted. Guards against fitting a relative pose from a handful of
    /// outliers.
    pub loop_closure_min_correspondences: usize,
    /// Scalar weight multiplied into the loop factor's information matrix.
    pub loop_closure_weight: f64,
    /// Maximum angular disagreement (degrees) between a loop's two-view RANSAC
    /// rotation and the current VIO relative rotation for the loop to be
    /// trusted. Genuine loops can have large *translation* drift but the
    /// relative rotation estimate is robust, so a large rotation disagreement
    /// indicates a false loop (repeated structure) and the loop factor is
    /// dropped. `loop_closure_factors` must be enabled.
    pub loop_closure_max_rotation_error_deg: f64,
    /// Maximum median angular residual (degrees) for the map-based metric loop
    /// translation solve before the loop is rejected as geometrically
    /// inconsistent.
    pub loop_closure_max_metric_residual_deg: f64,
    /// Covisibility local-BA window size in keyframes (0 disables). When
    /// non-zero, `optimize_pass` runs a local windowed BA around the newest
    /// keyframe after the global steps, cheaply refining the active part of the
    /// map between global merges.
    pub local_ba_window: usize,
    /// LM iteration budget for the local windowed BA.
    pub local_ba_iterations: usize,
    /// Equalize each keyframe image's local contrast before ORB detection
    /// and description (`crate::contrast`), for maps that must localize
    /// images taken under other lighting. `None` (the default) extracts from
    /// the raw image. The VIO always tracks the raw images.
    pub local_contrast: Option<LocalContrast>,
    /// Blur each keyframe image (after `local_contrast`) before ORB
    /// detection and description (`crate::smoothing`), for grainy images
    /// whose noise ORB's pixel comparisons would pick up. `None` (the
    /// default) extracts from the image as is.
    pub feature_smoothing: Option<GaussianSmoothing>,
    /// The ORB detector's quality level
    /// (`features::detect_keypoints_mapping_with`); `None` (the default) is
    /// upstream's 0.01. Lower keeps weak texture next to strong corners.
    pub feature_quality_level: Option<f32>,
}

impl Default for OnlineMapperConfig {
    fn default() -> Self {
        Self {
            imu_samples: Arc::from([]),
            imu_preintegration_weight: 0.0,
            joint_vi_ba_weight: 0.0,
            optimize_every_k: 100,
            periodic_iterations: 4,
            loop_gap_keyframes: 30,
            loop_matching: true,
            loop_match_max_rotation_error_deg: 30.0,
            match_top_k: 5,
            headless: NfrMapperHeadlessConfig::default(),
            projection_rematch: false,
            projection_search_radius_px: 12.0,
            projection_max_hamming: 70,
            projection_host_window: 20,
            incremental_local_mapping: false,
            loop_closure_factors: false,
            loop_closure_min_correspondences: 10,
            loop_closure_weight: 1.0,
            loop_closure_max_rotation_error_deg: 180.0,
            loop_closure_max_metric_residual_deg: 5.0,
            local_ba_window: 0,
            local_ba_iterations: 4,
            local_contrast: None,
            feature_smoothing: None,
            feature_quality_level: None,
        }
    }
}

/// Per-packet counters returned by [`OnlineNfrMapper::ingest_packet`], used
/// by the online demo's stats/RTF/lag reporting and by the exactness tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct OnlineIngestReport {
    pub processed_timestamp_count: usize,
    pub skipped_ineligible_timestamp_count: usize,
    pub new_key_count: usize,
    pub detect_seconds: f64,
    pub stereo_seconds: f64,
    pub match_seconds: f64,
    pub accepted_temporal_pair_count: usize,
    pub accepted_loop_pair_count: usize,
    /// A new background optimize job was *started* this packet (see
    /// [`BackgroundOptimizeBreakdown`] -- ingestion is not blocked by it;
    /// its result is merged, and reported via `optimize_merge`, on a later
    /// packet once the background thread finishes).
    pub optimize_triggered: bool,
    /// A previously started background optimize job *finished and was
    /// merged* into the live mapper state during this packet's call. `None`
    /// on every packet where no job happened to complete.
    pub optimize_merge: Option<BackgroundOptimizeBreakdown>,
    /// Bytes retained in `NfrMapper::img_data` immediately after this call
    /// returns. Phase A identified this field as the dominant term in the
    /// offline mapper's RSS; the online demo asserts this stays ~0 (only
    /// non-zero when a packet's images are not yet pose-eligible, which is
    /// not expected on a live VIO stream since a packet's own frame poses
    /// are installed by `add_marg_data` before this method runs).
    pub retained_image_bytes: usize,
}

/// Per-stage timing for one completed background optimize job (see rule 3
/// in [`OnlineNfrMapper`]'s trigger policy): where a periodic pass's wall
/// time actually goes. `build_tracks`/`setup_opt` are full rebuilds over the
/// whole accumulated match/track graph on every call (Sec2 of the design
/// doc -- there is no incremental union-find/triangulation port), so their
/// cost grows with total sequence length regardless of the LM iteration
/// cap; `optimize1`/`optimize2` are bounded by `periodic_iterations`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BackgroundOptimizeBreakdown {
    pub build_tracks_seconds: f64,
    pub setup_opt_seconds: f64,
    pub optimize1_seconds: f64,
    pub filter_seconds: f64,
    pub optimize2_seconds: f64,
    pub total_seconds: f64,
}

/// Result of the final `optimize_pass` run when the ingest channel closes.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineFinalReport {
    pub first_optimize: NfrMapperOptimizeReport,
    pub filter: NfrMapperFilterReport,
    pub second_optimize: NfrMapperOptimizeReport,
    pub result: NfrMapperResult,
    pub trajectory_tum: String,
}

/// Incremental wrapper around [`NfrMapper`]. See the module documentation
/// for the design rationale.
pub struct OnlineNfrMapper {
    mapper: NfrMapper,
    config: OnlineMapperConfig,
    last_imu_factor_keyframe: Option<u64>,
    // Cumulative history; poses are looked up afresh at each optimize pass.
    imu_pairs: Vec<(u64, u64, crate::mapper::imu_factor::CorrectedDelta)>,
    /// Same cadence/bookkeeping as `last_imu_factor_keyframe`/`imu_pairs`,
    /// but for the joint VI-BA solver: retains the *raw* preintegrated delta
    /// (bias Jacobians + covariance) instead of `imu_pairs`'s already
    /// bias-corrected, covariance-free `CorrectedDelta`, since the joint
    /// solver relinearizes around the current bias estimate at every LM
    /// iteration (see `mapper::imu_ba`). Independent of `imu_pairs`/
    /// `imu_preintegration_weight`: the two features are gated by separate
    /// config knobs and can be enabled independently (though only one is
    /// meant to be on at a time in practice).
    last_imu_ba_factor_keyframe: Option<u64>,
    imu_ba_pairs: Vec<(u64, u64, crate::imu::ImuPreintegratedDelta)>,
    keyframes_since_optimize: usize,
    total_accepted_loops: usize,
    total_optimize_passes: usize,
    /// Assigns each distinct `frame_id` a small monotonically increasing
    /// rank the first time an image for it is detected. `TimeCamId::frame_id`
    /// is the VIO's raw per-frame index (every processed frame, not just
    /// keyframes -- see `vio/margdata.rs`'s `OfImageData::frame_id` doc), so
    /// a fixed *frame_id* gap is not a stable proxy for "how many keyframes
    /// apart": consecutive mapper keyframes can already differ by dozens of
    /// raw frame IDs. `loop_gap_keyframes` in [`OnlineMapperConfig`] is
    /// compared against this rank gap instead.
    keyframe_rank: std::collections::BTreeMap<u64, u64>,
    next_keyframe_rank: u64,
    /// Inverted HashBoW index: hash bucket -> images whose `bow_vector`
    /// contains that hash. `query_bow_candidates` (batch `match_all`'s own
    /// primitive) only ever scores a database image as a candidate when it
    /// shares at least one hash bucket with the query (`shared` in its
    /// source); every other database image is skipped regardless of the
    /// database's size. So querying only `hash_index`'s union of
    /// hash-bucket members for the query's own hashes -- instead of every
    /// detected image -- is an *exact* reduction of the candidate set
    /// `query_bow_candidates` scans, not an approximation: see
    /// `tests::inverted_index_candidates_equal_full_scan` below.
    hash_index: std::collections::BTreeMap<u32, BTreeSet<TimeCamId>>,
    /// At most one background optimize job in flight at a time (rule 3):
    /// `ingest_packet` never blocks on it, but also never starts a second
    /// one while the first is still running, so the mapper thread's own
    /// ingestion work (detect/stereo/match, all cheap) is what determines
    /// whether the mapper keeps up with the VIO, not a growing backlog of
    /// stacked optimize jobs.
    pending_optimizer: Option<JoinHandle<BackgroundOptimizeResult>>,
    /// When the currently-pending (or most recently completed) job started,
    /// and how long the most recently *completed* one took. Used by the
    /// trigger's rate limit: don't start another job until at least
    /// `last_optimize_duration * 2` has elapsed since the last one started,
    /// even if loop pairs keep arriving in the meantime (they still get
    /// recorded as accepted match-graph factors immediately; they just
    /// don't each force their own optimize job the way the first cut of
    /// this trigger policy did -- confirmed by an MH_01 run where a
    /// loop-rich revisited corridor made `optimize_triggered` fire on
    /// nearly every packet, each costing 80-130s, because every single
    /// accepted loop pair triggered its own full rebuild).
    last_optimize_started_at: Option<Instant>,
    last_optimize_duration: Duration,
    total_optimize_triggers: usize,
    /// Accepted loop pairs (image pairs, query first) collected by
    /// `match_new_keyframe`. Loop factors are **re-derived from the current
    /// map** in `optimize_pass` after every `setup_opt`, so they must not be
    /// appended to the persistent `NfrMapper::factors` (which accumulates VIO
    /// marginalization factors by design). Keeping the pairs here lets the
    /// background clone rebuild a fresh, map-consistent loop factor set each
    /// merge.
    loop_pairs: BTreeSet<ImagePair>,
}

/// Owned inputs/outputs of one background optimize job (rule 3). Computed
/// entirely on a clone of the mapper state taken at trigger time -- the
/// live `NfrMapper` (and therefore `ingest_packet`) is never touched while
/// this runs.
struct BackgroundOptimizeResult {
    poses: std::collections::BTreeMap<u64, SE3>,
    optimizer_state: GlobalBaOptimizerState,
    /// Persistent track graph and landmark database produced by this job's
    /// `build_tracks` + `setup_opt` + `filter_outliers` sequence.  Before this
    /// was carried back, the live mapper's `feature_tracks`/`lmdb` stayed
    /// empty for the whole run (they are only otherwise populated by
    /// `finalize`'s live `optimize_pass`), so there was no persistent map for
    /// projection-based re-observation to use.
    feature_tracks: FeatureTracks,
    lmdb: NfrMapperLandmarkDb,
    /// Joint-VI-BA-refined velocity/gyro-bias/accel-bias, carried back the
    /// same way `poses` is. Empty (and ignored by `merge_background_result`)
    /// whenever the joint solver did not run this pass.
    velocities: std::collections::BTreeMap<u64, Vector3<f64>>,
    gyro_biases: std::collections::BTreeMap<u64, Vector3<f64>>,
    accel_biases: std::collections::BTreeMap<u64, Vector3<f64>>,
    breakdown: BackgroundOptimizeBreakdown,
}

impl OnlineNfrMapper {
    pub fn new(
        mapper_config: MapperConfig,
        calibration: BasaltCalibration,
        feature_config: OfflineMapperConfig,
        optimize_config: GlobalBaConfig,
        config: OnlineMapperConfig,
    ) -> Self {
        let mut mapper = NfrMapper::with_calibration(mapper_config, calibration);
        mapper.set_feature_config(feature_config);
        mapper.set_optimize_config(optimize_config);
        Self {
            mapper,
            config,
            last_imu_factor_keyframe: None,
            imu_pairs: Vec::new(),
            last_imu_ba_factor_keyframe: None,
            imu_ba_pairs: Vec::new(),
            keyframes_since_optimize: 0,
            total_accepted_loops: 0,
            total_optimize_passes: 0,
            keyframe_rank: std::collections::BTreeMap::new(),
            next_keyframe_rank: 0,
            hash_index: std::collections::BTreeMap::new(),
            pending_optimizer: None,
            last_optimize_started_at: None,
            last_optimize_duration: Duration::ZERO,
            total_optimize_triggers: 0,
            loop_pairs: BTreeSet::new(),
        }
    }

    pub const fn total_optimize_triggers(&self) -> usize {
        self.total_optimize_triggers
    }

    pub const fn total_accepted_loops(&self) -> usize {
        self.total_accepted_loops
    }

    pub const fn total_optimize_passes(&self) -> usize {
        self.total_optimize_passes
    }

    /// Bytes currently retained in the wrapped mapper's raw-image map. Should
    /// be ~0 after every [`Self::ingest_packet`] call on a live stream (see
    /// [`OnlineIngestReport::retained_image_bytes`]).
    pub fn retained_image_bytes(&self) -> usize {
        self.mapper
            .img_data
            .values()
            .flatten()
            .map(|image| image.data.len() * 2)
            .sum()
    }

    /// Direct access to the wrapped mapper, e.g. for `trajectory_tum`/
    /// `result` mid-run diagnostics. Prefer [`Self::finalize`] for the
    /// end-of-sequence output.
    pub const fn inner(&self) -> &NfrMapper {
        &self.mapper
    }

    /// Ingest one MargData packet: `add_marg_data`, then incremental
    /// detect/stereo/BoW-match for exactly the images this packet
    /// introduced, then (if the trigger policy fires) a periodic
    /// optimization pass. `seed` is `None` in production (matching
    /// `NfrMapper::match_all`'s wall-clock RANSAC seed) and `Some` only for
    /// deterministic tests.
    pub fn ingest_packet(
        &mut self,
        data: &mut MargData,
        seed: Option<u32>,
    ) -> Result<OnlineIngestReport, OnlineMapperError> {
        mapper_trace!("ingest_packet: enter, images={}", data.of_images.len());
        // Non-blocking: pick up a background optimize job's result if one
        // finished since the last packet (rule 3). Ingestion below proceeds
        // immediately either way.
        mapper_trace!("ingest_packet: poll_pending_optimizer start");
        let optimize_merge = self.poll_pending_optimizer();
        mapper_trace!(
            "ingest_packet: poll_pending_optimizer done merged={}",
            optimize_merge.is_some()
        );

        mapper_trace!("ingest_packet: add_marg_data start");
        let retain_velocity_bias =
            self.config.imu_preintegration_weight > 0.0 || self.config.joint_vi_ba_weight > 0.0;
        self.mapper
            .add_marg_data_with_velocity_bias(data, retain_velocity_bias)
            .map_err(OnlineMapperError::Ingest)?;
        if self.config.imu_preintegration_weight > 0.0 {
            use std::ops::Bound::{Excluded, Unbounded};
            // An unbounded initial range includes the legitimate frame ID zero.
            let lower = self.last_imu_factor_keyframe.map_or(Unbounded, Excluded);
            let new_ids = self
                .mapper
                .frame_poses
                .range((lower, Unbounded))
                .map(|(&id, _)| id)
                .collect::<Vec<_>>();
            let ids = self
                .last_imu_factor_keyframe
                .iter()
                .copied()
                .chain(new_ids.iter().copied())
                .collect::<Vec<_>>();
            for pair in ids.windows(2) {
                let (prev_id, cur_id) = (pair[0], pair[1]);
                let (Some(&start), Some(&end), Some(&(_, bg, ba))) = (
                    self.mapper.frame_timestamps.get(&prev_id),
                    self.mapper.frame_timestamps.get(&cur_id),
                    self.mapper.frame_velocity_bias.get(&prev_id),
                ) else {
                    continue;
                };
                if end <= start {
                    continue;
                }
                let samples = &self.config.imu_samples;
                let first = samples.partition_point(|sample| sample.timestamp_ns <= start);
                let last = samples.partition_point(|sample| sample.timestamp_ns <= end);
                if let Some(delta) = crate::mapper::imu_factor::preintegrate_corrected_delta(
                    bg,
                    ba,
                    start,
                    end,
                    &samples[first..last],
                ) {
                    self.imu_pairs.push((prev_id, cur_id, delta));
                }
            }
            if let Some(&last) = new_ids.last() {
                self.last_imu_factor_keyframe = Some(last);
            }
        }
        if self.config.joint_vi_ba_weight > 0.0 {
            if let Some(calibration) = self.mapper.calibration.clone() {
                let noise = crate::mapper::imu_ba::imu_noise_model_from_calibration(&calibration);
                use std::ops::Bound::{Excluded, Unbounded};
                let lower = self.last_imu_ba_factor_keyframe.map_or(Unbounded, Excluded);
                let new_ids = self
                    .mapper
                    .frame_poses
                    .range((lower, Unbounded))
                    .map(|(&id, _)| id)
                    .collect::<Vec<_>>();
                let ids = self
                    .last_imu_ba_factor_keyframe
                    .iter()
                    .copied()
                    .chain(new_ids.iter().copied())
                    .collect::<Vec<_>>();
                for pair in ids.windows(2) {
                    let (prev_id, cur_id) = (pair[0], pair[1]);
                    let (Some(&start), Some(&end), Some(&(_, bg, ba))) = (
                        self.mapper.frame_timestamps.get(&prev_id),
                        self.mapper.frame_timestamps.get(&cur_id),
                        self.mapper.frame_velocity_bias.get(&prev_id),
                    ) else {
                        continue;
                    };
                    if end <= start {
                        continue;
                    }
                    let samples = &self.config.imu_samples;
                    let first = samples.partition_point(|sample| sample.timestamp_ns <= start);
                    let last = samples.partition_point(|sample| sample.timestamp_ns <= end);
                    if let Some(delta) = crate::mapper::imu_ba::preintegrate_delta_for_joint_ba(
                        bg,
                        ba,
                        start,
                        end,
                        &samples[first..last],
                        noise,
                    ) {
                        self.imu_ba_pairs.push((prev_id, cur_id, delta));
                    }
                }
                if let Some(&last) = new_ids.last() {
                    self.last_imu_ba_factor_keyframe = Some(last);
                }
            }
        }

        mapper_trace!("ingest_packet: add_marg_data done");
        let calibration = self
            .mapper
            .calibration
            .clone()
            .ok_or(OnlineMapperError::MissingCalibration)?;

        // Snapshot the timestamps present right now. On a live stream this
        // is exactly the packet just ingested, because every prior call
        // purges every timestamp it successfully processed (rule 1 above);
        // an entry can only still be here if a previous call found it
        // ineligible (no installed pose yet) and left it for retry.
        let timestamps = self.mapper.img_data.keys().copied().collect::<Vec<_>>();

        let mut new_keys = Vec::new();
        let mut processed_timestamp_count = 0;
        let mut skipped_ineligible_timestamp_count = 0;
        let detect_stereo_start = Instant::now();
        let mut stereo_seconds_accum = 0.0;
        mapper_trace!(
            "ingest_packet: detect/stereo loop start, timestamps={}",
            timestamps.len()
        );

        // Phase 1 (sequential, cheap): snapshot every eligible timestamp's
        // images and collect the (still-empty) set of genuinely new keys
        // each one needs detected -- the same overlap-skip filter as
        // before (a MargData packet's `of_images` is the marginalization
        // event's whole AOM window, up to 16 images, and consecutive
        // packets' windows overlap heavily; re-detecting an already-known
        // key would reproduce batch `match_all`'s exact re-querying cost,
        // per the fix note this replaced). This pass only reads mapper
        // state, so it stays a plain sequential loop; the expensive part
        // (phase 2) is what moves to rayon.
        struct EligibleTimestamp {
            timestamp_ns: i64,
            images: Vec<OfImageData>,
            new_keys: Vec<TimeCamId>,
        }
        let mut eligible_timestamps = Vec::new();
        let mut jobs: Vec<(TimeCamId, usize, usize)> = Vec::new();
        for timestamp_ns in timestamps {
            let Some(images) = self.mapper.img_data.get(&timestamp_ns).cloned() else {
                continue;
            };
            let eligible = images
                .iter()
                .any(|image| self.mapper.frame_poses.contains_key(&image.frame_id));
            if !eligible {
                skipped_ineligible_timestamp_count += 1;
                continue;
            }
            let timestamp_index = eligible_timestamps.len();
            for (image_index, image) in images.iter().enumerate() {
                if !self.mapper.frame_poses.contains_key(&image.frame_id) {
                    continue;
                }
                let key = TimeCamId::new(image.frame_id, image.camera_id);
                if self.mapper.feature_corners.contains_key(&key) {
                    continue;
                }
                jobs.push((key, timestamp_index, image_index));
            }
            eligible_timestamps.push(EligibleTimestamp {
                timestamp_ns,
                images,
                new_keys: Vec::new(),
            });
        }

        // Phase 2 (parallel): pure per-image feature extraction (FAST
        // corners, descriptors, BoW hashing) on rayon's shared pool --
        // dominated detect_seconds at ~0.1s/packet on a full MH_01
        // --pipeline run before this change; every job reads only its own
        // raw pixel buffer, so this is embarrassingly parallel regardless
        // of which timestamp or camera it belongs to.
        mapper_trace!("ingest_packet: parallel detect start, jobs={}", jobs.len());
        let extracted: Vec<Result<(TimeCamId, usize, MapperImageFeatures), OnlineMapperError>> =
            jobs.par_iter()
                .map(|&(key, timestamp_index, image_index)| {
                    let image = &eligible_timestamps[timestamp_index].images[image_index];
                    extract_one_image_features(
                        image,
                        &calibration,
                        self.mapper.feature_config,
                        self.config.local_contrast,
                        self.config.feature_smoothing,
                        self.config.feature_quality_level,
                    )
                    .map(|features| (key, timestamp_index, features))
                })
                .collect();
        mapper_trace!("ingest_packet: parallel detect done");

        // Phase 3 (sequential): apply results in original job order,
        // propagating the first error exactly like the prior sequential
        // loop's `?` did -- an image after a failing one is never
        // inserted, matching the old early-return's observable state even
        // though every job was already computed in parallel.
        for result in extracted {
            let (key, timestamp_index, features) = result?;
            self.insert_detected_image(key, key.frame_id, features);
            new_keys.push(key);
            eligible_timestamps[timestamp_index].new_keys.push(key);
        }

        // Phase 4 (sequential): stereo matching + cleanup, unchanged from
        // the prior per-timestamp behavior -- stereo needs both of a
        // timestamp's images already detected (from this call or an
        // earlier one), and this stays cheap (~4ms/packet total) so it is
        // not itself a parallelization target.
        for entry in eligible_timestamps {
            // Same overlap reasoning as detection: if both of this
            // timestamp's camera images were already matched by an earlier
            // packet, re-running stereo matching would just recompute an
            // identical (deterministic) result for free-standing cost.
            if !entry.new_keys.is_empty() {
                mapper_trace!(
                    "ingest_packet: match_stereo_one_timestamp start ts={}",
                    entry.timestamp_ns
                );
                let stereo_start = Instant::now();
                self.match_stereo_one_timestamp(entry.timestamp_ns, &entry.images, &calibration)?;
                stereo_seconds_accum += stereo_start.elapsed().as_secs_f64();
                mapper_trace!(
                    "ingest_packet: match_stereo_one_timestamp done ts={}",
                    entry.timestamp_ns
                );
            }

            // Rule 1: drop the raw pixels now that detection and stereo
            // matching (which only needs already-extracted features, not
            // pixels) are both done for this timestamp.
            self.mapper.img_data.remove(&entry.timestamp_ns);
            processed_timestamp_count += 1;
        }
        let detect_seconds =
            (detect_stereo_start.elapsed().as_secs_f64() - stereo_seconds_accum).max(0.0);

        // Rule 2: query each newly detected key exactly once, in ascending
        // frame order, against the database of already-inserted keys.
        new_keys.sort();
        let match_start = Instant::now();
        let mut accepted_temporal_pair_count = 0;
        let mut accepted_loop_pair_count = 0;
        mapper_trace!(
            "ingest_packet: match loop start, new_keys={}",
            new_keys.len()
        );
        for &key in &new_keys {
            mapper_trace!("ingest_packet: match_new_keyframe start key={key:?}");
            let (accepted, loops) = self.match_new_keyframe(key, seed);
            mapper_trace!("ingest_packet: match_new_keyframe done key={key:?} accepted={accepted} loops={loops}");
            accepted_temporal_pair_count += accepted;
            accepted_loop_pair_count += loops;
        }
        let match_seconds = match_start.elapsed().as_secs_f64();
        self.total_accepted_loops += accepted_loop_pair_count;

        let new_keyframe_count = new_keys
            .iter()
            .map(|key| key.frame_id)
            .collect::<BTreeSet<_>>()
            .len();
        self.keyframes_since_optimize += new_keyframe_count;

        // Rate limit (rule 1): a keyframe-count threshold alone re-triggers
        // on literally every packet through a loop-rich revisited segment,
        // since each packet's own newly accepted loop pair satisfies
        // `accepted_loop_pair_count > 0` independently -- confirmed by an
        // MH_01 run where that fired ~120 times in a row at 80-130s each.
        // Loops accepted while a job is running or cooling down still land
        // in `feature_matches`/`feature_match_data` immediately (rule 2 is
        // unaffected); they are simply picked up by whichever optimize job
        // runs next instead of each forcing their own.
        let cooldown_elapsed = self.last_optimize_started_at.is_none_or(|started| {
            started.elapsed() >= self.last_optimize_duration.saturating_mul(2)
        });
        let should_trigger = self.pending_optimizer.is_none()
            && !new_keys.is_empty()
            && self.keyframes_since_optimize >= self.config.optimize_every_k
            && cooldown_elapsed;
        if should_trigger {
            mapper_trace!("ingest_packet: spawn_background_optimize start");
            self.spawn_background_optimize();
            mapper_trace!("ingest_packet: spawn_background_optimize done");
            self.keyframes_since_optimize = 0;
            self.total_optimize_triggers += 1;
        }

        mapper_trace!("ingest_packet: exit");
        Ok(OnlineIngestReport {
            processed_timestamp_count,
            skipped_ineligible_timestamp_count,
            new_key_count: new_keys.len(),
            detect_seconds,
            stereo_seconds: stereo_seconds_accum,
            match_seconds,
            accepted_temporal_pair_count,
            accepted_loop_pair_count,
            optimize_triggered: should_trigger,
            optimize_merge,
            retained_image_bytes: self.retained_image_bytes(),
        })
    }

    /// Mirrors `NfrMapper::detect_keypoints`'s per-image body
    /// (`session.rs`), scoped to one already pose-eligible image: runs the
    /// pure [`extract_one_image_features`] step and applies its result to
    /// mapper state. Detection itself (the expensive part) has moved to
    /// `ingest_packet`'s own parallel phase; this method now only does the
    /// cheap, must-stay-sequential bookkeeping insert.
    fn insert_detected_image(
        &mut self,
        key: TimeCamId,
        frame_id: u64,
        features: MapperImageFeatures,
    ) {
        for entry in &features.bow_vector {
            self.hash_index.entry(entry.hash).or_default().insert(key);
        }
        self.mapper.feature_corners.insert(key, features);
        self.keyframe_rank.entry(frame_id).or_insert_with(|| {
            let rank = self.next_keyframe_rank;
            self.next_keyframe_rank += 1;
            rank
        });
    }

    /// Mirrors `NfrMapper::match_stereo`'s per-timestamp body, scoped to one
    /// timestamp whose images were just detected.
    fn match_stereo_one_timestamp(
        &mut self,
        timestamp_ns: i64,
        images: &[OfImageData],
        calibration: &BasaltCalibration,
    ) -> Result<(), OnlineMapperError> {
        let camera_0 = calibration
            .camera_to_imu(0)
            .ok_or(OnlineMapperError::MissingCamera {
                timestamp_ns,
                camera_id: 0,
            })?;
        let camera_1 = calibration
            .camera_to_imu(1)
            .ok_or(OnlineMapperError::MissingCamera {
                timestamp_ns,
                camera_id: 1,
            })?;
        let t_0_1 = camera_0.inverse().compose(camera_1);

        let Some(frame_id) = stereo_frame_id(timestamp_ns, images, 0) else {
            return Ok(());
        };
        let left_id = TimeCamId::new(frame_id, 0);
        let right_id = TimeCamId::new(
            stereo_frame_id(timestamp_ns, images, 1).unwrap_or(frame_id),
            1,
        );
        self.mapper
            .feature_corners
            .entry(left_id)
            .or_insert_with(empty_mapper_features);
        self.mapper
            .feature_corners
            .entry(right_id)
            .or_insert_with(empty_mapper_features);

        let stereo = {
            let left = self
                .mapper
                .feature_corners
                .get(&left_id)
                .expect("stereo left feature map materialized");
            let right = self
                .mapper
                .feature_corners
                .get(&right_id)
                .expect("stereo right feature map materialized");
            match_stereo_features(left, right, &t_0_1, self.mapper.feature_config).map_err(
                |source| OnlineMapperError::Stereo {
                    timestamp_ns,
                    source,
                },
            )?
        };
        if stereo.mapper_feature_matches_stored {
            let inliers = stereo
                .essential_inliers
                .into_iter()
                .map(|(left, right)| (left as u64, right as u64))
                .collect::<Vec<_>>();
            self.mapper.feature_match_data.insert(
                (left_id, right_id),
                NfrMapperMatchData {
                    t_i_j: t_0_1,
                    matches: stereo
                        .raw_matches
                        .into_iter()
                        .map(|match_| (match_.left, match_.right))
                        .collect(),
                    inliers: inliers.clone(),
                },
            );
            self.mapper
                .feature_matches
                .insert((left_id, right_id), MatchData::new(inliers));
        }
        Ok(())
    }

    /// Query the BoW database (already-inserted keys only -- see rule 2 in
    /// the module documentation) for one newly detected image and run the
    /// same descriptor-match/RANSAC accept logic `NfrMapper::match_all`'s
    /// inner loop uses. Returns `(accepted_pair_count, loop_pair_count)`.
    /// Build the candidate list for one query using the inverted
    /// `hash_index` instead of a full scan of every detected image.
    ///
    /// Exact reduction, not an approximation: `query_bow_candidates` (the
    /// same primitive batch `NfrMapper::match_all` uses) only ever scores a
    /// database entry when it shares a hash bucket with the query (see
    /// `hash_index`'s field doc), so restricting the scanned set to the
    /// union of this query's own hash buckets' members is guaranteed to
    /// include every image that could possibly score, and excludes only
    /// images that would have scored zero/unshared anyway. Exposed as its
    /// own method (rather than inlined in `match_new_keyframe`) so
    /// `tests::inverted_index_candidates_equal_full_scan` can compare it
    /// directly against a full-scan call with the same inputs.
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
                self.mapper
                    .feature_corners
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

    fn match_new_keyframe(&mut self, query_id: TimeCamId, seed: Option<u32>) -> (usize, usize) {
        let config = self.mapper.feature_config;
        let mut match_config = config;
        match_config.max_hamming = 70;
        match_config.second_best_ratio = 1.2;

        let Some(query_features) = self.mapper.feature_corners.get(&query_id).cloned() else {
            return (0, 0);
        };
        // Deliberately `self.config.match_top_k`, not
        // `config.match_window` (which batch `match_all` still uses
        // unchanged) -- see `OnlineMapperConfig::match_top_k`'s field doc.
        let candidates =
            self.bow_candidates_via_index(query_id, &query_features, self.config.match_top_k);
        mapper_trace!(
            "match_new_keyframe: query={query_id:?} candidates={}",
            candidates.len()
        );

        // Per-candidate descriptor matching + 5-point RANSAC dominates
        // per-packet wall time (~0.5s/packet of the ~2.4s total measured on
        // a full MH_01 --pipeline run, vs ~0.1s/packet for detection) and is
        // embarrassingly parallel: each candidate's outcome depends only on
        // `feature_corners` (read-only here -- no candidate mutates it) and
        // its own gate/RANSAC result, never on another candidate's outcome.
        // Run every candidate's match+RANSAC concurrently on the shared
        // rayon pool the demo's `--threads` flag sizes, collecting outcomes
        // into a `Vec` (an `IndexedParallelIterator`, so `collect` preserves
        // the original candidate order regardless of completion order), and
        // apply the accepted ones -- the `feature_match_data`/
        // `feature_matches` inserts and the loop-count bookkeeping -- back
        // on `self` sequentially, in that same original order. This keeps
        // the accepted-pair set and insertion order bit-identical to the
        // fully sequential version; only the wall-clock cost of getting
        // there changes. (An earlier revision tried a small dedicated pool
        // here on the theory that sharing the global pool would
        // oversubscribe VIO's own concurrent frontend/estimator work; a
        // second full-MH_01 measurement showed that variant with *both*
        // worse per-packet mapper cost and worse whole-system wall time,
        // consistent with rising host contention across the measurement
        // session rather than a real effect of pool choice, so the simpler
        // shared-pool version is what's kept.)
        let feature_corners = &self.mapper.feature_corners;
        let outcomes: Vec<Option<CandidateMatchOutcome>> = candidates
            .par_iter()
            .map(|candidate| {
                evaluate_match_candidate(
                    feature_corners,
                    query_id,
                    candidate,
                    config,
                    match_config,
                    seed,
                )
            })
            .collect();

        let mut accepted_count = 0;
        let mut loop_count = 0;
        let mut loop_pairs: Vec<ImagePair> = Vec::new();
        for outcome in outcomes.into_iter().flatten() {
            let CandidateMatchOutcome {
                left_id,
                right_id,
                t_i_j,
                raw_matches,
                inliers,
            } = outcome;
            // Compare keyframe *rank* (assignment order), not raw
            // frame_id: frame_id is the VIO's per-frame index and
            // advances on every processed frame, not just keyframes, so
            // a frame_id gap is not a stable "how many keyframes apart"
            // proxy (see the `keyframe_rank` field doc).
            let left_rank = self
                .keyframe_rank
                .get(&left_id.frame_id)
                .copied()
                .unwrap_or(0);
            let right_rank = self
                .keyframe_rank
                .get(&right_id.frame_id)
                .copied()
                .unwrap_or(0);
            let gap = left_rank.abs_diff(right_rank);
            let is_loop = gap > self.config.loop_gap_keyframes;
            if is_loop && !self.config.loop_matching {
                continue;
            }
            if is_loop {
                let rotation_error = self.loop_match_rotation_error_deg(left_id, right_id, &t_i_j);
                let verdict = if self.config.loop_match_max_rotation_error_deg < 180.0
                    && rotation_error
                        .is_some_and(|angle| angle > self.config.loop_match_max_rotation_error_deg)
                {
                    "reject-rotation"
                } else {
                    "accept"
                };
                if loop_log_enabled() {
                    let (distance, travelled, seconds) = self
                        .loop_match_geometry(left_id, right_id)
                        .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
                    let timestamp = |id: TimeCamId| {
                        self.mapper
                            .frame_timestamps
                            .get(&id.frame_id)
                            .copied()
                            .unwrap_or(-1)
                    };
                    eprintln!(
                        "[loop] {verdict} left={} right={} left_ts={} right_ts={} gap={gap} inliers={} raw={} rot_err_deg={:.2} dist_m={distance:.2} travelled_m={travelled:.1} dt_s={seconds:.1}",
                        left_id.frame_id,
                        right_id.frame_id,
                        timestamp(left_id),
                        timestamp(right_id),
                        inliers.len(),
                        raw_matches.len(),
                        rotation_error.unwrap_or(f64::NAN),
                    );
                }
                if verdict != "accept" {
                    mapper_trace!("match_new_keyframe: {verdict} loop {left_id:?}/{right_id:?}");
                    continue;
                }
            }
            self.mapper.feature_match_data.insert(
                (left_id, right_id),
                NfrMapperMatchData {
                    t_i_j,
                    matches: raw_matches,
                    inliers: inliers.clone(),
                },
            );
            self.mapper
                .feature_matches
                .insert((left_id, right_id), MatchData::new(inliers));
            accepted_count += 1;
            if is_loop {
                loop_count += 1;
                if self.config.loop_closure_factors {
                    loop_pairs.push((left_id, right_id));
                }
            }
        }
        if self.config.loop_closure_factors {
            // Record the pairs only; `optimize_pass` re-derives factors from the
            // fresh map after `setup_opt`.
            for pair in loop_pairs {
                self.loop_pairs.insert(pair);
            }
        }
        if self.config.incremental_local_mapping {
            let (attempted, accepted, rejected) = self
                .mapper
                .local_map_new_keyframes(std::slice::from_ref(&query_id));
            mapper_trace!(
                "local_map_new_keyframes: query={query_id:?} attempted={attempted} accepted={accepted} rejected={rejected}"
            );
        }
        if self.config.projection_rematch {
            accepted_count += self.projection_rematch(query_id);
        }
        mapper_trace!(
            "match_new_keyframe: query={query_id:?} accepted={accepted_count} loops={loop_count}"
        );
        (accepted_count, loop_count)
    }

    /// Angular disagreement (degrees) between a loop match's two-view RANSAC
    /// rotation and the current pose-graph relative rotation. `None` when
    /// either frame pose is unknown. Used by the loop-match safety gate in
    /// `match_new_keyframe`.
    fn loop_match_rotation_error_deg(
        &self,
        left_id: TimeCamId,
        right_id: TimeCamId,
        t_i_j: &SE3,
    ) -> Option<f64> {
        let left_pose = self.mapper.frame_poses.get(&left_id.frame_id)?;
        let right_pose = self.mapper.frame_poses.get(&right_id.frame_id)?;
        let camera_relative =
            self.camera_relative_pose(left_id, left_pose, right_id, right_pose)?;
        let delta = camera_relative.rotation.inverse() * t_i_j.rotation;
        Some(delta.angle().to_degrees())
    }

    /// `T_cl_cr` from the map's IMU poses (`frame_poses` are `T_w_i`): the
    /// frame in which the two-view RANSAC result `t_i_j` lives. Comparing
    /// `t_i_j` with the IMU-frame relative pose instead mixes frames, which
    /// on a camera mounted at a large angle to the IMU (Aria) turns the
    /// viewpoint change itself into "rotation error".
    fn camera_relative_pose(
        &self,
        left_id: TimeCamId,
        left_pose: &SE3,
        right_id: TimeCamId,
        right_pose: &SE3,
    ) -> Option<SE3> {
        let calibration = self.mapper.calibration.as_ref()?;
        Some(camera_relative_from_imu(
            left_pose,
            calibration.camera_to_imu(left_id.cam_id)?,
            right_pose,
            calibration.camera_to_imu(right_id.cam_id)?,
        ))
    }

    /// `T_ir_il` (IMU frames) from a camera-frame relative pose `T_cl_cr`:
    /// `T_il_ir = T_i_cl * T_cl_cr * T_i_cr^-1`.
    fn imu_right_left_from_cameras(
        &self,
        left_id: TimeCamId,
        right_id: TimeCamId,
        t_cl_cr: &SE3,
    ) -> Option<SE3> {
        let calibration = self.mapper.calibration.as_ref()?;
        Some(imu_right_left_from_camera_relative(
            calibration.camera_to_imu(left_id.cam_id)?,
            t_cl_cr,
            calibration.camera_to_imu(right_id.cam_id)?,
        ))
    }

    /// Current-map geometry of a loop pair: (distance between the two
    /// keyframes' positions, path length along the keyframes in between,
    /// seconds between them). `None` when either pose is unknown.
    fn loop_match_geometry(
        &self,
        left_id: TimeCamId,
        right_id: TimeCamId,
    ) -> Option<(f64, f64, f64)> {
        let left = self.mapper.frame_poses.get(&left_id.frame_id)?.translation;
        let right = self.mapper.frame_poses.get(&right_id.frame_id)?.translation;
        let (lo, hi) = if left_id.frame_id <= right_id.frame_id {
            (left_id.frame_id, right_id.frame_id)
        } else {
            (right_id.frame_id, left_id.frame_id)
        };
        let mut travelled = 0.0_f64;
        let mut previous: Option<Vector3<f64>> = None;
        for pose in self
            .mapper
            .frame_poses
            .range(lo..=hi)
            .map(|(_, pose)| pose.translation)
        {
            if let Some(prev) = previous {
                travelled += (pose - prev).norm();
            }
            previous = Some(pose);
        }
        let seconds = match (
            self.mapper.frame_timestamps.get(&lo),
            self.mapper.frame_timestamps.get(&hi),
        ) {
            (Some(a), Some(b)) => (b - a) as f64 * 1e-9,
            _ => f64::NAN,
        };
        Some(((left - right).norm(), travelled, seconds))
    }

    /// Build a metric loop-closure `RelativePoseFactor` for an accepted loop
    /// pair `(left_id, right_id)`.
    ///
    /// The two-view RANSAC measurement (`t_i_j`) is scale-free because the
    /// temporal matcher works on unit rays, so it cannot constrain metric
    /// drift. Instead this gathers the map landmarks observed in *both*
    /// images, expresses each landmark in its observing camera's frame, and
    /// solves the point-set registration `p_left ≈ T_left_right * p_right`.
    /// The result is the metric relative pose ORB-SLAM3 derives from its loop
    /// map points, and it is the constraint that kills long-horizon drift.
    ///
    /// Returns `None` when fewer than `loop_closure_min_correspondences`
    /// landmarks are shared or the fit is degenerate.
    fn loop_closure_factor(
        &self,
        left_id: TimeCamId,
        right_id: TimeCamId,
    ) -> Option<RelativePoseFactor> {
        let calibration = self.mapper.calibration.as_ref()?;
        let left_pose = self.mapper.frame_poses.get(&left_id.frame_id)?.clone();
        let right_pose = self.mapper.frame_poses.get(&right_id.frame_id)?.clone();
        let t_imu_cam_r = calibration.camera_to_imu(right_id.cam_id)?.clone();
        let t_world_cam_r = right_pose.compose(&t_imu_cam_r);
        let t_cam_r_world = t_world_cam_r.inverse();

        // `loop_pairs` come straight from `match_new_keyframe`, which inserts
        // `feature_match_data` under `(query_id, other_id)`.
        let data = self.mapper.feature_match_data.get(&(left_id, right_id))?;

        // Geometric plausibility gate: compare the loop's two-view RANSAC
        // rotation with the current VIO relative rotation. A false loop from
        // repeated structure disagrees grossly; a genuine loop, whose
        // translation may have drifted, agrees on rotation.
        {
            let vio_relative =
                self.camera_relative_pose(left_id, &left_pose, right_id, &right_pose)?;
            let measured = data.t_i_j.clone();
            let delta = vio_relative.rotation.inverse() * measured.rotation;
            let angle_deg = delta.angle().to_degrees();
            if angle_deg > self.config.loop_closure_max_rotation_error_deg {
                mapper_trace!(
                    "loop_closure_factor: reject rotation mismatch {angle_deg:.1} deg for {left_id:?}/{right_id:?}"
                );
                return None;
            }
        }

        // Metric loop recovery without requiring the new keyframe to already
        // own map landmarks (it never does: `setup_opt` predates it). The old
        // keyframe's landmarks are metric world points; each matched query
        // feature supplies a bearing. With the reliable two-view RANSAC
        // rotation `R` (from `t_i_j`), the metric translation `t` solves the
        // bearing constraint `d × (R * p_old + t) = 0`, which is linear in `t`:
        //
        //     [d]x * t = -[d]x * (R * p_old)
        //
        // Stacking at least two non-degenerate inliers gives `t`. The per-inlier
        // angular residual is a genuine geometric verification, so false loops
        // (repeated structure) are rejected by their large residual rather than
        // by the blunt VIO-prior gate above.
        let query_features = self.mapper.feature_corners.get(&left_id)?;
        let rotation = data.t_i_j.rotation;
        let mut normal = Matrix3::zeros();
        let mut rhs = Vector3::zeros();
        let mut correspondences: Vec<(Vector3<f64>, Vector3<f64>, Vector3<f64>)> = Vec::new();
        for &(query_feature, old_feature) in &data.inliers {
            let Some(landmark) = self
                .mapper
                .lmdb
                .landmark_for_observation(right_id, old_feature)
            else {
                continue;
            };
            let Some(p_host) = landmark.position_in_host() else {
                continue;
            };
            let Some(host_pose) = self.mapper.frame_poses.get(&landmark.host.frame_id) else {
                continue;
            };
            let Some(host_extrinsic) = calibration.camera_to_imu(landmark.host.cam_id) else {
                continue;
            };
            let p_world = host_pose
                .compose(host_extrinsic)
                .transform_point(&Point3::from(p_host))
                .coords;
            let p_old_cam = t_cam_r_world.transform_point(&Point3::from(p_world)).coords;
            let Some(ray) = query_features.rays.get(query_feature as usize) else {
                continue;
            };
            let d = Vector3::new(ray[0], ray[1], ray[2]);
            if !d.iter().all(|v| v.is_finite()) || d.norm() < 1e-9 {
                continue;
            }
            let d = d.normalize();
            let rotated = rotation.transform_vector(&p_old_cam);
            let skew = skew_vector(&d);
            normal += skew.transpose() * skew;
            rhs += -skew.transpose() * (skew * rotated);
            correspondences.push((d, rotated, p_old_cam));
        }
        mapper_trace!(
            "loop_closure_factor: metric candidates {}/{} {left_id:?}/{right_id:?}",
            correspondences.len(),
            data.inliers.len()
        );
        // A well-conditioned bearing system needs at least two inliers; require
        // the configured minimum before trusting the translation.
        if correspondences.len() >= self.config.loop_closure_min_correspondences.max(3) {
            if let Some(normal_inv) = normal.try_inverse() {
                let translation = normal_inv * rhs;
                // Geometric verification: angular residual between each query
                // bearing and the reprojected point.
                let mut residuals: Vec<f64> = correspondences
                    .iter()
                    .filter_map(|(d, rotated, _)| {
                        let predicted = rotated + translation;
                        let norm = predicted.norm();
                        if norm < 1e-9 {
                            return None;
                        }
                        let cos = (d.dot(&predicted) / norm).clamp(-1.0, 1.0);
                        Some(cos.acos().to_degrees())
                    })
                    .collect();
                residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = residuals[residuals.len() / 2];
                if median <= self.config.loop_closure_max_metric_residual_deg {
                    // `rotation`/`translation` relate the two *cameras*; the
                    // factor constrains the IMU poses.
                    let t_cl_cr = SE3::new(rotation, translation);
                    let t_right_left =
                        self.imu_right_left_from_cameras(left_id, right_id, &t_cl_cr)?;
                    let q = t_right_left.rotation.quaternion();
                    return Some(self.build_loop_factor(
                        right_id.frame_id,
                        left_id.frame_id,
                        t_right_left.translation,
                        [q.w, q.i, q.j, q.k],
                        true,
                        correspondences.len(),
                    ));
                }
                mapper_trace!(
                    "loop_closure_factor: metric reject median residual {median:.1} deg ({}/{} corr) {left_id:?}/{right_id:?}",
                    correspondences.len(),
                    data.inliers.len()
                );
            }
        }

        // Fallback: the two-view RANSAC measurement is scale-free, but its
        // rotation is a valid, VIO-independent loop constraint. Keep the
        // translation at the current VIO estimate (neutral) and put near-zero
        // information on it, so only rotation closes the loop (what removes yaw
        // drift, the dominant long-horizon error mode).
        //
        // A 2-3 inlier "loop" must not carry the same authority as a 100-inlier
        // one, so the factor is gated on inlier count and its information is
        // scaled by that count (see `build_loop_factor`).
        let inliers = data.inliers.len();
        let min_rotation_inliers = self.config.loop_closure_min_correspondences / 2;
        if inliers < min_rotation_inliers.max(5) {
            return None;
        }
        let relative = right_pose.inverse().compose(&left_pose);
        // `data.t_i_j` is the cameras' `T_cl_cr`; the factor wants the IMU
        // poses' `T_ir_il`.
        let rotation = self
            .imu_right_left_from_cameras(left_id, right_id, &data.t_i_j)?
            .rotation;
        let q = rotation.quaternion();
        Some(self.build_loop_factor(
            right_id.frame_id,
            left_id.frame_id,
            relative.translation,
            [q.w, q.i, q.j, q.k],
            false,
            inliers,
        ))
    }

    /// Assemble a loop `RelativePoseFactor`.
    ///
    /// When `metric` is true the information matrix is strong on all six DOF;
    /// otherwise only rotation is constrained (`1e-6` on translation). The
    /// information is scaled by `support` (the correspondence/inlier count)
    /// relative to `loop_closure_min_correspondences`, capped at 3x, so weak
    /// loops contribute weakly.
    fn build_loop_factor(
        &self,
        from: u64,
        to: u64,
        translation: Vector3<f64>,
        rotation: [f64; 4],
        metric: bool,
        support: usize,
    ) -> RelativePoseFactor {
        let reference = self.config.loop_closure_min_correspondences.max(1) as f64;
        let support_scale = (support as f64 / reference).clamp(0.25, 3.0);
        let strong = self.config.loop_closure_weight * support_scale / 1e-3;
        let weak = 1e-6;
        let mut information = vec![0.0; 36];
        for i in 0..6 {
            let is_translation = i < 3;
            let value = if metric || !is_translation {
                strong
            } else {
                weak
            };
            information[i * 6 + i] = value;
        }
        RelativePoseFactor {
            from,
            to,
            translation: translation.into(),
            rotation,
            information,
            weight: 1.0,
        }
    }

    /// L1 projection re-observation: project persistent landmarks into
    /// `query_id` using the current pose and add descriptor matches found in a
    /// pixel window as new inlier pairs.
    ///
    /// Returns the number of accepted projection matches. Only landmarks
    /// hosted within `projection_host_window` keyframe-ranks of the query are
    /// considered (a cheap covisibility proxy), and only the highest-scoring
    /// descriptor within `projection_search_radius_px` under
    /// `projection_max_hamming` is accepted. New pairs are inserted into
    /// `feature_match_data`/`feature_matches` with the host as the older
    /// image, so `build_tracks` folds them into longer tracks.
    fn projection_rematch(&mut self, query_id: TimeCamId) -> usize {
        let Some(calibration) = self.mapper.calibration.clone() else {
            return 0;
        };
        let Some(query_pose) = self.mapper.frame_poses.get(&query_id.frame_id).cloned() else {
            return 0;
        };
        let Some(query_camera) = calibration.camera(query_id.cam_id).copied() else {
            return 0;
        };
        let Some(t_imu_cam_q) = calibration.camera_to_imu(query_id.cam_id).cloned() else {
            return 0;
        };
        let Some(query_features) = self.mapper.feature_corners.get(&query_id) else {
            return 0;
        };
        let query_rank = self
            .keyframe_rank
            .get(&query_id.frame_id)
            .copied()
            .unwrap_or(0);
        let host_window = self.config.projection_host_window;
        let radius = self.config.projection_search_radius_px as f32;
        let radius_sq = radius * radius;
        let max_hamming = self.config.projection_max_hamming;

        // Snapshot the candidate landmarks and their host observations before
        // mutating the match graph below (borrow checker: `self.mapper.lmdb`
        // is read while `feature_match_data` is written).
        let candidates: Vec<(TimeCamId, u64, Point3<f64>, [u8; 32])> = self
            .mapper
            .lmdb
            .landmarks
            .values()
            .filter_map(|landmark| {
                let host = landmark.host;
                if host.frame_id == query_id.frame_id {
                    return None;
                }
                let host_rank = self.keyframe_rank.get(&host.frame_id).copied().unwrap_or(0);
                if query_rank.abs_diff(host_rank) > host_window {
                    return None;
                }
                let position_host = landmark.position_in_host()?;
                let observation = landmark
                    .observations
                    .iter()
                    .find(|observation| observation.image == host)?;
                let host_features = self.mapper.feature_corners.get(&host)?;
                let descriptor = host_features
                    .descriptors
                    .get(usize::try_from(observation.feature_id).ok()?)?;
                Some((
                    host,
                    observation.feature_id,
                    Point3::from(position_host),
                    *descriptor,
                ))
            })
            .collect();

        mapper_trace!(
            "projection_rematch: query={query_id:?} landmarks={} candidates={}",
            self.mapper.lmdb.landmarks.len(),
            candidates.len()
        );
        let mut added = 0usize;
        let mut touched: BTreeSet<ImagePair> = BTreeSet::new();
        for (host, host_feature, position_host, descriptor) in candidates {
            let Some(t_imu_cam_h) = calibration.camera_to_imu(host.cam_id).cloned() else {
                continue;
            };
            let Some(host_pose) = self.mapper.frame_poses.get(&host.frame_id).cloned() else {
                continue;
            };
            // T_w_c = T_w_i * T_i_c, then express the host-camera point in the
            // query camera frame: T_cq_ch = (T_w_cq)^-1 * T_w_ch.
            let t_w_cq = query_pose.compose(&t_imu_cam_q);
            let t_w_ch = host_pose.compose(&t_imu_cam_h);
            let t_cq_ch = t_w_cq.inverse().compose(&t_w_ch);
            let point_query = t_cq_ch.transform_point(&position_host);
            let Some(projected) = query_camera.project(&point_query) else {
                continue;
            };
            if point_query.z <= 0.0 {
                continue;
            }
            // Best descriptor within the pixel window.
            let mut best: Option<(u32, u64)> = None;
            for (index, corner) in query_features.corners.iter().enumerate() {
                let dx = corner.x as f32 - projected.x as f32;
                let dy = corner.y as f32 - projected.y as f32;
                if dx * dx + dy * dy > radius_sq {
                    continue;
                }
                let Some(query_descriptor) = query_features.descriptors.get(index) else {
                    continue;
                };
                let distance = hamming_distance(&descriptor, query_descriptor);
                if distance > max_hamming {
                    continue;
                }
                if best.is_none_or(|(best_distance, _)| distance < best_distance) {
                    best = Some((distance, index as u64));
                }
            }
            let Some((_distance, query_feature)) = best else {
                continue;
            };
            let pair = (host, query_id);
            let entry = self
                .mapper
                .feature_match_data
                .entry(pair)
                .or_insert_with(|| NfrMapperMatchData {
                    t_i_j: t_cq_ch.clone(),
                    matches: Vec::new(),
                    inliers: Vec::new(),
                });
            // One-to-one: skip if either endpoint already claimed in this pair.
            if entry
                .inliers
                .iter()
                .any(|(left, right)| *left == host_feature || *right == query_feature)
            {
                continue;
            }
            entry.matches.push((host_feature, query_feature));
            entry.inliers.push((host_feature, query_feature));
            touched.insert(pair);
            added += 1;
        }
        mapper_trace!("projection_rematch: query={query_id:?} added={added}");
        // Reflect the projection inliers in the inlier-only `feature_matches`
        // view for exactly the pairs this call touched, so `build_tracks`
        // folds them into longer tracks. Existing pairs keep their prior
        // entries for untouched pairs.
        for pair in touched {
            if let Some(data) = self.mapper.feature_match_data.get(&pair) {
                self.mapper
                    .feature_matches
                    .insert(pair, MatchData::new(data.inliers.clone()));
            }
        }
        added
    }

    /// Rule 3: the exact tail of `NfrMapper::run_headless`
    /// (`build_tracks -> setup_opt -> optimize -> filter_outliers ->
    /// optimize`), called through the same public methods `run_headless`
    /// itself calls. `num_opt_iter` is the LM iteration budget passed to
    /// both `optimize` calls: the periodic trigger passes
    /// `config.periodic_iterations` (small, warm-started off the persistent
    /// `GlobalBaOptimizerState`), while [`Self::finalize`] passes
    /// `config.headless.num_opt_iter` (the same unbounded budget
    /// `run_headless` itself uses) so the shipped trajectory's last
    /// optimization is not a capped approximation of the offline path's.
    /// Re-derive loop-closure relative-pose factors from the current map for
    /// every collected loop pair. Called after `setup_opt` in `optimize_pass`,
    /// so the factors reflect the freshly rebuilt landmark positions.
    fn rebuild_loop_factors(&self) -> Vec<RelativePoseFactor> {
        if !self.config.loop_closure_factors || self.loop_pairs.is_empty() {
            return Vec::new();
        }
        let mut factors = Vec::new();
        for &(left_id, right_id) in &self.loop_pairs {
            if let Some(factor) = self.loop_closure_factor(left_id, right_id) {
                factors.push(factor);
            }
        }
        mapper_trace!(
            "loop_closure_factors: pairs={} accepted={}",
            self.loop_pairs.len(),
            factors.len()
        );
        factors
    }

    /// Cumulative refinement uses current poses, including background BA merges.
    /// Bound the dense 3N solve (quadratic storage, cubic worst-case work).
    fn refresh_imu_factors(&mut self) {
        if self.config.imu_preintegration_weight > 0.0 && !self.imu_pairs.is_empty() {
            const MAX_IMU_PAIRS: usize = 2000;
            if self.imu_pairs.len() > MAX_IMU_PAIRS {
                mapper_trace!(
                    "imu velocity refresh skipped: {} pairs",
                    self.imu_pairs.len()
                );
                return;
            }
            let pairs = self
                .imu_pairs
                .iter()
                .filter_map(|(from, to, delta)| {
                    Some((
                        *from,
                        *to,
                        self.mapper.frame_poses.get(from)?.clone(),
                        self.mapper.frame_poses.get(to)?.clone(),
                        delta.dt,
                        delta.position,
                        delta.velocity,
                    ))
                })
                .collect::<Vec<_>>();
            self.mapper.frame_velocities = crate::mapper::imu_factor::refine_pair_velocities(
                &pairs,
                &self.mapper.frame_velocities,
            );
            self.mapper.factors.imu_relative_pose = self
                .imu_pairs
                .iter()
                .filter_map(|(from, to, delta)| {
                    self.mapper.frame_poses.get(to)?;
                    crate::mapper::imu_factor::relative_pose_factor_from_delta(
                        *from,
                        *to,
                        self.mapper.frame_poses.get(from)?.clone(),
                        *self.mapper.frame_velocities.get(from)?,
                        delta,
                        self.config.imu_preintegration_weight,
                    )
                })
                .collect();
        }
    }

    /// Build the joint solver's transient factor lists from `imu_ba_pairs`,
    /// or two empty `Vec`s if joint VI-BA is disabled, calibration is
    /// missing, or the pair count exceeds the same bound
    /// `refresh_imu_factors` uses for the frozen-velocity path (bounding the
    /// per-pass cost of a very long uninterrupted sequence). Callers use an
    /// empty result as the signal to keep calling the unmodified pose-only
    /// `NfrMapper::optimize`/`optimize_with_extra_factors`, so this method
    /// alone gates every bit of extra joint-solver cost.
    fn build_joint_imu_factors(
        &self,
    ) -> (
        Vec<crate::mapper::imu_ba::ImuPreintegratedFactor>,
        Vec<crate::mapper::imu_ba::BiasRandomWalkFactor>,
    ) {
        if self.config.joint_vi_ba_weight <= 0.0 || self.imu_ba_pairs.is_empty() {
            return (Vec::new(), Vec::new());
        }
        const MAX_IMU_BA_PAIRS: usize = 2000;
        if self.imu_ba_pairs.len() > MAX_IMU_BA_PAIRS {
            mapper_trace!(
                "joint VI-BA factor build skipped: {} pairs",
                self.imu_ba_pairs.len()
            );
            return (Vec::new(), Vec::new());
        }
        let Some(calibration) = self.mapper.calibration.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        let bias_noise =
            crate::mapper::imu_ba::bias_random_walk_noise_from_calibration(calibration);
        let weight = self.config.joint_vi_ba_weight;
        let mut imu_factors = Vec::with_capacity(self.imu_ba_pairs.len());
        let mut bias_factors = Vec::with_capacity(self.imu_ba_pairs.len());
        for (from, to, delta) in &self.imu_ba_pairs {
            imu_factors.push(crate::mapper::imu_ba::ImuPreintegratedFactor {
                from: *from,
                to: *to,
                delta: delta.clone(),
                weight,
            });
            bias_factors.push(crate::mapper::imu_ba::BiasRandomWalkFactor {
                from: *from,
                to: *to,
                dt: delta.delta_time,
                noise: bias_noise,
                weight,
            });
        }
        (imu_factors, bias_factors)
    }

    fn optimize_pass(
        &mut self,
        num_opt_iter: usize,
    ) -> Result<
        (
            NfrMapperOptimizeReport,
            NfrMapperFilterReport,
            NfrMapperOptimizeReport,
        ),
        OnlineMapperError,
    > {
        let _ = self.mapper.build_tracks();
        let _ = self
            .mapper
            .setup_opt()
            .map_err(|_| OnlineMapperError::MissingCalibration)?;
        // Re-derive loop factors from the freshly rebuilt map, so they are
        // consistent with the current landmark positions rather than stale
        // snapshots from when the loop was first matched.
        let loop_factors = self.rebuild_loop_factors();
        self.refresh_imu_factors();
        let (imu_factors, bias_factors) = self.build_joint_imu_factors();
        let use_joint = !imu_factors.is_empty() || !bias_factors.is_empty();
        let gravity_world = crate::mapper::imu_factor::gravity_world();
        let first_optimize = if use_joint {
            self.mapper.optimize_joint_with_extra_factors(
                &loop_factors,
                &imu_factors,
                &bias_factors,
                gravity_world,
                num_opt_iter,
            )
        } else {
            self.mapper
                .optimize_with_extra_factors(&loop_factors, num_opt_iter)
        }
        .map_err(|_| OnlineMapperError::MissingCalibration)?;
        let filter = self
            .mapper
            .filter_outliers(
                self.config.headless.outlier_threshold,
                self.config.headless.min_num_obs,
            )
            .map_err(|_| OnlineMapperError::MissingCalibration)?;
        let second_optimize = if use_joint {
            self.mapper.optimize_joint_with_extra_factors(
                &loop_factors,
                &imu_factors,
                &bias_factors,
                gravity_world,
                num_opt_iter,
            )
        } else {
            self.mapper
                .optimize_with_extra_factors(&loop_factors, num_opt_iter)
        }
        .map_err(|_| OnlineMapperError::MissingCalibration)?;
        if self.config.local_ba_window > 0 {
            let local = self
                .mapper
                .optimize_local_window(self.config.local_ba_window, self.config.local_ba_iterations)
                .map_err(|_| OnlineMapperError::MissingCalibration)?;
            mapper_trace!(
                "local_ba: window={} poses={} landmarks={} cost {:.3}->{:.3}",
                self.config.local_ba_window,
                local.pose_count,
                local.landmark_count,
                local.initial_cost,
                local.final_cost
            );
        }
        Ok((first_optimize, filter, second_optimize))
    }

    /// Run a final, full-budget `optimize_pass` (unconditionally, regardless
    /// of the periodic trigger's cadence) and return the propagated
    /// trajectory. Call this once the packet channel/source is exhausted.
    ///
    /// Joins any in-flight background optimize job first (rule 3) so the
    /// final pass warm-starts from the freshest `GlobalBaOptimizerState`
    /// and does not race a background merge landing mid-solve; this is the
    /// one place `OnlineNfrMapper` blocks on the background thread, and it
    /// only does so once, at the natural end of the run.
    pub fn finalize(&mut self) -> Result<OnlineFinalReport, OnlineMapperError> {
        mapper_trace!(
            "finalize: enter, pending_optimizer={}",
            self.pending_optimizer.is_some()
        );
        if let Some(handle) = self.pending_optimizer.take() {
            mapper_trace!("finalize: joining in-flight background optimize job");
            let result = handle.join().expect("background optimize thread panicked");
            mapper_trace!("finalize: background optimize job joined");
            self.merge_background_result(result);
        }
        mapper_trace!("finalize: optimize_pass start");
        let (first_optimize, filter, second_optimize) =
            self.optimize_pass(self.config.headless.num_opt_iter)?;
        mapper_trace!("finalize: optimize_pass done");
        self.total_optimize_passes += 1;
        Ok(OnlineFinalReport {
            first_optimize,
            filter,
            second_optimize,
            result: self.mapper.result(),
            trajectory_tum: self.mapper.trajectory_tum(),
        })
    }

    /// Non-blocking: if the in-flight background optimize job (if any) has
    /// finished, join and merge it, returning its breakdown. Called at the
    /// top of every [`Self::ingest_packet`] so a completed job is merged
    /// promptly without ever making ingestion wait for it.
    fn poll_pending_optimizer(&mut self) -> Option<BackgroundOptimizeBreakdown> {
        let finished = self
            .pending_optimizer
            .as_ref()
            .is_some_and(JoinHandle::is_finished);
        if !finished {
            return None;
        }
        let handle = self.pending_optimizer.take().expect("checked Some above");
        let result = handle.join().expect("background optimize thread panicked");
        let breakdown = result.breakdown;
        self.merge_background_result(result);
        Some(breakdown)
    }

    /// Apply a finished background job's result to the live mapper state.
    /// Only poses present in the snapshot are overwritten -- a keyframe
    /// detected *after* the snapshot was taken keeps its live incremental
    /// pose estimate rather than being touched by a solve that never saw
    /// it. `optimizer_state` (lambda/lambda_vee) is always taken from the
    /// background result so the next job's warm start continues from it.
    fn merge_background_result(&mut self, result: BackgroundOptimizeResult) {
        for (frame_id, pose) in result.poses {
            self.mapper.frame_poses.insert(frame_id, pose);
        }
        // Joint-VI-BA-refined state (a no-op merge when the joint solver did
        // not run this pass: `optimizer_snapshot` seeds these three maps
        // from the same live values, so an unrun joint pass merges each
        // entry back onto itself).
        for (frame_id, velocity) in result.velocities {
            self.mapper.frame_velocities.insert(frame_id, velocity);
        }
        for (frame_id, bias) in result.gyro_biases {
            self.mapper.frame_gyro_bias.insert(frame_id, bias);
        }
        for (frame_id, bias) in result.accel_biases {
            self.mapper.frame_accel_bias.insert(frame_id, bias);
        }
        // Surface the persistent track graph and landmark database on the live
        // mapper.  The snapshot was cloned from the live mapper and then
        // advanced by `build_tracks`/`setup_opt`/`filter_outliers`, so its
        // landmark state is the freshest available; keyframes detected after
        // the snapshot was taken simply have no landmarks yet and are picked
        // up by the next `build_tracks`.  This is what makes the map
        // persistent between optimizations instead of empty until `finalize`.
        if !result.feature_tracks.is_empty() {
            self.mapper.feature_tracks = result.feature_tracks;
        }
        if !result.lmdb.landmarks.is_empty() {
            self.mapper.lmdb = result.lmdb;
        }
        self.mapper.optimizer_state = result.optimizer_state;
        self.last_optimize_duration = Duration::from_secs_f64(result.breakdown.total_seconds);
        self.total_optimize_passes += 1;
    }

    /// Rule 3: clone the current mapper state and run one periodic
    /// `optimize_pass`-equivalent sequence on the clone, on a dedicated
    /// thread, so ingestion (`detect`/`stereo`/`match_new_keyframe`, the
    /// cheap per-packet work) is never blocked by it. The clone is a
    /// snapshot: it does not observe any packet ingested after this call
    /// returns, and the live mapper is not touched until
    /// [`Self::poll_pending_optimizer`] or [`Self::finalize`] merges the
    /// result back.
    fn spawn_background_optimize(&mut self) {
        self.refresh_imu_factors();
        let (imu_factors, bias_factors) = self.build_joint_imu_factors();
        let use_joint = !imu_factors.is_empty() || !bias_factors.is_empty();
        let gravity_world = crate::mapper::imu_factor::gravity_world();
        let mut snapshot = self.mapper.optimizer_snapshot();
        let periodic_iterations = self.config.periodic_iterations;
        let outlier_threshold = self.config.headless.outlier_threshold;
        let min_num_obs = self.config.headless.min_num_obs;
        self.last_optimize_started_at = Some(Instant::now());
        self.pending_optimizer = Some(std::thread::spawn(move || {
            // Best-effort below-normal OS priority (see `crate::rt_priority`)
            // for this dedicated per-pass thread, same rationale as
            // `run_mapper_thread`'s: this pass must not win scheduling
            // contention against the real-time frontend/estimator threads.
            crate::rt_priority::set_current_thread_priority_below_normal_best_effort();
            // Run the whole periodic sequence inside a small, dedicated rayon
            // pool instead of the process-wide one. This pass's
            // `linearize_vision`/landmark-reduction `par_iter` work would
            // otherwise contend with the live VIO thread's own `--threads`
            // pool for cores; capping this background thread's own
            // parallelism keeps it from taking them, matching the joint VI-BA
            // background pass's fix for the same failure mode (uncapped, a
            // periodic background pass regressed VIO wall time 25-45% in
            // real runs -- see `docs/vi_slam_global_consistency_plan.md`
            // section 1.8 and the joint VI-BA background-thread commit
            // message). `1` leaves every other core to the real-time threads.
            const BACKGROUND_OPTIMIZE_THREADS: usize = 1;
            let run_pass = move || {
                let total_start = Instant::now();
                let start = Instant::now();
                let _ = snapshot.build_tracks();
                let build_tracks_seconds = start.elapsed().as_secs_f64();

                let start = Instant::now();
                // Calibration is always present on an online mapper (checked in
                // `OnlineNfrMapper::new`'s callers); a missing-calibration error
                // here would mean the snapshot itself is malformed, which
                // `ingest_packet` already could not have produced.
                let _ = snapshot.setup_opt();
                let setup_opt_seconds = start.elapsed().as_secs_f64();

                let start = Instant::now();
                if use_joint {
                    let _ = snapshot.optimize_joint_with_extra_factors(
                        &[],
                        &imu_factors,
                        &bias_factors,
                        gravity_world,
                        periodic_iterations,
                    );
                } else {
                    let _ = snapshot.optimize(periodic_iterations);
                }
                let optimize1_seconds = start.elapsed().as_secs_f64();

                let start = Instant::now();
                let _ = snapshot.filter_outliers(outlier_threshold, min_num_obs);
                let filter_seconds = start.elapsed().as_secs_f64();

                let start = Instant::now();
                if use_joint {
                    let _ = snapshot.optimize_joint_with_extra_factors(
                        &[],
                        &imu_factors,
                        &bias_factors,
                        gravity_world,
                        periodic_iterations,
                    );
                } else {
                    let _ = snapshot.optimize(periodic_iterations);
                }
                let optimize2_seconds = start.elapsed().as_secs_f64();

                BackgroundOptimizeResult {
                    poses: snapshot.frame_poses,
                    optimizer_state: snapshot.optimizer_state,
                    feature_tracks: snapshot.feature_tracks,
                    lmdb: snapshot.lmdb,
                    velocities: snapshot.frame_velocities,
                    gyro_biases: snapshot.frame_gyro_bias,
                    accel_biases: snapshot.frame_accel_bias,
                    breakdown: BackgroundOptimizeBreakdown {
                        build_tracks_seconds,
                        setup_opt_seconds,
                        optimize1_seconds,
                        filter_seconds,
                        optimize2_seconds,
                        total_seconds: total_start.elapsed().as_secs_f64(),
                    },
                }
            };
            if let Ok(pool) = rayon::ThreadPoolBuilder::new()
                .num_threads(BACKGROUND_OPTIMIZE_THREADS)
                .build()
            {
                return pool.install(run_pass);
            }
            run_pass()
        }));
    }
}

/// Reason a packet source stopped feeding [`run_mapper_thread`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapperThreadStopReason {
    /// The sender was dropped (the VIO thread finished the sequence).
    ChannelClosed,
}

/// Drive an [`OnlineNfrMapper`] from an `mpsc` channel of in-memory
/// `MargData` packets until the sender side closes, then run
/// [`OnlineNfrMapper::finalize`]. This is the mapper-thread half of the
/// two-thread design in `docs/basalt_online_mapper_design.md` Sec3.1: the
/// VIO thread owns the `Sender`, moves each mapper packet into it (no JSON/
/// base64 round trip -- `EstimatorOutput::marg_data` is already owned), and
/// drops the sender when the sequence ends; this function is meant to run on
/// a dedicated `std::thread` so the VIO thread is never blocked by it.
///
/// `on_packet` is called with each packet's [`OnlineIngestReport`] as it
/// completes (for live queue-lag/RTF instrumentation in the caller); it must
/// not block, since it runs on the mapper thread between packets.
///
/// Sets this thread's OS priority to below-normal (best-effort, see
/// `crate::rt_priority`) for the whole lifetime of the function -- so the
/// mapper's periodic background optimize pass (already capped to one rayon
/// thread, see `spawn_background_optimize`) does not also win scheduling
/// contention against the real-time frontend/estimator threads on a busy
/// machine. The caller's own `finalize()` call (made after joining this
/// thread, per its own doc comment) runs on whichever thread calls it and is
/// unaffected by this. Purely a scheduler hint: it cannot change any
/// packet's ingested result.
pub fn run_mapper_thread(
    mut mapper: OnlineNfrMapper,
    receiver: Receiver<MargData>,
    seed: Option<u32>,
    mut on_packet: impl FnMut(&OnlineIngestReport),
) -> (
    OnlineNfrMapper,
    MapperThreadStopReason,
    Vec<OnlineMapperError>,
) {
    crate::rt_priority::set_current_thread_priority_below_normal_best_effort();
    let mut errors = Vec::new();
    loop {
        match receiver.recv_timeout(Duration::from_secs(3600)) {
            Ok(mut packet) => match mapper.ingest_packet(&mut packet, seed) {
                Ok(report) => on_packet(&report),
                Err(error) => errors.push(error),
            },
            Err(RecvTimeoutError::Disconnected) => {
                return (mapper, MapperThreadStopReason::ChannelClosed, errors)
            }
            Err(RecvTimeoutError::Timeout) => {
                // No packet in an hour: treat like a closed channel rather
                // than spinning forever. A live VIO stream at EuRoC's ~20 Hz
                // camera rate never approaches this.
                return (mapper, MapperThreadStopReason::ChannelClosed, errors);
            }
        }
    }
}

/// Producer-side filter that drops raw images already sent to the mapper.
///
/// Each MargData packet carries the whole AOM window's images, so
/// consecutive packets repeat almost every image. The mapper ignores a
/// repeated image: a processed key is skipped, and an image still waiting for
/// its pose stays in `img_data` until it becomes eligible. Queued packets,
/// however, hold their full pixel buffers until the mapper reads them. When
/// the mapper lags, the repeats dominate memory. Stripping them before
/// sending leaves the mapper's inputs unchanged and bounds queued pixels to
/// about one copy per image.
#[derive(Debug, Default)]
pub struct SentImageFilter {
    sent: HashSet<(u64, i64, u16)>,
}

impl SentImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove images whose `(frame_id, timestamp_ns, camera_id)` was already
    /// forwarded, keeping the canonical order of the rest.
    pub fn strip_sent(&mut self, data: &mut MargData) {
        data.of_images.retain(|image| {
            self.sent
                .insert((image.frame_id, image.timestamp_ns, image.camera_id))
        });
    }
}

/// Producer handle used by the VIO thread.
///
/// Bounded (`SyncSender`), not a plain `Sender`. An earlier revision used an
/// unbounded channel on the theory that a queued `MargData` is cheap to hold
/// -- it is consumed (features extracted, raw pixels dropped) in the same
/// `ingest_packet` call that receives it, so *processed* queue growth only
/// costs keypoints/descriptors, not images (Sec1.2 of the design doc). That
/// reasoning misses the *unprocessed* case: a packet sitting in the channel
/// before `ingest_packet` ever sees it still carries its full `of_images`
/// raw pixel buffers, and `--pipeline`'s frontend/estimator overlap (PR
/// #153) makes the VIO producer meaningfully faster than the serial path --
/// fast enough to run many hundreds of packets ahead of a mapper that is
/// still working through an earlier, larger match graph. That gap has no
/// ceiling with an unbounded channel: multi-GB backlog growth was observed
/// on a full-MH_01 `--pipeline` run whose mapper thread also happened to hit
/// a slow packet (production `seed: None` -- see `OnlineIngestReport`'s
/// docs -- makes per-packet cost timing-dependent, and `--pipeline`'s higher
/// throughput changes that timing relative to the serial path), and the
/// resulting memory pressure turned a transient slow packet into a
/// multi-hour apparent stall (100% CPU, no forward progress) rather than a
/// bounded delay. A bounded channel keeps the "VIO never blocks on a
/// keeping-pace mapper" property from rule (1) above (the earlier "a small
/// bound stalled the VIO thread for minutes" regression was measured at a
/// bound small enough to contend during ordinary operation, not one sized
/// to the sliding-AOM-window packet rate this channel actually sees) while
/// putting a hard ceiling on backlog size, and therefore on backlog memory,
/// so a slow mapper packet degrades into bounded VIO backpressure instead of
/// unbounded queue growth. The online demo still reports queue depth/lag as
/// an honest diagnostic; the bound only caps how bad that diagnostic number
/// can get.
pub type MapperPacketSender = SyncSender<MargData>;

/// Pure per-image feature-extraction step of `NfrMapper::detect_keypoints`'s
/// per-image body (`session.rs`): reads only its own raw pixel buffer plus
/// calibration/config, and mutates no mapper state, so `ingest_packet` can
/// run it concurrently across every newly eligible image in a packet (up to
/// the AOM window's 16) without any image's outcome depending on another's
/// or on processing order.
fn extract_one_image_features(
    image: &OfImageData,
    calibration: &BasaltCalibration,
    feature_config: OfflineMapperConfig,
    local_contrast: Option<LocalContrast>,
    feature_smoothing: Option<GaussianSmoothing>,
    feature_quality_level: Option<f32>,
) -> Result<MapperImageFeatures, OnlineMapperError> {
    let timestamp_ns = image.timestamp_ns;
    let camera_id = image.camera_id;
    let camera = calibration
        .camera(camera_id)
        .ok_or(OnlineMapperError::MissingCamera {
            timestamp_ns,
            camera_id,
        })?;
    let width = usize::try_from(image.width).map_err(|_| OnlineMapperError::Image {
        timestamp_ns,
        camera_id,
        source: ImageError::DimensionOverflow,
    })?;
    let height = usize::try_from(image.height).map_err(|_| OnlineMapperError::Image {
        timestamp_ns,
        camera_id,
        source: ImageError::DimensionOverflow,
    })?;
    let raw = RawU16Image::new(width, height, image.data.clone()).map_err(|source| {
        OnlineMapperError::Image {
            timestamp_ns,
            camera_id,
            source,
        }
    })?;
    let raw = match local_contrast {
        Some(local_contrast) => local_contrast.apply(&raw),
        None => raw,
    };
    let raw = match feature_smoothing {
        Some(feature_smoothing) => feature_smoothing.apply(&raw),
        None => raw,
    };
    extract_mapper_features_with(
        &raw,
        camera,
        feature_config,
        feature_quality_level.unwrap_or(MAPPING_QUALITY_LEVEL),
    )
    .map_err(|source| OnlineMapperError::Extraction {
        timestamp_ns,
        camera_id,
        source,
    })
}

/// One accepted candidate pair's match-graph payload, computed by
/// [`evaluate_match_candidate`] on the shared rayon pool and applied back to
/// `self.mapper` sequentially by [`OnlineNfrMapper::match_new_keyframe`].
struct CandidateMatchOutcome {
    left_id: TimeCamId,
    right_id: TimeCamId,
    t_i_j: SE3,
    raw_matches: Vec<(u64, u64)>,
    inliers: Vec<(u64, u64)>,
}

/// Descriptor-match/RANSAC accept logic for one BoW candidate, factored out
/// of [`OnlineNfrMapper::match_new_keyframe`] so it can run on rayon's
/// shared pool: reads only `feature_corners` (never mutates it, and no
/// candidate's outcome depends on another's), so every candidate for a
/// query can be evaluated concurrently without changing which pairs get
/// accepted or their computed match data. Mirrors the sequential body this
/// replaced exactly -- same gate order, same RANSAC entry point selection,
/// same final `mutual_descriptor_matches` recomputation for the accepted
/// pair's stored raw matches.
fn evaluate_match_candidate(
    feature_corners: &std::collections::BTreeMap<TimeCamId, MapperImageFeatures>,
    query_id: TimeCamId,
    candidate: &BowQueryCandidate,
    config: OfflineMapperConfig,
    match_config: OfflineMapperConfig,
    seed: Option<u32>,
) -> Option<CandidateMatchOutcome> {
    if candidate.image.frame_id == query_id.frame_id
        || candidate.score <= config.frames_to_match_threshold
    {
        return None;
    }
    let other_id = TimeCamId::from(candidate.image);
    let (left_id, right_id) = (query_id, other_id);
    let left = feature_corners.get(&left_id)?;
    let right = feature_corners.get(&right_id)?;

    let stage = match_temporal_stage(left, right, match_config);
    if !stage.raw_match_gate_passed {
        return None;
    }

    let result = match seed {
        Some(seed) => match_temporal_ransac_seeded(left, right, match_config, seed),
        None => match_temporal_ransac(left, right, match_config),
    };
    if !result.accepted || result.refined_inlier_ids.is_empty() {
        return None;
    }

    let raw_matches =
        mutual_descriptor_matches(&left.descriptors, &right.descriptors, match_config)
            .into_iter()
            .map(|match_| (match_.left, match_.right))
            .collect::<Vec<_>>();
    Some(CandidateMatchOutcome {
        left_id,
        right_id,
        t_i_j: temporal_result_se3(&result),
        raw_matches,
        inliers: result.refined_inlier_ids.clone(),
    })
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

/// Verbatim copy of `session.rs`'s private helper of the same name (that
/// file is not modified -- see the module documentation).
fn stereo_frame_id(timestamp_ns: i64, images: &[OfImageData], camera_id: u16) -> Option<u64> {
    images
        .iter()
        .find(|image| image.camera_id == camera_id)
        .or_else(|| images.first())
        .map(|image| image.frame_id)
        .or_else(|| u64::try_from(timestamp_ns).ok())
}

/// Verbatim copy of `session.rs`'s private helper of the same name (that
/// file is not modified -- see the module documentation).
/// `T_cl_cr` of two cameras from their IMU poses `T_w_il`, `T_w_ir` and
/// extrinsics `T_i_cl`, `T_i_cr`.
fn camera_relative_from_imu(t_w_il: &SE3, t_i_cl: &SE3, t_w_ir: &SE3, t_i_cr: &SE3) -> SE3 {
    t_w_il
        .compose(t_i_cl)
        .inverse()
        .compose(&t_w_ir.compose(t_i_cr))
}

/// `T_ir_il` of the IMU poses from a camera-frame `T_cl_cr`:
/// `T_il_ir = T_i_cl * T_cl_cr * T_i_cr^-1`.
fn imu_right_left_from_camera_relative(t_i_cl: &SE3, t_cl_cr: &SE3, t_i_cr: &SE3) -> SE3 {
    t_i_cl.compose(t_cl_cr).compose(&t_i_cr.inverse()).inverse()
}

fn temporal_result_se3(result: &crate::mapper::features::TemporalRansacResult) -> SE3 {
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

/// Skew-symmetric matrix `[v]x` such that `[v]x * w = v × w`.
fn skew_vector(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Least-squares SE(3) registration aligning `source` onto `target`
/// (Umeyama / Horn without scale): returns `T` minimizing
/// `sum || T * source_i - target_i ||^2`.
///
/// Returns `None` for fewer than three points or a degenerate (rank-deficient)
/// cross-covariance.
fn fit_se3(source: &[Vector3<f64>], target: &[Vector3<f64>]) -> Option<SE3> {
    if source.len() != target.len() || source.len() < 3 {
        return None;
    }
    let n = source.len() as f64;
    let mean_source = source.iter().sum::<Vector3<f64>>() / n;
    let mean_target = target.iter().sum::<Vector3<f64>>() / n;
    let mut cov = Matrix3::zeros();
    for (s, t) in source.iter().zip(target) {
        cov += (s - mean_source) * (t - mean_target).transpose();
    }
    let svd = cov.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut d = Matrix3::identity();
    if (v_t.transpose() * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let rotation = v_t.transpose() * d * u.transpose();
    let rotation = UnitQuaternion::from_matrix(&rotation);
    let translation = mean_target - rotation.transform_vector(&mean_source);
    Some(SE3::new(rotation, translation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::DoubleSphereCamera;
    use crate::mapper::MapperConfig;
    use crate::vio::margdata::{
        AomBlockData, FramePoseData, FrameStateData, MarginalizationTargets, MatrixData,
        MARGDATA_SCHEMA_VERSION_V3,
    };
    use crate::BasaltCalibration;
    use nalgebra::UnitQuaternion;
    use serde_json::Value;

    const TEST_SEED: u32 = 12345;

    fn se3(axis: [f64; 3], angle_deg: f64, t: [f64; 3]) -> SE3 {
        SE3::new(
            UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(Vector3::from(axis)),
                angle_deg.to_radians(),
            ),
            Vector3::from(t),
        )
    }

    #[test]
    fn loop_frame_conversions_are_consistent_with_tilted_cameras() {
        // Aria-like: two cameras tilted far from the IMU axes.
        let t_i_cl = se3([0.3, 1.0, 0.2], 75.0, [0.05, 0.01, -0.02]);
        let t_i_cr = se3([-0.4, 0.8, 0.5], -70.0, [-0.05, 0.02, 0.01]);
        // A 40 deg yaw turn plus 3 m of motion between the keyframes.
        let t_w_il = se3([0.0, 0.0, 1.0], 10.0, [1.0, 2.0, 0.5]);
        let t_w_ir = se3([0.1, 0.0, 1.0], 50.0, [3.5, 3.0, 0.4]);

        let t_cl_cr = camera_relative_from_imu(&t_w_il, &t_i_cl, &t_w_ir, &t_i_cr);
        // A correct loop measurement agrees exactly with the map in camera frames.
        let delta = t_cl_cr.rotation.inverse() * t_cl_cr.rotation;
        assert!(delta.angle().to_degrees() < 1e-9);
        // The IMU-frame relative rotation is not the camera-frame one: the
        // frame mix-up this guards against reports a large error here.
        let imu_relative = t_w_il.inverse().compose(&t_w_ir);
        let mixed = (imu_relative.rotation.inverse() * t_cl_cr.rotation)
            .angle()
            .to_degrees();
        assert!(mixed > 10.0, "mixed-frame error {mixed}");

        // Converting the camera measurement back gives the IMU poses' T_ir_il.
        let t_ir_il = imu_right_left_from_camera_relative(&t_i_cl, &t_cl_cr, &t_i_cr);
        let expected = t_w_ir.inverse().compose(&t_w_il);
        assert!((t_ir_il.translation - expected.translation).norm() < 1e-9);
        assert!((t_ir_il.rotation.inverse() * expected.rotation).angle() < 1e-9);
    }

    // The exact real MH_01 M8a/M8b marginalization-event fixture the pinned
    // `pipelines/basalt/tests/m8f_nfr_session.rs` gate uses, reused here
    // unmodified so this module's tests exercise real AOM/pose data rather
    // than a synthetic identity system. Path is relative to this file
    // (`pipelines/basalt/src/mapper/online.rs`), four levels up to the repo
    // root.
    const FIXTURE: &str =
        include_str!("../../../../benchmarks/basalt/m8a_m8b_mh01_1403636579763555584.json");

    fn array(value: &Value) -> Vec<f64> {
        value
            .as_array()
            .expect("array")
            .iter()
            .map(|item| item.as_f64().expect("finite number"))
            .collect()
    }

    /// Verbatim adaptation of `tests/m8f_nfr_session.rs::fixture_packet`.
    fn fixture_packet() -> MargData {
        let root: Value = serde_json::from_str(FIXTURE).expect("M8a/M8b fixture JSON");
        let input = &root["input"];

        let h_rows = input["abs_H"].as_array().expect("input abs_H");
        let h_cols = h_rows
            .first()
            .and_then(Value::as_array)
            .expect("abs_H row")
            .len();
        let mut h_column_major = Vec::with_capacity(h_cols * h_rows.len());
        for col in 0..h_cols {
            for row in h_rows {
                h_column_major.push(
                    row.as_array()
                        .and_then(|values| values.get(col))
                        .and_then(Value::as_f64)
                        .expect("abs_H entry"),
                );
            }
        }

        let frame_poses = input["frame_poses"]
            .as_array()
            .expect("frame_poses")
            .iter()
            .map(|pose| {
                let q = array(&pose["pose"]["quaternion_xyzw"]);
                let t = array(&pose["pose"]["translation"]);
                FramePoseData {
                    frame_id: pose["id"].as_u64().expect("pose id"),
                    timestamp_ns: pose["t_ns"].as_i64().expect("pose timestamp"),
                    pose: [t[0], t[1], t[2], q[3], q[0], q[1], q[2]],
                    is_keyframe: true,
                }
            })
            .collect::<Vec<_>>();

        let frame_states = input["frame_states"]
            .as_array()
            .expect("frame_states")
            .iter()
            .map(|state| {
                let q = array(&state["pose"]["quaternion_xyzw"]);
                let t = array(&state["pose"]["translation"]);
                let velocity = array(&state["velocity"]);
                let gyro_bias = array(&state["bias_gyro"]);
                let accel_bias = array(&state["bias_accel"]);
                FrameStateData {
                    frame_id: state["id"].as_u64().expect("state id"),
                    timestamp_ns: state["t_ns"].as_i64().expect("state timestamp"),
                    pose: [t[0], t[1], t[2], q[3], q[0], q[1], q[2]],
                    velocity: velocity.try_into().expect("velocity length"),
                    gyro_bias: gyro_bias.try_into().expect("gyro bias length"),
                    accel_bias: accel_bias.try_into().expect("accel bias length"),
                    linearized: state["linearized"].as_bool().expect("linearized"),
                    is_keyframe: input["kfs_all"]
                        .as_array()
                        .expect("kfs_all")
                        .iter()
                        .any(|id| id.as_u64() == Some(state["id"].as_u64().unwrap())),
                    is_latest: false,
                }
            })
            .collect::<Vec<_>>();

        let aom_order = input["aom"]["blocks"]
            .as_array()
            .expect("AOM blocks")
            .iter()
            .map(|block| AomBlockData {
                frame_id: block["id"].as_u64().expect("block id"),
                offset: block["offset"].as_u64().expect("block offset") as usize,
                dof: block["size"].as_u64().expect("block size") as usize,
                kind: if block["size"].as_u64() == Some(6) {
                    "pose".into()
                } else {
                    "state".into()
                },
            })
            .collect::<Vec<_>>();

        MargData {
            schema_version: MARGDATA_SCHEMA_VERSION_V3,
            aom_sqrt_jacobian: MatrixData::new(0, h_cols, Vec::new()).expect("sqrt shape"),
            aom_sqrt_rhs: Vec::new(),
            aom_abs_h: Some(MatrixData::new(h_rows.len(), h_cols, h_column_major).unwrap()),
            aom_abs_b: Some(array(&input["abs_b"])),
            frame_poses,
            frame_states,
            keyframes: input["kfs_all"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap())
                .collect(),
            kf_to_marg: Vec::new(),
            kfs_all: input["kfs_all"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap())
                .collect(),
            kfs_to_marg: input["kfs_to_marg"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap())
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
            used_imu: input["use_imu"].as_bool().expect("use_imu"),
            provenance_version: "basalt-0f3b2b52-online-mapper-test-fixture".into(),
        }
    }

    /// Verbatim adaptation of `tests/m8f_nfr_session.rs::feature_calibration`.
    fn feature_calibration() -> BasaltCalibration {
        let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
            .expect("camera");
        BasaltCalibration {
            t_imu_cam: vec![
                SE3::identity(),
                SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0)),
            ],
            cameras: vec![camera, camera],
            resolutions: vec![(640, 480), (640, 480)],
            calib_accel_bias: vec![0.0; 3],
            calib_gyro_bias: vec![0.0; 3],
            imu_update_rate_hz: 200.0,
            accel_noise_std: Vector3::repeat(1.0),
            gyro_noise_std: Vector3::repeat(1.0),
            accel_bias_std: Vector3::repeat(1.0),
            gyro_bias_std: Vector3::repeat(1.0),
            t_mocap_world: SE3::identity(),
            t_imu_marker: SE3::identity(),
            mocap_time_offset_ns: 0,
            mocap_to_imu_offset_ns: 0,
            cam_time_offset_ns: 0,
        }
    }

    /// Verbatim adaptation of `tests/m8f_nfr_session.rs::feature_image`.
    fn feature_image(frame_id: u64, timestamp_ns: i64, camera_id: u16) -> OfImageData {
        let pixels = (0..128)
            .flat_map(|y| {
                (0..128).map(move |x| {
                    let mut value = (x as u32).wrapping_mul(0x9e37_79b9)
                        ^ (y as u32).wrapping_mul(0x85eb_ca6b)
                        ^ ((x as u32) << 16 | y as u32);
                    value ^= value >> 13;
                    value = value.wrapping_mul(0xc2b2_ae35);
                    (value ^ u32::from(camera_id)).wrapping_shr(16) as u16
                })
            })
            .collect();
        OfImageData::new(frame_id, timestamp_ns, camera_id, 128, 128, pixels).expect("image")
    }

    /// Two eligible stereo frames from the real fixture's own keyframe
    /// list, split across two packets so the equivalence test exercises
    /// more than one `ingest_packet`/`add_marg_data` call -- one keyframe
    /// per packet, matching how the online demo will actually receive
    /// packets from the VIO stream.
    fn two_packets_with_images() -> (MargData, MargData) {
        let base = fixture_packet();
        let first = base.frame_poses.first().expect("fixture pose").clone();
        let second = base
            .frame_poses
            .get(1)
            .expect("second fixture pose")
            .clone();

        let mut packet_one = base.clone();
        packet_one.of_images = vec![
            feature_image(first.frame_id, first.timestamp_ns, 0),
            feature_image(first.frame_id, first.timestamp_ns, 1),
        ];

        let mut packet_two = base;
        packet_two.of_images = vec![
            feature_image(second.frame_id, second.timestamp_ns, 0),
            feature_image(second.frame_id, second.timestamp_ns, 1),
        ];

        (packet_one, packet_two)
    }

    /// Like [`two_packets_with_images`], but `packet_two`'s window
    /// *overlaps* `packet_one`'s -- it carries images for both `first` and
    /// `second`, exactly like a real MargData stream's sliding AOM window
    /// (Phase A: one packet's `of_images` is the whole marginalization-event
    /// window, and consecutive packets' windows share all but the oldest
    /// keyframe). Used to prove the overlap-skip fix below.
    fn overlapping_packets_with_images() -> (MargData, MargData) {
        let base = fixture_packet();
        let first = base.frame_poses.first().expect("fixture pose").clone();
        let second = base
            .frame_poses
            .get(1)
            .expect("second fixture pose")
            .clone();

        let mut packet_one = base.clone();
        packet_one.of_images = vec![
            feature_image(first.frame_id, first.timestamp_ns, 0),
            feature_image(first.frame_id, first.timestamp_ns, 1),
        ];

        let mut packet_two = base;
        packet_two.of_images = vec![
            feature_image(first.frame_id, first.timestamp_ns, 0),
            feature_image(first.frame_id, first.timestamp_ns, 1),
            feature_image(second.frame_id, second.timestamp_ns, 0),
            feature_image(second.frame_id, second.timestamp_ns, 1),
        ];

        (packet_one, packet_two)
    }

    /// A packet whose window overlaps the previous one's must only detect
    /// and match the genuinely new images, not re-process the whole window
    /// every time -- the exact bug an MH_01 smoke run caught (every packet
    /// reporting `new_key_count=16`, the full window, with per-packet match
    /// cost growing monotonically as a result).
    #[test]
    fn imu_ingestion_retains_navigation_states_before_pose_conversion() {
        let mut packet = fixture_packet();
        let states = packet
            .frame_states
            .iter()
            .filter(|state| packet.kfs_all.contains(&state.frame_id))
            .cloned()
            .collect::<Vec<_>>();
        assert!(!states.is_empty());
        let mut mapper = NfrMapper::new(MapperConfig::default());
        let mut repeated_packet = packet.clone();
        assert!(mapper.add_marg_data(&mut packet).unwrap().accepted);
        for state in &states {
            assert_eq!(
                mapper.frame_velocities[&state.frame_id],
                Vector3::from(state.velocity)
            );
            assert!(!packet
                .frame_states
                .iter()
                .any(|s| s.frame_id == state.frame_id));
            assert_eq!(
                mapper.frame_velocity_bias[&state.frame_id],
                (
                    Vector3::from(state.velocity),
                    Vector3::from(state.gyro_bias),
                    Vector3::from(state.accel_bias)
                )
            );
        }
        let refined = Vector3::new(123.0, 456.0, 789.0);
        mapper.frame_velocities.insert(states[0].frame_id, refined);
        mapper.add_marg_data(&mut repeated_packet).unwrap();
        assert_eq!(mapper.frame_velocities[&states[0].frame_id], refined);
        mapper.calibration = Some(feature_calibration());
        // `frame_velocity_bias` (the frozen original VIO estimate) is dead
        // weight in every snapshot and stays empty. `frame_velocities` used
        // to be dropped too, back when only the pose-only frozen-velocity
        // IMU factor existed (it bakes the live velocity into a relative-pose
        // factor *before* the snapshot is taken, so the snapshot's own
        // `optimize()` call never reads it). The joint VI-BA solver
        // (`optimize_joint_with_extra_factors`) optimizes velocity as
        // first-class state instead, so it now needs the live value carried
        // into the snapshot; see `optimizer_snapshot`'s doc comment.
        let snapshot = mapper.optimizer_snapshot();
        assert!(snapshot.frame_velocity_bias.is_empty());
        assert_eq!(snapshot.frame_velocities[&states[0].frame_id], refined);
    }

    #[test]
    fn imu_ingestion_connects_packets_once_and_is_disabled_by_default() {
        let mut base = fixture_packet();
        // Supplement pose-only fixture entries with navigation estimates as if
        // retained from earlier VIO packets, including frame ID zero.
        base.frame_poses.push(FramePoseData {
            frame_id: 0,
            timestamp_ns: 0,
            pose: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            is_keyframe: true,
        });
        base.kfs_all.push(0);
        let mut all_ids = base.kfs_all.clone();
        all_ids.sort_unstable();
        all_ids.dedup();
        let last = *all_ids.last().unwrap();
        let mut next = base.clone();
        next.frame_poses.push(FramePoseData {
            frame_id: last + 1,
            timestamp_ns: last as i64 + 1_000_000_000,
            pose: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            is_keyframe: true,
        });
        next.kfs_all.push(last + 1);
        let samples: Arc<[ImuSample]> = all_ids
            .iter()
            .skip(1)
            .map(|&id| ImuSample::new(id as i64, Vector3::zeros(), Vector3::zeros()))
            .chain(std::iter::once(ImuSample::new(
                last as i64 + 1_000_000_000,
                Vector3::zeros(),
                Vector3::zeros(),
            )))
            .collect::<Vec<_>>()
            .into();
        for weight in [0.0, 7.0] {
            let mut online = OnlineNfrMapper::new(
                MapperConfig::default(),
                feature_calibration(),
                OfflineMapperConfig::default(),
                GlobalBaConfig::default(),
                OnlineMapperConfig {
                    imu_samples: samples.clone(),
                    imu_preintegration_weight: weight,
                    optimize_every_k: usize::MAX,
                    ..OnlineMapperConfig::default()
                },
            );
            if weight > 0.0 {
                // Pose-only entries have already lost their navigation fields;
                // model the history retained from their original VIO packets.
                for &id in &all_ids {
                    online
                        .mapper
                        .frame_velocity_bias
                        .insert(id, (Vector3::zeros(), Vector3::zeros(), Vector3::zeros()));
                }
            }
            online
                .ingest_packet(&mut base.clone(), Some(TEST_SEED))
                .unwrap();
            online
                .ingest_packet(&mut next.clone(), Some(TEST_SEED))
                .unwrap();
            online
                .ingest_packet(&mut next.clone(), Some(TEST_SEED))
                .unwrap();
            assert!(online.mapper.factors.imu_relative_pose.is_empty());
            online.refresh_imu_factors();
            let pairs = online
                .mapper
                .factors
                .imu_relative_pose
                .iter()
                .filter(|f| f.weight == 7.0)
                .map(|f| (f.from, f.to))
                .collect::<Vec<_>>();
            if weight == 0.0 {
                assert!(pairs.is_empty());
                assert!(online.mapper.frame_velocity_bias.is_empty());
                assert!(online.mapper.frame_velocities.is_empty());
                assert!(online.imu_pairs.is_empty());
                assert_eq!(online.last_imu_factor_keyframe, None);
            } else {
                let mut expected = all_ids
                    .windows(2)
                    .map(|ids| (ids[0], ids[1]))
                    .collect::<Vec<_>>();
                expected.push((last, last + 1));
                assert_eq!(pairs, expected);
                let original = online.mapper.factors.imu_relative_pose.clone();
                // A later pass must rebuild old pairs from the latest poses,
                // not append duplicate factors or reuse ingestion-time poses.
                online
                    .mapper
                    .frame_poses
                    .get_mut(&(last + 1))
                    .unwrap()
                    .translation
                    .x += 1.0;
                online.refresh_imu_factors();
                assert_eq!(
                    online.mapper.factors.imu_relative_pose.len(),
                    expected.len()
                );
                assert_ne!(online.mapper.factors.imu_relative_pose, original);
                let snapshot = online.mapper.optimizer_snapshot();
                assert_eq!(
                    snapshot.factors.imu_relative_pose,
                    online.mapper.factors.imu_relative_pose
                );
                // `frame_velocities` is now carried into every snapshot (see
                // `optimizer_snapshot`'s doc comment): the joint VI-BA solver
                // needs it as first-class state, and carrying it is harmless
                // for this pose-only frozen-velocity factor path, which never
                // reads it back off the snapshot.
                assert_eq!(snapshot.frame_velocities, online.mapper.frame_velocities);
            }
        }
    }

    /// Mirrors `imu_ingestion_connects_packets_once_and_is_disabled_by_default`
    /// for the joint VI-BA path: `joint_vi_ba_weight <= 0.0` (the default)
    /// must retain zero `imu_ba_pairs` and build zero joint factors, while a
    /// positive weight builds one `imu_ba_pairs` entry per consecutive
    /// keyframe pair and non-empty, equal-length preintegrated/bias-walk
    /// factor lists from `build_joint_imu_factors`.
    #[test]
    fn joint_vi_ba_ingestion_is_disabled_by_default_and_builds_pairs_when_enabled() {
        let mut base = fixture_packet();
        base.frame_poses.push(FramePoseData {
            frame_id: 0,
            timestamp_ns: 0,
            pose: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            is_keyframe: true,
        });
        base.kfs_all.push(0);
        let mut all_ids = base.kfs_all.clone();
        all_ids.sort_unstable();
        all_ids.dedup();
        let last = *all_ids.last().unwrap();
        let mut next = base.clone();
        next.frame_poses.push(FramePoseData {
            frame_id: last + 1,
            timestamp_ns: last as i64 + 1_000_000_000,
            pose: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            is_keyframe: true,
        });
        next.kfs_all.push(last + 1);
        let samples: Arc<[ImuSample]> = all_ids
            .iter()
            .skip(1)
            .map(|&id| ImuSample::new(id as i64, Vector3::zeros(), Vector3::zeros()))
            .chain(std::iter::once(ImuSample::new(
                last as i64 + 1_000_000_000,
                Vector3::zeros(),
                Vector3::zeros(),
            )))
            .collect::<Vec<_>>()
            .into();
        for weight in [0.0, 3.0] {
            let mut online = OnlineNfrMapper::new(
                MapperConfig::default(),
                feature_calibration(),
                OfflineMapperConfig::default(),
                GlobalBaConfig::default(),
                OnlineMapperConfig {
                    imu_samples: samples.clone(),
                    joint_vi_ba_weight: weight,
                    optimize_every_k: usize::MAX,
                    ..OnlineMapperConfig::default()
                },
            );
            if weight > 0.0 {
                for &id in &all_ids {
                    online
                        .mapper
                        .frame_velocity_bias
                        .insert(id, (Vector3::zeros(), Vector3::zeros(), Vector3::zeros()));
                }
            }
            online
                .ingest_packet(&mut base.clone(), Some(TEST_SEED))
                .unwrap();
            online
                .ingest_packet(&mut next.clone(), Some(TEST_SEED))
                .unwrap();
            online
                .ingest_packet(&mut next.clone(), Some(TEST_SEED))
                .unwrap();
            let (imu_factors, bias_factors) = online.build_joint_imu_factors();
            if weight == 0.0 {
                assert!(online.imu_ba_pairs.is_empty());
                assert_eq!(online.last_imu_ba_factor_keyframe, None);
                assert!(imu_factors.is_empty());
                assert!(bias_factors.is_empty());
                // Disabled joint mode must never touch the frozen-velocity
                // feature's own state.
                assert!(online.imu_pairs.is_empty());
            } else {
                let mut expected = all_ids
                    .windows(2)
                    .map(|ids| (ids[0], ids[1]))
                    .collect::<Vec<_>>();
                expected.push((last, last + 1));
                let pairs = online
                    .imu_ba_pairs
                    .iter()
                    .map(|(from, to, _)| (*from, *to))
                    .collect::<Vec<_>>();
                assert_eq!(pairs, expected);
                assert_eq!(imu_factors.len(), expected.len());
                assert_eq!(bias_factors.len(), expected.len());
                assert!(imu_factors.iter().all(|f| f.weight == 3.0));
                assert!(bias_factors.iter().all(|f| f.weight == 3.0));
                assert!(online.imu_pairs.is_empty());
            }
        }
    }

    /// Direct regression test for a real bug caught while running the joint
    /// solver online: `ingest_packet` only retained navigation states
    /// (`frame_velocity_bias`) when `imu_preintegration_weight > 0.0`, so
    /// `joint_vi_ba_weight`-only mode built zero `imu_ba_pairs` (the ingest
    /// block's `self.mapper.frame_velocity_bias.get(&prev_id)` was always
    /// `None`) despite loading raw IMU samples and appearing to run
    /// normally -- a silent no-op, not a crash, which the manual
    /// `frame_velocity_bias` pre-seed in the test above did not catch since
    /// it bypassed real ingestion. This test does not pre-seed anything: it
    /// only checks that `ingest_packet` itself populates
    /// `mapper.frame_velocity_bias` when only `joint_vi_ba_weight` is set.
    #[test]
    fn joint_vi_ba_only_mode_retains_velocity_bias_at_ingest() {
        let mut packet = fixture_packet();
        let calibration = feature_calibration();
        let mut online = OnlineNfrMapper::new(
            MapperConfig::default(),
            calibration,
            OfflineMapperConfig::default(),
            GlobalBaConfig::default(),
            OnlineMapperConfig {
                imu_samples: Arc::from([]),
                imu_preintegration_weight: 0.0,
                joint_vi_ba_weight: 5.0,
                ..OnlineMapperConfig::default()
            },
        );
        assert!(online.mapper.frame_velocity_bias.is_empty());
        online.ingest_packet(&mut packet, Some(TEST_SEED)).unwrap();
        assert!(
            !online.mapper.frame_velocity_bias.is_empty(),
            "joint_vi_ba_weight-only ingest must retain navigation states, \
             the same way imu_preintegration_weight-only ingest does"
        );
    }

    #[test]
    fn overlapping_packet_window_only_processes_new_images() {
        let (mut packet_one, mut packet_two) = overlapping_packets_with_images();
        let mut online = OnlineNfrMapper::new(
            MapperConfig::default(),
            feature_calibration(),
            OfflineMapperConfig::default(),
            GlobalBaConfig::default(),
            OnlineMapperConfig {
                optimize_every_k: usize::MAX,
                ..OnlineMapperConfig::default()
            },
        );
        let report_one = online
            .ingest_packet(&mut packet_one, Some(TEST_SEED))
            .expect("online ingest 1");
        assert_eq!(
            report_one.new_key_count, 2,
            "packet 1 detects its own 2 images"
        );

        let report_two = online
            .ingest_packet(&mut packet_two, Some(TEST_SEED))
            .expect("online ingest 2");
        assert_eq!(
            report_two.new_key_count, 2,
            "packet 2 must only detect the 2 genuinely new images (second's stereo pair), \
             not re-detect first's images already processed by packet 1"
        );
        assert_eq!(
            online.mapper.feature_corners.len(),
            4,
            "feature_corners must contain exactly 4 keys total, not a duplicate 6"
        );
    }

    fn synthetic_ba_mapper() -> NfrMapper {
        let calibration = feature_calibration();
        let camera = *calibration.camera(0).expect("camera");
        let mut mapper = NfrMapper::with_calibration(MapperConfig::default(), calibration);
        let mut state = 0x2545_f491_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(state >> 8) / f64::from(1u32 << 24) - 0.5
        };
        let points = (0..60)
            .map(|_| nalgebra::Point3::new(2.0 * next(), 1.4 * next(), 4.5 + 3.0 * next()))
            .collect::<Vec<_>>();
        // `min_track_length` defaults to 5, so tracks need six images.
        let frames = 6_u64;
        for frame in 0..frames {
            let pose = SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(0.15 * frame as f64, 0.02 * next(), 0.0),
            );
            let world_to_camera = pose.inverse();
            let corners = points
                .iter()
                .map(|point| {
                    let pixel = camera
                        .project(&world_to_camera.transform_point(point))
                        .expect("visible point");
                    nalgebra::Point2::new(pixel.x + 0.3 * next(), pixel.y + 0.3 * next())
                })
                .collect::<Vec<_>>();
            let count = corners.len();
            mapper.feature_corners.insert(
                TimeCamId::new(frame, 0),
                MapperImageFeatures {
                    corners,
                    corner_angles: vec![0.25; count],
                    descriptors: vec![[7; 32]; count],
                    rays: vec![[0.0, 0.0, 1.0, 0.0]; count],
                    hashes: vec![3; count],
                    bow_vector: Vec::new(),
                },
            );
            mapper.frame_poses.insert(frame, pose);
            mapper.frame_timestamps.insert(frame, 1_000 + frame as i64);
        }
        for step in [1, 2] {
            for frame in 0..frames.saturating_sub(step) {
                let inliers = (0..points.len() as u64)
                    .map(|id| (id, id))
                    .collect::<Vec<_>>();
                mapper.feature_matches.insert(
                    (TimeCamId::new(frame, 0), TimeCamId::new(frame + step, 0)),
                    MatchData::new(inliers),
                );
            }
        }
        mapper
    }

    /// The compact optimizer snapshot must drive the background pipeline to
    /// exactly the same poses, tracks, landmarks and optimizer state as a
    /// full clone of the mapper.  Synthetic state: 60 points seen by six
    /// translated cam0 frames with deterministic pixel noise, chained by
    /// consecutive and skip-one temporal matches.
    #[test]
    fn optimizer_snapshot_matches_full_clone_pipeline() {
        let mapper = synthetic_ba_mapper();
        let headless = OnlineMapperConfig::default().headless;
        let run = |mut mapper: NfrMapper| {
            let _ = mapper.build_tracks();
            mapper.setup_opt().expect("setup_opt");
            let _ = mapper.optimize(4);
            let _ = mapper.filter_outliers(headless.outlier_threshold, headless.min_num_obs);
            let _ = mapper.optimize(4);
            mapper
        };
        let full = run(mapper.clone());
        let compact = run(mapper.optimizer_snapshot());
        assert!(
            full.lmdb.landmarks.len() > 30,
            "fixture must triangulate landmarks, got {}",
            full.lmdb.landmarks.len()
        );
        assert_ne!(
            full.frame_poses, mapper.frame_poses,
            "BA must move the poses"
        );
        assert_eq!(full.frame_poses, compact.frame_poses);
        assert_eq!(full.feature_tracks, compact.feature_tracks);
        assert_eq!(full.lmdb, compact.lmdb);
        assert_eq!(full.optimizer_state, compact.optimizer_state);
        assert!(compact
            .feature_corners
            .values()
            .all(|f| f.descriptors.is_empty()));
    }

    /// Batching the observation-index rebuild in `filter_outliers` must give
    /// exactly the landmark database of the previous per-removal rebuild,
    /// including both whole-landmark and single-observation removals.
    #[test]
    fn filter_outliers_batched_rebuild_matches_per_removal_reference() {
        let mut mapper = synthetic_ba_mapper();
        // Corrupt a few pixels so some observations become outliers: points
        // 0..4 in one frame (observation removal) and points 10..14 in
        // three frames (landmark removal once too few observations remain).
        for (frame, ids) in [(3_u64, 0..4), (1, 10..14), (2, 10..14), (4, 10..14)] {
            let features = mapper
                .feature_corners
                .get_mut(&TimeCamId::new(frame, 0))
                .expect("frame features");
            for id in ids {
                features.corners[id].x += 40.0;
                features.corners[id].y -= 25.0;
            }
        }
        let _ = mapper.build_tracks();
        mapper.setup_opt().expect("setup_opt");
        let _ = mapper.optimize(4);
        let headless = OnlineMapperConfig::default().headless;
        let mut reference = mapper.clone();
        reference.filter_outliers_per_removal_reference(
            headless.outlier_threshold,
            headless.min_num_obs,
        );
        let report = mapper
            .filter_outliers(headless.outlier_threshold, headless.min_num_obs)
            .expect("filter");
        assert!(
            report.removed_landmark_count > 0 && report.removed_observation_count > 0,
            "fixture must exercise both removal kinds: {report:?}"
        );
        assert_eq!(mapper.lmdb, reference.lmdb);
    }

    /// Stripping images that an earlier packet already carried must leave
    /// the mapper's detection, stereo, and temporal-match state unchanged.
    #[test]
    fn sent_image_filter_preserves_online_mapper_state() {
        let new_online = || {
            OnlineNfrMapper::new(
                MapperConfig::default(),
                feature_calibration(),
                OfflineMapperConfig::default(),
                GlobalBaConfig::default(),
                OnlineMapperConfig {
                    optimize_every_k: usize::MAX,
                    ..OnlineMapperConfig::default()
                },
            )
        };
        let (packet_one, packet_two) = overlapping_packets_with_images();

        let mut full = new_online();
        for packet in [packet_one.clone(), packet_two.clone()] {
            let mut packet = packet;
            full.ingest_packet(&mut packet, Some(TEST_SEED))
                .expect("full ingest");
        }

        let mut filter = SentImageFilter::new();
        let mut stripped = new_online();
        let mut sizes = Vec::new();
        for packet in [packet_one, packet_two] {
            let mut packet = packet;
            filter.strip_sent(&mut packet);
            sizes.push(packet.of_images.len());
            stripped
                .ingest_packet(&mut packet, Some(TEST_SEED))
                .expect("stripped ingest");
        }

        assert_eq!(sizes, vec![2, 2], "packet 2 keeps only its 2 new images");
        assert_eq!(full.mapper.feature_corners, stripped.mapper.feature_corners);
        assert_eq!(full.mapper.feature_matches, stripped.mapper.feature_matches);
        assert_eq!(
            full.mapper.feature_match_data,
            stripped.mapper.feature_match_data
        );
        assert_eq!(full.mapper.frame_poses, stripped.mapper.frame_poses);
        assert_eq!(full.retained_image_bytes(), stripped.retained_image_bytes());
    }

    /// Guardrail (1): the incremental matcher must accept exactly the same
    /// pair set as a batch `match_all` call over the same packets, with the
    /// same seed. This is the same real fixture and the same
    /// two-frame/four-key shape `tests/m8f_nfr_session.rs::
    /// pinned_mh01_packet_match_all_preserves_bow_order_and_payload`
    /// already proves produces stereo *and* accepted temporal pairs (not a
    /// vacuous all-empty comparison).
    #[test]
    fn incremental_matches_equal_batch() {
        let (packet_one, packet_two) = two_packets_with_images();
        let calibration = feature_calibration();

        // Batch reference: ingest both packets, then detect/stereo/match_all
        // once over the fully accumulated state, exactly like the offline
        // demo (`run_headless`'s frontend order).
        let mut batch = NfrMapper::with_calibration(MapperConfig::default(), calibration.clone());
        let mut p1 = packet_one.clone();
        let mut p2 = packet_two.clone();
        batch.add_marg_data(&mut p1).expect("batch ingest 1");
        batch.add_marg_data(&mut p2).expect("batch ingest 2");
        batch.detect_keypoints().expect("batch detect");
        batch.match_stereo().expect("batch stereo");
        let batch_report = batch.match_all_seeded(TEST_SEED);
        assert!(
            batch_report.accepted_pair_count > 0,
            "fixture must produce at least one accepted pair for this test to be meaningful"
        );

        // Incremental: one packet at a time through OnlineNfrMapper.
        let mut online = OnlineNfrMapper::new(
            MapperConfig::default(),
            calibration,
            OfflineMapperConfig::default(),
            GlobalBaConfig::default(),
            OnlineMapperConfig {
                // Isolate the match-graph comparison from the periodic BA
                // trigger for this test.
                optimize_every_k: usize::MAX,
                ..OnlineMapperConfig::default()
            },
        );
        let mut p1 = packet_one;
        let mut p2 = packet_two;
        online
            .ingest_packet(&mut p1, Some(TEST_SEED))
            .expect("online ingest 1");
        online
            .ingest_packet(&mut p2, Some(TEST_SEED))
            .expect("online ingest 2");

        assert_eq!(
            batch.feature_matches, online.mapper.feature_matches,
            "incremental match graph must equal the batch match graph"
        );
        assert_eq!(
            batch.feature_corners, online.mapper.feature_corners,
            "incremental detection must produce exactly the batch features (same keys, same pixels, same descriptors), \
             not merely the same key count"
        );
        assert_eq!(
            batch.feature_match_data, online.mapper.feature_match_data,
            "incremental per-pair match data must equal the batch match data"
        );
    }

    /// The inverted `hash_index` lookup `match_new_keyframe` uses must
    /// return exactly the same candidate list `query_bow_candidates` returns
    /// from a full scan of every detected image, for every query in a
    /// multi-packet prefix -- not merely the same *final accepted pairs*
    /// (`incremental_matches_equal_batch` above already covers that; this
    /// test isolates the BoW candidate-selection step itself, per the
    /// specific guardrail that the index change must not silently narrow or
    /// reorder candidates before RANSAC ever runs).
    #[test]
    fn inverted_index_candidates_equal_full_scan() {
        let (mut packet_one, mut packet_two) = two_packets_with_images();
        let mut online = OnlineNfrMapper::new(
            MapperConfig::default(),
            feature_calibration(),
            OfflineMapperConfig::default(),
            GlobalBaConfig::default(),
            OnlineMapperConfig {
                optimize_every_k: usize::MAX,
                ..OnlineMapperConfig::default()
            },
        );
        online
            .ingest_packet(&mut packet_one, Some(TEST_SEED))
            .expect("online ingest 1");
        online
            .ingest_packet(&mut packet_two, Some(TEST_SEED))
            .expect("online ingest 2");

        let match_window = online.mapper.feature_config.match_window as usize;
        let keys = online
            .mapper
            .feature_corners
            .keys()
            .copied()
            .collect::<Vec<_>>();
        assert!(
            keys.len() >= 4,
            "fixture must produce more than one frame's worth of keys"
        );
        let mut checked_nonempty = false;
        for query_id in keys {
            let query_features = online
                .mapper
                .feature_corners
                .get(&query_id)
                .expect("key present")
                .clone();

            let via_index =
                online.bow_candidates_via_index(query_id, &query_features, match_window);

            let full_database = online
                .mapper
                .feature_corners
                .iter()
                .map(|(&id, features)| (MapperImageId::from(id), features))
                .collect::<Vec<_>>();
            let via_full_scan = query_bow_candidates(
                MapperImageId::from(query_id),
                &query_features,
                &full_database,
                match_window,
            );

            assert_eq!(
                via_index, via_full_scan,
                "inverted-index candidates must equal a full-scan call for query {query_id:?}"
            );
            checked_nonempty |= !via_full_scan.is_empty();
        }
        assert!(
            checked_nonempty,
            "fixture must produce at least one non-empty candidate list for this test to be meaningful"
        );
    }

    /// Guardrail (2): raw image bytes must not accumulate across packets.
    #[test]
    fn retained_image_bytes_stay_bounded() {
        let (mut packet_one, mut packet_two) = two_packets_with_images();
        let mut online = OnlineNfrMapper::new(
            MapperConfig::default(),
            feature_calibration(),
            OfflineMapperConfig::default(),
            GlobalBaConfig::default(),
            OnlineMapperConfig::default(),
        );
        for packet in [&mut packet_one, &mut packet_two] {
            let report = online
                .ingest_packet(packet, Some(TEST_SEED))
                .expect("online ingest");
            assert_eq!(
                report.skipped_ineligible_timestamp_count, 0,
                "fixture packets are expected to be immediately pose-eligible"
            );
            assert_eq!(
                report.retained_image_bytes, 0,
                "raw image bytes must be dropped in the same call that detects them"
            );
            assert_eq!(online.retained_image_bytes(), 0);
        }
    }

    /// The channel-close path exists and runs the tail exactly once even
    /// with zero packets (degenerate but must not panic).
    #[test]
    fn mapper_thread_runs_finalize_on_channel_close() {
        let online = OnlineNfrMapper::new(
            MapperConfig::default(),
            feature_calibration(),
            OfflineMapperConfig::default(),
            GlobalBaConfig::default(),
            OnlineMapperConfig::default(),
        );
        let (sender, receiver) = std::sync::mpsc::sync_channel::<MargData>(1);
        drop(sender);
        let (mut online, reason, errors) =
            run_mapper_thread(online, receiver, Some(TEST_SEED), |_report| {});
        assert_eq!(reason, MapperThreadStopReason::ChannelClosed);
        assert!(errors.is_empty());
        // No packets were ingested, so setup_opt has no tracks and
        // optimize_pass runs on an empty problem -- finalize must still not
        // panic.
        let _ = online.finalize();
    }

    #[test]
    fn fit_se3_recovers_a_known_transform() {
        let expected = SE3::new(
            UnitQuaternion::from_euler_angles(0.2, -0.35, 0.5),
            Vector3::new(0.4, -0.2, 1.3),
        );
        let source: Vec<Vector3<f64>> = vec![
            Vector3::new(1.0, 0.0, 0.5),
            Vector3::new(0.0, 1.0, -0.5),
            Vector3::new(-1.0, 0.5, 0.0),
            Vector3::new(0.3, -0.7, 2.0),
            Vector3::new(2.0, 1.0, -1.0),
        ];
        let target: Vec<Vector3<f64>> = source
            .iter()
            .map(|p| expected.transform_point(&Point3::from(*p)).coords)
            .collect();
        let fitted = fit_se3(&source, &target).expect("fit");
        for (a, b) in fitted.translation.iter().zip(expected.translation.iter()) {
            assert!((a - b).abs() < 1e-9, "{fitted:?} != {expected:?}");
        }
        let r = fitted.rotation.to_rotation_matrix().into_inner()
            * expected
                .rotation
                .to_rotation_matrix()
                .into_inner()
                .transpose();
        assert!((r - Matrix3::identity()).norm() < 1e-9);
    }

    #[test]
    fn fit_se3_rejects_degenerate_input() {
        assert!(fit_se3(&[], &[]).is_none());
        let p = vec![Vector3::zeros(), Vector3::new(1.0, 0.0, 0.0)];
        assert!(fit_se3(&p, &p).is_none());
    }
}
