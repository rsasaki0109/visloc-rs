//! Direct frame-to-frame Basalt-style KLT tracking.
//!
//! This module intentionally does not call the repository's descriptor,
//! generic optical-flow, or PnP paths. The public processing order is kept
//! explicit in [`TrackStage`]:
//!
//! 1. cam0 forward SE(2) IC,
//! 2. cam0 backward SE(2) IC,
//! 3. cam0 FB² rejection at `0.04`,
//! 4. cam0 grid FAST replenishment,
//! 5. new cam0 to cam1 stereo KLT,
//! 6. stereo backward KLT and FB² rejection,
//! 7. Double Sphere bearing essential residual at `0.005`,
//! 8. emit observations.
//!
//! Existing stereo observations are also tracked between frames before
//! replenishment, matching Basalt's frame-to-frame map behavior. Each camera's
//! observation map is tracked independently; cam1 is not dropped merely
//! because the corresponding cam0 ID failed. They have separate trace entries
//! so the new-point sequence above remains visible.

use std::collections::BTreeMap;

use nalgebra::{Matrix2, Matrix3, Point2, Point3, UnitQuaternion, Vector2, Vector3};
use rayon::prelude::*;
use thiserror::Error;

use crate::{
    calibration::BasaltCalibration,
    camera::DoubleSphereCamera,
    fast::{GridFastConfig, GridFastDetector},
    patch::{MeanNormalizedPatch51, PatchResidualError},
    pyramid::{ImageError, RawU16Pyramid},
    timing::{TimingBreakdown, TimingBucket},
    types::{BasaltFrame, FrameId, ImuSample, TrackId, TrackObservation},
    update::{AffineCompact2f, Se2UpdateError},
};

/// Configuration for the direct KLT stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectKltConfig {
    pub pyramid_levels: usize,
    pub max_iterations: usize,
    pub fb_squared_threshold: f32,
    pub essential_residual_threshold: f64,
    pub fast: GridFastConfig,
    /// Opt-in raw-gyro rotation initialization for temporal cam0 KLT.
    pub imu_seed_rotation: bool,
}

impl Default for DirectKltConfig {
    fn default() -> Self {
        Self {
            pyramid_levels: 3,
            max_iterations: 5,
            fb_squared_threshold: 0.04,
            essential_residual_threshold: 0.005,
            fast: GridFastConfig::default(),
            imu_seed_rotation: false,
        }
    }
}

/// How the cam1 search for a newly detected cam0 keypoint is initialized.
///
/// Mirrors newer upstream Basalt's `optical_flow_matching_guess_type`.  The
/// pinned commit only has [`Self::SamePixel`], which is adequate for a
/// fronto-parallel stereo pair (EuRoC) but places the seed hundreds of pixels
/// away from the true match on a divergent rig such as Project Aria's two
/// SLAM cameras (~75 degrees apart).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum StereoMatchingGuess {
    /// Seed the cam1 search at the cam0 pixel coordinates (pinned upstream).
    #[default]
    SamePixel,
    /// Upstream `REPROJ_FIX_DEPTH`: unproject the cam0 keypoint to a unit
    /// bearing, place the point at `depth_m` metres along that ray, move it
    /// into cam1 with the calibrated `T_cam1_cam0`, and project it.  A
    /// keypoint whose seed does not project into cam1 is not stereo-tracked.
    ReprojectFixedDepth { depth_m: f64 },
}

/// Opt-in multi-camera frontend extensions (not in the pinned upstream).
///
/// The default value reproduces the pinned cam0-centric frontend bit for
/// bit: new keypoints are detected in cam0 only and stereo-tracked into cam1
/// from the same pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MultiCameraFlowOptions {
    /// Initial cam1 position of the new-keypoint stereo KLT.
    pub stereo_guess: StereoMatchingGuess,
}

impl MultiCameraFlowOptions {
    /// True when every option has its pinned-upstream value.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// A stereo frame supplied directly to the Basalt KLT stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StereoFrame {
    pub frame_id: FrameId,
    pub timestamp_ns: i64,
    pub cam0: crate::RawU16Image,
    pub cam1: Option<crate::RawU16Image>,
}

impl StereoFrame {
    pub const fn new(
        frame_id: FrameId,
        timestamp_ns: i64,
        cam0: crate::RawU16Image,
        cam1: Option<crate::RawU16Image>,
    ) -> Self {
        Self {
            frame_id,
            timestamp_ns,
            cam0,
            cam1,
        }
    }
}

/// Public stages emitted by one `process_frame` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackStage {
    FrameForwardSe2Ic,
    FrameBackwardSe2Ic,
    FrameFbSquared,
    ExistingStereoForwardSe2Ic,
    ExistingStereoBackwardSe2Ic,
    ExistingStereoFbSquared,
    Cam0GridFastReplenish,
    StereoForwardSe2Ic,
    StereoBackwardSe2Ic,
    StereoFbSquared,
    DsBearingEssentialResidual,
    Emit,
}

/// Fine-grained failure from one direct SE(2) KLT direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KltFailure {
    SourcePatchInvalid,
    TargetNoValidSamples,
    TargetInsufficientOverlap,
    IncrementNonFinite,
    IncrementTooLarge,
    TargetOutOfBounds,
}

/// Reject reason counters are stage-qualified to make lifecycle failures
/// diagnosable without inspecting an image stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RejectReason {
    FrameForward(KltFailure),
    FrameBackward(KltFailure),
    FrameFbSquared,
    ExistingStereoForward(KltFailure),
    ExistingStereoBackward(KltFailure),
    ExistingStereoFbSquared,
    FastNoCandidate,
    StereoForward(KltFailure),
    StereoBackward(KltFailure),
    StereoFbSquared,
    StereoBearingInvalid,
    StereoEssentialResidual,
    /// Opt-in reprojection seed ([`StereoMatchingGuess::ReprojectFixedDepth`])
    /// did not land inside cam1, so no stereo search was started.
    StereoSeedOutOfView,
}

/// Per-frame and cumulative reject counters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RejectReasonCounters {
    counts: BTreeMap<RejectReason, usize>,
}

impl RejectReasonCounters {
    pub fn record(&mut self, reason: RejectReason) {
        *self.counts.entry(reason).or_default() += 1;
    }

    pub fn count(&self, reason: RejectReason) -> usize {
        self.counts.get(&reason).copied().unwrap_or(0)
    }

    pub fn total(&self) -> usize {
        self.counts.values().sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&RejectReason, &usize)> {
        self.counts.iter()
    }

    fn merge_from(&mut self, other: &Self) {
        for (reason, count) in &other.counts {
            *self.counts.entry(*reason).or_default() += count;
        }
    }
}

/// Emitted observations and lifecycle information for one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackFrameOutput {
    pub frame: BasaltFrame,
    pub observations: Vec<TrackObservation>,
    pub created_track_ids: Vec<TrackId>,
    pub retained_track_ids: Vec<TrackId>,
    pub rejected_track_ids: Vec<TrackId>,
    pub reject_counters: RejectReasonCounters,
    pub stage_trace: Vec<TrackStage>,
}

/// Errors that prevent a frame from entering the direct stream.
#[derive(Debug, Error)]
pub enum StreamError {
    #[error("direct KLT configuration is invalid: {0}")]
    InvalidConfig(&'static str),
    #[error("camera 0 calibration is missing")]
    MissingCamera0Calibration,
    #[error("stereo frame supplied cam1 but camera 1 calibration is missing")]
    MissingCamera1Calibration,
    #[error("frame id/timestamp is not newer than the previous frame")]
    NonMonotonicFrame,
    #[error("image/pyramid error: {0}")]
    Image(#[from] ImageError),
}

#[derive(Debug, Clone, Copy)]
struct ActiveTrack {
    /// Upstream keeps one observation map per camera.  They are intentionally
    /// independent: cam1 may survive when the corresponding cam0 point is
    /// rejected (and the reverse is also possible).
    cam0: Option<AffineCompact2f>,
    cam1: Option<AffineCompact2f>,
    created_frame_id: FrameId,
}

#[derive(Debug, Clone)]
struct FramePyramids {
    cam0: RawU16Pyramid,
    cam1: Option<RawU16Pyramid>,
}

/// Direct frame-to-frame KLT tracker with monotonic track IDs.
#[derive(Debug, Clone)]
pub struct DirectKltStream {
    calibration: BasaltCalibration,
    config: DirectKltConfig,
    /// Opt-in multi-camera extensions; the default is the pinned frontend.
    multi_camera: MultiCameraFlowOptions,
    detector: GridFastDetector,
    previous: Option<FramePyramids>,
    previous_frame: Option<BasaltFrame>,
    tracks: BTreeMap<TrackId, ActiveTrack>,
    next_track_id: TrackId,
    cumulative_rejects: RejectReasonCounters,
    // Separate cam0/cam1 scratch buffers (rather than one shared buffer)
    // let the two pyramids build concurrently below -- each subsample pass
    // only ever touches its own camera's buffer.
    pyramid_scratch_cam0: Vec<i32>,
    pyramid_scratch_cam1: Vec<i32>,
}

impl DirectKltStream {
    pub fn new(
        calibration: BasaltCalibration,
        config: DirectKltConfig,
    ) -> Result<Self, StreamError> {
        if calibration.camera(0).is_none() {
            return Err(StreamError::MissingCamera0Calibration);
        }
        if config.max_iterations == 0 {
            return Err(StreamError::InvalidConfig(
                "max_iterations must be positive",
            ));
        }
        if !config.fb_squared_threshold.is_finite() || config.fb_squared_threshold <= 0.0 {
            return Err(StreamError::InvalidConfig(
                "fb_squared_threshold must be finite and positive",
            ));
        }
        if !config.essential_residual_threshold.is_finite()
            || config.essential_residual_threshold <= 0.0
        {
            return Err(StreamError::InvalidConfig(
                "essential_residual_threshold must be finite and positive",
            ));
        }
        Ok(Self {
            detector: GridFastDetector::new(config.fast),
            calibration,
            config,
            multi_camera: MultiCameraFlowOptions::default(),
            previous: None,
            previous_frame: None,
            tracks: BTreeMap::new(),
            next_track_id: 0,
            cumulative_rejects: RejectReasonCounters::default(),
            pyramid_scratch_cam0: Vec::new(),
            pyramid_scratch_cam1: Vec::new(),
        })
    }

    /// Enables the opt-in multi-camera frontend extensions.  Passing
    /// [`MultiCameraFlowOptions::default()`] leaves the stream unchanged.
    pub fn with_multi_camera_options(
        mut self,
        options: MultiCameraFlowOptions,
    ) -> Result<Self, StreamError> {
        if let StereoMatchingGuess::ReprojectFixedDepth { depth_m } = options.stereo_guess {
            if !depth_m.is_finite() || depth_m <= 0.0 {
                return Err(StreamError::InvalidConfig(
                    "stereo matching default depth must be finite and positive",
                ));
            }
        }
        self.multi_camera = options;
        Ok(self)
    }

    /// The multi-camera options this stream runs with.
    pub const fn multi_camera_options(&self) -> &MultiCameraFlowOptions {
        &self.multi_camera
    }

    pub fn active_track_ids(&self) -> Vec<TrackId> {
        self.tracks.keys().copied().collect()
    }

    pub const fn next_track_id(&self) -> TrackId {
        self.next_track_id
    }

    pub const fn cumulative_reject_counters(&self) -> &RejectReasonCounters {
        &self.cumulative_rejects
    }

    /// Processes one stereo frame and emits only direct KLT observations.
    pub fn process_frame(&mut self, frame: StereoFrame) -> Result<TrackFrameOutput, StreamError> {
        let mut timing = TimingBreakdown::from_env();
        self.process_frame_with_timing(frame, &mut timing)
    }

    /// Processes one stereo frame while recording optional disjoint frontend
    /// sub-buckets in the adapter-owned timing collector.
    pub fn process_frame_with_timing(
        &mut self,
        frame: StereoFrame,
        timing: &mut TimingBreakdown,
    ) -> Result<TrackFrameOutput, StreamError> {
        self.process_frame_with_timing_impl(frame, &[], Vector3::zeros(), timing)
    }

    /// Processes a stereo frame with raw IMU samples for optional rotation
    /// seeding. `gyro_bias` is the estimator's most recently available
    /// gyro-bias estimate (IMU frame, rad/s; pass `Vector3::zeros()` when
    /// unavailable, e.g. the pipelined frontend thread, which does not share
    /// live estimator state with the frontend thread) -- it is subtracted
    /// from each raw gyro sample before integration.
    pub fn process_frame_with_timing_imu(
        &mut self,
        frame: StereoFrame,
        imu: &[ImuSample],
        gyro_bias: Vector3<f64>,
        timing: &mut TimingBreakdown,
    ) -> Result<TrackFrameOutput, StreamError> {
        self.process_frame_with_timing_impl(frame, imu, gyro_bias, timing)
    }

    fn process_frame_with_timing_impl(
        &mut self,
        frame: StereoFrame,
        imu: &[ImuSample],
        gyro_bias: Vector3<f64>,
        timing: &mut TimingBreakdown,
    ) -> Result<TrackFrameOutput, StreamError> {
        if let Some(previous_frame) = self.previous_frame {
            if frame.frame_id <= previous_frame.frame_id
                || frame.timestamp_ns <= previous_frame.timestamp_ns
            {
                return Err(StreamError::NonMonotonicFrame);
            }
        }
        if frame.cam1.is_some() && self.calibration.camera(1).is_none() {
            return Err(StreamError::MissingCamera1Calibration);
        }

        let pyramid_started = timing.start();
        // cam0 and cam1 pyramid construction are fully independent (each
        // reads only its own camera's decoded image and writes only its own
        // scratch buffer -- see the separate `pyramid_scratch_cam0` /
        // `pyramid_scratch_cam1` fields), so they run concurrently here.
        // `subsample_binomial5_with_scratch`'s own contract already
        // guarantees a scratch buffer's retained contents cannot affect the
        // computed pixels (every element is overwritten before it is read),
        // so this is bit-identical to building them one after another.
        let pyramid_levels = self.config.pyramid_levels;
        let cam1_image = frame.cam1;
        let (cam0_result, cam1_result) = rayon::join(
            || {
                RawU16Pyramid::from_image_with_scratch(
                    frame.cam0,
                    pyramid_levels,
                    &mut self.pyramid_scratch_cam0,
                )
            },
            || match cam1_image {
                Some(image) => RawU16Pyramid::from_image_with_scratch(
                    image,
                    pyramid_levels,
                    &mut self.pyramid_scratch_cam1,
                )
                .map(Some),
                None => Ok(None),
            },
        );
        let cam0 = cam0_result?;
        let cam1 = cam1_result?;
        let current = FramePyramids { cam0, cam1 };
        timing.finish(TimingBucket::FrontendPyramid, pyramid_started);
        let mut counters = RejectReasonCounters::default();
        let stage_trace = vec![
            TrackStage::FrameForwardSe2Ic,
            TrackStage::FrameBackwardSe2Ic,
            TrackStage::FrameFbSquared,
            TrackStage::ExistingStereoForwardSe2Ic,
            TrackStage::ExistingStereoBackwardSe2Ic,
            TrackStage::ExistingStereoFbSquared,
            TrackStage::Cam0GridFastReplenish,
            TrackStage::StereoForwardSe2Ic,
            TrackStage::StereoBackwardSe2Ic,
            TrackStage::StereoFbSquared,
            TrackStage::DsBearingEssentialResidual,
            TrackStage::Emit,
        ];

        let old_tracks = std::mem::take(&mut self.tracks);
        let mut current_tracks = BTreeMap::new();
        let mut retained_track_ids = Vec::new();
        let mut rejected_track_ids = Vec::new();

        let temporal_started = timing.start();
        if let Some(previous) = &self.previous {
            // Each track's forward/backward KLT and FB^2 gate reads only the
            // (read-only, shared) previous/current pyramids plus that one
            // track's own prior observation -- there is no shared mutable
            // state inside `temporal_track_update`. Computing the per-track
            // results with rayon and then folding them into `counters` /
            // `retained_track_ids` / `rejected_track_ids` / `current_tracks`
            // serially, in the same key order `BTreeMap` iteration already
            // used, reproduces the exact push/record sequence (and therefore
            // the exact `current_tracks` contents and the exact, purely
            // integer-count `counters`) the sequential loop produced: no
            // floating-point reduction crosses a track boundary here, so
            // this is bit-identical for any thread count.
            let entries: Vec<(TrackId, ActiveTrack)> =
                old_tracks.iter().map(|(&id, &track)| (id, track)).collect();
            let theta_cam0 = if self.config.imu_seed_rotation {
                self.calibration.t_imu_cam.first().and_then(|t_imu_cam0| {
                    integrate_gyro_imu_frame(imu, gyro_bias).map(|theta_imu| {
                        // T_imu_cam maps camera vectors into the IMU frame; its
                        // inverse rotation maps raw gyro rotation into cam0.
                        (t_imu_cam0.rotation.inverse() * theta_imu).cast::<f32>()
                    })
                })
            } else {
                None
            };
            let camera0 = self.calibration.camera(0);
            let results: Vec<TemporalTrackResult> = entries
                .par_iter()
                .map(|&(track_id, old_track)| {
                    temporal_track_update(
                        track_id,
                        old_track,
                        previous,
                        &current,
                        &self.config,
                        camera0,
                        theta_cam0,
                    )
                })
                .collect();
            for result in results {
                match result.cam0_classification {
                    Some(Cam0Classification::Retained) => retained_track_ids.push(result.track_id),
                    Some(Cam0Classification::Rejected(reason)) => {
                        counters.record(reason);
                        rejected_track_ids.push(result.track_id);
                    }
                    None => {}
                }
                if let Some(reason) = result.cam1_reject {
                    counters.record(reason);
                }
                // Preserve a shared ID whenever either independent upstream
                // observation map retained it.
                if result.cam0.is_some() || result.cam1.is_some() {
                    current_tracks.insert(
                        result.track_id,
                        ActiveTrack {
                            cam0: result.cam0,
                            cam1: result.cam1,
                            created_frame_id: result.created_frame_id,
                        },
                    );
                }
            }
        }
        timing.finish(TimingBucket::FrontendTemporalKlt, temporal_started);

        let fast_started = timing.start();
        let existing_positions: Vec<_> = current_tracks
            .values()
            .filter_map(|track| track.cam0.map(|cam0| *cam0.translation()))
            .collect();
        let new_positions = self.detector.detect(
            current.cam0.level(0).expect("level zero exists"),
            &existing_positions,
        );
        if new_positions.is_empty() {
            counters.record(RejectReason::FastNoCandidate);
        }
        let mut created_track_ids = Vec::new();
        for position in new_positions {
            let track_id = self.next_track_id;
            self.next_track_id += 1;
            current_tracks.insert(
                track_id,
                ActiveTrack {
                    cam0: Some(AffineCompact2f::new(Matrix2::identity(), position)),
                    cam1: None,
                    created_frame_id: frame.frame_id,
                },
            );
            created_track_ids.push(track_id);
        }
        timing.finish(TimingBucket::FrontendFastReplenish, fast_started);

        // New cam0 points are stereo-tracked only after replenishment, exactly
        // as Basalt's addPoints path does.
        let stereo_started = timing.start();
        if let (Some(current_cam1), Some(_camera1)) = (&current.cam1, self.calibration.camera(1)) {
            for track_id in &created_track_ids {
                let Some(track) = current_tracks.get_mut(track_id) else {
                    continue;
                };
                let Some(cam0_transform) = track.cam0 else {
                    continue;
                };
                // Opt-in reprojection seed.  With the pinned `SamePixel`
                // guess `seed` is `None` and both searches below are the
                // unchanged `track_direction` calls.
                let seed = match self.multi_camera.stereo_guess {
                    StereoMatchingGuess::SamePixel => None,
                    StereoMatchingGuess::ReprojectFixedDepth { depth_m } => {
                        match reprojected_stereo_seed_with_warp(
                            &self.calibration,
                            *cam0_transform.translation(),
                            depth_m,
                        ) {
                            Some(seed) => Some(seed),
                            None => {
                                counters.record(RejectReason::StereoSeedOutOfView);
                                continue;
                            }
                        }
                    }
                };
                let forward = match seed {
                    None => {
                        track_direction(&current.cam0, current_cam1, cam0_transform, &self.config)
                    }
                    // The cam1 patch is sampled through the predicted local
                    // cam0->cam1 warp, so the SE(2) search only has to absorb
                    // the residual rotation and translation.
                    Some((seed, warp)) => track_direction_from_warped_seed(
                        &current.cam0,
                        current_cam1,
                        *cam0_transform.translation(),
                        AffineCompact2f::new(warp, seed),
                        &self.config,
                    ),
                };
                let stereo_transform = match forward {
                    Ok(transform) => transform,
                    Err(failure) => {
                        counters.record(RejectReason::StereoForward(failure));
                        continue;
                    }
                };
                // The backward search of a reprojection-seeded match starts
                // from the cam0 keypoint itself (the inverse guess); the
                // same-pixel path keeps upstream's seed at the cam1 match.
                let backward = match seed {
                    None => {
                        track_direction(current_cam1, &current.cam0, stereo_transform, &self.config)
                    }
                    Some((_, warp)) => track_direction_from_warped_seed(
                        current_cam1,
                        &current.cam0,
                        *stereo_transform.translation(),
                        AffineCompact2f::new(
                            stereo_transform
                                .linear()
                                .try_inverse()
                                .or_else(|| warp.try_inverse())
                                .unwrap_or_else(Matrix2::identity),
                            *cam0_transform.translation(),
                        ),
                        &self.config,
                    ),
                };
                let recovered = match backward {
                    Ok(transform) => transform,
                    Err(failure) => {
                        counters.record(RejectReason::StereoBackward(failure));
                        continue;
                    }
                };
                if (cam0_transform.translation() - recovered.translation()).norm_squared()
                    >= self.config.fb_squared_threshold
                {
                    counters.record(RejectReason::StereoFbSquared);
                } else {
                    track.cam1 = Some(stereo_transform);
                }
            }
        }
        timing.finish(TimingBucket::FrontendNewStereoKlt, stereo_started);

        let essential_started = timing.start();
        if self.calibration.camera(1).is_some() {
            for track in current_tracks.values_mut() {
                let Some(cam1_transform) = track.cam1 else {
                    continue;
                };
                // `filterPoints` only evaluates cam1 observations whose ID is
                // also present in cam0.  An independently tracked cam1 point
                // without a cam0 counterpart remains in the upstream map.
                let Some(cam0_transform) = track.cam0 else {
                    continue;
                };
                let Some(residual) = essential_residual(
                    &self.calibration,
                    *cam0_transform.translation(),
                    *cam1_transform.translation(),
                ) else {
                    counters.record(RejectReason::StereoBearingInvalid);
                    track.cam1 = None;
                    continue;
                };
                if residual > self.config.essential_residual_threshold {
                    counters.record(RejectReason::StereoEssentialResidual);
                    track.cam1 = None;
                }
            }
        }
        timing.finish(TimingBucket::FrontendEssentialFilter, essential_started);

        let output_started = timing.start();
        let mut observations = Vec::new();
        for (track_id, track) in &current_tracks {
            if let Some(cam0) = track.cam0 {
                observations.push(TrackObservation {
                    track_id: *track_id,
                    frame_id: frame.frame_id,
                    timestamp_ns: frame.timestamp_ns,
                    camera_id: 0,
                    pixel: Point2::new(cam0.translation().x as f64, cam0.translation().y as f64),
                });
            }
            if let Some(cam1) = track.cam1 {
                observations.push(TrackObservation {
                    track_id: *track_id,
                    frame_id: frame.frame_id,
                    timestamp_ns: frame.timestamp_ns,
                    camera_id: 1,
                    pixel: Point2::new(cam1.translation().x as f64, cam1.translation().y as f64),
                });
            }
        }
        observations.sort_by_key(|observation| (observation.track_id, observation.camera_id));
        created_track_ids.sort_unstable();
        retained_track_ids.sort_unstable();
        rejected_track_ids.sort_unstable();

        self.cumulative_rejects.merge_from(&counters);
        self.tracks = current_tracks;
        self.previous = Some(current);
        let output = TrackFrameOutput {
            frame: BasaltFrame::new(frame.frame_id, frame.timestamp_ns, 0),
            observations,
            created_track_ids,
            retained_track_ids,
            rejected_track_ids,
            reject_counters: counters,
            stage_trace,
        };
        self.previous_frame = Some(output.frame);
        timing.finish(TimingBucket::FrontendOutput, output_started);
        Ok(output)
    }
}

/// One track's cam0 classification from [`temporal_track_update`], carried
/// out of the parallel worker instead of mutating a shared
/// `RejectReasonCounters` / `Vec<TrackId>` pair so the per-track computation
/// has no shared mutable state.
#[derive(Debug, Clone, Copy)]
enum Cam0Classification {
    Retained,
    Rejected(RejectReason),
}

/// Output of one track's independent temporal KLT update, folded back into
/// the caller's `counters` / `retained_track_ids` / `rejected_track_ids` /
/// `current_tracks` serially in the same order [`DirectKltStream::tracks`]
/// (a `BTreeMap`) already iterates in.
struct TemporalTrackResult {
    track_id: TrackId,
    created_frame_id: FrameId,
    cam0: Option<AffineCompact2f>,
    cam0_classification: Option<Cam0Classification>,
    cam1: Option<AffineCompact2f>,
    cam1_reject: Option<RejectReason>,
}

/// Integrates raw gyro samples in the IMU frame using rectangular integration
/// between consecutive timestamps: sum (gyro[k] - bias) * dt[k, k+1]. Returns
/// None with fewer than two samples. `bias` is the estimator's most recently
/// available gyro-bias estimate (zero when unavailable); subtracting it
/// matters here because this integration has no other bias-observability
/// mechanism the way the estimator's own preintegration does -- an
/// uncorrected EuRoC-scale gyro bias (order 1e-2 rad/s) integrated over one
/// frame interval (order 1/20 s) is a non-negligible fraction of a typical
/// per-frame rotation on the fast/blurred sequences this seed targets.
fn integrate_gyro_imu_frame(imu: &[ImuSample], bias: Vector3<f64>) -> Option<Vector3<f64>> {
    if imu.len() < 2 {
        return None;
    }
    let mut theta = Vector3::<f64>::zeros();
    for pair in imu.windows(2) {
        let dt = (pair[1].timestamp_ns - pair[0].timestamp_ns) as f64 * 1e-9;
        if dt > 0.0 {
            theta += (pair[0].gyro_rad_s - bias) * dt;
        }
    }
    Some(theta)
}

/// Predicts the current pixel from the previous pixel and the camera's own
/// integrated rotation (camera-frame axis-angle in radians).
///
/// Derivation, matching this codebase's own IMU-preintegration convention
/// (`pipelines/basalt/src/vio/estimator.rs`: `predicted.imu_to_world.rotation
/// *= delta.delta_rotation`, i.e. `R(t2) = R(t1) * exp([theta]_x)` for
/// `imu_to_world` R and integrated gyro `theta`, the standard Forster-style
/// preintegration delta applied on the right): a world-fixed point's bearing
/// in the IMU/camera frame is `b(t) = R(t)^T * direction`, so `b(t2) =
/// R(t2)^T R(t1) b(t1) = exp([theta]_x)^T b(t1) = exp(-[theta]_x) b(t1)` --
/// rotate the OLD bearing by the NEGATED integrated rotation vector.
fn rotation_seeded_pixel(
    camera: &DoubleSphereCamera,
    old_pixel: Vector2<f32>,
    theta_cam: Vector3<f32>,
) -> Option<Vector2<f32>> {
    let bearing = camera.unproject_f32(&Point2::new(old_pixel.x, old_pixel.y))?;
    let rotation = UnitQuaternion::from_scaled_axis(-theta_cam);
    let rotated = rotation * bearing;
    let projected = camera.project_f32(&Point3::new(rotated.x, rotated.y, rotated.z))?;
    Some(Vector2::new(projected.x, projected.y))
}

/// Pure, side-effect-free per-track temporal KLT update: forward/backward
/// cam0 tracking plus its FB^2 gate, and independently, forward/backward
/// cam1 tracking plus its FB^2 gate. Reads only `previous`/`current`
/// (shared, read-only) and this one track's own prior observation, so many
/// tracks can run this concurrently with no coordination and no change to
/// any individual track's arithmetic.
fn temporal_track_update(
    track_id: TrackId,
    old_track: ActiveTrack,
    previous: &FramePyramids,
    current: &FramePyramids,
    config: &DirectKltConfig,
    camera0: Option<&DoubleSphereCamera>,
    theta_cam0: Option<Vector3<f32>>,
) -> TemporalTrackResult {
    // `FrameToFrameOpticalFlow::trackPoints` runs once for each camera map.
    // Do not gate the cam1 search on cam0 success.
    let mut cam0_classification = None;
    let cam0 = old_track.cam0.and_then(|old_cam0| {
        // `source_position` anchors the reference patch sampled from the OLD
        // image and must stay at the point's true previous-frame pixel; only
        // the NEW-frame search seed (`initial_transform`'s translation) is
        // replaced with the rotation-predicted position. When
        // `camera0`/`theta_cam0` is `None` (feature off, or fewer than two
        // IMU samples this frame), `predicted_position` falls back to
        // `*old_cam0.translation()`, so `seeded_initial == old_cam0` and this
        // is byte-identical to the unseeded `track_direction(&previous.cam0,
        // &current.cam0, old_cam0, config)` call it replaces.
        let predicted_position = camera0
            .zip(theta_cam0)
            .and_then(|(camera, theta)| {
                rotation_seeded_pixel(camera, *old_cam0.translation(), theta)
            })
            .unwrap_or(*old_cam0.translation());
        let seeded_initial = AffineCompact2f::new(*old_cam0.linear(), predicted_position);
        let frame_transform = match track_direction_from_seed(
            &previous.cam0,
            &current.cam0,
            *old_cam0.translation(),
            seeded_initial,
            config,
        ) {
            Ok(transform) => transform,
            Err(failure) => {
                cam0_classification = Some(Cam0Classification::Rejected(
                    RejectReason::FrameForward(failure),
                ));
                return None;
            }
        };

        let recovered =
            match track_direction(&current.cam0, &previous.cam0, frame_transform, config) {
                Ok(transform) => transform,
                Err(failure) => {
                    cam0_classification = Some(Cam0Classification::Rejected(
                        RejectReason::FrameBackward(failure),
                    ));
                    return None;
                }
            };
        let fb_squared = (old_cam0.translation() - recovered.translation()).norm_squared();
        if fb_squared >= config.fb_squared_threshold {
            cam0_classification = Some(Cam0Classification::Rejected(RejectReason::FrameFbSquared));
            None
        } else {
            cam0_classification = Some(Cam0Classification::Retained);
            Some(frame_transform)
        }
    });

    let mut cam1_reject = None;
    let cam1 = match (
        old_track.cam1,
        previous.cam1.as_ref(),
        current.cam1.as_ref(),
    ) {
        (Some(old_cam1), Some(previous_cam1), Some(current_cam1)) => {
            match track_direction(previous_cam1, current_cam1, old_cam1, config) {
                Ok(stereo_transform) => {
                    let recovered = match track_direction(
                        current_cam1,
                        previous_cam1,
                        stereo_transform,
                        config,
                    ) {
                        Ok(transform) => Some(transform),
                        Err(failure) => {
                            cam1_reject = Some(RejectReason::ExistingStereoBackward(failure));
                            None
                        }
                    };
                    recovered.and_then(|recovered| {
                        if (old_cam1.translation() - recovered.translation()).norm_squared()
                            >= config.fb_squared_threshold
                        {
                            cam1_reject = Some(RejectReason::ExistingStereoFbSquared);
                            None
                        } else {
                            Some(stereo_transform)
                        }
                    })
                }
                Err(failure) => {
                    cam1_reject = Some(RejectReason::ExistingStereoForward(failure));
                    None
                }
            }
        }
        _ => None,
    };

    TemporalTrackResult {
        track_id,
        created_frame_id: old_track.created_frame_id,
        cam0,
        cam0_classification,
        cam1,
        cam1_reject,
    }
}

fn track_direction(
    old_pyramid: &RawU16Pyramid,
    current_pyramid: &RawU16Pyramid,
    old_transform: AffineCompact2f,
    config: &DirectKltConfig,
) -> Result<AffineCompact2f, KltFailure> {
    track_direction_from_seed(
        old_pyramid,
        current_pyramid,
        *old_transform.translation(),
        old_transform,
        config,
    )
}

fn track_direction_from_seed(
    old_pyramid: &RawU16Pyramid,
    current_pyramid: &RawU16Pyramid,
    source_position: Vector2<f32>,
    initial_transform: AffineCompact2f,
    config: &DirectKltConfig,
) -> Result<AffineCompact2f, KltFailure> {
    track_direction_from_seed_impl(
        old_pyramid,
        current_pyramid,
        source_position,
        initial_transform,
        false,
        config,
        #[cfg(test)]
        None,
    )
}

/// Like [`track_direction_from_seed`], but the IC search starts from
/// `initial_transform`'s linear part (a predicted patch warp) instead of the
/// identity, and the found transform is returned without composing a base.
/// Used only by the opt-in reprojection stereo seed.
fn track_direction_from_warped_seed(
    old_pyramid: &RawU16Pyramid,
    current_pyramid: &RawU16Pyramid,
    source_position: Vector2<f32>,
    initial_transform: AffineCompact2f,
    config: &DirectKltConfig,
) -> Result<AffineCompact2f, KltFailure> {
    track_direction_from_seed_impl(
        old_pyramid,
        current_pyramid,
        source_position,
        initial_transform,
        true,
        config,
        #[cfg(test)]
        None,
    )
}

#[cfg(test)]
trait KltIterationObserver {
    fn record_iteration(
        &mut self,
        level: usize,
        iteration: usize,
        patch: &MeanNormalizedPatch51,
        residual: &crate::patch::PatchData51,
        increment: nalgebra::Vector3<f32>,
        before: AffineCompact2f,
        update: crate::update::Se2,
        after: AffineCompact2f,
    );
}

#[cfg(test)]
fn track_direction_from_seed_with_observer(
    old_pyramid: &RawU16Pyramid,
    current_pyramid: &RawU16Pyramid,
    source_position: Vector2<f32>,
    initial_transform: AffineCompact2f,
    config: &DirectKltConfig,
    observer: &mut dyn KltIterationObserver,
) -> Result<AffineCompact2f, KltFailure> {
    track_direction_from_seed_impl(
        old_pyramid,
        current_pyramid,
        source_position,
        initial_transform,
        false,
        config,
        Some(observer),
    )
}

fn track_direction_from_seed_impl(
    old_pyramid: &RawU16Pyramid,
    current_pyramid: &RawU16Pyramid,
    source_position: Vector2<f32>,
    initial_transform: AffineCompact2f,
    search_from_initial_linear: bool,
    config: &DirectKltConfig,
    #[cfg(test)] mut observer: Option<&mut dyn KltIterationObserver>,
) -> Result<AffineCompact2f, KltFailure> {
    // Basalt resets the linear part for the current IC search. The previous
    // affine linear part is composed back onto the result after all levels;
    // the old point centre seeds the translation search.
    //
    // Opt-in (`search_from_initial_linear`, used only by the reprojection
    // stereo seed): the search instead starts from the initial linear part,
    // a predicted cam0->cam1 patch warp, and the result is returned as found.
    let (base_linear, search_linear) = if search_from_initial_linear {
        (Matrix2::identity(), *initial_transform.linear())
    } else {
        (*initial_transform.linear(), Matrix2::identity())
    };
    let mut transform = AffineCompact2f::new(search_linear, *initial_transform.translation());
    for level in (0..=config.pyramid_levels).rev() {
        let scale = (1_u32 << level) as f32;
        let old_position = source_position / scale;
        let level_image = old_pyramid
            .level(level)
            .ok_or(KltFailure::SourcePatchInvalid)?;
        let current_image = current_pyramid
            .level(level)
            .ok_or(KltFailure::TargetOutOfBounds)?;
        let patch = MeanNormalizedPatch51::from_image(level_image, old_position);
        if !patch.valid {
            return Err(KltFailure::SourcePatchInvalid);
        }
        let mut level_transform =
            AffineCompact2f::new(*transform.linear(), *transform.translation() / scale);
        // `iteration` is read only by the `#[cfg(test)]` observer hook below;
        // keep the name (rather than `_iteration`) so that branch still compiles.
        #[allow(unused_variables)]
        for iteration in 0..config.max_iterations {
            let residual = patch
                .residual(current_image, &level_transform)
                .map_err(map_patch_error)?;
            let increment = patch.ic_increment(&residual);
            #[cfg(test)]
            let before = level_transform;
            level_transform
                .try_right_compose_se2(increment)
                .map_err(map_update_error)?;
            #[cfg(test)]
            if let Some(observer) = observer.as_deref_mut() {
                observer.record_iteration(
                    level,
                    iteration,
                    &patch,
                    &residual,
                    increment,
                    before,
                    crate::update::Se2::exp(increment),
                    level_transform,
                );
            }
            if !current_image.in_bounds(*level_transform.translation(), 2.0) {
                return Err(KltFailure::TargetOutOfBounds);
            }
        }
        transform = AffineCompact2f::new(
            *level_transform.linear(),
            *level_transform.translation() * scale,
        );
    }
    // The upstream map carries the accumulated affine linear part between
    // frames, while resetting it for the current IC search above.
    Ok(AffineCompact2f::new(
        base_linear * *transform.linear(),
        *transform.translation(),
    ))
}

const fn map_patch_error(error: PatchResidualError) -> KltFailure {
    match error {
        PatchResidualError::NoValidTargetSamples => KltFailure::TargetNoValidSamples,
        PatchResidualError::InsufficientOverlap { .. } => KltFailure::TargetInsufficientOverlap,
    }
}

const fn map_update_error(error: Se2UpdateError) -> KltFailure {
    match error {
        Se2UpdateError::NonFiniteIncrement => KltFailure::IncrementNonFinite,
        Se2UpdateError::IncrementTooLarge { .. } => KltFailure::IncrementTooLarge,
    }
}

/// Upstream `REPROJ_FIX_DEPTH` seed: the cam1 pixel of the point `depth_m`
/// metres along the (unit) cam0 bearing of `cam0_pixel`.  Returns `None` when
/// the bearing or projection is invalid or the projection lies outside cam1.
pub(crate) fn reprojected_stereo_seed(
    calibration: &BasaltCalibration,
    cam0_pixel: Vector2<f32>,
    depth_m: f64,
) -> Option<Vector2<f32>> {
    let camera0 = calibration.camera(0)?;
    let camera1 = calibration.camera(1)?;
    let bearing0 = camera0.unproject(&Point2::new(cam0_pixel.x as f64, cam0_pixel.y as f64))?;
    let norm = bearing0.norm();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return None;
    }
    let point_cam0 = Point3::from(bearing0 / norm * depth_m);
    // T_cam1_cam0 = T_imu_cam1^-1 * T_imu_cam0.
    let t_cam1_cam0 = calibration
        .imu_to_camera(1)?
        .compose(calibration.camera_to_imu(0)?);
    let point_cam1 = t_cam1_cam0.transform_point(&point_cam0);
    let pixel = camera1.project(&point_cam1)?;
    if !pixel.x.is_finite() || !pixel.y.is_finite() || !camera1.contains_pixel(&pixel) {
        return None;
    }
    Some(Vector2::new(pixel.x as f32, pixel.y as f32))
}

/// [`reprojected_stereo_seed`] plus the local linear warp of the cam0 ->
/// cam1 reprojection at that depth (central differences over +-1 px).  The
/// warp is the identity when a neighbouring sample does not reproject into
/// cam1 or the differences are not invertible.
pub(crate) fn reprojected_stereo_seed_with_warp(
    calibration: &BasaltCalibration,
    cam0_pixel: Vector2<f32>,
    depth_m: f64,
) -> Option<(Vector2<f32>, Matrix2<f32>)> {
    let seed = reprojected_stereo_seed(calibration, cam0_pixel, depth_m)?;
    let step = 1.0_f32;
    let column = |offset: Vector2<f32>| {
        let plus = reprojected_stereo_seed(calibration, cam0_pixel + offset, depth_m)?;
        let minus = reprojected_stereo_seed(calibration, cam0_pixel - offset, depth_m)?;
        Some((plus - minus) / (2.0 * step))
    };
    let warp = column(Vector2::new(step, 0.0))
        .zip(column(Vector2::new(0.0, step)))
        .map(|(dx, dy)| Matrix2::from_columns(&[dx, dy]))
        .filter(|warp| {
            let determinant = warp.determinant();
            determinant.is_finite() && determinant.abs() > 1e-3
        })
        .unwrap_or_else(Matrix2::identity);
    Some((seed, warp))
}

fn essential_residual(
    calibration: &BasaltCalibration,
    cam0_pixel: Vector2<f32>,
    cam1_pixel: Vector2<f32>,
) -> Option<f64> {
    let camera0 = calibration.camera(0)?;
    let camera1 = calibration.camera(1)?;
    let bearing0 = camera0.unproject(&Point2::new(cam0_pixel.x as f64, cam0_pixel.y as f64))?;
    let bearing1 = camera1.unproject(&Point2::new(cam1_pixel.x as f64, cam1_pixel.y as f64))?;
    let t_imu_cam0 = calibration.camera_to_imu(0)?;
    let t_imu_cam1 = calibration.camera_to_imu(1)?;
    let t_cam0_cam1 = t_imu_cam0.inverse().compose(t_imu_cam1);
    let translation = t_cam0_cam1.translation;
    let norm = translation.norm();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return None;
    }
    let t = translation / norm;
    let skew = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
    let essential = skew * t_cam0_cam1.rotation.to_rotation_matrix().into_inner();
    Some((bearing0.dot(&(essential * bearing1))).abs())
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix2, Vector2};

    use super::*;
    use crate::{camera::DoubleSphereCamera, types::BasaltNavState};

    #[test]
    fn gyro_integration_uses_left_samples_and_positive_intervals() {
        let sample = |timestamp_ns, gyro_rad_s| ImuSample {
            timestamp_ns,
            gyro_rad_s,
            accel_m_s2: Vector3::zeros(),
        };
        let imu = [
            sample(0, Vector3::new(1.0, 2.0, 3.0)),
            sample(10_000_000, Vector3::new(4.0, 5.0, 6.0)),
            sample(10_000_000, Vector3::new(7.0, 8.0, 9.0)),
            sample(5_000_000, Vector3::new(10.0, 11.0, 12.0)),
            sample(25_000_000, Vector3::repeat(999.0)),
        ];
        let zero_bias = Vector3::zeros();
        assert_eq!(integrate_gyro_imu_frame(&[], zero_bias), None);
        assert_eq!(integrate_gyro_imu_frame(&imu[..1], zero_bias), None);
        let expected = Vector3::new(0.21, 0.24, 0.27);
        assert!((integrate_gyro_imu_frame(&imu, zero_bias).unwrap() - expected).norm() < 1e-12);

        // Bias is subtracted from each leading sample before integration:
        // the two positive-dt intervals used above (0.01 s and 0.02 s) sum
        // to 0.03 s, so a constant bias offset should subtract `bias * 0.03`.
        let bias = Vector3::new(1.0, 1.0, 1.0);
        let expected_biased = expected - bias * 0.03;
        assert!((integrate_gyro_imu_frame(&imu, bias).unwrap() - expected_biased).norm() < 1e-12);
    }

    #[test]
    fn rotation_seeded_pixel_matches_independently_rotated_bearing() {
        let camera = DoubleSphereCamera::new(40.0, 40.0, 48.0, 48.0, 0.0, 0.5, 96, 96).unwrap();
        let old_pixel = Vector2::new(53.0, 45.0);
        let b0 = camera.unproject_f32(&Point2::from(old_pixel)).unwrap();
        let axis = Vector3::new(0.0, 0.05, 0.0);
        let r = UnitQuaternion::from_scaled_axis(axis).to_rotation_matrix();
        let b_expected = r * b0;
        let p_expected = camera.project_f32(&Point3::from(b_expected)).unwrap();
        let actual = rotation_seeded_pixel(&camera, old_pixel, -axis).unwrap();
        assert!((actual - p_expected.coords).norm() < 1e-3);
    }

    #[test]
    fn reject_reason_counters_are_stage_specific_and_deterministic() {
        let mut counters = RejectReasonCounters::default();
        counters.record(RejectReason::FrameFbSquared);
        counters.record(RejectReason::FrameFbSquared);
        counters.record(RejectReason::FrameForward(KltFailure::SourcePatchInvalid));
        assert_eq!(counters.count(RejectReason::FrameFbSquared), 2);
        assert_eq!(counters.total(), 3);
        assert_eq!(counters.iter().count(), 2);
    }

    #[test]
    fn essential_residual_is_zero_for_a_horizontal_stereo_pair() {
        let camera = DoubleSphereCamera::new(40.0, 40.0, 48.0, 48.0, 0.0, 0.5, 96, 96).unwrap();
        let mut calibration = synthetic_calibration(camera);
        calibration.t_imu_cam[1].translation.x = 0.1;
        let residual = essential_residual(
            &calibration,
            Vector2::new(48.0, 48.0),
            Vector2::new(52.0, 48.0),
        )
        .unwrap();
        assert!(residual < 1e-12);
    }

    /// Independent pinhole reference for the `REPROJ_FIX_DEPTH` seed: a
    /// divergent rig whose cam1 is rotated 75 degrees about the cam0 x axis
    /// and offset by a 0.138 m baseline (Project Aria geometry).
    #[test]
    fn reprojected_stereo_seed_matches_independent_pinhole_projection() {
        let (f, cx, cy) = (241.6, 382.4, 286.5);
        let camera = DoubleSphereCamera::new(f, f, cx, cy, 0.0, 0.0, 758, 572).unwrap();
        let mut calibration = synthetic_calibration(camera);
        calibration.resolutions = vec![(758, 572); 2];
        let r_imu_cam0 = UnitQuaternion::from_scaled_axis(Vector3::new(0.1, -0.2, 0.05));
        let t_imu_cam0 = Vector3::new(0.02, -0.1, 0.07);
        // cam1 = cam0 rotated +75 degrees about its own x axis.
        let r_cam0_cam1 =
            UnitQuaternion::from_scaled_axis(Vector3::new(75_f64.to_radians(), 0.0, 0.0));
        let t_cam0_cam1 = Vector3::new(0.004, -0.109, -0.085);
        calibration.t_imu_cam = vec![
            visloc_core::geometry::SE3::new(r_imu_cam0, t_imu_cam0),
            visloc_core::geometry::SE3::new(
                r_imu_cam0 * r_cam0_cam1,
                t_imu_cam0 + r_imu_cam0 * t_cam0_cam1,
            ),
        ];

        // A pixel in the top band of cam0, which this rig shares with cam1.
        let pixel0 = Vector2::new(600.0_f32, 120.0_f32);
        for depth in [2.0, 5.0, 20.0] {
            let ray = Vector3::new((pixel0.x as f64 - cx) / f, (pixel0.y as f64 - cy) / f, 1.0);
            let point_cam0 = ray.normalize() * depth;
            let point_cam1 = r_cam0_cam1.inverse() * (point_cam0 - t_cam0_cam1);
            assert!(point_cam1.z > 0.0);
            let expected = Vector2::new(
                f * point_cam1.x / point_cam1.z + cx,
                f * point_cam1.y / point_cam1.z + cy,
            );
            let seed =
                reprojected_stereo_seed(&calibration, pixel0, depth).expect("point is inside cam1");
            assert!(
                (seed.cast::<f64>() - expected).norm() < 1e-3,
                "depth {depth}: seed {seed:?} expected {expected:?}"
            );
            // The same-pixel guess is hundreds of pixels off on this rig.
            assert!((seed - pixel0).norm() > 200.0, "seed {seed:?}");
        }
        // At 1 m the baseline pushes the same ray below cam1's last row.
        assert_eq!(reprojected_stereo_seed(&calibration, pixel0, 1.0), None);
        // A pixel at the bottom of cam0 looks away from cam1: no seed.
        assert_eq!(
            reprojected_stereo_seed(&calibration, Vector2::new(382.0, 560.0), 2.0),
            None
        );

        // Degenerate rig (identical cameras, no baseline): the seed is the
        // same pixel at every depth.
        let identity = synthetic_calibration(camera);
        let seed = reprojected_stereo_seed(&identity, pixel0, 3.0).unwrap();
        assert!((seed - pixel0).norm() < 1e-3);
        let (_, warp) = reprojected_stereo_seed_with_warp(&identity, pixel0, 3.0).unwrap();
        assert!((warp - Matrix2::identity()).norm() < 1e-3);

        // The patch warp is the local derivative of the reprojection: map a
        // small cam0 offset and compare with the independent projection.
        let (seed, warp) = reprojected_stereo_seed_with_warp(&calibration, pixel0, 5.0).unwrap();
        let project_at = |pixel: Vector2<f32>| {
            let ray = Vector3::new((pixel.x as f64 - cx) / f, (pixel.y as f64 - cy) / f, 1.0);
            let point_cam1 = r_cam0_cam1.inverse() * (ray.normalize() * 5.0 - t_cam0_cam1);
            Vector2::new(
                f * point_cam1.x / point_cam1.z + cx,
                f * point_cam1.y / point_cam1.z + cy,
            )
        };
        let offset = Vector2::new(3.0_f32, -2.0_f32);
        let predicted = (seed + warp * offset).cast::<f64>();
        let actual = project_at(pixel0 + offset);
        assert!(
            (predicted - actual).norm() < 0.05,
            "{predicted:?} vs {actual:?}"
        );
        // On this rig the warp is far from a pure rotation (scale/shear).
        assert!((warp.determinant() - 1.0).abs() > 0.05, "{warp:?}");
    }

    #[test]
    fn multi_camera_options_validate_the_default_depth() {
        let camera = DoubleSphereCamera::new(40.0, 40.0, 48.0, 48.0, 0.0, 0.5, 96, 96).unwrap();
        let stream =
            DirectKltStream::new(synthetic_calibration(camera), DirectKltConfig::default())
                .unwrap();
        assert!(stream.multi_camera_options().is_default());
        for depth_m in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(stream
                .clone()
                .with_multi_camera_options(MultiCameraFlowOptions {
                    stereo_guess: StereoMatchingGuess::ReprojectFixedDepth { depth_m },
                })
                .is_err());
        }
    }

    fn synthetic_calibration(camera: DoubleSphereCamera) -> BasaltCalibration {
        BasaltCalibration {
            t_imu_cam: vec![
                visloc_core::geometry::SE3::identity(),
                visloc_core::geometry::SE3::identity(),
            ],
            cameras: vec![camera, camera],
            resolutions: vec![(96, 96), (96, 96)],
            calib_accel_bias: vec![0.0; 9],
            calib_gyro_bias: vec![0.0; 12],
            imu_update_rate_hz: 200.0,
            accel_noise_std: nalgebra::Vector3::repeat(0.01),
            gyro_noise_std: nalgebra::Vector3::repeat(0.01),
            accel_bias_std: nalgebra::Vector3::repeat(0.01),
            gyro_bias_std: nalgebra::Vector3::repeat(0.01),
            t_mocap_world: visloc_core::geometry::SE3::identity(),
            t_imu_marker: visloc_core::geometry::SE3::identity(),
            mocap_time_offset_ns: 0,
            mocap_to_imu_offset_ns: 0,
            cam_time_offset_ns: 0,
        }
    }

    #[allow(dead_code)]
    fn _keep_types_linked(_: Matrix2<f32>, _: BasaltNavState) {}

    /// Run the real two-image cam1 KLT path on the first frame-12 divergence
    /// witness.  The normal test suite leaves this opt-in when the input-root
    /// variables are absent; a harness supplies the sensor root and trace
    /// path to make the operation reproducible on both MSVC and Linux.
    #[test]
    #[ignore = "diagnostic-only real-image KLT repro; requires explicit input/config/calibration/trace env"]
    fn m11_frame12_cam1_real_path_klt_trace() {
        let input_root = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_ROOT").expect("VISLOC_BASALT_KLT_REPRO_ROOT"),
        );
        let trace_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_TRACE")
                .expect("VISLOC_BASALT_KLT_REPRO_TRACE"),
        );
        let calibration_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_CALIBRATION")
                .expect("VISLOC_BASALT_KLT_REPRO_CALIBRATION"),
        );
        let config_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_CONFIG")
                .expect("VISLOC_BASALT_KLT_REPRO_CONFIG"),
        );
        let source_timestamp_ns = 1_403_636_580_313_555_456_i64;
        let target_timestamp_ns = 1_403_636_580_363_555_584_i64;
        let dataset =
            crate::euroc::EurocSensorDataset::open(&input_root, &calibration_path, &config_path)
                .expect("open KLT repro EuRoC dataset");
        let source_frame = dataset.frame(11).expect("load KLT repro source frame");
        let target_frame = dataset.frame(12).expect("load KLT repro target frame");
        assert_eq!(source_frame.timestamp_ns, source_timestamp_ns);
        assert_eq!(target_frame.timestamp_ns, target_timestamp_ns);
        let source_path = source_frame.cam1_path.clone().expect("source cam1 path");
        let target_path = target_frame.cam1_path.clone().expect("target cam1 path");
        let config = DirectKltConfig::default();
        let source = RawU16Pyramid::from_image(
            source_frame.cam1.expect("source cam1 image"),
            config.pyramid_levels,
        )
        .expect("source KLT repro pyramid");
        let target = RawU16Pyramid::from_image(
            target_frame.cam1.expect("target cam1 image"),
            config.pyramid_levels,
        )
        .expect("target KLT repro pyramid");
        let source_position = Vector2::new(469.9930419921875_f32, 50.48072052001953_f32);
        let initial = AffineCompact2f::new(Matrix2::identity(), source_position);
        let result = track_direction_from_seed(&source, &target, source_position, initial, &config)
            .expect("real frame-11 to frame-12 cam1 KLT path");
        let record = serde_json::json!({
            "schema": "visloc.basalt.m11.klt_real_path_trace.v1",
            "source": "rust",
            "camera_id": 1,
            "source_frame_id": 11,
            "target_frame_id": 12,
            "source_timestamp_ns": source_timestamp_ns,
            "target_timestamp_ns": target_timestamp_ns,
            "source_path": source_path,
            "target_path": target_path,
            "decoder": "EurocSensorDataset::frame -> euroc::read_raw_u16_png -> dynamic_to_raw_u16",
            "calibration_path": calibration_path,
            "config_path": config_path,
            "image_dimensions": {
                "source": [source.level(0).expect("source level zero").width(), source.level(0).expect("source level zero").height()],
                "target": [target.level(0).expect("target level zero").width(), target.level(0).expect("target level zero").height()]
            },
            "config": {
                "pyramid_levels": config.pyramid_levels,
                "max_iterations": config.max_iterations,
                "source_position_f32": [source_position.x, source_position.y],
                "source_position_f32_bits": [format!("{:08x}", source_position.x.to_bits()), format!("{:08x}", source_position.y.to_bits())],
                "initial_linear_f32_bits_column_major": matrix2_bits(initial.linear()),
                "initial_translation_f32_bits": vector2_bits(initial.translation()),
            },
            "result": {
                "linear_f32": matrix2_values(result.linear()),
                "linear_f32_bits_column_major": matrix2_bits(result.linear()),
                "translation_f32": [result.translation().x, result.translation().y],
                "translation_f32_bits": vector2_bits(result.translation()),
            },
        });
        if let Some(parent) = trace_path.parent() {
            std::fs::create_dir_all(parent).expect("KLT repro trace parent");
        }
        std::fs::write(
            &trace_path,
            serde_json::to_vec_pretty(&record).expect("KLT repro trace JSON"),
        )
        .expect("KLT repro trace output");
        println!(
            "{}",
            serde_json::to_string(&record).expect("KLT repro trace line")
        );
    }

    /// Qualify the finite-angle Arm-derived cosine path on the real decoder and
    /// real two-image KLT path.  This remains ignored and requires explicit
    /// input/config/calibration variables; the assertion is the pinned
    /// native/Linux endpoint witness recorded in
    /// work/m11_frame12_native_endpoint_binding_20260907.
    #[test]
    #[ignore = "diagnostic-only candidate endpoint; requires explicit input/config/calibration env"]
    fn m11_frame12_cam1_real_path_klt_portable_cos_candidate() {
        let input_root = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_ROOT").expect("VISLOC_BASALT_KLT_REPRO_ROOT"),
        );
        let calibration_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_CALIBRATION")
                .expect("VISLOC_BASALT_KLT_REPRO_CALIBRATION"),
        );
        let config_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_CONFIG")
                .expect("VISLOC_BASALT_KLT_REPRO_CONFIG"),
        );
        let dataset =
            crate::euroc::EurocSensorDataset::open(&input_root, &calibration_path, &config_path)
                .expect("open candidate KLT repro EuRoC dataset");
        let source_frame = dataset.frame(11).expect("load candidate KLT source frame");
        let target_frame = dataset.frame(12).expect("load candidate KLT target frame");
        assert_eq!(source_frame.timestamp_ns, 1_403_636_580_313_555_456_i64);
        assert_eq!(target_frame.timestamp_ns, 1_403_636_580_363_555_584_i64);
        let config = DirectKltConfig::default();
        let source = RawU16Pyramid::from_image(
            source_frame.cam1.expect("candidate source cam1 image"),
            config.pyramid_levels,
        )
        .expect("candidate source KLT pyramid");
        let target = RawU16Pyramid::from_image(
            target_frame.cam1.expect("candidate target cam1 image"),
            config.pyramid_levels,
        )
        .expect("candidate target KLT pyramid");
        let source_position = Vector2::new(469.9930419921875_f32, 50.48072052001953_f32);
        let initial = AffineCompact2f::new(Matrix2::identity(), source_position);
        let result = track_direction_from_seed(&source, &target, source_position, initial, &config)
            .expect("candidate frame-11 to frame-12 cam1 KLT path");
        assert_eq!(
            result.translation().x.to_bits(),
            0x43ebb126,
            "candidate endpoint x must match pinned native/Linux endpoint"
        );
        assert_eq!(
            result.translation().y.to_bits(),
            0x4232f6b6,
            "candidate endpoint y must match pinned native/Linux endpoint"
        );
        println!(
            "{{\"schema\":\"visloc.basalt.m11.klt_portable_cos_candidate_endpoint.v1\",\"candidate_gate\":true,\"translation_f32_bits\":[\"{:08x}\",\"{:08x}\"]}}",
            result.translation().x.to_bits(),
            result.translation().y.to_bits()
        );
    }

    #[derive(Default)]
    struct KltIterationTraceRecorder {
        records: Vec<serde_json::Value>,
    }

    impl KltIterationTraceRecorder {
        fn f32_bits(values: &[f32]) -> Vec<String> {
            values
                .iter()
                .map(|value| format!("{:08x}", value.to_bits()))
                .collect()
        }

        fn patch_record(patch: &crate::patch::MeanNormalizedPatch51) -> serde_json::Value {
            let data: Vec<f32> = patch.data.iter().copied().collect();
            let jacobian: Vec<f32> = (0..52)
                .flat_map(|row| (0..3).map(move |column| patch.jacobian_se2[(row, column)]))
                .collect();
            let inverse_jacobian: Vec<f32> = (0..3)
                .flat_map(|row| (0..52).map(move |column| patch.h_se2_inv_j_se2_t[(row, column)]))
                .collect();
            serde_json::json!({
                "position_f32": [patch.position.x, patch.position.y],
                "position_f32_bits": Self::f32_bits(&[patch.position.x, patch.position.y]),
                "mean_f32": patch.mean,
                "mean_f32_bits": format!("{:08x}", patch.mean.to_bits()),
                "valid_samples": patch.valid_samples,
                "valid": patch.valid,
                "data_f32": data,
                "data_f32_bits": Self::f32_bits(&data),
                "jacobian_se2_f32_row_major": jacobian,
                "jacobian_se2_f32_bits_row_major": Self::f32_bits(&jacobian),
                "h_se2_inv_j_se2_t_f32_row_major": inverse_jacobian,
                "h_se2_inv_j_se2_t_f32_bits_row_major": Self::f32_bits(&inverse_jacobian),
            })
        }

        fn residual_record(residual: &crate::patch::PatchData51) -> serde_json::Value {
            let values: Vec<f32> = residual.iter().copied().collect();
            serde_json::json!({
                "f32": values,
                "f32_bits": Self::f32_bits(&values),
            })
        }

        fn vector3_record(vector: nalgebra::Vector3<f32>) -> serde_json::Value {
            let values = [vector.x, vector.y, vector.z];
            serde_json::json!({
                "f32": values,
                "f32_bits": Self::f32_bits(&values),
            })
        }

        fn affine_record(transform: AffineCompact2f) -> serde_json::Value {
            serde_json::json!({
                "linear_f32": matrix2_values(transform.linear()),
                "linear_f32_bits_column_major": matrix2_bits(transform.linear()),
                "translation_f32": [transform.translation().x, transform.translation().y],
                "translation_f32_bits": vector2_bits(transform.translation()),
            })
        }

        fn se2_exp_record(
            increment: nalgebra::Vector3<f32>,
            update: crate::update::Se2,
        ) -> serde_json::Value {
            let theta = increment.z;
            let sin_theta = update.rotation[(1, 0)];
            let cos_theta = update.rotation[(0, 0)];
            let (sin_over_theta, one_minus_cos_over_theta) = if theta.abs() < 1e-5 {
                let theta_sq = theta * theta;
                (
                    1.0 - (1.0 / 6.0) * theta_sq,
                    0.5 * theta - (1.0 / 24.0) * theta * theta_sq,
                )
            } else {
                (sin_theta / theta, (1.0 - cos_theta) / theta)
            };
            serde_json::json!({
                "theta_f32": theta,
                "theta_f32_bits": format!("{:08x}", theta.to_bits()),
                "sin_theta_f32": sin_theta,
                "sin_theta_f32_bits": format!("{:08x}", sin_theta.to_bits()),
                "cos_theta_f32": cos_theta,
                "cos_theta_f32_bits": format!("{:08x}", cos_theta.to_bits()),
                "sin_over_theta_f32": sin_over_theta,
                "sin_over_theta_f32_bits": format!("{:08x}", sin_over_theta.to_bits()),
                "one_minus_cos_over_theta_f32": one_minus_cos_over_theta,
                "one_minus_cos_over_theta_f32_bits":
                    format!("{:08x}", one_minus_cos_over_theta.to_bits()),
                "rotation_f32": [
                    [update.rotation[(0, 0)], update.rotation[(0, 1)]],
                    [update.rotation[(1, 0)], update.rotation[(1, 1)]],
                ],
                "rotation_f32_bits_column_major": [
                    format!("{:08x}", update.rotation[(0, 0)].to_bits()),
                    format!("{:08x}", update.rotation[(1, 0)].to_bits()),
                    format!("{:08x}", update.rotation[(0, 1)].to_bits()),
                    format!("{:08x}", update.rotation[(1, 1)].to_bits()),
                ],
                "translation_f32": [update.translation.x, update.translation.y],
                "translation_f32_bits": vector2_bits(&update.translation),
            })
        }
    }

    impl KltIterationObserver for KltIterationTraceRecorder {
        fn record_iteration(
            &mut self,
            level: usize,
            iteration: usize,
            patch: &crate::patch::MeanNormalizedPatch51,
            residual: &crate::patch::PatchData51,
            increment: nalgebra::Vector3<f32>,
            before: AffineCompact2f,
            update: crate::update::Se2,
            after: AffineCompact2f,
        ) {
            self.records.push(serde_json::json!({
                "level": level,
                "iteration": iteration,
                "scale_f32": (1_u32 << level) as f32,
                "patch": Self::patch_record(patch),
                "residual": Self::residual_record(residual),
                "increment": Self::vector3_record(increment),
                "recomputed_se2_exp": Self::se2_exp_record(increment, update),
                "affine_before": Self::affine_record(before),
                "affine_after": Self::affine_record(after),
            }));
        }
    }

    #[test]
    #[ignore = "diagnostic-only real-image KLT iteration trace; requires explicit input/config/calibration/trace env"]
    fn m11_frame12_cam1_real_path_klt_iteration_trace() {
        let input_root = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_ROOT").expect("VISLOC_BASALT_KLT_REPRO_ROOT"),
        );
        let trace_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_ITER_TRACE")
                .expect("VISLOC_BASALT_KLT_REPRO_ITER_TRACE"),
        );
        let calibration_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_CALIBRATION")
                .expect("VISLOC_BASALT_KLT_REPRO_CALIBRATION"),
        );
        let config_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_CONFIG")
                .expect("VISLOC_BASALT_KLT_REPRO_CONFIG"),
        );
        let source_timestamp_ns = 1_403_636_580_313_555_456_i64;
        let target_timestamp_ns = 1_403_636_580_363_555_584_i64;
        let dataset =
            crate::euroc::EurocSensorDataset::open(&input_root, &calibration_path, &config_path)
                .expect("open KLT iteration repro EuRoC dataset");
        let source_frame = dataset.frame(11).expect("load KLT iteration source frame");
        let target_frame = dataset.frame(12).expect("load KLT iteration target frame");
        assert_eq!(source_frame.timestamp_ns, source_timestamp_ns);
        assert_eq!(target_frame.timestamp_ns, target_timestamp_ns);
        let source_path = source_frame.cam1_path.clone().expect("source cam1 path");
        let target_path = target_frame.cam1_path.clone().expect("target cam1 path");
        let config = DirectKltConfig::default();
        let source = RawU16Pyramid::from_image(
            source_frame.cam1.expect("source cam1 image"),
            config.pyramid_levels,
        )
        .expect("source KLT iteration pyramid");
        let target = RawU16Pyramid::from_image(
            target_frame.cam1.expect("target cam1 image"),
            config.pyramid_levels,
        )
        .expect("target KLT iteration pyramid");
        let source_position = Vector2::new(469.9930419921875_f32, 50.48072052001953_f32);
        let initial = AffineCompact2f::new(Matrix2::identity(), source_position);
        let mut observer = KltIterationTraceRecorder::default();
        let result = track_direction_from_seed_with_observer(
            &source,
            &target,
            source_position,
            initial,
            &config,
            &mut observer,
        )
        .expect("real frame-11 to frame-12 KLT iteration path");
        let unobserved_result =
            track_direction_from_seed(&source, &target, source_position, initial, &config)
                .expect("unobserved real frame-11 to frame-12 KLT iteration path");
        assert_eq!(
            result, unobserved_result,
            "iteration observer must not alter the KLT result"
        );
        let record = serde_json::json!({
            "schema": "visloc.basalt.m11.klt_real_path_iteration_trace.v1",
            "source": "rust",
            "capture_status": "TRACE_CAPTURED",
            "observer_result_match_unobserved": true,
            "observer_notes": [
                "residual, increment, affine_before, and affine_after are captured from the real loop",
                "recomputed_se2_exp is evaluated from the same increment after the production update and is not an authoritative intermediate capture"
            ],
            "profile": std::env::var("VISLOC_BASALT_KLT_REPRO_PROFILE")
                .unwrap_or_else(|_| "unspecified".to_owned()),
            "camera_id": 1,
            "source_frame_id": 11,
            "target_frame_id": 12,
            "source_timestamp_ns": source_timestamp_ns,
            "target_timestamp_ns": target_timestamp_ns,
            "source_path": source_path,
            "target_path": target_path,
            "decoder": "EurocSensorDataset::frame -> euroc::read_raw_u16_png -> dynamic_to_raw_u16",
            "calibration_path": calibration_path,
            "config_path": config_path,
            "config": {
                "pyramid_levels": config.pyramid_levels,
                "max_iterations": config.max_iterations,
                "source_position_f32": [source_position.x, source_position.y],
                "source_position_f32_bits": [
                    format!("{:08x}", source_position.x.to_bits()),
                    format!("{:08x}", source_position.y.to_bits())
                ],
                "initial_linear_f32_bits_column_major": matrix2_bits(initial.linear()),
                "initial_translation_f32_bits": vector2_bits(initial.translation()),
            },
            "result": {
                "linear_f32": matrix2_values(result.linear()),
                "linear_f32_bits_column_major": matrix2_bits(result.linear()),
                "translation_f32": [result.translation().x, result.translation().y],
                "translation_f32_bits": vector2_bits(result.translation()),
            },
            "iteration_count": observer.records.len(),
            "iterations": observer.records,
        });
        if let Some(parent) = trace_path.parent() {
            std::fs::create_dir_all(parent).expect("KLT iteration trace parent");
        }
        std::fs::write(
            &trace_path,
            serde_json::to_vec(&record).expect("KLT iteration trace JSON"),
        )
        .expect("KLT iteration trace output");
        println!(
            "{}",
            serde_json::to_string(&record).expect("KLT iteration trace line")
        );
    }

    /// Capture the first current cross-platform endpoint mismatch through the
    /// private real KLT path.  The observer result is compared with an
    /// unobserved invocation so this diagnostic cannot change the endpoint.
    #[test]
    #[ignore = "diagnostic-only frame-14 cam0 KLT endpoint; requires explicit EuRoC paths"]
    fn m11_frame14_cam0_track203_real_path_klt_endpoint_diagnostic() {
        let input_root = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_ROOT").expect("VISLOC_BASALT_KLT_REPRO_ROOT"),
        );
        let trace_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_ITER_TRACE")
                .expect("VISLOC_BASALT_KLT_REPRO_ITER_TRACE"),
        );
        let calibration_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_CALIBRATION")
                .expect("VISLOC_BASALT_KLT_REPRO_CALIBRATION"),
        );
        let config_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_KLT_REPRO_CONFIG")
                .expect("VISLOC_BASALT_KLT_REPRO_CONFIG"),
        );
        const SOURCE_FRAME: usize = 13;
        const TARGET_FRAME: usize = 14;
        const TRACK_ID: u64 = 203;
        const SOURCE_TIMESTAMP_NS: i64 = 1_403_636_580_413_555_456;
        const TARGET_TIMESTAMP_NS: i64 = 1_403_636_580_463_555_584;
        const SOURCE_X: f32 = 170.6181640625;
        const SOURCE_Y: f32 = 90.51874542236328;
        const NATIVE_TARGET_X: f32 = 170.787841796875;
        const NATIVE_TARGET_Y: f32 = 81.55065155029297;

        let dataset =
            crate::euroc::EurocSensorDataset::open(&input_root, &calibration_path, &config_path)
                .expect("open frame-14 KLT diagnostic EuRoC dataset");
        let source_frame = dataset.frame(SOURCE_FRAME).expect("load frame 13");
        let target_frame = dataset.frame(TARGET_FRAME).expect("load frame 14");
        assert_eq!(source_frame.timestamp_ns, SOURCE_TIMESTAMP_NS);
        assert_eq!(target_frame.timestamp_ns, TARGET_TIMESTAMP_NS);
        let source_path = source_frame.cam0_path.clone();
        let target_path = target_frame.cam0_path.clone();
        let config = DirectKltConfig::default();
        let source = RawU16Pyramid::from_image(source_frame.cam0, config.pyramid_levels)
            .expect("frame-13 cam0 KLT pyramid");
        let target = RawU16Pyramid::from_image(target_frame.cam0, config.pyramid_levels)
            .expect("frame-14 cam0 KLT pyramid");
        let source_position = Vector2::new(SOURCE_X, SOURCE_Y);
        let initial = AffineCompact2f::new(Matrix2::identity(), source_position);
        let mut observer = KltIterationTraceRecorder::default();
        let observed = track_direction_from_seed_with_observer(
            &source,
            &target,
            source_position,
            initial,
            &config,
            &mut observer,
        )
        .expect("observed frame-13 to frame-14 cam0 KLT path");
        let unobserved =
            track_direction_from_seed(&source, &target, source_position, initial, &config)
                .expect("unobserved frame-13 to frame-14 cam0 KLT path");
        assert_eq!(
            observed, unobserved,
            "observer must be observationally neutral"
        );

        let camera = dataset
            .calibration()
            .camera(0)
            .expect("cam0 calibration for endpoint diagnostic");
        let source_bearing = camera
            .unproject(&Point2::new(
                source_position.x as f64,
                source_position.y as f64,
            ))
            .expect("unproject source seed");
        let target_bearing = camera
            .unproject(&Point2::new(
                observed.translation().x as f64,
                observed.translation().y as f64,
            ))
            .expect("unproject observed endpoint");
        let target_x = observed.translation().x;
        let target_y = observed.translation().y;
        assert_eq!(
            target_x.to_bits(),
            NATIVE_TARGET_X.to_bits(),
            "frame-14 cam0 track203 endpoint x must match the pinned native packet"
        );
        assert_eq!(
            target_y.to_bits(),
            NATIVE_TARGET_Y.to_bits(),
            "frame-14 cam0 track203 endpoint y must match the pinned native packet"
        );
        let record = serde_json::json!({
            "schema": "visloc.basalt.m11.frame14_raw_frontend_endpoint_iteration.v1",
            "source": "rust_direct_klt_stream_private_path",
            "camera_id": 0,
            "track_id": TRACK_ID,
            "source_frame_id": SOURCE_FRAME,
            "target_frame_id": TARGET_FRAME,
            "source_timestamp_ns": SOURCE_TIMESTAMP_NS,
            "target_timestamp_ns": TARGET_TIMESTAMP_NS,
            "source_path": source_path,
            "target_path": target_path,
            "decoder": "EurocSensorDataset::frame -> euroc::read_raw_u16_png -> dynamic_to_raw_u16",
            "config": {
                "pyramid_levels": config.pyramid_levels,
                "max_iterations": config.max_iterations,
                "source_position_f32": [source_position.x, source_position.y],
                "source_position_f32_bits": [format!("{:08x}", source_position.x.to_bits()), format!("{:08x}", source_position.y.to_bits())],
                "initial_linear_f32_bits_column_major": matrix2_bits(initial.linear()),
                "initial_translation_f32_bits": vector2_bits(initial.translation())
            },
            "observational_neutrality": {
                "observer_result_equals_unobserved": true,
                "observed_iteration_count": observer.records.len()
            },
            "endpoint": {
                "observed_f32": [target_x, target_y],
                "observed_f32_bits": [format!("{:08x}", target_x.to_bits()), format!("{:08x}", target_y.to_bits())],
                "pinned_native_f32": [NATIVE_TARGET_X, NATIVE_TARGET_Y],
                "pinned_native_f32_bits": [format!("{:08x}", NATIVE_TARGET_X.to_bits()), format!("{:08x}", NATIVE_TARGET_Y.to_bits())],
                "delta_f32": [target_x - NATIVE_TARGET_X, target_y - NATIVE_TARGET_Y]
            },
            "camera_unproject": {
                "source_seed_f64": [source_bearing.x, source_bearing.y, source_bearing.z],
                "target_endpoint_f64": [target_bearing.x, target_bearing.y, target_bearing.z]
            },
            "iterations": observer.records
        });
        let bytes = serde_json::to_vec(&record).expect("frame-14 KLT diagnostic JSON");
        assert!(
            bytes.len() <= 8 * 1024 * 1024,
            "diagnostic trace must remain bounded"
        );
        if let Some(parent) = trace_path.parent() {
            std::fs::create_dir_all(parent).expect("frame-14 KLT trace parent");
        }
        std::fs::write(&trace_path, &bytes).expect("frame-14 KLT trace output");
        println!(
            "{}",
            serde_json::to_string(&record).expect("frame-14 KLT trace line")
        );
    }

    #[test]
    #[ignore = "frame19 track30 cam1 real KLT diagnostic; explicit sensor-only paths required"]
    fn m11_frame19_cam1_track30_real_path_klt_diagnostic() {
        let env_path = |name| std::path::PathBuf::from(std::env::var_os(name).expect(name));
        let dataset = crate::euroc::EurocSensorDataset::open(
            &env_path("VISLOC_BASALT_KLT_REPRO_ROOT"),
            &env_path("VISLOC_BASALT_KLT_REPRO_CALIBRATION"),
            &env_path("VISLOC_BASALT_KLT_REPRO_CONFIG"),
        )
        .expect("sensor-only dataset");
        let source = dataset.frame(18).expect("frame18");
        let target = dataset.frame(19).expect("frame19");
        assert_eq!(target.timestamp_ns, 1_403_636_580_713_555_456);
        let config = DirectKltConfig::default();
        let source =
            RawU16Pyramid::from_image(source.cam1.expect("source cam1"), config.pyramid_levels)
                .unwrap();
        let target =
            RawU16Pyramid::from_image(target.cam1.expect("target cam1"), config.pyramid_levels)
                .unwrap();
        // Verified native and Rust frame18 observation. Search resets the
        // linear part; base_linear only changes the final affine linear part.
        let position = Vector2::new(f32::from_bits(0x433c7382), f32::from_bits(0x42cbefbf));
        let seed = AffineCompact2f::new(Matrix2::identity(), position);
        let mut observer = KltIterationTraceRecorder::default();
        let observed = track_direction_from_seed_with_observer(
            &source,
            &target,
            position,
            seed,
            &config,
            &mut observer,
        )
        .unwrap();
        let plain = track_direction_from_seed(&source, &target, position, seed, &config).unwrap();
        assert_eq!(observed, plain, "diagnostic neutrality");
        let endpoint = [
            observed.translation().x.to_bits(),
            observed.translation().y.to_bits(),
        ];
        let record = serde_json::json!({
            "schema": "visloc.basalt.frame19.track30.klt.v1",
            "source_frame": 18, "target_frame": 19, "camera": 1, "track": 30,
            "source_bits": ["433c7382", "42cbefbf"],
            "endpoint_bits": endpoint.map(|v| format!("{v:08x}")),
            "native_endpoint_bits": ["433c2bc2", "42d65f34"],
            "observer_neutral": true, "iterations": observer.records
        });
        let path = env_path("VISLOC_BASALT_KLT_REPRO_ITER_TRACE");
        let bytes = serde_json::to_vec(&record).unwrap();
        assert!(bytes.len() < 8 * 1024 * 1024);
        use std::io::Write;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        // This bounded endpoint oracle does not replace full VIO parity gates.
        assert_eq!(endpoint, [0x433c2bc2, 0x42d65f34]);
    }

    fn matrix2_values(matrix: &Matrix2<f32>) -> [[f32; 2]; 2] {
        [
            [matrix[(0, 0)], matrix[(0, 1)]],
            [matrix[(1, 0)], matrix[(1, 1)]],
        ]
    }

    fn matrix2_bits(matrix: &Matrix2<f32>) -> [String; 4] {
        [
            format!("{:08x}", matrix[(0, 0)].to_bits()),
            format!("{:08x}", matrix[(1, 0)].to_bits()),
            format!("{:08x}", matrix[(0, 1)].to_bits()),
            format!("{:08x}", matrix[(1, 1)].to_bits()),
        ]
    }

    fn vector2_bits(vector: &Vector2<f32>) -> [String; 2] {
        [
            format!("{:08x}", vector.x.to_bits()),
            format!("{:08x}", vector.y.to_bits()),
        ]
    }
}
