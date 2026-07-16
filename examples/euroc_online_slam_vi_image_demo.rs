//! End-to-end EuRoC VIO pipeline-integration demo on **real cam0 pixels**.
//!
//! Companion to [`euroc_online_slam_vi_demo`](./euroc_online_slam_vi_demo.rs)
//! (the synthetic-landmark variant). Both demos drive
//! [`OnlineSlamPipeline`] with the real EuRoC IMU stream + cam0 cadence and
//! the auto-bootstrap stage enabled. The difference is the **visual signal**:
//!
//! * `euroc_online_slam_vi_demo` projects a fixed 5×5 synthetic landmark grid
//!   through the GT camera pose into each frame. This keeps the pipeline's
//!   tracker fed end-to-end without ever decoding a pixel, and was useful for
//!   validating that the recently-shipped VI-init / pipeline integration runs
//!   on real IMU data.
//! * **This demo** decodes the cam0 PNG for every frame, runs
//!   [`CornerFeatureExtractor`] on it, and feeds the real corner keypoints +
//!   patch descriptors into [`OnlineSlamPipeline::process_frame`]. The
//!   initial visual map uses the first camera frame as its world origin. Stereo
//!   triangulation supplies metric depth without reading the EuRoC ground-truth
//!   stream. From that point on, both the corner detection and
//!   the descriptor matching the pipeline does inside its tracker are
//!   driven by genuine pixel intensities.
//!
//! What this validates on top of the synthetic variant:
//! 1. The pipeline accepts a real EuRoC pixel-derived feature stream — the
//!    distribution of corners (number, position, descriptor signal) is no
//!    longer a clean grid but a function of actual scene texture.
//! 2. The default `BruteForceMatcher` + `PnPRansac` inside the tracker
//!    survive real corner-patch descriptors well enough to deliver a usable
//!    localisation rate across the bootstrap and short subsequent motion.
//! 3. The auto-bootstrap stage still fires under a more realistic visual
//!    failure profile — when tracking briefly drops, IMU integration carries
//!    through and the stale-factor gate still discards anything staged
//!    before promotion.
//!
//! Known limitations (intentional, scope-controlled):
//! * Cam0's radial-tangential distortion is now undistorted by default
//!   using the published `distortion_coefficients` from the cam0
//!   calibration (`visloc_vision::distortion::RadialTangential`). Each
//!   extracted corner has its pixel coordinates mapped to the "ideal
//!   pinhole" position before back-projection or `process_frame`,
//!   removing the edge-region error that previously inflated rigid
//!   ATE. The corner *descriptors* are still extracted from the raw
//!   distorted image patches — this is the standard VIO simplification
//!   (the patch signal-to-noise is unchanged; only the position is
//!   corrected). Pass `--no-undistort` to disable the correction and
//!   reproduce the pre-correction behaviour for A/B comparison.
//! * Each cam0 seed corner is now stereo-matched against cam1 and
//!   triangulated via DLT using the published `T_BS` extrinsics; the
//!   resulting metric-scale 3D point replaces the fixed-`bootstrap_depth`
//!   back-projection for that corner. Corners that fail to match a cam1
//!   descriptor (or whose triangulation falls outside the configured
//!   depth / reprojection-error gates) are dropped by default, so every seed
//!   landmark has measured stereo depth. `--allow-fixed-depth-bootstrap`
//!   restores the historical fixed-depth fallback strictly for diagnostic A/B
//!   runs; it must not be used for ORB-SLAM3 parity claims.
//!
//! Requires the `image-io` feature.
//!
//! Usage:
//! ```sh
//! cargo run --release --features image-io \
//!     --example euroc_online_slam_vi_image_demo -- \
//!     --euroc-dir /path/to/MH_01_easy \
//!     --out-dir target/euroc_online_slam_vi_image_demo \
//!     --max-frames 400
//! ```
//! Add `--observation-confidence-ba` to run the same local-BA windows with
//! relative learned visual weights for a uniform-vs-weighted A/B comparison.
//!
//! Output is the same four files as the synthetic variant — `slam_trajectory.csv`,
//! `slam_errors.csv`, `vi_init_log.txt`, `summary.txt` — so the two demos
//! can be diffed apples-to-apples on the same EuRoC sequence.

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!(
        "this example requires the `image-io` feature; rebuild with \
         `cargo run --release --features image-io --example euroc_online_slam_vi_image_demo`"
    );
    std::process::exit(2);
}

#[cfg(feature = "image-io")]
use std::env;
#[cfg(feature = "image-io")]
use std::fs;
#[cfg(feature = "image-io")]
use std::path::PathBuf;
#[cfg(feature = "image-io")]
use std::time::Instant;

#[cfg(feature = "image-io")]
use nalgebra::{Matrix4, Point2, Point3, UnitQuaternion, Vector3};
#[cfg(feature = "image-io")]
use visloc_rs::core::geometry::{Pose, SE3};
#[cfg(feature = "image-io")]
use visloc_rs::core::types::{
    Camera, Frame, Keyframe, Landmark, LocalizationResult, Observation, StereoObservation,
    VisualMap,
};
#[cfg(feature = "image-io")]
use visloc_rs::io::euroc::{
    read_euroc_dataset_dir, EurocCameraCalibration, EurocGroundTruthSample,
};
#[cfg(feature = "image-io")]
use visloc_rs::io::images::read_common_image;
#[cfg(feature = "image-io")]
use visloc_rs::slam::{ImuNoiseModel, OnlineSlamImuConfig};
#[cfg(feature = "image-io")]
use visloc_rs::vision::distortion::RadialTangential;
#[cfg(feature = "image-io")]
use visloc_rs::vision::features::{
    CornerFeatureConfig, CornerFeatureExtractor, FeatureExtractor, FeatureSet, GrayscaleImage,
    HogLikeFeatureConfig, HogLikeFeatureExtractor,
};
#[cfg(feature = "image-io")]
use visloc_rs::vision::stereo_bootstrap::{
    bootstrap_stereo_landmarks, StereoBootstrapConfig, StereoBootstrapLandmark,
};
#[cfg(feature = "image-io")]
use visloc_rs::{
    build_stereo_replenish_candidates, read_external_deep_features_txt,
    umeyama_similarity_transform, AdaptiveImuPoseMotionModel, AdaptiveImuPoseMotionModelConfig,
    AdaptiveMotionMode, AdaptiveVelocityGateConfig, BaConfig, BiasReleaseSchedule,
    BruteForceMatcher, ConstantPoseMotionModel, ConstantVelocityMotionModel,
    CovisibilityLocalBaConfig, CovisibilityLocalBaError, CovisibilityLocalMapConfig,
    CrossCheckMatcher,
    CrossSubmapAlignmentConfig, CrossSubmapAlignmentResult, CrossSubmapBoundaryFactorResult,
    DescriptorMatch, GravityVelocityAlignment, GyroBiasAlignment, ImuPredictiveMotionModel,
    ImuPredictiveMotionModelConfig, ImuVelocityRefreshPolicy, KeyframeDecisionReason,
    KeyframePolicyConfig, LandmarkCandidate,
    LocalMappingPipeline, LocalMappingResult, LocalizationConfig, LocalizationPipeline,
    LoopAppearanceCandidateConfig, LoopClosureCandidateSource, LoopClosureConfig,
    LoopClosureVerifierConfig, LoopRefinementSolver, LoopRefinementVerifier, MapAtlas,
    MapProviderStats, Matcher, MotionBasedViInitializerConfig, MotionModel,
    MotionViInitializationEvent, MotionViInitializationStatus, MotionViRawResidualActivationConfig,
    MutualSoftmaxConfig,
    MutualSoftmaxMatcher, OnlineSlamConfig, OnlineSlamCovisibilityLocalBaConfig,
    OnlineSlamLocalBaConfig, OnlineSlamLoopClosureRefinementConfig, OnlineSlamMotionViInitConfig,
    OnlineSlamPipeline, OnlineSlamRelocalizationStats, OnlineSlamViInitConfig,
    PnPLoopClosureVerifierConfig, PoseGraphSe3Config, PosePriorVisualOverrideConfig,
    PoseTrajectory, ProjectionGuidedTrackingConfig, RelativePoseErrorConfig, Sim3PoseGraphConfig,
    SimpleKeyframePolicy, StereoReplenishConfig, Tracker, TrackingConfig, TrackingEvent,
    TrackingResult, TrackingState, TrajectorySimilarityTransform, ViInitFallback,
    ViInitializationEvent, Viba2Config, VisualInertialInitializerConfig,
};

/// Runtime-dispatched motion model. Both inner models implement
/// [`MotionModel`], but `Tracker<P, M>` is generic in `M` and Rust's
/// monomorphisation forces a single concrete `M` for the lifetime of
/// the `Tracker`. Wrap the two stock models in an enum so the demo
/// can pick between them via a CLI flag without duplicating the
/// downstream pipeline construction.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
enum DemoMotionModel {
    Pose(ConstantPoseMotionModel),
    Velocity(ConstantVelocityMotionModel),
    ImuPredictive(ImuPredictiveMotionModel),
    AdaptiveImuPose(AdaptiveImuPoseMotionModel),
}

#[cfg(feature = "image-io")]
impl DemoMotionModel {
    /// Forward a raw body-frame IMU sample into the wrapped motion
    /// model. No-op for `Pose` / `Velocity`; the IMU-predictive variant
    /// buffers the sample for the next `predict_pose` call.
    fn push_imu_measurement(&mut self, gyro: Vector3<f64>, accel: Vector3<f64>, dt: f64) {
        match self {
            DemoMotionModel::ImuPredictive(inner) => inner.push_imu_measurement(gyro, accel, dt),
            DemoMotionModel::AdaptiveImuPose(inner) => {
                inner.imu_mut().push_imu_measurement(gyro, accel, dt);
            }
            _ => {}
        }
    }

    /// Refresh `velocity_world` on the IMU-predictive variant from two
    /// successive observed camera poses. The IMU strapdown
    /// integrator's initial-velocity slot is otherwise only updated by
    /// a downstream VI-BA pass (not exercised in this demo); without
    /// this hook `velocity_world` stays at zero forever and the
    /// per-frame position prediction only captures the quadratic accel
    /// term, systematically under-predicting the body's true motion.
    /// No-op for the `Pose` / `Velocity` variants — they maintain
    /// their own state via the standard `observe` path.
    fn update_velocity_from_pose_diff(&mut self, prev: &Pose, curr: &Pose, dt_seconds: f64) {
        match self {
            DemoMotionModel::ImuPredictive(inner) => {
                inner.update_velocity_from_camera_pose_difference(prev, curr, dt_seconds);
            }
            DemoMotionModel::AdaptiveImuPose(inner) => {
                inner
                    .imu_mut()
                    .update_velocity_from_camera_pose_difference(prev, curr, dt_seconds);
            }
            _ => {}
        }
    }

    /// Push the local-VI-BA-refined per-keyframe `(velocity_world,
    /// bias_gyro, bias_acc)` into the IMU-predictive variant's
    /// integration state, matching the documented contract on
    /// `ImuPredictiveMotionModel::set_velocity_world` /
    /// `set_biases`. No-op for the `Pose` / `Velocity` variants.
    fn mirror_vi_ba_state(
        &mut self,
        velocity_world: Vector3<f64>,
        bias_gyro: Vector3<f64>,
        bias_acc: Vector3<f64>,
    ) {
        match self {
            DemoMotionModel::ImuPredictive(inner) => {
                inner.set_velocity_world(velocity_world);
                inner.set_biases(bias_gyro, bias_acc);
            }
            DemoMotionModel::AdaptiveImuPose(inner) => {
                let imu = inner.imu_mut();
                imu.set_velocity_world(velocity_world);
                imu.set_biases(bias_gyro, bias_acc);
            }
            _ => {}
        }
    }

    /// `Some((switches_to_pose, switches_to_imu,
    /// velocity_refreshes_on_switch_to_imu, current_mode))` when the
    /// wrapped model is the Phase-23 #4 / Phase-24 adaptive variant.
    /// `None` otherwise — the summary line then reports `n/a`.
    fn adaptive_stats(&self) -> Option<(u64, u64, u64, AdaptiveMotionMode)> {
        match self {
            DemoMotionModel::AdaptiveImuPose(inner) => Some((
                inner.switches_to_pose(),
                inner.switches_to_imu(),
                inner.velocity_refreshes_on_switch_to_imu(),
                inner.mode(),
            )),
            _ => None,
        }
    }
}

#[cfg(feature = "image-io")]
impl MotionModel for DemoMotionModel {
    fn predict_pose(
        &self,
        frame: &Frame,
        last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        match self {
            DemoMotionModel::Pose(inner) => {
                inner.predict_pose(frame, last_result, last_successful_pose)
            }
            DemoMotionModel::Velocity(inner) => {
                inner.predict_pose(frame, last_result, last_successful_pose)
            }
            DemoMotionModel::ImuPredictive(inner) => {
                inner.predict_pose(frame, last_result, last_successful_pose)
            }
            DemoMotionModel::AdaptiveImuPose(inner) => {
                inner.predict_pose(frame, last_result, last_successful_pose)
            }
        }
    }

    fn observe(&mut self, result: &TrackingResult) {
        match self {
            DemoMotionModel::Pose(inner) => inner.observe(result),
            DemoMotionModel::Velocity(inner) => inner.observe(result),
            DemoMotionModel::ImuPredictive(inner) => inner.observe(result),
            DemoMotionModel::AdaptiveImuPose(inner) => inner.observe(result),
        }
    }

    fn allows_pnp_pose_prior_warm_start(&self) -> bool {
        match self {
            DemoMotionModel::Pose(inner) => inner.allows_pnp_pose_prior_warm_start(),
            DemoMotionModel::Velocity(inner) => inner.allows_pnp_pose_prior_warm_start(),
            DemoMotionModel::ImuPredictive(inner) => inner.allows_pnp_pose_prior_warm_start(),
            DemoMotionModel::AdaptiveImuPose(inner) => inner.allows_pnp_pose_prior_warm_start(),
        }
    }

    fn reset(&mut self) {
        match self {
            DemoMotionModel::Pose(inner) => inner.reset(),
            DemoMotionModel::Velocity(inner) => inner.reset(),
            DemoMotionModel::ImuPredictive(inner) => inner.reset(),
            DemoMotionModel::AdaptiveImuPose(inner) => inner.reset(),
        }
    }
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MotionModelKind {
    Pose,
    Velocity,
    ImuPredictive,
    AdaptiveImuPose,
}

/// Runtime-dispatched descriptor matcher. `LocalizationPipeline<M, ...>`
/// is generic in `M` and monomorphisation fixes a single concrete type
/// for the lifetime of the pipeline; wrap `BruteForceMatcher` and
/// `CrossCheckMatcher<BruteForceMatcher>` so a single CLI flag can
/// pick between them without duplicating the pipeline construction.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone)]
enum DemoMatcher {
    BruteForce(BruteForceMatcher),
    CrossCheck(CrossCheckMatcher<BruteForceMatcher>),
    MutualSoftmax(MutualSoftmaxMatcher),
}

#[cfg(feature = "image-io")]
impl Matcher for DemoMatcher {
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch> {
        match self {
            DemoMatcher::BruteForce(m) => m.match_descriptors(query, train),
            DemoMatcher::CrossCheck(m) => m.match_descriptors(query, train),
            DemoMatcher::MutualSoftmax(m) => m.match_descriptors(query, train),
        }
    }
}

/// Runtime-dispatched feature extractor. The demo's default
/// `CornerFeatureExtractor` produces a small raw-pixel patch descriptor;
/// the alternative `HogLikeFeatureExtractor` produces a unit-norm
/// 128-D HOG/SIFT-style descriptor with optionally oriented bins,
/// raising the descriptor signal-to-noise at the cost of ~4× CPU per
/// keypoint. `SuperPointOffline` replays pre-exported SuperPoint
/// 256-D descriptors from disk (produced by
/// `scripts/export_superpoint_lightglue.py --mono-dir …`), so the
/// demo can A/B against a true deep descriptor without dragging an
/// ONNX runtime into the workspace. The error types differ across
/// the three backing extractors, so the enum unifies them through
/// `String`.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone)]
enum DemoExtractor {
    Corner(CornerFeatureExtractor),
    Hog(HogLikeFeatureExtractor),
    SuperPointOffline(SuperPointOfflineExtractor),
    /// Phase-27: in-Rust SuperPoint ONNX inference. The wrapped
    /// extractor's runtime behaviour depends on whether
    /// `visloc-vision` was built with `--features onnx-inference`:
    /// without the feature `extract()` returns
    /// `FeatureDisabled` and the demo aborts at the first frame
    /// with a clear error message pointing the operator at the
    /// feature flag and the model path.
    SuperPointOnnx(visloc_vision::features::superpoint_onnx::SuperPointOnnxExtractor),
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SuperPointOnnxBackendArg {
    CudaThenCpu,
    Cuda,
    Cpu,
}

#[cfg(feature = "image-io")]
impl SuperPointOnnxBackendArg {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cuda-then-cpu" => Ok(Self::CudaThenCpu),
            "cuda" => Ok(Self::Cuda),
            "cpu" => Ok(Self::Cpu),
            other => Err(format!(
                "--superpoint-onnx-backend: expected 'cuda-then-cpu', 'cuda', or 'cpu', got {other:?}"
            )),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::CudaThenCpu => "cuda-then-cpu",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

#[cfg(feature = "image-io")]
impl FeatureExtractor for DemoExtractor {
    type Image = GrayscaleImage;
    type Error = String;

    fn extract(&self, image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        use visloc_vision::features::deep::DeepFeatureExtractor;
        match self {
            DemoExtractor::Corner(e) => e.extract(image).map_err(|err| err.to_string()),
            DemoExtractor::Hog(e) => e.extract(image).map_err(|err| err.to_string()),
            DemoExtractor::SuperPointOffline(e) => e.extract(image),
            DemoExtractor::SuperPointOnnx(e) => e
                .extract_deep(image)
                .map(|deep| deep.into_feature_set())
                .map_err(|err| err.to_string()),
        }
    }
}

#[cfg(feature = "image-io")]
impl DemoExtractor {
    /// Set the index of the next-to-replay frame for the offline
    /// extractor. A no-op for `Corner` / `Hog` (which derive features
    /// from the supplied image on the fly). Call this before every
    /// `extract()` when running the offline path so the replay stays
    /// in sync with the EuRoC frame stream — the offline extractor
    /// returns the preloaded features at the configured index instead
    /// of running an extractor on the image.
    fn set_frame_idx(&self, frame_idx: usize) {
        if let DemoExtractor::SuperPointOffline(e) = self {
            e.set_frame_idx(frame_idx);
        }
    }

    /// Set the active camera for the offline extractor. A no-op for
    /// `Corner` / `Hog`. Used by the stereo-bootstrap path to switch
    /// from cam0 (loop default) to cam1 for the one-shot cam1
    /// `extract()` call.
    fn set_camera(&self, camera: SuperPointCamera) {
        if let DemoExtractor::SuperPointOffline(e) = self {
            e.set_camera(camera);
        }
    }
}

/// Camera selector for [`SuperPointOfflineExtractor`]. The demo loop
/// runs the cam0 path; the stereo bootstrap path makes a single
/// cam1 `extract()` call at the seed frame.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SuperPointCamera {
    Cam0,
    Cam1,
}

/// Offline SuperPoint descriptor replay. Pre-loads `frame_NNNNNN_features.txt`
/// files emitted by the Python `scripts/export_superpoint_lightglue.py
/// --mono-dir <dir>` helper for one or both cameras into in-memory
/// `Vec<FeatureSet>` tables at construction, then returns the indexed
/// entry on every `extract()` call. The replay index and active camera
/// are set explicitly by the demo via [`DemoExtractor::set_frame_idx`]
/// and [`DemoExtractor::set_camera`] before each `extract()` — auto-
/// increment would race against the stereo-bootstrap path (which makes
/// a single cam1 call at the seed frame), so the explicit setters are
/// mandatory.
///
/// When cam1 features are loaded, `--stereo-bootstrap` is permitted:
/// the seed-frame cam0 and cam1 SuperPoint feature sets are fed into
/// the existing stereo-bootstrap path so triangulated metric-depth
/// landmarks replace the fixed `bootstrap_depth` back-projection. This
/// is the Phase-17 lever that addresses the Phase-15 finding that
/// fixed 4 m depth is the bench's binding constraint for SuperPoint.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone)]
struct SuperPointOfflineExtractor {
    cam0_frames: std::sync::Arc<Vec<FeatureSet>>,
    cam1_frames: Option<std::sync::Arc<Vec<FeatureSet>>>,
    current_idx: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// `0` = Cam0, `1` = Cam1. Atomics over enums require a small
    /// integer round-trip; `set_camera` / extract decode this.
    current_camera: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(feature = "image-io")]
fn load_frame_features(dir: &std::path::Path) -> Result<Vec<FeatureSet>, String> {
    let mut entries: Vec<(usize, std::path::PathBuf)> = Vec::new();
    let read_dir = std::fs::read_dir(dir).map_err(|e| {
        format!(
            "SuperPointOfflineExtractor: read_dir {}: {e}",
            dir.display()
        )
    })?;
    for entry in read_dir {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        const PREFIX: &str = "frame_";
        const SUFFIX: &str = "_features.txt";
        if !name.starts_with(PREFIX) || !name.ends_with(SUFFIX) {
            continue;
        }
        let middle = &name[PREFIX.len()..name.len() - SUFFIX.len()];
        let Ok(idx) = middle.parse::<usize>() else {
            continue;
        };
        entries.push((idx, path));
    }
    entries.sort_by_key(|(idx, _)| *idx);
    if entries.is_empty() {
        return Err(format!(
            "SuperPointOfflineExtractor: no frame_*_features.txt files found in {}",
            dir.display(),
        ));
    }
    for (i, (idx, path)) in entries.iter().enumerate() {
        if *idx != i {
            return Err(format!(
                "SuperPointOfflineExtractor: feature files must be contiguous starting at frame_000000; gap at slot {i}, file {}",
                path.display(),
            ));
        }
    }
    let mut frames = Vec::with_capacity(entries.len());
    for (_idx, path) in &entries {
        let ext_set = read_external_deep_features_txt(path)
            .map_err(|e| format!("SuperPointOfflineExtractor: read {}: {e}", path.display()))?;
        let fs = ext_set.to_feature_set().map_err(|e| {
            format!(
                "SuperPointOfflineExtractor: to_feature_set {}: {e}",
                path.display()
            )
        })?;
        frames.push(fs);
    }
    Ok(frames)
}

#[cfg(feature = "image-io")]
impl SuperPointOfflineExtractor {
    fn load_from_dir(cam0_dir: &std::path::Path) -> Result<Self, String> {
        let cam0_frames = load_frame_features(cam0_dir)?;
        Ok(Self {
            cam0_frames: std::sync::Arc::new(cam0_frames),
            cam1_frames: None,
            current_idx: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            current_camera: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    fn load_with_cam1(
        cam0_dir: &std::path::Path,
        cam1_dir: &std::path::Path,
    ) -> Result<Self, String> {
        let cam0_frames = load_frame_features(cam0_dir)?;
        let cam1_frames = load_frame_features(cam1_dir)?;
        if cam0_frames.len() != cam1_frames.len() {
            return Err(format!(
                "SuperPointOfflineExtractor: cam0 ({}) and cam1 ({}) feature counts differ — re-export with the same --frames count",
                cam0_frames.len(),
                cam1_frames.len(),
            ));
        }
        Ok(Self {
            cam0_frames: std::sync::Arc::new(cam0_frames),
            cam1_frames: Some(std::sync::Arc::new(cam1_frames)),
            current_idx: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            current_camera: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    fn set_frame_idx(&self, idx: usize) {
        self.current_idx
            .store(idx, std::sync::atomic::Ordering::SeqCst);
    }

    fn set_camera(&self, camera: SuperPointCamera) {
        let value = match camera {
            SuperPointCamera::Cam0 => 0,
            SuperPointCamera::Cam1 => 1,
        };
        self.current_camera
            .store(value, std::sync::atomic::Ordering::SeqCst);
    }

    fn has_cam1(&self) -> bool {
        self.cam1_frames.is_some()
    }

    fn len(&self) -> usize {
        self.cam0_frames.len()
    }
}

#[cfg(feature = "image-io")]
impl FeatureExtractor for SuperPointOfflineExtractor {
    type Image = GrayscaleImage;
    type Error = String;

    fn extract(&self, _image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        let idx = self.current_idx.load(std::sync::atomic::Ordering::SeqCst);
        let cam = self
            .current_camera
            .load(std::sync::atomic::Ordering::SeqCst);
        let frames = match cam {
            0 => &self.cam0_frames,
            1 => self.cam1_frames.as_ref().ok_or_else(|| {
                "SuperPointOfflineExtractor: cam1 features not loaded; pass --superpoint-cam1-features-dir to enable stereo bootstrap".to_string()
            })?,
            other => return Err(format!("SuperPointOfflineExtractor: invalid camera id {other}")),
        };
        if idx >= frames.len() {
            return Err(format!(
                "SuperPointOfflineExtractor: requested frame_idx {idx} for cam{cam} but only {} frames preloaded; \
                 re-export with --frames covering the demo's --max-frames",
                frames.len()
            ));
        }
        Ok(frames[idx].clone())
    }
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeatureExtractorKind {
    Corner,
    Hog,
    SuperPointOffline,
    /// Phase-27: in-Rust SuperPoint ONNX inference. Requires
    /// `visloc-vision` built with `--features onnx-inference` and
    /// `--superpoint-onnx-model <path>` pointing at a SuperPoint
    /// ONNX model with the LightGlue-ONNX-style I/O contract (see
    /// `docs/superpoint_onnx_runtime_plan.md`).
    SuperPointOnnx,
}

#[cfg(feature = "image-io")]
#[derive(Debug)]
struct CliArgs {
    euroc_dir: PathBuf,
    out_dir: PathBuf,
    max_frames: usize,
    gravity_world: Vector3<f64>,
    vi_init_max_wait_seconds: f64,
    vi_init_gyro_std_limit: Option<f64>,
    vi_init_mean_gyro_magnitude_limit: Option<f64>,
    vi_init_accel_std_limit: Option<f64>,
    /// Phase-19 lever. Maps to
    /// `OnlineSlamViInitConfig::try_initialize_on_every_frame`.
    /// `false` (default) keeps the historical "VI-init's try_initialize
    /// only fires on new-keyframe frames" contract. `true` lets the
    /// stage attempt promotion on every frame; on success without a new
    /// KF this frame, the promotion binds to the latest existing
    /// keyframe.
    vi_init_try_initialize_on_every_frame: bool,
    /// Fixed metric depth assumed for every first-frame corner during the
    /// landmark-map bootstrap. EuRoC MH indoor scenes hover ~4 m from the
    /// camera, so the default keeps the rigid-ATE error reasonable while
    /// still letting the tracker exercise real corner descriptors.
    bootstrap_depth_meters: f64,
    /// Corner extractor knobs. Defaults match the patch sizes
    /// `online_slam_image_vo_loop_demo` settled on for real KITTI imagery —
    /// they generalise well to EuRoC.
    corner_max_features: usize,
    corner_min_score: f32,
    corner_descriptor_radius: usize,
    /// When `true` (the default) every extracted corner is mapped from
    /// its raw distorted pixel position to the "ideal pinhole" pixel
    /// using cam0's published radial-tangential distortion model. Set
    /// to `false` via `--no-undistort` to reproduce the
    /// pre-correction behaviour for A/B comparison.
    undistort: bool,
    /// When `true` (the default) the seed frame's cam0 corners are
    /// stereo-matched against cam1 and triangulated via DLT using the
    /// published `T_BS` extrinsics. Each surviving match becomes a
    /// metric-scale landmark. Seed keypoints without stereo depth are dropped
    /// by default; fixed-depth fallback is an explicit diagnostic opt-in.
    stereo_bootstrap: bool,
    /// Enable the motion-based VI init stage (VIBA1 / optional VIBA2).
    /// Off by default so the demo stays backwards compatible with the
    /// existing baseline. The stage normally gates on static VI init.
    motion_vi_init_enabled: bool,
    /// After static VI init gives up, permit motion-VI to start from the
    /// configured running IMU biases. Intended for sequences that start in
    /// motion and therefore have no valid stationary window.
    motion_vi_init_after_static_give_up: bool,
    /// Start motion VI immediately from configured IMU biases. Explicit
    /// moving-start opt-in; does not wait for stationary init to terminate.
    motion_vi_init_from_configured_bias: bool,
    /// Minimum keyframes after static success or the opted-in give-up fallback
    /// before the motion-VI trigger is allowed to fire. Mirrors
    /// `MotionBasedViInitializerConfig::min_keyframes`.
    motion_vi_init_min_keyframes: usize,
    /// Minimum cumulative camera-centre translation (m) before the
    /// motion-VI trigger fires. Mirrors
    /// `MotionBasedViInitializerConfig::min_translation_meters`.
    motion_vi_init_min_translation_meters: f64,
    /// When set, enables VIBA2 (alternating scale-recovery outer loop)
    /// instead of VIBA1-only. Defaults `false` because EuRoC is a
    /// stereo-bootstrapped run (known scale) — flip to `true` for a
    /// monocular sanity check.
    motion_vi_init_recover_scale: bool,
    /// Opt into the local VI-BA sliding-window stage so the refined
    /// per-keyframe `(velocity, bias)` produced by motion-VI init has a
    /// downstream consumer. Off by default to preserve the existing
    /// baseline behaviour. Mirrors `OnlineSlamLocalBaConfig::default()`
    /// with `gravity_world` rebased onto `--gravity`.
    local_vi_ba_enabled: bool,
    /// Global A/B lever shared by local VI-BA and covisibility local BA.
    /// Experimental and off by default until a multi-sequence tracking-cliff
    /// A/B demonstrates a net win.
    observation_confidence_ba_enabled: bool,
    /// Schur-marginalize the outgoing navigation state into a dense FEJ prior
    /// on the next window anchor.
    local_vi_ba_marginalization: bool,
    /// Optional finite initialization uncertainty `(velocity, gyro bias,
    /// accel bias)` used by the first marginal prior.
    local_vi_ba_initial_prior_std_devs: Option<(f64, f64, f64)>,
    /// When `Some(v)`, runs the local VI-BA conditioning fallback:
    /// after the joint solve, if `final_cost / initial_cost > v`, the
    /// window is re-solved with biases gauge-frozen and the bias
    /// updates are discarded. `None` (the default) preserves the legacy
    /// always-update-biases behaviour.
    local_vi_ba_freeze_biases_above: Option<f64>,
    /// When `Some(v)`, rejects the entire local VI-BA writeback when
    /// the selected solve has `final_cost / initial_cost > v`. This is
    /// a stricter safety gate than `local_vi_ba_freeze_biases_above`:
    /// rejected passes return diagnostics but leave map poses,
    /// landmarks, velocities, and biases untouched.
    local_vi_ba_reject_writeback_above: Option<f64>,
    local_vi_ba_reject_final_imu_nis_per_dof_above: Option<f64>,
    /// When `Some(v)`, rejects the entire local VI-BA writeback when
    /// any refined in-window `||velocity_world|| > v`. This catches
    /// non-physical tight-VIO updates that can still reduce reprojection
    /// cost by injecting bad velocity state.
    local_vi_ba_reject_velocity_above_mps: Option<f64>,
    local_vi_ba_reject_pose_translation_above_meters: Option<f64>,
    local_vi_ba_reject_pose_rotation_above_degrees: Option<f64>,
    /// Enable the adaptive local VI-BA refined-velocity writeback gate.
    /// The gate derives a per-trigger velocity threshold from the local
    /// window's existing velocity state, pose-delta / IMU-`dt` finite
    /// differences, and IMU-predicted next-keyframe velocities, avoiding
    /// a raw scene-scale `m/s` threshold as the primary decision.
    local_vi_ba_adaptive_velocity_gate: bool,
    /// Adaptive local VI-BA velocity gate robust reference quantile.
    local_vi_ba_adaptive_velocity_quantile: f64,
    /// Adaptive local VI-BA velocity gate multiplier.
    local_vi_ba_adaptive_velocity_multiplier: f64,
    /// Adaptive local VI-BA velocity gate additive slack in m/s.
    local_vi_ba_adaptive_velocity_margin_mps: f64,
    /// Adaptive local VI-BA velocity gate lower bound in m/s.
    local_vi_ba_adaptive_velocity_min_mps: f64,
    /// Optional adaptive local VI-BA velocity gate upper bound in m/s.
    local_vi_ba_adaptive_velocity_max_mps: Option<f64>,
    /// Minimum finite local velocity references required before the
    /// adaptive local VI-BA velocity gate can reject writeback.
    local_vi_ba_adaptive_velocity_min_references: usize,
    /// When `Some(v)`, runs the motion-VI post-solve velocity sanity
    /// gate: if any per-keyframe `||velocity_world|| > v`, the inner LM
    /// result is rejected and the stage stays in `Waiting`. `None`
    /// (the default) preserves legacy behaviour. EuRoC V1-class indoor
    /// sequences run safely at `Some(10.0)`.
    motion_vi_init_max_velocity_mps: Option<f64>,
    /// Optional post-solve magnitude bound for every recovered gyro bias.
    motion_vi_init_max_gyro_bias_rad_s: Option<f64>,
    /// Optional post-solve magnitude bound for every recovered accel bias.
    motion_vi_init_max_accel_bias_mps2: Option<f64>,
    /// Optional upper bound on the final whitened IMU NIS per residual DoF.
    motion_vi_init_max_imu_nis_per_dof: Option<f64>,
    motion_vi_init_max_rotation_residual_rms_rad: Option<f64>,
    motion_vi_init_max_velocity_residual_rms_mps: Option<f64>,
    motion_vi_init_max_position_residual_rms_meters: Option<f64>,
    /// Optional staged bias-release schedule
    /// ([`visloc_rs::BiasReleaseSchedule`]): `Some` when either
    /// `--vi-bias-release-min-keyframes` or
    /// `--vi-bias-release-min-translation` is passed. The other knob falls
    /// back to the schedule's documented default (10 keyframes / 2.0 m)
    /// when only one flag is given.
    vi_bias_release_min_keyframes: Option<usize>,
    vi_bias_release_min_translation_meters: Option<f64>,
    /// Enable gravity-direction recovery
    /// ([`visloc_rs::MotionBasedViInitializerConfig::estimate_gravity`]):
    /// estimate the world-frame gravity vector from this window's IMU
    /// preintegration factors + fixed visual poses instead of trusting
    /// `--gravity` as ground truth. Off by default. See
    /// `docs/motion_based_vi_alignment.md`'s "Gravity-direction recovery"
    /// section for the motivating real-data diagnosis (a ~90-degree
    /// misalignment on a moving-start sequence where the static VI
    /// initializer never fires).
    motion_vi_init_estimate_gravity: bool,
    /// Override
    /// [`visloc_rs::MotionBasedViInitializerConfig::max_gravity_norm_deviation_ratio`]
    /// (default `0.3`). Ignored unless `motion_vi_init_estimate_gravity` is
    /// set.
    motion_vi_init_max_gravity_norm_deviation: Option<f64>,
    /// Enable gyro-bias recovery
    /// ([`visloc_rs::MotionBasedViInitializerConfig::estimate_gyro_bias`]):
    /// estimate the shared gyro bias from rotation-only alignment against
    /// this window's fixed visual poses, BEFORE gravity/velocity alignment
    /// and before the staged solve — the classical first inertial-init step
    /// ORB-SLAM3 / VINS-Mono run before gravity alignment. Off by default.
    /// See `docs/motion_based_vi_alignment.md`'s "Gyro-bias recovery"
    /// section for the motivating diagnosis: the final fitted IMU rotation
    /// residual RMS sat at 0.014-0.022 rad against the 0.01 gate,
    /// bit-identical with/without gravity estimation (the rotation residual
    /// is gravity-independent).
    motion_vi_init_estimate_gyro_bias: bool,
    /// When `Some(n)`, restricts descriptor matching during tracking to
    /// landmarks observed by the reference keyframe and up to `n`
    /// co-visible neighbour keyframes (ranked by shared-landmark count).
    /// `None` (the default) preserves legacy whole-map matching.
    covisibility_local_map_max_keyframes: Option<usize>,
    /// Minimum shared-landmark count required for a candidate keyframe
    /// to enter the covisibility-derived local map. Only used when
    /// `covisibility_local_map_max_keyframes` is set.
    covisibility_local_map_min_shared: usize,
    /// Opt into visual-only covisibility local BA inside
    /// `OnlineSlamPipeline`. This is distinct from the tracker-side
    /// covisibility local map: it runs a backend BA solve after newly
    /// applied keyframes, using co-visible keyframes and fixed boundary
    /// keyframes from the accumulated `VisualMap`. Off by default so
    /// baseline runs stay unchanged.
    covisibility_local_ba_enabled: bool,
    /// Enable sparse DROID-style temporal/proximity/stereo factor lifecycle.
    sparse_factor_graph_enabled: bool,
    /// Minimum map keyframe count before online covisibility local BA
    /// can run. Skips startup windows that cannot anchor a useful local
    /// solve.
    covisibility_local_ba_min_keyframes: usize,
    covisibility_local_ba_max_keyframes: Option<usize>,
    covisibility_local_ba_motion_vi_raw_activation: Option<(f64, f64, f64)>,
    covisibility_local_ba_max_seed_landmarks_for_activation: Option<usize>,
    /// Run online covisibility local BA after every N newly-applied
    /// keyframes. Values below 1 are rejected during argument parsing.
    covisibility_local_ba_trigger_every: usize,
    /// Maximum optimized co-visible neighbor keyframes, excluding the
    /// active keyframe.
    covisibility_local_ba_max_neighbor_keyframes: usize,
    /// Minimum shared landmarks for a keyframe to become an optimized
    /// covisibility-BA neighbor.
    covisibility_local_ba_min_shared: usize,
    /// Maximum fixed boundary keyframes that observe the selected local
    /// landmarks.
    covisibility_local_ba_max_boundary_keyframes: usize,
    /// Minimum local-landmark observations for a keyframe to become a
    /// fixed boundary keyframe.
    covisibility_local_ba_min_boundary_observations: usize,
    /// Optional lower fixed-boundary threshold used only when the primary
    /// boundary threshold produces no local BA landmarks.
    covisibility_local_ba_fallback_min_boundary_observations: Option<usize>,
    /// Optional cap on local landmarks passed to the BA solve.
    covisibility_local_ba_max_landmarks: Option<usize>,
    /// Minimum selected local-landmark observations on the active keyframe
    /// before online covisibility local BA is allowed to run.
    covisibility_local_ba_min_active_observations: usize,
    /// Optional post-BA reprojection threshold for outlier diagnostics.
    covisibility_local_ba_outlier_threshold_px: Option<f64>,
    /// Remove post-BA observations above
    /// `covisibility_local_ba_outlier_threshold_px`.
    covisibility_local_ba_remove_outliers: bool,
    /// Use calibrated non-rectified cam0/cam1 factors for stored stereo
    /// observations in covisibility local BA.
    covisibility_local_ba_general_stereo: bool,
    covisibility_local_ba_general_stereo_max_right_reprojection_px: Option<f64>,
    /// Reject covisibility BA map write-back when post-BA outlier
    /// observations exceed this fraction of selected observations.
    covisibility_local_ba_max_outlier_observation_ratio: Option<f64>,
    /// Optional pre-solve guard for large optimized windows with too few
    /// fixed boundary keyframes.
    covisibility_local_ba_boundary_support_min_optimized_keyframes: Option<usize>,
    /// Fixed-boundary keyframe floor used with the boundary-support guard.
    covisibility_local_ba_boundary_support_min_fixed_keyframes: usize,
    /// Reject covisibility BA map write-back when the fraction of solved
    /// optimized landmarks projecting behind an observing optimized camera
    /// exceeds this value. Off (`None`) by default.
    covisibility_local_ba_max_behind_camera_ratio: Option<f64>,
    /// Reject covisibility BA map write-back unless
    /// `fixed >= ceil(optimized * ratio)`. Off (`None`) by default.
    covisibility_local_ba_min_fixed_to_optimized_ratio: Option<f64>,
    /// Reject a solved covisibility BA update above this camera-centre shift.
    covisibility_local_ba_max_pose_translation_correction_m: Option<f64>,
    /// Reject a solved covisibility BA update above this rotation change.
    covisibility_local_ba_max_pose_rotation_correction_deg: Option<f64>,
    /// Optional gauge/global-anchoring pose-prior weight for optimized
    /// covisibility-BA keyframes. Off (`None`) by default.
    covisibility_local_ba_anchor_weight: Option<f64>,
    /// When `true`, replenishes the map beyond the stereo bootstrap by
    /// stereo-matching each frame's cam0 keypoints that tracking did NOT
    /// match to an existing landmark against a freshly loaded/undistorted
    /// cam1 image, triangulating the survivors, and staging them as
    /// `LandmarkCandidate`s for `process_frame`. Off by default so the
    /// map stays frozen at the bootstrap landmark count, preserving
    /// legacy behaviour.
    stereo_landmark_replenish: bool,
    /// Cap on new replenishment candidates built per frame. Only
    /// meaningful when `stereo_landmark_replenish` is set.
    stereo_landmark_replenish_max_per_frame: usize,
    /// Radius (px) around the reprojected anchor pixel within which a real
    /// detected anchor-keyframe keypoint must exist. Maps to
    /// [`StereoReplenishConfig::anchor_keypoint_match_radius_px`].
    stereo_landmark_replenish_anchor_match_radius_px: Option<f64>,
    /// Optional descriptor-distance gate on the associated anchor keypoint.
    /// Maps to [`StereoReplenishConfig::anchor_keypoint_max_descriptor_distance`].
    stereo_landmark_replenish_anchor_max_descriptor_distance: Option<f32>,
    /// Radius (px) for geometric duplicate suppression against the anchor
    /// keyframe's own landmarks. Maps to
    /// [`StereoReplenishConfig::duplicate_suppression_radius_px`].
    stereo_landmark_replenish_duplicate_radius_px: Option<f64>,
    /// Minimum anchor↔current parallax (degrees). Maps to
    /// [`StereoReplenishConfig::min_parallax_deg`].
    stereo_landmark_replenish_min_parallax_deg: Option<f64>,
    /// Minimum triangulated depth (m). Maps to
    /// [`StereoReplenishConfig::min_depth_meters`].
    stereo_landmark_replenish_min_depth_meters: Option<f64>,
    /// Maximum triangulated depth (m). Maps to
    /// [`StereoReplenishConfig::max_depth_meters`].
    stereo_landmark_replenish_max_depth_meters: Option<f64>,
    /// When `Some(d)`, rejects a tracked frame whose PnP camera-centre
    /// drifts more than `d` metres from the motion-model pose prior
    /// (`ConstantPoseMotionModel` returns the last successful pose, so
    /// this effectively becomes a per-frame translation cap). Catches
    /// catastrophic PnP outliers that teleport the trajectory to
    /// unphysical positions. `None` (the default) preserves legacy
    /// always-accept behaviour. For EuRoC V1-class indoor sequences
    /// (~0.5 m/s walking pace, 50 ms cam0 cadence) values in the
    /// `0.2`–`1.0` m range are reasonable.
    max_pose_jump_meters: Option<f64>,
    /// Allow a bounded pose-prior gate widening for PnP solutions with at
    /// least 100 inliers, 0.6 inlier ratio, and 3 px mean reprojection error.
    pose_prior_visual_override: bool,
    /// When `true`, scale `--max-pose-jump-meters` by the number of frames
    /// elapsed since the last successful track (capped at
    /// `--pose-jump-gap-scaling-max-multiplier`, floored at 1) before
    /// comparing. Addresses a permanent-tracking-loss failure mode
    /// observed on EuRoC MH_01: a single tracking failure freezes the
    /// motion-model pose prior (`ConstantPoseMotionModel` returns the last
    /// successful pose), and a subsequent frame with a genuinely good PnP
    /// solution gets rejected by the fixed-radius gate because the prior
    /// is stale, not because the solution is bad. Scaling the gate by the
    /// gap lets a good post-gap solution through while still catching
    /// same-gap outliers at the original radius. Off by default; no effect
    /// unless `--max-pose-jump-meters` is also set.
    pose_jump_gap_scaling: bool,
    /// Companion to `--pose-jump-gap-scaling`: caps the gap-scaling
    /// multiplier so an extended tracking outage doesn't inflate the gate
    /// to an unbounded radius. Defaults to `10`.
    pose_jump_gap_scaling_max_multiplier: usize,
    /// Minimum PnP inlier count required after localization. Defaults
    /// to `0`, preserving legacy behaviour. Useful when relaxing
    /// pose-prior gates: frames with only a handful of correspondences
    /// should not refresh the tracker or enter the map.
    tracking_min_inliers: usize,
    /// Minimum PnP inlier ratio required after localization. Defaults
    /// to `0.0`, preserving legacy behaviour. Combine with
    /// `tracking_min_inliers` to separate visual-rescue frames from
    /// near-lost outliers.
    tracking_min_inlier_ratio: f64,
    /// Optional mean reprojection error ceiling for accepted
    /// localizations. Defaults to `None`.
    tracking_max_reprojection_error: Option<f64>,
    /// Override the tracker's PnP RANSAC reprojection-error inlier
    /// threshold (`LocalizationConfig::reprojection_threshold`,
    /// default 4.0 px). Larger values admit more correspondences as
    /// inliers and accept more frames at marginal accuracy cost; the
    /// Phase-26 #3a use case is allowing SuperPoint trajectories on
    /// V-class to track further past the universal cliff (where the
    /// stricter default refuses post-cliff frames that the wider
    /// gate would still trust). `None` preserves the default.
    pnp_reprojection_threshold_px: Option<f64>,
    /// When `true`, hand the motion-model pose prior to the PnP RANSAC
    /// as a warm-start hypothesis (ORB-SLAM3-style motion-only BA seed).
    /// Random samples must beat the prior's inlier count to win, so a
    /// well-aligned prior short-circuits RANSAC on hard scenes (e.g.,
    /// faster motion where standard PnP would diverge) while a
    /// misaligned prior gracefully degrades to the standard random
    /// search. Combine with `--max-pose-jump-meters` to gate
    /// post-hoc on the warm-start result. Off by default.
    pnp_pose_prior_warm_start: bool,
    /// Enable ORB-SLAM3-style projection-guided tracking: when a pose
    /// prior is available, restrict descriptor matching to a per-landmark
    /// projection window instead of an appearance-global search, with a
    /// widen-retry ladder that falls back to today's appearance-global
    /// path if every projection attempt fails, plus a post-hoc local-map
    /// refinement pass. Off by default; enabling it can only ADD tracking
    /// chances since the widen-retry ladder's last rung IS today's
    /// unmodified appearance-global path.
    projection_guided_tracking: bool,
    /// Initial per-landmark projection-window radius, in pixels. Only
    /// meaningful when `--projection-guided-tracking` is set.
    projection_search_radius_px: f64,
    /// Multiplier applied to the projection-window radius on each
    /// widen-retry attempt after a stage-1 projection attempt fails.
    projection_widen_factor: f64,
    /// Maximum number of widen-retry attempts after the initial
    /// projection-window attempt. Total projection attempts per frame is
    /// `1 + projection_max_widen_retries`.
    projection_max_widen_retries: u32,
    /// Reverse descriptor ambiguity ratio across projected landmarks that
    /// compete for one query keypoint. `None` restores the legacy
    /// deterministic first-wins behavior.
    projection_query_landmark_distance_ratio: Option<f32>,
    /// Disable stage-3 local-map refinement (on by default when
    /// `--projection-guided-tracking` is set): re-projects the
    /// covisibility local map with the estimated pose, harvests
    /// additional correspondences, and re-optimizes the pose, accepting
    /// the refined result only when its inlier count does not decrease.
    projection_no_local_map_refinement: bool,
    /// Per-landmark projection-window radius, in pixels, used by the
    /// local-map refinement stage.
    projection_refinement_search_radius_px: f64,
    /// Alternating fresh-correspondence / pose-update rounds in stage 3.
    projection_refinement_iterations: u32,
    /// Projection radius multiplier after every refinement round.
    projection_refinement_radius_shrink_factor: f64,
    /// Permit fresh rounds to reassign a query keypoint to another landmark.
    projection_refinement_reassign_correspondences: bool,
    /// Minimum retained fraction of current inlier query/landmark pairs.
    projection_refinement_min_inlier_pair_retention_ratio: f64,
    /// Optional per-round camera-centre correction trust region.
    projection_refinement_max_translation_correction_m: Option<f64>,
    /// Optional per-round rotation correction trust region, in degrees.
    projection_refinement_max_rotation_correction_deg: Option<f64>,
    /// Motion model fed to the tracker. `pose` (default) returns the
    /// last successful pose as the prior (`ConstantPoseMotionModel`);
    /// `velocity` extrapolates the last 2 successful poses to predict
    /// where the body will be at this frame
    /// (`ConstantVelocityMotionModel`), or **integrate the raw inter-
    /// frame IMU samples through a strapdown predictor**
    /// (`ImuPredictiveMotionModel`). The velocity model is a strict
    /// superset of constant-pose — it falls back to the constant-pose
    /// model before 2 successes have accumulated — and the IMU
    /// predictor is the strictest of the three (uses gyro / accel to
    /// extrapolate the previous pose forward by the inter-frame Δt).
    /// `imu` is the appropriate choice for sequences with non-trivial
    /// linear / angular acceleration (e.g., EuRoC MH_01 takeoff),
    /// where constant-velocity breaks down. All three integrate
    /// cleanly with `--pnp-pose-prior-warm-start`, because the
    /// warm-start is only as good as the prior is predictive.
    motion_model: MotionModelKind,
    /// When `true`, pass cam0's published `T_BS` (body-from-sensor)
    /// extrinsic into `ImuPredictiveMotionModel` so the strapdown
    /// integration runs in body frame and the input/output stay
    /// camera-pose; **also** wire the per-frame body-velocity update
    /// (finite-difference of two successive successful poses) so the
    /// integrator's initial velocity tracks the body's motion instead
    /// of being pinned at zero. Off by default because in the absence
    /// of a downstream VI-BA refining velocity, the visual tracker's
    /// per-frame noise leaks into the finite-difference velocity and
    /// the resulting (geometrically correct but noisy) prior leads to
    /// a higher rigid-ATE than the `body == camera` approximation —
    /// the trade-off is a perfect metric-scale recovery on
    /// accelerating sequences (MH_01: scale 1.001 vs 1.112). The flag
    /// has no effect when `--motion-model` is not `imu`.
    imu_extrinsic_from_cam0: bool,
    /// When `true`, set `ImuPredictiveMotionModelConfig
    /// ::carry_forward_velocity_world` so the motion model commits the
    /// post-strapdown `v_w` back into its initial-velocity slot during
    /// `observe()`. Without it, every per-frame `predict_pose` re-seeds
    /// from the last VI-BA mirror (KF time) — so on frames between
    /// mirrors the predicted velocity does not advance. Off by default;
    /// no effect when `--motion-model` is not `imu`.
    imu_motion_model_carry_forward_velocity: bool,
    /// When `true`, wrap the default `BruteForceMatcher` (Lowe ratio
    /// 0.8) in a `CrossCheckMatcher`, keeping only query↔train pairs
    /// where each side picks the other as its single best match. Off
    /// by default — the existing matcher uses the ratio test only.
    /// Cross-check is a standard ORB-SLAM-style filter that raises the
    /// per-frame inlier ratio at the cost of ~2× matcher CPU. The
    /// motivation is to register more keyframes through the strict
    /// `--max-pose-jump-meters 0.2` gate so the local-VI-BA chain (the
    /// Phase-9 mirror) actually fires on EuRoC.
    cross_check_matcher: bool,
    /// When `true`, replace the default `BruteForceMatcher` (and any
    /// `--cross-check-matcher` wrap) with `MutualSoftmaxMatcher`,
    /// LightGlue-style temperature-scaled mutual-softmax over the
    /// full cosine-similarity matrix. Defaults `temperature = 20.0`,
    /// `min_confidence = 0.2`. Aimed at cliff-region cross-attitude
    /// matching where the cross-check filter alone leaves SuperPoint's
    /// signal under-exploited; mutually exclusive with
    /// `--cross-check-matcher`. Off by default.
    mutual_softmax_matcher: bool,
    /// Selects the feature extractor backing the per-frame descriptor
    /// stream. `corner` (the default) is the existing
    /// `CornerFeatureExtractor` with raw patch descriptors. `hog` is
    /// `HogLikeFeatureExtractor` — a unit-norm 128-D HOG/SIFT-flavored
    /// descriptor that raises descriptor signal-to-noise at the cost
    /// of ~4× CPU per keypoint. Motivation: EuRoC's keyframe-
    /// registration floor at the strict pose-jump gate is dominated by
    /// descriptor match recall, not match-filter strictness, so a
    /// stronger descriptor is the natural next lever after Phase-10's
    /// `--cross-check-matcher` proved precision-only doesn't help.
    feature_extractor: FeatureExtractorKind,
    /// Companion to `feature_extractor = hog`: maximum keypoints
    /// retained per frame from the HOG corner-response NMS. Larger
    /// numbers favour recall at the cost of CPU; smaller numbers
    /// favour speed.
    hog_max_features: usize,
    /// Companion to `feature_extractor = hog`: minimum corner response
    /// score to admit a keypoint candidate (before NMS). Lower numbers
    /// admit more (often weaker) candidates.
    hog_min_corner_score: f32,
    /// Companion to `feature_extractor = hog`: enable SIFT-style
    /// dominant-gradient-orientation alignment of the HOG bins (off by
    /// default — adds variance without clear gain on a forward-driving
    /// camera, but worth trying on rotation-heavy EuRoC drone takeoffs).
    hog_orient: bool,
    /// When `Some(m)`, overrides `KeyframePolicyConfig.min_translation`
    /// (the cumulative-from-last-keyframe translation a successful
    /// frame must clear to register as a new keyframe). The library
    /// default is `1.0 m`, calibrated for KITTI-class driving cadence;
    /// on EuRoC's slower indoor / takeoff motion plus the strict
    /// `--max-pose-jump-meters 0.2` gate, `1.0 m` is too coarse to
    /// register more than the seed keyframe in 400 frames. Values
    /// around `0.05–0.2 m` register a useful keyframe stream on EuRoC
    /// without flooding the map.
    keyframe_min_translation: Option<f64>,
    /// Override `KeyframePolicyConfig.min_frame_id_gap`. Lower values
    /// allow earlier rescue keyframes when tracking quality drops soon
    /// after a promotion; `None` preserves the library default.
    keyframe_min_frame_gap: Option<u64>,
    /// Force a keyframe after this many frames when tracking quality passes,
    /// even if estimated translation remains below the metric threshold.
    keyframe_max_frame_gap: Option<u64>,
    /// When `Some(r)`, also promotes a keyframe after the normal
    /// frame-id gap if the current PnP inlier count drops to `r` times
    /// the last keyframe's tracked-landmark count. This mirrors the
    /// ORB-SLAM-style "tracked local-map points dropped" trigger while
    /// keeping the existing metric translation threshold for A/B.
    keyframe_tracked_landmark_ratio: Option<f64>,
    /// Reference/current-count floor for `keyframe_tracked_landmark_ratio`.
    /// Prevents sparse startup frames or nearly-lost frames from making
    /// the ratio trigger look artificially severe.
    keyframe_min_tracked_landmarks_for_ratio: usize,
    /// When `Some(n)`, rejects promoting a frame to a keyframe (even
    /// after it clears the frame-id-gap and translation gates) if its PnP
    /// inlier count is below `n`. Targets the EuRoC MH_01 frame-1096
    /// failure mode where a garbage localization (4 inliers) that
    /// survives the pose-prior gate gets promoted to a keyframe and
    /// poisons the local map / covisibility graph for subsequent frames.
    /// `None` (the default) preserves legacy always-promote behaviour.
    keyframe_min_inliers: Option<usize>,
    /// Companion to `--keyframe-min-inliers`: also requires the PnP
    /// inlier ratio to be at least this value for a keyframe promotion to
    /// go through. `None` (the default) preserves legacy behaviour.
    keyframe_min_inlier_ratio: Option<f64>,
    /// Override `VisualInertialInitializerConfig.min_samples`; default
    /// `50`. Lowering accelerates VI-init promotion so more KFs
    /// register POST-init and feed the local-VI-BA chain.
    vi_init_min_samples: Option<usize>,
    /// Override `VisualInertialInitializerConfig.min_stationary_window_seconds`;
    /// default `0.5 s`. Pairs with `--vi-init-min-samples` to bring
    /// VI-init forward.
    vi_init_min_stationary_window_seconds: Option<f64>,
    /// When `true`, sets `OnlineSlamConfig.keep_pre_promotion_imu_factors
    /// = true` so IMU factors staged on keyframes registered BEFORE the
    /// auto-bootstrap stage promotes still flow downstream (into the
    /// local-VI-BA factor history). The factors carry placeholder bias
    /// linearisations; the BA's Gauss-Newton iterations are expected
    /// to absorb the resulting bias error. Empirically on EuRoC this
    /// is the lever that unblocks local-VI-BA's per-trigger
    /// keyframe-state mirror — the strict (default) gate leaves the
    /// chain at 1 trigger per 400 frames, the relaxed gate lifts it
    /// to ~7. See `docs/motion_based_vi_alignment.md` §Phase-13.
    keep_pre_promotion_imu_factors: bool,
    /// When `Some((g, a))`, sets
    /// `OnlineSlamLocalBaConfig.relinearise_imu_factor_bias_thresholds`
    /// so banked IMU factors get re-linearised in-place before each BA
    /// pass if their stored `bias_*_linearisation` drifts more than `g`
    /// rad/s (gyro) or `a` m/s² (accel) from the per-keyframe state
    /// estimate. Pairs with `--keep-pre-promotion-imu-factors`: without
    /// the re-linearisation, banked pre-promotion factors carry their
    /// placeholder zero bias_linearisation through the post-promotion
    /// BA, and the linear bias-correction approximation breaks down on
    /// non-trivial biases — producing unphysical mirror velocities even
    /// when rigid ATE improves. See `docs/motion_based_vi_alignment.md`
    /// §Phase-14.
    relinearise_imu_factor_bias_thresholds: Option<(f64, f64)>,
    /// Directory holding pre-exported SuperPoint mono features
    /// (`frame_NNNNNN_features.txt` from
    /// `scripts/export_superpoint_lightglue.py --mono-dir <cam0_dir>`)
    /// to replay during the demo run. Required when
    /// `--feature-extractor superpoint-offline` is selected; otherwise
    /// ignored.
    superpoint_features_dir: Option<PathBuf>,
    /// Optional companion directory holding pre-exported SuperPoint
    /// cam1 features. When provided alongside
    /// `--superpoint-features-dir <cam0_dir>` and
    /// `--feature-extractor superpoint-offline`, the offline extractor
    /// loads both feature streams and `--stereo-bootstrap` is
    /// permitted. The stereo bootstrap path triangulates cam0 ↔ cam1
    /// SuperPoint matches via the cam0/cam1 extrinsics, producing
    /// metric-depth landmarks that replace the fixed `bootstrap_depth`
    /// back-projection. Phase-17 lever for the Phase-15 bootstrap-
    /// depth bottleneck.
    superpoint_cam1_features_dir: Option<PathBuf>,
    /// Phase-27: path to a SuperPoint ONNX model file. Required when
    /// `--feature-extractor superpoint-onnx` is selected. The model
    /// must follow the LightGlue-ONNX-style I/O contract documented
    /// in `docs/superpoint_onnx_runtime_plan.md`: input `image:
    /// (1, 1, H, W) f32`, outputs `keypoints (N, 2) i64`, `scores
    /// (N,) f32`, `descriptors (256, N) f32` (or `(N, 256)`). The
    /// extractor only runs when `visloc-vision` was built with
    /// `--features onnx-inference`; otherwise the first `extract()`
    /// call fails with `FeatureDisabled`.
    superpoint_onnx_model: Option<PathBuf>,
    /// ONNX execution-provider policy. `cuda` is strict and aborts at model
    /// loading when CUDA cannot be registered, preventing an evaluation from
    /// silently reporting CPU inference as a GPU run.
    superpoint_onnx_backend: SuperPointOnnxBackendArg,
    /// Export one L2-normalized mean local descriptor per processed cam0
    /// frame to `frame_appearance_descriptors.csv`. This is a diagnostic
    /// input for offline retrieval-candidate recall scripts; it does not
    /// affect tracking, mapping, or recovery.
    export_frame_appearance_descriptors: bool,
    /// Phase-16 lever. When `true`, sets
    /// `OnlineSlamLocalBaConfig.run_at_vi_init_promotion = true` so
    /// the local-VI-BA pass fires at the same `process_frame` that
    /// promotes VI-init, consuming the banked pre-promotion factors
    /// without waiting for the next keyframe registration. Useful
    /// when the visual tracker is fragile post-promotion so the next
    /// keyframe arrives late or never. Default `false` preserves
    /// Phase-13/14 cadence.
    run_local_vi_ba_at_vi_init_promotion: bool,
    /// Phase-23 #1 lever. When `true`, enables the
    /// relocalization-on-tracker-death stage in
    /// [`OnlineSlamConfig::relocalization`]. On every frame whose
    /// primary tracking fails, the pipeline runs a separate
    /// [`LocalizationPipeline`] against the full map and overrides
    /// the tracker's state when the recovered solution clears
    /// `--relocalization-min-inliers` / `--relocalization-min-inlier-ratio`
    /// / `--relocalization-max-reprojection-error`. Default `false`
    /// preserves the universal-cliff behaviour from Phase-21.
    relocalization_enabled: bool,
    /// Acceptance threshold for the relocalization recovery PnP solve.
    relocalization_min_inliers: usize,
    /// Acceptance threshold for the relocalization recovery PnP solve.
    relocalization_min_inlier_ratio: f64,
    /// Optional acceptance threshold for the relocalization recovery
    /// PnP solve.
    relocalization_max_reprojection_error: Option<f64>,
    /// Phase-23 #4 lever. When `--motion-model adaptive-imu-pose` is
    /// selected, this is the number of consecutive failed-tracking
    /// frames under the IMU branch that triggers a switch to the
    /// constant-pose branch. Default `2` reacts fast at the cliff
    /// transition; raise to dampen oscillation on noisy regimes.
    adaptive_motion_failures_to_switch_to_pose: usize,
    /// Phase-23 #4 lever. Number of consecutive successful frames
    /// under the constant-pose branch that triggers a switch back to
    /// the IMU branch. Default `5` biases for staying in the
    /// (cliff-survival) pose mode once it has fired.
    adaptive_motion_successes_to_switch_to_imu: usize,
    /// Phase-24 / Phase-25 lever. Selects which
    /// [`ImuVelocityRefreshPolicy`] the adaptive wrapper uses at every
    /// switch-back-to-IMU event. `FiniteDifference` (default,
    /// Phase-24 behavior) recomputes the IMU's `velocity_world` from
    /// a single finite-difference of the two most recent successful
    /// visual poses. `ZeroReset` (Phase-25 #1) zeroes the velocity
    /// instead. `ThreePoseSmoother` (Phase-25 #2) averages two
    /// finite-differences over the three most recent successful
    /// poses. `None` recovers the Phase-23 #4 no-refresh behavior for
    /// A/B testing. Configurable via
    /// `--adaptive-motion-refresh-policy`.
    adaptive_motion_imu_velocity_refresh_policy: ImuVelocityRefreshPolicy,
    /// Phase-23 #2 lever. When `true`, the bootstrap map drops every
    /// keypoint that did NOT receive a stereo-triangulated cam0↔cam1
    /// depth, refusing to fall back to `--bootstrap-depth`. The map
    /// is smaller (only the 30–43 % of cam0 keypoints that survived
    /// stereo matching) but every landmark has a real metric depth.
    /// Empirically, the Phase-23 #1 sweep showed that cross-attitude
    /// HOG descriptor mismatch dominates the recovery-PnP failure
    /// mode; dropping the stale 4-m-depth fallback landmarks reduces
    /// the false-positive matches that the recovery localizer would
    /// otherwise have to filter out. Default `false` preserves the
    /// legacy mixed-depth bootstrap.
    stereo_bootstrap_strict: bool,
    /// Phase-23 #1b lever. When `Some(radius_m)`, the recovery PnP
    /// uses the tracker's motion-model pose prior as a warm-start
    /// hypothesis AND filters the candidate landmark set to those
    /// within `radius_m` of the prior's camera centre. Empirically on
    /// EuRoC the Phase-23 #1 sweep showed full-map BruteForce PnP
    /// without a pose prior accepts < 0.3 % of recovery attempts at
    /// the strict gate; threading the prior in short-circuits the
    /// matcher to the local landmark set + seeds RANSAC, both of
    /// which should lift recovery quality. `None` (default) preserves
    /// the no-prior global path from Phase-23 #1.
    relocalization_pose_prior_radius_meters: Option<f64>,
    /// Phase-26 #4a active-frontier submap selection. When
    /// `Some(window)`, the recovery PnP's descriptor store is
    /// restricted to landmarks observed by any of the most-recent
    /// `window` keyframes in the map. Targets the Phase-26 #2
    /// V1_01 false-positive failure mode (full-map recovery
    /// accepts wrong-scale solutions because the candidate set
    /// spans the whole map). `None` preserves Phase-23 #1
    /// full-map behaviour.
    relocalization_recent_keyframe_window: Option<usize>,
    /// Phase-26 #4b post-acceptance IMU sanity check. When
    /// `Some(max_translation_m)`, recoveries that pass the
    /// inlier-count / inlier-ratio / reprojection-error gates are
    /// further rejected if the recovered camera centre is more
    /// than `max_translation_m` from the tracker's motion-model
    /// prediction. Filters wrong-scale recoveries identified by
    /// Phase-26 #2 diagnosis. `None` preserves Phase-23 #1
    /// no-IMU-sanity-check behaviour.
    relocalization_max_translation_from_imu_prediction_meters: Option<f64>,
    /// Minimum frame-id gap between relocalization attempts. Default
    /// `1` preserves the existing every-failed-frame behaviour.
    relocalization_attempt_interval_frames: u64,
    /// Optional cap on consecutive failed relocalization attempts while
    /// the primary tracker remains lost. `None` keeps the legacy
    /// unbounded retry behaviour.
    relocalization_max_consecutive_failed_attempts: Option<u64>,
    /// Optional pose-continuity gate on accepted relocalization
    /// candidates. When `Some(max_m_per_frame)`, the recovered camera
    /// centre must stay within this translation-per-frame budget from
    /// the last successful tracker pose. `None` preserves the existing
    /// acceptance behaviour.
    relocalization_max_translation_per_frame_from_last_success_meters: Option<f64>,
    /// Optional lower bound for recovery inlier median-depth ratio vs
    /// the last successful pose, measured on the same inlier landmarks.
    relocalization_min_inlier_depth_median_ratio_to_last_success: Option<f64>,
    /// Optional upper bound for recovery inlier median-depth ratio vs
    /// the last successful pose.
    relocalization_max_inlier_depth_median_ratio_to_last_success: Option<f64>,
    /// Enable covisibility-local recovery descriptor-store selection
    /// and cap the number of neighbor keyframes. `None` keeps the
    /// existing full-map / recent-window recovery store.
    relocalization_covisibility_max_keyframes: Option<usize>,
    /// Minimum shared landmarks for a neighbor keyframe to enter the
    /// recovery covisibility store.
    relocalization_covisibility_min_shared: usize,
    /// Minimum descriptor count required to use the recovery
    /// covisibility store; otherwise the demo falls back to the
    /// broader recovery descriptor store.
    relocalization_covisibility_min_landmarks: usize,
    /// Retry with the broader full-map / recent-window recovery store
    /// when the covisibility-local first pass fails the acceptance
    /// gates.
    relocalization_covisibility_broader_fallback: bool,
    /// Minimum frame-id gap between broader descriptor-store retries.
    relocalization_covisibility_broader_fallback_interval_frames: u64,
    /// Also run the broader store when the covisibility-local first
    /// pass succeeds, then keep the accepted result with the stronger
    /// inlier / reprojection score.
    relocalization_covisibility_compare_broader_store: bool,
    /// Enable appearance-retrieval recovery descriptor-store selection
    /// and cap the number of retrieved keyframes. `None` keeps this
    /// policy disabled.
    relocalization_appearance_max_keyframes: Option<usize>,
    /// Optional cap on ranked appearance candidates written to
    /// `relocalization_appearance_candidates.csv`. This can be higher
    /// than `relocalization_appearance_max_keyframes` to evaluate
    /// retrieval recall@K without increasing recovery-PnP cost.
    relocalization_appearance_candidate_log_limit: Option<usize>,
    /// Minimum mean-descriptor cosine similarity for a retrieved
    /// keyframe to seed recovery.
    relocalization_appearance_min_similarity: f32,
    /// Exclude keyframes within this frame-id gap from appearance
    /// retrieval.
    relocalization_appearance_exclude_recent_frame_gap: u64,
    /// Minimum descriptor count required to use the appearance-retrieval
    /// recovery store.
    relocalization_appearance_min_landmarks: usize,
    /// Retry with the broader full-map / recent-window recovery store
    /// when the appearance first pass fails the acceptance gates.
    relocalization_appearance_broader_fallback: bool,
    /// Minimum frame-id gap between broader descriptor-store retries
    /// after an appearance first pass.
    relocalization_appearance_broader_fallback_interval_frames: u64,
    /// Also run the broader store when the appearance first pass
    /// succeeds, then keep the accepted result with the stronger score.
    relocalization_appearance_compare_broader_store: bool,
    /// Number of consecutive recovery hypotheses required before the
    /// tracker accepts relocalization. `1` preserves immediate accept.
    relocalization_confirmation_required_recoveries: usize,
    /// Optional max translation per frame between consecutive recovery
    /// hypotheses inside the confirmation window.
    relocalization_confirmation_max_translation_per_frame_meters: Option<f64>,
    /// When `Some(n)`, re-seed tracking from the last accepted pose and the
    /// current calibrated stereo pair after `n` consecutive lost frames once
    /// relocalization (if enabled) has not recovered. This is causal and
    /// GT-free. By default it continues in the existing gauge;
    /// `rebootstrap_independent_submap` starts a fresh Atlas submap instead.
    rebootstrap_after_lost_frames: Option<usize>,
    /// Minimum frame gap between accepted re-bootstraps. Default `60`.
    rebootstrap_cooldown_frames: usize,
    /// Start every accepted stereo re-bootstrap in a fresh local gauge and
    /// retain the previous map in a [`MapAtlas`]. Until a verified cross-map
    /// SE(3) bridge is available, local poses are exported only to
    /// `submap_trajectory.csv` and never mixed into the global trajectory.
    rebootstrap_independent_submap: bool,
    /// Retained current/boundary source keyframes tried per Atlas-level
    /// broader recovery cycle.
    atlas_broader_recovery_max_source_keyframes: usize,
    /// Run the more expensive retained-keyframe Atlas search every N logical
    /// bridge attempts. Other attempts use only the current keyframe.
    atlas_broader_recovery_interval_attempts: usize,
    /// Opt into the online loop-closure + pose-graph refinement stage
    /// (`OnlineSlamConfig::pose_graph_refinement`). When `true`, the
    /// pipeline mirrors registered keyframe poses into a running pose
    /// graph, verifies every `detect_loop_closure_candidates` output with
    /// an essential-matrix verifier, and periodically re-solves the graph
    /// and writes the optimised poses back into the map. Off by default so
    /// baseline runs stay byte-identical.
    pose_graph_refinement_enabled: bool,
    /// Minimum newly-verified loop-closure constraints that must
    /// accumulate before a fresh pose-graph solve fires. Mirrors
    /// `OnlineSlamLoopClosureRefinementConfig::trigger_every_new_constraints`.
    /// Defaults to `1`. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pose_graph_refinement_trigger_every: usize,
    /// Optional fixed isotropic loop-edge weight. `None` (default) preserves
    /// the legacy raw-PnP-inlier-count weight. A value such as `1.0` prevents
    /// correspondence count from being mistaken for calibrated inverse pose
    /// covariance relative to unit-weight sequential edges.
    pose_graph_refinement_fixed_loop_edge_weight: Option<f64>,
    /// Use covariance-aware anisotropic 6×6 information for PnP loop edges.
    pose_graph_refinement_loop_pose_information: bool,
    /// Spectral cap applied to covariance-derived loop/sequential information.
    pose_graph_refinement_loop_pose_information_max_eigenvalue: f64,
    /// Relative strength applied to loop matrices only; sequential PnP edge
    /// information remains unchanged.
    pose_graph_refinement_loop_pose_information_loop_edge_scale: f64,
    /// Enable the Graduated Non-Convexity robust back-end solve
    /// (`OnlineSlamLoopClosureRefinementConfig::gnc`, using
    /// `gnc::GncConfig::default()`) instead of the plain iterative
    /// M-estimator solve. Off by default. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pose_graph_refinement_gnc: bool,
    /// Enable the Pairwise Consistency Maximization front-end screen
    /// (`OnlineSlamLoopClosureRefinementConfig::pcm`, using
    /// `pcm::PcmConfig::default()`). Off by default. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pose_graph_refinement_pcm: bool,
    /// Disable PCM's additional single-loop-vs-odometry pre-filter and rely on
    /// pairwise consistency between verified loop measurements. This matches
    /// the paper's drift-tolerant consistency-set test; the legacy individual
    /// gate remains enabled by default.
    pose_graph_refinement_pcm_require_individual: bool,
    /// Optional covariance-based chi-squared gate on verified loop
    /// closures. Mirrors
    /// `OnlineSlamLoopClosureRefinementConfig::covariance_gate`. `None`
    /// (default) applies no metric gate. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pose_graph_refinement_covariance_gate: Option<f64>,
    /// Which geometric verifier the refinement stage runs on loop-closure
    /// candidates (`OnlineSlamLoopClosureRefinementConfig::verifier`).
    /// `"essential"` (default) keeps the original scale-free two-view
    /// verifier, whose accepted constraints all carry the same
    /// `default_translation_scale` translation regardless of the true
    /// baseline. `"pnp"` verifies on 2D-3D correspondences against the map's
    /// triangulated landmarks instead, so accepted constraints carry the
    /// metric relative translation. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pose_graph_refinement_verifier: LoopRefinementVerifierKind,
    /// Opt into the appearance-based long-range loop candidate source
    /// (`OnlineSlamLoopClosureRefinementConfig::appearance_candidates`).
    /// Diagnoses the short-range hypothesis for why PnP-verified loop
    /// closures don't improve ATE: the default shared-landmark detector can
    /// only propose candidates that still reference the SAME map landmark
    /// ids the current frame's inliers carry, which drift re-triangulation
    /// makes inherently short-range. This stream instead ranks past
    /// keyframes by appearance (independent of shared ids) and PnP-verifies
    /// them via descriptor-matched 2D-3D correspondences. Off by default so
    /// baseline runs stay byte-identical. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pose_graph_refinement_appearance_loops: bool,
    /// Minimum keyframe-id gap for an appearance candidate (mirrors
    /// `LoopAppearanceCandidateConfig::min_keyframe_id_gap`). Defaults to
    /// `150` — comfortably above the shared-landmark detector's span, so
    /// this stream only ever proposes genuinely long-range loops. Only
    /// meaningful when `pose_graph_refinement_appearance_loops` is set.
    pose_graph_refinement_appearance_min_gap: u64,
    /// Maximum number of appearance-ranked candidate keyframes verified per
    /// frame (mirrors `LoopAppearanceCandidateConfig::max_candidates_per_frame`).
    /// Defaults to `3`. Only meaningful when
    /// `pose_graph_refinement_appearance_loops` is set.
    pose_graph_refinement_appearance_max_candidates: usize,
    /// Minimum PnP RANSAC inlier count an appearance candidate must produce
    /// to be admitted (mirrors `LoopAppearanceCandidateConfig::pnp_verifier`'s
    /// `min_inliers`). Defaults to `30` — higher than the shared-landmark
    /// PnP path's default, since a false long-range loop closure is
    /// catastrophic. Only meaningful when
    /// `pose_graph_refinement_appearance_loops` is set.
    pose_graph_refinement_appearance_min_inliers: usize,
    /// Separate current keyframes required to confirm one covisible region.
    pose_graph_refinement_appearance_confirmation_keyframes: usize,
    /// Consecutive keyframe misses tolerated before a pending region resets.
    pose_graph_refinement_appearance_confirmation_max_misses: usize,
    /// Optional post-PnP projection-window rematch radius in pixels.
    pose_graph_refinement_appearance_projection_radius_px: Option<f64>,
    /// Minimum projection-guided matches required for refined PnP.
    pose_graph_refinement_appearance_projection_min_matches: usize,
    /// Persist accepted appearance-PnP inliers as cross-loop observations.
    pose_graph_refinement_fuse_loop_observations: bool,
    /// Run covisibility welding BA after a solved, fused loop.
    pose_graph_refinement_loop_welding_ba: bool,
    /// Propagate each solved keyframe's pose correction to its anchored
    /// landmarks and to the tracker's continuation state
    /// (`OnlineSlamLoopClosureRefinementConfig::propagate_corrections`).
    /// Off by default so baseline pose-graph-refinement runs stay
    /// byte-identical (only keyframe poses move on write-back). Only
    /// meaningful when `pose_graph_refinement_enabled` is set.
    pose_graph_refinement_propagate: bool,
    /// Which pose-graph solver backs the periodic PGO trigger
    /// (`OnlineSlamLoopClosureRefinementConfig::solver`). `"se3"`
    /// (default) keeps the original rigid solve, byte-identical to
    /// today's behaviour. `"sim3"` opts into the `Sim(3)` pose graph
    /// instead, which can absorb scale drift (e.g. from a learned
    /// projection-tracking front end) a rigid solve smears across the
    /// whole trajectory. Only meaningful when
    /// `pose_graph_refinement_enabled` is set. Combining `"sim3"` with
    /// `--pose-graph-refinement-gnc` prints a warning: `Sim3PoseGraph`
    /// has no GNC variant yet, so the flag is silently ignored on that
    /// path.
    pose_graph_refinement_solver: LoopRefinementSolverKind,
}

/// CLI-selectable mirror of [`LoopRefinementVerifier`]'s variants. A
/// separate enum (rather than dispatching straight to the library type) so
/// this demo can carry its own `Display`/parsing without coupling to the
/// library enum's exact shape.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopRefinementVerifierKind {
    Essential,
    Pnp,
}

/// CLI-selectable mirror of [`LoopRefinementSolver`]'s variants (minus its
/// `Sim3` payload config, which this demo always constructs from
/// `Sim3PoseGraphConfig::default()` — no CLI knobs for the Sim3 solver's
/// own LM settings yet).
#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopRefinementSolverKind {
    Se3,
    Sim3,
}

#[cfg(feature = "image-io")]
fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut euroc_dir: Option<PathBuf> = None;
    let mut out_dir = PathBuf::from("target/euroc_online_slam_vi_image_demo");
    let mut max_frames: usize = 400;
    let mut gravity_world = Vector3::new(0.0, 0.0, -9.81);
    let mut vi_init_max_wait_seconds: f64 = 5.0;
    let mut vi_init_try_initialize_on_every_frame: bool = false;
    let mut vi_init_gyro_std_limit: Option<f64> = None;
    let mut vi_init_mean_gyro_magnitude_limit: Option<f64> = None;
    let mut vi_init_accel_std_limit: Option<f64> = None;
    let mut bootstrap_depth_meters: f64 = 4.0;
    let mut corner_max_features: usize = 1500;
    let mut corner_min_score: f32 = 0.02;
    let mut corner_descriptor_radius: usize = 5;
    let mut undistort: bool = true;
    let mut stereo_bootstrap: bool = true;
    let mut stereo_bootstrap_strict: bool = true;
    let mut adaptive_motion_failures_to_switch_to_pose: usize = 2;
    let mut adaptive_motion_successes_to_switch_to_imu: usize = 5;
    let mut adaptive_motion_imu_velocity_refresh_policy: ImuVelocityRefreshPolicy =
        ImuVelocityRefreshPolicy::default();
    let mut motion_vi_init_enabled: bool = false;
    let mut motion_vi_init_after_static_give_up: bool = false;
    let mut motion_vi_init_from_configured_bias: bool = false;
    let mut motion_vi_init_min_keyframes: usize = 10;
    let mut motion_vi_init_min_translation_meters: f64 = 2.0;
    let mut motion_vi_init_recover_scale: bool = false;
    let mut local_vi_ba_enabled: bool = false;
    let mut observation_confidence_ba_enabled: bool = false;
    let mut local_vi_ba_marginalization: bool = false;
    let mut local_vi_ba_initial_prior_std_devs: Option<(f64, f64, f64)> = None;
    let mut local_vi_ba_freeze_biases_above: Option<f64> = None;
    let mut local_vi_ba_reject_writeback_above: Option<f64> = None;
    let mut local_vi_ba_reject_final_imu_nis_per_dof_above: Option<f64> = None;
    let mut local_vi_ba_reject_velocity_above_mps: Option<f64> = None;
    let mut local_vi_ba_reject_pose_translation_above_meters: Option<f64> = None;
    let mut local_vi_ba_reject_pose_rotation_above_degrees: Option<f64> = None;
    let mut local_vi_ba_adaptive_velocity_gate: bool = false;
    let default_adaptive_velocity_gate = AdaptiveVelocityGateConfig::default();
    let mut local_vi_ba_adaptive_velocity_quantile: f64 =
        default_adaptive_velocity_gate.reference_quantile;
    let mut local_vi_ba_adaptive_velocity_multiplier: f64 =
        default_adaptive_velocity_gate.multiplier;
    let mut local_vi_ba_adaptive_velocity_margin_mps: f64 =
        default_adaptive_velocity_gate.margin_mps;
    let mut local_vi_ba_adaptive_velocity_min_mps: f64 =
        default_adaptive_velocity_gate.min_threshold_mps;
    let mut local_vi_ba_adaptive_velocity_max_mps: Option<f64> =
        default_adaptive_velocity_gate.max_threshold_mps;
    let mut local_vi_ba_adaptive_velocity_min_references: usize =
        default_adaptive_velocity_gate.min_reference_count;
    let mut motion_vi_init_max_velocity_mps: Option<f64> = None;
    let mut motion_vi_init_max_gyro_bias_rad_s: Option<f64> = None;
    let mut motion_vi_init_max_accel_bias_mps2: Option<f64> = None;
    let mut motion_vi_init_max_imu_nis_per_dof: Option<f64> = None;
    let mut motion_vi_init_max_rotation_residual_rms_rad: Option<f64> = None;
    let mut motion_vi_init_max_velocity_residual_rms_mps: Option<f64> = None;
    let mut motion_vi_init_max_position_residual_rms_meters: Option<f64> = None;
    let mut vi_bias_release_min_keyframes: Option<usize> = None;
    let mut vi_bias_release_min_translation_meters: Option<f64> = None;
    let mut motion_vi_init_estimate_gravity: bool = false;
    let mut motion_vi_init_max_gravity_norm_deviation: Option<f64> = None;
    let mut motion_vi_init_estimate_gyro_bias: bool = false;
    let mut covisibility_local_map_max_keyframes: Option<usize> = None;
    let mut covisibility_local_map_min_shared: usize = 15;
    let mut covisibility_local_ba_enabled: bool = false;
    let mut sparse_factor_graph_enabled: bool = false;
    let mut covisibility_local_ba_min_keyframes: usize = 3;
    let mut covisibility_local_ba_max_keyframes: Option<usize> = None;
    let mut covisibility_local_ba_motion_vi_raw_activation: Option<(f64, f64, f64)> = None;
    let mut covisibility_local_ba_max_seed_landmarks_for_activation: Option<usize> = None;
    let mut covisibility_local_ba_trigger_every: usize = 1;
    let mut covisibility_local_ba_max_neighbor_keyframes: usize = 10;
    let mut covisibility_local_ba_min_shared: usize = 15;
    let mut covisibility_local_ba_max_boundary_keyframes: usize = 10;
    let mut covisibility_local_ba_min_boundary_observations: usize = 5;
    let mut covisibility_local_ba_fallback_min_boundary_observations: Option<usize> = None;
    let mut covisibility_local_ba_max_landmarks: Option<usize> = None;
    let mut covisibility_local_ba_min_active_observations: usize = 1;
    let mut covisibility_local_ba_outlier_threshold_px: Option<f64> = Some(5.0);
    let mut covisibility_local_ba_remove_outliers: bool = false;
    let mut covisibility_local_ba_general_stereo: bool = false;
    let mut covisibility_local_ba_general_stereo_max_right_reprojection_px: Option<f64> = Some(5.0);
    let mut covisibility_local_ba_max_outlier_observation_ratio: Option<f64> = None;
    let mut covisibility_local_ba_boundary_support_min_optimized_keyframes: Option<usize> = None;
    let mut covisibility_local_ba_boundary_support_min_fixed_keyframes: usize = 0;
    let mut covisibility_local_ba_max_behind_camera_ratio: Option<f64> = None;
    let mut covisibility_local_ba_min_fixed_to_optimized_ratio: Option<f64> = None;
    let mut covisibility_local_ba_max_pose_translation_correction_m: Option<f64> = None;
    let mut covisibility_local_ba_max_pose_rotation_correction_deg: Option<f64> = None;
    let mut covisibility_local_ba_anchor_weight: Option<f64> = None;
    let mut stereo_landmark_replenish: bool = false;
    let mut stereo_landmark_replenish_max_per_frame: usize = 100;
    let mut stereo_landmark_replenish_anchor_match_radius_px: Option<f64> = None;
    let mut stereo_landmark_replenish_anchor_max_descriptor_distance: Option<f32> = None;
    let mut stereo_landmark_replenish_duplicate_radius_px: Option<f64> = None;
    let mut stereo_landmark_replenish_min_parallax_deg: Option<f64> = None;
    let mut stereo_landmark_replenish_min_depth_meters: Option<f64> = None;
    let mut stereo_landmark_replenish_max_depth_meters: Option<f64> = None;
    let mut max_pose_jump_meters: Option<f64> = None;
    let mut pose_prior_visual_override: bool = false;
    let mut pose_jump_gap_scaling: bool = false;
    let mut pose_jump_gap_scaling_max_multiplier: usize = 10;
    let mut tracking_min_inliers: usize = 0;
    let mut tracking_min_inlier_ratio: f64 = 0.0;
    let mut tracking_max_reprojection_error: Option<f64> = None;
    let mut pnp_pose_prior_warm_start: bool = false;
    let mut projection_guided_tracking: bool = false;
    let mut projection_search_radius_px: f64 = 15.0;
    let mut projection_widen_factor: f64 = 2.0;
    let mut projection_max_widen_retries: u32 = 2;
    let mut projection_query_landmark_distance_ratio: Option<f32> = None;
    let mut projection_no_local_map_refinement: bool = false;
    let mut projection_refinement_search_radius_px: f64 = 8.0;
    let mut projection_refinement_iterations: u32 = 1;
    let mut projection_refinement_radius_shrink_factor: f64 = 1.0;
    let mut projection_refinement_reassign_correspondences: bool = false;
    let mut projection_refinement_min_inlier_pair_retention_ratio: f64 = 0.0;
    let mut projection_refinement_max_translation_correction_m: Option<f64> = None;
    let mut projection_refinement_max_rotation_correction_deg: Option<f64> = None;
    let mut pnp_reprojection_threshold_px: Option<f64> = None;
    let mut motion_model: MotionModelKind = MotionModelKind::Pose;
    let mut imu_extrinsic_from_cam0: bool = false;
    let mut imu_motion_model_carry_forward_velocity: bool = false;
    let mut cross_check_matcher: bool = false;
    let mut mutual_softmax_matcher: bool = false;
    let mut feature_extractor: FeatureExtractorKind = FeatureExtractorKind::Corner;
    let mut hog_max_features: usize = 1500;
    let mut hog_min_corner_score: f32 = 0.05;
    let mut hog_orient: bool = false;
    let mut keyframe_min_translation: Option<f64> = None;
    let mut keyframe_min_frame_gap: Option<u64> = None;
    let mut keyframe_max_frame_gap: Option<u64> = None;
    let mut keyframe_tracked_landmark_ratio: Option<f64> = None;
    let mut keyframe_min_tracked_landmarks_for_ratio: usize = 20;
    let mut keyframe_min_inliers: Option<usize> = None;
    let mut keyframe_min_inlier_ratio: Option<f64> = None;
    let mut vi_init_min_samples: Option<usize> = None;
    let mut vi_init_min_stationary_window_seconds: Option<f64> = None;
    let mut keep_pre_promotion_imu_factors: bool = false;
    let mut relinearise_imu_factor_bias_thresholds: Option<(f64, f64)> = None;
    let mut superpoint_features_dir: Option<PathBuf> = None;
    let mut superpoint_cam1_features_dir: Option<PathBuf> = None;
    let mut superpoint_onnx_model: Option<PathBuf> = None;
    let mut superpoint_onnx_backend = SuperPointOnnxBackendArg::CudaThenCpu;
    let mut export_frame_appearance_descriptors: bool = false;
    let mut run_local_vi_ba_at_vi_init_promotion: bool = false;
    let mut relocalization_enabled: bool = false;
    let mut relocalization_min_inliers: usize = 20;
    let mut relocalization_min_inlier_ratio: f64 = 0.3;
    let mut relocalization_max_reprojection_error: Option<f64> = Some(8.0);
    let mut relocalization_pose_prior_radius_meters: Option<f64> = None;
    let mut relocalization_recent_keyframe_window: Option<usize> = None;
    let mut relocalization_max_translation_from_imu_prediction_meters: Option<f64> = None;
    let mut relocalization_attempt_interval_frames: u64 = 1;
    let mut relocalization_max_consecutive_failed_attempts: Option<u64> = None;
    let mut relocalization_max_translation_per_frame_from_last_success_meters: Option<f64> = None;
    let mut relocalization_min_inlier_depth_median_ratio_to_last_success: Option<f64> = None;
    let mut relocalization_max_inlier_depth_median_ratio_to_last_success: Option<f64> = None;
    let mut relocalization_covisibility_max_keyframes: Option<usize> = None;
    let mut relocalization_covisibility_min_shared: usize = 15;
    let mut relocalization_covisibility_min_landmarks: usize = 30;
    let mut relocalization_covisibility_broader_fallback: bool = true;
    let mut relocalization_covisibility_broader_fallback_interval_frames: u64 = 10;
    let mut relocalization_covisibility_compare_broader_store: bool = false;
    let mut relocalization_appearance_max_keyframes: Option<usize> = None;
    let mut relocalization_appearance_candidate_log_limit: Option<usize> = None;
    let mut relocalization_appearance_min_similarity: f32 = 0.2;
    let mut relocalization_appearance_exclude_recent_frame_gap: u64 = 30;
    let mut relocalization_appearance_min_landmarks: usize = 30;
    let mut relocalization_appearance_broader_fallback: bool = true;
    let mut relocalization_appearance_broader_fallback_interval_frames: u64 = 10;
    let mut relocalization_appearance_compare_broader_store: bool = false;
    let mut relocalization_confirmation_required_recoveries: usize = 1;
    let mut relocalization_confirmation_max_translation_per_frame_meters: Option<f64> = None;
    let mut rebootstrap_after_lost_frames: Option<usize> = None;
    let mut rebootstrap_cooldown_frames: usize = 60;
    let mut rebootstrap_independent_submap: bool = false;
    let mut atlas_broader_recovery_max_source_keyframes: usize = 1;
    let mut atlas_broader_recovery_interval_attempts: usize = 5;
    let mut pose_graph_refinement_enabled: bool = false;
    let mut pose_graph_refinement_trigger_every: usize = 1;
    let mut pose_graph_refinement_fixed_loop_edge_weight: Option<f64> = None;
    let mut pose_graph_refinement_loop_pose_information: bool = false;
    let mut pose_graph_refinement_loop_pose_information_max_eigenvalue: f64 = 1.0;
    let mut pose_graph_refinement_loop_pose_information_loop_edge_scale: f64 = 1.0;
    let mut pose_graph_refinement_gnc: bool = false;
    let mut pose_graph_refinement_pcm: bool = false;
    let mut pose_graph_refinement_pcm_require_individual: bool = true;
    let mut pose_graph_refinement_covariance_gate: Option<f64> = None;
    let mut pose_graph_refinement_verifier: LoopRefinementVerifierKind =
        LoopRefinementVerifierKind::Essential;
    let mut pose_graph_refinement_solver: LoopRefinementSolverKind = LoopRefinementSolverKind::Se3;
    let mut pose_graph_refinement_appearance_loops: bool = false;
    let mut pose_graph_refinement_appearance_min_gap: u64 = 150;
    let mut pose_graph_refinement_appearance_max_candidates: usize = 3;
    let mut pose_graph_refinement_appearance_min_inliers: usize = 30;
    let mut pose_graph_refinement_appearance_confirmation_keyframes: usize = 3;
    let mut pose_graph_refinement_appearance_confirmation_max_misses: usize = 2;
    let mut pose_graph_refinement_appearance_projection_radius_px: Option<f64> = None;
    let mut pose_graph_refinement_appearance_projection_min_matches: usize = 50;
    let mut pose_graph_refinement_fuse_loop_observations: bool = false;
    let mut pose_graph_refinement_loop_welding_ba: bool = false;
    let mut pose_graph_refinement_propagate: bool = false;

    let mut args: Vec<String> = env::args().skip(1).collect();
    let i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--euroc-dir" => {
                euroc_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--out-dir" => {
                out_dir = PathBuf::from(args.remove(i + 1));
                args.remove(i);
            }
            "--max-frames" => {
                max_frames = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--gravity" => {
                let xyz: Vec<f64> = args
                    .remove(i + 1)
                    .split(',')
                    .map(|tok| tok.trim().parse::<f64>())
                    .collect::<Result<_, _>>()?;
                if xyz.len() != 3 {
                    return Err("--gravity expects 'gx,gy,gz'".into());
                }
                gravity_world = Vector3::new(xyz[0], xyz[1], xyz[2]);
                args.remove(i);
            }
            "--vi-init-try-initialize-on-every-frame" => {
                vi_init_try_initialize_on_every_frame = true;
                args.remove(i);
            }
            "--vi-init-max-wait-seconds" => {
                vi_init_max_wait_seconds = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--vi-init-gyro-std-limit" => {
                vi_init_gyro_std_limit = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-init-mean-gyro-magnitude-limit" => {
                vi_init_mean_gyro_magnitude_limit = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-init-accel-std-limit" => {
                vi_init_accel_std_limit = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--bootstrap-depth" => {
                bootstrap_depth_meters = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--corner-max-features" => {
                corner_max_features = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--corner-min-score" => {
                corner_min_score = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--corner-descriptor-radius" => {
                corner_descriptor_radius = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--no-undistort" => {
                undistort = false;
                args.remove(i);
            }
            "--no-stereo-bootstrap" => {
                stereo_bootstrap = false;
                args.remove(i);
            }
            "--stereo-bootstrap-strict" => {
                stereo_bootstrap_strict = true;
                args.remove(i);
            }
            "--allow-fixed-depth-bootstrap" => {
                stereo_bootstrap_strict = false;
                args.remove(i);
            }
            "--adaptive-motion-failures-to-switch-to-pose" => {
                adaptive_motion_failures_to_switch_to_pose = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--adaptive-motion-successes-to-switch-to-imu" => {
                adaptive_motion_successes_to_switch_to_imu = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--adaptive-motion-no-refresh-imu-velocity-on-switch" => {
                // Backward-compat alias preserved through the Phase-25
                // refactor: equivalent to `--adaptive-motion-refresh-policy none`.
                adaptive_motion_imu_velocity_refresh_policy = ImuVelocityRefreshPolicy::None;
                args.remove(i);
            }
            "--adaptive-motion-refresh-policy" => {
                let raw = args.remove(i + 1);
                adaptive_motion_imu_velocity_refresh_policy = match raw.as_str() {
                    "none" => ImuVelocityRefreshPolicy::None,
                    "finite-diff" | "finite-difference" => {
                        ImuVelocityRefreshPolicy::FiniteDifference
                    }
                    "zero-reset" => ImuVelocityRefreshPolicy::ZeroReset,
                    "three-pose-smoother" => ImuVelocityRefreshPolicy::ThreePoseSmoother,
                    other => {
                        return Err(format!(
                            "--adaptive-motion-refresh-policy expects one of \
                             none|finite-diff|zero-reset|three-pose-smoother; got {other:?}"
                        )
                        .into());
                    }
                };
                args.remove(i);
            }
            "--motion-vi-init" => {
                motion_vi_init_enabled = true;
                args.remove(i);
            }
            "--motion-vi-init-after-static-give-up" => {
                motion_vi_init_after_static_give_up = true;
                args.remove(i);
            }
            "--motion-vi-init-from-configured-bias" => {
                motion_vi_init_from_configured_bias = true;
                args.remove(i);
            }
            "--motion-vi-init-min-keyframes" => {
                motion_vi_init_min_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--motion-vi-init-min-translation" => {
                motion_vi_init_min_translation_meters = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--motion-vi-init-recover-scale" => {
                motion_vi_init_recover_scale = true;
                args.remove(i);
            }
            "--local-vi-ba" => {
                local_vi_ba_enabled = true;
                args.remove(i);
            }
            "--observation-confidence-ba" => {
                observation_confidence_ba_enabled = true;
                args.remove(i);
            }
            "--local-vi-ba-marginalization" => {
                local_vi_ba_marginalization = true;
                args.remove(i);
            }
            "--local-vi-ba-initial-prior-std-devs" => {
                let value = args.remove(i + 1);
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() != 3 {
                    return Err("--local-vi-ba-initial-prior-std-devs expects '<velocity_mps>,<gyro_bias_rad_s>,<accel_bias_mps2>'".into());
                }
                let parsed = (
                    parts[0].parse::<f64>()?,
                    parts[1].parse::<f64>()?,
                    parts[2].parse::<f64>()?,
                );
                if [parsed.0, parsed.1, parsed.2]
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                {
                    return Err(
                        "--local-vi-ba-initial-prior-std-devs values must be finite and > 0".into(),
                    );
                }
                local_vi_ba_initial_prior_std_devs = Some(parsed);
                args.remove(i);
            }
            "--local-vi-ba-freeze-biases-above" => {
                local_vi_ba_freeze_biases_above = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--local-vi-ba-reject-writeback-above" => {
                local_vi_ba_reject_writeback_above = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--local-vi-ba-reject-final-imu-nis-per-dof-above" => {
                local_vi_ba_reject_final_imu_nis_per_dof_above = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--local-vi-ba-reject-velocity-above" => {
                let threshold: f64 = args.remove(i + 1).parse()?;
                if threshold < 0.0 {
                    return Err("--local-vi-ba-reject-velocity-above must be >= 0".into());
                }
                local_vi_ba_reject_velocity_above_mps = Some(threshold);
                args.remove(i);
            }
            "--local-vi-ba-reject-pose-translation-above" => {
                let threshold: f64 = args.remove(i + 1).parse()?;
                if !threshold.is_finite() || threshold < 0.0 {
                    return Err(
                        "--local-vi-ba-reject-pose-translation-above must be finite and >= 0"
                            .into(),
                    );
                }
                local_vi_ba_reject_pose_translation_above_meters = Some(threshold);
                args.remove(i);
            }
            "--local-vi-ba-reject-pose-rotation-above-deg" => {
                let threshold: f64 = args.remove(i + 1).parse()?;
                if !threshold.is_finite() || threshold < 0.0 {
                    return Err(
                        "--local-vi-ba-reject-pose-rotation-above-deg must be finite and >= 0"
                            .into(),
                    );
                }
                local_vi_ba_reject_pose_rotation_above_degrees = Some(threshold);
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-gate" => {
                local_vi_ba_adaptive_velocity_gate = true;
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-quantile" => {
                let value: f64 = args.remove(i + 1).parse()?;
                if !(0.0..=1.0).contains(&value) {
                    return Err("--local-vi-ba-adaptive-velocity-quantile must be in [0, 1]".into());
                }
                local_vi_ba_adaptive_velocity_quantile = value;
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-multiplier" => {
                let value: f64 = args.remove(i + 1).parse()?;
                if value < 0.0 {
                    return Err("--local-vi-ba-adaptive-velocity-multiplier must be >= 0".into());
                }
                local_vi_ba_adaptive_velocity_multiplier = value;
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-margin" => {
                let value: f64 = args.remove(i + 1).parse()?;
                if value < 0.0 {
                    return Err("--local-vi-ba-adaptive-velocity-margin must be >= 0".into());
                }
                local_vi_ba_adaptive_velocity_margin_mps = value;
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-min" => {
                let value: f64 = args.remove(i + 1).parse()?;
                if value < 0.0 {
                    return Err("--local-vi-ba-adaptive-velocity-min must be >= 0".into());
                }
                local_vi_ba_adaptive_velocity_min_mps = value;
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-max" => {
                let value: f64 = args.remove(i + 1).parse()?;
                if value < 0.0 {
                    return Err("--local-vi-ba-adaptive-velocity-max must be >= 0".into());
                }
                local_vi_ba_adaptive_velocity_max_mps = Some(value);
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-min-references" => {
                local_vi_ba_adaptive_velocity_min_references = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--motion-vi-init-max-velocity" => {
                motion_vi_init_max_velocity_mps = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-gyro-bias" => {
                motion_vi_init_max_gyro_bias_rad_s = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-accel-bias" => {
                motion_vi_init_max_accel_bias_mps2 = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-imu-nis-per-dof" => {
                motion_vi_init_max_imu_nis_per_dof = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-rotation-residual-rms-rad" => {
                motion_vi_init_max_rotation_residual_rms_rad = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-velocity-residual-rms-mps" => {
                motion_vi_init_max_velocity_residual_rms_mps = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-position-residual-rms-m" => {
                motion_vi_init_max_position_residual_rms_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-bias-release-min-keyframes" => {
                vi_bias_release_min_keyframes = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-bias-release-min-translation" => {
                vi_bias_release_min_translation_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-estimate-gravity" => {
                motion_vi_init_estimate_gravity = true;
                args.remove(i);
            }
            "--motion-vi-init-max-gravity-norm-deviation" => {
                motion_vi_init_max_gravity_norm_deviation = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-estimate-gyro-bias" => {
                motion_vi_init_estimate_gyro_bias = true;
                args.remove(i);
            }
            "--covisibility-local-map-max-keyframes" => {
                covisibility_local_map_max_keyframes = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--covisibility-local-map-min-shared" => {
                covisibility_local_map_min_shared = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba" => {
                covisibility_local_ba_enabled = true;
                args.remove(i);
            }
            "--sparse-factor-graph" => {
                sparse_factor_graph_enabled = true;
                args.remove(i);
            }
            "--covisibility-local-ba-min-keyframes" => {
                covisibility_local_ba_min_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-max-keyframes" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_max_keyframes = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--covisibility-local-ba-motion-vi-raw-activation" => {
                let raw = args.remove(i + 1);
                let values = raw
                    .split(',')
                    .map(str::parse::<f64>)
                    .collect::<Result<Vec<_>, _>>()?;
                if values.len() != 3
                    || values
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
                {
                    return Err("--covisibility-local-ba-motion-vi-raw-activation expects three positive finite values: '<rotation_rad>,<velocity_mps>,<position_m>'".into());
                }
                covisibility_local_ba_motion_vi_raw_activation =
                    Some((values[0], values[1], values[2]));
                args.remove(i);
            }
            "--covisibility-local-ba-max-seed-landmarks-for-activation" => {
                covisibility_local_ba_max_seed_landmarks_for_activation =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--covisibility-local-ba-trigger-every" => {
                covisibility_local_ba_trigger_every = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-max-neighbor-keyframes" => {
                covisibility_local_ba_max_neighbor_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-min-shared" => {
                covisibility_local_ba_min_shared = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-max-boundary-keyframes" => {
                covisibility_local_ba_max_boundary_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-min-boundary-observations" => {
                covisibility_local_ba_min_boundary_observations = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-fallback-min-boundary-observations" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_fallback_min_boundary_observations =
                    if raw.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(raw.parse()?)
                    };
                args.remove(i);
            }
            "--covisibility-local-ba-max-landmarks" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_max_landmarks = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--covisibility-local-ba-min-active-observations" => {
                covisibility_local_ba_min_active_observations = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-outlier-threshold-px" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_outlier_threshold_px = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--covisibility-local-ba-remove-outliers" => {
                covisibility_local_ba_remove_outliers = true;
                args.remove(i);
            }
            "--covisibility-local-ba-general-stereo" => {
                covisibility_local_ba_general_stereo = true;
                args.remove(i);
            }
            "--covisibility-local-ba-general-stereo-max-right-reprojection-px" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_general_stereo_max_right_reprojection_px =
                    if raw.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(raw.parse()?)
                    };
                args.remove(i);
            }
            "--covisibility-local-ba-max-outlier-observation-ratio" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_max_outlier_observation_ratio =
                    if raw.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(raw.parse()?)
                    };
                args.remove(i);
            }
            "--covisibility-local-ba-boundary-support-min-optimized-keyframes" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_boundary_support_min_optimized_keyframes =
                    if raw.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(raw.parse()?)
                    };
                args.remove(i);
            }
            "--covisibility-local-ba-boundary-support-min-fixed-keyframes" => {
                covisibility_local_ba_boundary_support_min_fixed_keyframes =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-max-behind-camera-ratio" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_max_behind_camera_ratio = if raw.eq_ignore_ascii_case("none")
                {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--covisibility-local-ba-min-fixed-to-optimized-ratio" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_min_fixed_to_optimized_ratio =
                    if raw.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(raw.parse()?)
                    };
                args.remove(i);
            }
            "--covisibility-local-ba-anchor-weight" => {
                covisibility_local_ba_anchor_weight = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--covisibility-local-ba-max-pose-translation-correction-m" => {
                covisibility_local_ba_max_pose_translation_correction_m =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--covisibility-local-ba-max-pose-rotation-correction-deg" => {
                covisibility_local_ba_max_pose_rotation_correction_deg =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--stereo-landmark-replenish" => {
                stereo_landmark_replenish = true;
                args.remove(i);
            }
            "--stereo-landmark-replenish-max-per-frame" => {
                stereo_landmark_replenish_max_per_frame = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--stereo-landmark-replenish-anchor-match-radius-px" => {
                stereo_landmark_replenish_anchor_match_radius_px =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--stereo-landmark-replenish-anchor-max-descriptor-distance" => {
                stereo_landmark_replenish_anchor_max_descriptor_distance =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--stereo-landmark-replenish-duplicate-radius-px" => {
                stereo_landmark_replenish_duplicate_radius_px = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--stereo-landmark-replenish-min-parallax-deg" => {
                stereo_landmark_replenish_min_parallax_deg = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--stereo-landmark-replenish-min-depth-meters" => {
                stereo_landmark_replenish_min_depth_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--stereo-landmark-replenish-max-depth-meters" => {
                stereo_landmark_replenish_max_depth_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--max-pose-jump-meters" => {
                max_pose_jump_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pose-prior-visual-override" => {
                pose_prior_visual_override = true;
                args.remove(i);
            }
            "--pose-jump-gap-scaling" => {
                pose_jump_gap_scaling = true;
                args.remove(i);
            }
            "--pose-jump-gap-scaling-max-multiplier" => {
                pose_jump_gap_scaling_max_multiplier = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--tracking-min-inliers" => {
                tracking_min_inliers = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--tracking-min-inlier-ratio" => {
                tracking_min_inlier_ratio = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--tracking-max-reprojection-error" => {
                let raw = args.remove(i + 1);
                tracking_max_reprojection_error = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--pnp-pose-prior-warm-start" => {
                pnp_pose_prior_warm_start = true;
                args.remove(i);
            }
            "--projection-guided-tracking" => {
                projection_guided_tracking = true;
                args.remove(i);
            }
            "--projection-query-landmark-distance-ratio" => {
                let raw = args.remove(i + 1);
                projection_query_landmark_distance_ratio = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--projection-search-radius-px" => {
                projection_search_radius_px = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-widen-factor" => {
                projection_widen_factor = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-max-widen-retries" => {
                projection_max_widen_retries = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-no-local-map-refinement" => {
                projection_no_local_map_refinement = true;
                args.remove(i);
            }
            "--projection-refinement-search-radius-px" => {
                projection_refinement_search_radius_px = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-refinement-iterations" => {
                projection_refinement_iterations = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-refinement-radius-shrink-factor" => {
                projection_refinement_radius_shrink_factor = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-refinement-reassign-correspondences" => {
                projection_refinement_reassign_correspondences = true;
                args.remove(i);
            }
            "--projection-refinement-min-inlier-pair-retention-ratio" => {
                projection_refinement_min_inlier_pair_retention_ratio =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-refinement-max-translation-correction-m" => {
                projection_refinement_max_translation_correction_m =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--projection-refinement-max-rotation-correction-deg" => {
                projection_refinement_max_rotation_correction_deg =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pnp-reprojection-threshold-px" => {
                pnp_reprojection_threshold_px = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-model" => {
                let kind = args.remove(i + 1);
                motion_model = match kind.as_str() {
                    "pose" => MotionModelKind::Pose,
                    "velocity" => MotionModelKind::Velocity,
                    "imu" => MotionModelKind::ImuPredictive,
                    "adaptive-imu-pose" => MotionModelKind::AdaptiveImuPose,
                    other => {
                        return Err(format!(
                            "--motion-model: expected 'pose', 'velocity', 'imu', or 'adaptive-imu-pose', got {other:?}"
                        )
                        .into());
                    }
                };
                args.remove(i);
            }
            "--imu-extrinsic-from-cam0" => {
                imu_extrinsic_from_cam0 = true;
                args.remove(i);
            }
            "--imu-motion-model-carry-forward-velocity" => {
                imu_motion_model_carry_forward_velocity = true;
                args.remove(i);
            }
            "--cross-check-matcher" => {
                cross_check_matcher = true;
                args.remove(i);
            }
            "--mutual-softmax-matcher" => {
                mutual_softmax_matcher = true;
                args.remove(i);
            }
            "--feature-extractor" => {
                let kind = args.remove(i + 1);
                feature_extractor = match kind.as_str() {
                    "corner" => FeatureExtractorKind::Corner,
                    "hog" => FeatureExtractorKind::Hog,
                    "superpoint-offline" => FeatureExtractorKind::SuperPointOffline,
                    "superpoint-onnx" => FeatureExtractorKind::SuperPointOnnx,
                    other => {
                        return Err(format!(
                            "--feature-extractor: expected 'corner', 'hog', 'superpoint-offline', or 'superpoint-onnx', got {other:?}"
                        )
                        .into());
                    }
                };
                args.remove(i);
            }
            "--superpoint-features-dir" => {
                superpoint_features_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--superpoint-cam1-features-dir" => {
                superpoint_cam1_features_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--superpoint-onnx-model" => {
                superpoint_onnx_model = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--superpoint-onnx-backend" => {
                superpoint_onnx_backend = SuperPointOnnxBackendArg::parse(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--export-frame-appearance-descriptors" => {
                export_frame_appearance_descriptors = true;
                args.remove(i);
            }
            "--run-local-vi-ba-at-vi-init-promotion" => {
                run_local_vi_ba_at_vi_init_promotion = true;
                args.remove(i);
            }
            "--hog-max-features" => {
                hog_max_features = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--hog-min-corner-score" => {
                hog_min_corner_score = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--hog-orient" => {
                hog_orient = true;
                args.remove(i);
            }
            "--keyframe-min-translation" => {
                keyframe_min_translation = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keyframe-min-frame-gap" => {
                keyframe_min_frame_gap = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keyframe-max-frame-gap" => {
                keyframe_max_frame_gap = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keyframe-tracked-landmark-ratio" => {
                keyframe_tracked_landmark_ratio = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keyframe-min-tracked-landmarks-for-ratio" => {
                keyframe_min_tracked_landmarks_for_ratio = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--keyframe-min-inliers" => {
                keyframe_min_inliers = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keyframe-min-inlier-ratio" => {
                keyframe_min_inlier_ratio = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-init-min-samples" => {
                vi_init_min_samples = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-init-min-stationary-window-seconds" => {
                vi_init_min_stationary_window_seconds = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keep-pre-promotion-imu-factors" => {
                keep_pre_promotion_imu_factors = true;
                args.remove(i);
            }
            "--relinearise-imu-factor-bias-thresholds" => {
                let raw = args.remove(i + 1);
                let mut parts = raw.split(',');
                let gyro_str = parts.next().ok_or_else(|| {
                    "--relinearise-imu-factor-bias-thresholds: expected '<gyro_rad_s>,<accel_m_s2>'".to_string()
                })?;
                let accel_str = parts.next().ok_or_else(|| {
                    "--relinearise-imu-factor-bias-thresholds: missing accel component".to_string()
                })?;
                let gyro_thresh: f64 = gyro_str.trim().parse()?;
                let accel_thresh: f64 = accel_str.trim().parse()?;
                if gyro_thresh < 0.0 || accel_thresh < 0.0 {
                    return Err(
                        "--relinearise-imu-factor-bias-thresholds: thresholds must be >= 0".into(),
                    );
                }
                relinearise_imu_factor_bias_thresholds = Some((gyro_thresh, accel_thresh));
                args.remove(i);
            }
            "--relocalization-enabled" => {
                relocalization_enabled = true;
                args.remove(i);
            }
            "--relocalization-min-inliers" => {
                relocalization_min_inliers = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-min-inlier-ratio" => {
                relocalization_min_inlier_ratio = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-max-reprojection-error" => {
                let raw = args.remove(i + 1);
                relocalization_max_reprojection_error = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--relocalization-pose-prior-radius" => {
                relocalization_pose_prior_radius_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-recent-keyframe-window" => {
                relocalization_recent_keyframe_window = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-max-translation-from-imu-prediction-meters" => {
                relocalization_max_translation_from_imu_prediction_meters =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-attempt-interval-frames" => {
                relocalization_attempt_interval_frames = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-max-consecutive-failed-attempts" => {
                relocalization_max_consecutive_failed_attempts = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-max-translation-per-frame-from-last-success-meters" => {
                relocalization_max_translation_per_frame_from_last_success_meters =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-min-inlier-depth-median-ratio-to-last-success" => {
                relocalization_min_inlier_depth_median_ratio_to_last_success =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-max-inlier-depth-median-ratio-to-last-success" => {
                relocalization_max_inlier_depth_median_ratio_to_last_success =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-covisibility-max-keyframes" => {
                relocalization_covisibility_max_keyframes = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-covisibility-min-shared" => {
                relocalization_covisibility_min_shared = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-covisibility-min-landmarks" => {
                relocalization_covisibility_min_landmarks = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-covisibility-no-broader-fallback" => {
                relocalization_covisibility_broader_fallback = false;
                args.remove(i);
            }
            "--relocalization-covisibility-broader-fallback-interval-frames" => {
                relocalization_covisibility_broader_fallback_interval_frames =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-covisibility-compare-broader-store" => {
                relocalization_covisibility_compare_broader_store = true;
                args.remove(i);
            }
            "--relocalization-appearance-max-keyframes" => {
                relocalization_appearance_max_keyframes = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-appearance-candidate-log-limit" => {
                relocalization_appearance_candidate_log_limit = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-appearance-min-similarity" => {
                relocalization_appearance_min_similarity = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-appearance-exclude-recent-frame-gap" => {
                relocalization_appearance_exclude_recent_frame_gap = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-appearance-min-landmarks" => {
                relocalization_appearance_min_landmarks = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-appearance-no-broader-fallback" => {
                relocalization_appearance_broader_fallback = false;
                args.remove(i);
            }
            "--relocalization-appearance-broader-fallback-interval-frames" => {
                relocalization_appearance_broader_fallback_interval_frames =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-appearance-compare-broader-store" => {
                relocalization_appearance_compare_broader_store = true;
                args.remove(i);
            }
            "--relocalization-confirmation-required-recoveries" => {
                relocalization_confirmation_required_recoveries = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-confirmation-max-translation-per-frame-meters" => {
                relocalization_confirmation_max_translation_per_frame_meters =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--rebootstrap-after-lost-frames" => {
                rebootstrap_after_lost_frames = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--rebootstrap-cooldown-frames" => {
                rebootstrap_cooldown_frames = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--rebootstrap-independent-submap" => {
                rebootstrap_independent_submap = true;
                args.remove(i);
            }
            "--atlas-broader-recovery-max-source-keyframes" => {
                atlas_broader_recovery_max_source_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--atlas-broader-recovery-interval-attempts" => {
                atlas_broader_recovery_interval_attempts = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement" => {
                pose_graph_refinement_enabled = true;
                args.remove(i);
            }
            "--pose-graph-refinement-trigger-every" => {
                pose_graph_refinement_trigger_every = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-fixed-loop-edge-weight" => {
                pose_graph_refinement_fixed_loop_edge_weight = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pose-graph-refinement-loop-pose-information" => {
                pose_graph_refinement_loop_pose_information = true;
                args.remove(i);
            }
            "--pose-graph-refinement-loop-pose-information-max-eigenvalue" => {
                pose_graph_refinement_loop_pose_information_max_eigenvalue =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-loop-pose-information-loop-edge-scale" => {
                pose_graph_refinement_loop_pose_information_loop_edge_scale =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-gnc" => {
                pose_graph_refinement_gnc = true;
                args.remove(i);
            }
            "--pose-graph-refinement-pcm" => {
                pose_graph_refinement_pcm = true;
                args.remove(i);
            }
            "--pose-graph-refinement-pcm-pairwise-only" => {
                pose_graph_refinement_pcm_require_individual = false;
                args.remove(i);
            }
            "--pose-graph-refinement-covariance-gate" => {
                pose_graph_refinement_covariance_gate = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pose-graph-refinement-verifier" => {
                let kind = args.remove(i + 1);
                pose_graph_refinement_verifier = match kind.as_str() {
                    "essential" => LoopRefinementVerifierKind::Essential,
                    "pnp" => LoopRefinementVerifierKind::Pnp,
                    other => {
                        return Err(format!(
                            "--pose-graph-refinement-verifier: expected 'essential' or 'pnp', got {other:?}"
                        )
                        .into());
                    }
                };
                args.remove(i);
            }
            "--pose-graph-refinement-solver" => {
                let kind = args.remove(i + 1);
                pose_graph_refinement_solver = match kind.as_str() {
                    "se3" => LoopRefinementSolverKind::Se3,
                    "sim3" => LoopRefinementSolverKind::Sim3,
                    other => {
                        return Err(format!(
                            "--pose-graph-refinement-solver: expected 'se3' or 'sim3', got {other:?}"
                        )
                        .into());
                    }
                };
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-loops" => {
                pose_graph_refinement_appearance_loops = true;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-min-gap" => {
                pose_graph_refinement_appearance_min_gap = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-max-candidates" => {
                pose_graph_refinement_appearance_max_candidates = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-min-inliers" => {
                pose_graph_refinement_appearance_min_inliers = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-confirmation-keyframes" => {
                pose_graph_refinement_appearance_confirmation_keyframes =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-confirmation-max-misses" => {
                pose_graph_refinement_appearance_confirmation_max_misses =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-projection-radius" => {
                pose_graph_refinement_appearance_projection_radius_px =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-projection-min-matches" => {
                pose_graph_refinement_appearance_projection_min_matches =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-fuse-loop-observations" => {
                pose_graph_refinement_fuse_loop_observations = true;
                args.remove(i);
            }
            "--pose-graph-refinement-loop-welding-ba" => {
                pose_graph_refinement_loop_welding_ba = true;
                args.remove(i);
            }
            "--pose-graph-refinement-propagate" => {
                pose_graph_refinement_propagate = true;
                args.remove(i);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let euroc_dir = euroc_dir.ok_or("--euroc-dir <path/to/MH_01_easy> is required")?;
    if bootstrap_depth_meters <= 0.0 {
        return Err("--bootstrap-depth must be positive".into());
    }
    if !stereo_bootstrap && stereo_bootstrap_strict {
        return Err(
            "--no-stereo-bootstrap requires --allow-fixed-depth-bootstrap; fixed-depth seeding is diagnostic-only"
                .into(),
        );
    }
    if let Some(ratio) = keyframe_tracked_landmark_ratio {
        if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
            return Err("--keyframe-tracked-landmark-ratio must be in [0, 1]".into());
        }
        if keyframe_min_tracked_landmarks_for_ratio == 0 {
            return Err("--keyframe-min-tracked-landmarks-for-ratio must be >= 1".into());
        }
    }
    if let Some(gap) = keyframe_min_frame_gap {
        if gap == 0 {
            return Err("--keyframe-min-frame-gap must be >= 1".into());
        }
    }
    if let Some(gap) = keyframe_max_frame_gap {
        if gap == 0 {
            return Err("--keyframe-max-frame-gap must be >= 1".into());
        }
        if keyframe_min_frame_gap.is_some_and(|minimum| gap < minimum) {
            return Err("--keyframe-max-frame-gap must be >= --keyframe-min-frame-gap".into());
        }
    }
    if !tracking_min_inlier_ratio.is_finite() || !(0.0..=1.0).contains(&tracking_min_inlier_ratio) {
        return Err("--tracking-min-inlier-ratio must be in [0, 1]".into());
    }
    if max_pose_jump_meters.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err("--max-pose-jump-meters must be finite and positive".into());
    }
    if pose_prior_visual_override && max_pose_jump_meters.is_none() {
        return Err("--pose-prior-visual-override requires --max-pose-jump-meters".into());
    }
    if let Some(max_reprojection_error) = tracking_max_reprojection_error {
        if !max_reprojection_error.is_finite() || max_reprojection_error <= 0.0 {
            return Err("--tracking-max-reprojection-error must be positive or 'none'".into());
        }
    }
    if let Some(ratio) = projection_query_landmark_distance_ratio {
        if !ratio.is_finite() || ratio <= 0.0 || ratio >= 1.0 {
            return Err(
                "--projection-query-landmark-distance-ratio must be in (0, 1) or 'none'".into(),
            );
        }
    }
    if !projection_search_radius_px.is_finite() || projection_search_radius_px <= 0.0 {
        return Err("--projection-search-radius-px must be positive".into());
    }
    if !projection_widen_factor.is_finite() || projection_widen_factor <= 0.0 {
        return Err("--projection-widen-factor must be positive".into());
    }
    if !projection_refinement_search_radius_px.is_finite()
        || projection_refinement_search_radius_px <= 0.0
    {
        return Err("--projection-refinement-search-radius-px must be positive".into());
    }
    if projection_refinement_iterations == 0 {
        return Err("--projection-refinement-iterations must be >= 1".into());
    }
    if !projection_refinement_radius_shrink_factor.is_finite()
        || projection_refinement_radius_shrink_factor <= 0.0
        || projection_refinement_radius_shrink_factor > 1.0
    {
        return Err("--projection-refinement-radius-shrink-factor must be in (0, 1]".into());
    }
    if relocalization_attempt_interval_frames == 0 {
        return Err("--relocalization-attempt-interval-frames must be >= 1".into());
    }
    if let Some(max_attempts) = relocalization_max_consecutive_failed_attempts {
        if max_attempts == 0 {
            return Err("--relocalization-max-consecutive-failed-attempts must be >= 1".into());
        }
    }
    if let Some(max_per_frame) = relocalization_max_translation_per_frame_from_last_success_meters {
        if !max_per_frame.is_finite() || max_per_frame <= 0.0 {
            return Err("--relocalization-max-translation-per-frame-from-last-success-meters must be positive".into());
        }
    }
    if let Some(min_ratio) = relocalization_min_inlier_depth_median_ratio_to_last_success {
        if !min_ratio.is_finite() || min_ratio <= 0.0 {
            return Err(
                "--relocalization-min-inlier-depth-median-ratio-to-last-success must be positive"
                    .into(),
            );
        }
    }
    if let Some(max_ratio) = relocalization_max_inlier_depth_median_ratio_to_last_success {
        if !max_ratio.is_finite() || max_ratio <= 0.0 {
            return Err(
                "--relocalization-max-inlier-depth-median-ratio-to-last-success must be positive"
                    .into(),
            );
        }
    }
    if let (Some(min_ratio), Some(max_ratio)) = (
        relocalization_min_inlier_depth_median_ratio_to_last_success,
        relocalization_max_inlier_depth_median_ratio_to_last_success,
    ) {
        if min_ratio > max_ratio {
            return Err("--relocalization-min-inlier-depth-median-ratio-to-last-success must be <= --relocalization-max-inlier-depth-median-ratio-to-last-success".into());
        }
    }
    if let Some(max_keyframes) = relocalization_covisibility_max_keyframes {
        if max_keyframes == 0 {
            return Err("--relocalization-covisibility-max-keyframes must be >= 1".into());
        }
    }
    if relocalization_covisibility_min_shared == 0 {
        return Err("--relocalization-covisibility-min-shared must be >= 1".into());
    }
    if relocalization_covisibility_min_landmarks == 0 {
        return Err("--relocalization-covisibility-min-landmarks must be >= 1".into());
    }
    if relocalization_covisibility_broader_fallback_interval_frames == 0 {
        return Err(
            "--relocalization-covisibility-broader-fallback-interval-frames must be >= 1".into(),
        );
    }
    if !projection_refinement_min_inlier_pair_retention_ratio.is_finite()
        || !(0.0..=1.0).contains(&projection_refinement_min_inlier_pair_retention_ratio)
    {
        return Err(
            "--projection-refinement-min-inlier-pair-retention-ratio must be in [0, 1]".into(),
        );
    }
    if projection_refinement_max_translation_correction_m
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("--projection-refinement-max-translation-correction-m must be positive".into());
    }
    if projection_refinement_max_rotation_correction_deg
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("--projection-refinement-max-rotation-correction-deg must be positive".into());
    }
    if let Some(max_keyframes) = relocalization_appearance_max_keyframes {
        if max_keyframes == 0 {
            return Err("--relocalization-appearance-max-keyframes must be >= 1".into());
        }
    }
    if let Some(candidate_log_limit) = relocalization_appearance_candidate_log_limit {
        if candidate_log_limit == 0 {
            return Err("--relocalization-appearance-candidate-log-limit must be >= 1".into());
        }
    }
    if !relocalization_appearance_min_similarity.is_finite() {
        return Err("--relocalization-appearance-min-similarity must be finite".into());
    }
    if relocalization_appearance_min_landmarks == 0 {
        return Err("--relocalization-appearance-min-landmarks must be >= 1".into());
    }
    if relocalization_appearance_broader_fallback_interval_frames == 0 {
        return Err(
            "--relocalization-appearance-broader-fallback-interval-frames must be >= 1".into(),
        );
    }
    if relocalization_confirmation_required_recoveries == 0 {
        return Err("--relocalization-confirmation-required-recoveries must be >= 1".into());
    }
    if let Some(max_per_frame) = relocalization_confirmation_max_translation_per_frame_meters {
        if !max_per_frame.is_finite() || max_per_frame <= 0.0 {
            return Err(
                "--relocalization-confirmation-max-translation-per-frame-meters must be positive"
                    .into(),
            );
        }
    }
    if let Some(lost_frames) = rebootstrap_after_lost_frames {
        if lost_frames == 0 {
            return Err("--rebootstrap-after-lost-frames must be >= 1".into());
        }
        if !stereo_bootstrap && !stereo_landmark_replenish {
            return Err(
                "--rebootstrap-after-lost-frames requires --stereo-bootstrap and/or --stereo-landmark-replenish (cam1 stereo matching)".into(),
            );
        }
    }
    if rebootstrap_cooldown_frames == 0 {
        return Err("--rebootstrap-cooldown-frames must be >= 1".into());
    }
    if rebootstrap_independent_submap && rebootstrap_after_lost_frames.is_none() {
        return Err(
            "--rebootstrap-independent-submap requires --rebootstrap-after-lost-frames".into(),
        );
    }
    if atlas_broader_recovery_max_source_keyframes == 0 {
        return Err("--atlas-broader-recovery-max-source-keyframes must be >= 1".into());
    }
    if atlas_broader_recovery_interval_attempts == 0 {
        return Err("--atlas-broader-recovery-interval-attempts must be >= 1".into());
    }
    if motion_vi_init_after_static_give_up && !motion_vi_init_enabled {
        return Err("--motion-vi-init-after-static-give-up requires --motion-vi-init".into());
    }
    if motion_vi_init_from_configured_bias && !motion_vi_init_enabled {
        return Err("--motion-vi-init-from-configured-bias requires --motion-vi-init".into());
    }
    if !pose_graph_refinement_pcm_require_individual && !pose_graph_refinement_pcm {
        return Err(
            "--pose-graph-refinement-pcm-pairwise-only requires --pose-graph-refinement-pcm".into(),
        );
    }
    if !motion_vi_init_enabled
        && (motion_vi_init_max_velocity_mps.is_some()
            || motion_vi_init_max_gyro_bias_rad_s.is_some()
            || motion_vi_init_max_accel_bias_mps2.is_some()
            || motion_vi_init_max_imu_nis_per_dof.is_some()
            || motion_vi_init_max_rotation_residual_rms_rad.is_some()
            || motion_vi_init_max_velocity_residual_rms_mps.is_some()
            || motion_vi_init_max_position_residual_rms_meters.is_some())
    {
        return Err("motion-VI sanity limits require --motion-vi-init".into());
    }
    if !motion_vi_init_enabled
        && (vi_bias_release_min_keyframes.is_some()
            || vi_bias_release_min_translation_meters.is_some())
    {
        return Err(
            "--vi-bias-release-min-keyframes / --vi-bias-release-min-translation require --motion-vi-init"
                .into(),
        );
    }
    if !motion_vi_init_enabled
        && (motion_vi_init_estimate_gravity || motion_vi_init_max_gravity_norm_deviation.is_some())
    {
        return Err(
            "--motion-vi-init-estimate-gravity / --motion-vi-init-max-gravity-norm-deviation require --motion-vi-init"
                .into(),
        );
    }
    if !motion_vi_init_estimate_gravity && motion_vi_init_max_gravity_norm_deviation.is_some() {
        return Err(
            "--motion-vi-init-max-gravity-norm-deviation requires --motion-vi-init-estimate-gravity"
                .into(),
        );
    }
    if !motion_vi_init_enabled && motion_vi_init_estimate_gyro_bias {
        return Err("--motion-vi-init-estimate-gyro-bias requires --motion-vi-init".into());
    }
    for (value, flag) in [
        (
            motion_vi_init_max_velocity_mps,
            "--motion-vi-init-max-velocity",
        ),
        (
            motion_vi_init_max_gyro_bias_rad_s,
            "--motion-vi-init-max-gyro-bias",
        ),
        (
            motion_vi_init_max_accel_bias_mps2,
            "--motion-vi-init-max-accel-bias",
        ),
        (
            motion_vi_init_max_imu_nis_per_dof,
            "--motion-vi-init-max-imu-nis-per-dof",
        ),
        (
            motion_vi_init_max_rotation_residual_rms_rad,
            "--motion-vi-init-max-rotation-residual-rms-rad",
        ),
        (
            motion_vi_init_max_velocity_residual_rms_mps,
            "--motion-vi-init-max-velocity-residual-rms-mps",
        ),
        (
            motion_vi_init_max_position_residual_rms_meters,
            "--motion-vi-init-max-position-residual-rms-m",
        ),
    ] {
        if value.is_some_and(|limit| !limit.is_finite() || limit <= 0.0) {
            return Err(format!("{flag} must be finite and > 0").into());
        }
    }
    if local_vi_ba_reject_final_imu_nis_per_dof_above
        .is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
    {
        return Err(
            "--local-vi-ba-reject-final-imu-nis-per-dof-above must be finite and > 0".into(),
        );
    }
    if covisibility_local_ba_motion_vi_raw_activation.is_some()
        && (!covisibility_local_ba_enabled || !motion_vi_init_enabled)
    {
        return Err("--covisibility-local-ba-motion-vi-raw-activation requires --covisibility-local-ba and --motion-vi-init".into());
    }
    if let Some(maximum) = covisibility_local_ba_max_seed_landmarks_for_activation {
        if !covisibility_local_ba_enabled || maximum == 0 {
            return Err("--covisibility-local-ba-max-seed-landmarks-for-activation requires --covisibility-local-ba and a value >= 1".into());
        }
    }
    if covisibility_local_ba_enabled {
        if covisibility_local_ba_min_keyframes == 0 {
            return Err("--covisibility-local-ba-min-keyframes must be >= 1".into());
        }
        if covisibility_local_ba_max_keyframes
            .is_some_and(|maximum| maximum < covisibility_local_ba_min_keyframes)
        {
            return Err(
                "--covisibility-local-ba-max-keyframes must be >= the minimum or 'none'".into(),
            );
        }
        if covisibility_local_ba_trigger_every == 0 {
            return Err("--covisibility-local-ba-trigger-every must be >= 1".into());
        }
        if covisibility_local_ba_min_shared == 0 {
            return Err("--covisibility-local-ba-min-shared must be >= 1".into());
        }
        if covisibility_local_ba_min_active_observations == 0 {
            return Err("--covisibility-local-ba-min-active-observations must be >= 1".into());
        }
        if let Some(fallback_min) = covisibility_local_ba_fallback_min_boundary_observations {
            if fallback_min == 0 {
                return Err(
                    "--covisibility-local-ba-fallback-min-boundary-observations must be >= 1 or 'none'"
                        .into(),
                );
            }
        }
        if let Some(px) = covisibility_local_ba_outlier_threshold_px {
            if !px.is_finite() || px <= 0.0 {
                return Err(
                    "--covisibility-local-ba-outlier-threshold-px must be positive or 'none'"
                        .into(),
                );
            }
        }
        if let Some(px) = covisibility_local_ba_general_stereo_max_right_reprojection_px {
            if !px.is_finite() || px <= 0.0 {
                return Err(
                    "--covisibility-local-ba-general-stereo-max-right-reprojection-px must be positive or 'none'"
                        .into(),
                );
            }
        }
        if covisibility_local_ba_remove_outliers
            && covisibility_local_ba_outlier_threshold_px.is_none()
        {
            return Err(
                "--covisibility-local-ba-remove-outliers requires an outlier threshold".into(),
            );
        }
        if let Some(ratio) = covisibility_local_ba_max_outlier_observation_ratio {
            if !(0.0..=1.0).contains(&ratio) {
                return Err(
                    "--covisibility-local-ba-max-outlier-observation-ratio must be in [0, 1]"
                        .into(),
                );
            }
        }
        if let Some(ratio) = covisibility_local_ba_max_behind_camera_ratio {
            if !(0.0..=1.0).contains(&ratio) {
                return Err(
                    "--covisibility-local-ba-max-behind-camera-ratio must be in [0, 1] or 'none'"
                        .into(),
                );
            }
        }
        if let Some(ratio) = covisibility_local_ba_min_fixed_to_optimized_ratio {
            if !(ratio.is_finite() && ratio > 0.0) {
                return Err(
                    "--covisibility-local-ba-min-fixed-to-optimized-ratio must be > 0 or 'none'"
                        .into(),
                );
            }
        }
        if let Some(limit) = covisibility_local_ba_max_pose_translation_correction_m {
            if !limit.is_finite() || limit < 0.0 {
                return Err("--covisibility-local-ba-max-pose-translation-correction-m must be finite and >= 0".into());
            }
        }
        if let Some(limit) = covisibility_local_ba_max_pose_rotation_correction_deg {
            if !limit.is_finite() || limit < 0.0 {
                return Err("--covisibility-local-ba-max-pose-rotation-correction-deg must be finite and >= 0".into());
            }
        }
        if let Some(min_optimized) = covisibility_local_ba_boundary_support_min_optimized_keyframes
        {
            if min_optimized < 1 {
                return Err(
                    "--covisibility-local-ba-boundary-support-min-optimized-keyframes must be >= 1 or 'none'"
                        .into(),
                );
            }
            if covisibility_local_ba_boundary_support_min_fixed_keyframes < 1 {
                return Err(
                    "--covisibility-local-ba-boundary-support-min-fixed-keyframes must be >= 1 when boundary support gate is enabled"
                        .into(),
                );
            }
        }
    }
    if pose_graph_refinement_enabled {
        if pose_graph_refinement_fuse_loop_observations && !pose_graph_refinement_appearance_loops {
            return Err(
                "--pose-graph-refinement-fuse-loop-observations requires --pose-graph-refinement-appearance-loops"
                    .into(),
            );
        }
        if pose_graph_refinement_loop_welding_ba && !pose_graph_refinement_fuse_loop_observations {
            return Err(
                "--pose-graph-refinement-loop-welding-ba requires --pose-graph-refinement-fuse-loop-observations"
                    .into(),
            );
        }
        if pose_graph_refinement_trigger_every == 0 {
            return Err("--pose-graph-refinement-trigger-every must be >= 1".into());
        }
        if pose_graph_refinement_fixed_loop_edge_weight
            .is_some_and(|weight| !weight.is_finite() || weight <= 0.0)
        {
            return Err(
                "--pose-graph-refinement-fixed-loop-edge-weight must be finite and > 0".into(),
            );
        }
        if pose_graph_refinement_fixed_loop_edge_weight.is_some()
            && pose_graph_refinement_loop_pose_information
        {
            return Err(
                "fixed loop weight and loop pose information are mutually exclusive".into(),
            );
        }
        if !pose_graph_refinement_loop_pose_information_max_eigenvalue.is_finite()
            || pose_graph_refinement_loop_pose_information_max_eigenvalue <= 0.0
        {
            return Err(
                "--pose-graph-refinement-loop-pose-information-max-eigenvalue must be finite and > 0"
                    .into(),
            );
        }
        if !pose_graph_refinement_loop_pose_information_loop_edge_scale.is_finite()
            || pose_graph_refinement_loop_pose_information_loop_edge_scale <= 0.0
        {
            return Err(
                "--pose-graph-refinement-loop-pose-information-loop-edge-scale must be finite and > 0"
                    .into(),
            );
        }
        if let Some(gate) = pose_graph_refinement_covariance_gate {
            if !gate.is_finite() || gate <= 0.0 {
                return Err("--pose-graph-refinement-covariance-gate must be positive".into());
            }
        }
        if pose_graph_refinement_appearance_loops {
            if pose_graph_refinement_appearance_min_gap == 0 {
                return Err("--pose-graph-refinement-appearance-min-gap must be >= 1".into());
            }
            if pose_graph_refinement_appearance_max_candidates == 0 {
                return Err(
                    "--pose-graph-refinement-appearance-max-candidates must be >= 1".into(),
                );
            }
            if pose_graph_refinement_appearance_min_inliers == 0 {
                return Err("--pose-graph-refinement-appearance-min-inliers must be >= 1".into());
            }
            if pose_graph_refinement_appearance_confirmation_keyframes == 0 {
                return Err(
                    "--pose-graph-refinement-appearance-confirmation-keyframes must be >= 1".into(),
                );
            }
            if pose_graph_refinement_appearance_confirmation_max_misses == 0 {
                return Err(
                    "--pose-graph-refinement-appearance-confirmation-max-misses must be >= 1"
                        .into(),
                );
            }
            if pose_graph_refinement_appearance_projection_radius_px
                .is_some_and(|radius| !radius.is_finite() || radius <= 0.0)
            {
                return Err(
                    "--pose-graph-refinement-appearance-projection-radius must be finite and > 0"
                        .into(),
                );
            }
            if pose_graph_refinement_appearance_projection_min_matches == 0 {
                return Err(
                    "--pose-graph-refinement-appearance-projection-min-matches must be >= 1".into(),
                );
            }
        }
        if pose_graph_refinement_solver == LoopRefinementSolverKind::Sim3
            && pose_graph_refinement_gnc
        {
            eprintln!(
                "warning: --pose-graph-refinement-gnc has no effect with \
                 --pose-graph-refinement-solver sim3 — Sim3PoseGraph has no GNC \
                 robust-outlier variant yet, so the flag is silently ignored on \
                 that path (see LoopRefinementSolver::Sim3's doc comment)."
            );
        }
        if pose_graph_refinement_solver == LoopRefinementSolverKind::Sim3
            && (stereo_bootstrap || stereo_landmark_replenish)
        {
            return Err(
                "--pose-graph-refinement-solver sim3 is not valid for metric stereo maps; \
                 ORB-SLAM2/3 fix loop scale to 1 for stereo/RGB-D, so use the default \
                 --pose-graph-refinement-solver se3"
                    .into(),
            );
        }
    }
    Ok(CliArgs {
        euroc_dir,
        out_dir,
        max_frames,
        gravity_world,
        vi_init_max_wait_seconds,
        vi_init_try_initialize_on_every_frame,
        vi_init_gyro_std_limit,
        vi_init_mean_gyro_magnitude_limit,
        vi_init_accel_std_limit,
        bootstrap_depth_meters,
        corner_max_features,
        corner_min_score,
        corner_descriptor_radius,
        undistort,
        stereo_bootstrap,
        stereo_bootstrap_strict,
        adaptive_motion_failures_to_switch_to_pose,
        adaptive_motion_successes_to_switch_to_imu,
        adaptive_motion_imu_velocity_refresh_policy,
        motion_vi_init_enabled,
        motion_vi_init_after_static_give_up,
        motion_vi_init_from_configured_bias,
        motion_vi_init_min_keyframes,
        motion_vi_init_min_translation_meters,
        motion_vi_init_recover_scale,
        local_vi_ba_enabled,
        observation_confidence_ba_enabled,
        local_vi_ba_marginalization,
        local_vi_ba_initial_prior_std_devs,
        local_vi_ba_freeze_biases_above,
        local_vi_ba_reject_writeback_above,
        local_vi_ba_reject_final_imu_nis_per_dof_above,
        local_vi_ba_reject_velocity_above_mps,
        local_vi_ba_reject_pose_translation_above_meters,
        local_vi_ba_reject_pose_rotation_above_degrees,
        local_vi_ba_adaptive_velocity_gate,
        local_vi_ba_adaptive_velocity_quantile,
        local_vi_ba_adaptive_velocity_multiplier,
        local_vi_ba_adaptive_velocity_margin_mps,
        local_vi_ba_adaptive_velocity_min_mps,
        local_vi_ba_adaptive_velocity_max_mps,
        local_vi_ba_adaptive_velocity_min_references,
        motion_vi_init_max_velocity_mps,
        motion_vi_init_max_gyro_bias_rad_s,
        motion_vi_init_max_accel_bias_mps2,
        motion_vi_init_max_imu_nis_per_dof,
        motion_vi_init_max_rotation_residual_rms_rad,
        motion_vi_init_max_velocity_residual_rms_mps,
        motion_vi_init_max_position_residual_rms_meters,
        vi_bias_release_min_keyframes,
        vi_bias_release_min_translation_meters,
        motion_vi_init_estimate_gravity,
        motion_vi_init_max_gravity_norm_deviation,
        motion_vi_init_estimate_gyro_bias,
        covisibility_local_map_max_keyframes,
        covisibility_local_map_min_shared,
        covisibility_local_ba_enabled,
        sparse_factor_graph_enabled,
        covisibility_local_ba_min_keyframes,
        covisibility_local_ba_max_keyframes,
        covisibility_local_ba_motion_vi_raw_activation,
        covisibility_local_ba_max_seed_landmarks_for_activation,
        covisibility_local_ba_trigger_every,
        covisibility_local_ba_max_neighbor_keyframes,
        covisibility_local_ba_min_shared,
        covisibility_local_ba_max_boundary_keyframes,
        covisibility_local_ba_min_boundary_observations,
        covisibility_local_ba_fallback_min_boundary_observations,
        covisibility_local_ba_max_landmarks,
        covisibility_local_ba_min_active_observations,
        covisibility_local_ba_outlier_threshold_px,
        covisibility_local_ba_remove_outliers,
        covisibility_local_ba_general_stereo,
        covisibility_local_ba_general_stereo_max_right_reprojection_px,
        covisibility_local_ba_max_outlier_observation_ratio,
        covisibility_local_ba_boundary_support_min_optimized_keyframes,
        covisibility_local_ba_boundary_support_min_fixed_keyframes,
        covisibility_local_ba_max_behind_camera_ratio,
        covisibility_local_ba_min_fixed_to_optimized_ratio,
        covisibility_local_ba_max_pose_translation_correction_m,
        covisibility_local_ba_max_pose_rotation_correction_deg,
        covisibility_local_ba_anchor_weight,
        stereo_landmark_replenish,
        stereo_landmark_replenish_max_per_frame,
        stereo_landmark_replenish_anchor_match_radius_px,
        stereo_landmark_replenish_anchor_max_descriptor_distance,
        stereo_landmark_replenish_duplicate_radius_px,
        stereo_landmark_replenish_min_parallax_deg,
        stereo_landmark_replenish_min_depth_meters,
        stereo_landmark_replenish_max_depth_meters,
        max_pose_jump_meters,
        pose_prior_visual_override,
        pose_jump_gap_scaling,
        pose_jump_gap_scaling_max_multiplier,
        tracking_min_inliers,
        tracking_min_inlier_ratio,
        tracking_max_reprojection_error,
        pnp_pose_prior_warm_start,
        projection_guided_tracking,
        projection_search_radius_px,
        projection_widen_factor,
        projection_max_widen_retries,
        projection_query_landmark_distance_ratio,
        projection_no_local_map_refinement,
        projection_refinement_search_radius_px,
        projection_refinement_iterations,
        projection_refinement_radius_shrink_factor,
        projection_refinement_reassign_correspondences,
        projection_refinement_min_inlier_pair_retention_ratio,
        projection_refinement_max_translation_correction_m,
        projection_refinement_max_rotation_correction_deg,
        pnp_reprojection_threshold_px,
        motion_model,
        imu_extrinsic_from_cam0,
        imu_motion_model_carry_forward_velocity,
        cross_check_matcher,
        mutual_softmax_matcher,
        feature_extractor,
        hog_max_features,
        hog_min_corner_score,
        hog_orient,
        keyframe_min_translation,
        keyframe_min_frame_gap,
        keyframe_max_frame_gap,
        keyframe_tracked_landmark_ratio,
        keyframe_min_tracked_landmarks_for_ratio,
        keyframe_min_inliers,
        keyframe_min_inlier_ratio,
        vi_init_min_samples,
        vi_init_min_stationary_window_seconds,
        keep_pre_promotion_imu_factors,
        relinearise_imu_factor_bias_thresholds,
        superpoint_features_dir,
        superpoint_cam1_features_dir,
        superpoint_onnx_model,
        superpoint_onnx_backend,
        export_frame_appearance_descriptors,
        run_local_vi_ba_at_vi_init_promotion,
        relocalization_enabled,
        relocalization_min_inliers,
        relocalization_min_inlier_ratio,
        relocalization_max_reprojection_error,
        relocalization_pose_prior_radius_meters,
        relocalization_recent_keyframe_window,
        relocalization_max_translation_from_imu_prediction_meters,
        relocalization_attempt_interval_frames,
        relocalization_max_consecutive_failed_attempts,
        relocalization_max_translation_per_frame_from_last_success_meters,
        relocalization_min_inlier_depth_median_ratio_to_last_success,
        relocalization_max_inlier_depth_median_ratio_to_last_success,
        relocalization_covisibility_max_keyframes,
        relocalization_covisibility_min_shared,
        relocalization_covisibility_min_landmarks,
        relocalization_covisibility_broader_fallback,
        relocalization_covisibility_broader_fallback_interval_frames,
        relocalization_covisibility_compare_broader_store,
        relocalization_appearance_max_keyframes,
        relocalization_appearance_candidate_log_limit,
        relocalization_appearance_min_similarity,
        relocalization_appearance_exclude_recent_frame_gap,
        relocalization_appearance_min_landmarks,
        relocalization_appearance_broader_fallback,
        relocalization_appearance_broader_fallback_interval_frames,
        relocalization_appearance_compare_broader_store,
        relocalization_confirmation_required_recoveries,
        relocalization_confirmation_max_translation_per_frame_meters,
        rebootstrap_after_lost_frames,
        rebootstrap_cooldown_frames,
        rebootstrap_independent_submap,
        atlas_broader_recovery_max_source_keyframes,
        atlas_broader_recovery_interval_attempts,
        pose_graph_refinement_enabled,
        pose_graph_refinement_trigger_every,
        pose_graph_refinement_fixed_loop_edge_weight,
        pose_graph_refinement_loop_pose_information,
        pose_graph_refinement_loop_pose_information_max_eigenvalue,
        pose_graph_refinement_loop_pose_information_loop_edge_scale,
        pose_graph_refinement_gnc,
        pose_graph_refinement_pcm,
        pose_graph_refinement_pcm_require_individual,
        pose_graph_refinement_covariance_gate,
        pose_graph_refinement_verifier,
        pose_graph_refinement_appearance_loops,
        pose_graph_refinement_appearance_min_gap,
        pose_graph_refinement_appearance_max_candidates,
        pose_graph_refinement_appearance_min_inliers,
        pose_graph_refinement_appearance_confirmation_keyframes,
        pose_graph_refinement_appearance_confirmation_max_misses,
        pose_graph_refinement_appearance_projection_radius_px,
        pose_graph_refinement_appearance_projection_min_matches,
        pose_graph_refinement_fuse_loop_observations,
        pose_graph_refinement_loop_welding_ba,
        pose_graph_refinement_propagate,
        pose_graph_refinement_solver,
    })
}

/// Decompose an EuRoC `T_BS` 4×4 body-to-sensor matrix into an [`SE3`].
#[cfg(feature = "image-io")]
fn se3_from_t_bs(t_bs: &Matrix4<f64>) -> SE3 {
    let rotation_matrix = t_bs.fixed_view::<3, 3>(0, 0).into_owned();
    let translation = Vector3::new(t_bs[(0, 3)], t_bs[(1, 3)], t_bs[(2, 3)]);
    let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
    SE3::new(rotation, translation)
}

/// Build a [`Camera`] from EuRoC's pinhole `(fu, fv, cu, cv)` intrinsics.
/// Distortion coefficients are intentionally ignored (see file docs).
#[cfg(feature = "image-io")]
fn camera_from_cam0(cam0: &EurocCameraCalibration, camera_id: u64) -> Camera {
    let (fu, fv, cu, cv) = (
        cam0.intrinsics[0],
        cam0.intrinsics[1],
        cam0.intrinsics[2],
        cam0.intrinsics[3],
    );
    Camera::pinhole(
        camera_id,
        cam0.resolution.0,
        cam0.resolution.1,
        fu,
        fv,
        cu,
        cv,
    )
}

/// Back-project a pixel into the world frame at fixed depth, given the
/// world-to-camera pose. The ray from the camera centre through `pixel`,
/// taken to camera-frame depth `depth_meters`, is mapped into the world
/// frame via the inverse of the supplied pose.
#[cfg(feature = "image-io")]
fn back_project_pixel_to_world(
    camera: &Camera,
    pose_world_to_camera: &Pose,
    pixel: &Point2<f64>,
    depth_meters: f64,
) -> Option<Point3<f64>> {
    let normalized = camera.normalize_pixel(pixel)?;
    let p_cam = Point3::new(
        normalized.x * depth_meters,
        normalized.y * depth_meters,
        depth_meters,
    );
    let r_cw = pose_world_to_camera.world_to_camera.rotation;
    let t_cw = pose_world_to_camera.world_to_camera.translation;
    let r_wc = r_cw.inverse();
    // World point = R_cw⁻¹ · (p_cam - t_cw).
    Some(r_wc.transform_point(&Point3::from(p_cam.coords - t_cw)))
}

/// Cam1 camera model, cam0-to-cam1 extrinsic, and cam1 distortion model,
/// computed once and shared by the seed-frame stereo bootstrap and the
/// per-frame stereo landmark replenishment path.
#[cfg(feature = "image-io")]
struct Cam1StereoSetup {
    camera: Camera,
    cam0_to_cam1: SE3,
    distortion: RadialTangential,
}

/// Build a [`VisualMap`] by seeding one landmark per keypoint in
/// `features`. When `stereo_world_points[i]` is `Some(world_point)`, that
/// metric-scale 3D point is used directly (typically the output of a
/// stereo-triangulation bootstrap); otherwise the keypoint is
/// back-projected through `pose_world_to_camera` at
/// `bootstrap_depth_meters` as the fixed-depth fall-back. The corner
/// descriptor is attached to every seeded landmark.
///
/// `stereo_world_points` must be either empty (no override) or the same
/// length as `features.keypoints`.
#[cfg(feature = "image-io")]
fn bootstrap_map_from_first_frame(
    camera: &Camera,
    pose_world_to_camera: &Pose,
    features: &FeatureSet,
    bootstrap_depth_meters: f64,
    stereo_world_points: &[Option<Point3<f64>>],
    stereo_world_covariances: &[Option<nalgebra::Matrix3<f64>>],
    strict_stereo: bool,
) -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let use_overrides = !stereo_world_points.is_empty();
    if use_overrides {
        debug_assert_eq!(stereo_world_points.len(), features.keypoints.len());
        debug_assert_eq!(stereo_world_covariances.len(), features.keypoints.len());
    }
    for (index, (keypoint, descriptor)) in features
        .keypoints
        .iter()
        .zip(features.descriptors.iter())
        .enumerate()
    {
        let world_point = if use_overrides && stereo_world_points[index].is_some() {
            stereo_world_points[index].unwrap()
        } else if strict_stereo {
            // Phase-23 #2: strict-stereo mode drops keypoints without a
            // triangulated cam0↔cam1 depth, refusing to fall back to
            // the fixed `bootstrap_depth_meters` back-projection. The
            // resulting map is smaller but every landmark has a real
            // metric depth, which empirically lifts recovery-PnP
            // quality (post-cliff recovery, loop-closure verifier) by
            // avoiding the cross-attitude descriptor mismatch that
            // dominates when stale fixed-depth landmarks pollute the
            // matcher.
            continue;
        } else {
            let Some(point) = back_project_pixel_to_world(
                camera,
                pose_world_to_camera,
                keypoint,
                bootstrap_depth_meters,
            ) else {
                continue;
            };
            point
        };
        let mut landmark = Landmark::new(index as u64 + 1, world_point);
        landmark.descriptor = Some(descriptor.clone());
        if use_overrides {
            if let Some(covariance_world) = stereo_world_covariances[index] {
                map.landmark_position_covariances
                    .insert(landmark.id, covariance_world);
            }
        }
        map.landmarks.insert(landmark.id, landmark);
    }
    map
}

/// Mirror of [`OnlineSlamPipeline`]'s private `keyframe_from_tracking_result`.
#[cfg(feature = "image-io")]
fn keyframe_from_tracking_result(frame: &Frame, tracking: &TrackingResult) -> Keyframe {
    let mut frame = frame.clone();
    frame.pose = tracking.localization.pose.clone();

    let observations = tracking
        .localization
        .inlier_query_indices
        .iter()
        .zip(tracking.localization.inlier_landmark_ids.iter())
        .filter_map(|(keypoint_index, landmark_id)| {
            frame.keypoints.get(*keypoint_index).map(|xy| Observation {
                frame_id: frame.id,
                landmark_id: *landmark_id,
                keypoint_index: *keypoint_index,
                xy: *xy,
            })
        })
        .collect();

    Keyframe {
        frame,
        observations,
    }
}

/// Preserve the last successfully tracked frame at an Atlas boundary even
/// when the online keyframe policy did not promote it. DROID-style frame
/// graphs retain this overlap; dropping it can leave the next independent
/// submap with only a much older target keyframe.
#[cfg(feature = "image-io")]
fn insert_atlas_boundary_keyframe_snapshot(map: &mut VisualMap, keyframe: &Keyframe) -> bool {
    if map.keyframes.contains_key(&keyframe.frame.id) || keyframe.observations.is_empty() {
        return false;
    }

    let mut snapshot = keyframe.clone();
    snapshot
        .observations
        .retain(|observation| map.landmarks.contains_key(&observation.landmark_id));
    if snapshot.observations.is_empty() {
        return false;
    }
    for observation in &snapshot.observations {
        let landmark = map
            .landmarks
            .get_mut(&observation.landmark_id)
            .expect("snapshot observations were filtered against this map");
        if !landmark.observations.iter().any(|existing| {
            existing.frame_id == observation.frame_id
                && existing.landmark_id == observation.landmark_id
        }) {
            landmark.observations.push(observation.clone());
        }
    }
    map.keyframes.insert(snapshot.frame.id, snapshot);
    true
}

/// Append stereo-triangulated landmarks to an existing map. Returns the
/// new landmark ids in `matches` order.
#[cfg(feature = "image-io")]
fn append_stereo_bootstrap_landmarks_to_map(
    map: &mut VisualMap,
    pose_world_to_camera: &Pose,
    features: &FeatureSet,
    matches: &[StereoBootstrapLandmark],
    next_landmark_id: &mut u64,
) -> Vec<u64> {
    let cam0_pose_camera_to_world = pose_world_to_camera.camera_to_world();
    let mut landmark_ids = Vec::with_capacity(matches.len());
    for survivor in matches {
        let world_point =
            cam0_pose_camera_to_world.transform_point(&survivor.point_left_camera_frame);
        let id = *next_landmark_id;
        *next_landmark_id += 1;
        let mut landmark = Landmark::new(id, world_point);
        landmark.descriptor = Some(features.descriptors[survivor.left_keypoint_index].clone());
        let rotation_camera_to_world = cam0_pose_camera_to_world
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let covariance_world = rotation_camera_to_world
            * survivor.point_covariance_left_camera_frame
            * rotation_camera_to_world.transpose();
        map.landmark_position_covariances
            .insert(id, covariance_world);
        map.landmarks.insert(id, landmark);
        landmark_ids.push(id);
    }
    landmark_ids
}

/// Build a successful [`TrackingResult`] for a stereo segment restart.
#[cfg(feature = "image-io")]
fn build_stereo_segment_restart_tracking_result(
    frame_id: u64,
    seed_pose: Pose,
    matches: &[StereoBootstrapLandmark],
    landmark_ids: &[u64],
    map_stats: MapProviderStats,
) -> TrackingResult {
    let inlier_count = matches.len();
    let inlier_query_indices: Vec<usize> = matches.iter().map(|m| m.left_keypoint_index).collect();
    let inlier_landmark_ids = landmark_ids.to_vec();
    let localization = LocalizationResult {
        success: true,
        pose: Some(seed_pose.clone()),
        failure_reason: None,
        candidate_landmark_count: inlier_count,
        match_count: inlier_count,
        correspondence_count: inlier_count,
        inlier_count,
        outlier_count: 0,
        inlier_ratio: 1.0,
        reprojection_error: Some(0.0),
        median_reprojection_error: Some(0.0),
        max_reprojection_error: Some(0.0),
        inlier_reprojection_errors: vec![0.0; inlier_count],
        inliers: (0..inlier_count).collect(),
        inlier_query_indices,
        inlier_landmark_ids,
        inlier_confidences: vec![None; inlier_count],
        estimator_diagnostics: None,
        pose_failure_diagnostics: None,
    };
    TrackingResult {
        frame_id,
        state: TrackingState::Tracking,
        event: TrackingEvent::Relocalized,
        successive_failures: 0,
        pose_prior: Some(seed_pose),
        used_pose_prior: false,
        used_external_localization_prior: false,
        external_localization_prior_radius: None,
        tracking_failure_reason: None,
        map_landmark_count: map_stats.landmark_count,
        map_stats,
        localization,
        covisibility_local_map_size: None,
    }
}

#[cfg(feature = "image-io")]
fn map_provider_stats_from_map(map: &VisualMap) -> MapProviderStats {
    MapProviderStats {
        camera_count: map.cameras.len(),
        landmark_count: map.landmarks.len(),
        keyframe_count: map.keyframes.len(),
        descriptor_count: map
            .landmarks
            .values()
            .filter(|landmark| landmark.descriptor.is_some())
            .count(),
    }
}

/// Return a new [`FeatureSet`] whose keypoints have been mapped from
/// their raw distorted pixel positions to the "ideal pinhole" pixel
/// positions implied by cam0's calibration. Descriptors are copied
/// verbatim — they are extracted from raw image patches and the
/// undistortion only corrects the *position* at which each descriptor
/// was sampled. Returns the input unchanged when `distortion` is the
/// identity (zero-coefficient case) so the no-op path is allocation-free
/// modulo the clone.
///
/// A keypoint whose `undistort_pixel` returns `None` (camera intrinsics
/// missing, which should not happen for the EuRoC `Pinhole` camera) is
/// silently dropped along with its descriptor — keypoints / descriptors
/// must stay in lock-step so a half-failed undistortion can never poison
/// the matcher's index-based lookup.
#[cfg(feature = "image-io")]
fn undistort_feature_keypoints(
    distortion: &RadialTangential,
    camera: &Camera,
    features: &FeatureSet,
) -> FeatureSet {
    if distortion.is_identity() {
        return features.clone();
    }
    let mut keypoints = Vec::with_capacity(features.keypoints.len());
    let mut descriptors = Vec::with_capacity(features.descriptors.len());
    for (kp, desc) in features.keypoints.iter().zip(features.descriptors.iter()) {
        let Some(undistorted) = distortion.undistort_pixel(camera, *kp) else {
            continue;
        };
        keypoints.push(undistorted);
        descriptors.push(desc.clone());
    }
    FeatureSet {
        keypoints,
        descriptors,
    }
}

/// Adapt a [`FeatureSet`] into a [`Frame`] addressed to the cam0 camera id.
#[cfg(feature = "image-io")]
fn frame_from_features(frame_id: u64, camera_id: u64, features: &FeatureSet) -> Frame {
    let mut frame = Frame::new(frame_id, camera_id);
    frame.keypoints = features.keypoints.clone();
    frame.descriptors = features.descriptors.clone();
    frame
}

#[cfg(feature = "image-io")]
fn normalized_mean_descriptor(descriptors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = descriptors.first()?;
    if first.is_empty() {
        return None;
    }
    let dim = first.len();
    let mut mean = vec![0.0f32; dim];
    let mut count = 0usize;
    for descriptor in descriptors {
        if descriptor.len() != dim {
            return None;
        }
        for (acc, value) in mean.iter_mut().zip(descriptor) {
            *acc += *value;
        }
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let inv = 1.0 / count as f32;
    for value in &mut mean {
        *value *= inv;
    }
    let norm = mean.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut mean {
            *value /= norm;
        }
    }
    Some(mean)
}

#[cfg(feature = "image-io")]
fn nearest_ground_truth(
    samples: &[EurocGroundTruthSample],
    target_ts: i128,
) -> Option<&EurocGroundTruthSample> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples
        .binary_search_by_key(&target_ts, |sample| sample.timestamp_nanoseconds)
        .unwrap_or_else(|insert| {
            if insert == 0 {
                0
            } else if insert >= samples.len() {
                samples.len() - 1
            } else {
                let before = samples[insert - 1].timestamp_nanoseconds;
                let after = samples[insert].timestamp_nanoseconds;
                if (target_ts - before).abs() <= (after - target_ts).abs() {
                    insert - 1
                } else {
                    insert
                }
            }
        });
    samples.get(idx)
}

#[cfg(feature = "image-io")]
fn format_vi_init_event(event: &ViInitializationEvent) -> String {
    match event {
        ViInitializationEvent::StillBuffering { reason } => {
            format!("StillBuffering reason={reason:?}")
        }
        ViInitializationEvent::Succeeded {
            result,
            first_keyframe_id,
            discarded_stale_factor_count,
        } => format!(
            "Succeeded first_keyframe={first_keyframe_id:?} discarded_stale={discarded_stale_factor_count} bias_gyro={:?} bias_acc={:?} rotation_angle_deg={:.4}",
            result.bias_gyro.as_slice(),
            result.bias_acc.as_slice(),
            result
                .initial_rotation_body_to_world
                .angle()
                .to_degrees(),
        ),
        ViInitializationEvent::GaveUp {
            last_reason,
            fallback,
        } => format!(
            "GaveUp last_reason={last_reason:?} fallback={fallback:?}",
        ),
    }
}

/// `last_gravity_alignment` is the initializer's most recent
/// `estimate_gravity_and_velocities` attempt (see
/// `MotionBasedViInitializer::last_gravity_alignment`), sampled at the same
/// frame as `event`. It is `None` when `--motion-vi-init-estimate-gravity`
/// is off. Threading it through separately from `event` is deliberate: on
/// real data most attempts are rejected by a downstream residual gate
/// (`ImuRawResidualOutOfRange` etc.), which carries no gravity vector of its
/// own, so without this the recovered estimate would be invisible on every
/// `StillWaiting` line — see `docs/motion_based_vi_alignment.md`'s
/// "Gravity-direction recovery" section.
/// `last_gyro_bias_alignment` is the initializer's most recent
/// `estimate_gyro_bias` attempt (see
/// `MotionBasedViInitializer::last_gyro_bias_alignment`), sampled at the
/// same frame as `event`, threaded through for the same reason as
/// `last_gravity_alignment` (see that parameter's doc comment): most
/// real-data attempts are rejected by a downstream gate that carries no
/// gyro-bias vector of its own.
#[cfg(feature = "image-io")]
fn format_motion_vi_init_event(
    event: &MotionViInitializationEvent,
    last_gravity_alignment: Option<&GravityVelocityAlignment>,
    last_gyro_bias_alignment: Option<&GyroBiasAlignment>,
) -> String {
    match event {
        MotionViInitializationEvent::StillWaiting { reason } => {
            let alignment = last_gravity_alignment
                .map(|a| {
                    format!(
                        " gravity_alignment_gravity_world=[{:.6},{:.6},{:.6}] gravity_alignment_raw_norm={:.6} gravity_alignment_mean_residual_after={:.6} gravity_alignment_window_keyframes={}",
                        a.gravity_world.x,
                        a.gravity_world.y,
                        a.gravity_world.z,
                        a.raw_gravity_norm,
                        a.mean_residual_after,
                        a.window_keyframes,
                    )
                })
                .unwrap_or_default();
            let gyro_bias_alignment = last_gyro_bias_alignment
                .map(|a| {
                    format!(
                        " gyro_bias_alignment_bias_gyro=[{:.6},{:.6},{:.6}] gyro_bias_alignment_iterations={} gyro_bias_alignment_rms_before={:.6} gyro_bias_alignment_rms_after={:.6}",
                        a.bias_gyro.x,
                        a.bias_gyro.y,
                        a.bias_gyro.z,
                        a.iterations,
                        a.rotation_residual_rms_before,
                        a.rotation_residual_rms_after,
                    )
                })
                .unwrap_or_default();
            format!("StillWaiting reason={reason:?}{alignment}{gyro_bias_alignment}")
        }
        MotionViInitializationEvent::Succeeded { result } => format!(
            "Succeeded keyframes={} imu_factors={} scale={:.6} viba2_iters={} trigger_translation_m={:.3} bias_released={} estimated_gravity_world={:?} estimated_gyro_bias={:?}",
            result.keyframe_ids.len(),
            result.imu_factors_used,
            result.scale,
            result.viba2_iterations_run,
            result.trigger_translation_meters,
            result.bias_released,
            result.estimated_gravity_world.as_ref().map(|g| g.as_slice().to_vec()),
            result.estimated_gyro_bias.as_ref().map(|b| b.as_slice().to_vec()),
        ),
    }
}

#[cfg(feature = "image-io")]
fn format_optional_frame_id(frame_id: Option<u64>) -> String {
    frame_id.map(|id| id.to_string()).unwrap_or_default()
}

#[cfg(feature = "image-io")]
fn format_optional_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

#[cfg(feature = "image-io")]
fn format_optional_f64(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.9}")).unwrap_or_default()
}

#[cfg(feature = "image-io")]
fn bool_as_u8(value: bool) -> u8 {
    if value {
        1
    } else {
        0
    }
}

#[cfg(feature = "image-io")]
fn quote_csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(feature = "image-io")]
fn append_atlas_alignment_attempt(
    csv: &mut String,
    attempt_idx: usize,
    frame_idx: usize,
    timestamp_ns: i128,
    alignment: &CrossSubmapAlignmentResult,
) {
    if alignment.diagnostics.is_empty() {
        csv.push_str(&format!(
            "{attempt_idx},{frame_idx},{timestamp_ns},{},{},,,,,,,,,,,,,,,0,no_ranked_candidate\n",
            alignment.source_submap_id, alignment.source_frame_id,
        ));
        return;
    }
    for diagnostic in &alignment.diagnostics {
        let scale = diagnostic.scale_estimate;
        let failure = diagnostic
            .failure_reason
            .as_ref()
            .map(|reason| quote_csv_field(&format!("{reason:?}")))
            .unwrap_or_default();
        let verified = alignment.verified_merge.as_ref().is_some_and(|merge| {
            merge.evidence.target_submap_id == diagnostic.target_submap_id
                && diagnostic.failure_reason.is_none()
        });
        let weldable_landmark_matches = alignment
            .verified_merge
            .as_ref()
            .filter(|merge| {
                merge.evidence.target_submap_id == diagnostic.target_submap_id
                    && diagnostic.failure_reason.is_none()
            })
            .map(|merge| merge.landmark_matches.len().to_string())
            .unwrap_or_default();
        csv.push_str(&format!(
            "{attempt_idx},{frame_idx},{timestamp_ns},{},{},{},{},{:.6},{},{},{},{},{:.6},{},{},{},{},{},{},{},{}\n",
            alignment.source_submap_id,
            alignment.source_frame_id,
            diagnostic.target_submap_id,
            diagnostic.target_frame_id,
            diagnostic.appearance_similarity,
            diagnostic.candidate_landmark_count,
            u8::from(diagnostic.used_projection_prior),
            diagnostic.localization_correspondence_count,
            diagnostic.localization_inlier_count,
            diagnostic.localization_inlier_ratio,
            diagnostic
                .mean_reprojection_error_px
                .map(|value| format!("{value:.6}"))
                .unwrap_or_default(),
            scale
                .map(|estimate| format!("{:.9}", estimate.estimated_scale))
                .unwrap_or_default(),
            scale
                .map(|estimate| estimate.pair_count.to_string())
                .unwrap_or_default(),
            scale
                .map(|estimate| format!("{:.6}", estimate.inlier_ratio))
                .unwrap_or_default(),
            scale
                .map(|estimate| format!("{:.9}", estimate.median_absolute_deviation))
                .unwrap_or_default(),
            weldable_landmark_matches,
            u8::from(verified),
            failure,
        ));
    }
}

#[cfg(feature = "image-io")]
fn append_atlas_boundary_factor(
    csv: &mut String,
    frame_idx: usize,
    timestamp_ns: i128,
    factor: &CrossSubmapBoundaryFactorResult,
) {
    let failure = factor
        .failure_reason
        .as_ref()
        .map(|reason| quote_csv_field(&format!("{reason:?}")))
        .unwrap_or_default();
    let scale = factor.scale_estimate;
    csv.push_str(&format!(
        "{frame_idx},{timestamp_ns},{},{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{}\n",
        factor.source_submap_id,
        factor.source_frame_id,
        factor.target_submap_id,
        factor.target_frame_id,
        factor.descriptor_match_count,
        factor.spatial_mutual_match_count,
        factor.metric_correspondence_count,
        factor.projection_refined_correspondence_count,
        factor.projection_refinement_iterations,
        bool_as_u8(factor.used_transform_prior),
        factor.rigid_inlier_count,
        factor.rigid_inlier_ratio,
        factor
            .mean_target_reprojection_error_px
            .map(|value| format!("{value:.6}"))
            .unwrap_or_default(),
        scale
            .map(|estimate| format!("{:.9}", estimate.estimated_scale))
            .unwrap_or_default(),
        scale
            .map(|estimate| estimate.pair_count.to_string())
            .unwrap_or_default(),
        scale
            .map(|estimate| format!("{:.6}", estimate.inlier_ratio))
            .unwrap_or_default(),
        factor
            .verified_merge
            .as_ref()
            .map(|merge| merge.landmark_matches.len().to_string())
            .unwrap_or_default(),
        bool_as_u8(factor.verified_merge.is_some()),
        failure,
    ));
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RelocalizationAttemptGateStatus {
    success_pass: bool,
    min_inliers_pass: bool,
    min_inlier_ratio_pass: bool,
    reprojection_pass: bool,
    continuity_pass: bool,
    depth_ratio_pass: bool,
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq)]
struct RelocalizationAttemptGateConfig {
    min_inliers: usize,
    min_inlier_ratio: f64,
    max_reprojection_error: Option<f64>,
    max_translation_per_frame_from_last_success_meters: Option<f64>,
    min_inlier_depth_median_ratio_to_last_success: Option<f64>,
    max_inlier_depth_median_ratio_to_last_success: Option<f64>,
}

#[cfg(feature = "image-io")]
impl RelocalizationAttemptGateConfig {
    fn from_args(args: &CliArgs) -> Self {
        Self {
            min_inliers: args.relocalization_min_inliers,
            min_inlier_ratio: args.relocalization_min_inlier_ratio,
            max_reprojection_error: args.relocalization_max_reprojection_error,
            max_translation_per_frame_from_last_success_meters: args
                .relocalization_max_translation_per_frame_from_last_success_meters,
            min_inlier_depth_median_ratio_to_last_success: args
                .relocalization_min_inlier_depth_median_ratio_to_last_success,
            max_inlier_depth_median_ratio_to_last_success: args
                .relocalization_max_inlier_depth_median_ratio_to_last_success,
        }
    }
}

#[cfg(feature = "image-io")]
fn relocalization_attempt_gate_status(
    stats: &OnlineSlamRelocalizationStats,
    config: RelocalizationAttemptGateConfig,
) -> RelocalizationAttemptGateStatus {
    let reprojection_pass = match (config.max_reprojection_error, stats.mean_reprojection_error) {
        (Some(max), Some(actual)) => actual <= max,
        (Some(_), None) => false,
        (None, _) => true,
    };
    let continuity_pass = match (
        config.max_translation_per_frame_from_last_success_meters,
        stats.translation_per_frame_from_last_success_meters,
    ) {
        (Some(max), Some(actual)) => actual <= max,
        (Some(_), None) => true,
        (None, _) => true,
    };
    let depth_ratio_pass = match stats.inlier_depth_median_ratio_to_last_success {
        Some(ratio) => {
            config
                .min_inlier_depth_median_ratio_to_last_success
                .is_none_or(|min| ratio >= min)
                && config
                    .max_inlier_depth_median_ratio_to_last_success
                    .is_none_or(|max| ratio <= max)
        }
        None => true,
    };
    RelocalizationAttemptGateStatus {
        success_pass: stats.localization_success,
        min_inliers_pass: stats.inlier_count >= config.min_inliers,
        min_inlier_ratio_pass: stats.inlier_ratio >= config.min_inlier_ratio,
        reprojection_pass,
        continuity_pass,
        depth_ratio_pass,
    }
}

#[cfg(feature = "image-io")]
fn relocalization_attempt_reject_reason(
    stats: &OnlineSlamRelocalizationStats,
    gates: RelocalizationAttemptGateStatus,
) -> &'static str {
    if stats.succeeded {
        "accepted"
    } else if !gates.success_pass {
        "no_pnp_solution"
    } else if !gates.min_inliers_pass {
        "min_inliers"
    } else if !gates.min_inlier_ratio_pass {
        "min_inlier_ratio"
    } else if !gates.reprojection_pass {
        "max_reprojection_error"
    } else if !gates.continuity_pass {
        "translation_per_frame_from_last_success"
    } else if !gates.depth_ratio_pass {
        "inlier_depth_median_ratio_to_last_success"
    } else if stats.passed_acceptance_gates
        && stats.confirmation_count < stats.confirmation_required_count
    {
        "confirmation_waiting"
    } else if stats.passed_acceptance_gates {
        "tracker_acceptance"
    } else {
        "unknown_gate"
    }
}

#[cfg(feature = "image-io")]
fn keyframe_reason_csv_fields(
    reason: &KeyframeDecisionReason,
) -> (&'static str, String, String, String, String, String, String) {
    match reason {
        KeyframeDecisionReason::NotLocalized => (
            "NotLocalized",
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::MissingPose => (
            "MissingPose",
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::FirstSuccessfulFrame => (
            "FirstSuccessfulFrame",
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::Relocalized => (
            "Relocalized",
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::TrackedLandmarkDrop {
            frame_id_gap,
            tracked_landmarks,
            last_keyframe_tracked_landmarks,
            min_tracked_landmark_ratio,
        } => (
            "TrackedLandmarkDrop",
            frame_id_gap.to_string(),
            String::new(),
            String::new(),
            tracked_landmarks.to_string(),
            last_keyframe_tracked_landmarks.to_string(),
            format!("{min_tracked_landmark_ratio:.6}"),
        ),
        KeyframeDecisionReason::ThresholdsMet {
            frame_id_gap,
            translation,
        } => (
            "ThresholdsMet",
            frame_id_gap.to_string(),
            format!("{translation:.6}"),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::FrameIdGapTooSmall {
            frame_id_gap,
            min_frame_id_gap: _,
        } => (
            "FrameIdGapTooSmall",
            frame_id_gap.to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::MaximumFrameIdGap {
            frame_id_gap,
            max_frame_id_gap: _,
        } => (
            "MaximumFrameIdGap",
            frame_id_gap.to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::TranslationTooSmall {
            translation,
            min_translation,
        } => (
            "TranslationTooSmall",
            String::new(),
            format!("{translation:.6}"),
            format!("{min_translation:.6}"),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::InsufficientTrackingQuality { .. } => (
            "InsufficientTrackingQuality",
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
    }
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let demo_started_at = Instant::now();
    let args = parse_args()?;
    let dataset = read_euroc_dataset_dir(&args.euroc_dir)?;
    println!(
        "loaded euroc cam0_frames={} imu_samples={} gt_samples={} cam0_rate={:.1}Hz imu_rate={:.1}Hz",
        dataset.cam0_images.len(),
        dataset.imu_samples.len(),
        dataset.ground_truth.len(),
        dataset.cam0_calibration.rate_hz,
        dataset.imu_calibration.rate_hz,
    );
    let evaluation_ground_truth_available = !dataset.ground_truth.is_empty();
    if !evaluation_ground_truth_available {
        println!(
            "ground truth absent: running estimation-only; ATE/RPE and loop correctness fields will be unavailable"
        );
    }
    if dataset.imu_samples.len() < 100 {
        return Err(format!(
            "too few IMU samples ({}) — is this an EuRoC recording?",
            dataset.imu_samples.len()
        )
        .into());
    }

    let camera_id: u64 = 1;
    let camera = camera_from_cam0(&dataset.cam0_calibration, camera_id);
    let body_to_camera = se3_from_t_bs(&dataset.cam0_calibration.t_body_sensor);
    let distortion = if args.undistort {
        RadialTangential::from_euroc_coefficients(
            &dataset.cam0_calibration.distortion_coefficients,
        )
        .ok_or_else(|| {
            format!(
                "cam0 distortion_coefficients has unexpected length ({}); expected 4 for radial-tangential. Pass --no-undistort to skip the correction.",
                dataset.cam0_calibration.distortion_coefficients.len(),
            )
        })?
    } else {
        RadialTangential::IDENTITY
    };
    println!(
        "cam0 fx={} fy={} cx={} cy={} resolution={}x{} body_to_camera_t=[{:.3},{:.3},{:.3}] undistort={} distortion_model={} distortion_coefficients={:?}",
        camera.params[0],
        camera.params[1],
        camera.params[2],
        camera.params[3],
        camera.width,
        camera.height,
        body_to_camera.translation.x,
        body_to_camera.translation.y,
        body_to_camera.translation.z,
        args.undistort,
        dataset.cam0_calibration.distortion_model,
        dataset.cam0_calibration.distortion_coefficients,
    );

    let imu_first_ts = dataset.imu_samples.first().unwrap().timestamp_nanoseconds;
    let imu_last_ts = dataset.imu_samples.last().unwrap().timestamp_nanoseconds;
    // Establish an arbitrary SLAM gauge at the first camera frame covered by
    // the IMU stream. Stereo depth makes the gauge metric; GT is never needed.
    let seed_frame_idx = dataset
        .cam0_images
        .iter()
        .position(|entry| {
            entry.timestamp_nanoseconds >= imu_first_ts
                && entry.timestamp_nanoseconds <= imu_last_ts
        })
        .ok_or("camera and IMU streams do not overlap")?;
    let seed_image_entry = &dataset.cam0_images[seed_frame_idx];
    let seed_image_path = dataset.cam0_image_dir.join(&seed_image_entry.filename);
    let seed_image: GrayscaleImage = read_common_image(&seed_image_path).map_err(|err| {
        format!(
            "failed to read seed image {}: {err}",
            seed_image_path.display()
        )
    })?;

    let extractor = match args.feature_extractor {
        FeatureExtractorKind::Corner => {
            DemoExtractor::Corner(CornerFeatureExtractor::new(CornerFeatureConfig {
                max_features: args.corner_max_features,
                min_score: args.corner_min_score,
                descriptor_radius: args.corner_descriptor_radius,
            }))
        }
        FeatureExtractorKind::Hog => {
            DemoExtractor::Hog(HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
                max_features: args.hog_max_features,
                min_corner_score: args.hog_min_corner_score,
                orient: args.hog_orient,
                ..HogLikeFeatureConfig::default()
            }))
        }
        FeatureExtractorKind::SuperPointOffline => {
            let dir = args.superpoint_features_dir.as_ref().ok_or(
                "--feature-extractor superpoint-offline requires --superpoint-features-dir <path>",
            )?;
            let loaded = if let Some(cam1_dir) = args.superpoint_cam1_features_dir.as_ref() {
                let extractor = SuperPointOfflineExtractor::load_with_cam1(dir, cam1_dir)?;
                println!(
                    "loaded {} SuperPoint cam0 + cam1 feature files from {} & {}",
                    extractor.len(),
                    dir.display(),
                    cam1_dir.display(),
                );
                extractor
            } else {
                if args.stereo_bootstrap {
                    return Err(
                        "--feature-extractor superpoint-offline with --stereo-bootstrap requires --superpoint-cam1-features-dir <cam1_path>".into(),
                    );
                }
                let extractor = SuperPointOfflineExtractor::load_from_dir(dir)?;
                println!(
                    "loaded {} SuperPoint cam0 feature files from {}",
                    extractor.len(),
                    dir.display(),
                );
                extractor
            };
            if args.stereo_bootstrap && !loaded.has_cam1() {
                return Err(
                    "--stereo-bootstrap with superpoint-offline requires cam1 features; pass --superpoint-cam1-features-dir <path>".into(),
                );
            }
            DemoExtractor::SuperPointOffline(loaded)
        }
        FeatureExtractorKind::SuperPointOnnx => {
            let model_path = args.superpoint_onnx_model.as_ref().ok_or(
                "--feature-extractor superpoint-onnx requires --superpoint-onnx-model <path>",
            )?;
            #[cfg(feature = "onnx-inference")]
            let extractor = {
                use visloc_vision::features::superpoint_onnx::{
                    OnnxBackend, SuperPointOnnxConfig, SuperPointOnnxExtractor,
                };
                let backend = match args.superpoint_onnx_backend {
                    SuperPointOnnxBackendArg::CudaThenCpu => OnnxBackend::CudaThenCpu,
                    SuperPointOnnxBackendArg::Cuda => OnnxBackend::Cuda,
                    SuperPointOnnxBackendArg::Cpu => OnnxBackend::Cpu,
                };
                SuperPointOnnxExtractor::load_from_path_with_backend(
                    model_path,
                    SuperPointOnnxConfig::default(),
                    backend,
                )
                .map_err(|err| format!("SuperPoint ONNX load failed: {err}"))?
            };
            #[cfg(not(feature = "onnx-inference"))]
            let extractor =
                visloc_vision::features::superpoint_onnx::SuperPointOnnxExtractor::load_from_path(
                    model_path,
                    visloc_vision::features::superpoint_onnx::SuperPointOnnxConfig::default(),
                )
                .map_err(|err| format!("SuperPoint ONNX load_from_path failed: {err}"))?;
            println!(
                "loaded SuperPoint ONNX model from {} (backend={}, onnx-inference feature: {})",
                model_path.display(),
                args.superpoint_onnx_backend.as_str(),
                cfg!(feature = "onnx-inference"),
            );
            DemoExtractor::SuperPointOnnx(extractor)
        }
    };
    extractor.set_camera(SuperPointCamera::Cam0);
    extractor.set_frame_idx(seed_frame_idx);
    let seed_features_raw = extractor
        .extract(&seed_image)
        .map_err(|err| format!("seed-frame feature extraction failed: {err}"))?;
    let seed_features = undistort_feature_keypoints(&distortion, &camera, &seed_features_raw);
    println!(
        "seed frame_idx={seed_frame_idx} path={} extracted_features={} after_undistort={}",
        seed_image_path.display(),
        seed_features_raw.len(),
        seed_features.len(),
    );
    if seed_features.is_empty() {
        return Err(
            "seed-frame extractor returned 0 corners after undistort — try reducing --corner-min-score or passing --no-undistort".into(),
        );
    }
    let seed_pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());

    // Optional stereo bootstrap: pair the seed cam0 frame with the nearest
    // cam1 frame, match descriptors, and triangulate each surviving pair
    // via DLT in the cam0 frame. The resulting 3D points are transformed
    // into the world frame using `seed_pose` and override the
    // fixed-`bootstrap_depth` back-projection for the matched cam0
    // keypoints below.
    let mut stereo_world_points: Vec<Option<Point3<f64>>> =
        vec![None; seed_features.keypoints.len()];
    let mut stereo_world_covariances: Vec<Option<nalgebra::Matrix3<f64>>> =
        vec![None; seed_features.keypoints.len()];
    let mut stereo_right_pixels: Vec<Option<Point2<f64>>> =
        vec![None; seed_features.keypoints.len()];
    let mut stereo_bootstrap_matches: Vec<StereoBootstrapLandmark> = Vec::new();
    let mut stereo_cam1_features_count: usize = 0;
    let mut stereo_cam1_features_after_undistort_count: usize = 0;
    // Shared cam1 camera/extrinsic/distortion setup, computed once whenever
    // either the seed-frame stereo bootstrap or the per-frame stereo
    // landmark replenishment needs to stereo-match cam0 keypoints against
    // cam1.
    let cam1_stereo_setup: Option<Cam1StereoSetup> = if args.stereo_bootstrap
        || args.stereo_landmark_replenish
    {
        let cam1_camera_id: u64 = 2;
        let cam1_camera = camera_from_cam0(&dataset.cam1_calibration, cam1_camera_id);
        let body_to_cam1 = se3_from_t_bs(&dataset.cam1_calibration.t_body_sensor);
        let cam0_to_cam1 = body_to_cam1.inverse().compose(&body_to_camera);
        let cam1_distortion = if args.undistort {
            RadialTangential::from_euroc_coefficients(
                    &dataset.cam1_calibration.distortion_coefficients,
                )
                .ok_or_else(|| {
                    format!(
                        "cam1 distortion_coefficients has unexpected length ({}); expected 4 for radial-tangential. Pass --no-undistort (or --no-stereo-bootstrap / --no stereo-landmark-replenish) to skip.",
                        dataset.cam1_calibration.distortion_coefficients.len(),
                    )
                })?
        } else {
            RadialTangential::IDENTITY
        };
        Some(Cam1StereoSetup {
            camera: cam1_camera,
            cam0_to_cam1,
            distortion: cam1_distortion,
        })
    } else {
        None
    };
    if args.stereo_bootstrap {
        let cam1_setup = cam1_stereo_setup
            .as_ref()
            .expect("computed above whenever stereo_bootstrap is set");
        let cam1_camera = &cam1_setup.camera;
        let cam0_to_cam1 = &cam1_setup.cam0_to_cam1;
        let cam1_distortion = &cam1_setup.distortion;
        let cam1_seed_idx = dataset
            .cam1_images
            .iter()
            .position(|entry| entry.timestamp_nanoseconds == seed_image_entry.timestamp_nanoseconds)
            .ok_or_else(|| {
                format!(
                    "no cam1 frame matches the cam0 seed timestamp {}; pass --no-stereo-bootstrap to skip.",
                    seed_image_entry.timestamp_nanoseconds,
                )
            })?;
        let cam1_image_entry = &dataset.cam1_images[cam1_seed_idx];
        let cam1_image_path = dataset.cam1_image_dir.join(&cam1_image_entry.filename);
        let cam1_image: GrayscaleImage = read_common_image(&cam1_image_path).map_err(|err| {
            format!(
                "failed to read cam1 seed image {}: {err}",
                cam1_image_path.display()
            )
        })?;
        // For the offline-replay path, switch the extractor to cam1
        // and to the cam1 seed frame index. The cam0/cam1 SuperPoint
        // pre-exports are aligned by frame index (the EuRoC streams
        // share a 20 Hz cadence), so the cam1 features at
        // `cam1_seed_idx` correspond to the cam0 seed timestamp.
        extractor.set_camera(SuperPointCamera::Cam1);
        extractor.set_frame_idx(cam1_seed_idx);
        let cam1_features_raw = extractor
            .extract(&cam1_image)
            .map_err(|err| format!("cam1 seed-frame feature extraction failed: {err}"))?;
        // Restore the loop-path defaults so subsequent cam0 extracts
        // pick up the correct stream.
        extractor.set_camera(SuperPointCamera::Cam0);
        stereo_cam1_features_count = cam1_features_raw.len();
        let cam1_features =
            undistort_feature_keypoints(cam1_distortion, cam1_camera, &cam1_features_raw);
        stereo_cam1_features_after_undistort_count = cam1_features.len();
        stereo_bootstrap_matches = bootstrap_stereo_landmarks(
            &camera,
            cam1_camera,
            cam0_to_cam1,
            &seed_features,
            &cam1_features,
            &StereoBootstrapConfig::default(),
        );
        let cam0_pose_camera_to_world = seed_pose.camera_to_world();
        let rotation_camera_to_world = cam0_pose_camera_to_world
            .rotation
            .to_rotation_matrix()
            .into_inner();
        for survivor in &stereo_bootstrap_matches {
            let world_point =
                cam0_pose_camera_to_world.transform_point(&survivor.point_left_camera_frame);
            stereo_world_points[survivor.left_keypoint_index] = Some(world_point);
            stereo_world_covariances[survivor.left_keypoint_index] = Some(
                rotation_camera_to_world
                    * survivor.point_covariance_left_camera_frame
                    * rotation_camera_to_world.transpose(),
            );
            stereo_right_pixels[survivor.left_keypoint_index] =
                Some(cam1_features.keypoints[survivor.right_keypoint_index]);
        }
        println!(
            "stereo_bootstrap cam1_seed_idx={cam1_seed_idx} cam1_features={} after_undistort={} triangulated_matches={}",
            stereo_cam1_features_count,
            stereo_cam1_features_after_undistort_count,
            stereo_bootstrap_matches.len(),
        );
    }

    let mut map = bootstrap_map_from_first_frame(
        &camera,
        &seed_pose,
        &seed_features,
        args.bootstrap_depth_meters,
        &stereo_world_points,
        &stereo_world_covariances,
        args.stereo_bootstrap_strict,
    );
    let mut seed_stereo_observations = Vec::new();
    if let Some(cam1_setup) = cam1_stereo_setup.as_ref() {
        map.cameras
            .insert(cam1_setup.camera.id, cam1_setup.camera.clone());
        for survivor in &stereo_bootstrap_matches {
            let landmark_id = survivor.left_keypoint_index as u64 + 1;
            let Some(xy_right) = stereo_right_pixels[survivor.left_keypoint_index] else {
                continue;
            };
            if map.landmarks.contains_key(&landmark_id) {
                seed_stereo_observations.push(StereoObservation {
                    frame_id: seed_frame_idx as u64,
                    landmark_id,
                    right_camera_id: cam1_setup.camera.id,
                    xy_right,
                    left_to_right: cam1_setup.cam0_to_cam1.clone(),
                });
            }
        }
    }
    let bootstrap_landmark_count = map.landmarks.len();
    println!(
        "bootstrap landmarks={} bootstrap_depth={:.2}m stereo_overrides={} seed_t_ns={} seed_pose_source=identity_first_camera",
        map.landmarks.len(),
        args.bootstrap_depth_meters,
        stereo_bootstrap_matches.len(),
        seed_image_entry.timestamp_nanoseconds,
    );

    let mut initializer_config = VisualInertialInitializerConfig {
        gravity_world: args.gravity_world,
        ..VisualInertialInitializerConfig::default()
    };
    if let Some(limit) = args.vi_init_gyro_std_limit {
        initializer_config.max_gyro_std = limit;
    }
    if let Some(limit) = args.vi_init_mean_gyro_magnitude_limit {
        initializer_config.max_mean_gyro_magnitude = limit;
    }
    if let Some(limit) = args.vi_init_accel_std_limit {
        initializer_config.max_accel_std = limit;
    }
    if let Some(min_samples) = args.vi_init_min_samples {
        initializer_config.min_samples = min_samples;
    }
    if let Some(window) = args.vi_init_min_stationary_window_seconds {
        initializer_config.min_stationary_window_seconds = window;
    }
    let vi_init_config = OnlineSlamViInitConfig {
        initializer: initializer_config,
        body_to_camera: body_to_camera.clone(),
        seed_first_keyframe_rotation: true,
        on_persistent_rejection: ViInitFallback::KeepExistingSeed,
        max_wait_duration_seconds: args.vi_init_max_wait_seconds,
        max_buffered_samples: 4000,
        try_initialize_on_every_frame: args.vi_init_try_initialize_on_every_frame,
    };
    let imu_config = OnlineSlamImuConfig {
        gravity_world: args.gravity_world,
        noise_model: Some(ImuNoiseModel {
            gyroscope_noise_density: dataset.imu_calibration.gyroscope_noise_density,
            accelerometer_noise_density: dataset.imu_calibration.accelerometer_noise_density,
        }),
        ..OnlineSlamImuConfig::default()
    };
    let vi_motion_init_config = if args.motion_vi_init_enabled {
        let viba2 = if args.motion_vi_init_recover_scale {
            Some(Viba2Config {
                recover_scale: true,
                ..Viba2Config::default()
            })
        } else {
            None
        };
        let bias_release = if args.vi_bias_release_min_keyframes.is_some()
            || args.vi_bias_release_min_translation_meters.is_some()
        {
            Some(BiasReleaseSchedule {
                min_keyframes: args.vi_bias_release_min_keyframes.unwrap_or(10),
                min_translation_meters: args
                    .vi_bias_release_min_translation_meters
                    .unwrap_or(2.0),
            })
        } else {
            None
        };
        Some(OnlineSlamMotionViInitConfig {
            initializer: MotionBasedViInitializerConfig {
                min_keyframes: args.motion_vi_init_min_keyframes,
                min_translation_meters: args.motion_vi_init_min_translation_meters,
                gravity_world: args.gravity_world,
                body_to_camera: body_to_camera.clone(),
                viba2,
                max_velocity_magnitude_mps: args.motion_vi_init_max_velocity_mps,
                max_gyro_bias_magnitude_rad_s: args.motion_vi_init_max_gyro_bias_rad_s,
                max_accel_bias_magnitude_mps2: args.motion_vi_init_max_accel_bias_mps2,
                max_final_imu_nis_per_dof: args.motion_vi_init_max_imu_nis_per_dof,
                max_final_imu_rotation_residual_rms_rad: args
                    .motion_vi_init_max_rotation_residual_rms_rad,
                max_final_imu_velocity_residual_rms_mps: args
                    .motion_vi_init_max_velocity_residual_rms_mps,
                max_final_imu_position_residual_rms_meters: args
                    .motion_vi_init_max_position_residual_rms_meters,
                bias_release,
                estimate_gravity: args.motion_vi_init_estimate_gravity,
                max_gravity_norm_deviation_ratio: args
                    .motion_vi_init_max_gravity_norm_deviation
                    .unwrap_or(0.3),
                estimate_gyro_bias: args.motion_vi_init_estimate_gyro_bias,
                ..MotionBasedViInitializerConfig::default()
            },
            allow_after_static_give_up: args.motion_vi_init_after_static_give_up,
            allow_from_configured_bias_before_static: args.motion_vi_init_from_configured_bias,
            ..OnlineSlamMotionViInitConfig::default()
        })
    } else {
        None
    };
    let local_vi_ba_config = if args.local_vi_ba_enabled {
        Some(OnlineSlamLocalBaConfig {
            gravity_world: args.gravity_world,
            body_to_camera: body_to_camera.clone(),
            bias_random_walk_noise_densities: Some((
                dataset.imu_calibration.gyroscope_random_walk,
                dataset.imu_calibration.accelerometer_random_walk,
            )),
            freeze_biases_when_cost_ratio_above: args.local_vi_ba_freeze_biases_above,
            reject_writeback_when_cost_ratio_above: args.local_vi_ba_reject_writeback_above,
            reject_writeback_when_final_imu_nis_per_dof_above: args
                .local_vi_ba_reject_final_imu_nis_per_dof_above,
            reject_writeback_when_velocity_norm_above_mps: args
                .local_vi_ba_reject_velocity_above_mps,
            reject_writeback_when_pose_translation_above_meters: args
                .local_vi_ba_reject_pose_translation_above_meters,
            reject_writeback_when_pose_rotation_above_radians: args
                .local_vi_ba_reject_pose_rotation_above_degrees
                .map(f64::to_radians),
            adaptive_velocity_gate: args.local_vi_ba_adaptive_velocity_gate.then_some(
                AdaptiveVelocityGateConfig {
                    reference_quantile: args.local_vi_ba_adaptive_velocity_quantile,
                    multiplier: args.local_vi_ba_adaptive_velocity_multiplier,
                    margin_mps: args.local_vi_ba_adaptive_velocity_margin_mps,
                    min_threshold_mps: args.local_vi_ba_adaptive_velocity_min_mps,
                    max_threshold_mps: args.local_vi_ba_adaptive_velocity_max_mps,
                    min_reference_count: args.local_vi_ba_adaptive_velocity_min_references,
                },
            ),
            relinearise_imu_factor_bias_thresholds: args.relinearise_imu_factor_bias_thresholds,
            run_at_vi_init_promotion: args.run_local_vi_ba_at_vi_init_promotion,
            marginalize_navigation_state: args.local_vi_ba_marginalization,
            initial_navigation_prior_std_devs: args.local_vi_ba_initial_prior_std_devs,
            use_observation_confidence_weights: args.observation_confidence_ba_enabled,
            ..OnlineSlamLocalBaConfig::default()
        })
    } else {
        None
    };
    let covisibility_local_ba_config = if args.covisibility_local_ba_enabled {
        Some(OnlineSlamCovisibilityLocalBaConfig {
            min_keyframes: args.covisibility_local_ba_min_keyframes,
            max_keyframes: args.covisibility_local_ba_max_keyframes,
            motion_vi_raw_residual_activation: args
                .covisibility_local_ba_motion_vi_raw_activation
                .map(
                    |(rotation, velocity, position)| MotionViRawResidualActivationConfig {
                        max_rotation_residual_rms_rad: rotation,
                        max_velocity_residual_rms_mps: velocity,
                        max_position_residual_rms_meters: position,
                    },
                ),
            max_seed_landmarks_for_activation: args
                .covisibility_local_ba_max_seed_landmarks_for_activation,
            trigger_every_new_keyframes: args.covisibility_local_ba_trigger_every,
            max_outlier_observation_ratio: args.covisibility_local_ba_max_outlier_observation_ratio,
            max_behind_camera_landmark_ratio: args.covisibility_local_ba_max_behind_camera_ratio,
            min_fixed_to_optimized_ratio: args.covisibility_local_ba_min_fixed_to_optimized_ratio,
            max_pose_translation_correction_m: args
                .covisibility_local_ba_max_pose_translation_correction_m,
            max_pose_rotation_correction_rad: args
                .covisibility_local_ba_max_pose_rotation_correction_deg
                .map(f64::to_radians),
            ba: CovisibilityLocalBaConfig {
                max_neighbor_keyframes: args.covisibility_local_ba_max_neighbor_keyframes,
                min_shared_landmarks: args.covisibility_local_ba_min_shared,
                max_boundary_keyframes: args.covisibility_local_ba_max_boundary_keyframes,
                min_boundary_observations: args.covisibility_local_ba_min_boundary_observations,
                fallback_min_boundary_observations: args
                    .covisibility_local_ba_fallback_min_boundary_observations,
                max_landmarks: args.covisibility_local_ba_max_landmarks,
                min_active_observations: args.covisibility_local_ba_min_active_observations,
                boundary_support_min_optimized_keyframes: args
                    .covisibility_local_ba_boundary_support_min_optimized_keyframes,
                boundary_support_min_fixed_keyframes: args
                    .covisibility_local_ba_boundary_support_min_fixed_keyframes,
                outlier_reprojection_threshold_px: args.covisibility_local_ba_outlier_threshold_px,
                remove_outlier_observations: args.covisibility_local_ba_remove_outliers,
                use_general_stereo_observations: args.covisibility_local_ba_general_stereo,
                general_stereo_max_initial_right_reprojection_error_px: args
                    .covisibility_local_ba_general_stereo_max_right_reprojection_px,
                use_observation_confidence_weights: args.observation_confidence_ba_enabled,
                pose_anchor_prior_weight: args.covisibility_local_ba_anchor_weight,
                ..CovisibilityLocalBaConfig::default()
            },
        })
    } else {
        None
    };
    // Sensible defaults for every field the CLI does not expose directly:
    // `verifier_config`/`pose_graph_config` use their library `Default`s,
    // `pcm_batch_rescreen`/`marginalization_window`/`marginalization_sparsify`
    // keep the documented off-by-default behaviour (see
    // `OnlineSlamLoopClosureRefinementConfig`'s field docs), and `camera`
    // reuses cam0's intrinsics since both verifiers are single-monocular.
    // `--pose-graph-refinement-verifier pnp` uses the library's default PnP
    // RANSAC thresholds (`PnPLoopClosureVerifierConfig::default()`); only the
    // essential-vs-PnP choice is CLI-selectable for now.
    let pose_graph_refinement_config = if args.pose_graph_refinement_enabled {
        Some(OnlineSlamLoopClosureRefinementConfig {
            camera: camera.clone(),
            verifier_config: LoopClosureVerifierConfig::default(),
            verifier: match args.pose_graph_refinement_verifier {
                LoopRefinementVerifierKind::Essential => LoopRefinementVerifier::EssentialMatrix,
                LoopRefinementVerifierKind::Pnp => {
                    LoopRefinementVerifier::Pnp(PnPLoopClosureVerifierConfig::default())
                }
            },
            pose_graph_config: PoseGraphSe3Config::default(),
            fixed_loop_edge_weight: args.pose_graph_refinement_fixed_loop_edge_weight,
            loop_pose_information: args.pose_graph_refinement_loop_pose_information.then(|| {
                visloc_rs::slam::LoopPoseInformationConfig {
                    max_information_eigenvalue: args
                        .pose_graph_refinement_loop_pose_information_max_eigenvalue,
                    loop_edge_scale: args
                        .pose_graph_refinement_loop_pose_information_loop_edge_scale,
                    ..visloc_rs::slam::LoopPoseInformationConfig::default()
                }
            }),
            gnc: if args.pose_graph_refinement_gnc {
                Some(visloc_rs::slam::gnc::GncConfig::default())
            } else {
                None
            },
            pcm: if args.pose_graph_refinement_pcm {
                Some(visloc_rs::slam::pcm::PcmConfig {
                    require_individual: args.pose_graph_refinement_pcm_require_individual,
                    ..visloc_rs::slam::pcm::PcmConfig::default()
                })
            } else {
                None
            },
            covariance_gate: args.pose_graph_refinement_covariance_gate,
            pcm_batch_rescreen: false,
            marginalization_window: None,
            marginalization_sparsify: false,
            trigger_every_new_constraints: args.pose_graph_refinement_trigger_every,
            appearance_candidates: if args.pose_graph_refinement_appearance_loops {
                Some(LoopAppearanceCandidateConfig {
                    min_keyframe_id_gap: args.pose_graph_refinement_appearance_min_gap,
                    max_candidates_per_frame: args.pose_graph_refinement_appearance_max_candidates,
                    pnp_verifier: PnPLoopClosureVerifierConfig {
                        min_inliers: args.pose_graph_refinement_appearance_min_inliers,
                        ..PnPLoopClosureVerifierConfig::default()
                    },
                    region_confirmation_required_keyframes: args
                        .pose_graph_refinement_appearance_confirmation_keyframes,
                    region_confirmation_max_misses: args
                        .pose_graph_refinement_appearance_confirmation_max_misses,
                    projection_search_radius_px: args
                        .pose_graph_refinement_appearance_projection_radius_px,
                    min_projection_correspondence_count: args
                        .pose_graph_refinement_appearance_projection_min_matches,
                    ..LoopAppearanceCandidateConfig::default()
                })
            } else {
                None
            },
            fuse_loop_observations: args.pose_graph_refinement_fuse_loop_observations,
            loop_welding_ba: args.pose_graph_refinement_loop_welding_ba.then(|| {
                CovisibilityLocalBaConfig {
                    max_neighbor_keyframes: 20,
                    min_shared_landmarks: 5,
                    max_boundary_keyframes: 20,
                    min_boundary_observations: 3,
                    min_active_observations: 8,
                    boundary_support_min_optimized_keyframes: Some(2),
                    boundary_support_min_fixed_keyframes: 1,
                    pose_anchor_prior_weight: Some(100.0),
                    // Welding starts from the geometrically verified PnP
                    // alignment. Keep this synchronous pass deliberately
                    // short so it refines the seam without rotating the
                    // whole local trajectory away from that initialization.
                    ba_config: BaConfig {
                        max_iterations: 3,
                        ..BaConfig::default()
                    },
                    ..CovisibilityLocalBaConfig::default()
                }
            }),
            propagate_corrections: args.pose_graph_refinement_propagate,
            solver: match args.pose_graph_refinement_solver {
                LoopRefinementSolverKind::Se3 => LoopRefinementSolver::Se3,
                LoopRefinementSolverKind::Sim3 => {
                    LoopRefinementSolver::Sim3(Sim3PoseGraphConfig::default())
                }
            },
        })
    } else {
        None
    };
    let covisibility_config = args
        .covisibility_local_map_max_keyframes
        .map(|max_keyframes| CovisibilityLocalMapConfig {
            max_keyframes: Some(max_keyframes),
            min_shared_landmarks: args.covisibility_local_map_min_shared,
            ..CovisibilityLocalMapConfig::default()
        });
    let relocalization_covisibility_config =
        args.relocalization_covisibility_max_keyframes
            .map(
                |max_keyframes| visloc_rs::OnlineSlamRelocalizationCovisibilityConfig {
                    max_neighbor_keyframes: Some(max_keyframes),
                    min_shared_landmarks: args.relocalization_covisibility_min_shared,
                    min_local_map_landmarks: args.relocalization_covisibility_min_landmarks,
                    fallback_to_broader_store_on_failure: args
                        .relocalization_covisibility_broader_fallback,
                    broader_store_retry_interval_frames: args
                        .relocalization_covisibility_broader_fallback_interval_frames,
                    compare_broader_store_on_success: args
                        .relocalization_covisibility_compare_broader_store,
                },
            );
    let relocalization_appearance_config =
        args.relocalization_appearance_max_keyframes
            .map(
                |max_keyframes| visloc_rs::OnlineSlamRelocalizationAppearanceConfig {
                    max_keyframes,
                    candidate_log_limit: args.relocalization_appearance_candidate_log_limit,
                    min_similarity: args.relocalization_appearance_min_similarity,
                    exclude_recent_frame_gap: args
                        .relocalization_appearance_exclude_recent_frame_gap,
                    min_local_map_landmarks: args.relocalization_appearance_min_landmarks,
                    fallback_to_broader_store_on_failure: args
                        .relocalization_appearance_broader_fallback,
                    broader_store_retry_interval_frames: args
                        .relocalization_appearance_broader_fallback_interval_frames,
                    compare_broader_store_on_success: args
                        .relocalization_appearance_compare_broader_store,
                },
            );
    let projection_guided_tracking_config = if args.projection_guided_tracking {
        Some(ProjectionGuidedTrackingConfig {
            search_radius_px: args.projection_search_radius_px,
            widen_factor: args.projection_widen_factor,
            max_widen_retries: args.projection_max_widen_retries,
            max_query_landmark_distance_ratio: args.projection_query_landmark_distance_ratio,
            local_map_refinement: !args.projection_no_local_map_refinement,
            refinement_search_radius_px: args.projection_refinement_search_radius_px,
            refinement_iterations: args.projection_refinement_iterations,
            refinement_radius_shrink_factor: args.projection_refinement_radius_shrink_factor,
            refinement_reassign_correspondences: args
                .projection_refinement_reassign_correspondences,
            refinement_min_inlier_pair_retention_ratio: args
                .projection_refinement_min_inlier_pair_retention_ratio,
            refinement_max_pose_translation_correction_m: args
                .projection_refinement_max_translation_correction_m,
            refinement_max_pose_rotation_correction_rad: args
                .projection_refinement_max_rotation_correction_deg
                .map(f64::to_radians),
        })
    } else {
        None
    };
    let tracking_config = TrackingConfig {
        covisibility_local_map: covisibility_config,
        max_pose_prior_translation_error: args.max_pose_jump_meters,
        pose_prior_visual_override: args
            .pose_prior_visual_override
            .then(PosePriorVisualOverrideConfig::default),
        min_inliers: args.tracking_min_inliers,
        min_inlier_ratio: args.tracking_min_inlier_ratio,
        max_mean_reprojection_error: args.tracking_max_reprojection_error,
        pnp_pose_prior_warm_start: args.pnp_pose_prior_warm_start,
        pose_jump_gap_scaling: args.pose_jump_gap_scaling,
        pose_jump_gap_scaling_max_multiplier: args.pose_jump_gap_scaling_max_multiplier,
        projection_guided_tracking: projection_guided_tracking_config,
        ..TrackingConfig::default()
    };
    let build_imu_config = || {
        let body_to_sensor = if args.imu_extrinsic_from_cam0 {
            body_to_camera.clone()
        } else {
            SE3::identity()
        };
        ImuPredictiveMotionModelConfig {
            gravity_world: args.gravity_world,
            bias_gyro: Vector3::zeros(),
            bias_acc: Vector3::zeros(),
            body_to_sensor,
            carry_forward_velocity_world: args.imu_motion_model_carry_forward_velocity,
        }
    };
    let motion_model = match args.motion_model {
        MotionModelKind::Pose => DemoMotionModel::Pose(ConstantPoseMotionModel),
        MotionModelKind::Velocity => DemoMotionModel::Velocity(ConstantVelocityMotionModel::new()),
        MotionModelKind::ImuPredictive => {
            // Plumb cam0's `T_BS` (body-from-sensor) only when the
            // `--imu-extrinsic-from-cam0` opt-in flag is set. With the
            // flag off (default), `body == camera` is approximated —
            // the integration is still done in body-as-camera frame
            // but the math reduces to the original Phase-7 wire-up.
            DemoMotionModel::ImuPredictive(ImuPredictiveMotionModel::new(build_imu_config()))
        }
        MotionModelKind::AdaptiveImuPose => {
            // Phase-23 #4: IMU↔ConstantPose adaptive dispatch. Default
            // thresholds (2 consecutive failures → switch to pose,
            // 5 consecutive successes → switch back to IMU) bias for
            // staying in IMU mode in steady-state but fall back to
            // pose mode quickly when the cliff begins to fire.
            let adaptive_config = AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: args.adaptive_motion_failures_to_switch_to_pose,
                successes_to_switch_to_imu: args.adaptive_motion_successes_to_switch_to_imu,
                imu_velocity_refresh_policy: args.adaptive_motion_imu_velocity_refresh_policy,
            };
            DemoMotionModel::AdaptiveImuPose(AdaptiveImuPoseMotionModel::new(
                ImuPredictiveMotionModel::new(build_imu_config()),
                ConstantPoseMotionModel,
                adaptive_config,
            ))
        }
    };
    let mut localization_config = LocalizationConfig::default();
    if let Some(px) = args.pnp_reprojection_threshold_px {
        localization_config.reprojection_threshold = px;
    }
    if args.cross_check_matcher && args.mutual_softmax_matcher {
        return Err(
            "--cross-check-matcher and --mutual-softmax-matcher are mutually exclusive".into(),
        );
    }
    let demo_matcher = if args.mutual_softmax_matcher {
        DemoMatcher::MutualSoftmax(MutualSoftmaxMatcher::new(MutualSoftmaxConfig::default()))
    } else if args.cross_check_matcher {
        DemoMatcher::CrossCheck(CrossCheckMatcher::new(BruteForceMatcher {
            ratio: localization_config.ratio,
        }))
    } else {
        DemoMatcher::BruteForce(BruteForceMatcher {
            ratio: localization_config.ratio,
        })
    };
    let atlas_bridge_localizer = LocalizationPipeline::new(
        CrossCheckMatcher::new(demo_matcher.clone()),
        localization_config.clone(),
    );
    let atlas_bridge_config = CrossSubmapAlignmentConfig::default();
    let localization_pipeline = LocalizationPipeline::new(demo_matcher, localization_config);
    let keyframe_policy_config = {
        let mut cfg = KeyframePolicyConfig::default();
        if let Some(m) = args.keyframe_min_translation {
            cfg.min_translation = m;
        }
        if let Some(gap) = args.keyframe_min_frame_gap {
            cfg.min_frame_id_gap = gap;
        }
        cfg.max_frame_id_gap = args.keyframe_max_frame_gap;
        cfg.tracked_landmark_keyframe_ratio = args.keyframe_tracked_landmark_ratio;
        cfg.min_tracked_landmarks_for_quality_keyframe =
            args.keyframe_min_tracked_landmarks_for_ratio;
        cfg.min_inliers = args.keyframe_min_inliers;
        cfg.min_inlier_ratio = args.keyframe_min_inlier_ratio;
        cfg
    };
    let local_mapping = LocalMappingPipeline {
        keyframe_policy: SimpleKeyframePolicy::new(keyframe_policy_config),
        ..LocalMappingPipeline::default()
    };
    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::with_motion_model(localization_pipeline, motion_model, tracking_config),
        local_mapping,
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                enabled: !(args.pose_graph_refinement_enabled
                    && args.pose_graph_refinement_appearance_loops),
                // Shared-landmark overlap at a 5-frame gap is ordinary local
                // covisibility, not evidence of a completed loop. Keep the
                // geometric and appearance candidate streams on the same
                // long-range revisit horizon when online PGO is enabled.
                min_frame_id_gap: if args.pose_graph_refinement_enabled {
                    args.pose_graph_refinement_appearance_min_gap
                } else {
                    LoopClosureConfig::default().min_frame_id_gap
                },
                ..LoopClosureConfig::default()
            },
            imu: Some(imu_config),
            local_vi_ba: local_vi_ba_config,
            covisibility_local_ba: covisibility_local_ba_config,
            sparse_factor_graph: args
                .sparse_factor_graph_enabled
                .then(visloc_rs::SparseFactorGraphConfig::default),
            vi_init: Some(vi_init_config),
            vi_motion_init: vi_motion_init_config,
            keep_pre_promotion_imu_factors: args.keep_pre_promotion_imu_factors,
            pose_graph_refinement: pose_graph_refinement_config,
            relocalization: if args.relocalization_enabled {
                Some(visloc_rs::OnlineSlamRelocalizationConfig {
                    min_inliers: args.relocalization_min_inliers,
                    min_inlier_ratio: args.relocalization_min_inlier_ratio,
                    max_mean_reprojection_error: args.relocalization_max_reprojection_error,
                    pose_prior_candidate_radius_meters: args
                        .relocalization_pose_prior_radius_meters,
                    recent_keyframe_window: args.relocalization_recent_keyframe_window,
                    covisibility_local_map: relocalization_covisibility_config,
                    appearance_retrieval_map: relocalization_appearance_config,
                    max_translation_from_imu_prediction_meters: args
                        .relocalization_max_translation_from_imu_prediction_meters,
                    attempt_interval_frames: args.relocalization_attempt_interval_frames,
                    max_consecutive_failed_attempts: args
                        .relocalization_max_consecutive_failed_attempts,
                    max_translation_per_frame_from_last_success_meters: args
                        .relocalization_max_translation_per_frame_from_last_success_meters,
                    min_inlier_depth_median_ratio_to_last_success: args
                        .relocalization_min_inlier_depth_median_ratio_to_last_success,
                    max_inlier_depth_median_ratio_to_last_success: args
                        .relocalization_max_inlier_depth_median_ratio_to_last_success,
                    confirmation_required_recoveries: args
                        .relocalization_confirmation_required_recoveries,
                    confirmation_max_translation_per_frame_meters: args
                        .relocalization_confirmation_max_translation_per_frame_meters,
                })
            } else {
                None
            },
        },
    );

    // Atlas mode is deliberately opt-in. The root owns a snapshot of the
    // initial online map and is aligned to the Atlas identity. Each later
    // stereo restart gets its own local gauge and remains unaligned until a
    // verified cross-submap bridge is admitted.
    let mut map_atlas = args
        .rebootstrap_independent_submap
        .then(|| MapAtlas::new(slam.map().clone()));
    let mut active_submap_id = map_atlas
        .as_ref()
        .and_then(MapAtlas::root_submap_id)
        .unwrap_or(0);
    // Unverified causal placement used only to project bridge candidates.
    // It never marks a submap aligned and is discarded after verified merge.
    let mut active_submap_atlas_prior: Option<SE3> = None;

    fs::create_dir_all(&args.out_dir)?;
    let traj_path = args.out_dir.join("slam_trajectory.csv");
    let submap_trajectory_path = args.out_dir.join("submap_trajectory.csv");
    let atlas_submaps_path = args.out_dir.join("atlas_submaps.csv");
    let atlas_submap_landmarks_path = args.out_dir.join("atlas_submap_landmarks.csv");
    let atlas_merge_attempts_path = args.out_dir.join("atlas_merge_attempts.csv");
    let atlas_boundary_factors_path = args.out_dir.join("atlas_boundary_factors.csv");
    let err_path = args.out_dir.join("slam_errors.csv");
    let frame_groundtruth_path = args.out_dir.join("frame_groundtruth.csv");
    let frame_appearance_descriptors_path = args.out_dir.join("frame_appearance_descriptors.csv");
    let vi_init_log_path = args.out_dir.join("vi_init_log.txt");
    let motion_vi_init_log_path = args.out_dir.join("motion_vi_init_log.txt");
    let covisibility_local_ba_log_path = args.out_dir.join("covisibility_ba_log.txt");
    let local_vi_ba_diagnostics_path = args.out_dir.join("local_vi_ba_diagnostics.csv");
    let keyframe_decision_log_path = args.out_dir.join("keyframe_decisions.csv");
    let tracking_diagnostics_path = args.out_dir.join("tracking_diagnostics.csv");
    let relocalization_appearance_candidates_path = args
        .out_dir
        .join("relocalization_appearance_candidates.csv");
    let relocalization_attempts_path = args.out_dir.join("relocalization_attempts.csv");
    let rebootstrap_log_path = args.out_dir.join("rebootstrap_log.csv");
    let loop_constraints_path = args.out_dir.join("loop_constraints.csv");
    let loop_rejected_constraints_path = args.out_dir.join("loop_rejected_constraints.csv");
    let loop_candidate_diagnostics_path = args.out_dir.join("loop_candidate_diagnostics.csv");
    let loop_candidate_evidence_path = args.out_dir.join("loop_candidate_evidence.csv");
    // Post-run views computed once the frame loop (and any pose-graph
    // loop-closure back-propagation into `map.keyframes`) has finished:
    // `slam_trajectory.csv`/`slam_errors.csv` are causal, per-frame *live*
    // estimates and can never reflect a later loop-closure correction. These
    // two give the "final optimized trajectory" view that published SLAM ATE
    // numbers (ORB-SLAM3 etc.) are actually computed on.
    let keyframe_trajectory_path = args.out_dir.join("keyframe_trajectory.csv");
    let final_keyframe_errors_path = args.out_dir.join("final_keyframe_errors.csv");

    let mut traj_csv =
        String::from("timestamp_ns,frame_idx,px,py,pz,qw,qx,qy,qz,tracking_success\n");
    let mut submap_trajectory_csv = String::from(
        "timestamp_ns,frame_idx,submap_id,local_px,local_py,local_pz,local_qw,local_qx,local_qy,local_qz,local_tracking_success,atlas_aligned\n",
    );
    let mut atlas_merge_attempts_csv = String::from(
        "attempt_idx,frame_idx,timestamp_ns,source_submap_id,source_frame_id,target_submap_id,target_frame_id,appearance_similarity,candidate_landmarks,used_projection_prior,localization_correspondences,localization_inliers,localization_inlier_ratio,mean_reprojection_error_px,estimated_scale,scale_pair_count,scale_inlier_ratio,scale_mad,weldable_landmark_matches,verified,failure_reason\n",
    );
    let mut atlas_boundary_factors_csv = String::from(
        "frame_idx,timestamp_ns,source_submap_id,source_frame_id,target_submap_id,target_frame_id,descriptor_matches,spatial_mutual_matches,metric_correspondences,projection_refined_correspondences,projection_refinement_iterations,used_transform_prior,rigid_inliers,rigid_inlier_ratio,mean_target_reprojection_error_px,estimated_scale,scale_pair_count,scale_inlier_ratio,weldable_landmark_matches,verified,failure_reason\n",
    );
    let mut err_csv = String::from(
        "timestamp_ns,frame_idx,segment_id,gt_px,gt_py,gt_pz,est_px,est_py,est_pz,position_error_m,orientation_error_deg\n",
    );
    let mut frame_groundtruth_csv = String::from("timestamp_ns,frame_idx,gt_px,gt_py,gt_pz\n");
    let mut frame_appearance_descriptor_rows: Vec<String> = Vec::new();
    let mut frame_appearance_descriptor_dim: Option<usize> = None;
    let mut frame_appearance_descriptor_count = 0usize;
    let mut vi_init_log = String::new();
    let mut motion_vi_init_log = String::new();
    let mut covisibility_local_ba_log = String::from(
        "frame_idx,timestamp_ns,success,error,elapsed_ms,factor_graph_neighbors,optimized_keyframes,fixed_keyframes,landmarks,observations,boundary_fallback_used,mean_reprojection_before_px,mean_reprojection_after_px,max_pose_translation_correction_m,max_pose_rotation_correction_rad,updated_keyframes,updated_landmarks,outlier_observations,outlier_observation_ratio,quality_gate_rejected,pose_correction_gate_rejected,removed_observations\n",
    );
    let mut keyframe_decision_log = String::from(
        "frame_idx,timestamp_ns,tracking_success,selected,reason,last_keyframe_frame_id,selected_keyframe_count,localization_inliers,localization_inlier_ratio,frame_id_gap,translation_m,min_translation_m,tracked_landmarks,last_keyframe_tracked_landmarks,min_tracked_landmark_ratio,tracking_failure_reason\n",
    );
    let mut tracking_diagnostics_csv = String::from(
        "frame_idx,timestamp_ns,state,event,success,successive_failures,used_pose_prior,pose_present,estimated_px,estimated_py,estimated_pz,estimated_qw,estimated_qx,estimated_qy,estimated_qz,tracking_path,projection_attempt_count,projection_widen_retry_count,local_map_refinement_accepted,local_map_refinement_rejected,candidate_landmark_count,match_count,correspondence_count,inlier_count,outlier_count,inlier_ratio,mean_reprojection_error_px,median_reprojection_error_px,max_reprojection_error_px,pose_prior_translation_innovation_m,pose_prior_rotation_innovation_deg,tracking_failure_reason,localization_failure_reason,refinement_applied,pre_refinement_mean_reprojection_error_px,post_refinement_mean_reprojection_error_px,refinement_error_delta_px,pose_prior_visual_override_used\n",
    );
    let mut relocalization_appearance_candidates_csv = String::from(
        "query_frame_id,matched_keyframe_id,score,rank,timestamp_ns,descriptor_store_landmark_count,appearance_descriptor_store_landmark_count,recovery_attempted,recovery_succeeded,passed_acceptance_gates,used_appearance_store,used_broader_fallback\n",
    );
    let mut relocalization_attempts_csv = String::from(
        "frame_idx,timestamp_ns,attempted,succeeded,localization_success,reject_reason,inlier_count,min_inliers,min_inliers_pass,inlier_ratio,min_inlier_ratio,min_inlier_ratio_pass,correspondence_count,mean_reprojection_error,max_reprojection_error,reprojection_pass,translation_per_frame_from_last_success_meters,max_translation_per_frame_from_last_success_meters,continuity_pass,inlier_depth_median_ratio_to_last_success,min_inlier_depth_median_ratio_to_last_success,max_inlier_depth_median_ratio_to_last_success,depth_ratio_pass,passed_acceptance_gates,confirmation_count,confirmation_required_count,confirmation_translation_per_frame_from_previous_meters,descriptor_store_landmark_count,covisibility_local_descriptor_store_landmark_count,appearance_descriptor_store_landmark_count,broader_descriptor_store_landmark_count,tried_covisibility_store,used_covisibility_store,tried_appearance_store,used_appearance_store,tried_broader_fallback,broader_fallback_skipped_by_interval,used_broader_fallback\n",
    );
    let mut rebootstrap_log_csv = String::from(
        "event_idx,segment_id,frame_idx,timestamp_ns,seed_source,stereo_matches,landmarks_added,keyframe_selected,consecutive_lost_frames\n",
    );
    // Diagnostic for the short-range loop-closure hypothesis: every folded
    // loop constraint (both the shared-landmark and, when
    // `--pose-graph-refinement-appearance-loops` is set, the appearance
    // stream), with the keyframe-id gap so an offline pass can bucket
    // constraints by range and check whether appearance-sourced constraints
    // (necessarily long-range) are the ones actually moving ATE.
    let mut loop_constraints_csv = String::from(
        "frame_idx,from_keyframe_id,to_keyframe_id,keyframe_id_gap,translation_norm_m,relative_tx,relative_ty,relative_tz,relative_qw,relative_qx,relative_qy,relative_qz,inlier_count,source,gt_translation_error_m,gt_rotation_error_deg,gt_correct_0p5m_10deg\n",
    );
    let mut loop_rejected_constraints_csv = String::from(
        "frame_idx,from_keyframe_id,to_keyframe_id,keyframe_id_gap,translation_norm_m,relative_tx,relative_ty,relative_tz,relative_qw,relative_qx,relative_qy,relative_qz,inlier_count,source,rejection_reason,gt_translation_error_m,gt_rotation_error_deg,gt_correct_0p5m_10deg\n",
    );
    let mut loop_candidate_diagnostics_csv = String::from(
        "frame_idx,timestamp_ns,shared_candidates_seen,appearance_candidates_ranked,appearance_connected_region_rejected,appearance_primary_pnp_verified,appearance_projection_rejected,appearance_covisibility_rejected,appearance_pending_projection_attempted,appearance_pending_projection_verified,appearance_pending_projection_search_radius_px,appearance_pending_projection_correspondences,appearance_pending_projection_inliers,appearance_region_confirmation_waiting,appearance_region_confirmed,appearance_candidates_admitted,pcm_rejected,covariance_rejected\n",
    );
    let mut loop_candidate_evidence_csv = String::from(
        "query_frame_id,matched_keyframe_id,appearance_similarity,matched_region_keyframes,matched_region_landmarks,primary_correspondences,primary_inliers,projection_attempted,projection_correspondences,projection_inliers,projection_accepted,current_covisible_keyframes,neighbor_pnp_verified,consistent_keyframes,min_translation_disagreement_m,min_rotation_disagreement_rad,accepted\n",
    );
    let mut pose_graph_refinement_constraints_shared: usize = 0;
    let mut pose_graph_refinement_constraints_appearance: usize = 0;
    let mut pose_graph_refinement_gt_evaluated: usize = 0;
    let mut pose_graph_refinement_gt_correct: usize = 0;
    let mut pose_graph_refinement_rejected_gt_evaluated: usize = 0;
    let mut pose_graph_refinement_rejected_gt_correct: usize = 0;

    let frame_cap = if args.max_frames == 0 {
        usize::MAX
    } else {
        args.max_frames
    };

    let mut imu_idx = 0usize;
    let mut prev_imu_ts = imu_first_ts;
    let mut vi_init_preseed_samples = 0usize;
    while imu_idx < dataset.imu_samples.len()
        && dataset.imu_samples[imu_idx].timestamp_nanoseconds
            < seed_image_entry.timestamp_nanoseconds
    {
        let sample = &dataset.imu_samples[imu_idx];
        let dt_ns = sample.timestamp_nanoseconds - prev_imu_ts;
        if dt_ns > 0 {
            slam.push_vi_initialization_measurement(
                sample.gyro,
                sample.accel,
                dt_ns as f64 * 1.0e-9,
            );
            vi_init_preseed_samples += 1;
        }
        prev_imu_ts = sample.timestamp_nanoseconds;
        imu_idx += 1;
    }

    let mut frames_recorded = 0usize;
    let mut frame_processing_times_ms: Vec<f64> = Vec::new();
    let mut tracking_successes = 0usize;
    let mut globally_aligned_tracking_successes = 0usize;
    let mut keyframe_insufficient_tracking_quality_rejections = 0usize;
    let mut relocalization_attempts = 0u64;
    let mut relocalization_successes = 0u64;
    let mut relocalization_gate_passes = 0u64;
    let mut relocalization_confirmation_waiting = 0u64;
    let mut relocalization_confirmation_tx_per_frame_count = 0u64;
    let mut relocalization_confirmation_tx_per_frame_sum = 0.0f64;
    let mut relocalization_confirmation_tx_per_frame_max: Option<f64> = None;
    let mut relocalization_tx_per_frame_count = 0u64;
    let mut relocalization_tx_per_frame_sum = 0.0f64;
    let mut relocalization_tx_per_frame_max: Option<f64> = None;
    let mut relocalization_success_tx_per_frame_count = 0u64;
    let mut relocalization_success_tx_per_frame_sum = 0.0f64;
    let mut relocalization_success_tx_per_frame_max: Option<f64> = None;
    let mut relocalization_depth_ratio_count = 0u64;
    let mut relocalization_depth_ratio_sum = 0.0f64;
    let mut relocalization_depth_ratio_min: Option<f64> = None;
    let mut relocalization_depth_ratio_max: Option<f64> = None;
    let mut relocalization_success_depth_ratio_count = 0u64;
    let mut relocalization_success_depth_ratio_sum = 0.0f64;
    let mut relocalization_success_depth_ratio_min: Option<f64> = None;
    let mut relocalization_success_depth_ratio_max: Option<f64> = None;
    let mut relocalization_descriptor_store_count_observations = 0u64;
    let mut relocalization_descriptor_store_count_sum = 0usize;
    let mut relocalization_descriptor_store_count_min: Option<usize> = None;
    let mut relocalization_descriptor_store_count_max: Option<usize> = None;
    let mut relocalization_covisibility_descriptor_store_tried_frames = 0u64;
    let mut relocalization_covisibility_descriptor_store_used_frames = 0u64;
    let mut relocalization_appearance_descriptor_store_tried_frames = 0u64;
    let mut relocalization_appearance_descriptor_store_used_frames = 0u64;
    let mut relocalization_appearance_candidate_keyframe_count_observations = 0u64;
    let mut relocalization_appearance_candidate_keyframe_count_sum = 0usize;
    let mut relocalization_appearance_best_similarity_count = 0u64;
    let mut relocalization_appearance_best_similarity_sum = 0.0f64;
    let mut relocalization_appearance_best_similarity_max: Option<f32> = None;
    let mut relocalization_broader_descriptor_store_retry_frames = 0u64;
    let mut relocalization_broader_descriptor_store_retry_interval_skips = 0u64;
    let mut relocalization_broader_descriptor_store_used_frames = 0u64;
    let mut relocalization_covisibility_reference_keyframe_count = 0u64;
    let mut covisibility_local_map_frames = 0usize;
    let mut covisibility_local_map_size_sum: usize = 0;
    let mut feature_count_sum: usize = 0;
    let mut feature_count_min: usize = usize::MAX;
    let mut feature_count_max: usize = 0;
    let mut vi_init_first_event_at_frame: Option<usize> = None;
    let mut vi_init_succeeded_at_frame: Option<usize> = None;
    let mut motion_vi_init_first_event_at_frame: Option<usize> = None;
    let mut motion_vi_init_succeeded_at_frame: Option<usize> = None;
    let mut motion_vi_init_recovered_scale: Option<f64> = None;
    let mut motion_vi_init_viba2_iterations: Option<usize> = None;
    let mut motion_vi_init_estimated_gravity: Option<Vector3<f64>> = None;
    let mut motion_vi_init_estimated_gyro_bias: Option<Vector3<f64>> = None;

    let rebootstrap_enabled = args.rebootstrap_after_lost_frames.is_some();
    let mut segment_id: usize = 0;
    let mut consecutive_lost_frames: usize = 0;
    let mut last_rebootstrap_frame_idx: Option<usize> = None;
    let mut rebootstrap_events: usize = 0;
    let mut rebootstrap_event_idx: usize = 0;
    let mut atlas_merge_attempts: usize = 0;
    let mut atlas_merge_source_keyframe_attempts: usize = 0;
    let mut atlas_broader_recovery_cycles: usize = 0;
    let mut atlas_merge_successes: usize = 0;
    let mut atlas_boundary_keyframe_snapshots: usize = 0;
    let mut atlas_boundary_factor_attempts: usize = 0;
    let mut atlas_boundary_factor_successes: usize = 0;
    let mut last_successful_atlas_boundary_keyframe: Option<Keyframe> = None;
    let mut next_rebootstrap_landmark_id: u64 = 10_000_000_000;

    let mut estimated_positions: Vec<Point3<f64>> = Vec::new();
    let mut reference_positions: Vec<Point3<f64>> = Vec::new();
    let mut sum_position_sq = 0.0_f64;
    let mut sum_orientation_sq_deg = 0.0_f64;
    let mut max_position_err = 0.0_f64;
    let mut max_orientation_err_deg = 0.0_f64;
    let mut error_samples = 0usize;
    // Previous successful (pose, timestamp_ns) — used to finite-difference
    // body velocity after each successful frame and feed it back into the
    // IMU-predictive motion model. The model otherwise leaves
    // `velocity_world` at zero forever (no VI-BA running) and
    // systematically under-predicts motion on the next frame.
    let mut prev_successful_pose: Option<Pose> = None;
    let mut prev_successful_ts: Option<i128> = None;
    // Per-trigger local-VI-BA bookkeeping. `mirror_count` counts how many
    // times we pushed refined `(v, b)` into the IMU motion model.
    let mut local_vi_ba_triggers: usize = 0;
    let mut local_vi_ba_mirrors: usize = 0;
    let mut imu_factors_staged: usize = 0;
    let mut local_vi_ba_relinearised_factor_total: usize = 0;
    let mut local_vi_ba_stereo_observation_total: usize = 0;
    let mut local_vi_ba_marginalization_priors_applied: usize = 0;
    let mut local_vi_ba_marginalization_successes: usize = 0;
    let mut local_vi_ba_quality_gate_rejections: usize = 0;
    let mut local_vi_ba_cost_ratio_gate_rejections: usize = 0;
    let mut local_vi_ba_imu_nis_gate_rejections: usize = 0;
    let mut local_vi_ba_velocity_gate_rejections: usize = 0;
    let mut local_vi_ba_pose_correction_gate_rejections: usize = 0;
    let mut local_vi_ba_max_pose_translation_correction_meters: f64 = 0.0;
    let mut local_vi_ba_max_pose_rotation_correction_degrees: f64 = 0.0;
    let mut local_vi_ba_adaptive_velocity_gate_rejections: usize = 0;
    let mut local_vi_ba_last_adaptive_velocity_threshold_mps: Option<f64> = None;
    let mut local_vi_ba_diagnostic_rows = vec![
        "frame_idx,timestamp_ns,window_keyframe_ids,observations,stereo_observations,imu_factors,quality_gate_rejected,imu_nis_gate_rejected,max_pose_translation_correction_m,max_pose_rotation_correction_deg,initial_total,initial_visual,initial_imu,initial_bias_random_walk,initial_navigation_prior,initial_other_structural,initial_imu_nis_per_dof,initial_imu_rotation_rms_rad,initial_imu_velocity_rms_mps,initial_imu_position_rms_m,final_total,final_visual,final_imu,final_bias_random_walk,final_navigation_prior,final_other_structural,final_imu_nis_per_dof,final_imu_rotation_rms_rad,final_imu_velocity_rms_mps,final_imu_position_rms_m".to_string(),
    ];
    let mut last_mirrored_velocity_world: Option<Vector3<f64>> = None;
    let mut last_mirrored_bias_gyro: Option<Vector3<f64>> = None;
    let mut last_mirrored_bias_acc: Option<Vector3<f64>> = None;
    let mut covisibility_local_ba_triggers: usize = 0;
    let mut sparse_factor_graph_updates: usize = 0;
    let mut sparse_factor_graph_added: usize = 0;
    let mut sparse_factor_graph_reactivated: usize = 0;
    let mut sparse_factor_graph_inactivated_low_confidence: usize = 0;
    let mut sparse_factor_graph_inactivated_window_age: usize = 0;
    let mut sparse_factor_graph_inactivated_budget: usize = 0;
    let mut sparse_factor_graph_pruned: usize = 0;
    let mut sparse_factor_graph_active_temporal: usize = 0;
    let mut sparse_factor_graph_active_proximity: usize = 0;
    let mut sparse_factor_graph_active_stereo: usize = 0;
    let mut sparse_factor_graph_inactive: usize = 0;
    let mut covisibility_local_ba_successes: usize = 0;
    let mut covisibility_local_ba_failures: usize = 0;
    let mut covisibility_local_ba_updated_keyframes_total: usize = 0;
    let mut covisibility_local_ba_updated_landmarks_total: usize = 0;
    let mut covisibility_local_ba_removed_observations_total: usize = 0;
    let mut covisibility_local_ba_boundary_fallback_successes: usize = 0;
    let mut covisibility_local_ba_reprojection_before_sum: f64 = 0.0;
    let mut covisibility_local_ba_reprojection_after_sum: f64 = 0.0;
    let mut covisibility_local_ba_elapsed_ms_total: f64 = 0.0;
    let mut covisibility_local_ba_elapsed_ms_max: f64 = 0.0;
    let mut covisibility_local_ba_last_error: Option<String> = None;
    let mut covisibility_local_ba_active_observation_gate_failures: usize = 0;
    let mut covisibility_local_ba_boundary_fallback_active_gate_failures: usize = 0;
    let mut covisibility_local_ba_no_local_landmarks_failures: usize = 0;
    let mut covisibility_local_ba_no_observations_failures: usize = 0;
    let mut covisibility_local_ba_solver_failures: usize = 0;
    let mut covisibility_local_ba_quality_gate_failures: usize = 0;
    let mut covisibility_local_ba_boundary_support_failures: usize = 0;
    let mut covisibility_local_ba_behind_camera_gate_failures: usize = 0;
    let mut covisibility_local_ba_fixed_ratio_gate_failures: usize = 0;
    let mut covisibility_local_ba_pose_correction_gate_failures: usize = 0;
    let mut covisibility_local_ba_other_failures: usize = 0;
    // Online loop-closure + pose-graph refinement stage bookkeeping (opt-in
    // via `--pose-graph-refinement`).
    let mut pose_graph_refinement_candidates_seen: usize = 0;
    let mut pose_graph_refinement_verified_constraints: usize = 0;
    let mut pose_graph_refinement_appearance_ranked: usize = 0;
    let mut pose_graph_refinement_appearance_connected_region_rejected: usize = 0;
    let mut pose_graph_refinement_appearance_pnp_verified: usize = 0;
    let mut pose_graph_refinement_appearance_projection_rejected: usize = 0;
    let mut pose_graph_refinement_appearance_covisibility_rejected: usize = 0;
    let mut pose_graph_refinement_appearance_pending_projection_attempted: usize = 0;
    let mut pose_graph_refinement_appearance_pending_projection_verified: usize = 0;
    let mut pose_graph_refinement_appearance_pending_projection_correspondences: usize = 0;
    let mut pose_graph_refinement_appearance_pending_projection_inliers: usize = 0;
    let mut pose_graph_refinement_appearance_pending_projection_correspondences_max: usize = 0;
    let mut pose_graph_refinement_appearance_pending_projection_inliers_max: usize = 0;
    let mut pose_graph_refinement_appearance_pending_projection_radius_min: Option<f64> = None;
    let mut pose_graph_refinement_appearance_pending_projection_radius_max: Option<f64> = None;
    let mut pose_graph_refinement_appearance_region_confirmation_waiting: usize = 0;
    let mut pose_graph_refinement_appearance_region_confirmed: usize = 0;
    let mut pose_graph_refinement_appearance_scale_failed: usize = 0;
    let mut pose_graph_refinement_appearance_scale_insufficient_points: usize = 0;
    let mut pose_graph_refinement_appearance_scale_insufficient_matches: usize = 0;
    let mut pose_graph_refinement_appearance_scale_no_consensus: usize = 0;
    let mut pose_graph_refinement_appearance_near_unit: usize = 0;
    let mut pose_graph_refinement_pgo_solves: usize = 0;
    let mut pose_graph_refinement_gnc_rejected_loop_edges_max: usize = 0;
    let mut pose_graph_refinement_pcm_rejected: usize = 0;
    let mut pose_graph_refinement_covariance_rejected: usize = 0;
    let mut pose_graph_refinement_pose_information_rejected: usize = 0;
    let mut pose_graph_refinement_with_pose_information: usize = 0;
    let mut pose_graph_refinement_pose_information_estimates: usize = 0;
    let mut pose_graph_refinement_pose_information_used_correspondences: usize = 0;
    let mut pose_graph_refinement_pose_information_stereo_covariance_correspondences: usize = 0;
    let mut pose_graph_refinement_pose_information_raw_condition_max: f64 = 0.0;
    let mut pose_graph_refinement_pose_information_spectral_scale_min: Option<f64> = None;
    let mut pose_graph_refinement_pose_information_failures =
        visloc_rs::slam::LoopPoseInformationFailureCounts::default();
    let mut pose_graph_refinement_sequential_with_pose_information: usize = 0;
    let mut pose_graph_refinement_sequential_pose_information_fallbacks: usize = 0;
    let mut pose_graph_refinement_sequential_pose_information_used_correspondences: usize = 0;
    let mut pose_graph_refinement_sequential_pose_information_stereo_covariance_correspondences:
        usize = 0;
    let mut pose_graph_refinement_sequential_pose_information_raw_condition_max: f64 = 0.0;
    let mut pose_graph_refinement_sequential_pose_information_spectral_scale_min: Option<f64> =
        None;
    let mut pose_graph_refinement_sequential_pose_information_failures =
        visloc_rs::slam::LoopPoseInformationFailureCounts::default();
    let mut pose_graph_refinement_loop_fusion_pairs_considered: usize = 0;
    let mut pose_graph_refinement_loop_fusion_observations_inserted: usize = 0;
    let mut pose_graph_refinement_loop_fusion_observations_reassigned: usize = 0;
    let mut pose_graph_refinement_loop_fusion_pairs_skipped: usize = 0;
    let mut pose_graph_refinement_loop_fusion_pairs_robust_rejected: usize = 0;
    let mut pose_graph_refinement_loop_fusion_pairs_reprojection_rejected: usize = 0;
    let mut pose_graph_refinement_loop_welding_ba_attempted: usize = 0;
    let mut pose_graph_refinement_loop_welding_ba_succeeded: usize = 0;
    let mut pose_graph_refinement_loop_welding_ba_rejected_or_failed: usize = 0;
    let mut pose_graph_refinement_loop_welding_ba_updated_keyframes: usize = 0;
    let mut pose_graph_refinement_loop_welding_ba_updated_landmarks: usize = 0;
    let mut pose_graph_refinement_loop_welding_post_pgo_attempted: usize = 0;
    let mut pose_graph_refinement_loop_welding_post_pgo_succeeded: usize = 0;
    let mut pose_graph_refinement_loop_welding_post_pgo_reprojection_max: Option<f64> = None;
    let mut pose_graph_refinement_loop_welding_post_ba_last_error: Option<String> = None;
    let mut pose_graph_refinement_loop_welding_post_ba_behind_camera_ratio_max: Option<f64> = None;
    let mut pose_graph_refinement_loop_welding_initial_translation_max: Option<f64> = None;
    let mut pose_graph_refinement_loop_welding_initial_rotation_max: Option<f64> = None;
    let mut pose_graph_refinement_landmarks_moved: usize = 0;
    let mut pose_graph_refinement_max_landmark_displacement_meters: f64 = 0.0;
    let mut pose_graph_refinement_tracker_corrections_applied: usize = 0;
    // `Sim3` solver observability: the last fired solve's per-node scale
    // spread (see `LoopRefinementSolver::Sim3` / `--pose-graph-refinement-solver
    // sim3`). `None` on the `Se3` path (default) or before any Sim3 solve
    // has fired.
    let mut pose_graph_refinement_last_solve_scale_spread: Option<(f64, f64)> = None;
    // Stereo landmark replenishment (opt-in via `--stereo-landmark-replenish`).
    // Candidates built from frame N's stereo match are queued here and
    // submitted on frame N+1's `process_frame` call, once frame N's
    // keyframe (used as the second, anchor observation) is guaranteed to
    // already exist in the map.
    let mut pending_replenish_candidates: Vec<LandmarkCandidate> = Vec::new();
    let mut next_replenish_candidate_id: u64 = 1_000_000_000;
    let mut stereo_landmark_replenish_candidates_total: usize = 0;
    // Built once from the CLI knobs (permissive library defaults, overridden
    // where a flag was supplied). Shared, immutable, across the frame loop.
    let stereo_replenish_config = {
        let defaults = StereoReplenishConfig::default();
        StereoReplenishConfig {
            max_candidates_per_frame: args.stereo_landmark_replenish_max_per_frame,
            anchor_keypoint_match_radius_px: args
                .stereo_landmark_replenish_anchor_match_radius_px
                .unwrap_or(defaults.anchor_keypoint_match_radius_px),
            anchor_keypoint_max_descriptor_distance: args
                .stereo_landmark_replenish_anchor_max_descriptor_distance
                .or(defaults.anchor_keypoint_max_descriptor_distance),
            duplicate_suppression_radius_px: args
                .stereo_landmark_replenish_duplicate_radius_px
                .unwrap_or(defaults.duplicate_suppression_radius_px),
            min_parallax_deg: args
                .stereo_landmark_replenish_min_parallax_deg
                .unwrap_or(defaults.min_parallax_deg),
            min_depth_meters: args
                .stereo_landmark_replenish_min_depth_meters
                .unwrap_or(defaults.min_depth_meters),
            max_depth_meters: args
                .stereo_landmark_replenish_max_depth_meters
                .unwrap_or(defaults.max_depth_meters),
            bootstrap_config: defaults.bootstrap_config,
        }
    };

    for (frame_idx, image_entry) in dataset.cam0_images.iter().enumerate().skip(seed_frame_idx) {
        if frames_recorded >= frame_cap {
            break;
        }
        let frame_processing_started_at = Instant::now();
        // Drain IMU samples whose timestamp falls on or before this cam0
        // frame.
        while imu_idx < dataset.imu_samples.len() {
            let sample = &dataset.imu_samples[imu_idx];
            if sample.timestamp_nanoseconds > image_entry.timestamp_nanoseconds {
                break;
            }
            let dt_ns = sample.timestamp_nanoseconds - prev_imu_ts;
            if dt_ns > 0 {
                let dt = dt_ns as f64 * 1.0e-9;
                slam.push_imu_measurement(sample.gyro, sample.accel, dt);
                slam.tracker
                    .motion_model_mut()
                    .push_imu_measurement(sample.gyro, sample.accel, dt);
            }
            prev_imu_ts = sample.timestamp_nanoseconds;
            imu_idx += 1;
        }

        // Decode the cam0 PNG and extract real corner features. If the
        // image fails to load (corrupt frame, etc.) we skip it rather than
        // poisoning the pipeline.
        let image_path = dataset.cam0_image_dir.join(&image_entry.filename);
        let image: GrayscaleImage = match read_common_image(&image_path) {
            Ok(image) => image,
            Err(err) => {
                eprintln!(
                    "skipping frame_idx={frame_idx} due to image-decode error: {} ({err})",
                    image_path.display()
                );
                continue;
            }
        };
        extractor.set_frame_idx(frame_idx);
        let features = match extractor.extract(&image) {
            Ok(features) => undistort_feature_keypoints(&distortion, &camera, &features),
            Err(err) => {
                eprintln!("skipping frame_idx={frame_idx} due to feature-extraction error: {err}");
                continue;
            }
        };
        feature_count_sum += features.len();
        feature_count_min = feature_count_min.min(features.len());
        feature_count_max = feature_count_max.max(features.len());
        if args.export_frame_appearance_descriptors {
            if let Some(mean) = normalized_mean_descriptor(&features.descriptors) {
                if let Some(dim) = frame_appearance_descriptor_dim {
                    if dim != mean.len() {
                        return Err(format!(
                            "frame_idx={frame_idx}: appearance descriptor dim {} != {dim}",
                            mean.len()
                        )
                        .into());
                    }
                } else {
                    frame_appearance_descriptor_dim = Some(mean.len());
                }
                let values = mean
                    .iter()
                    .map(|value| format!("{value:.9}"))
                    .collect::<Vec<_>>()
                    .join(",");
                frame_appearance_descriptor_rows.push(format!(
                    "{},{frame_idx},{},{}\n",
                    image_entry.timestamp_nanoseconds,
                    features.descriptors.len(),
                    values
                ));
                frame_appearance_descriptor_count += 1;
            }
        }
        let frame = frame_from_features(frame_idx as u64, camera_id, &features);

        // Extract cam1 once per frame and share the undistorted features
        // between re-bootstrap and landmark replenishment below. Metric
        // stereo loop closure fixes scale to one (ORB-SLAM2/3 convention),
        // so this demo deliberately does not build a second 3D point set for
        // per-loop Sim3 scale estimation.
        let cam1_features_for_frame = if args.stereo_landmark_replenish {
            if let Some(cam1_setup) = cam1_stereo_setup.as_ref() {
                if let Some(cam1_idx) = dataset.cam1_images.iter().position(|entry| {
                    entry.timestamp_nanoseconds == image_entry.timestamp_nanoseconds
                }) {
                    let cam1_image_path = dataset
                        .cam1_image_dir
                        .join(&dataset.cam1_images[cam1_idx].filename);
                    if let Ok(cam1_image) = read_common_image(&cam1_image_path) {
                        extractor.set_camera(SuperPointCamera::Cam1);
                        extractor.set_frame_idx(cam1_idx);
                        let extracted = extractor.extract(&cam1_image);
                        extractor.set_camera(SuperPointCamera::Cam0);
                        if let Ok(raw) = extracted {
                            let cam1_features = undistort_feature_keypoints(
                                &cam1_setup.distortion,
                                &cam1_setup.camera,
                                &raw,
                            );
                            Some(cam1_features)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // The most recently applied keyframe, captured *before* this
        // frame's own `process_frame` call, anchors any replenishment
        // candidates built below: `process_keyframe`'s triangulator only
        // accepts observations against keyframes already present in the
        // map, so a second gated real observation from this pre-existing
        // keyframe is what lets a same-frame stereo match become a valid
        // two-view `LandmarkCandidate`.
        let replenish_anchor_frame_id = if args.stereo_landmark_replenish {
            slam.map().keyframes.keys().copied().max()
        } else {
            None
        };
        // Cloned rather than drained: `process_keyframe` only ever looks
        // at `candidates` when this call's tracked frame is itself
        // selected as a new keyframe (an early return skips them
        // otherwise), so a queued candidate must survive across however
        // many non-keyframe frames occur before the next keyframe is
        // selected. The queue is cleared below only once a call actually
        // considers it.
        let candidates_to_submit: Vec<LandmarkCandidate> = if args.stereo_landmark_replenish {
            pending_replenish_candidates.clone()
        } else {
            Vec::new()
        };
        let tracking_stats_before = slam.tracker.stats().clone();
        let mut result = slam.process_frame(&frame, candidates_to_submit);
        if result.tracking_succeeded() {
            last_successful_atlas_boundary_keyframe =
                Some(keyframe_from_tracking_result(&frame, &result.tracking));
        }
        let tracking_stats_after = slam.tracker.stats();
        let projection_attempt_count = tracking_stats_after
            .projection_guided_attempt_count
            .saturating_sub(tracking_stats_before.projection_guided_attempt_count);
        let projection_widen_retry_count = tracking_stats_after
            .projection_guided_widen_retry_count
            .saturating_sub(tracking_stats_before.projection_guided_widen_retry_count);
        let projection_succeeded = tracking_stats_after.projection_guided_success_count
            > tracking_stats_before.projection_guided_success_count;
        let appearance_fallback_succeeded = tracking_stats_after
            .projection_guided_fallback_success_count
            > tracking_stats_before.projection_guided_fallback_success_count;
        let local_map_refinement_accepted = tracking_stats_after
            .local_map_refinement_accepted_count
            > tracking_stats_before.local_map_refinement_accepted_count;
        let local_map_refinement_rejected = tracking_stats_after
            .local_map_refinement_rejected_count
            > tracking_stats_before.local_map_refinement_rejected_count;
        let pose_prior_visual_override_used = tracking_stats_after.pose_prior_visual_override_count
            > tracking_stats_before.pose_prior_visual_override_count;
        let tracking_path = if projection_succeeded {
            "projection"
        } else if appearance_fallback_succeeded {
            "appearance_fallback"
        } else if projection_attempt_count > 0 {
            "projection_and_fallback_failed"
        } else {
            "appearance_global"
        };
        // Preserve the raw tracker decision before optional relocalization or
        // rebootstrap replaces it below. In particular, quality-gate rejected
        // PnP estimates intentionally retain their pose and reprojection
        // statistics, which makes this log able to distinguish an observation
        // cliff from a motion-prior innovation-gate rejection.
        let (pose_prior_translation_innovation_m, pose_prior_rotation_innovation_deg) = result
            .tracking
            .pose_prior
            .as_ref()
            .zip(result.tracking.localization.pose.as_ref())
            .map(|(prior, estimated)| {
                let translation =
                    (estimated.camera_center_world() - prior.camera_center_world()).norm();
                let prior_rotation = prior.camera_to_world().rotation;
                let estimated_rotation = estimated.camera_to_world().rotation;
                let rotation_deg = prior_rotation
                    .rotation_to(&estimated_rotation)
                    .angle()
                    .to_degrees();
                (Some(translation), Some(rotation_deg))
            })
            .unwrap_or((None, None));
        let estimator_diagnostics = result.tracking.localization.estimator_diagnostics.as_ref();
        let estimated_pose_fields = result
            .tracking
            .localization
            .pose
            .as_ref()
            .map(|pose| {
                let center = pose.camera_center_world();
                let rotation = pose.camera_to_world().rotation;
                (
                    Some(center.x),
                    Some(center.y),
                    Some(center.z),
                    Some(rotation.w),
                    Some(rotation.i),
                    Some(rotation.j),
                    Some(rotation.k),
                )
            })
            .unwrap_or((None, None, None, None, None, None, None));
        tracking_diagnostics_csv.push_str(&format!(
            "{frame_idx},{},{:?},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            image_entry.timestamp_nanoseconds,
            result.tracking.state,
            result.tracking.event,
            bool_as_u8(result.tracking.localization.success),
            result.tracking.successive_failures,
            bool_as_u8(result.tracking.used_pose_prior),
            bool_as_u8(result.tracking.localization.pose.is_some()),
            format_optional_f64(estimated_pose_fields.0),
            format_optional_f64(estimated_pose_fields.1),
            format_optional_f64(estimated_pose_fields.2),
            format_optional_f64(estimated_pose_fields.3),
            format_optional_f64(estimated_pose_fields.4),
            format_optional_f64(estimated_pose_fields.5),
            format_optional_f64(estimated_pose_fields.6),
            tracking_path,
            projection_attempt_count,
            projection_widen_retry_count,
            bool_as_u8(local_map_refinement_accepted),
            bool_as_u8(local_map_refinement_rejected),
            result.tracking.localization.candidate_landmark_count,
            result.tracking.localization.match_count,
            result.tracking.localization.correspondence_count,
            result.tracking.localization.inlier_count,
            result.tracking.localization.outlier_count,
            result.tracking.localization.inlier_ratio,
            format_optional_f64(result.tracking.localization.reprojection_error),
            format_optional_f64(result.tracking.localization.median_reprojection_error),
            format_optional_f64(result.tracking.localization.max_reprojection_error),
            format_optional_f64(pose_prior_translation_innovation_m),
            format_optional_f64(pose_prior_rotation_innovation_deg),
            quote_csv_field(
                &result
                    .tracking
                    .tracking_failure_reason
                    .as_ref()
                    .map(|reason| format!("{reason:?}"))
                    .unwrap_or_default(),
            ),
            quote_csv_field(
                &result
                    .tracking
                    .localization
                    .failure_reason
                    .as_ref()
                    .map(|reason| format!("{reason:?}"))
                    .unwrap_or_default(),
            ),
            estimator_diagnostics
                .map(|diagnostics| bool_as_u8(diagnostics.refinement_applied).to_string())
                .unwrap_or_default(),
            format_optional_f64(
                estimator_diagnostics
                    .and_then(|diagnostics| diagnostics.pre_refinement_mean_reprojection_error),
            ),
            format_optional_f64(
                estimator_diagnostics
                    .and_then(|diagnostics| diagnostics.post_refinement_mean_reprojection_error),
            ),
            format_optional_f64(
                estimator_diagnostics.and_then(|diagnostics| diagnostics.refinement_error_delta),
            ),
            bool_as_u8(pose_prior_visual_override_used),
        ));
        if frame_idx == seed_frame_idx {
            let observed_landmarks = slam
                .map()
                .keyframes
                .get(&(frame_idx as u64))
                .map(|keyframe| {
                    keyframe
                        .observations
                        .iter()
                        .map(|observation| observation.landmark_id)
                        .collect::<std::collections::HashSet<_>>()
                })
                .unwrap_or_default();
            seed_stereo_observations
                .retain(|stereo| observed_landmarks.contains(&stereo.landmark_id));
            slam.map_mut()
                .stereo_observations
                .append(&mut seed_stereo_observations);
            result.sparse_factor_graph = slam.sync_sparse_factor_graph_keyframe(frame_idx as u64);
        }
        let mut success = result.tracking_succeeded();
        if args.stereo_landmark_replenish
            && result
                .mapping
                .as_ref()
                .is_some_and(|mapping| mapping.keyframe_decision.selected)
        {
            pending_replenish_candidates.clear();
        }
        // Treat quality-gate rejections as "no estimate this frame": their
        // `localization.pose` still carries the rejected PnP result (the
        // tracker only flips `success = false`), which would otherwise leak
        // a known-bad pose into the trajectory CSV and ATE summation.
        let mut tracked = if success {
            result.tracking.localization.pose.clone()
        } else {
            None
        };
        let mut rebootstrap_mapping_override: Option<LocalMappingResult> = None;
        let mut atlas_bridge_attempted_this_frame = false;
        if rebootstrap_enabled {
            if success {
                consecutive_lost_frames = 0;
            } else {
                consecutive_lost_frames += 1;
                let reloc_recovered = result
                    .relocalization
                    .as_ref()
                    .is_some_and(|reloc| reloc.succeeded);
                let cooldown_ok = last_rebootstrap_frame_idx
                    .map(|last| frame_idx.saturating_sub(last) >= args.rebootstrap_cooldown_frames)
                    .unwrap_or(true);
                if !reloc_recovered
                    && cooldown_ok
                    && consecutive_lost_frames
                        >= args.rebootstrap_after_lost_frames.unwrap_or(usize::MAX)
                {
                    let min_stereo_matches = args.keyframe_min_inliers.unwrap_or(15).max(1);
                    if let (Some(cam1_setup), Some(cam1_features)) =
                        (cam1_stereo_setup.as_ref(), cam1_features_for_frame.as_ref())
                    {
                        let stereo_matches = bootstrap_stereo_landmarks(
                            &camera,
                            &cam1_setup.camera,
                            &cam1_setup.cam0_to_cam1,
                            &features,
                            cam1_features,
                            &StereoBootstrapConfig::default(),
                        );
                        if stereo_matches.len() >= min_stereo_matches {
                            let independent_restart = args.rebootstrap_independent_submap;
                            let adjacent_boundary_target = independent_restart.then(|| {
                                last_successful_atlas_boundary_keyframe
                                    .as_ref()
                                    .map(|keyframe| (active_submap_id, keyframe.frame.id))
                            });
                            let adjacent_boundary_target = adjacent_boundary_target.flatten();
                            let restart_atlas_pose_prior = independent_restart
                                .then(|| {
                                    let last_pose = slam.tracker.last_successful_pose()?.clone();
                                    let atlas_from_old = map_atlas
                                        .as_ref()?
                                        .submap(active_submap_id)?
                                        .atlas_from_local
                                        .as_ref()?;
                                    Some(Pose {
                                        world_to_camera: last_pose
                                            .world_to_camera
                                            .compose(&atlas_from_old.inverse()),
                                    })
                                })
                                .flatten();
                            let seed_pose = if independent_restart {
                                Pose::from_world_to_camera(
                                    UnitQuaternion::identity(),
                                    Vector3::zeros(),
                                )
                            } else {
                                // Legacy same-gauge continuation is causal and
                                // GT-free, but can silently inherit a damaged
                                // scale/pose branch. Atlas mode below avoids
                                // that coupling by starting at identity.
                                slam.tracker
                                    .last_successful_pose()
                                    .cloned()
                                    .unwrap_or_else(|| {
                                        Pose::from_world_to_camera(
                                            UnitQuaternion::identity(),
                                            Vector3::zeros(),
                                        )
                                    })
                            };

                            if independent_restart {
                                let atlas = map_atlas
                                    .as_mut()
                                    .expect("independent re-bootstrap requires an Atlas");
                                let mut atlas_snapshot = slam.map().clone();
                                if let Some(boundary_keyframe) =
                                    last_successful_atlas_boundary_keyframe.as_ref()
                                {
                                    if insert_atlas_boundary_keyframe_snapshot(
                                        &mut atlas_snapshot,
                                        boundary_keyframe,
                                    ) {
                                        atlas_boundary_keyframe_snapshots += 1;
                                    }
                                }
                                atlas.replace_submap_map(active_submap_id, atlas_snapshot)?;

                                let mut independent_map = VisualMap::new();
                                independent_map.cameras = slam.map().cameras.clone();
                                active_submap_id =
                                    atlas.try_insert_independent(independent_map.clone())?;
                                active_submap_atlas_prior =
                                    restart_atlas_pose_prior.as_ref().map(|atlas_pose| {
                                        atlas_pose
                                            .world_to_camera
                                            .inverse()
                                            .compose(&seed_pose.world_to_camera)
                                    });

                                // Reset every per-map continuation state. In
                                // particular, no old motion prior, keyframe
                                // policy, VI linearization, relocalization
                                // cache, or pose graph may cross the gauge
                                // boundary. Raw IMU processing resumes on the
                                // next frame in the new segment.
                                let mut restart_tracker = slam.tracker.clone();
                                restart_tracker.reset();
                                let mut restart_mapper = slam.mapper.clone();
                                restart_mapper.reset();
                                slam = OnlineSlamPipeline::new(
                                    independent_map,
                                    restart_tracker,
                                    restart_mapper,
                                    slam.config.clone(),
                                );
                            }
                            let landmark_ids = append_stereo_bootstrap_landmarks_to_map(
                                slam.map_mut(),
                                &seed_pose,
                                &features,
                                &stereo_matches,
                                &mut next_rebootstrap_landmark_id,
                            );
                            let map_stats = map_provider_stats_from_map(slam.map());
                            let restart_tracking = build_stereo_segment_restart_tracking_result(
                                frame_idx as u64,
                                seed_pose,
                                &stereo_matches,
                                &landmark_ids,
                                map_stats,
                            );
                            slam.tracker
                                .accept_segment_restart_result(restart_tracking.clone());
                            if let Some(state) = slam.relocalization_state.as_mut() {
                                state.consecutive_failed_attempts = 0;
                                state.pending_confirmation = None;
                            }
                            pending_replenish_candidates.clear();
                            let keyframe = keyframe_from_tracking_result(&frame, &restart_tracking);
                            let map_snapshot = slam.map().clone();
                            let mapping_result = slam.mapper.process_keyframe(
                                &map_snapshot,
                                &restart_tracking,
                                keyframe,
                                Vec::<LandmarkCandidate>::new(),
                            );
                            if slam.config.apply_map_updates
                                && mapping_result.staged_update_validation.is_valid()
                            {
                                let _ = mapping_result
                                    .staged_update
                                    .clone()
                                    .apply_to(slam.map_mut());
                            }
                            last_successful_atlas_boundary_keyframe =
                                slam.map().keyframes.get(&(frame_idx as u64)).cloned();
                            let observed_landmarks = slam
                                .map()
                                .keyframes
                                .get(&(frame_idx as u64))
                                .map(|keyframe| {
                                    keyframe
                                        .observations
                                        .iter()
                                        .map(|observation| observation.landmark_id)
                                        .collect::<std::collections::HashSet<_>>()
                                })
                                .unwrap_or_default();
                            for (stereo_match, landmark_id) in
                                stereo_matches.iter().zip(landmark_ids.iter().copied())
                            {
                                if observed_landmarks.contains(&landmark_id) {
                                    slam.map_mut().stereo_observations.push(StereoObservation {
                                        frame_id: frame_idx as u64,
                                        landmark_id,
                                        right_camera_id: cam1_setup.camera.id,
                                        xy_right: cam1_features.keypoints
                                            [stereo_match.right_keypoint_index],
                                        left_to_right: cam1_setup.cam0_to_cam1.clone(),
                                    });
                                }
                            }
                            result.sparse_factor_graph =
                                slam.sync_sparse_factor_graph_keyframe(frame_idx as u64);
                            if independent_restart {
                                let atlas = map_atlas
                                    .as_mut()
                                    .expect("independent re-bootstrap requires an Atlas");
                                atlas.replace_submap_map(active_submap_id, slam.map().clone())?;
                                atlas_merge_attempts += 1;
                                let mut boundary_verified = false;
                                if let Some((target_submap_id, target_frame_id)) =
                                    adjacent_boundary_target
                                {
                                    atlas_boundary_factor_attempts += 1;
                                    let target_from_source_prior = active_submap_atlas_prior
                                        .as_ref()
                                        .zip(
                                            atlas.submap(target_submap_id).and_then(|submap| {
                                                submap.atlas_from_local.as_ref()
                                            }),
                                        )
                                        .map(|(atlas_from_source, atlas_from_target)| {
                                            atlas_from_target.inverse().compose(atlas_from_source)
                                        });
                                    let factor = atlas.align_adjacent_boundary_factor(
                                        active_submap_id,
                                        frame_idx as u64,
                                        target_submap_id,
                                        target_frame_id,
                                        target_from_source_prior.as_ref(),
                                        &atlas_bridge_localizer.matcher,
                                        &atlas_bridge_config,
                                    )?;
                                    append_atlas_boundary_factor(
                                        &mut atlas_boundary_factors_csv,
                                        frame_idx,
                                        image_entry.timestamp_nanoseconds,
                                        &factor,
                                    );
                                    if factor.verified_merge.is_some() {
                                        atlas_boundary_factor_successes += 1;
                                        atlas_merge_successes += 1;
                                        active_submap_atlas_prior = None;
                                        boundary_verified = true;
                                    }
                                    println!(
                                        "atlas boundary factor source_submap={} target_submap={} frame_idx={} descriptor_matches={} spatial_matches={} metric={} refined_metric={} refinement_iterations={} used_prior={} rigid_inliers={} verified={}",
                                        active_submap_id,
                                        target_submap_id,
                                        frame_idx,
                                        factor.descriptor_match_count,
                                        factor.spatial_mutual_match_count,
                                        factor.metric_correspondence_count,
                                        factor.projection_refined_correspondence_count,
                                        factor.projection_refinement_iterations,
                                        factor.used_transform_prior,
                                        factor.rigid_inlier_count,
                                        boundary_verified,
                                    );
                                }
                                if !boundary_verified {
                                    let source_window = atlas.source_keyframe_recovery_window(
                                        active_submap_id,
                                        frame_idx as u64,
                                        args.atlas_broader_recovery_max_source_keyframes,
                                    )?;
                                    let window_alignment = atlas.align_submap_keyframe_window(
                                        active_submap_id,
                                        &source_window,
                                        &camera,
                                        active_submap_atlas_prior.as_ref(),
                                        &atlas_bridge_localizer,
                                        &atlas_bridge_config,
                                    )?;
                                    atlas_merge_source_keyframe_attempts +=
                                        window_alignment.attempted_source_frame_ids.len();
                                    if window_alignment.verified_merge.is_some() {
                                        atlas_merge_successes += 1;
                                        active_submap_atlas_prior = None;
                                    }
                                    for alignment in &window_alignment.alignments {
                                        append_atlas_alignment_attempt(
                                            &mut atlas_merge_attempts_csv,
                                            atlas_merge_attempts,
                                            frame_idx,
                                            image_entry.timestamp_nanoseconds,
                                            alignment,
                                        );
                                    }
                                    println!(
                                        "atlas bridge source_submap={} frame_idx={} source_views={} target_attempts={} verified={}",
                                        active_submap_id,
                                        frame_idx,
                                        window_alignment.attempted_source_frame_ids.len(),
                                        window_alignment
                                            .alignments
                                            .iter()
                                            .map(|alignment| alignment.diagnostics.len())
                                            .sum::<usize>(),
                                        window_alignment.verified_merge.is_some(),
                                    );
                                }
                                atlas_bridge_attempted_this_frame = true;
                            }
                            rebootstrap_mapping_override = Some(mapping_result);
                            segment_id = if independent_restart {
                                active_submap_id as usize
                            } else {
                                segment_id + 1
                            };
                            rebootstrap_events += 1;
                            rebootstrap_event_idx += 1;
                            let seed_source = if independent_restart {
                                "independent_identity"
                            } else {
                                "last_successful_pose"
                            };
                            rebootstrap_log_csv.push_str(&format!(
                                        "{rebootstrap_event_idx},{segment_id},{frame_idx},{},{seed_source},{},{},{},{}\n",
                                        image_entry.timestamp_nanoseconds,
                                        stereo_matches.len(),
                                        landmark_ids.len(),
                                        rebootstrap_mapping_override
                                            .as_ref()
                                            .map(|mapping| {
                                                u8::from(mapping.keyframe_decision.selected)
                                            })
                                            .unwrap_or(0),
                                        consecutive_lost_frames,
                                    ));
                            last_rebootstrap_frame_idx = Some(frame_idx);
                            consecutive_lost_frames = 0;
                            result.tracking = restart_tracking;
                            success = true;
                            tracked = result.tracking.localization.pose.clone();
                            println!(
                                        "rebootstrap segment_id={segment_id} frame_idx={frame_idx} stereo_matches={} landmarks_added={} keyframe_selected={}",
                                        stereo_matches.len(),
                                        landmark_ids.len(),
                                        rebootstrap_mapping_override
                                            .as_ref()
                                            .map(|mapping| mapping.keyframe_decision.selected)
                                            .unwrap_or(false),
                                    );
                        }
                    }
                }
            }
        }
        if success {
            tracking_successes += 1;
        }
        // Stereo landmark replenishment: for cam0 keypoints tracking did NOT
        // match to an existing landmark this frame, reuse the cam1 features
        // extracted above and stage two-observation
        // `LandmarkCandidate`s (this frame's real pixel + a gated real
        // observation in the most-recent keyframe, predicted by the stereo
        // reprojection) for
        // submission on the *next* frame's `process_frame` call. The anchor
        // observation is a gated real keypoint measurement; the stereo 3D
        // reprojection is used only to predict that association. All the
        // matching / triangulation / gating logic (the three-defect-hardened
        // module) lives in `build_stereo_replenish_candidates`; here we only
        // assemble its inputs — cam1 features for this instant, the
        // just-solved pose, and the PnP-inlier index set — and queue the
        // result.
        if args.stereo_landmark_replenish && success {
            if let (
                Some(anchor_frame_id),
                Some(cam1_setup),
                Some(current_pose),
                Some(cam1_features),
            ) = (
                replenish_anchor_frame_id,
                cam1_stereo_setup.as_ref(),
                tracked.as_ref(),
                cam1_features_for_frame.as_ref(),
            ) {
                let matched_cam0_indices: std::collections::HashSet<usize> = result
                    .tracking
                    .localization
                    .inlier_query_indices
                    .iter()
                    .copied()
                    .collect();
                let new_candidates = build_stereo_replenish_candidates(
                    slam.map(),
                    anchor_frame_id,
                    frame_idx as u64,
                    &cam1_setup.camera,
                    &cam1_setup.cam0_to_cam1,
                    &features,
                    cam1_features,
                    &matched_cam0_indices,
                    current_pose,
                    next_replenish_candidate_id,
                    &stereo_replenish_config,
                );
                next_replenish_candidate_id += new_candidates.len() as u64;
                stereo_landmark_replenish_candidates_total += new_candidates.len();
                pending_replenish_candidates.extend(new_candidates);
            }
        }
        if let Some(reloc) = result.relocalization.as_ref() {
            if reloc.attempted {
                relocalization_attempts += 1;
                let gate_config = RelocalizationAttemptGateConfig::from_args(&args);
                let gates = relocalization_attempt_gate_status(reloc, gate_config);
                let reject_reason = relocalization_attempt_reject_reason(reloc, gates);
                relocalization_attempts_csv.push_str(&format!(
                    "{frame_idx},{},{},{},{},{},{},{},{},{:.9},{:.9},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    image_entry.timestamp_nanoseconds,
                    bool_as_u8(reloc.attempted),
                    bool_as_u8(reloc.succeeded),
                    bool_as_u8(reloc.localization_success),
                    quote_csv_field(reject_reason),
                    reloc.inlier_count,
                    args.relocalization_min_inliers,
                    bool_as_u8(gates.min_inliers_pass),
                    reloc.inlier_ratio,
                    args.relocalization_min_inlier_ratio,
                    bool_as_u8(gates.min_inlier_ratio_pass),
                    reloc.correspondence_count,
                    format_optional_f64(reloc.mean_reprojection_error),
                    format_optional_f64(args.relocalization_max_reprojection_error),
                    bool_as_u8(gates.reprojection_pass),
                    format_optional_f64(reloc.translation_per_frame_from_last_success_meters),
                    format_optional_f64(
                        args.relocalization_max_translation_per_frame_from_last_success_meters
                    ),
                    bool_as_u8(gates.continuity_pass),
                    format_optional_f64(reloc.inlier_depth_median_ratio_to_last_success),
                    format_optional_f64(
                        args.relocalization_min_inlier_depth_median_ratio_to_last_success
                    ),
                    format_optional_f64(
                        args.relocalization_max_inlier_depth_median_ratio_to_last_success
                    ),
                    bool_as_u8(gates.depth_ratio_pass),
                    bool_as_u8(reloc.passed_acceptance_gates),
                    reloc.confirmation_count,
                    reloc.confirmation_required_count,
                    format_optional_f64(
                        reloc.confirmation_translation_per_frame_from_previous_meters
                    ),
                    reloc.descriptor_store_landmark_count,
                    format_optional_usize(reloc.covisibility_local_descriptor_store_landmark_count),
                    format_optional_usize(reloc.appearance_descriptor_store_landmark_count),
                    format_optional_usize(reloc.broader_descriptor_store_landmark_count),
                    bool_as_u8(reloc.tried_covisibility_local_descriptor_store),
                    bool_as_u8(reloc.used_covisibility_local_descriptor_store),
                    bool_as_u8(reloc.tried_appearance_descriptor_store),
                    bool_as_u8(reloc.used_appearance_descriptor_store),
                    bool_as_u8(reloc.tried_broader_descriptor_store_fallback),
                    bool_as_u8(reloc.broader_descriptor_store_retry_skipped_by_interval),
                    bool_as_u8(reloc.used_broader_descriptor_store_fallback),
                ));
                relocalization_descriptor_store_count_observations += 1;
                relocalization_descriptor_store_count_sum += reloc.descriptor_store_landmark_count;
                relocalization_descriptor_store_count_min = Some(
                    relocalization_descriptor_store_count_min
                        .map_or(reloc.descriptor_store_landmark_count, |current| {
                            current.min(reloc.descriptor_store_landmark_count)
                        }),
                );
                relocalization_descriptor_store_count_max = Some(
                    relocalization_descriptor_store_count_max
                        .map_or(reloc.descriptor_store_landmark_count, |current| {
                            current.max(reloc.descriptor_store_landmark_count)
                        }),
                );
                if reloc.tried_covisibility_local_descriptor_store {
                    relocalization_covisibility_descriptor_store_tried_frames += 1;
                }
                if reloc.used_covisibility_local_descriptor_store {
                    relocalization_covisibility_descriptor_store_used_frames += 1;
                }
                if reloc.tried_appearance_descriptor_store {
                    relocalization_appearance_descriptor_store_tried_frames += 1;
                    relocalization_appearance_candidate_keyframe_count_observations += 1;
                    relocalization_appearance_candidate_keyframe_count_sum +=
                        reloc.appearance_candidate_keyframe_count;
                }
                if reloc.used_appearance_descriptor_store {
                    relocalization_appearance_descriptor_store_used_frames += 1;
                }
                if let Some(similarity) = reloc.appearance_best_similarity {
                    relocalization_appearance_best_similarity_count += 1;
                    relocalization_appearance_best_similarity_sum += similarity as f64;
                    relocalization_appearance_best_similarity_max = Some(
                        relocalization_appearance_best_similarity_max
                            .map_or(similarity, |current| current.max(similarity)),
                    );
                }
                for (rank, candidate) in reloc.appearance_candidates.iter().enumerate() {
                    relocalization_appearance_candidates_csv.push_str(&format!(
                        "{},{},{:.9},{},{},{},{},{},{},{},{},{}\n",
                        frame_idx,
                        candidate.keyframe_id,
                        candidate.similarity,
                        rank + 1,
                        image_entry.timestamp_nanoseconds,
                        reloc.descriptor_store_landmark_count,
                        reloc
                            .appearance_descriptor_store_landmark_count
                            .map(|count| count.to_string())
                            .unwrap_or_default(),
                        reloc.attempted as u8,
                        reloc.succeeded as u8,
                        reloc.passed_acceptance_gates as u8,
                        reloc.used_appearance_descriptor_store as u8,
                        reloc.used_broader_descriptor_store_fallback as u8
                    ));
                }
                if reloc.tried_broader_descriptor_store_fallback {
                    relocalization_broader_descriptor_store_retry_frames += 1;
                }
                if reloc.broader_descriptor_store_retry_skipped_by_interval {
                    relocalization_broader_descriptor_store_retry_interval_skips += 1;
                }
                if reloc.used_broader_descriptor_store_fallback {
                    relocalization_broader_descriptor_store_used_frames += 1;
                }
                if reloc.covisibility_reference_keyframe_id.is_some() {
                    relocalization_covisibility_reference_keyframe_count += 1;
                }
            }
            if reloc.passed_acceptance_gates {
                relocalization_gate_passes += 1;
                if !reloc.succeeded {
                    relocalization_confirmation_waiting += 1;
                }
            }
            if let Some(tx_per_frame) =
                reloc.confirmation_translation_per_frame_from_previous_meters
            {
                relocalization_confirmation_tx_per_frame_count += 1;
                relocalization_confirmation_tx_per_frame_sum += tx_per_frame;
                relocalization_confirmation_tx_per_frame_max = Some(
                    relocalization_confirmation_tx_per_frame_max
                        .map(|current| current.max(tx_per_frame))
                        .unwrap_or(tx_per_frame),
                );
            }
            if let Some(tx_per_frame) = reloc.translation_per_frame_from_last_success_meters {
                relocalization_tx_per_frame_count += 1;
                relocalization_tx_per_frame_sum += tx_per_frame;
                relocalization_tx_per_frame_max = Some(
                    relocalization_tx_per_frame_max
                        .map(|current| current.max(tx_per_frame))
                        .unwrap_or(tx_per_frame),
                );
            }
            if let Some(depth_ratio) = reloc.inlier_depth_median_ratio_to_last_success {
                relocalization_depth_ratio_count += 1;
                relocalization_depth_ratio_sum += depth_ratio;
                relocalization_depth_ratio_min = Some(
                    relocalization_depth_ratio_min
                        .map(|current| current.min(depth_ratio))
                        .unwrap_or(depth_ratio),
                );
                relocalization_depth_ratio_max = Some(
                    relocalization_depth_ratio_max
                        .map(|current| current.max(depth_ratio))
                        .unwrap_or(depth_ratio),
                );
            }
            if reloc.succeeded {
                relocalization_successes += 1;
                if let Some(tx_per_frame) = reloc.translation_per_frame_from_last_success_meters {
                    relocalization_success_tx_per_frame_count += 1;
                    relocalization_success_tx_per_frame_sum += tx_per_frame;
                    relocalization_success_tx_per_frame_max = Some(
                        relocalization_success_tx_per_frame_max
                            .map(|current| current.max(tx_per_frame))
                            .unwrap_or(tx_per_frame),
                    );
                }
                if let Some(depth_ratio) = reloc.inlier_depth_median_ratio_to_last_success {
                    relocalization_success_depth_ratio_count += 1;
                    relocalization_success_depth_ratio_sum += depth_ratio;
                    relocalization_success_depth_ratio_min = Some(
                        relocalization_success_depth_ratio_min
                            .map(|current| current.min(depth_ratio))
                            .unwrap_or(depth_ratio),
                    );
                    relocalization_success_depth_ratio_max = Some(
                        relocalization_success_depth_ratio_max
                            .map(|current| current.max(depth_ratio))
                            .unwrap_or(depth_ratio),
                    );
                }
            }
        }
        let atlas_retry_keyframe_selected = result
            .mapping
            .as_ref()
            .or(rebootstrap_mapping_override.as_ref())
            .is_some_and(|mapping| mapping.keyframe_decision.selected);
        if !atlas_bridge_attempted_this_frame
            && success
            && atlas_retry_keyframe_selected
            && map_atlas.as_ref().is_some_and(|atlas| {
                atlas
                    .submap(active_submap_id)
                    .is_some_and(|submap| submap.atlas_from_local.is_none())
            })
            && slam.map().keyframes.contains_key(&(frame_idx as u64))
        {
            let atlas = map_atlas
                .as_mut()
                .expect("Atlas retry requires independent-submap mode");
            atlas.replace_submap_map(active_submap_id, slam.map().clone())?;
            atlas_merge_attempts += 1;
            let max_source_keyframes = if args.atlas_broader_recovery_max_source_keyframes > 1
                && atlas_merge_attempts % args.atlas_broader_recovery_interval_attempts == 0
            {
                atlas_broader_recovery_cycles += 1;
                args.atlas_broader_recovery_max_source_keyframes
            } else {
                1
            };
            let source_window = atlas.source_keyframe_recovery_window(
                active_submap_id,
                frame_idx as u64,
                max_source_keyframes,
            )?;
            let window_alignment = atlas.align_submap_keyframe_window(
                active_submap_id,
                &source_window,
                &camera,
                active_submap_atlas_prior.as_ref(),
                &atlas_bridge_localizer,
                &atlas_bridge_config,
            )?;
            atlas_merge_source_keyframe_attempts +=
                window_alignment.attempted_source_frame_ids.len();
            if window_alignment.verified_merge.is_some() {
                atlas_merge_successes += 1;
                active_submap_atlas_prior = None;
            }
            for alignment in &window_alignment.alignments {
                append_atlas_alignment_attempt(
                    &mut atlas_merge_attempts_csv,
                    atlas_merge_attempts,
                    frame_idx,
                    image_entry.timestamp_nanoseconds,
                    alignment,
                );
            }
            println!(
                "atlas bridge retry source_submap={} frame_idx={} source_views={} target_attempts={} verified={}",
                active_submap_id,
                frame_idx,
                window_alignment.attempted_source_frame_ids.len(),
                window_alignment
                    .alignments
                    .iter()
                    .map(|alignment| alignment.diagnostics.len())
                    .sum::<usize>(),
                window_alignment.verified_merge.is_some(),
            );
        }
        if let Some(mapping) = result
            .mapping
            .as_ref()
            .or(rebootstrap_mapping_override.as_ref())
        {
            let decision = &mapping.keyframe_decision;
            if matches!(
                decision.reason,
                KeyframeDecisionReason::InsufficientTrackingQuality { .. }
            ) {
                keyframe_insufficient_tracking_quality_rejections += 1;
            }
            let tracking_failure_reason = result
                .tracking
                .tracking_failure_reason
                .as_ref()
                .map(|reason| quote_csv_field(&format!("{reason:?}")))
                .unwrap_or_default();
            let (
                reason,
                frame_id_gap,
                translation_m,
                min_translation_m,
                tracked_landmarks,
                last_keyframe_tracked_landmarks,
                min_tracked_landmark_ratio,
            ) = keyframe_reason_csv_fields(&decision.reason);
            keyframe_decision_log.push_str(&format!(
                "{frame_idx},{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{}\n",
                image_entry.timestamp_nanoseconds,
                if success { 1 } else { 0 },
                if decision.selected { 1 } else { 0 },
                reason,
                format_optional_frame_id(decision.last_keyframe_frame_id),
                decision.selected_keyframe_count,
                result.tracking.localization.inlier_count,
                result.tracking.localization.inlier_ratio,
                frame_id_gap,
                translation_m,
                min_translation_m,
                tracked_landmarks,
                last_keyframe_tracked_landmarks,
                min_tracked_landmark_ratio,
                tracking_failure_reason,
            ));
        } else {
            let tracking_failure_reason = result
                .tracking
                .tracking_failure_reason
                .as_ref()
                .map(|reason| quote_csv_field(&format!("{reason:?}")))
                .unwrap_or_default();
            keyframe_decision_log.push_str(&format!(
                "{frame_idx},{},{},0,NoMapping,,,{},{:.6},,,,,,,{}\n",
                image_entry.timestamp_nanoseconds,
                if success { 1 } else { 0 },
                result.tracking.localization.inlier_count,
                result.tracking.localization.inlier_ratio,
                tracking_failure_reason,
            ));
        }

        // Static VI initialization promotes its calibrated velocity and IMU
        // biases into the SLAM preintegrator. Mirror the same state into the
        // optional IMU motion model; otherwise `--motion-model imu` would keep
        // integrating with its construction-time zero biases even after the
        // estimator has accepted nonzero calibrated values.
        if let Some(ViInitializationEvent::Succeeded { result, .. }) = result.vi_init.as_ref() {
            slam.tracker.motion_model_mut().mirror_vi_ba_state(
                result.initial_velocity_world,
                result.bias_gyro,
                result.bias_acc,
            );
        }

        // Refresh the IMU motion model's `velocity_world` from the
        // finite-difference of two successive successful poses, but
        // only when `--imu-extrinsic-from-cam0` is on — the flag is
        // the documented atomic opt-in for "use the cam0 T_BS AND
        // maintain velocity from pose history." With the flag off,
        // `velocity_world` stays at zero (Phase-7 behaviour), which
        // empirically gives a tighter rigid ATE on the bench
        // sequences than the finite-difference path until a proper
        // VI-BA-driven velocity update is wired in.
        if args.imu_extrinsic_from_cam0 {
            if let (Some(curr_pose), Some(prev_pose), Some(prev_ts)) = (
                tracked.as_ref(),
                prev_successful_pose.as_ref(),
                prev_successful_ts,
            ) {
                let dt_ns = image_entry.timestamp_nanoseconds - prev_ts;
                if dt_ns > 0 {
                    let dt = dt_ns as f64 * 1.0e-9;
                    slam.tracker
                        .motion_model_mut()
                        .update_velocity_from_pose_diff(prev_pose, curr_pose, dt);
                }
            }
            if let Some(curr_pose) = tracked.as_ref() {
                prev_successful_pose = Some(curr_pose.clone());
                prev_successful_ts = Some(image_entry.timestamp_nanoseconds);
            }
        }

        // Local-VI-BA writeback: when the BA fired this frame and didn't
        // freeze biases (the conditioning fallback), pull the refined
        // per-keyframe `(velocity_world, bias_gyro, bias_acc)` for the
        // window's most recent keyframe and mirror them into the IMU
        // motion model. Pairs with `--imu-extrinsic-from-cam0` — the
        // mirror is a no-op when the motion model isn't IMU-predictive.
        if result.imu_factor.is_some() {
            imu_factors_staged += 1;
        }
        if let Some(stats) = result.local_vi_ba.as_ref() {
            local_vi_ba_triggers += 1;
            local_vi_ba_relinearised_factor_total += stats.relinearised_factor_count;
            local_vi_ba_stereo_observation_total += stats.stereo_observation_count;
            if stats.marginalization_prior_applied {
                local_vi_ba_marginalization_priors_applied += 1;
            }
            if stats.marginalization_succeeded {
                local_vi_ba_marginalization_successes += 1;
            }
            if stats.quality_gate_rejected {
                local_vi_ba_quality_gate_rejections += 1;
            }
            if stats.cost_ratio_gate_rejected {
                local_vi_ba_cost_ratio_gate_rejections += 1;
            }
            if stats.imu_nis_gate_rejected {
                local_vi_ba_imu_nis_gate_rejections += 1;
            }
            if stats.velocity_gate_rejected {
                local_vi_ba_velocity_gate_rejections += 1;
            }
            if stats.pose_correction_gate_rejected {
                local_vi_ba_pose_correction_gate_rejections += 1;
            }
            local_vi_ba_max_pose_translation_correction_meters =
                local_vi_ba_max_pose_translation_correction_meters
                    .max(stats.max_pose_translation_correction_meters);
            local_vi_ba_max_pose_rotation_correction_degrees =
                local_vi_ba_max_pose_rotation_correction_degrees
                    .max(stats.max_pose_rotation_correction_radians.to_degrees());
            if stats.adaptive_velocity_gate_rejected {
                local_vi_ba_adaptive_velocity_gate_rejections += 1;
            }
            local_vi_ba_last_adaptive_velocity_threshold_mps =
                stats.adaptive_velocity_gate_threshold_mps;
            let fmt_optional =
                |value: Option<f64>| value.map(|value| format!("{value:.9}")).unwrap_or_default();
            let window_ids = stats
                .window_keyframe_ids
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(";");
            let initial = stats.initial_cost_breakdown;
            let final_cost = stats.final_cost_breakdown;
            local_vi_ba_diagnostic_rows.push(format!(
                "{frame_idx},{},{window_ids},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{}",
                image_entry.timestamp_nanoseconds,
                stats.observation_count,
                stats.stereo_observation_count,
                stats.imu_factor_count,
                stats.quality_gate_rejected,
                stats.imu_nis_gate_rejected,
                stats.max_pose_translation_correction_meters,
                stats.max_pose_rotation_correction_radians.to_degrees(),
                initial.total,
                initial.visual,
                initial.imu,
                initial.bias_random_walk,
                initial.navigation_prior,
                initial.other_structural,
                fmt_optional(initial.imu_normalized_squared_residual_per_dof),
                fmt_optional(initial.imu_rotation_residual_rms_rad),
                fmt_optional(initial.imu_velocity_residual_rms_mps),
                fmt_optional(initial.imu_position_residual_rms_meters),
                final_cost.total,
                final_cost.visual,
                final_cost.imu,
                final_cost.bias_random_walk,
                final_cost.navigation_prior,
                final_cost.other_structural,
                fmt_optional(final_cost.imu_normalized_squared_residual_per_dof),
                fmt_optional(final_cost.imu_rotation_residual_rms_rad),
                fmt_optional(final_cost.imu_velocity_residual_rms_mps),
                fmt_optional(final_cost.imu_position_residual_rms_meters),
            ));
            if !stats.bias_frozen && !stats.quality_gate_rejected {
                if let (Some(state), Some(latest_kf)) = (
                    slam.local_vi_ba_state.as_ref(),
                    stats.window_keyframe_ids.last().copied(),
                ) {
                    if let Some(per_kf) = state.keyframe_state.get(&latest_kf) {
                        slam.tracker.motion_model_mut().mirror_vi_ba_state(
                            per_kf.velocity_world,
                            per_kf.bias_gyro,
                            per_kf.bias_acc,
                        );
                        local_vi_ba_mirrors += 1;
                        last_mirrored_velocity_world = Some(per_kf.velocity_world);
                        last_mirrored_bias_gyro = Some(per_kf.bias_gyro);
                        last_mirrored_bias_acc = Some(per_kf.bias_acc);
                    }
                }
            }
        }
        if let Some(stats) = result.sparse_factor_graph.as_ref() {
            sparse_factor_graph_updates += 1;
            sparse_factor_graph_added += stats.added;
            sparse_factor_graph_reactivated += stats.reactivated;
            sparse_factor_graph_inactivated_low_confidence += stats.inactivated_low_confidence;
            sparse_factor_graph_inactivated_window_age += stats.inactivated_window_age;
            sparse_factor_graph_inactivated_budget += stats.inactivated_budget;
            sparse_factor_graph_pruned += stats.pruned;
            sparse_factor_graph_active_temporal = stats.active_temporal;
            sparse_factor_graph_active_proximity = stats.active_proximity;
            sparse_factor_graph_active_stereo = stats.active_stereo;
            sparse_factor_graph_inactive = stats.inactive;
        }
        if let Some(stats) = result.covisibility_local_ba.as_ref() {
            covisibility_local_ba_triggers += 1;
            covisibility_local_ba_elapsed_ms_total += stats.elapsed_ms;
            covisibility_local_ba_elapsed_ms_max =
                covisibility_local_ba_elapsed_ms_max.max(stats.elapsed_ms);
            if stats.success {
                covisibility_local_ba_successes += 1;
                covisibility_local_ba_updated_keyframes_total += stats.updated_keyframe_count;
                covisibility_local_ba_updated_landmarks_total += stats.updated_landmark_count;
                covisibility_local_ba_removed_observations_total += stats.removed_observation_count;
                if let (Some(before), Some(after)) = (
                    stats.mean_reprojection_before_px,
                    stats.mean_reprojection_after_px,
                ) {
                    covisibility_local_ba_reprojection_before_sum += before;
                    covisibility_local_ba_reprojection_after_sum += after;
                }
                if stats
                    .selection
                    .as_ref()
                    .map(|selection| selection.boundary_fallback_used)
                    .unwrap_or(false)
                {
                    covisibility_local_ba_boundary_fallback_successes += 1;
                }
            } else {
                covisibility_local_ba_failures += 1;
                match stats.error.as_ref() {
                    Some(CovisibilityLocalBaError::InsufficientActiveObservations {
                        boundary_fallback_used,
                        ..
                    }) => {
                        covisibility_local_ba_active_observation_gate_failures += 1;
                        if *boundary_fallback_used {
                            covisibility_local_ba_boundary_fallback_active_gate_failures += 1;
                        }
                    }
                    Some(CovisibilityLocalBaError::NoLocalLandmarks) => {
                        covisibility_local_ba_no_local_landmarks_failures += 1;
                    }
                    Some(CovisibilityLocalBaError::NoObservations) => {
                        covisibility_local_ba_no_observations_failures += 1;
                    }
                    Some(CovisibilityLocalBaError::QualityGateRejected { .. }) => {
                        covisibility_local_ba_quality_gate_failures += 1;
                    }
                    Some(CovisibilityLocalBaError::InsufficientBoundaryKeyframes { .. }) => {
                        covisibility_local_ba_boundary_support_failures += 1;
                    }
                    Some(CovisibilityLocalBaError::BehindCameraGateRejected { .. }) => {
                        covisibility_local_ba_behind_camera_gate_failures += 1;
                    }
                    Some(CovisibilityLocalBaError::FixedSupportRatioRejected { .. }) => {
                        covisibility_local_ba_fixed_ratio_gate_failures += 1;
                    }
                    Some(CovisibilityLocalBaError::PoseCorrectionGateRejected { .. }) => {
                        covisibility_local_ba_pose_correction_gate_failures += 1;
                    }
                    Some(CovisibilityLocalBaError::Ba(_)) => {
                        covisibility_local_ba_solver_failures += 1;
                    }
                    Some(_) | None => {
                        covisibility_local_ba_other_failures += 1;
                    }
                }
                covisibility_local_ba_last_error = stats.error.as_ref().map(|err| err.to_string());
            }
            let error = stats
                .error
                .as_ref()
                .map(|err| err.to_string())
                .unwrap_or_else(|| "none".to_string());
            let (
                optimized_keyframes,
                fixed_keyframes,
                landmarks,
                observations,
                boundary_fallback_used,
            ) = if let Some(selection) = stats.selection.as_ref() {
                (
                    selection.optimized_keyframe_ids.len(),
                    selection.fixed_keyframe_ids.len(),
                    selection.landmark_ids.len(),
                    selection.observation_count,
                    selection.boundary_fallback_used,
                )
            } else if let Some(CovisibilityLocalBaError::InsufficientBoundaryKeyframes {
                optimized_keyframe_count,
                fixed_keyframe_count,
                boundary_fallback_used,
                ..
            }) = stats.error.as_ref()
            {
                (
                    *optimized_keyframe_count,
                    *fixed_keyframe_count,
                    0,
                    0,
                    *boundary_fallback_used,
                )
            } else {
                (0, 0, 0, 0, false)
            };
            covisibility_local_ba_log.push_str(&format!(
                "{frame_idx},{},{},{},{:.6},{:?},{optimized_keyframes},{fixed_keyframes},{landmarks},{observations},{boundary_fallback_used},{:?},{:?},{:?},{:?},{},{},{},{:?},{},{},{}\n",
                image_entry.timestamp_nanoseconds,
                stats.success,
                error.replace(',', ";"),
                stats.elapsed_ms,
                stats.factor_graph_neighbor_count,
                stats.mean_reprojection_before_px,
                stats.mean_reprojection_after_px,
                stats.max_pose_translation_correction_m,
                stats.max_pose_rotation_correction_rad,
                stats.updated_keyframe_count,
                stats.updated_landmark_count,
                stats.outlier_observation_count,
                stats.outlier_observation_ratio,
                stats.quality_gate_rejected,
                stats.pose_correction_gate_rejected,
                stats.removed_observation_count,
            ));
        }
        if let Some(stats) = result.pose_graph_refinement.as_ref() {
            loop_candidate_diagnostics_csv.push_str(&format!(
                "{frame_idx},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                image_entry.timestamp_nanoseconds,
                stats.verified_candidate_count,
                stats.appearance_ranked_candidate_count,
                stats.appearance_connected_region_rejected_count,
                stats.appearance_pnp_verified_count,
                stats.appearance_projection_rejected_count,
                stats.appearance_covisibility_rejected_count,
                stats.appearance_pending_projection_attempted_count,
                stats.appearance_pending_projection_verified_count,
                format_optional_f64(stats.appearance_pending_projection_search_radius_px),
                stats.appearance_pending_projection_correspondence_count,
                stats.appearance_pending_projection_inlier_count,
                stats.appearance_region_confirmation_waiting,
                stats.appearance_region_confirmed_count,
                stats
                    .admitted_constraints
                    .iter()
                    .filter(|constraint| {
                        constraint.source == LoopClosureCandidateSource::Appearance
                    })
                    .count(),
                stats.loop_closures_pcm_rejected,
                stats.loop_closures_covariance_rejected,
            ));
            for diagnostic in &stats.appearance_candidate_diagnostics {
                loop_candidate_evidence_csv.push_str(&format!(
                    "{},{},{:.9},{},{},{},{},{},{},{},{},{},{},{},{:?},{:?},{}\n",
                    diagnostic.query_frame_id,
                    diagnostic.matched_keyframe_id,
                    diagnostic.appearance_similarity,
                    diagnostic.matched_region_keyframe_count,
                    diagnostic.matched_region_landmark_count,
                    diagnostic.primary_correspondence_count,
                    diagnostic.primary_inlier_count,
                    diagnostic.projection_attempted,
                    diagnostic.projection_correspondence_count,
                    diagnostic.projection_inlier_count,
                    diagnostic.projection_accepted,
                    diagnostic.current_covisible_keyframe_count,
                    diagnostic.neighbor_pnp_verified_count,
                    diagnostic.consistent_keyframe_count,
                    diagnostic.minimum_translation_disagreement_meters,
                    diagnostic.minimum_rotation_disagreement_radians,
                    diagnostic.accepted,
                ));
            }
            pose_graph_refinement_candidates_seen += stats.verified_candidate_count;
            pose_graph_refinement_verified_constraints += stats.accepted_count;
            pose_graph_refinement_appearance_ranked += stats.appearance_ranked_candidate_count;
            pose_graph_refinement_appearance_connected_region_rejected +=
                stats.appearance_connected_region_rejected_count;
            pose_graph_refinement_appearance_pnp_verified += stats.appearance_pnp_verified_count;
            pose_graph_refinement_appearance_projection_rejected +=
                stats.appearance_projection_rejected_count;
            pose_graph_refinement_appearance_covisibility_rejected +=
                stats.appearance_covisibility_rejected_count;
            pose_graph_refinement_appearance_pending_projection_attempted +=
                stats.appearance_pending_projection_attempted_count;
            pose_graph_refinement_appearance_pending_projection_verified +=
                stats.appearance_pending_projection_verified_count;
            pose_graph_refinement_appearance_pending_projection_correspondences +=
                stats.appearance_pending_projection_correspondence_count;
            pose_graph_refinement_appearance_pending_projection_inliers +=
                stats.appearance_pending_projection_inlier_count;
            pose_graph_refinement_appearance_pending_projection_correspondences_max =
                pose_graph_refinement_appearance_pending_projection_correspondences_max
                    .max(stats.appearance_pending_projection_correspondence_count);
            pose_graph_refinement_appearance_pending_projection_inliers_max =
                pose_graph_refinement_appearance_pending_projection_inliers_max
                    .max(stats.appearance_pending_projection_inlier_count);
            if let Some(radius) = stats.appearance_pending_projection_search_radius_px {
                pose_graph_refinement_appearance_pending_projection_radius_min = Some(
                    pose_graph_refinement_appearance_pending_projection_radius_min
                        .map_or(radius, |value| value.min(radius)),
                );
                pose_graph_refinement_appearance_pending_projection_radius_max = Some(
                    pose_graph_refinement_appearance_pending_projection_radius_max
                        .map_or(radius, |value| value.max(radius)),
                );
            }
            pose_graph_refinement_appearance_region_confirmation_waiting +=
                usize::from(stats.appearance_region_confirmation_waiting);
            pose_graph_refinement_appearance_region_confirmed +=
                stats.appearance_region_confirmed_count;
            pose_graph_refinement_appearance_scale_failed +=
                stats.appearance_scale_estimation_failed_count;
            pose_graph_refinement_appearance_scale_insufficient_points +=
                stats.appearance_scale_insufficient_points_count;
            pose_graph_refinement_appearance_scale_insufficient_matches +=
                stats.appearance_scale_insufficient_matches_count;
            pose_graph_refinement_appearance_scale_no_consensus +=
                stats.appearance_scale_no_consensus_count;
            pose_graph_refinement_appearance_near_unit += stats.appearance_near_unit_scale_count;
            if stats.pose_graph_result.is_some()
                || stats.gnc_result.is_some()
                || stats.sim3_pose_graph_result.is_some()
            {
                pose_graph_refinement_pgo_solves += 1;
            }
            pose_graph_refinement_gnc_rejected_loop_edges_max =
                pose_graph_refinement_gnc_rejected_loop_edges_max.max(stats.loop_closures_rejected);
            if let Some(spread) = stats.sim3_scale_spread {
                pose_graph_refinement_last_solve_scale_spread = Some(spread);
            }
            pose_graph_refinement_pcm_rejected += stats.loop_closures_pcm_rejected;
            pose_graph_refinement_covariance_rejected += stats.loop_closures_covariance_rejected;
            pose_graph_refinement_pose_information_rejected +=
                stats.loop_closures_pose_information_rejected;
            pose_graph_refinement_with_pose_information +=
                stats.loop_closures_with_pose_information;
            for diagnostic in &stats.loop_pose_information_diagnostics {
                pose_graph_refinement_pose_information_estimates += 1;
                pose_graph_refinement_pose_information_used_correspondences +=
                    diagnostic.used_correspondence_count;
                pose_graph_refinement_pose_information_stereo_covariance_correspondences +=
                    diagnostic.stereo_covariance_correspondence_count;
                pose_graph_refinement_pose_information_raw_condition_max =
                    pose_graph_refinement_pose_information_raw_condition_max
                        .max(diagnostic.raw_condition_number);
                pose_graph_refinement_pose_information_spectral_scale_min = Some(
                    pose_graph_refinement_pose_information_spectral_scale_min
                        .map_or(diagnostic.applied_spectral_scale, |value| {
                            value.min(diagnostic.applied_spectral_scale)
                        }),
                );
            }
            pose_graph_refinement_pose_information_failures
                .merge(stats.loop_pose_information_failures);
            pose_graph_refinement_sequential_with_pose_information +=
                stats.sequential_edges_with_pose_information;
            pose_graph_refinement_sequential_pose_information_fallbacks +=
                stats.sequential_pose_information_fallbacks;
            for diagnostic in &stats.sequential_pose_information_diagnostics {
                pose_graph_refinement_sequential_pose_information_used_correspondences +=
                    diagnostic.used_correspondence_count;
                pose_graph_refinement_sequential_pose_information_stereo_covariance_correspondences +=
                    diagnostic.stereo_covariance_correspondence_count;
                pose_graph_refinement_sequential_pose_information_raw_condition_max =
                    pose_graph_refinement_sequential_pose_information_raw_condition_max
                        .max(diagnostic.raw_condition_number);
                pose_graph_refinement_sequential_pose_information_spectral_scale_min = Some(
                    pose_graph_refinement_sequential_pose_information_spectral_scale_min
                        .map_or(diagnostic.applied_spectral_scale, |value| {
                            value.min(diagnostic.applied_spectral_scale)
                        }),
                );
            }
            pose_graph_refinement_sequential_pose_information_failures
                .merge(stats.sequential_pose_information_failures);
            pose_graph_refinement_loop_fusion_pairs_considered +=
                stats.loop_fusion_pairs_considered;
            pose_graph_refinement_loop_fusion_observations_inserted +=
                stats.loop_fusion_observations_inserted;
            pose_graph_refinement_loop_fusion_observations_reassigned +=
                stats.loop_fusion_observations_reassigned;
            pose_graph_refinement_loop_fusion_pairs_skipped += stats.loop_fusion_pairs_skipped;
            pose_graph_refinement_loop_fusion_pairs_robust_rejected +=
                stats.loop_fusion_pairs_robust_rejected;
            pose_graph_refinement_loop_fusion_pairs_reprojection_rejected +=
                stats.loop_fusion_pairs_reprojection_rejected;
            pose_graph_refinement_loop_welding_ba_attempted +=
                usize::from(stats.loop_welding_ba_attempted);
            pose_graph_refinement_loop_welding_ba_succeeded +=
                usize::from(stats.loop_welding_ba_succeeded);
            pose_graph_refinement_loop_welding_ba_rejected_or_failed +=
                usize::from(stats.loop_welding_ba_rejected_or_failed);
            pose_graph_refinement_loop_welding_ba_updated_keyframes +=
                stats.loop_welding_ba_updated_keyframes;
            pose_graph_refinement_loop_welding_ba_updated_landmarks +=
                stats.loop_welding_ba_updated_landmarks;
            pose_graph_refinement_loop_welding_post_pgo_attempted +=
                usize::from(stats.loop_welding_post_pgo_attempted);
            pose_graph_refinement_loop_welding_post_pgo_succeeded +=
                usize::from(stats.loop_welding_post_pgo_succeeded);
            if let Some(value) = stats.loop_welding_post_pgo_mean_reprojection_px {
                pose_graph_refinement_loop_welding_post_pgo_reprojection_max = Some(
                    pose_graph_refinement_loop_welding_post_pgo_reprojection_max
                        .map_or(value, |current| current.max(value)),
                );
            }
            if let Some(error) = &stats.loop_welding_post_ba_error {
                pose_graph_refinement_loop_welding_post_ba_last_error = Some(format!("{error:?}"));
            }
            if let Some(value) = stats.loop_welding_post_ba_behind_camera_ratio {
                pose_graph_refinement_loop_welding_post_ba_behind_camera_ratio_max = Some(
                    pose_graph_refinement_loop_welding_post_ba_behind_camera_ratio_max
                        .map_or(value, |current| current.max(value)),
                );
            }
            if let Some(value) = stats.loop_welding_initial_translation_meters {
                pose_graph_refinement_loop_welding_initial_translation_max = Some(
                    pose_graph_refinement_loop_welding_initial_translation_max
                        .map_or(value, |current| current.max(value)),
                );
            }
            if let Some(value) = stats.loop_welding_initial_rotation_radians {
                pose_graph_refinement_loop_welding_initial_rotation_max = Some(
                    pose_graph_refinement_loop_welding_initial_rotation_max
                        .map_or(value, |current| current.max(value)),
                );
            }
            pose_graph_refinement_landmarks_moved += stats.landmarks_moved;
            if let Some(max_displacement) = stats.max_landmark_displacement_meters {
                if max_displacement > pose_graph_refinement_max_landmark_displacement_meters {
                    pose_graph_refinement_max_landmark_displacement_meters = max_displacement;
                }
            }
            if stats.tracker_correction_applied {
                pose_graph_refinement_tracker_corrections_applied += 1;
            }
            for admitted in &stats.admitted_constraints {
                let source = match admitted.source {
                    LoopClosureCandidateSource::SharedLandmark => {
                        pose_graph_refinement_constraints_shared += 1;
                        "shared_landmark"
                    }
                    LoopClosureCandidateSource::Appearance => {
                        pose_graph_refinement_constraints_appearance += 1;
                        "appearance"
                    }
                };
                let keyframe_id_gap = admitted
                    .to_keyframe_id
                    .saturating_sub(admitted.from_keyframe_id);
                let gt_error = dataset
                    .cam0_images
                    .get(admitted.from_keyframe_id as usize)
                    .zip(dataset.cam0_images.get(admitted.to_keyframe_id as usize))
                    .and_then(|(from_image, to_image)| {
                        let from_gt = nearest_ground_truth(
                            &dataset.ground_truth,
                            from_image.timestamp_nanoseconds,
                        )?;
                        let to_gt = nearest_ground_truth(
                            &dataset.ground_truth,
                            to_image.timestamp_nanoseconds,
                        )?;
                        let camera_to_world = |gt: &EurocGroundTruthSample| {
                            let center = gt.position_world
                                + gt.orientation_world
                                    .transform_vector(&body_to_camera.translation);
                            visloc_rs::SE3::new(
                                gt.orientation_world * body_to_camera.rotation,
                                center,
                            )
                        };
                        let from_camera_to_world = camera_to_world(from_gt);
                        let to_world_to_camera = camera_to_world(to_gt).inverse();
                        let gt_relative = to_world_to_camera.compose(&from_camera_to_world);
                        let residual = admitted.relative_pose.compose(&gt_relative.inverse());
                        let translation_error = residual.translation.norm();
                        let rotation_error_deg = residual.rotation.angle().to_degrees();
                        let correct = translation_error <= 0.5 && rotation_error_deg <= 10.0;
                        Some((translation_error, rotation_error_deg, correct))
                    });
                if let Some((_, _, correct)) = gt_error {
                    pose_graph_refinement_gt_evaluated += 1;
                    pose_graph_refinement_gt_correct += usize::from(correct);
                }
                let relative_q = admitted.relative_pose.rotation.quaternion();
                let (gt_translation_error, gt_rotation_error, gt_correct) = gt_error
                    .map(|(translation, rotation, correct)| {
                        (
                            translation.to_string(),
                            rotation.to_string(),
                            correct.to_string(),
                        )
                    })
                    .unwrap_or_else(|| (String::new(), String::new(), String::new()));
                loop_constraints_csv.push_str(&format!(
                    "{frame_idx},{},{},{},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{}\n",
                    admitted.from_keyframe_id,
                    admitted.to_keyframe_id,
                    keyframe_id_gap,
                    admitted.translation_norm_m,
                    admitted.relative_pose.translation.x,
                    admitted.relative_pose.translation.y,
                    admitted.relative_pose.translation.z,
                    relative_q.w,
                    relative_q.i,
                    relative_q.j,
                    relative_q.k,
                    admitted.inlier_count,
                    source,
                    gt_translation_error,
                    gt_rotation_error,
                    gt_correct,
                ));
            }
            for rejected in &stats.rejected_constraints {
                let source = match rejected.source {
                    LoopClosureCandidateSource::SharedLandmark => "shared_landmark",
                    LoopClosureCandidateSource::Appearance => "appearance",
                };
                let keyframe_id_gap = rejected
                    .to_keyframe_id
                    .saturating_sub(rejected.from_keyframe_id);
                let gt_error = dataset
                    .cam0_images
                    .get(rejected.from_keyframe_id as usize)
                    .zip(dataset.cam0_images.get(rejected.to_keyframe_id as usize))
                    .and_then(|(from_image, to_image)| {
                        let from_gt = nearest_ground_truth(
                            &dataset.ground_truth,
                            from_image.timestamp_nanoseconds,
                        )?;
                        let to_gt = nearest_ground_truth(
                            &dataset.ground_truth,
                            to_image.timestamp_nanoseconds,
                        )?;
                        let camera_to_world = |gt: &EurocGroundTruthSample| {
                            let center = gt.position_world
                                + gt.orientation_world
                                    .transform_vector(&body_to_camera.translation);
                            visloc_rs::SE3::new(
                                gt.orientation_world * body_to_camera.rotation,
                                center,
                            )
                        };
                        let from_camera_to_world = camera_to_world(from_gt);
                        let to_world_to_camera = camera_to_world(to_gt).inverse();
                        let gt_relative = to_world_to_camera.compose(&from_camera_to_world);
                        let residual = rejected.relative_pose.compose(&gt_relative.inverse());
                        let translation_error = residual.translation.norm();
                        let rotation_error_deg = residual.rotation.angle().to_degrees();
                        let correct = translation_error <= 0.5 && rotation_error_deg <= 10.0;
                        Some((translation_error, rotation_error_deg, correct))
                    });
                if let Some((_, _, correct)) = gt_error {
                    pose_graph_refinement_rejected_gt_evaluated += 1;
                    pose_graph_refinement_rejected_gt_correct += usize::from(correct);
                }
                let relative_q = rejected.relative_pose.rotation.quaternion();
                let (gt_translation_error, gt_rotation_error, gt_correct) = gt_error
                    .map(|(translation, rotation, correct)| {
                        (
                            translation.to_string(),
                            rotation.to_string(),
                            correct.to_string(),
                        )
                    })
                    .unwrap_or_else(|| (String::new(), String::new(), String::new()));
                loop_rejected_constraints_csv.push_str(&format!(
                    "{frame_idx},{},{},{},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{:?},{},{},{}\n",
                    rejected.from_keyframe_id,
                    rejected.to_keyframe_id,
                    keyframe_id_gap,
                    rejected.translation_norm_m,
                    rejected.relative_pose.translation.x,
                    rejected.relative_pose.translation.y,
                    rejected.relative_pose.translation.z,
                    relative_q.w,
                    relative_q.i,
                    relative_q.j,
                    relative_q.k,
                    rejected.inlier_count,
                    source,
                    rejected.reason,
                    gt_translation_error,
                    gt_rotation_error,
                    gt_correct,
                ));
            }
        }
        if let Some(size) = result.tracking.covisibility_local_map_size {
            covisibility_local_map_frames += 1;
            covisibility_local_map_size_sum += size;
        }

        if let Some(event) = &result.vi_init {
            let entry = format!(
                "frame_idx={frame_idx} timestamp_ns={} {}\n",
                image_entry.timestamp_nanoseconds,
                format_vi_init_event(event),
            );
            print!("vi_init {entry}");
            vi_init_log.push_str(&entry);
            if vi_init_first_event_at_frame.is_none() {
                vi_init_first_event_at_frame = Some(frame_idx);
            }
            if matches!(event, ViInitializationEvent::Succeeded { .. })
                && vi_init_succeeded_at_frame.is_none()
            {
                vi_init_succeeded_at_frame = Some(frame_idx);
            }
        }

        if let Some(event) = &result.vi_motion_init {
            let (last_gravity_alignment, last_gyro_bias_alignment) =
                match slam.motion_vi_initialization_status() {
                    MotionViInitializationStatus::Waiting {
                        last_gravity_alignment,
                        last_gyro_bias_alignment,
                        ..
                    } => (last_gravity_alignment, last_gyro_bias_alignment),
                    _ => (None, None),
                };
            let entry = format!(
                "frame_idx={frame_idx} timestamp_ns={} {}\n",
                image_entry.timestamp_nanoseconds,
                format_motion_vi_init_event(
                    event,
                    last_gravity_alignment.as_ref(),
                    last_gyro_bias_alignment.as_ref(),
                ),
            );
            print!("vi_motion_init {entry}");
            motion_vi_init_log.push_str(&entry);
            if motion_vi_init_first_event_at_frame.is_none() {
                motion_vi_init_first_event_at_frame = Some(frame_idx);
            }
            if let MotionViInitializationEvent::Succeeded { result } = event {
                if motion_vi_init_succeeded_at_frame.is_none() {
                    motion_vi_init_succeeded_at_frame = Some(frame_idx);
                    motion_vi_init_recovered_scale = Some(result.scale);
                    motion_vi_init_viba2_iterations = Some(result.viba2_iterations_run);
                    motion_vi_init_estimated_gravity = result.estimated_gravity_world;
                    motion_vi_init_estimated_gyro_bias = result.estimated_gyro_bias;
                }
            }
        }

        let atlas_from_local = map_atlas.as_ref().and_then(|atlas| {
            atlas
                .submap(active_submap_id)
                .and_then(|submap| submap.atlas_from_local.clone())
        });
        let globally_tracked = if map_atlas.is_some() {
            tracked.as_ref().and_then(|pose| {
                atlas_from_local.as_ref().map(|atlas_from_local| Pose {
                    world_to_camera: pose.world_to_camera.compose(&atlas_from_local.inverse()),
                })
            })
        } else {
            tracked.clone()
        };
        let global_success = success && globally_tracked.is_some();
        if global_success {
            globally_aligned_tracking_successes += 1;
        }

        if let Some(local_pose) = tracked.as_ref() {
            let local_center = local_pose.camera_center_world();
            let local_rotation_wc = local_pose.world_to_camera.rotation.inverse();
            submap_trajectory_csv.push_str(&format!(
                "{},{frame_idx},{active_submap_id},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
                image_entry.timestamp_nanoseconds,
                local_center.x,
                local_center.y,
                local_center.z,
                local_rotation_wc.w,
                local_rotation_wc.i,
                local_rotation_wc.j,
                local_rotation_wc.k,
                u8::from(success),
                u8::from(atlas_from_local.is_some() || map_atlas.is_none()),
            ));
        } else {
            submap_trajectory_csv.push_str(&format!(
                "{},{frame_idx},{active_submap_id},,,,,,,,{},{}\n",
                image_entry.timestamp_nanoseconds,
                u8::from(success),
                u8::from(atlas_from_local.is_some() || map_atlas.is_none()),
            ));
        }

        let (estimated_center, estimated_rotation_wc) =
            if let Some(pose) = globally_tracked.as_ref() {
                let center = pose.camera_center_world();
                let rotation_wc = pose.world_to_camera.rotation.inverse();
                (Some(center), Some(rotation_wc))
            } else {
                (None, None)
            };

        let gt = nearest_ground_truth(&dataset.ground_truth, image_entry.timestamp_nanoseconds);
        let gt_camera_pose = gt.map(|sample| {
            let center = sample.position_world
                + sample
                    .orientation_world
                    .transform_vector(&body_to_camera.translation);
            let rotation = sample.orientation_world * body_to_camera.rotation;
            (center, rotation)
        });
        if let Some((gt_center, _)) = gt_camera_pose.as_ref() {
            frame_groundtruth_csv.push_str(&format!(
                "{},{frame_idx},{:.6},{:.6},{:.6}\n",
                image_entry.timestamp_nanoseconds, gt_center.x, gt_center.y, gt_center.z,
            ));
        }

        if let (Some(center), Some(rot_wc)) = (estimated_center, estimated_rotation_wc) {
            let q = rot_wc;
            traj_csv.push_str(&format!(
                "{},{frame_idx},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                image_entry.timestamp_nanoseconds,
                center.x,
                center.y,
                center.z,
                q.w,
                q.i,
                q.j,
                q.k,
                if global_success { 1 } else { 0 },
            ));

            if let Some((gt_center, gt_rotation_wc)) = gt_camera_pose.as_ref() {
                let position_error =
                    (Vector3::new(center.x, center.y, center.z) - gt_center).norm();
                let orientation_error_deg = q.rotation_to(gt_rotation_wc).angle().to_degrees();
                err_csv.push_str(&format!(
                    "{},{frame_idx},{segment_id},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                    image_entry.timestamp_nanoseconds,
                    gt_center.x,
                    gt_center.y,
                    gt_center.z,
                    center.x,
                    center.y,
                    center.z,
                    position_error,
                    orientation_error_deg,
                ));
                sum_position_sq += position_error * position_error;
                sum_orientation_sq_deg += orientation_error_deg * orientation_error_deg;
                if position_error > max_position_err {
                    max_position_err = position_error;
                }
                if orientation_error_deg > max_orientation_err_deg {
                    max_orientation_err_deg = orientation_error_deg;
                }
                estimated_positions.push(Point3::new(center.x, center.y, center.z));
                reference_positions.push(Point3::from(*gt_center));
                error_samples += 1;
            }
        } else {
            traj_csv.push_str(&format!(
                "{},{frame_idx},,,,,,,,{}\n",
                image_entry.timestamp_nanoseconds,
                if global_success { 1 } else { 0 },
            ));
        }

        frame_processing_times_ms.push(frame_processing_started_at.elapsed().as_secs_f64() * 1e3);
        frames_recorded += 1;
    }

    if let Some(atlas) = map_atlas.as_mut() {
        atlas.replace_submap_map(active_submap_id, slam.map().clone())?;
    }
    let materialized_atlas = map_atlas
        .as_ref()
        .map(MapAtlas::materialize_aligned)
        .transpose()?;
    let final_output_map = materialized_atlas
        .as_ref()
        .map(|materialized| &materialized.map)
        .unwrap_or_else(|| slam.map());
    let materialized_frame_source_ids: std::collections::HashMap<u64, u64> = materialized_atlas
        .as_ref()
        .map(|materialized| {
            materialized
                .id_remaps
                .values()
                .flat_map(|remap| {
                    remap
                        .frame_ids
                        .iter()
                        .map(|(source_id, output_id)| (*output_id, *source_id))
                })
                .collect()
        })
        .unwrap_or_default();

    fs::write(&traj_path, traj_csv)?;
    if map_atlas.is_some() {
        fs::write(&submap_trajectory_path, submap_trajectory_csv)?;
        fs::write(&atlas_merge_attempts_path, atlas_merge_attempts_csv)?;
        fs::write(&atlas_boundary_factors_path, atlas_boundary_factors_csv)?;
    }
    if let Some(atlas) = map_atlas.as_ref() {
        let mut submaps_csv = String::from(
            "submap_id,aligned,atlas_tx,atlas_ty,atlas_tz,atlas_qw,atlas_qx,atlas_qy,atlas_qz,cameras,keyframes,landmarks,stereo_observations,observation_confidences\n",
        );
        let mut submap_landmarks_csv = String::from(
            "submap_id,landmark_id,local_x,local_y,local_z,atlas_x,atlas_y,atlas_z,observations,aligned\n",
        );
        for submap in atlas.active_submaps() {
            let transform = submap.atlas_from_local.as_ref();
            let transform_fields = transform
                .map(|transform| {
                    format!(
                        "{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                        transform.rotation.w,
                        transform.rotation.i,
                        transform.rotation.j,
                        transform.rotation.k,
                    )
                })
                .unwrap_or_else(|| ",,,,,,".to_string());
            submaps_csv.push_str(&format!(
                "{},{},{transform_fields},{},{},{},{},{}\n",
                submap.id,
                u8::from(transform.is_some()),
                submap.map.cameras.len(),
                submap.map.keyframes.len(),
                submap.map.landmarks.len(),
                submap.map.stereo_observations.len(),
                submap.map.observation_confidence_count(),
            ));
            for landmark in submap.map.landmarks.values() {
                let atlas_point =
                    transform.map(|transform| transform.transform_point(&landmark.position));
                let atlas_fields = atlas_point
                    .map(|point| format!("{:.9},{:.9},{:.9}", point.x, point.y, point.z))
                    .unwrap_or_else(|| ",,".to_string());
                submap_landmarks_csv.push_str(&format!(
                    "{},{},{:.9},{:.9},{:.9},{atlas_fields},{},{}\n",
                    submap.id,
                    landmark.id,
                    landmark.position.x,
                    landmark.position.y,
                    landmark.position.z,
                    landmark.observations.len(),
                    u8::from(transform.is_some()),
                ));
            }
        }
        fs::write(&atlas_submaps_path, submaps_csv)?;
        fs::write(&atlas_submap_landmarks_path, submap_landmarks_csv)?;
    }
    fs::write(&err_path, err_csv)?;
    fs::write(&frame_groundtruth_path, frame_groundtruth_csv)?;
    if args.export_frame_appearance_descriptors {
        let mut csv = String::from("timestamp_ns,frame_idx,descriptor_count");
        if let Some(dim) = frame_appearance_descriptor_dim {
            for i in 0..dim {
                csv.push_str(&format!(",d{i}"));
            }
        }
        csv.push('\n');
        for row in &frame_appearance_descriptor_rows {
            csv.push_str(row);
        }
        fs::write(&frame_appearance_descriptors_path, csv)?;
    }
    fs::write(&vi_init_log_path, &vi_init_log)?;
    fs::write(&motion_vi_init_log_path, &motion_vi_init_log)?;
    fs::write(&covisibility_local_ba_log_path, &covisibility_local_ba_log)?;
    fs::write(
        &local_vi_ba_diagnostics_path,
        local_vi_ba_diagnostic_rows.join("\n") + "\n",
    )?;
    fs::write(&keyframe_decision_log_path, &keyframe_decision_log)?;
    fs::write(&tracking_diagnostics_path, &tracking_diagnostics_csv)?;
    fs::write(
        &relocalization_appearance_candidates_path,
        &relocalization_appearance_candidates_csv,
    )?;
    fs::write(&relocalization_attempts_path, &relocalization_attempts_csv)?;
    if rebootstrap_enabled {
        fs::write(&rebootstrap_log_path, &rebootstrap_log_csv)?;
    }
    if args.pose_graph_refinement_enabled {
        fs::write(&loop_constraints_path, &loop_constraints_csv)?;
        fs::write(
            &loop_rejected_constraints_path,
            &loop_rejected_constraints_csv,
        )?;
        fs::write(
            &loop_candidate_diagnostics_path,
            &loop_candidate_diagnostics_csv,
        )?;
        fs::write(&loop_candidate_evidence_path, &loop_candidate_evidence_csv)?;
    }

    // Dump the metric landmark cloud (world frame) so downstream tools can seed
    // a 3D Gaussian Splat from real on-surface SLAM points instead of random
    // volumetric init (which gives gsplat a huge kNN init-scale -> fog).
    let mut lm_csv = String::from("id,x,y,z,observations\n");
    for lm in final_output_map.landmarks.values() {
        lm_csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{}\n",
            lm.id,
            lm.position.x,
            lm.position.y,
            lm.position.z,
            lm.observations.len()
        ));
    }
    fs::write(args.out_dir.join("slam_landmarks.csv"), lm_csv)?;

    let (rmse_pos, rmse_rot_deg) = if error_samples > 0 {
        (
            (sum_position_sq / error_samples as f64).sqrt(),
            (sum_orientation_sq_deg / error_samples as f64).sqrt(),
        )
    } else {
        (0.0, 0.0)
    };

    let aligned_rigid =
        umeyama_similarity_transform(&estimated_positions, &reference_positions, false)
            .unwrap_or_else(TrajectorySimilarityTransform::identity);
    let aligned_similarity =
        umeyama_similarity_transform(&estimated_positions, &reference_positions, true)
            .unwrap_or_else(TrajectorySimilarityTransform::identity);

    let mut rmse_sq_rigid = 0.0_f64;
    let mut max_rigid = 0.0_f64;
    let mut rmse_sq_sim = 0.0_f64;
    let mut max_sim = 0.0_f64;
    for (est, gt) in estimated_positions.iter().zip(reference_positions.iter()) {
        let rigid_err = (aligned_rigid.apply(est) - gt).norm();
        let sim_err = (aligned_similarity.apply(est) - gt).norm();
        rmse_sq_rigid += rigid_err * rigid_err;
        rmse_sq_sim += sim_err * sim_err;
        if rigid_err > max_rigid {
            max_rigid = rigid_err;
        }
        if sim_err > max_sim {
            max_sim = sim_err;
        }
    }
    let (ate_rmse_rigid, ate_rmse_sim) = if !estimated_positions.is_empty() {
        let n = estimated_positions.len() as f64;
        ((rmse_sq_rigid / n).sqrt(), (rmse_sq_sim / n).sqrt())
    } else {
        (0.0, 0.0)
    };

    // Final-trajectory view: `slam_trajectory.csv`/`slam_errors.csv` above are
    // causal, per-frame *live* estimates logged as each frame was processed,
    // so a loop-closure PGO correction applied to `map.keyframes` later in
    // the run (`--pose-graph-refinement-propagate`) can never retroactively
    // fix an already-logged row. `final_output_map.keyframes` reflects every
    // correction folded in by the time the frame loop finished, so re-running
    // the same per-frame-error machinery over the final keyframe poses gives
    // the "final optimized trajectory" ATE that published SLAM numbers
    // (ORB-SLAM3 etc.) are computed on. Keyframe ids are assigned as
    // `frame_idx as u64` when each `Frame` is built for `slam.process_frame`
    // (see `frame_from_features` call sites above), so `dataset.cam0_images`
    // is indexable directly by keyframe id to recover the frame timestamp —
    // the same association `keyframe_decisions.csv` relies on implicitly via
    // `frame_idx`.
    let mut keyframe_ids: Vec<u64> = final_output_map.keyframes.keys().copied().collect();
    keyframe_ids.sort_unstable();

    let mut keyframe_trajectory_csv =
        String::from("keyframe_id,timestamp_ns,px,py,pz,qw,qx,qy,qz\n");
    let mut final_keyframe_estimated_positions: Vec<Point3<f64>> = Vec::new();
    let mut final_keyframe_reference_positions: Vec<Point3<f64>> = Vec::new();
    let mut final_keyframe_timestamps: Vec<i128> = Vec::new();
    let mut final_keyframe_ids: Vec<u64> = Vec::new();
    let mut final_keyframe_estimated_tum = String::new();
    let mut final_keyframe_reference_tum = String::new();
    let mut final_keyframe_trajectory_count = 0usize;

    for &keyframe_id in &keyframe_ids {
        let keyframe = &final_output_map.keyframes[&keyframe_id];
        let Some(pose) = keyframe.frame.pose.as_ref() else {
            continue;
        };
        let source_frame_id = materialized_frame_source_ids
            .get(&keyframe_id)
            .copied()
            .unwrap_or(keyframe_id);
        let Some(image_entry) = dataset.cam0_images.get(source_frame_id as usize) else {
            continue;
        };
        let timestamp_ns = image_entry.timestamp_nanoseconds;
        let center = pose.camera_center_world();
        let q = pose.world_to_camera.rotation.inverse();
        keyframe_trajectory_csv.push_str(&format!(
            "{keyframe_id},{timestamp_ns},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            center.x, center.y, center.z, q.w, q.i, q.j, q.k,
        ));
        final_keyframe_trajectory_count += 1;

        if let Some(gt) = nearest_ground_truth(&dataset.ground_truth, timestamp_ns) {
            let gt_center = gt.position_world
                + gt.orientation_world
                    .transform_vector(&body_to_camera.translation);
            let gt_camera_to_world_rotation = gt.orientation_world * body_to_camera.rotation;
            let gt_q = gt_camera_to_world_rotation.quaternion();
            final_keyframe_estimated_tum.push_str(&format!(
                "{keyframe_id} {} {} {} {} {} {} {}\n",
                center.x, center.y, center.z, q.i, q.j, q.k, q.w,
            ));
            final_keyframe_reference_tum.push_str(&format!(
                "{keyframe_id} {} {} {} {} {} {} {}\n",
                gt_center.x, gt_center.y, gt_center.z, gt_q.i, gt_q.j, gt_q.k, gt_q.w,
            ));

            final_keyframe_estimated_positions.push(Point3::new(center.x, center.y, center.z));
            final_keyframe_reference_positions.push(Point3::new(
                gt_center.x,
                gt_center.y,
                gt_center.z,
            ));
            final_keyframe_timestamps.push(timestamp_ns);
            final_keyframe_ids.push(keyframe_id);
        }
    }
    fs::write(&keyframe_trajectory_path, keyframe_trajectory_csv)?;

    let final_keyframe_count = final_keyframe_estimated_positions.len();
    let final_keyframe_aligned_rigid = umeyama_similarity_transform(
        &final_keyframe_estimated_positions,
        &final_keyframe_reference_positions,
        false,
    )
    .unwrap_or_else(TrajectorySimilarityTransform::identity);
    let final_keyframe_aligned_similarity = umeyama_similarity_transform(
        &final_keyframe_estimated_positions,
        &final_keyframe_reference_positions,
        true,
    )
    .unwrap_or_else(TrajectorySimilarityTransform::identity);

    let mut final_keyframe_errors_csv = String::from("keyframe_id,timestamp_ns,position_error_m\n");
    let mut final_keyframe_rmse_sq_rigid = 0.0_f64;
    let mut final_keyframe_rmse_sq_sim = 0.0_f64;
    for (((id, ts), est), gt) in final_keyframe_ids
        .iter()
        .zip(final_keyframe_timestamps.iter())
        .zip(final_keyframe_estimated_positions.iter())
        .zip(final_keyframe_reference_positions.iter())
    {
        let rigid_err = (final_keyframe_aligned_rigid.apply(est) - gt).norm();
        let sim_err = (final_keyframe_aligned_similarity.apply(est) - gt).norm();
        final_keyframe_rmse_sq_rigid += rigid_err * rigid_err;
        final_keyframe_rmse_sq_sim += sim_err * sim_err;
        final_keyframe_errors_csv.push_str(&format!("{id},{ts},{rigid_err:.6}\n"));
    }
    fs::write(&final_keyframe_errors_path, final_keyframe_errors_csv)?;

    let (final_keyframe_ate_rigid_rmse_m, final_keyframe_ate_similarity_rmse_m) =
        if final_keyframe_count > 0 {
            let n = final_keyframe_count as f64;
            (
                (final_keyframe_rmse_sq_rigid / n).sqrt(),
                (final_keyframe_rmse_sq_sim / n).sqrt(),
            )
        } else {
            (0.0, 0.0)
        };
    let final_keyframe_ate_similarity_scale = final_keyframe_aligned_similarity.scale;
    let final_keyframe_estimated_trajectory =
        PoseTrajectory::from_tum_poses_str(&final_keyframe_estimated_tum)?;
    let final_keyframe_reference_trajectory =
        PoseTrajectory::from_tum_poses_str(&final_keyframe_reference_tum)?;
    let final_keyframe_rpe_delta1 = final_keyframe_estimated_trajectory
        .relative_pose_error_against(
            &final_keyframe_reference_trajectory,
            &RelativePoseErrorConfig {
                delta: 1,
                start_step: 1,
            },
        );
    let final_keyframe_rpe_delta10 = final_keyframe_estimated_trajectory
        .relative_pose_error_against(
            &final_keyframe_reference_trajectory,
            &RelativePoseErrorConfig {
                delta: 10,
                start_step: 1,
            },
        );
    let final_keyframe_rpe_delta1_translation_rmse_m = final_keyframe_rpe_delta1
        .translation
        .map(|stats| stats.rmse);
    let final_keyframe_rpe_delta1_rotation_rmse_deg = final_keyframe_rpe_delta1
        .rotation_deg
        .map(|stats| stats.rmse);
    let final_keyframe_rpe_delta10_translation_rmse_m = final_keyframe_rpe_delta10
        .translation
        .map(|stats| stats.rmse);
    let final_keyframe_rpe_delta10_rotation_rmse_deg = final_keyframe_rpe_delta10
        .rotation_deg
        .map(|stats| stats.rmse);

    let final_vi_status = slam.vi_initialization_status();
    let final_motion_vi_status = slam.motion_vi_initialization_status();
    let map_keyframes = final_output_map.keyframes.len();
    let map_landmarks = final_output_map.landmarks.len();
    let map_landmark_covariances = final_output_map.landmark_position_covariances.len();
    let map_stereo_observations = final_output_map.stereo_observations.len();
    let map_observation_confidences = final_output_map.observation_confidence_count();
    let map_observation_confidence_stats = final_output_map.observation_confidence_stats();
    let atlas_submap_count = map_atlas
        .as_ref()
        .map(|atlas| atlas.active_submaps().count())
        .unwrap_or(1);
    let atlas_aligned_submap_count = map_atlas
        .as_ref()
        .map(MapAtlas::aligned_submap_count)
        .unwrap_or(1);
    let atlas_independent_submap_count = map_atlas
        .as_ref()
        .map(MapAtlas::independent_submap_count)
        .unwrap_or(0);
    let atlas_verified_merge_count = map_atlas
        .as_ref()
        .map(|atlas| atlas.verified_merges().len())
        .unwrap_or(0);
    let atlas_verified_landmark_match_count = map_atlas
        .as_ref()
        .map(|atlas| {
            atlas
                .verified_merges()
                .iter()
                .map(|merge| merge.landmark_matches.len())
                .sum::<usize>()
        })
        .unwrap_or(0);
    let atlas_aligned_owned_landmark_count = map_atlas
        .as_ref()
        .map(|atlas| {
            atlas
                .active_submaps()
                .filter(|submap| submap.atlas_from_local.is_some())
                .map(|submap| submap.map.landmarks.len())
                .sum::<usize>()
        })
        .unwrap_or(map_landmarks);
    let atlas_welded_landmark_reduction =
        atlas_aligned_owned_landmark_count.saturating_sub(map_landmarks);
    let global_tracking_success_rate = if frames_recorded > 0 {
        globally_aligned_tracking_successes as f64 / frames_recorded as f64
    } else {
        0.0
    };
    let mean_features = if frames_recorded > 0 {
        feature_count_sum as f64 / frames_recorded as f64
    } else {
        0.0
    };
    let feature_count_min = if feature_count_min == usize::MAX {
        0
    } else {
        feature_count_min
    };
    let covisibility_local_ba_mean_reprojection_before_px = if covisibility_local_ba_successes > 0 {
        Some(covisibility_local_ba_reprojection_before_sum / covisibility_local_ba_successes as f64)
    } else {
        None
    };
    let covisibility_local_ba_mean_reprojection_after_px = if covisibility_local_ba_successes > 0 {
        Some(covisibility_local_ba_reprojection_after_sum / covisibility_local_ba_successes as f64)
    } else {
        None
    };
    let covisibility_local_ba_elapsed_ms_mean = if covisibility_local_ba_triggers > 0 {
        Some(covisibility_local_ba_elapsed_ms_total / covisibility_local_ba_triggers as f64)
    } else {
        None
    };

    let wall_clock_seconds = demo_started_at.elapsed().as_secs_f64();
    let wall_clock_ms_per_frame = if frames_recorded > 0 {
        wall_clock_seconds * 1_000.0 / frames_recorded as f64
    } else {
        0.0
    };
    let wall_clock_fps = if wall_clock_seconds > 0.0 {
        frames_recorded as f64 / wall_clock_seconds
    } else {
        0.0
    };
    frame_processing_times_ms.sort_by(f64::total_cmp);
    let frame_time_percentile_ms = |quantile: f64| {
        if frame_processing_times_ms.is_empty() {
            return None;
        }
        let index = ((frame_processing_times_ms.len() - 1) as f64 * quantile)
            .round()
            .clamp(0.0, (frame_processing_times_ms.len() - 1) as f64) as usize;
        frame_processing_times_ms.get(index).copied()
    };
    let frame_processing_ms_p50 = frame_time_percentile_ms(0.50);
    let frame_processing_ms_p95 = frame_time_percentile_ms(0.95);
    let frame_processing_ms_p99 = frame_time_percentile_ms(0.99);
    let evaluation_metric = |value: f64, precision: usize| {
        if evaluation_ground_truth_available {
            format!("{value:.precision$}")
        } else {
            "None".to_string()
        }
    };
    let rmse_pos_summary = evaluation_metric(rmse_pos, 4);
    let max_position_err_summary = evaluation_metric(max_position_err, 4);
    let rmse_rot_deg_summary = evaluation_metric(rmse_rot_deg, 4);
    let max_orientation_err_deg_summary = evaluation_metric(max_orientation_err_deg, 4);
    let ate_rmse_rigid_summary = evaluation_metric(ate_rmse_rigid, 4);
    let max_rigid_summary = evaluation_metric(max_rigid, 4);
    let ate_rmse_sim_summary = evaluation_metric(ate_rmse_sim, 4);
    let max_sim_summary = evaluation_metric(max_sim, 4);
    let scale_summary = evaluation_metric(aligned_similarity.scale, 6);
    let final_keyframe_ate_rigid_summary = evaluation_metric(final_keyframe_ate_rigid_rmse_m, 4);
    let final_keyframe_ate_similarity_summary =
        evaluation_metric(final_keyframe_ate_similarity_rmse_m, 4);
    let final_keyframe_ate_similarity_scale_summary =
        evaluation_metric(final_keyframe_ate_similarity_scale, 6);
    let summary = format!(
        "euroc_dir={}\n\
         frames_recorded={frames_recorded}\n\
         wall_clock_seconds={wall_clock_seconds:.6}\n\
         wall_clock_ms_per_frame={wall_clock_ms_per_frame:.6}\n\
         wall_clock_fps={wall_clock_fps:.6}\n\
         frame_processing_ms_p50={frame_processing_ms_p50:?}\n\
         frame_processing_ms_p95={frame_processing_ms_p95:?}\n\
         frame_processing_ms_p99={frame_processing_ms_p99:?}\n\
         evaluation_ground_truth_available={evaluation_ground_truth_available}\n\
         seed_pose_source=identity_first_camera\n\
         tracking_success_rate={success_rate:.3}\n\
         globally_aligned_tracking_success_rate={global_tracking_success_rate:.3}\n\
         globally_aligned_tracking_successes={globally_aligned_tracking_successes}\n\
         atlas_enabled={rebootstrap_independent_submap}\n\
         atlas_submap_count={atlas_submap_count}\n\
         atlas_aligned_submap_count={atlas_aligned_submap_count}\n\
         atlas_independent_submap_count={atlas_independent_submap_count}\n\
         atlas_active_submap_id={active_submap_id}\n\
         atlas_merge_attempts={atlas_merge_attempts}\n\
         atlas_merge_source_keyframe_attempts={atlas_merge_source_keyframe_attempts}\n\
         atlas_merge_successes={atlas_merge_successes}\n\
         atlas_verified_merge_count={atlas_verified_merge_count}\n\
         atlas_verified_landmark_match_count={atlas_verified_landmark_match_count}\n\
         atlas_aligned_owned_landmark_count={atlas_aligned_owned_landmark_count}\n\
         atlas_welded_landmark_reduction={atlas_welded_landmark_reduction}\n\
         pose_prior_visual_override_count={pose_prior_visual_override_count}\n\
         imu_samples_consumed={imu_idx}\n\
         vi_init_preseed_samples={vi_init_preseed_samples}\n\
         seed_frame_idx={seed_frame_idx}\n\
         undistort={undistort}\n\
         stereo_bootstrap={stereo_bootstrap_enabled}\n\
         stereo_bootstrap_strict={stereo_bootstrap_strict}\n\
         stereo_bootstrap_cam1_features={stereo_cam1_features}\n\
         stereo_bootstrap_cam1_features_after_undistort={stereo_cam1_features_after_undistort}\n\
         stereo_bootstrap_matches={stereo_bootstrap_matches_count}\n\
         stereo_landmark_replenish_enabled={stereo_landmark_replenish_enabled}\n\
         stereo_landmark_replenish_max_per_frame={stereo_landmark_replenish_max_per_frame}\n\
         stereo_landmark_replenish_candidates_total={stereo_landmark_replenish_candidates_total}\n\
         bootstrap_depth_meters={bootstrap_depth:.3}\n\
         bootstrap_landmarks={bootstrap_landmarks}\n\
         feature_count_mean={mean_features:.1}\n\
         feature_count_min={feature_count_min}\n\
         feature_count_max={feature_count_max}\n\
         frame_appearance_descriptors_exported={appearance_descriptors_exported}\n\
         frame_appearance_descriptor_count={appearance_descriptor_count}\n\
         map_keyframes={map_keyframes}\n\
         map_landmarks={map_landmarks}\n\
         map_landmark_position_covariances={map_landmark_covariances}\n\
         map_stereo_observations={map_stereo_observations}\n\
         map_observation_confidences={map_observation_confidences}\n\
         map_observation_confidence_stats_min_mean_max={map_observation_confidence_stats:?}\n\
         observation_confidence_ba_enabled={observation_confidence_ba_enabled}\n\
         vi_init_first_event_frame={vi_first:?}\n\
         vi_init_succeeded_frame={vi_succeeded:?}\n\
         vi_init_status_final={final_vi_status:?}\n\
         motion_vi_init_enabled={motion_enabled}\n\
         motion_vi_init_after_static_give_up={motion_after_static_give_up}\n\
         motion_vi_init_from_configured_bias={motion_from_configured_bias}\n\
         motion_vi_init_first_event_frame={motion_first:?}\n\
         motion_vi_init_succeeded_frame={motion_succeeded:?}\n\
         motion_vi_init_recovered_scale={motion_scale:?}\n\
         motion_vi_init_viba2_iterations={motion_iters:?}\n\
         motion_vi_init_estimate_gravity={motion_estimate_gravity}\n\
         motion_vi_init_max_gravity_norm_deviation={motion_max_gravity_deviation:?}\n\
         motion_vi_init_estimated_gravity={motion_estimated_gravity:?}\n\
         motion_vi_init_estimate_gyro_bias={motion_estimate_gyro_bias}\n\
         motion_vi_init_estimated_gyro_bias={motion_estimated_gyro_bias:?}\n\
         motion_vi_init_status_final={final_motion_vi_status:?}\n\
         local_vi_ba_enabled={local_vi_ba_enabled}\n\
         local_vi_ba_marginalization={local_vi_ba_marginalization}\n\
         local_vi_ba_general_stereo=true\n\
         local_vi_ba_initial_prior_std_devs={local_vi_ba_initial_prior_std_devs:?}\n\
         local_vi_ba_freeze_biases_above={local_vi_ba_freeze:?}\n\
         local_vi_ba_reject_writeback_above={local_vi_ba_reject_writeback:?}\n\
         local_vi_ba_reject_final_imu_nis_per_dof_above={local_vi_ba_reject_final_imu_nis:?}\n\
         local_vi_ba_reject_velocity_above_mps={local_vi_ba_reject_velocity:?}\n\
         local_vi_ba_reject_pose_translation_above_meters={local_vi_ba_reject_pose_translation:?}\n\
         local_vi_ba_reject_pose_rotation_above_degrees={local_vi_ba_reject_pose_rotation:?}\n\
         local_vi_ba_adaptive_velocity_gate={local_vi_ba_adaptive_velocity_gate}\n\
         local_vi_ba_adaptive_velocity_quantile={local_vi_ba_adaptive_velocity_quantile:.3}\n\
         local_vi_ba_adaptive_velocity_multiplier={local_vi_ba_adaptive_velocity_multiplier:.3}\n\
         local_vi_ba_adaptive_velocity_margin_mps={local_vi_ba_adaptive_velocity_margin_mps:.3}\n\
         local_vi_ba_adaptive_velocity_min_mps={local_vi_ba_adaptive_velocity_min_mps:.3}\n\
         local_vi_ba_adaptive_velocity_max_mps={local_vi_ba_adaptive_velocity_max_mps:?}\n\
         local_vi_ba_adaptive_velocity_min_references={local_vi_ba_adaptive_velocity_min_references}\n\
         motion_vi_init_max_velocity_mps={motion_vi_max_vel:?}\n\
         motion_vi_init_max_gyro_bias_rad_s={motion_vi_max_gyro_bias:?}\n\
         motion_vi_init_max_accel_bias_mps2={motion_vi_max_accel_bias:?}\n\
         motion_vi_init_max_imu_nis_per_dof={motion_vi_max_imu_nis:?}\n\
         motion_vi_init_max_rotation_residual_rms_rad={motion_vi_max_rotation_residual_rms_rad:?}\n\
         motion_vi_init_max_velocity_residual_rms_mps={motion_vi_max_velocity_residual_rms_mps:?}\n\
         motion_vi_init_max_position_residual_rms_m={motion_vi_max_position_residual_rms_m:?}\n\
         vi_bias_release_min_keyframes={bias_release_min_kf:?}\n\
         vi_bias_release_min_translation_meters={bias_release_min_translation:?}\n\
         covisibility_local_map_max_keyframes={covisibility_max_kf:?}\n\
         covisibility_local_map_min_shared={covisibility_min_shared}\n\
         covisibility_local_map_used_frames={covisibility_used_frames}\n\
         covisibility_local_map_mean_size={covisibility_mean_size:.2}\n\
         sparse_factor_graph_enabled={sparse_factor_graph_enabled}\n\
         sparse_factor_graph_updates={sparse_factor_graph_updates}\n\
         sparse_factor_graph_added={sparse_factor_graph_added}\n\
         sparse_factor_graph_reactivated={sparse_factor_graph_reactivated}\n\
         sparse_factor_graph_inactivated_low_confidence={sparse_factor_graph_inactivated_low_confidence}\n\
         sparse_factor_graph_inactivated_window_age={sparse_factor_graph_inactivated_window_age}\n\
         sparse_factor_graph_inactivated_budget={sparse_factor_graph_inactivated_budget}\n\
         sparse_factor_graph_pruned={sparse_factor_graph_pruned}\n\
         sparse_factor_graph_active_temporal={sparse_factor_graph_active_temporal}\n\
         sparse_factor_graph_active_proximity={sparse_factor_graph_active_proximity}\n\
         sparse_factor_graph_active_stereo={sparse_factor_graph_active_stereo}\n\
         sparse_factor_graph_inactive={sparse_factor_graph_inactive}\n\
         covisibility_local_ba_enabled={covis_ba_enabled}\n\
         covisibility_local_ba_min_keyframes={covis_ba_min_keyframes}\n\
         covisibility_local_ba_max_keyframes={covis_ba_max_keyframes:?}\n\
         covisibility_local_ba_motion_vi_raw_activation={covis_ba_motion_vi_raw_activation:?}\n\
         covisibility_local_ba_max_seed_landmarks_for_activation={covis_ba_max_seed_landmarks:?}\n\
         covisibility_local_ba_trigger_every={covis_ba_trigger_every}\n\
         covisibility_local_ba_max_neighbor_keyframes={covis_ba_max_neighbors}\n\
         covisibility_local_ba_min_shared={covis_ba_min_shared}\n\
         covisibility_local_ba_max_boundary_keyframes={covis_ba_max_boundary}\n\
         covisibility_local_ba_min_boundary_observations={covis_ba_min_boundary_obs}\n\
         covisibility_local_ba_fallback_min_boundary_observations={covis_ba_fallback_min_boundary_obs:?}\n\
         covisibility_local_ba_max_landmarks={covis_ba_max_landmarks:?}\n\
         covisibility_local_ba_min_active_observations={covis_ba_min_active_obs}\n\
         covisibility_local_ba_outlier_threshold_px={covis_ba_outlier_threshold:?}\n\
         covisibility_local_ba_remove_outliers={covis_ba_remove_outliers}\n\
         covisibility_local_ba_general_stereo={covis_ba_general_stereo}\n\
         covisibility_local_ba_general_stereo_max_right_reprojection_px={covis_ba_general_stereo_max_right_reprojection_px:?}\n\
         covisibility_local_ba_max_outlier_observation_ratio={covis_ba_max_outlier_ratio:?}\n\
         covisibility_local_ba_boundary_support_min_optimized_keyframes={covis_ba_boundary_support_min_optimized:?}\n\
         covisibility_local_ba_boundary_support_min_fixed_keyframes={covis_ba_boundary_support_min_fixed}\n\
         covisibility_local_ba_max_behind_camera_ratio={covis_ba_max_behind_camera_ratio:?}\n\
         covisibility_local_ba_min_fixed_to_optimized_ratio={covis_ba_min_fixed_to_optimized_ratio:?}\n\
         covisibility_local_ba_max_pose_translation_correction_m={covis_ba_max_pose_translation_correction_m:?}\n\
         covisibility_local_ba_max_pose_rotation_correction_deg={covis_ba_max_pose_rotation_correction_deg:?}\n\
         covisibility_local_ba_triggers={covis_ba_triggers}\n\
         covisibility_local_ba_successes={covis_ba_successes}\n\
         covisibility_local_ba_failures={covis_ba_failures}\n\
         covisibility_local_ba_active_observation_gate_failures={covis_ba_active_gate_failures}\n\
         covisibility_local_ba_boundary_fallback_active_gate_failures={covis_ba_boundary_fallback_active_gate_failures}\n\
         covisibility_local_ba_quality_gate_failures={covis_ba_quality_gate_failures}\n\
         covisibility_local_ba_boundary_support_failures={covis_ba_boundary_support_failures}\n\
         covisibility_local_ba_behind_camera_gate_failures={covis_ba_behind_camera_gate_failures}\n\
         covisibility_local_ba_fixed_ratio_gate_failures={covis_ba_fixed_ratio_gate_failures}\n\
         covisibility_local_ba_pose_correction_gate_failures={covis_ba_pose_correction_gate_failures}\n\
         covisibility_local_ba_no_local_landmarks_failures={covis_ba_no_local_landmarks_failures}\n\
         covisibility_local_ba_no_observations_failures={covis_ba_no_observations_failures}\n\
         covisibility_local_ba_solver_failures={covis_ba_solver_failures}\n\
         covisibility_local_ba_other_failures={covis_ba_other_failures}\n\
         covisibility_local_ba_boundary_fallback_successes={covis_ba_boundary_fallback_successes}\n\
         covisibility_local_ba_updated_keyframes_total={covis_ba_updated_keyframes}\n\
         covisibility_local_ba_updated_landmarks_total={covis_ba_updated_landmarks}\n\
         covisibility_local_ba_removed_observations_total={covis_ba_removed_observations}\n\
         covisibility_local_ba_mean_reprojection_before_px={covis_ba_mean_before:?}\n\
         covisibility_local_ba_mean_reprojection_after_px={covis_ba_mean_after:?}\n\
         covisibility_local_ba_elapsed_ms_total={covis_ba_elapsed_ms_total:.6}\n\
         covisibility_local_ba_elapsed_ms_mean={covis_ba_elapsed_ms_mean:?}\n\
         covisibility_local_ba_elapsed_ms_max={covis_ba_elapsed_ms_max:.6}\n\
         covisibility_local_ba_last_error={covis_ba_last_error:?}\n\
         pose_graph_refinement={pgr_enabled}\n\
         pose_graph_refinement_trigger_every={pgr_trigger_every}\n\
         pose_graph_refinement_fixed_loop_edge_weight={pgr_fixed_loop_edge_weight:?}\n\
         pose_graph_refinement_loop_pose_information={pgr_loop_pose_information}\n\
         pose_graph_refinement_loop_pose_information_max_eigenvalue={pgr_loop_pose_information_max_eigenvalue:.9}\n\
         pose_graph_refinement_loop_pose_information_loop_edge_scale={pgr_loop_pose_information_loop_edge_scale:.9}\n\
         pose_graph_refinement_gnc={pgr_gnc}\n\
         pose_graph_refinement_pcm={pgr_pcm}\n\
         pose_graph_refinement_pcm_require_individual={pgr_pcm_require_individual}\n\
         pose_graph_refinement_covariance_gate={pgr_covariance_gate:?}\n\
         pose_graph_refinement_verifier={pgr_verifier}\n\
         pose_graph_refinement_solver={pgr_solver}\n\
         pose_graph_refinement_last_solve_scale_spread={pgr_scale_spread:?}\n\
         pose_graph_refinement_candidates_seen={pgr_candidates_seen}\n\
         pose_graph_refinement_verified_constraints={pgr_verified_constraints}\n\
         pose_graph_refinement_pgo_solves={pgr_pgo_solves}\n\
         pose_graph_refinement_gnc_rejected_loop_edges_max={pgr_gnc_rejected_loop_edges_max}\n\
         pose_graph_refinement_pcm_rejected={pgr_pcm_rejected}\n\
         pose_graph_refinement_covariance_rejected={pgr_covariance_rejected}\n\
         pose_graph_refinement_pose_information_rejected={pgr_pose_information_rejected}\n\
         pose_graph_refinement_with_pose_information={pgr_with_pose_information}\n\
         pose_graph_refinement_pose_information_estimates={pgr_pose_information_estimates}\n\
         pose_graph_refinement_pose_information_used_correspondences={pgr_pose_information_used_correspondences}\n\
         pose_graph_refinement_pose_information_stereo_covariance_correspondences={pgr_pose_information_stereo_covariance_correspondences}\n\
         pose_graph_refinement_pose_information_raw_condition_max={pgr_pose_information_raw_condition_max:.9}\n\
         pose_graph_refinement_pose_information_spectral_scale_min={pgr_pose_information_spectral_scale_min:?}\n\
         pose_graph_refinement_pose_information_failure_counts={pgr_pose_information_failure_counts}\n\
         pose_graph_refinement_sequential_with_pose_information={pgr_sequential_with_pose_information}\n\
         pose_graph_refinement_sequential_pose_information_fallbacks={pgr_sequential_pose_information_fallbacks}\n\
         pose_graph_refinement_sequential_pose_information_used_correspondences={pgr_sequential_pose_information_used_correspondences}\n\
         pose_graph_refinement_sequential_pose_information_stereo_covariance_correspondences={pgr_sequential_pose_information_stereo_covariance_correspondences}\n\
         pose_graph_refinement_sequential_pose_information_raw_condition_max={pgr_sequential_pose_information_raw_condition_max:.9}\n\
         pose_graph_refinement_sequential_pose_information_spectral_scale_min={pgr_sequential_pose_information_spectral_scale_min:?}\n\
         pose_graph_refinement_sequential_pose_information_failure_counts={pgr_sequential_pose_information_failure_counts}\n\
         pose_graph_refinement_appearance_loops={pgr_appearance_enabled}\n\
         pose_graph_refinement_appearance_min_gap={pgr_appearance_min_gap}\n\
         pose_graph_refinement_appearance_max_candidates={pgr_appearance_max_candidates}\n\
         pose_graph_refinement_appearance_min_inliers={pgr_appearance_min_inliers}\n\
         pose_graph_refinement_appearance_confirmation_keyframes={pgr_appearance_confirmation_keyframes}\n\
         pose_graph_refinement_appearance_confirmation_max_misses={pgr_appearance_confirmation_max_misses}\n\
         pose_graph_refinement_appearance_projection_radius_px={pgr_appearance_projection_radius:?}\n\
         pose_graph_refinement_appearance_projection_min_matches={pgr_appearance_projection_min_matches}\n\
         pose_graph_refinement_fuse_loop_observations={pgr_fuse_loop_observations}\n\
         pose_graph_refinement_loop_fusion_pairs_considered={pgr_loop_fusion_pairs_considered}\n\
         pose_graph_refinement_loop_fusion_observations_inserted={pgr_loop_fusion_observations_inserted}\n\
         pose_graph_refinement_loop_fusion_observations_reassigned={pgr_loop_fusion_observations_reassigned}\n\
         pose_graph_refinement_loop_fusion_pairs_skipped={pgr_loop_fusion_pairs_skipped}\n\
         pose_graph_refinement_loop_fusion_pairs_robust_rejected={pgr_loop_fusion_pairs_robust_rejected}\n\
         pose_graph_refinement_loop_fusion_pairs_reprojection_rejected={pgr_loop_fusion_pairs_reprojection_rejected}\n\
         pose_graph_refinement_loop_welding_ba={pgr_loop_welding_ba}\n\
         pose_graph_refinement_loop_welding_ba_attempted={pgr_loop_welding_ba_attempted}\n\
         pose_graph_refinement_loop_welding_ba_succeeded={pgr_loop_welding_ba_succeeded}\n\
         pose_graph_refinement_loop_welding_ba_rejected_or_failed={pgr_loop_welding_ba_rejected_or_failed}\n\
         pose_graph_refinement_loop_welding_ba_updated_keyframes={pgr_loop_welding_ba_updated_keyframes}\n\
         pose_graph_refinement_loop_welding_ba_updated_landmarks={pgr_loop_welding_ba_updated_landmarks}\n\
         pose_graph_refinement_loop_welding_post_pgo_attempted={pgr_loop_welding_post_pgo_attempted}\n\
         pose_graph_refinement_loop_welding_post_pgo_succeeded={pgr_loop_welding_post_pgo_succeeded}\n\
         pose_graph_refinement_loop_welding_post_pgo_reprojection_max_px={pgr_loop_welding_post_pgo_reprojection_max:?}\n\
         pose_graph_refinement_loop_welding_post_ba_last_error={pgr_loop_welding_post_ba_last_error:?}\n\
         pose_graph_refinement_loop_welding_post_ba_behind_camera_ratio_max={pgr_loop_welding_post_ba_behind_camera_ratio_max:?}\n\
         pose_graph_refinement_loop_welding_initial_translation_max_m={pgr_loop_welding_initial_translation_max:?}\n\
         pose_graph_refinement_loop_welding_initial_rotation_max_rad={pgr_loop_welding_initial_rotation_max:?}\n\
         pose_graph_refinement_appearance_ranked={pgr_appearance_ranked}\n\
         pose_graph_refinement_appearance_connected_region_rejected={pgr_appearance_connected_region_rejected}\n\
         pose_graph_refinement_appearance_pnp_verified={pgr_appearance_pnp_verified}\n\
         pose_graph_refinement_appearance_projection_rejected={pgr_appearance_projection_rejected}\n\
         pose_graph_refinement_appearance_covisibility_rejected={pgr_appearance_covisibility_rejected}\n\
         pose_graph_refinement_appearance_pending_projection_attempted={pgr_appearance_pending_projection_attempted}\n\
         pose_graph_refinement_appearance_pending_projection_verified={pgr_appearance_pending_projection_verified}\n\
         pose_graph_refinement_appearance_pending_projection_correspondences={pgr_appearance_pending_projection_correspondences}\n\
         pose_graph_refinement_appearance_pending_projection_inliers={pgr_appearance_pending_projection_inliers}\n\
         pose_graph_refinement_appearance_pending_projection_correspondences_max={pgr_appearance_pending_projection_correspondences_max}\n\
         pose_graph_refinement_appearance_pending_projection_inliers_max={pgr_appearance_pending_projection_inliers_max}\n\
         pose_graph_refinement_appearance_pending_projection_radius_min={pgr_appearance_pending_projection_radius_min:?}\n\
         pose_graph_refinement_appearance_pending_projection_radius_max={pgr_appearance_pending_projection_radius_max:?}\n\
         pose_graph_refinement_appearance_region_confirmation_waiting={pgr_appearance_region_confirmation_waiting}\n\
         pose_graph_refinement_appearance_region_confirmed={pgr_appearance_region_confirmed}\n\
         pose_graph_refinement_appearance_scale_failed={pgr_appearance_scale_failed}\n\
         pose_graph_refinement_appearance_scale_insufficient_points={pgr_appearance_scale_insufficient_points}\n\
         pose_graph_refinement_appearance_scale_insufficient_matches={pgr_appearance_scale_insufficient_matches}\n\
         pose_graph_refinement_appearance_scale_no_consensus={pgr_appearance_scale_no_consensus}\n\
         pose_graph_refinement_appearance_near_unit={pgr_appearance_near_unit}\n\
         pose_graph_refinement_constraints_shared={pgr_constraints_shared}\n\
         pose_graph_refinement_constraints_appearance={pgr_constraints_appearance}\n\
         pose_graph_refinement_gt_evaluated={pgr_gt_evaluated}\n\
         pose_graph_refinement_gt_correct_0p5m_10deg={pgr_gt_correct}\n\
         pose_graph_refinement_gt_precision_0p5m_10deg={pgr_gt_precision:?}\n\
         pose_graph_refinement_rejected_gt_evaluated={pgr_rejected_gt_evaluated}\n\
         pose_graph_refinement_rejected_gt_correct_0p5m_10deg={pgr_rejected_gt_correct}\n\
         pose_graph_refinement_rejected_gt_correct_ratio_0p5m_10deg={pgr_rejected_gt_correct_ratio:?}\n\
         pose_graph_refinement_propagate={pgr_propagate}\n\
         pose_graph_refinement_landmarks_moved={pgr_landmarks_moved}\n\
         pose_graph_refinement_max_landmark_displacement_meters={pgr_max_landmark_displacement_meters:.6}\n\
         pose_graph_refinement_tracker_corrections_applied={pgr_tracker_corrections_applied}\n\
         max_pose_jump_meters={max_pose_jump:?}\n\
         pose_prior_visual_override={pose_prior_visual_override}\n\
         pose_jump_gap_scaling={pose_jump_gap_scaling}\n\
         pose_jump_gap_scaling_max_multiplier={pose_jump_gap_scaling_max_multiplier}\n\
         tracking_min_inliers={tracking_min_inliers}\n\
         tracking_min_inlier_ratio={tracking_min_inlier_ratio:.6}\n\
         tracking_max_reprojection_error={tracking_max_reprojection_error:?}\n\
         pnp_pose_prior_warm_start={pnp_warm_start}\n\
         projection_guided_tracking={projection_guided_tracking}\n\
         projection_query_landmark_distance_ratio={projection_query_landmark_ratio:?}\n\
         projection_search_radius_px={projection_search_radius_px:.3}\n\
         projection_widen_factor={projection_widen_factor:.3}\n\
         projection_max_widen_retries={projection_max_widen_retries}\n\
         projection_local_map_refinement={projection_local_map_refinement}\n\
         projection_refinement_search_radius_px={projection_refinement_search_radius_px:.3}\n\
         projection_refinement_iterations={projection_refinement_iterations}\n\
         projection_refinement_radius_shrink_factor={projection_refinement_radius_shrink_factor:.3}\n\
         projection_refinement_reassign_correspondences={projection_refinement_reassign_correspondences}\n\
         projection_refinement_min_inlier_pair_retention_ratio={projection_refinement_min_inlier_pair_retention_ratio:.3}\n\
         projection_refinement_max_translation_correction_m={projection_refinement_max_translation_correction_m:?}\n\
         projection_refinement_max_rotation_correction_deg={projection_refinement_max_rotation_correction_deg:?}\n\
         projection_guided_attempt_count={projection_guided_attempt_count}\n\
         projection_guided_widen_retry_count={projection_guided_widen_retry_count}\n\
         projection_guided_success_count={projection_guided_success_count}\n\
         projection_guided_fallback_success_count={projection_guided_fallback_success_count}\n\
         local_map_refinement_correspondence_gain_total={local_map_refinement_correspondence_gain_total}\n\
         local_map_refinement_accepted_count={local_map_refinement_accepted_count}\n\
         local_map_refinement_rejected_count={local_map_refinement_rejected_count}\n\
         pnp_reprojection_threshold_px={pnp_reproj_thresh:?}\n\
         motion_model={motion_model_kind}\n\
         imu_extrinsic_from_cam0={imu_tbs}\n\
         imu_motion_model_carry_forward_velocity={imu_carry_fwd}\n\
         adaptive_motion_failures_to_switch_to_pose={adaptive_fail_thresh}\n\
         adaptive_motion_successes_to_switch_to_imu={adaptive_succ_thresh}\n\
         adaptive_motion_imu_velocity_refresh_policy={adaptive_refresh_policy}\n\
         adaptive_motion_switches_to_pose={adaptive_switches_pose}\n\
         adaptive_motion_switches_to_imu={adaptive_switches_imu}\n\
         adaptive_motion_velocity_refreshes_on_switch_to_imu={adaptive_velocity_refreshes}\n\
         adaptive_motion_final_mode={adaptive_final_mode}\n\
         cross_check_matcher={cross_check}\n\
         mutual_softmax_matcher={mutual_softmax}\n\
         feature_extractor={feature_extractor_kind}\n\
         superpoint_features_dir={superpoint_features_dir:?}\n\
         superpoint_cam1_features_dir={superpoint_cam1_features_dir:?}\n\
         superpoint_onnx_model={superpoint_onnx_model:?}\n\
         superpoint_onnx_backend={superpoint_onnx_backend}\n\
         keyframe_min_translation={kf_min_translation:?}\n\
         keyframe_min_frame_gap={kf_min_frame_gap:?}\n\
         keyframe_max_frame_gap={kf_max_frame_gap:?}\n\
         keyframe_tracked_landmark_ratio={kf_tracked_ratio:?}\n\
         keyframe_min_tracked_landmarks_for_ratio={kf_min_tracked_for_ratio}\n\
         keyframe_min_inliers={kf_min_inliers:?}\n\
         keyframe_min_inlier_ratio={kf_min_inlier_ratio:?}\n\
         keyframe_insufficient_tracking_quality_rejections={kf_insufficient_tracking_quality_rejections}\n\
         keep_pre_promotion_imu_factors={keep_pre_promotion}\n\
         relinearise_imu_factor_bias_thresholds={relinearise_thresholds:?}\n\
         run_local_vi_ba_at_vi_init_promotion={run_at_promotion}\n\
         relocalization_enabled={reloc_enabled}\n\
         relocalization_min_inliers={reloc_min_inliers}\n\
         relocalization_min_inlier_ratio={reloc_min_inlier_ratio}\n\
         relocalization_max_mean_reprojection_error={reloc_max_rep_err:?}\n\
         relocalization_pose_prior_radius={reloc_pose_prior_radius:?}\n\
         relocalization_recent_keyframe_window={reloc_recent_kf_window:?}\n\
         relocalization_covisibility_max_keyframes={reloc_covis_max_kf:?}\n\
         relocalization_covisibility_min_shared={reloc_covis_min_shared}\n\
         relocalization_covisibility_min_landmarks={reloc_covis_min_landmarks}\n\
         relocalization_covisibility_broader_fallback={reloc_covis_broader_fallback}\n\
         relocalization_covisibility_broader_fallback_interval_frames={reloc_covis_broader_fallback_interval_frames}\n\
         relocalization_covisibility_compare_broader_store={reloc_covis_compare_broader}\n\
         relocalization_appearance_max_keyframes={reloc_appearance_max_kf:?}\n\
         relocalization_appearance_candidate_log_limit={reloc_appearance_candidate_log_limit:?}\n\
         relocalization_appearance_min_similarity={reloc_appearance_min_similarity:.6}\n\
         relocalization_appearance_exclude_recent_frame_gap={reloc_appearance_exclude_recent_frame_gap}\n\
         relocalization_appearance_min_landmarks={reloc_appearance_min_landmarks}\n\
         relocalization_appearance_broader_fallback={reloc_appearance_broader_fallback}\n\
         relocalization_appearance_broader_fallback_interval_frames={reloc_appearance_broader_fallback_interval_frames}\n\
         relocalization_appearance_compare_broader_store={reloc_appearance_compare_broader}\n\
         relocalization_max_translation_from_imu_prediction_meters={reloc_max_tx_from_imu:?}\n\
         relocalization_attempt_interval_frames={reloc_attempt_interval_frames}\n\
         relocalization_max_consecutive_failed_attempts={reloc_max_consecutive_failed_attempts:?}\n\
         relocalization_max_translation_per_frame_from_last_success_meters={reloc_max_tx_per_frame_from_last_success:?}\n\
         relocalization_min_inlier_depth_median_ratio_to_last_success={reloc_min_depth_ratio_to_last_success:?}\n\
         relocalization_max_inlier_depth_median_ratio_to_last_success={reloc_max_depth_ratio_to_last_success:?}\n\
         relocalization_confirmation_required_recoveries={reloc_confirmation_required}\n\
         relocalization_confirmation_max_translation_per_frame_meters={reloc_confirmation_max_tx_per_frame:?}\n\
         relocalization_attempts={reloc_attempts}\n\
         relocalization_successes={reloc_successes}\n\
         rebootstrap_after_lost_frames={rebootstrap_after_lost_frames:?}\n\
         rebootstrap_cooldown_frames={rebootstrap_cooldown_frames}\n\
         rebootstrap_independent_submap={rebootstrap_independent_submap}\n\
         atlas_broader_recovery_max_source_keyframes={atlas_broader_recovery_max_source_keyframes}\n\
         atlas_broader_recovery_interval_attempts={atlas_broader_recovery_interval_attempts}\n\
         atlas_broader_recovery_cycles={atlas_broader_recovery_cycles}\n\
         atlas_boundary_keyframe_snapshots={atlas_boundary_keyframe_snapshots}\n\
         atlas_boundary_factor_attempts={atlas_boundary_factor_attempts}\n\
         atlas_boundary_factor_successes={atlas_boundary_factor_successes}\n\
         rebootstrap_events={rebootstrap_events}\n\
         rebootstrap_final_segment_id={segment_id}\n\
         relocalization_gate_passes={reloc_gate_passes}\n\
         relocalization_descriptor_store_landmark_count_observations={reloc_descriptor_store_count_observations}\n\
         relocalization_descriptor_store_landmark_count_mean={reloc_descriptor_store_count_mean:?}\n\
         relocalization_descriptor_store_landmark_count_min={reloc_descriptor_store_count_min:?}\n\
         relocalization_descriptor_store_landmark_count_max={reloc_descriptor_store_count_max:?}\n\
         relocalization_covisibility_descriptor_store_tried_frames={reloc_covis_descriptor_store_tried_frames}\n\
         relocalization_covisibility_descriptor_store_used_frames={reloc_covis_descriptor_store_used_frames}\n\
         relocalization_appearance_descriptor_store_tried_frames={reloc_appearance_descriptor_store_tried_frames}\n\
         relocalization_appearance_descriptor_store_used_frames={reloc_appearance_descriptor_store_used_frames}\n\
         relocalization_appearance_candidate_keyframe_count_observations={reloc_appearance_candidate_count_observations}\n\
         relocalization_appearance_candidate_keyframe_count_mean={reloc_appearance_candidate_count_mean:?}\n\
         relocalization_appearance_best_similarity_count={reloc_appearance_best_similarity_count}\n\
         relocalization_appearance_best_similarity_mean={reloc_appearance_best_similarity_mean:?}\n\
         relocalization_appearance_best_similarity_max={reloc_appearance_best_similarity_max:?}\n\
         relocalization_broader_descriptor_store_retry_frames={reloc_broader_descriptor_store_retry_frames}\n\
         relocalization_broader_descriptor_store_retry_interval_skips={reloc_broader_descriptor_store_retry_interval_skips}\n\
         relocalization_broader_descriptor_store_used_frames={reloc_broader_descriptor_store_used_frames}\n\
         relocalization_budget_skips={reloc_budget_skips}\n\
         relocalization_covisibility_reference_keyframe_count={reloc_covis_reference_keyframe_count}\n\
         relocalization_confirmation_waiting={reloc_confirmation_waiting}\n\
         relocalization_confirmation_translation_per_frame_from_previous_count={reloc_confirmation_tx_per_frame_count}\n\
         relocalization_confirmation_translation_per_frame_from_previous_mean={reloc_confirmation_tx_per_frame_mean:?}\n\
         relocalization_confirmation_translation_per_frame_from_previous_max={reloc_confirmation_tx_per_frame_max:?}\n\
         relocalization_translation_per_frame_from_last_success_count={reloc_tx_per_frame_count}\n\
         relocalization_translation_per_frame_from_last_success_mean={reloc_tx_per_frame_mean:?}\n\
         relocalization_translation_per_frame_from_last_success_max={reloc_tx_per_frame_max:?}\n\
         relocalization_success_translation_per_frame_from_last_success_count={reloc_success_tx_per_frame_count}\n\
         relocalization_success_translation_per_frame_from_last_success_mean={reloc_success_tx_per_frame_mean:?}\n\
         relocalization_success_translation_per_frame_from_last_success_max={reloc_success_tx_per_frame_max:?}\n\
         relocalization_inlier_depth_median_ratio_to_last_success_count={reloc_depth_ratio_count}\n\
         relocalization_inlier_depth_median_ratio_to_last_success_mean={reloc_depth_ratio_mean:?}\n\
         relocalization_inlier_depth_median_ratio_to_last_success_min={reloc_depth_ratio_min:?}\n\
         relocalization_inlier_depth_median_ratio_to_last_success_max={reloc_depth_ratio_max:?}\n\
         relocalization_success_inlier_depth_median_ratio_to_last_success_count={reloc_success_depth_ratio_count}\n\
         relocalization_success_inlier_depth_median_ratio_to_last_success_mean={reloc_success_depth_ratio_mean:?}\n\
         relocalization_success_inlier_depth_median_ratio_to_last_success_min={reloc_success_depth_ratio_min:?}\n\
         relocalization_success_inlier_depth_median_ratio_to_last_success_max={reloc_success_depth_ratio_max:?}\n\
         vi_init_try_initialize_on_every_frame={vi_init_try_every_frame}\n\
         imu_factors_staged={imu_factors_staged}\n\
         local_vi_ba_triggers={local_vi_ba_triggers}\n\
         local_vi_ba_relinearised_factor_total={local_vi_ba_relinearised_factor_total}\n\
         local_vi_ba_stereo_observation_total={local_vi_ba_stereo_observation_total}\n\
         local_vi_ba_marginalization_priors_applied={local_vi_ba_marginalization_priors_applied}\n\
         local_vi_ba_marginalization_successes={local_vi_ba_marginalization_successes}\n\
         local_vi_ba_quality_gate_rejections={local_vi_ba_quality_gate_rejections}\n\
         local_vi_ba_cost_ratio_gate_rejections={local_vi_ba_cost_ratio_gate_rejections}\n\
         local_vi_ba_imu_nis_gate_rejections={local_vi_ba_imu_nis_gate_rejections}\n\
         local_vi_ba_velocity_gate_rejections={local_vi_ba_velocity_gate_rejections}\n\
         local_vi_ba_pose_correction_gate_rejections={local_vi_ba_pose_correction_gate_rejections}\n\
         local_vi_ba_max_pose_translation_correction_meters={local_vi_ba_max_pose_translation_correction_meters:.9}\n\
         local_vi_ba_max_pose_rotation_correction_degrees={local_vi_ba_max_pose_rotation_correction_degrees:.9}\n\
         local_vi_ba_adaptive_velocity_gate_rejections={local_vi_ba_adaptive_velocity_gate_rejections}\n\
         local_vi_ba_last_adaptive_velocity_threshold_mps={local_vi_ba_last_adaptive_velocity_threshold_mps:?}\n\
         local_vi_ba_mirrors_into_imu_motion_model={local_vi_ba_mirrors}\n\
         last_mirrored_velocity_world={last_mirrored_v:?}\n\
         last_mirrored_bias_gyro={last_mirrored_bg:?}\n\
         last_mirrored_bias_acc={last_mirrored_ba:?}\n\
         tracking_quality_gate_failures={quality_gate_failures}\n\
         ate_position_rmse_m={rmse_pos_summary}\n\
         ate_position_max_m={max_position_err_summary}\n\
         ate_orientation_rmse_deg={rmse_rot_deg_summary}\n\
         ate_orientation_max_deg={max_orientation_err_deg_summary}\n\
         ate_rigid_rmse_m={ate_rmse_rigid_summary}\n\
         ate_rigid_max_m={max_rigid_summary}\n\
         ate_similarity_rmse_m={ate_rmse_sim_summary}\n\
         ate_similarity_max_m={max_sim_summary}\n\
         ate_similarity_scale={scale_summary}\n\
         final_keyframe_ate_rigid_rmse_m={final_keyframe_ate_rigid_summary}\n\
         final_keyframe_ate_similarity_rmse_m={final_keyframe_ate_similarity_summary}\n\
         final_keyframe_ate_similarity_scale={final_keyframe_ate_similarity_scale_summary}\n\
         final_keyframe_rpe_delta1_pairs={}\n\
         final_keyframe_rpe_delta1_translation_rmse_m={final_keyframe_rpe_delta1_translation_rmse_m:?}\n\
         final_keyframe_rpe_delta1_rotation_rmse_deg={final_keyframe_rpe_delta1_rotation_rmse_deg:?}\n\
         final_keyframe_rpe_delta10_pairs={}\n\
         final_keyframe_rpe_delta10_translation_rmse_m={final_keyframe_rpe_delta10_translation_rmse_m:?}\n\
         final_keyframe_rpe_delta10_rotation_rmse_deg={final_keyframe_rpe_delta10_rotation_rmse_deg:?}\n\
         final_keyframe_count={final_keyframe_trajectory_count}\n\
         final_keyframe_evaluation_count={final_keyframe_count}\n",
        args.euroc_dir.display(),
        final_keyframe_rpe_delta1.pair_count,
        final_keyframe_rpe_delta10.pair_count,
        success_rate = if frames_recorded > 0 {
            tracking_successes as f64 / frames_recorded as f64
        } else {
            0.0
        },
        pose_prior_visual_override_count = slam.tracker.stats().pose_prior_visual_override_count,
        undistort = args.undistort,
        stereo_bootstrap_enabled = args.stereo_bootstrap,
        stereo_bootstrap_strict = args.stereo_bootstrap_strict,
        stereo_cam1_features = stereo_cam1_features_count,
        stereo_cam1_features_after_undistort = stereo_cam1_features_after_undistort_count,
        stereo_bootstrap_matches_count = stereo_bootstrap_matches.len(),
        stereo_landmark_replenish_enabled = args.stereo_landmark_replenish,
        stereo_landmark_replenish_max_per_frame = args.stereo_landmark_replenish_max_per_frame,
        stereo_landmark_replenish_candidates_total = stereo_landmark_replenish_candidates_total,
        bootstrap_depth = args.bootstrap_depth_meters,
        bootstrap_landmarks = bootstrap_landmark_count,
        appearance_descriptors_exported = args.export_frame_appearance_descriptors,
        appearance_descriptor_count = frame_appearance_descriptor_count,
        vi_first = vi_init_first_event_at_frame,
        vi_succeeded = vi_init_succeeded_at_frame,
        motion_enabled = args.motion_vi_init_enabled,
        motion_after_static_give_up = args.motion_vi_init_after_static_give_up,
        motion_from_configured_bias = args.motion_vi_init_from_configured_bias,
        motion_first = motion_vi_init_first_event_at_frame,
        motion_succeeded = motion_vi_init_succeeded_at_frame,
        motion_scale = motion_vi_init_recovered_scale,
        motion_iters = motion_vi_init_viba2_iterations,
        motion_estimate_gravity = args.motion_vi_init_estimate_gravity,
        motion_max_gravity_deviation = args.motion_vi_init_max_gravity_norm_deviation,
        motion_estimated_gravity = motion_vi_init_estimated_gravity.map(|g| g.as_slice().to_vec()),
        motion_estimate_gyro_bias = args.motion_vi_init_estimate_gyro_bias,
        motion_estimated_gyro_bias = motion_vi_init_estimated_gyro_bias.map(|b| b.as_slice().to_vec()),
        local_vi_ba_enabled = args.local_vi_ba_enabled,
        observation_confidence_ba_enabled = args.observation_confidence_ba_enabled,
        local_vi_ba_marginalization = args.local_vi_ba_marginalization,
        local_vi_ba_initial_prior_std_devs = args.local_vi_ba_initial_prior_std_devs,
        local_vi_ba_freeze = args.local_vi_ba_freeze_biases_above,
        local_vi_ba_reject_writeback = args.local_vi_ba_reject_writeback_above,
        local_vi_ba_reject_final_imu_nis =
            args.local_vi_ba_reject_final_imu_nis_per_dof_above,
        local_vi_ba_reject_velocity = args.local_vi_ba_reject_velocity_above_mps,
        local_vi_ba_reject_pose_translation =
            args.local_vi_ba_reject_pose_translation_above_meters,
        local_vi_ba_reject_pose_rotation = args.local_vi_ba_reject_pose_rotation_above_degrees,
        local_vi_ba_adaptive_velocity_gate = args.local_vi_ba_adaptive_velocity_gate,
        local_vi_ba_adaptive_velocity_quantile = args.local_vi_ba_adaptive_velocity_quantile,
        local_vi_ba_adaptive_velocity_multiplier = args.local_vi_ba_adaptive_velocity_multiplier,
        local_vi_ba_adaptive_velocity_margin_mps =
            args.local_vi_ba_adaptive_velocity_margin_mps,
        local_vi_ba_adaptive_velocity_min_mps = args.local_vi_ba_adaptive_velocity_min_mps,
        local_vi_ba_adaptive_velocity_max_mps = args.local_vi_ba_adaptive_velocity_max_mps,
        local_vi_ba_adaptive_velocity_min_references =
            args.local_vi_ba_adaptive_velocity_min_references,
        motion_vi_max_vel = args.motion_vi_init_max_velocity_mps,
        motion_vi_max_gyro_bias = args.motion_vi_init_max_gyro_bias_rad_s,
        motion_vi_max_accel_bias = args.motion_vi_init_max_accel_bias_mps2,
        motion_vi_max_imu_nis = args.motion_vi_init_max_imu_nis_per_dof,
        motion_vi_max_rotation_residual_rms_rad =
            args.motion_vi_init_max_rotation_residual_rms_rad,
        motion_vi_max_velocity_residual_rms_mps =
            args.motion_vi_init_max_velocity_residual_rms_mps,
        motion_vi_max_position_residual_rms_m =
            args.motion_vi_init_max_position_residual_rms_meters,
        bias_release_min_kf = args.vi_bias_release_min_keyframes,
        bias_release_min_translation = args.vi_bias_release_min_translation_meters,
        covisibility_max_kf = args.covisibility_local_map_max_keyframes,
        covisibility_min_shared = args.covisibility_local_map_min_shared,
        covisibility_used_frames = covisibility_local_map_frames,
        covisibility_mean_size = if covisibility_local_map_frames > 0 {
            covisibility_local_map_size_sum as f64 / covisibility_local_map_frames as f64
        } else {
            0.0
        },
        sparse_factor_graph_enabled = args.sparse_factor_graph_enabled,
        sparse_factor_graph_updates = sparse_factor_graph_updates,
        sparse_factor_graph_added = sparse_factor_graph_added,
        sparse_factor_graph_reactivated = sparse_factor_graph_reactivated,
        sparse_factor_graph_inactivated_low_confidence =
            sparse_factor_graph_inactivated_low_confidence,
        sparse_factor_graph_inactivated_window_age =
            sparse_factor_graph_inactivated_window_age,
        sparse_factor_graph_inactivated_budget = sparse_factor_graph_inactivated_budget,
        sparse_factor_graph_pruned = sparse_factor_graph_pruned,
        sparse_factor_graph_active_temporal = sparse_factor_graph_active_temporal,
        sparse_factor_graph_active_proximity = sparse_factor_graph_active_proximity,
        sparse_factor_graph_active_stereo = sparse_factor_graph_active_stereo,
        sparse_factor_graph_inactive = sparse_factor_graph_inactive,
        covis_ba_enabled = args.covisibility_local_ba_enabled,
        covis_ba_min_keyframes = args.covisibility_local_ba_min_keyframes,
        covis_ba_max_keyframes = args.covisibility_local_ba_max_keyframes,
        covis_ba_motion_vi_raw_activation = args.covisibility_local_ba_motion_vi_raw_activation,
        covis_ba_max_seed_landmarks = args
            .covisibility_local_ba_max_seed_landmarks_for_activation,
        covis_ba_trigger_every = args.covisibility_local_ba_trigger_every,
        covis_ba_max_neighbors = args.covisibility_local_ba_max_neighbor_keyframes,
        covis_ba_min_shared = args.covisibility_local_ba_min_shared,
        covis_ba_max_boundary = args.covisibility_local_ba_max_boundary_keyframes,
        covis_ba_min_boundary_obs = args.covisibility_local_ba_min_boundary_observations,
        covis_ba_fallback_min_boundary_obs = args
            .covisibility_local_ba_fallback_min_boundary_observations,
        covis_ba_max_landmarks = args.covisibility_local_ba_max_landmarks,
        covis_ba_min_active_obs = args.covisibility_local_ba_min_active_observations,
        covis_ba_outlier_threshold = args.covisibility_local_ba_outlier_threshold_px,
        covis_ba_remove_outliers = args.covisibility_local_ba_remove_outliers,
        covis_ba_general_stereo = args.covisibility_local_ba_general_stereo,
        covis_ba_general_stereo_max_right_reprojection_px = args
            .covisibility_local_ba_general_stereo_max_right_reprojection_px,
        covis_ba_max_outlier_ratio = args.covisibility_local_ba_max_outlier_observation_ratio,
        covis_ba_boundary_support_min_optimized = args
            .covisibility_local_ba_boundary_support_min_optimized_keyframes,
        covis_ba_boundary_support_min_fixed = args
            .covisibility_local_ba_boundary_support_min_fixed_keyframes,
        covis_ba_max_behind_camera_ratio = args.covisibility_local_ba_max_behind_camera_ratio,
        covis_ba_min_fixed_to_optimized_ratio =
            args.covisibility_local_ba_min_fixed_to_optimized_ratio,
        covis_ba_max_pose_translation_correction_m = args
            .covisibility_local_ba_max_pose_translation_correction_m,
        covis_ba_max_pose_rotation_correction_deg = args
            .covisibility_local_ba_max_pose_rotation_correction_deg,
        covis_ba_triggers = covisibility_local_ba_triggers,
        covis_ba_successes = covisibility_local_ba_successes,
        covis_ba_failures = covisibility_local_ba_failures,
        covis_ba_active_gate_failures = covisibility_local_ba_active_observation_gate_failures,
        covis_ba_boundary_fallback_active_gate_failures =
            covisibility_local_ba_boundary_fallback_active_gate_failures,
        covis_ba_quality_gate_failures = covisibility_local_ba_quality_gate_failures,
        covis_ba_boundary_support_failures =
            covisibility_local_ba_boundary_support_failures,
        covis_ba_behind_camera_gate_failures =
            covisibility_local_ba_behind_camera_gate_failures,
        covis_ba_fixed_ratio_gate_failures = covisibility_local_ba_fixed_ratio_gate_failures,
        covis_ba_pose_correction_gate_failures =
            covisibility_local_ba_pose_correction_gate_failures,
        covis_ba_no_local_landmarks_failures = covisibility_local_ba_no_local_landmarks_failures,
        covis_ba_no_observations_failures = covisibility_local_ba_no_observations_failures,
        covis_ba_solver_failures = covisibility_local_ba_solver_failures,
        covis_ba_other_failures = covisibility_local_ba_other_failures,
        covis_ba_boundary_fallback_successes =
            covisibility_local_ba_boundary_fallback_successes,
        covis_ba_updated_keyframes = covisibility_local_ba_updated_keyframes_total,
        covis_ba_updated_landmarks = covisibility_local_ba_updated_landmarks_total,
        covis_ba_removed_observations = covisibility_local_ba_removed_observations_total,
        covis_ba_mean_before = covisibility_local_ba_mean_reprojection_before_px,
        covis_ba_mean_after = covisibility_local_ba_mean_reprojection_after_px,
        covis_ba_elapsed_ms_total = covisibility_local_ba_elapsed_ms_total,
        covis_ba_elapsed_ms_mean = covisibility_local_ba_elapsed_ms_mean,
        covis_ba_elapsed_ms_max = covisibility_local_ba_elapsed_ms_max,
        covis_ba_last_error = covisibility_local_ba_last_error,
        pgr_enabled = args.pose_graph_refinement_enabled,
        pgr_trigger_every = args.pose_graph_refinement_trigger_every,
        pgr_fixed_loop_edge_weight = args.pose_graph_refinement_fixed_loop_edge_weight,
        pgr_loop_pose_information = args.pose_graph_refinement_loop_pose_information,
        pgr_loop_pose_information_max_eigenvalue =
            args.pose_graph_refinement_loop_pose_information_max_eigenvalue,
        pgr_loop_pose_information_loop_edge_scale =
            args.pose_graph_refinement_loop_pose_information_loop_edge_scale,
        pgr_gnc = args.pose_graph_refinement_gnc,
        pgr_pcm = args.pose_graph_refinement_pcm,
        pgr_pcm_require_individual = args.pose_graph_refinement_pcm_require_individual,
        pgr_covariance_gate = args.pose_graph_refinement_covariance_gate,
        pgr_verifier = match args.pose_graph_refinement_verifier {
            LoopRefinementVerifierKind::Essential => "essential",
            LoopRefinementVerifierKind::Pnp => "pnp",
        },
        pgr_solver = match args.pose_graph_refinement_solver {
            LoopRefinementSolverKind::Se3 => "se3",
            LoopRefinementSolverKind::Sim3 => "sim3",
        },
        pgr_scale_spread = pose_graph_refinement_last_solve_scale_spread,
        pgr_candidates_seen = pose_graph_refinement_candidates_seen,
        pgr_verified_constraints = pose_graph_refinement_verified_constraints,
        pgr_pgo_solves = pose_graph_refinement_pgo_solves,
        pgr_gnc_rejected_loop_edges_max = pose_graph_refinement_gnc_rejected_loop_edges_max,
        pgr_pcm_rejected = pose_graph_refinement_pcm_rejected,
        pgr_covariance_rejected = pose_graph_refinement_covariance_rejected,
        pgr_pose_information_rejected = pose_graph_refinement_pose_information_rejected,
        pgr_with_pose_information = pose_graph_refinement_with_pose_information,
        pgr_pose_information_estimates = pose_graph_refinement_pose_information_estimates,
        pgr_pose_information_used_correspondences =
            pose_graph_refinement_pose_information_used_correspondences,
        pgr_pose_information_stereo_covariance_correspondences =
            pose_graph_refinement_pose_information_stereo_covariance_correspondences,
        pgr_pose_information_raw_condition_max =
            pose_graph_refinement_pose_information_raw_condition_max,
        pgr_pose_information_spectral_scale_min =
            pose_graph_refinement_pose_information_spectral_scale_min,
        pgr_pose_information_failure_counts =
            pose_graph_refinement_pose_information_failures,
        pgr_sequential_with_pose_information =
            pose_graph_refinement_sequential_with_pose_information,
        pgr_sequential_pose_information_fallbacks =
            pose_graph_refinement_sequential_pose_information_fallbacks,
        pgr_sequential_pose_information_used_correspondences =
            pose_graph_refinement_sequential_pose_information_used_correspondences,
        pgr_sequential_pose_information_stereo_covariance_correspondences =
            pose_graph_refinement_sequential_pose_information_stereo_covariance_correspondences,
        pgr_sequential_pose_information_raw_condition_max =
            pose_graph_refinement_sequential_pose_information_raw_condition_max,
        pgr_sequential_pose_information_spectral_scale_min =
            pose_graph_refinement_sequential_pose_information_spectral_scale_min,
        pgr_sequential_pose_information_failure_counts =
            pose_graph_refinement_sequential_pose_information_failures,
        pgr_appearance_enabled = args.pose_graph_refinement_appearance_loops,
        pgr_appearance_min_gap = args.pose_graph_refinement_appearance_min_gap,
        pgr_appearance_max_candidates = args.pose_graph_refinement_appearance_max_candidates,
        pgr_appearance_min_inliers = args.pose_graph_refinement_appearance_min_inliers,
        pgr_appearance_confirmation_keyframes =
            args.pose_graph_refinement_appearance_confirmation_keyframes,
        pgr_appearance_confirmation_max_misses =
            args.pose_graph_refinement_appearance_confirmation_max_misses,
        pgr_appearance_projection_radius =
            args.pose_graph_refinement_appearance_projection_radius_px,
        pgr_appearance_projection_min_matches =
            args.pose_graph_refinement_appearance_projection_min_matches,
        pgr_fuse_loop_observations = args.pose_graph_refinement_fuse_loop_observations,
        pgr_loop_fusion_pairs_considered = pose_graph_refinement_loop_fusion_pairs_considered,
        pgr_loop_fusion_observations_inserted =
            pose_graph_refinement_loop_fusion_observations_inserted,
        pgr_loop_fusion_observations_reassigned =
            pose_graph_refinement_loop_fusion_observations_reassigned,
        pgr_loop_fusion_pairs_skipped = pose_graph_refinement_loop_fusion_pairs_skipped,
        pgr_loop_fusion_pairs_robust_rejected =
            pose_graph_refinement_loop_fusion_pairs_robust_rejected,
        pgr_loop_fusion_pairs_reprojection_rejected =
            pose_graph_refinement_loop_fusion_pairs_reprojection_rejected,
        pgr_loop_welding_ba = args.pose_graph_refinement_loop_welding_ba,
        pgr_loop_welding_ba_attempted = pose_graph_refinement_loop_welding_ba_attempted,
        pgr_loop_welding_ba_succeeded = pose_graph_refinement_loop_welding_ba_succeeded,
        pgr_loop_welding_ba_rejected_or_failed =
            pose_graph_refinement_loop_welding_ba_rejected_or_failed,
        pgr_loop_welding_ba_updated_keyframes =
            pose_graph_refinement_loop_welding_ba_updated_keyframes,
        pgr_loop_welding_ba_updated_landmarks =
            pose_graph_refinement_loop_welding_ba_updated_landmarks,
        pgr_loop_welding_post_pgo_attempted =
            pose_graph_refinement_loop_welding_post_pgo_attempted,
        pgr_loop_welding_post_pgo_succeeded =
            pose_graph_refinement_loop_welding_post_pgo_succeeded,
        pgr_loop_welding_post_pgo_reprojection_max =
            pose_graph_refinement_loop_welding_post_pgo_reprojection_max,
        pgr_loop_welding_post_ba_last_error =
            pose_graph_refinement_loop_welding_post_ba_last_error,
        pgr_loop_welding_post_ba_behind_camera_ratio_max =
            pose_graph_refinement_loop_welding_post_ba_behind_camera_ratio_max,
        pgr_loop_welding_initial_translation_max =
            pose_graph_refinement_loop_welding_initial_translation_max,
        pgr_loop_welding_initial_rotation_max =
            pose_graph_refinement_loop_welding_initial_rotation_max,
        pgr_appearance_ranked = pose_graph_refinement_appearance_ranked,
        pgr_appearance_connected_region_rejected =
            pose_graph_refinement_appearance_connected_region_rejected,
        pgr_appearance_pnp_verified = pose_graph_refinement_appearance_pnp_verified,
        pgr_appearance_projection_rejected =
            pose_graph_refinement_appearance_projection_rejected,
        pgr_appearance_covisibility_rejected =
            pose_graph_refinement_appearance_covisibility_rejected,
        pgr_appearance_pending_projection_attempted =
            pose_graph_refinement_appearance_pending_projection_attempted,
        pgr_appearance_pending_projection_verified =
            pose_graph_refinement_appearance_pending_projection_verified,
        pgr_appearance_pending_projection_correspondences =
            pose_graph_refinement_appearance_pending_projection_correspondences,
        pgr_appearance_pending_projection_inliers =
            pose_graph_refinement_appearance_pending_projection_inliers,
        pgr_appearance_pending_projection_correspondences_max =
            pose_graph_refinement_appearance_pending_projection_correspondences_max,
        pgr_appearance_pending_projection_inliers_max =
            pose_graph_refinement_appearance_pending_projection_inliers_max,
        pgr_appearance_pending_projection_radius_min =
            pose_graph_refinement_appearance_pending_projection_radius_min,
        pgr_appearance_pending_projection_radius_max =
            pose_graph_refinement_appearance_pending_projection_radius_max,
        pgr_appearance_region_confirmation_waiting =
            pose_graph_refinement_appearance_region_confirmation_waiting,
        pgr_appearance_region_confirmed =
            pose_graph_refinement_appearance_region_confirmed,
        pgr_appearance_scale_failed = pose_graph_refinement_appearance_scale_failed,
        pgr_appearance_scale_insufficient_points =
            pose_graph_refinement_appearance_scale_insufficient_points,
        pgr_appearance_scale_insufficient_matches =
            pose_graph_refinement_appearance_scale_insufficient_matches,
        pgr_appearance_scale_no_consensus =
            pose_graph_refinement_appearance_scale_no_consensus,
        pgr_appearance_near_unit = pose_graph_refinement_appearance_near_unit,
        pgr_constraints_shared = pose_graph_refinement_constraints_shared,
        pgr_constraints_appearance = pose_graph_refinement_constraints_appearance,
        pgr_gt_evaluated = pose_graph_refinement_gt_evaluated,
        pgr_gt_correct = pose_graph_refinement_gt_correct,
        pgr_gt_precision = (pose_graph_refinement_gt_evaluated > 0).then(|| {
            pose_graph_refinement_gt_correct as f64
                / pose_graph_refinement_gt_evaluated as f64
        }),
        pgr_rejected_gt_evaluated = pose_graph_refinement_rejected_gt_evaluated,
        pgr_rejected_gt_correct = pose_graph_refinement_rejected_gt_correct,
        pgr_rejected_gt_correct_ratio = (pose_graph_refinement_rejected_gt_evaluated > 0).then(|| {
            pose_graph_refinement_rejected_gt_correct as f64
                / pose_graph_refinement_rejected_gt_evaluated as f64
        }),
        pgr_propagate = args.pose_graph_refinement_propagate,
        pgr_landmarks_moved = pose_graph_refinement_landmarks_moved,
        pgr_max_landmark_displacement_meters = pose_graph_refinement_max_landmark_displacement_meters,
        pgr_tracker_corrections_applied = pose_graph_refinement_tracker_corrections_applied,
        max_pose_jump = args.max_pose_jump_meters,
        pose_prior_visual_override = args.pose_prior_visual_override,
        pose_jump_gap_scaling = args.pose_jump_gap_scaling,
        pose_jump_gap_scaling_max_multiplier = args.pose_jump_gap_scaling_max_multiplier,
        tracking_min_inliers = args.tracking_min_inliers,
        tracking_min_inlier_ratio = args.tracking_min_inlier_ratio,
        tracking_max_reprojection_error = args.tracking_max_reprojection_error,
        pnp_warm_start = args.pnp_pose_prior_warm_start,
        projection_guided_tracking = args.projection_guided_tracking,
        projection_query_landmark_ratio = projection_guided_tracking_config
            .and_then(|config| config.max_query_landmark_distance_ratio),
        projection_search_radius_px = args.projection_search_radius_px,
        projection_widen_factor = args.projection_widen_factor,
        projection_max_widen_retries = args.projection_max_widen_retries,
        projection_local_map_refinement = !args.projection_no_local_map_refinement,
        projection_refinement_search_radius_px = args.projection_refinement_search_radius_px,
        projection_refinement_iterations = args.projection_refinement_iterations,
        projection_refinement_radius_shrink_factor = args
            .projection_refinement_radius_shrink_factor,
        projection_refinement_reassign_correspondences = args
            .projection_refinement_reassign_correspondences,
        projection_refinement_min_inlier_pair_retention_ratio = args
            .projection_refinement_min_inlier_pair_retention_ratio,
        projection_refinement_max_translation_correction_m = args
            .projection_refinement_max_translation_correction_m,
        projection_refinement_max_rotation_correction_deg = args
            .projection_refinement_max_rotation_correction_deg,
        projection_guided_attempt_count = slam.tracker.stats().projection_guided_attempt_count,
        projection_guided_widen_retry_count =
            slam.tracker.stats().projection_guided_widen_retry_count,
        projection_guided_success_count = slam.tracker.stats().projection_guided_success_count,
        projection_guided_fallback_success_count =
            slam.tracker.stats().projection_guided_fallback_success_count,
        local_map_refinement_correspondence_gain_total = slam
            .tracker
            .stats()
            .local_map_refinement_correspondence_gain_total,
        local_map_refinement_accepted_count =
            slam.tracker.stats().local_map_refinement_accepted_count,
        local_map_refinement_rejected_count =
            slam.tracker.stats().local_map_refinement_rejected_count,
        pnp_reproj_thresh = args.pnp_reprojection_threshold_px,
        motion_model_kind = match args.motion_model {
            MotionModelKind::Pose => "pose",
            MotionModelKind::Velocity => "velocity",
            MotionModelKind::ImuPredictive => "imu",
            MotionModelKind::AdaptiveImuPose => "adaptive-imu-pose",
        },
        imu_tbs = args.imu_extrinsic_from_cam0,
        imu_carry_fwd = args.imu_motion_model_carry_forward_velocity,
        adaptive_fail_thresh = args.adaptive_motion_failures_to_switch_to_pose,
        adaptive_succ_thresh = args.adaptive_motion_successes_to_switch_to_imu,
        adaptive_refresh_policy = match args.adaptive_motion_imu_velocity_refresh_policy {
            ImuVelocityRefreshPolicy::None => "none",
            ImuVelocityRefreshPolicy::FiniteDifference => "finite-diff",
            ImuVelocityRefreshPolicy::ZeroReset => "zero-reset",
            ImuVelocityRefreshPolicy::ThreePoseSmoother => "three-pose-smoother",
        },
        adaptive_switches_pose = slam
            .tracker
            .motion_model()
            .adaptive_stats()
            .map(|(p, _, _, _)| p as i64)
            .unwrap_or(-1),
        adaptive_switches_imu = slam
            .tracker
            .motion_model()
            .adaptive_stats()
            .map(|(_, i, _, _)| i as i64)
            .unwrap_or(-1),
        adaptive_velocity_refreshes = slam
            .tracker
            .motion_model()
            .adaptive_stats()
            .map(|(_, _, r, _)| r as i64)
            .unwrap_or(-1),
        adaptive_final_mode = match slam.tracker.motion_model().adaptive_stats() {
            Some((_, _, _, AdaptiveMotionMode::Imu)) => "imu",
            Some((_, _, _, AdaptiveMotionMode::Pose)) => "pose",
            None => "n/a",
        },
        cross_check = args.cross_check_matcher,
        mutual_softmax = args.mutual_softmax_matcher,
        feature_extractor_kind = match args.feature_extractor {
            FeatureExtractorKind::Corner => "corner",
            FeatureExtractorKind::Hog => "hog",
            FeatureExtractorKind::SuperPointOffline => "superpoint-offline",
            FeatureExtractorKind::SuperPointOnnx => "superpoint-onnx",
        },
        superpoint_features_dir = args.superpoint_features_dir,
        superpoint_cam1_features_dir = args.superpoint_cam1_features_dir,
        superpoint_onnx_model = args.superpoint_onnx_model,
        superpoint_onnx_backend = args.superpoint_onnx_backend.as_str(),
        kf_min_translation = args.keyframe_min_translation,
        kf_min_frame_gap = args.keyframe_min_frame_gap,
        kf_max_frame_gap = args.keyframe_max_frame_gap,
        kf_tracked_ratio = args.keyframe_tracked_landmark_ratio,
        kf_min_tracked_for_ratio = args.keyframe_min_tracked_landmarks_for_ratio,
        kf_min_inliers = args.keyframe_min_inliers,
        kf_min_inlier_ratio = args.keyframe_min_inlier_ratio,
        kf_insufficient_tracking_quality_rejections =
            keyframe_insufficient_tracking_quality_rejections,
        keep_pre_promotion = args.keep_pre_promotion_imu_factors,
        relinearise_thresholds = args.relinearise_imu_factor_bias_thresholds,
        run_at_promotion = args.run_local_vi_ba_at_vi_init_promotion,
        reloc_enabled = args.relocalization_enabled,
        reloc_min_inliers = args.relocalization_min_inliers,
        reloc_min_inlier_ratio = args.relocalization_min_inlier_ratio,
        reloc_max_rep_err = args.relocalization_max_reprojection_error,
        reloc_pose_prior_radius = args.relocalization_pose_prior_radius_meters,
        reloc_recent_kf_window = args.relocalization_recent_keyframe_window,
        reloc_covis_max_kf = args.relocalization_covisibility_max_keyframes,
        reloc_covis_min_shared = args.relocalization_covisibility_min_shared,
        reloc_covis_min_landmarks = args.relocalization_covisibility_min_landmarks,
        reloc_covis_broader_fallback = args.relocalization_covisibility_broader_fallback,
        reloc_covis_broader_fallback_interval_frames = args
            .relocalization_covisibility_broader_fallback_interval_frames,
        reloc_covis_compare_broader = args.relocalization_covisibility_compare_broader_store,
        reloc_appearance_max_kf = args.relocalization_appearance_max_keyframes,
        reloc_appearance_candidate_log_limit =
            args.relocalization_appearance_candidate_log_limit,
        reloc_appearance_min_similarity = args.relocalization_appearance_min_similarity,
        reloc_appearance_exclude_recent_frame_gap = args
            .relocalization_appearance_exclude_recent_frame_gap,
        reloc_appearance_min_landmarks = args.relocalization_appearance_min_landmarks,
        reloc_appearance_broader_fallback = args.relocalization_appearance_broader_fallback,
        reloc_appearance_broader_fallback_interval_frames = args
            .relocalization_appearance_broader_fallback_interval_frames,
        reloc_appearance_compare_broader = args.relocalization_appearance_compare_broader_store,
        reloc_max_tx_from_imu = args.relocalization_max_translation_from_imu_prediction_meters,
        reloc_attempt_interval_frames = args.relocalization_attempt_interval_frames,
        reloc_max_consecutive_failed_attempts =
            args.relocalization_max_consecutive_failed_attempts,
        reloc_max_tx_per_frame_from_last_success = args
            .relocalization_max_translation_per_frame_from_last_success_meters,
        reloc_min_depth_ratio_to_last_success = args
            .relocalization_min_inlier_depth_median_ratio_to_last_success,
        reloc_max_depth_ratio_to_last_success = args
            .relocalization_max_inlier_depth_median_ratio_to_last_success,
        reloc_confirmation_required = args.relocalization_confirmation_required_recoveries,
        reloc_confirmation_max_tx_per_frame = args
            .relocalization_confirmation_max_translation_per_frame_meters,
        reloc_attempts = relocalization_attempts,
        reloc_successes = relocalization_successes,
        rebootstrap_after_lost_frames = args.rebootstrap_after_lost_frames,
        rebootstrap_cooldown_frames = args.rebootstrap_cooldown_frames,
        rebootstrap_independent_submap = args.rebootstrap_independent_submap,
        atlas_broader_recovery_max_source_keyframes =
            args.atlas_broader_recovery_max_source_keyframes,
        atlas_broader_recovery_interval_attempts =
            args.atlas_broader_recovery_interval_attempts,
        atlas_broader_recovery_cycles = atlas_broader_recovery_cycles,
        atlas_boundary_keyframe_snapshots = atlas_boundary_keyframe_snapshots,
        atlas_boundary_factor_attempts = atlas_boundary_factor_attempts,
        atlas_boundary_factor_successes = atlas_boundary_factor_successes,
        rebootstrap_events = rebootstrap_events,
        segment_id = segment_id,
        reloc_gate_passes = relocalization_gate_passes,
        reloc_descriptor_store_count_observations =
            relocalization_descriptor_store_count_observations,
        reloc_descriptor_store_count_mean =
            if relocalization_descriptor_store_count_observations > 0 {
                Some(
                    relocalization_descriptor_store_count_sum as f64
                        / relocalization_descriptor_store_count_observations as f64,
                )
            } else {
                None
            },
        reloc_descriptor_store_count_min = relocalization_descriptor_store_count_min,
        reloc_descriptor_store_count_max = relocalization_descriptor_store_count_max,
        reloc_covis_descriptor_store_tried_frames =
            relocalization_covisibility_descriptor_store_tried_frames,
        reloc_covis_descriptor_store_used_frames =
            relocalization_covisibility_descriptor_store_used_frames,
        reloc_appearance_descriptor_store_tried_frames =
            relocalization_appearance_descriptor_store_tried_frames,
        reloc_appearance_descriptor_store_used_frames =
            relocalization_appearance_descriptor_store_used_frames,
        reloc_appearance_candidate_count_observations =
            relocalization_appearance_candidate_keyframe_count_observations,
        reloc_appearance_candidate_count_mean =
            if relocalization_appearance_candidate_keyframe_count_observations > 0 {
                Some(
                    relocalization_appearance_candidate_keyframe_count_sum as f64
                        / relocalization_appearance_candidate_keyframe_count_observations as f64,
                )
            } else {
                None
            },
        reloc_appearance_best_similarity_count =
            relocalization_appearance_best_similarity_count,
        reloc_appearance_best_similarity_mean =
            if relocalization_appearance_best_similarity_count > 0 {
                Some(
                    relocalization_appearance_best_similarity_sum
                        / relocalization_appearance_best_similarity_count as f64,
                )
            } else {
                None
            },
        reloc_appearance_best_similarity_max = relocalization_appearance_best_similarity_max,
        reloc_broader_descriptor_store_retry_frames =
            relocalization_broader_descriptor_store_retry_frames,
        reloc_broader_descriptor_store_retry_interval_skips =
            relocalization_broader_descriptor_store_retry_interval_skips,
        reloc_broader_descriptor_store_used_frames =
            relocalization_broader_descriptor_store_used_frames,
        reloc_budget_skips = slam
            .relocalization_state
            .as_ref()
            .map(|state| state.budget_skip_count)
            .unwrap_or(0),
        reloc_covis_reference_keyframe_count =
            relocalization_covisibility_reference_keyframe_count,
        reloc_confirmation_waiting = relocalization_confirmation_waiting,
        reloc_confirmation_tx_per_frame_count = relocalization_confirmation_tx_per_frame_count,
        reloc_confirmation_tx_per_frame_mean = if relocalization_confirmation_tx_per_frame_count
            > 0
        {
            Some(
                relocalization_confirmation_tx_per_frame_sum
                    / relocalization_confirmation_tx_per_frame_count as f64,
            )
        } else {
            None
        },
        reloc_confirmation_tx_per_frame_max = relocalization_confirmation_tx_per_frame_max,
        reloc_tx_per_frame_count = relocalization_tx_per_frame_count,
        reloc_tx_per_frame_mean = if relocalization_tx_per_frame_count > 0 {
            Some(
                relocalization_tx_per_frame_sum
                    / relocalization_tx_per_frame_count as f64,
            )
        } else {
            None
        },
        reloc_tx_per_frame_max = relocalization_tx_per_frame_max,
        reloc_success_tx_per_frame_count = relocalization_success_tx_per_frame_count,
        reloc_success_tx_per_frame_mean = if relocalization_success_tx_per_frame_count > 0 {
            Some(
                relocalization_success_tx_per_frame_sum
                    / relocalization_success_tx_per_frame_count as f64,
            )
        } else {
            None
        },
        reloc_success_tx_per_frame_max = relocalization_success_tx_per_frame_max,
        reloc_depth_ratio_count = relocalization_depth_ratio_count,
        reloc_depth_ratio_mean = if relocalization_depth_ratio_count > 0 {
            Some(relocalization_depth_ratio_sum / relocalization_depth_ratio_count as f64)
        } else {
            None
        },
        reloc_depth_ratio_min = relocalization_depth_ratio_min,
        reloc_depth_ratio_max = relocalization_depth_ratio_max,
        reloc_success_depth_ratio_count = relocalization_success_depth_ratio_count,
        reloc_success_depth_ratio_mean = if relocalization_success_depth_ratio_count > 0 {
            Some(
                relocalization_success_depth_ratio_sum
                    / relocalization_success_depth_ratio_count as f64,
            )
        } else {
            None
        },
        reloc_success_depth_ratio_min = relocalization_success_depth_ratio_min,
        reloc_success_depth_ratio_max = relocalization_success_depth_ratio_max,
        vi_init_try_every_frame = args.vi_init_try_initialize_on_every_frame,
        imu_factors_staged = imu_factors_staged,
        local_vi_ba_triggers = local_vi_ba_triggers,
        local_vi_ba_relinearised_factor_total = local_vi_ba_relinearised_factor_total,
        local_vi_ba_stereo_observation_total = local_vi_ba_stereo_observation_total,
        local_vi_ba_marginalization_priors_applied =
            local_vi_ba_marginalization_priors_applied,
        local_vi_ba_marginalization_successes = local_vi_ba_marginalization_successes,
        local_vi_ba_quality_gate_rejections = local_vi_ba_quality_gate_rejections,
        local_vi_ba_cost_ratio_gate_rejections = local_vi_ba_cost_ratio_gate_rejections,
        local_vi_ba_velocity_gate_rejections = local_vi_ba_velocity_gate_rejections,
        local_vi_ba_pose_correction_gate_rejections =
            local_vi_ba_pose_correction_gate_rejections,
        local_vi_ba_max_pose_translation_correction_meters =
            local_vi_ba_max_pose_translation_correction_meters,
        local_vi_ba_max_pose_rotation_correction_degrees =
            local_vi_ba_max_pose_rotation_correction_degrees,
        local_vi_ba_adaptive_velocity_gate_rejections =
            local_vi_ba_adaptive_velocity_gate_rejections,
        local_vi_ba_last_adaptive_velocity_threshold_mps =
            local_vi_ba_last_adaptive_velocity_threshold_mps,
        local_vi_ba_mirrors = local_vi_ba_mirrors,
        last_mirrored_v = last_mirrored_velocity_world.map(|v| [v.x, v.y, v.z]),
        last_mirrored_bg = last_mirrored_bias_gyro.map(|v| [v.x, v.y, v.z]),
        last_mirrored_ba = last_mirrored_bias_acc.map(|v| [v.x, v.y, v.z]),
        quality_gate_failures = slam.tracker.stats().tracking_quality_gate_failure_count,
    );
    println!("{summary}");
    fs::write(args.out_dir.join("summary.txt"), &summary)?;
    println!(
        "wrote {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} (+ summary.txt)",
        traj_path.display(),
        err_path.display(),
        frame_groundtruth_path.display(),
        vi_init_log_path.display(),
        motion_vi_init_log_path.display(),
        covisibility_local_ba_log_path.display(),
        local_vi_ba_diagnostics_path.display(),
        keyframe_decision_log_path.display(),
        relocalization_appearance_candidates_path.display(),
        relocalization_attempts_path.display(),
        keyframe_trajectory_path.display(),
        final_keyframe_errors_path.display(),
    );
    if rebootstrap_enabled {
        println!("wrote {}", rebootstrap_log_path.display());
    }
    if args.export_frame_appearance_descriptors {
        println!("wrote {}", frame_appearance_descriptors_path.display());
    }
    Ok(())
}

#[cfg(all(test, feature = "image-io"))]
mod tests {
    use super::*;

    #[test]
    fn superpoint_onnx_backend_arg_parses_strict_cuda_without_fallback() {
        assert_eq!(
            SuperPointOnnxBackendArg::parse("cuda"),
            Ok(SuperPointOnnxBackendArg::Cuda)
        );
        assert!(SuperPointOnnxBackendArg::parse("gpu").is_err());
    }

    fn fake_cam0() -> EurocCameraCalibration {
        EurocCameraCalibration {
            t_body_sensor: Matrix4::identity(),
            rate_hz: 20.0,
            resolution: (752, 480),
            camera_model: "pinhole".to_string(),
            intrinsics: [458.0, 457.0, 367.0, 248.0],
            distortion_model: "radial-tangential".to_string(),
            distortion_coefficients: vec![0.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn se3_from_t_bs_identity_round_trip() {
        let se3 = se3_from_t_bs(&Matrix4::identity());
        assert!((se3.rotation.angle()).abs() < 1.0e-12);
        assert!(se3.translation.norm() < 1.0e-12);
    }

    #[test]
    fn back_project_pixel_to_world_at_identity_camera_returns_camera_frame_point() {
        // World-to-camera = identity → camera centre at origin, looking
        // along +Z. The principal-point pixel back-projects to a world
        // point at (0, 0, depth).
        let camera = camera_from_cam0(&fake_cam0(), 1);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let principal_pixel = Point2::new(camera.params[2], camera.params[3]);
        let world =
            back_project_pixel_to_world(&camera, &pose, &principal_pixel, 5.0).expect("valid");
        assert!((world - Point3::new(0.0, 0.0, 5.0)).norm() < 1.0e-9);
    }

    #[test]
    fn back_project_then_project_round_trip_identity_camera() {
        // Back-projecting an arbitrary pixel and re-projecting through the
        // same camera must return the same pixel within floating-point
        // tolerance — the projection / unprojection are exact inverses
        // when the camera is identity-posed and the assumed depth is
        // honoured.
        let camera = camera_from_cam0(&fake_cam0(), 1);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let pixel = Point2::new(200.0, 150.0);
        let world = back_project_pixel_to_world(&camera, &pose, &pixel, 7.5).expect("valid");
        let p_cam = pose.transform_world_point(&world);
        let projected = camera.project(&p_cam).expect("in front of camera");
        assert!((projected.x - pixel.x).abs() < 1.0e-9);
        assert!((projected.y - pixel.y).abs() < 1.0e-9);
    }

    #[test]
    fn bootstrap_map_seeds_one_landmark_per_keypoint() {
        let camera = camera_from_cam0(&fake_cam0(), 1);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let features = FeatureSet::new(
            vec![Point2::new(300.0, 200.0), Point2::new(450.0, 250.0)],
            vec![vec![0.1, 0.2, 0.3], vec![-0.1, 0.0, 0.5]],
        )
        .expect("valid feature set");
        let map = bootstrap_map_from_first_frame(&camera, &pose, &features, 4.0, &[], &[], false);
        assert_eq!(map.landmarks.len(), 2);
        // Each seeded landmark must carry the source corner descriptor.
        for landmark in map.landmarks.values() {
            assert!(landmark.descriptor.is_some());
            assert_eq!(landmark.descriptor.as_ref().unwrap().len(), 3);
        }
    }

    #[test]
    fn frame_from_features_preserves_keypoint_and_descriptor_order() {
        let features = FeatureSet::new(
            vec![Point2::new(1.0, 2.0), Point2::new(3.0, 4.0)],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .expect("valid feature set");
        let frame = frame_from_features(42, 1, &features);
        assert_eq!(frame.id, 42);
        assert_eq!(frame.camera_id, 1);
        assert_eq!(frame.keypoints, features.keypoints);
        assert_eq!(frame.descriptors, features.descriptors);
    }

    #[test]
    fn atlas_boundary_snapshot_preserves_unpromoted_tracking_overlap() {
        let mut map = VisualMap::new();
        map.landmarks
            .insert(7, Landmark::new(7, Point3::new(0.0, 0.0, 4.0)));
        let mut frame = Frame::new(42, 1);
        frame.keypoints.push(Point2::new(320.0, 240.0));
        frame.descriptors.push(vec![1.0, 0.0]);
        frame.pose = Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::zeros(),
        ));
        let snapshot = Keyframe {
            frame,
            observations: vec![Observation {
                frame_id: 42,
                landmark_id: 7,
                keypoint_index: 0,
                xy: Point2::new(320.0, 240.0),
            }],
        };

        assert!(insert_atlas_boundary_keyframe_snapshot(&mut map, &snapshot));
        assert!(map.keyframes.contains_key(&42));
        assert!(map.landmarks[&7]
            .observations
            .iter()
            .any(|observation| observation.frame_id == 42));
        assert!(!insert_atlas_boundary_keyframe_snapshot(
            &mut map, &snapshot
        ));
    }

    #[test]
    fn normalized_mean_descriptor_averages_and_l2_normalizes() {
        let mean = normalized_mean_descriptor(&[vec![2.0, 0.0], vec![0.0, 2.0]])
            .expect("non-empty descriptor set");

        assert!((mean[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-6);
        assert!((mean[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-6);
        assert_eq!(normalized_mean_descriptor(&[]), None);
    }

    #[test]
    fn relocalization_attempt_reject_reason_reports_first_failed_gate() {
        let config = RelocalizationAttemptGateConfig {
            min_inliers: 20,
            min_inlier_ratio: 0.5,
            max_reprojection_error: Some(8.0),
            max_translation_per_frame_from_last_success_meters: None,
            min_inlier_depth_median_ratio_to_last_success: None,
            max_inlier_depth_median_ratio_to_last_success: None,
        };
        let stats = OnlineSlamRelocalizationStats {
            attempted: true,
            localization_success: true,
            inlier_count: 30,
            inlier_ratio: 0.8,
            mean_reprojection_error: Some(9.0),
            ..Default::default()
        };

        let gates = relocalization_attempt_gate_status(&stats, config);

        assert!(!gates.reprojection_pass);
        assert_eq!(
            relocalization_attempt_reject_reason(&stats, gates),
            "max_reprojection_error"
        );
    }

    #[test]
    fn relocalization_attempt_reject_reason_reports_confirmation_waiting() {
        let config = RelocalizationAttemptGateConfig {
            min_inliers: 20,
            min_inlier_ratio: 0.5,
            max_reprojection_error: Some(8.0),
            max_translation_per_frame_from_last_success_meters: None,
            min_inlier_depth_median_ratio_to_last_success: None,
            max_inlier_depth_median_ratio_to_last_success: None,
        };
        let stats = OnlineSlamRelocalizationStats {
            attempted: true,
            localization_success: true,
            inlier_count: 30,
            inlier_ratio: 0.8,
            mean_reprojection_error: Some(2.0),
            passed_acceptance_gates: true,
            confirmation_count: 1,
            confirmation_required_count: 2,
            ..Default::default()
        };

        let gates = relocalization_attempt_gate_status(&stats, config);

        assert_eq!(
            relocalization_attempt_reject_reason(&stats, gates),
            "confirmation_waiting"
        );
    }

    #[test]
    fn undistort_feature_keypoints_is_noop_under_identity_distortion() {
        let camera = camera_from_cam0(&fake_cam0(), 1);
        let features = FeatureSet::new(
            vec![Point2::new(100.0, 80.0), Point2::new(450.0, 320.0)],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        )
        .expect("valid feature set");
        let undistorted =
            undistort_feature_keypoints(&RadialTangential::IDENTITY, &camera, &features);
        assert_eq!(undistorted.keypoints, features.keypoints);
        assert_eq!(undistorted.descriptors, features.descriptors);
    }

    #[test]
    fn bootstrap_map_overrides_landmark_position_when_stereo_point_provided() {
        // When `stereo_world_points[i]` is `Some(...)`, that 3D point must
        // win over the fixed-depth back-projection for keypoint `i`. The
        // other keypoint must still fall back to the depth-based seeding.
        let camera = camera_from_cam0(&fake_cam0(), 1);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let principal_pixel = Point2::new(camera.params[2], camera.params[3]);
        let other_pixel = Point2::new(450.0, 250.0);
        let features = FeatureSet::new(
            vec![principal_pixel, other_pixel],
            vec![vec![0.1, 0.2], vec![-0.1, 0.0]],
        )
        .expect("valid feature set");
        // Override the first keypoint with an explicit world point.
        let override_point = Point3::new(7.0, -1.5, 9.0);
        let stereo_overrides = vec![Some(override_point), None];
        let covariance = nalgebra::Matrix3::from_diagonal_element(0.25);
        let stereo_covariances = vec![Some(covariance), None];
        let map = bootstrap_map_from_first_frame(
            &camera,
            &pose,
            &features,
            4.0,
            &stereo_overrides,
            &stereo_covariances,
            false,
        );
        assert_eq!(map.landmarks.len(), 2);
        // Landmark id = index + 1.
        let first = map.landmarks.get(&1).expect("override landmark seeded");
        assert!((first.position - override_point).norm() < 1.0e-9);
        assert_eq!(map.landmark_position_covariances.get(&1), Some(&covariance));
        let second = map.landmarks.get(&2).expect("fallback landmark seeded");
        // Fallback path: back-project the other pixel at depth 4.0 m and
        // confirm the seeded landmark sits there.
        let expected_fallback =
            back_project_pixel_to_world(&camera, &pose, &other_pixel, 4.0).unwrap();
        assert!((second.position - expected_fallback).norm() < 1.0e-9);
    }

    #[test]
    fn bootstrap_map_strict_stereo_drops_keypoints_without_triangulated_depth() {
        // Phase-23 #2: with `strict_stereo = true`, every keypoint that
        // has no corresponding stereo-triangulated 3D point is dropped
        // from the map instead of falling back to the fixed-depth
        // back-projection. On a 2-keypoint frame with one stereo
        // override and one without, the strict map must hold exactly
        // one landmark — the overridden one.
        let camera = camera_from_cam0(&fake_cam0(), 1);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let principal_pixel = Point2::new(camera.params[2], camera.params[3]);
        let other_pixel = Point2::new(450.0, 250.0);
        let features = FeatureSet::new(
            vec![principal_pixel, other_pixel],
            vec![vec![0.1, 0.2], vec![-0.1, 0.0]],
        )
        .expect("valid feature set");
        let override_point = Point3::new(7.0, -1.5, 9.0);
        let stereo_overrides = vec![Some(override_point), None];
        let stereo_covariances = vec![Some(nalgebra::Matrix3::identity()), None];
        let strict_map = bootstrap_map_from_first_frame(
            &camera,
            &pose,
            &features,
            4.0,
            &stereo_overrides,
            &stereo_covariances,
            true,
        );
        assert_eq!(strict_map.landmarks.len(), 1);
        let kept = strict_map
            .landmarks
            .get(&1)
            .expect("strict mode keeps the overridden landmark");
        assert!((kept.position - override_point).norm() < 1.0e-9);
        assert!(strict_map.landmarks.get(&2).is_none());
    }

    #[test]
    fn undistort_feature_keypoints_shifts_edge_pixels_under_euroc_distortion() {
        let camera = camera_from_cam0(&fake_cam0(), 1);
        // EuRoC MH_01_easy cam0 published coefficients.
        let distortion = RadialTangential {
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 0.0000176187114,
        };
        let principal = Point2::new(camera.params[2], camera.params[3]);
        let edge = Point2::new(10.0, 10.0);
        let features = FeatureSet::new(vec![principal, edge], vec![vec![0.0], vec![1.0]])
            .expect("valid feature set");
        let undistorted = undistort_feature_keypoints(&distortion, &camera, &features);
        assert_eq!(undistorted.keypoints.len(), 2);
        assert_eq!(undistorted.descriptors, features.descriptors);
        // Principal-point pixel barely moves; corner pixel shifts by
        // several pixels under the EuRoC distortion magnitude.
        let principal_shift = (undistorted.keypoints[0].coords - principal.coords).norm();
        let edge_shift = (undistorted.keypoints[1].coords - edge.coords).norm();
        assert!(
            principal_shift < 0.01,
            "principal shift = {principal_shift}"
        );
        assert!(edge_shift > 5.0, "edge shift = {edge_shift}");
    }

    #[test]
    fn stereo_segment_restart_tracking_result_marks_all_stereo_matches_inliers() {
        let camera = camera_from_cam0(&fake_cam0(), 1);
        let pose = Pose::identity();
        let features = FeatureSet::new(
            vec![Point2::new(100.0, 120.0), Point2::new(200.0, 220.0)],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .expect("valid feature set");
        let matches = vec![
            StereoBootstrapLandmark {
                left_keypoint_index: 0,
                right_keypoint_index: 0,
                point_left_camera_frame: Point3::new(0.1, 0.2, 2.0),
                left_reprojection_error_pixels: 0.5,
                right_reprojection_error_pixels: 0.5,
                point_covariance_left_camera_frame: nalgebra::Matrix3::identity(),
            },
            StereoBootstrapLandmark {
                left_keypoint_index: 1,
                right_keypoint_index: 1,
                point_left_camera_frame: Point3::new(-0.2, 0.1, 3.0),
                left_reprojection_error_pixels: 0.4,
                right_reprojection_error_pixels: 0.4,
                point_covariance_left_camera_frame: nalgebra::Matrix3::identity(),
            },
        ];
        let mut map = VisualMap::new();
        map.cameras.insert(camera.id, camera);
        let mut next_id = 100_u64;
        let landmark_ids = append_stereo_bootstrap_landmarks_to_map(
            &mut map,
            &pose,
            &features,
            &matches,
            &mut next_id,
        );
        let stats = map_provider_stats_from_map(&map);
        let tracking =
            build_stereo_segment_restart_tracking_result(42, pose, &matches, &landmark_ids, stats);
        assert!(tracking.localization.success);
        assert_eq!(tracking.localization.inlier_count, 2);
        assert_eq!(tracking.localization.inlier_landmark_ids, landmark_ids);
    }
}
