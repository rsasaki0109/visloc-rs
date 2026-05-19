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
//!   initial visual map is bootstrapped by back-projecting the first frame's
//!   real corners through the first GT camera pose at a fixed depth — so the
//!   landmarks live in the **metric** EuRoC world frame even though their
//!   depth is approximate. From that point on, both the corner detection and
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
//!   depth / reprojection-error gates) still fall back to the depth-only
//!   seed, so the matcher's input always has the same keypoint count as
//!   the cam0 extractor produced. Pass `--no-stereo-bootstrap` to skip
//!   the stereo pass entirely and reproduce the pre-stereo behaviour for
//!   A/B comparison. `--bootstrap-depth` is still honoured for the
//!   fall-back corners.
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
use nalgebra::{Matrix4, Point2, Point3, UnitQuaternion, Vector3};
#[cfg(feature = "image-io")]
use visloc_rs::core::geometry::{Pose, SE3};
#[cfg(feature = "image-io")]
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
#[cfg(feature = "image-io")]
use visloc_rs::io::euroc::{
    read_euroc_dataset_dir, EurocCameraCalibration, EurocGroundTruthSample,
};
#[cfg(feature = "image-io")]
use visloc_rs::io::images::read_common_image;
#[cfg(feature = "image-io")]
use visloc_rs::slam::OnlineSlamImuConfig;
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
    read_external_deep_features_txt, umeyama_similarity_transform, AdaptiveImuPoseMotionModel,
    AdaptiveImuPoseMotionModelConfig, AdaptiveMotionMode, BruteForceMatcher,
    ConstantPoseMotionModel, ConstantVelocityMotionModel, CovisibilityLocalMapConfig,
    CrossCheckMatcher, DescriptorMatch, ImuPredictiveMotionModel, ImuPredictiveMotionModelConfig,
    ImuVelocityRefreshPolicy, KeyframePolicyConfig, LocalMappingPipeline, LocalizationConfig,
    LocalizationPipeline, LoopClosureConfig, Matcher, MotionBasedViInitializerConfig, MotionModel,
    MotionViInitializationEvent, MutualSoftmaxConfig, MutualSoftmaxMatcher, OnlineSlamConfig,
    OnlineSlamLocalBaConfig, OnlineSlamMotionViInitConfig, OnlineSlamPipeline,
    OnlineSlamViInitConfig, SimpleKeyframePolicy, Tracker, TrackingConfig, TrackingResult,
    TrajectorySimilarityTransform, ViInitFallback, ViInitializationEvent, Viba2Config,
    VisualInertialInitializerConfig,
};

/// Runtime-dispatched motion model. Both inner models implement
/// [`MotionModel`], but `Tracker<P, M>` is generic in `M` and Rust's
/// monomorphisation forces a single concrete `M` for the lifetime of
/// the `Tracker`. Wrap the two stock models in an enum so the demo
/// can pick between them via a CLI flag without duplicating the
/// downstream pipeline construction.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone)]
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
    /// metric-scale landmark, replacing the fixed-`bootstrap_depth`
    /// back-projection for that corner. Set to `false` via
    /// `--no-stereo-bootstrap` to fall back to depth-only seeding for
    /// A/B comparison.
    stereo_bootstrap: bool,
    /// Enable the motion-based VI init stage (VIBA1 / optional VIBA2).
    /// Off by default so the demo stays backwards compatible with the
    /// existing baseline. The stage is gated on the static VI init
    /// completing first.
    motion_vi_init_enabled: bool,
    /// Minimum keyframes (post-static-seed) before the motion-VI trigger
    /// is allowed to fire. Mirrors
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
    /// When `Some(v)`, runs the local VI-BA conditioning fallback:
    /// after the joint solve, if `final_cost / initial_cost > v`, the
    /// window is re-solved with biases gauge-frozen and the bias
    /// updates are discarded. `None` (the default) preserves the legacy
    /// always-update-biases behaviour.
    local_vi_ba_freeze_biases_above: Option<f64>,
    /// When `Some(v)`, runs the motion-VI post-solve velocity sanity
    /// gate: if any per-keyframe `||velocity_world|| > v`, the inner LM
    /// result is rejected and the stage stays in `Waiting`. `None`
    /// (the default) preserves legacy behaviour. EuRoC V1-class indoor
    /// sequences run safely at `Some(10.0)`.
    motion_vi_init_max_velocity_mps: Option<f64>,
    /// When `Some(n)`, restricts descriptor matching during tracking to
    /// landmarks observed by the reference keyframe and up to `n`
    /// co-visible neighbour keyframes (ranked by shared-landmark count).
    /// `None` (the default) preserves legacy whole-map matching.
    covisibility_local_map_max_keyframes: Option<usize>,
    /// Minimum shared-landmark count required for a candidate keyframe
    /// to enter the covisibility-derived local map. Only used when
    /// `covisibility_local_map_max_keyframes` is set.
    covisibility_local_map_min_shared: usize,
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
    let mut vi_init_accel_std_limit: Option<f64> = None;
    let mut bootstrap_depth_meters: f64 = 4.0;
    let mut corner_max_features: usize = 1500;
    let mut corner_min_score: f32 = 0.02;
    let mut corner_descriptor_radius: usize = 5;
    let mut undistort: bool = true;
    let mut stereo_bootstrap: bool = true;
    let mut stereo_bootstrap_strict: bool = false;
    let mut adaptive_motion_failures_to_switch_to_pose: usize = 2;
    let mut adaptive_motion_successes_to_switch_to_imu: usize = 5;
    let mut adaptive_motion_imu_velocity_refresh_policy: ImuVelocityRefreshPolicy =
        ImuVelocityRefreshPolicy::default();
    let mut motion_vi_init_enabled: bool = false;
    let mut motion_vi_init_min_keyframes: usize = 10;
    let mut motion_vi_init_min_translation_meters: f64 = 2.0;
    let mut motion_vi_init_recover_scale: bool = false;
    let mut local_vi_ba_enabled: bool = false;
    let mut local_vi_ba_freeze_biases_above: Option<f64> = None;
    let mut motion_vi_init_max_velocity_mps: Option<f64> = None;
    let mut covisibility_local_map_max_keyframes: Option<usize> = None;
    let mut covisibility_local_map_min_shared: usize = 15;
    let mut max_pose_jump_meters: Option<f64> = None;
    let mut pnp_pose_prior_warm_start: bool = false;
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
    let mut vi_init_min_samples: Option<usize> = None;
    let mut vi_init_min_stationary_window_seconds: Option<f64> = None;
    let mut keep_pre_promotion_imu_factors: bool = false;
    let mut relinearise_imu_factor_bias_thresholds: Option<(f64, f64)> = None;
    let mut superpoint_features_dir: Option<PathBuf> = None;
    let mut superpoint_cam1_features_dir: Option<PathBuf> = None;
    let mut superpoint_onnx_model: Option<PathBuf> = None;
    let mut run_local_vi_ba_at_vi_init_promotion: bool = false;
    let mut relocalization_enabled: bool = false;
    let mut relocalization_min_inliers: usize = 20;
    let mut relocalization_min_inlier_ratio: f64 = 0.3;
    let mut relocalization_max_reprojection_error: Option<f64> = Some(8.0);
    let mut relocalization_pose_prior_radius_meters: Option<f64> = None;
    let mut relocalization_recent_keyframe_window: Option<usize> = None;
    let mut relocalization_max_translation_from_imu_prediction_meters: Option<f64> = None;

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
            "--local-vi-ba-freeze-biases-above" => {
                local_vi_ba_freeze_biases_above = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-velocity" => {
                motion_vi_init_max_velocity_mps = Some(args.remove(i + 1).parse()?);
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
            "--max-pose-jump-meters" => {
                max_pose_jump_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pnp-pose-prior-warm-start" => {
                pnp_pose_prior_warm_start = true;
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
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let euroc_dir = euroc_dir.ok_or("--euroc-dir <path/to/MH_01_easy> is required")?;
    if bootstrap_depth_meters <= 0.0 {
        return Err("--bootstrap-depth must be positive".into());
    }
    Ok(CliArgs {
        euroc_dir,
        out_dir,
        max_frames,
        gravity_world,
        vi_init_max_wait_seconds,
        vi_init_try_initialize_on_every_frame,
        vi_init_gyro_std_limit,
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
        motion_vi_init_min_keyframes,
        motion_vi_init_min_translation_meters,
        motion_vi_init_recover_scale,
        local_vi_ba_enabled,
        local_vi_ba_freeze_biases_above,
        motion_vi_init_max_velocity_mps,
        covisibility_local_map_max_keyframes,
        covisibility_local_map_min_shared,
        max_pose_jump_meters,
        pnp_pose_prior_warm_start,
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
        vi_init_min_samples,
        vi_init_min_stationary_window_seconds,
        keep_pre_promotion_imu_factors,
        relinearise_imu_factor_bias_thresholds,
        superpoint_features_dir,
        superpoint_cam1_features_dir,
        superpoint_onnx_model,
        run_local_vi_ba_at_vi_init_promotion,
        relocalization_enabled,
        relocalization_min_inliers,
        relocalization_min_inlier_ratio,
        relocalization_max_reprojection_error,
        relocalization_pose_prior_radius_meters,
        relocalization_recent_keyframe_window,
        relocalization_max_translation_from_imu_prediction_meters,
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

/// Build the world-to-camera [`Pose`] implied by a GT body pose and the
/// body-to-camera rig calibration.
#[cfg(feature = "image-io")]
fn world_to_camera_pose(
    body_rotation_world: &UnitQuaternion<f64>,
    body_position_world: &Vector3<f64>,
    body_to_camera: &SE3,
) -> Pose {
    let r_wc = body_rotation_world * body_to_camera.rotation;
    let camera_center_world =
        body_position_world + body_rotation_world.transform_vector(&body_to_camera.translation);
    let r_cw = r_wc.inverse();
    let t_cw = -(r_cw.transform_vector(&camera_center_world));
    Pose::from_world_to_camera(r_cw, t_cw)
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
    strict_stereo: bool,
) -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let use_overrides = !stereo_world_points.is_empty();
    if use_overrides {
        debug_assert_eq!(stereo_world_points.len(), features.keypoints.len());
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
        map.landmarks.insert(landmark.id, landmark);
    }
    map
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
fn nearest_ground_truth(
    samples: &[EurocGroundTruthSample],
    target_ts: i128,
) -> &EurocGroundTruthSample {
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
    &samples[idx]
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

#[cfg(feature = "image-io")]
fn format_motion_vi_init_event(event: &MotionViInitializationEvent) -> String {
    match event {
        MotionViInitializationEvent::StillWaiting { reason } => {
            format!("StillWaiting reason={reason:?}")
        }
        MotionViInitializationEvent::Succeeded { result } => format!(
            "Succeeded keyframes={} imu_factors={} scale={:.6} viba2_iters={} trigger_translation_m={:.3}",
            result.keyframe_ids.len(),
            result.imu_factors_used,
            result.scale,
            result.viba2_iterations_run,
            result.trigger_translation_meters,
        ),
    }
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    if dataset.ground_truth.is_empty() {
        return Err(format!(
            "ground truth missing under {}/mav0/state_groundtruth_estimate0/data.csv",
            args.euroc_dir.display()
        )
        .into());
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
    let seed_gt = dataset
        .ground_truth
        .iter()
        .find(|gt| {
            gt.timestamp_nanoseconds >= imu_first_ts && gt.timestamp_nanoseconds <= imu_last_ts
        })
        .ok_or("ground truth and IMU streams do not overlap")?
        .clone();

    // Locate the first cam0 frame whose timestamp is on or after `seed_gt`.
    // That frame is decoded and feature-extracted to seed the map.
    let seed_frame_idx = dataset
        .cam0_images
        .iter()
        .position(|entry| entry.timestamp_nanoseconds >= seed_gt.timestamp_nanoseconds)
        .ok_or("no cam0 frame at or after the GT seed timestamp")?;
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
            let extractor = visloc_vision::features::superpoint_onnx::SuperPointOnnxExtractor::load_from_path(
                model_path,
                visloc_vision::features::superpoint_onnx::SuperPointOnnxConfig::default(),
            )
            .map_err(|err| format!("SuperPoint ONNX load_from_path failed: {err}"))?;
            println!(
                "loaded SuperPoint ONNX model from {} (onnx-inference feature: {})",
                model_path.display(),
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

    let seed_pose = world_to_camera_pose(
        &seed_gt.orientation_world,
        &seed_gt.position_world,
        &body_to_camera,
    );

    // Optional stereo bootstrap: pair the seed cam0 frame with the nearest
    // cam1 frame, match descriptors, and triangulate each surviving pair
    // via DLT in the cam0 frame. The resulting 3D points are transformed
    // into the world frame using `seed_pose` and override the
    // fixed-`bootstrap_depth` back-projection for the matched cam0
    // keypoints below.
    let mut stereo_world_points: Vec<Option<Point3<f64>>> =
        vec![None; seed_features.keypoints.len()];
    let mut stereo_bootstrap_matches: Vec<StereoBootstrapLandmark> = Vec::new();
    let mut stereo_cam1_features_count: usize = 0;
    let mut stereo_cam1_features_after_undistort_count: usize = 0;
    if args.stereo_bootstrap {
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
                    "cam1 distortion_coefficients has unexpected length ({}); expected 4 for radial-tangential. Pass --no-undistort (or --no-stereo-bootstrap) to skip.",
                    dataset.cam1_calibration.distortion_coefficients.len(),
                )
            })?
        } else {
            RadialTangential::IDENTITY
        };
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
            undistort_feature_keypoints(&cam1_distortion, &cam1_camera, &cam1_features_raw);
        stereo_cam1_features_after_undistort_count = cam1_features.len();
        stereo_bootstrap_matches = bootstrap_stereo_landmarks(
            &camera,
            &cam1_camera,
            &cam0_to_cam1,
            &seed_features,
            &cam1_features,
            &StereoBootstrapConfig::default(),
        );
        let cam0_pose_camera_to_world = seed_pose.camera_to_world();
        for survivor in &stereo_bootstrap_matches {
            let world_point =
                cam0_pose_camera_to_world.transform_point(&survivor.point_left_camera_frame);
            stereo_world_points[survivor.left_keypoint_index] = Some(world_point);
        }
        println!(
            "stereo_bootstrap cam1_seed_idx={cam1_seed_idx} cam1_features={} after_undistort={} triangulated_matches={}",
            stereo_cam1_features_count,
            stereo_cam1_features_after_undistort_count,
            stereo_bootstrap_matches.len(),
        );
    }

    let map = bootstrap_map_from_first_frame(
        &camera,
        &seed_pose,
        &seed_features,
        args.bootstrap_depth_meters,
        &stereo_world_points,
        args.stereo_bootstrap_strict,
    );
    println!(
        "bootstrap landmarks={} bootstrap_depth={:.2}m stereo_overrides={} gt_seed_t_ns={} body_position=[{:.3},{:.3},{:.3}]",
        map.landmarks.len(),
        args.bootstrap_depth_meters,
        stereo_bootstrap_matches.len(),
        seed_gt.timestamp_nanoseconds,
        seed_gt.position_world.x,
        seed_gt.position_world.y,
        seed_gt.position_world.z,
    );

    let mut initializer_config = VisualInertialInitializerConfig {
        gravity_world: args.gravity_world,
        ..VisualInertialInitializerConfig::default()
    };
    if let Some(limit) = args.vi_init_gyro_std_limit {
        initializer_config.max_gyro_std = limit;
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
        Some(OnlineSlamMotionViInitConfig {
            initializer: MotionBasedViInitializerConfig {
                min_keyframes: args.motion_vi_init_min_keyframes,
                min_translation_meters: args.motion_vi_init_min_translation_meters,
                gravity_world: args.gravity_world,
                viba2,
                max_velocity_magnitude_mps: args.motion_vi_init_max_velocity_mps,
                ..MotionBasedViInitializerConfig::default()
            },
            ..OnlineSlamMotionViInitConfig::default()
        })
    } else {
        None
    };
    let local_vi_ba_config = if args.local_vi_ba_enabled {
        Some(OnlineSlamLocalBaConfig {
            gravity_world: args.gravity_world,
            freeze_biases_when_cost_ratio_above: args.local_vi_ba_freeze_biases_above,
            relinearise_imu_factor_bias_thresholds: args.relinearise_imu_factor_bias_thresholds,
            run_at_vi_init_promotion: args.run_local_vi_ba_at_vi_init_promotion,
            ..OnlineSlamLocalBaConfig::default()
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
    let tracking_config = TrackingConfig {
        covisibility_local_map: covisibility_config,
        max_pose_prior_translation_error: args.max_pose_jump_meters,
        pnp_pose_prior_warm_start: args.pnp_pose_prior_warm_start,
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
    let localization_pipeline = LocalizationPipeline::new(demo_matcher, localization_config);
    let keyframe_policy_config = {
        let mut cfg = KeyframePolicyConfig::default();
        if let Some(m) = args.keyframe_min_translation {
            cfg.min_translation = m;
        }
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
            loop_closure: LoopClosureConfig::default(),
            imu: Some(imu_config),
            local_vi_ba: local_vi_ba_config,
            vi_init: Some(vi_init_config),
            vi_motion_init: vi_motion_init_config,
            keep_pre_promotion_imu_factors: args.keep_pre_promotion_imu_factors,
            pose_graph_refinement: None,
            relocalization: if args.relocalization_enabled {
                Some(visloc_rs::OnlineSlamRelocalizationConfig {
                    min_inliers: args.relocalization_min_inliers,
                    min_inlier_ratio: args.relocalization_min_inlier_ratio,
                    max_mean_reprojection_error: args.relocalization_max_reprojection_error,
                    pose_prior_candidate_radius_meters: args
                        .relocalization_pose_prior_radius_meters,
                    recent_keyframe_window: args.relocalization_recent_keyframe_window,
                    max_translation_from_imu_prediction_meters: args
                        .relocalization_max_translation_from_imu_prediction_meters,
                })
            } else {
                None
            },
        },
    );

    fs::create_dir_all(&args.out_dir)?;
    let traj_path = args.out_dir.join("slam_trajectory.csv");
    let err_path = args.out_dir.join("slam_errors.csv");
    let vi_init_log_path = args.out_dir.join("vi_init_log.txt");
    let motion_vi_init_log_path = args.out_dir.join("motion_vi_init_log.txt");

    let mut traj_csv =
        String::from("timestamp_ns,frame_idx,px,py,pz,qw,qx,qy,qz,tracking_success\n");
    let mut err_csv = String::from(
        "timestamp_ns,frame_idx,gt_px,gt_py,gt_pz,est_px,est_py,est_pz,position_error_m,orientation_error_deg\n",
    );
    let mut vi_init_log = String::new();
    let mut motion_vi_init_log = String::new();

    let frame_cap = if args.max_frames == 0 {
        usize::MAX
    } else {
        args.max_frames
    };

    let mut imu_idx = 0usize;
    let mut prev_imu_ts = imu_first_ts;
    while imu_idx < dataset.imu_samples.len()
        && dataset.imu_samples[imu_idx].timestamp_nanoseconds < seed_gt.timestamp_nanoseconds
    {
        prev_imu_ts = dataset.imu_samples[imu_idx].timestamp_nanoseconds;
        imu_idx += 1;
    }

    let mut frames_recorded = 0usize;
    let mut tracking_successes = 0usize;
    let mut relocalization_attempts = 0u64;
    let mut relocalization_successes = 0u64;
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
    let mut last_mirrored_velocity_world: Option<Vector3<f64>> = None;
    let mut last_mirrored_bias_gyro: Option<Vector3<f64>> = None;
    let mut last_mirrored_bias_acc: Option<Vector3<f64>> = None;

    for (frame_idx, image_entry) in dataset.cam0_images.iter().enumerate().skip(seed_frame_idx) {
        if frames_recorded >= frame_cap {
            break;
        }
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
        let frame = frame_from_features(frame_idx as u64, camera_id, &features);

        let result = slam.process_frame(&frame, []);
        let success = result.tracking_succeeded();
        // Treat quality-gate rejections as "no estimate this frame": their
        // `localization.pose` still carries the rejected PnP result (the
        // tracker only flips `success = false`), which would otherwise leak
        // a known-bad pose into the trajectory CSV and ATE summation.
        let tracked = if success {
            result.tracking.localization.pose.clone()
        } else {
            None
        };
        if success {
            tracking_successes += 1;
        }
        if let Some(reloc) = result.relocalization.as_ref() {
            if reloc.attempted {
                relocalization_attempts += 1;
            }
            if reloc.succeeded {
                relocalization_successes += 1;
            }
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
            if !stats.bias_frozen {
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
            let entry = format!(
                "frame_idx={frame_idx} timestamp_ns={} {}\n",
                image_entry.timestamp_nanoseconds,
                format_motion_vi_init_event(event),
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
                }
            }
        }

        let (estimated_center, estimated_rotation_wc) = if let Some(pose) = tracked.as_ref() {
            let center = pose.camera_center_world();
            let rotation_wc = pose.world_to_camera.rotation.inverse();
            (Some(center), Some(rotation_wc))
        } else {
            (None, None)
        };

        let gt = nearest_ground_truth(&dataset.ground_truth, image_entry.timestamp_nanoseconds);
        let (gt_center_x, gt_center_y, gt_center_z) = {
            let camera_center_world = gt.position_world
                + gt.orientation_world
                    .transform_vector(&body_to_camera.translation);
            (
                camera_center_world.x,
                camera_center_world.y,
                camera_center_world.z,
            )
        };

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
                if success { 1 } else { 0 },
            ));

            let gt_center = Vector3::new(gt_center_x, gt_center_y, gt_center_z);
            let position_error = (Vector3::new(center.x, center.y, center.z) - gt_center).norm();
            let gt_rotation_wc = gt.orientation_world * body_to_camera.rotation;
            let orientation_error_deg = q.rotation_to(&gt_rotation_wc).angle().to_degrees();
            err_csv.push_str(&format!(
                "{},{frame_idx},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                image_entry.timestamp_nanoseconds,
                gt_center_x,
                gt_center_y,
                gt_center_z,
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
            reference_positions.push(Point3::new(gt_center_x, gt_center_y, gt_center_z));
            error_samples += 1;
        } else {
            traj_csv.push_str(&format!(
                "{},{frame_idx},,,,,,,,{}\n",
                image_entry.timestamp_nanoseconds,
                if success { 1 } else { 0 },
            ));
        }

        frames_recorded += 1;
    }

    fs::write(&traj_path, traj_csv)?;
    fs::write(&err_path, err_csv)?;
    fs::write(&vi_init_log_path, &vi_init_log)?;
    fs::write(&motion_vi_init_log_path, &motion_vi_init_log)?;

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
    let final_vi_status = slam.vi_initialization_status();
    let final_motion_vi_status = slam.motion_vi_initialization_status();
    let map_keyframes = slam.map().keyframes.len();
    let map_landmarks = slam.map().landmarks.len();
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

    let summary = format!(
        "euroc_dir={}\n\
         frames_recorded={frames_recorded}\n\
         tracking_success_rate={success_rate:.3}\n\
         imu_samples_consumed={imu_idx}\n\
         seed_frame_idx={seed_frame_idx}\n\
         undistort={undistort}\n\
         stereo_bootstrap={stereo_bootstrap_enabled}\n\
         stereo_bootstrap_strict={stereo_bootstrap_strict}\n\
         stereo_bootstrap_cam1_features={stereo_cam1_features}\n\
         stereo_bootstrap_cam1_features_after_undistort={stereo_cam1_features_after_undistort}\n\
         stereo_bootstrap_matches={stereo_bootstrap_matches_count}\n\
         bootstrap_depth_meters={bootstrap_depth:.3}\n\
         bootstrap_landmarks={bootstrap_landmarks}\n\
         feature_count_mean={mean_features:.1}\n\
         feature_count_min={feature_count_min}\n\
         feature_count_max={feature_count_max}\n\
         map_keyframes={map_keyframes}\n\
         map_landmarks={map_landmarks}\n\
         vi_init_first_event_frame={vi_first:?}\n\
         vi_init_succeeded_frame={vi_succeeded:?}\n\
         vi_init_status_final={final_vi_status:?}\n\
         motion_vi_init_enabled={motion_enabled}\n\
         motion_vi_init_first_event_frame={motion_first:?}\n\
         motion_vi_init_succeeded_frame={motion_succeeded:?}\n\
         motion_vi_init_recovered_scale={motion_scale:?}\n\
         motion_vi_init_viba2_iterations={motion_iters:?}\n\
         motion_vi_init_status_final={final_motion_vi_status:?}\n\
         local_vi_ba_enabled={local_vi_ba_enabled}\n\
         local_vi_ba_freeze_biases_above={local_vi_ba_freeze:?}\n\
         motion_vi_init_max_velocity_mps={motion_vi_max_vel:?}\n\
         covisibility_local_map_max_keyframes={covisibility_max_kf:?}\n\
         covisibility_local_map_min_shared={covisibility_min_shared}\n\
         covisibility_local_map_used_frames={covisibility_used_frames}\n\
         covisibility_local_map_mean_size={covisibility_mean_size:.2}\n\
         max_pose_jump_meters={max_pose_jump:?}\n\
         pnp_pose_prior_warm_start={pnp_warm_start}\n\
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
         keyframe_min_translation={kf_min_translation:?}\n\
         keep_pre_promotion_imu_factors={keep_pre_promotion}\n\
         relinearise_imu_factor_bias_thresholds={relinearise_thresholds:?}\n\
         run_local_vi_ba_at_vi_init_promotion={run_at_promotion}\n\
         relocalization_enabled={reloc_enabled}\n\
         relocalization_min_inliers={reloc_min_inliers}\n\
         relocalization_min_inlier_ratio={reloc_min_inlier_ratio}\n\
         relocalization_max_mean_reprojection_error={reloc_max_rep_err:?}\n\
         relocalization_pose_prior_radius={reloc_pose_prior_radius:?}\n\
         relocalization_recent_keyframe_window={reloc_recent_kf_window:?}\n\
         relocalization_max_translation_from_imu_prediction_meters={reloc_max_tx_from_imu:?}\n\
         relocalization_attempts={reloc_attempts}\n\
         relocalization_successes={reloc_successes}\n\
         vi_init_try_initialize_on_every_frame={vi_init_try_every_frame}\n\
         imu_factors_staged={imu_factors_staged}\n\
         local_vi_ba_triggers={local_vi_ba_triggers}\n\
         local_vi_ba_relinearised_factor_total={local_vi_ba_relinearised_factor_total}\n\
         local_vi_ba_mirrors_into_imu_motion_model={local_vi_ba_mirrors}\n\
         last_mirrored_velocity_world={last_mirrored_v:?}\n\
         last_mirrored_bias_gyro={last_mirrored_bg:?}\n\
         last_mirrored_bias_acc={last_mirrored_ba:?}\n\
         tracking_quality_gate_failures={quality_gate_failures}\n\
         ate_position_rmse_m={rmse_pos:.4}\n\
         ate_position_max_m={max_position_err:.4}\n\
         ate_orientation_rmse_deg={rmse_rot_deg:.4}\n\
         ate_orientation_max_deg={max_orientation_err_deg:.4}\n\
         ate_rigid_rmse_m={ate_rmse_rigid:.4}\n\
         ate_rigid_max_m={max_rigid:.4}\n\
         ate_similarity_rmse_m={ate_rmse_sim:.4}\n\
         ate_similarity_max_m={max_sim:.4}\n\
         ate_similarity_scale={scale:.6}\n",
        args.euroc_dir.display(),
        success_rate = if frames_recorded > 0 {
            tracking_successes as f64 / frames_recorded as f64
        } else {
            0.0
        },
        undistort = args.undistort,
        stereo_bootstrap_enabled = args.stereo_bootstrap,
        stereo_bootstrap_strict = args.stereo_bootstrap_strict,
        stereo_cam1_features = stereo_cam1_features_count,
        stereo_cam1_features_after_undistort = stereo_cam1_features_after_undistort_count,
        stereo_bootstrap_matches_count = stereo_bootstrap_matches.len(),
        bootstrap_depth = args.bootstrap_depth_meters,
        bootstrap_landmarks = seed_features.len(),
        vi_first = vi_init_first_event_at_frame,
        vi_succeeded = vi_init_succeeded_at_frame,
        motion_enabled = args.motion_vi_init_enabled,
        motion_first = motion_vi_init_first_event_at_frame,
        motion_succeeded = motion_vi_init_succeeded_at_frame,
        motion_scale = motion_vi_init_recovered_scale,
        motion_iters = motion_vi_init_viba2_iterations,
        local_vi_ba_enabled = args.local_vi_ba_enabled,
        local_vi_ba_freeze = args.local_vi_ba_freeze_biases_above,
        motion_vi_max_vel = args.motion_vi_init_max_velocity_mps,
        covisibility_max_kf = args.covisibility_local_map_max_keyframes,
        covisibility_min_shared = args.covisibility_local_map_min_shared,
        covisibility_used_frames = covisibility_local_map_frames,
        covisibility_mean_size = if covisibility_local_map_frames > 0 {
            covisibility_local_map_size_sum as f64 / covisibility_local_map_frames as f64
        } else {
            0.0
        },
        max_pose_jump = args.max_pose_jump_meters,
        pnp_warm_start = args.pnp_pose_prior_warm_start,
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
        kf_min_translation = args.keyframe_min_translation,
        keep_pre_promotion = args.keep_pre_promotion_imu_factors,
        relinearise_thresholds = args.relinearise_imu_factor_bias_thresholds,
        run_at_promotion = args.run_local_vi_ba_at_vi_init_promotion,
        reloc_enabled = args.relocalization_enabled,
        reloc_min_inliers = args.relocalization_min_inliers,
        reloc_min_inlier_ratio = args.relocalization_min_inlier_ratio,
        reloc_max_rep_err = args.relocalization_max_reprojection_error,
        reloc_pose_prior_radius = args.relocalization_pose_prior_radius_meters,
        reloc_recent_kf_window = args.relocalization_recent_keyframe_window,
        reloc_max_tx_from_imu = args.relocalization_max_translation_from_imu_prediction_meters,
        reloc_attempts = relocalization_attempts,
        reloc_successes = relocalization_successes,
        vi_init_try_every_frame = args.vi_init_try_initialize_on_every_frame,
        imu_factors_staged = imu_factors_staged,
        local_vi_ba_triggers = local_vi_ba_triggers,
        local_vi_ba_relinearised_factor_total = local_vi_ba_relinearised_factor_total,
        local_vi_ba_mirrors = local_vi_ba_mirrors,
        last_mirrored_v = last_mirrored_velocity_world.map(|v| [v.x, v.y, v.z]),
        last_mirrored_bg = last_mirrored_bias_gyro.map(|v| [v.x, v.y, v.z]),
        last_mirrored_ba = last_mirrored_bias_acc.map(|v| [v.x, v.y, v.z]),
        quality_gate_failures = slam.tracker.stats().tracking_quality_gate_failure_count,
        scale = aligned_similarity.scale,
    );
    println!("{summary}");
    fs::write(args.out_dir.join("summary.txt"), &summary)?;
    println!(
        "wrote {}, {}, {}, {} (+ summary.txt)",
        traj_path.display(),
        err_path.display(),
        vi_init_log_path.display(),
        motion_vi_init_log_path.display(),
    );
    Ok(())
}

#[cfg(all(test, feature = "image-io"))]
mod tests {
    use super::*;

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
    fn world_to_camera_pose_round_trip_identity_rig() {
        let body_rot = UnitQuaternion::identity();
        let body_pos = Vector3::new(1.0, 2.0, 3.0);
        let rig = SE3::identity();
        let pose = world_to_camera_pose(&body_rot, &body_pos, &rig);
        let camera_center = pose.camera_center_world();
        assert!(
            (Vector3::new(camera_center.x, camera_center.y, camera_center.z) - body_pos).norm()
                < 1.0e-9
        );
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
        let map = bootstrap_map_from_first_frame(&camera, &pose, &features, 4.0, &[], false);
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
        let map = bootstrap_map_from_first_frame(
            &camera,
            &pose,
            &features,
            4.0,
            &stereo_overrides,
            false,
        );
        assert_eq!(map.landmarks.len(), 2);
        // Landmark id = index + 1.
        let first = map.landmarks.get(&1).expect("override landmark seeded");
        assert!((first.position - override_point).norm() < 1.0e-9);
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
        let strict_map =
            bootstrap_map_from_first_frame(&camera, &pose, &features, 4.0, &stereo_overrides, true);
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
}
