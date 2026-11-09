//! End-to-end EuRoC VIO pipeline-integration demo on **real cam0 pixels**.
//!
//! Companion to [`euroc_online_slam_vi_demo`](../euroc_online_slam_vi_demo.rs)
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
//! Basalt-faithful VIO knobs (window size / sqrt marg from the checked-in
//! `configs/basalt/euroc_config.json`):
//! ```sh
//! ... --basalt-euroc-profile --motion-vi-init --local-vi-ba
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
    bootstrap_stereo_landmarks, bootstrap_stereo_landmarks_from_correspondences,
    StereoBootstrapConfig, StereoBootstrapLandmark,
};
#[cfg(feature = "image-io")]
use visloc_rs::{
    build_stereo_replenish_candidates, read_external_deep_features_txt,
    umeyama_similarity_transform, AdaptiveImuPoseMotionModel, AdaptiveImuPoseMotionModelConfig,
    AdaptiveMotionMode, AdaptiveVelocityGateConfig, BaConfig, BiasReleaseSchedule,
    BruteForceMatcher, ConstantPoseMotionModel, ConstantVelocityMotionModel,
    CovisibilityLocalBaConfig, CovisibilityLocalBaError, CovisibilityLocalMapConfig,
    CrossCheckMatcher, CrossSubmapAlignmentConfig, CrossSubmapAlignmentResult,
    CrossSubmapBoundaryFactorResult, DescriptorMatch, GravityVelocityAlignment, GyroBiasAlignment,
    ImuPredictiveMotionModel, ImuPredictiveMotionModelConfig, ImuVelocityRefreshPolicy,
    KeyframeDecisionReason, KeyframePolicyConfig, LandmarkCandidate, LocalMappingPipeline,
    LocalMappingResult, LocalizationConfig, LocalizationPipeline, LoopAppearanceCandidateConfig,
    LoopClosureCandidateSource, LoopClosureConfig, LoopClosureVerifierConfig, LoopRefinementSolver,
    LoopRefinementVerifier, MapAtlas, MapProviderStats, Matcher, MotionBasedViInitializerConfig,
    MotionModel, MotionViInitializationEvent, MotionViInitializationStatus,
    MotionViRawResidualActivationConfig, MutualSoftmaxConfig, MutualSoftmaxMatcher,
    OnlineSlamConfig, OnlineSlamCovisibilityLocalBaConfig, OnlineSlamLocalBaConfig,
    OnlineSlamLoopClosureRefinementConfig, OnlineSlamMotionViInitConfig, OnlineSlamPipeline,
    OnlineSlamRelocalizationStats, OnlineSlamViInitConfig, PnPLoopClosureVerifierConfig,
    PoseGraphSe3Config, PosePriorVisualOverrideConfig, PoseTrajectory,
    ProjectionGuidedTrackingConfig, RelativePoseErrorConfig, Sim3PoseGraphConfig,
    SimpleKeyframePolicy, StereoReplenishConfig, Tracker, TrackingConfig, TrackingEvent,
    TrackingResult, TrackingState, TrajectorySimilarityTransform, ViInitFallback,
    ViInitializationEvent, Viba2Config, VisualInertialInitializerConfig,
};

#[cfg(feature = "image-io")]
mod bootstrap;
#[cfg(feature = "image-io")]
mod cli;
#[cfg(feature = "image-io")]
mod extractor;
#[cfg(feature = "image-io")]
mod frontend;
#[cfg(feature = "image-io")]
mod matcher;
#[cfg(feature = "image-io")]
mod motion_model;
#[cfg(feature = "image-io")]
mod relocalization_gate;
#[cfg(feature = "image-io")]
mod report;

#[cfg(feature = "image-io")]
use bootstrap::*;
#[cfg(feature = "image-io")]
use cli::*;
#[cfg(feature = "image-io")]
use extractor::*;
#[cfg(feature = "image-io")]
use frontend::*;
#[cfg(feature = "image-io")]
use matcher::*;
#[cfg(feature = "image-io")]
use motion_model::*;
#[cfg(feature = "image-io")]
use relocalization_gate::*;
#[cfg(feature = "image-io")]
use report::*;

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
        FeatureExtractorKind::OpticalFlow => {
            println!(
                "using Basalt-style optical-flow frontend (levels={}, pattern={}, grid={}, max_iters={})",
                args.optical_flow_config.optical_flow_levels,
                args.optical_flow_config.optical_flow_pattern,
                args.optical_flow_config.optical_flow_detection_grid_size,
                args.optical_flow_config.optical_flow_max_iterations,
            );
            DemoExtractor::OpticalFlow(
                visloc_rs::vision::optical_flow::OpticalFlowFeatureExtractor::new(
                    args.optical_flow_config.clone(),
                ),
            )
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
        if matches!(args.feature_extractor, FeatureExtractorKind::OpticalFlow) {
            // Basalt stereo: LK each cam0 track into cam1 at the same timestamp
            // on *raw* pixels, then undistort both sides before triangulation.
            let DemoExtractor::OpticalFlow(ref of_extractor) = extractor else {
                unreachable!("OpticalFlow kind requires OpticalFlow extractor");
            };
            let left_pts: Vec<(f32, f32)> = seed_features_raw
                .keypoints
                .iter()
                .map(|p| (p.x as f32, p.y as f32))
                .collect();
            let tracked = of_extractor.track_points_to(&seed_image, &cam1_image, &left_pts);
            stereo_cam1_features_count = tracked.iter().filter(|p| p.is_some()).count();

            let mut right_undist_by_raw = vec![None; seed_features_raw.keypoints.len()];
            for (raw_i, right) in tracked.into_iter().enumerate() {
                let Some((rx, ry)) = right else {
                    continue;
                };
                right_undist_by_raw[raw_i] =
                    cam1_distortion.undistort_pixel(cam1_camera, Point2::new(rx as f64, ry as f64));
            }

            let mut correspondences = Vec::new();
            let mut left_undist_idx = 0usize;
            for (raw_i, kp) in seed_features_raw.keypoints.iter().enumerate() {
                let Some(undist_left) = distortion.undistort_pixel(&camera, *kp) else {
                    continue;
                };
                debug_assert_eq!(
                    seed_features.keypoints[left_undist_idx], undist_left,
                    "undistort filter order must match seed_features"
                );
                if let Some(undist_right) = right_undist_by_raw[raw_i] {
                    correspondences.push((left_undist_idx, undist_right));
                    stereo_right_pixels[left_undist_idx] = Some(undist_right);
                }
                left_undist_idx += 1;
            }
            stereo_cam1_features_after_undistort_count = correspondences.len();
            let epi_thresh = args
                .optical_flow_config
                .optical_flow_epipolar_error
                .max(0.0) as f64;
            let before_epi = correspondences.len();
            let correspondences =
                visloc_rs::vision::stereo_bootstrap::filter_correspondences_by_epipolar(
                    &camera,
                    cam1_camera,
                    cam0_to_cam1,
                    &seed_features.keypoints,
                    &correspondences,
                    epi_thresh,
                );
            if before_epi != correspondences.len() {
                eprintln!(
                    "basalt OF stereo epipolar: {before_epi} -> {} (thresh={epi_thresh})",
                    correspondences.len()
                );
            }
            // Refresh right-pixel slots to epipolar survivors only.
            for slot in stereo_right_pixels.iter_mut() {
                *slot = None;
            }
            for &(left_idx, right_px) in &correspondences {
                stereo_right_pixels[left_idx] = Some(right_px);
            }
            stereo_bootstrap_matches = bootstrap_stereo_landmarks_from_correspondences(
                &camera,
                cam1_camera,
                cam0_to_cam1,
                &seed_features.keypoints,
                &correspondences,
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
            }
            println!(
                "stereo_bootstrap(optical-flow) cam1_seed_idx={cam1_seed_idx} \
                 lk_survivors={} triangulated_matches={}",
                stereo_cam1_features_count,
                stereo_bootstrap_matches.len(),
            );
        } else {
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
                min_translation_meters: args.vi_bias_release_min_translation_meters.unwrap_or(2.0),
            })
        } else {
            None
        };
        Some(OnlineSlamMotionViInitConfig {
            initializer: MotionBasedViInitializerConfig {
                min_keyframes: args.motion_vi_init_min_keyframes,
                min_translation_meters: args.motion_vi_init_min_translation_meters,
                max_solve_keyframes: Some(5),
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
            reject_writeback_when_observation_count_below: args
                .motion_vi_init_enabled
                .then_some(20),
            writeback_navigation_state_despite_pose_gate: args.motion_vi_init_enabled,
            reject_gyro_bias_above_rad_s: args.local_vi_ba_reject_gyro_bias_above_rad_s,
            reject_accel_bias_above_mps2: args.local_vi_ba_reject_accel_bias_above_mps2,
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
            marginalize_navigation_state: args.local_vi_ba_marginalization
                || args.local_vi_ba_use_sqrt_window_marginalization,
            use_sqrt_window_marginalization: args.local_vi_ba_use_sqrt_window_marginalization,
            window_size: args.local_vi_ba_window_size,
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
    } else if args.local_vi_ba_marginalization && args.local_vi_ba_use_sqrt_window_marginalization {
        Some(OnlineSlamLoopClosureRefinementConfig::recovered_factor_sink(camera.clone()))
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
        pose_prior_visual_override: args.pose_prior_visual_override.then(|| {
            if args.motion_vi_init_enabled {
                // Gate58 (50/5×) and gate60 (80/0.30/2×) both raised
                // tracking but collapsed Sim(3) scale (→0.28 / →0.19).
                // Gate59 (80/0.5/2×) never fired (PPT≥80 ratio p50≈0.45).
                // Keep a conservative opt-in bar; do not default — the
                // 0.2 m jump lock is what preserves metric scale.
                PosePriorVisualOverrideConfig {
                    min_inliers: 80,
                    min_inlier_ratio: 0.50,
                    max_mean_reprojection_error: Some(3.0),
                    max_rotation_error_radians: Some(2.0_f64.to_radians()),
                    max_translation_error_multiplier: 2.0,
                    ..PosePriorVisualOverrideConfig::default()
                }
            } else {
                PosePriorVisualOverrideConfig::default()
            }
        }),
        accept_motion_prior_on_failure: args.accept_motion_prior_on_failure,
        max_consecutive_motion_prior_coasts: args.max_consecutive_motion_prior_coasts,
        temporal_landmark_tracking: args.temporal_landmark_tracking,
        temporal_landmark_tracking_history_frames: args.temporal_landmark_tracking_history_frames,
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
        DemoMatcher::MutualSoftmax(MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
            temperature: args.mutual_softmax_temperature,
            min_confidence: args.mutual_softmax_min_confidence,
            ..MutualSoftmaxConfig::default()
        }))
    } else if matches!(args.feature_extractor, FeatureExtractorKind::OpticalFlow) {
        // Track-id descriptors are exact; Lowe ratio is harmful and any
        // non-zero NN latch onto a neighbour id poisons PnP.
        DemoMatcher::MaxDistance {
            inner: BruteForceMatcher { ratio: None },
            max_distance: 1.0e-3,
        }
    } else if let Some(dll) = args.gpu_matcher_dll.as_ref() {
        // Exact GPU nearest-neighbour; the ratio policy matches BruteForce and
        // the matcher degrades to the CPU path if the library is unusable.
        #[cfg(feature = "gpu-matcher")]
        {
            let matcher = visloc_rs::vision::matching::GpuDescriptorMatcher::load(
                BruteForceMatcher {
                    ratio: localization_config.ratio,
                },
                dll,
            );
            println!(
                "gpu matcher: dll={} active={}",
                dll.display(),
                matcher.is_gpu_active()
            );
            if !matcher.is_gpu_active() {
                eprintln!(
                    "warning: --gpu-matcher-dll {} could not be loaded; using the CPU matcher",
                    dll.display()
                );
            }
            DemoMatcher::Gpu(matcher)
        }
        #[cfg(not(feature = "gpu-matcher"))]
        {
            eprintln!(
                "warning: --gpu-matcher-dll requires the `gpu-matcher` feature; using the CPU matcher"
            );
            let _ = dll;
            DemoMatcher::BruteForce(BruteForceMatcher {
                ratio: localization_config.ratio,
            })
        }
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
        } else if args.motion_vi_init_enabled {
            // 1.0 m (library default) yields ~20 KFs / 400f on MH_01; gyro
            // alignment then sits at ~0.3 rad. 0.05 m is dense enough for a
            // 3-keyframe IMU window before VO rotation drifts.
            cfg.min_translation = 0.05;
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
    let mut local_vi_ba_recovered_marginal_factors_injected: usize = 0;
    let mut local_vi_ba_recovered_marginal_factors_enqueued: usize = 0;
    let mut local_vi_ba_quality_gate_rejections: usize = 0;
    let mut local_vi_ba_cost_ratio_gate_rejections: usize = 0;
    let mut local_vi_ba_imu_nis_gate_rejections: usize = 0;
    let mut local_vi_ba_velocity_gate_rejections: usize = 0;
    let mut local_vi_ba_pose_correction_gate_rejections: usize = 0;
    let mut local_vi_ba_navigation_state_partial_writebacks: usize = 0;
    let mut local_vi_ba_max_pose_translation_correction_meters: f64 = 0.0;
    let mut local_vi_ba_max_pose_rotation_correction_degrees: f64 = 0.0;
    let mut local_vi_ba_bias_magnitude_gate_rejections: usize = 0;
    let mut local_vi_ba_max_gyro_bias_norm_rad_s: f64 = 0.0;
    let mut local_vi_ba_max_accel_bias_norm_mps2: f64 = 0.0;
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
        if args.adaptive_keypoints {
            // Pace the SuperPoint cap to the rolling frame-time budget. The
            // matcher is O(N^2) in the keypoint count, so the cap is scaled by
            // sqrt(budget / measured) to move the dominant term toward the
            // target; it is clamped to the configured cap and a small floor.
            apply_adaptive_keypoint_cap(
                &extractor,
                args.adaptive_frame_budget_ms,
                &frame_processing_times_ms,
            );
        }
        let (features_raw, features) =
            if matches!(args.feature_extractor, FeatureExtractorKind::OpticalFlow)
                && frame_idx == seed_frame_idx
            {
                // Reuse the exact FeatureSet that seeded the map so track-id
                // descriptors match 1:1. Re-running KLT on the seed image can
                // drop tracks and mint new ids, zeroing localization.
                (seed_features_raw.clone(), seed_features.clone())
            } else {
                match extractor.extract(&image) {
                    Ok(raw) => {
                        let undist = undistort_feature_keypoints(&distortion, &camera, &raw);
                        (raw, undist)
                    }
                    Err(err) => {
                        eprintln!(
                            "skipping frame_idx={frame_idx} due to feature-extraction error: {err}"
                        );
                        continue;
                    }
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
                        if matches!(args.feature_extractor, FeatureExtractorKind::OpticalFlow) {
                            // Stereo LK via track_points_to — does not touch
                            // the cam0-temporal OF tracker state.
                            let DemoExtractor::OpticalFlow(ref of_extractor) = extractor else {
                                unreachable!("OpticalFlow kind requires OpticalFlow extractor");
                            };
                            optical_flow_stereo_features_for_frame(
                                of_extractor,
                                &image,
                                &cam1_image,
                                &features_raw,
                                &features,
                                &distortion,
                                &camera,
                                &cam1_setup.distortion,
                                &cam1_setup.camera,
                                &cam1_setup.cam0_to_cam1,
                                args.optical_flow_config
                                    .optical_flow_epipolar_error
                                    .max(0.0) as f64,
                            )
                        } else {
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
                && atlas_merge_attempts
                    .is_multiple_of(args.atlas_broader_recovery_interval_attempts)
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
            local_vi_ba_recovered_marginal_factors_injected +=
                stats.recovered_marginal_factors_injected;
            local_vi_ba_recovered_marginal_factors_enqueued +=
                stats.recovered_marginal_factors_enqueued;
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
            if stats.navigation_state_partially_written {
                local_vi_ba_navigation_state_partial_writebacks += 1;
            }
            local_vi_ba_max_pose_translation_correction_meters =
                local_vi_ba_max_pose_translation_correction_meters
                    .max(stats.max_pose_translation_correction_meters);
            local_vi_ba_max_pose_rotation_correction_degrees =
                local_vi_ba_max_pose_rotation_correction_degrees
                    .max(stats.max_pose_rotation_correction_radians.to_degrees());
            if stats.bias_magnitude_gate_rejected {
                local_vi_ba_bias_magnitude_gate_rejections += 1;
            }
            local_vi_ba_max_gyro_bias_norm_rad_s =
                local_vi_ba_max_gyro_bias_norm_rad_s.max(stats.max_refined_gyro_bias_norm_rad_s);
            local_vi_ba_max_accel_bias_norm_mps2 =
                local_vi_ba_max_accel_bias_norm_mps2.max(stats.max_refined_accel_bias_norm_mps2);
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
            if !stats.bias_frozen
                && (!stats.quality_gate_rejected || stats.navigation_state_partially_written)
            {
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
         motion_prior_coast_count={motion_prior_coast_count}\n\
         temporal_landmark_attempt_count={temporal_landmark_attempt_count}\n\
         temporal_landmark_success_count={temporal_landmark_success_count}\n\
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
         local_vi_ba_use_sqrt_window_marginalization={local_vi_ba_use_sqrt_window_marginalization}\n\
         local_vi_ba_window_size={local_vi_ba_window_size}\n\
         basalt_config_path={basalt_config_path}\n\
         local_vi_ba_general_stereo=true\n\
         local_vi_ba_initial_prior_std_devs={local_vi_ba_initial_prior_std_devs:?}\n\
         local_vi_ba_freeze_biases_above={local_vi_ba_freeze:?}\n\
         local_vi_ba_reject_writeback_above={local_vi_ba_reject_writeback:?}\n\
         local_vi_ba_reject_final_imu_nis_per_dof_above={local_vi_ba_reject_final_imu_nis:?}\n\
         local_vi_ba_reject_velocity_above_mps={local_vi_ba_reject_velocity:?}\n\
         local_vi_ba_reject_pose_translation_above_meters={local_vi_ba_reject_pose_translation:?}\n\
         local_vi_ba_reject_pose_rotation_above_degrees={local_vi_ba_reject_pose_rotation:?}\n\
         local_vi_ba_reject_gyro_bias_above_rad_s={local_vi_ba_reject_gyro_bias:?}\n\
         local_vi_ba_reject_accel_bias_above_mps2={local_vi_ba_reject_accel_bias:?}\n\
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
         accept_motion_prior_on_failure={accept_motion_prior_on_failure}\n\
         temporal_landmark_tracking={temporal_landmark_tracking}\n\
         temporal_landmark_tracking_history_frames={temporal_landmark_history}\n\
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
         mutual_softmax_temperature={mutual_softmax_temperature}\n\
         mutual_softmax_min_confidence={mutual_softmax_min_confidence}\n\
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
         local_vi_ba_recovered_marginal_factors_enqueued={local_vi_ba_recovered_marginal_factors_enqueued}\n\
         local_vi_ba_recovered_marginal_factors_injected={local_vi_ba_recovered_marginal_factors_injected}\n\
         local_vi_ba_quality_gate_rejections={local_vi_ba_quality_gate_rejections}\n\
         local_vi_ba_cost_ratio_gate_rejections={local_vi_ba_cost_ratio_gate_rejections}\n\
         local_vi_ba_imu_nis_gate_rejections={local_vi_ba_imu_nis_gate_rejections}\n\
         local_vi_ba_velocity_gate_rejections={local_vi_ba_velocity_gate_rejections}\n\
         local_vi_ba_pose_correction_gate_rejections={local_vi_ba_pose_correction_gate_rejections}\n\
         local_vi_ba_navigation_state_partial_writebacks={local_vi_ba_navigation_state_partial_writebacks}\n\
         local_vi_ba_max_pose_translation_correction_meters={local_vi_ba_max_pose_translation_correction_meters:.9}\n\
         local_vi_ba_max_pose_rotation_correction_degrees={local_vi_ba_max_pose_rotation_correction_degrees:.9}\n\
         local_vi_ba_bias_magnitude_gate_rejections={local_vi_ba_bias_magnitude_gate_rejections}\n\
         local_vi_ba_max_gyro_bias_norm_rad_s={local_vi_ba_max_gyro_bias_norm_rad_s:.9}\n\
         local_vi_ba_max_accel_bias_norm_mps2={local_vi_ba_max_accel_bias_norm_mps2:.9}\n\
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
        motion_prior_coast_count = slam.tracker.stats().motion_prior_coast_count,
        temporal_landmark_attempt_count = slam.tracker.stats().temporal_landmark_attempt_count,
        temporal_landmark_success_count = slam.tracker.stats().temporal_landmark_success_count,
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
        local_vi_ba_use_sqrt_window_marginalization = args
            .local_vi_ba_use_sqrt_window_marginalization,
        local_vi_ba_window_size = args.local_vi_ba_window_size,
        basalt_config_path = format!("{:?}", args.basalt_config_path),
        local_vi_ba_initial_prior_std_devs = args.local_vi_ba_initial_prior_std_devs,
        local_vi_ba_freeze = args.local_vi_ba_freeze_biases_above,
        local_vi_ba_reject_writeback = args.local_vi_ba_reject_writeback_above,
        local_vi_ba_reject_final_imu_nis =
            args.local_vi_ba_reject_final_imu_nis_per_dof_above,
        local_vi_ba_reject_velocity = args.local_vi_ba_reject_velocity_above_mps,
        local_vi_ba_reject_pose_translation =
            args.local_vi_ba_reject_pose_translation_above_meters,
        local_vi_ba_reject_pose_rotation = args.local_vi_ba_reject_pose_rotation_above_degrees,
        local_vi_ba_reject_gyro_bias = args.local_vi_ba_reject_gyro_bias_above_rad_s,
        local_vi_ba_reject_accel_bias = args.local_vi_ba_reject_accel_bias_above_mps2,
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
        accept_motion_prior_on_failure = args.accept_motion_prior_on_failure,
        temporal_landmark_tracking = args.temporal_landmark_tracking,
        temporal_landmark_history = args.temporal_landmark_tracking_history_frames,
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
        mutual_softmax_temperature = args.mutual_softmax_temperature,
        mutual_softmax_min_confidence = args.mutual_softmax_min_confidence,
        feature_extractor_kind = match args.feature_extractor {
            FeatureExtractorKind::Corner => "corner",
            FeatureExtractorKind::Hog => "hog",
            FeatureExtractorKind::SuperPointOffline => "superpoint-offline",
            FeatureExtractorKind::SuperPointOnnx => "superpoint-onnx",
            FeatureExtractorKind::OpticalFlow => "optical-flow",
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
        local_vi_ba_recovered_marginal_factors_enqueued =
            local_vi_ba_recovered_marginal_factors_enqueued,
        local_vi_ba_recovered_marginal_factors_injected =
            local_vi_ba_recovered_marginal_factors_injected,
        local_vi_ba_quality_gate_rejections = local_vi_ba_quality_gate_rejections,
        local_vi_ba_cost_ratio_gate_rejections = local_vi_ba_cost_ratio_gate_rejections,
        local_vi_ba_velocity_gate_rejections = local_vi_ba_velocity_gate_rejections,
        local_vi_ba_pose_correction_gate_rejections =
            local_vi_ba_pose_correction_gate_rejections,
        local_vi_ba_navigation_state_partial_writebacks =
            local_vi_ba_navigation_state_partial_writebacks,
        local_vi_ba_max_pose_translation_correction_meters =
            local_vi_ba_max_pose_translation_correction_meters,
        local_vi_ba_max_pose_rotation_correction_degrees =
            local_vi_ba_max_pose_rotation_correction_degrees,
        local_vi_ba_bias_magnitude_gate_rejections = local_vi_ba_bias_magnitude_gate_rejections,
        local_vi_ba_max_gyro_bias_norm_rad_s = local_vi_ba_max_gyro_bias_norm_rad_s,
        local_vi_ba_max_accel_bias_norm_mps2 = local_vi_ba_max_accel_bias_norm_mps2,
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
mod tests;
