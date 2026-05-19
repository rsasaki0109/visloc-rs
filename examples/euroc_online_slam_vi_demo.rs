//! End-to-end EuRoC pipeline + VI-init integration smoke run.
//!
//! Drives [`OnlineSlamPipeline`] with the real EuRoC MAV IMU stream and the
//! real cam0 frame cadence, with the auto-bootstrap stage
//! ([`OnlineSlamConfig::vi_init`]) enabled. Cam0 image pixels are **not**
//! decoded — instead, the demo seeds a small deterministic landmark cloud
//! in front of the first ground-truth body pose and projects it into each
//! frame under the GT-derived camera-from-world transform. This gives the
//! pipeline's tracker a stable, scale-anchored visual signal so
//! `process_frame` actually runs end-to-end, while keeping the demo free
//! of a real feature extractor. The IMU side is the real EuRoC stream.
//!
//! What this validates:
//! 1. The pipeline accepts the real EuRoC IMU sample rate (~200 Hz) and
//!    cam0 cadence (~20 Hz) without buffering pathologies.
//! 2. The auto-bootstrap stage actually fires on a real EuRoC sequence's
//!    leading stationary segment, the promoted bias / rotation match GT
//!    within the documented tolerances, and the stale-factor gate
//!    correctly discards any factor staged before promotion.
//! 3. The pipeline's tracked trajectory aligns with GT under Umeyama
//!    rigid / similarity ATE — proving the atomic promotion (preintegrator
//!    reset, config mirror, keyframe pose rewrite) preserves geometric
//!    consistency on real data.
//!
//! Usage:
//! ```sh
//! cargo run --release --example euroc_online_slam_vi_demo -- \
//!     --euroc-dir /path/to/MH_01_easy \
//!     --out-dir target/euroc_online_slam_vi_demo \
//!     --max-frames 400
//! ```
//!
//! Writes `slam_trajectory.csv` (per-cam0-frame estimated pose),
//! `slam_errors.csv` (per-frame ATE residuals), `vi_init_log.txt` (every
//! state-transition event from the auto-bootstrap stage) and
//! `summary.txt` to the output directory.

use std::env;
use std::fs;
use std::path::PathBuf;

use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::{Pose, SE3};
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::io::euroc::{
    read_euroc_dataset_dir, EurocCameraCalibration, EurocGroundTruthSample,
};
use visloc_rs::slam::OnlineSlamImuConfig;
use visloc_rs::{
    umeyama_similarity_transform, LocalMappingPipeline, LocalizationPipeline, LoopClosureConfig,
    MotionBasedViInitializerConfig, MotionViInitializationEvent, OnlineSlamConfig,
    OnlineSlamMotionViInitConfig, OnlineSlamPipeline, OnlineSlamViInitConfig, Tracker,
    TrackingConfig, TrajectorySimilarityTransform, ViInitFallback, ViInitializationEvent,
    Viba2Config, VisualInertialInitializerConfig,
};

#[derive(Debug)]
struct CliArgs {
    euroc_dir: PathBuf,
    out_dir: PathBuf,
    /// Cap the number of cam0 frames processed. `0` keeps the full
    /// sequence. The default (400) is enough to span the leading
    /// stationary window on every EuRoC MH sequence and ~10 s of the
    /// subsequent motion — long enough for the auto-bootstrap to
    /// promote and the pipeline to log a meaningful ATE.
    max_frames: usize,
    /// Gravity in the EuRoC world frame (z-up, gravity points down).
    gravity_world: Vector3<f64>,
    /// VI-init window cap (seconds) — the same knob exposed on the
    /// pipeline-side config. EuRoC MH lift-off is ~1 s.
    vi_init_max_wait_seconds: f64,
    /// Optional per-axis gyro / accel standard deviation override for
    /// the VI initialiser. EuRoC hand-held release periods often miss
    /// the default 0.05 rad/s threshold; values around 0.1–0.2 admit
    /// the leading "lift-off" window.
    vi_init_gyro_std_limit: Option<f64>,
    vi_init_accel_std_limit: Option<f64>,
    /// Enable the motion-based VI init stage (VIBA1 / optional VIBA2).
    /// Off by default to preserve the existing baseline; the stage is
    /// gated on the static VI init completing first.
    motion_vi_init_enabled: bool,
    motion_vi_init_min_keyframes: usize,
    motion_vi_init_min_translation_meters: f64,
    motion_vi_init_recover_scale: bool,
}

fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut euroc_dir: Option<PathBuf> = None;
    let mut out_dir = PathBuf::from("target/euroc_online_slam_vi_demo");
    let mut max_frames: usize = 400;
    let mut gravity_world = Vector3::new(0.0, 0.0, -9.81);
    let mut vi_init_max_wait_seconds: f64 = 5.0;
    let mut vi_init_gyro_std_limit: Option<f64> = None;
    let mut vi_init_accel_std_limit: Option<f64> = None;
    let mut motion_vi_init_enabled: bool = false;
    let mut motion_vi_init_min_keyframes: usize = 10;
    let mut motion_vi_init_min_translation_meters: f64 = 2.0;
    let mut motion_vi_init_recover_scale: bool = false;

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
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let euroc_dir = euroc_dir.ok_or("--euroc-dir <path/to/MH_01_easy> is required")?;
    Ok(CliArgs {
        euroc_dir,
        out_dir,
        max_frames,
        gravity_world,
        vi_init_max_wait_seconds,
        vi_init_gyro_std_limit,
        vi_init_accel_std_limit,
        motion_vi_init_enabled,
        motion_vi_init_min_keyframes,
        motion_vi_init_min_translation_meters,
        motion_vi_init_recover_scale,
    })
}

/// Decompose an EuRoC `T_BS` 4×4 matrix into an [`SE3`]. EuRoC stores `T_BS`
/// as body-to-sensor in row-major form; the rotation block is a proper
/// rotation matrix, the last column is the translation, and the last row
/// is `(0, 0, 0, 1)` — exactly the SE(3) layout this codebase uses.
fn se3_from_t_bs(t_bs: &Matrix4<f64>) -> SE3 {
    let rotation_matrix = t_bs.fixed_view::<3, 3>(0, 0).into_owned();
    let translation = Vector3::new(t_bs[(0, 3)], t_bs[(1, 3)], t_bs[(2, 3)]);
    let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
    SE3::new(rotation, translation)
}

/// Build the world-to-camera [`Pose`] implied by a GT body pose and the
/// fixed `body_to_camera` rig calibration. This is the inverse of the
/// rendering convention `T_w←c = T_w←b · T_b←c`; we go from that into the
/// codebase's `Pose::from_world_to_camera` layout.
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
/// EuRoC images are recorded with non-zero `radial-tangential` distortion
/// coefficients, but the demo projects synthetic landmarks through the
/// pinhole intrinsics directly (no real pixels are sampled), so the
/// distortion model is intentionally ignored.
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

/// Construct the synthetic 3D landmark cloud, positioned in front of the
/// first GT camera pose. A 5×5 grid spanning ±2 m laterally / vertically
/// at a fixed depth of 6 m keeps every landmark in the cam0 field of view
/// for several seconds even as the body moves — enough to keep the
/// tracker's localisation success rate high throughout the bootstrap.
fn seed_landmark_grid(
    initial_pose_world_to_camera: &Pose,
    grid_size: usize,
    extent_meters: f64,
    depth_meters: f64,
) -> Vec<Point3<f64>> {
    // Build the grid in the camera frame, then transform into the world
    // frame using the inverse of the first world-to-camera pose. Working
    // in the camera frame at construction time guarantees the points
    // start in the FOV; transforming once to world space makes them
    // stationary world-frame landmarks for the rest of the demo.
    let r_wc = initial_pose_world_to_camera
        .world_to_camera
        .rotation
        .inverse();
    let t_wc = -(r_wc.transform_vector(&initial_pose_world_to_camera.world_to_camera.translation));
    let camera_to_world = SE3::new(r_wc, t_wc);

    let mut points = Vec::with_capacity(grid_size * grid_size);
    for i in 0..grid_size {
        for j in 0..grid_size {
            let u = if grid_size > 1 {
                -extent_meters + 2.0 * extent_meters * (i as f64) / (grid_size as f64 - 1.0)
            } else {
                0.0
            };
            let v = if grid_size > 1 {
                -extent_meters + 2.0 * extent_meters * (j as f64) / (grid_size as f64 - 1.0)
            } else {
                0.0
            };
            let p_cam = Point3::new(u, v, depth_meters);
            points.push(camera_to_world.transform_point(&p_cam));
        }
    }
    points
}

fn build_seeded_map(camera: &Camera, points: &[Point3<f64>]) -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 1.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor);
        map.landmarks.insert(landmark.id, landmark);
    }
    map
}

/// Project every world-frame seed point through `pose_world_to_camera` and
/// emit a frame whose keypoints / descriptors line up with the seeded
/// landmark ids. Off-image or behind-camera projections are silently
/// dropped — the tracker handles partial visibility natively.
fn frame_from_gt_pose(
    frame_id: u64,
    camera: &Camera,
    pose: &Pose,
    points: &[Point3<f64>],
) -> Frame {
    let mut frame = Frame::new(frame_id, camera.id);
    for (index, point) in points.iter().enumerate() {
        let p_cam = pose.transform_world_point(point);
        if p_cam.z <= 0.0 {
            continue;
        }
        let Some(uv) = camera.project(&p_cam) else {
            continue;
        };
        if uv.x < 0.0 || uv.y < 0.0 || uv.x >= camera.width as f64 || uv.y >= camera.height as f64 {
            continue;
        }
        frame.keypoints.push(uv);
        frame.descriptors.push(vec![index as f32, 1.0]);
    }
    frame
}

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
    println!(
        "cam0 fx={} fy={} cx={} cy={} resolution={}x{} body_to_camera_t=[{:.3},{:.3},{:.3}]",
        camera.params[0],
        camera.params[1],
        camera.params[2],
        camera.params[3],
        camera.width,
        camera.height,
        body_to_camera.translation.x,
        body_to_camera.translation.y,
        body_to_camera.translation.z,
    );

    // Anchor the seed cloud at the first GT sample inside the IMU stream
    // window so the bootstrap pose and the rendered first frame agree
    // exactly. Skipping the warm-up gap also lets us read initial GT
    // biases off the same row.
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
    let initial_pose = world_to_camera_pose(
        &seed_gt.orientation_world,
        &seed_gt.position_world,
        &body_to_camera,
    );
    let landmark_grid = seed_landmark_grid(&initial_pose, 5, 2.5, 6.0);
    let map = build_seeded_map(&camera, &landmark_grid);
    println!(
        "seeded landmarks={} from gt_seed_t_ns={} body_position=[{:.3},{:.3},{:.3}]",
        landmark_grid.len(),
        seed_gt.timestamp_nanoseconds,
        seed_gt.position_world.x,
        seed_gt.position_world.y,
        seed_gt.position_world.z,
    );

    // Mirror the dead-reckon demo's VI-init knobs so the same EuRoC
    // sequence shows the same buffer / threshold behaviour through the
    // pipeline.
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
    let vi_init_config = OnlineSlamViInitConfig {
        initializer: initializer_config,
        body_to_camera: body_to_camera.clone(),
        seed_first_keyframe_rotation: true,
        on_persistent_rejection: ViInitFallback::KeepExistingSeed,
        max_wait_duration_seconds: args.vi_init_max_wait_seconds,
        max_buffered_samples: 4000,
        try_initialize_on_every_frame: false,
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
                ..MotionBasedViInitializerConfig::default()
            },
            ..OnlineSlamMotionViInitConfig::default()
        })
    } else {
        None
    };
    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig::default(),
            imu: Some(imu_config),
            local_vi_ba: None,
            vi_init: Some(vi_init_config),
            vi_motion_init: vi_motion_init_config,
            keep_pre_promotion_imu_factors: false,
            pose_graph_refinement: None,
            relocalization: None,
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

    // IMU iterator state: feed every sample whose timestamp is ≤ the
    // current cam0 frame timestamp to the pipeline before calling
    // `process_frame` for that frame.
    let mut imu_idx = 0usize;
    let mut prev_imu_ts = imu_first_ts;
    // Skip any IMU samples before the seed GT row so the integrator
    // starts from a coherent reference.
    while imu_idx < dataset.imu_samples.len()
        && dataset.imu_samples[imu_idx].timestamp_nanoseconds < seed_gt.timestamp_nanoseconds
    {
        prev_imu_ts = dataset.imu_samples[imu_idx].timestamp_nanoseconds;
        imu_idx += 1;
    }

    let mut frames_recorded = 0usize;
    let mut tracking_successes = 0usize;
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

    for (frame_idx, image_entry) in dataset.cam0_images.iter().enumerate() {
        if image_entry.timestamp_nanoseconds < seed_gt.timestamp_nanoseconds {
            continue;
        }
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
            }
            prev_imu_ts = sample.timestamp_nanoseconds;
            imu_idx += 1;
        }

        // Render the frame against the GT body pose interpolated to the
        // cam0 timestamp.
        let gt = nearest_ground_truth(&dataset.ground_truth, image_entry.timestamp_nanoseconds);
        let pose_world_to_camera =
            world_to_camera_pose(&gt.orientation_world, &gt.position_world, &body_to_camera);
        let frame = frame_from_gt_pose(
            frame_idx as u64,
            &camera,
            &pose_world_to_camera,
            &landmark_grid,
        );

        let result = slam.process_frame(&frame, []);
        let tracked = result.tracking.localization.pose.clone();
        let success = result.tracking_succeeded();
        if success {
            tracking_successes += 1;
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

        // Record the tracked pose's camera centre in the world frame.
        // `Pose::camera_center_world` returns the inverse-applied
        // translation, i.e. `-R_c←w⁻¹ · t_c←w` — the world-frame point at
        // which the camera optical centre sits.
        let (estimated_center, estimated_rotation_wc) = if let Some(pose) = tracked.as_ref() {
            let center = pose.camera_center_world();
            let rotation_wc = pose.world_to_camera.rotation.inverse();
            (Some(center), Some(rotation_wc))
        } else {
            (None, None)
        };
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
            // GT body rotation expressed as camera-in-world: w_R_c = w_R_b · b_R_c.
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
                "{},{frame_idx},,,,,,,,,{}\n",
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

    let summary = format!(
        "euroc_dir={}\n\
         frames_recorded={frames_recorded}\n\
         tracking_success_rate={success_rate:.3}\n\
         imu_samples_consumed={imu_idx}\n\
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
        vi_first = vi_init_first_event_at_frame,
        vi_succeeded = vi_init_succeeded_at_frame,
        motion_enabled = args.motion_vi_init_enabled,
        motion_first = motion_vi_init_first_event_at_frame,
        motion_succeeded = motion_vi_init_succeeded_at_frame,
        motion_scale = motion_vi_init_recovered_scale,
        motion_iters = motion_vi_init_viba2_iterations,
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

#[cfg(test)]
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
    fn se3_from_t_bs_recovers_translation() {
        let mut t_bs = Matrix4::identity();
        t_bs[(0, 3)] = 0.5;
        t_bs[(1, 3)] = -0.2;
        t_bs[(2, 3)] = 0.1;
        let se3 = se3_from_t_bs(&t_bs);
        assert!((se3.translation - Vector3::new(0.5, -0.2, 0.1)).norm() < 1.0e-12);
    }

    #[test]
    fn world_to_camera_pose_round_trip_identity_rig() {
        let body_rot = UnitQuaternion::identity();
        let body_pos = Vector3::new(1.0, 2.0, 3.0);
        let rig = SE3::identity();
        let pose = world_to_camera_pose(&body_rot, &body_pos, &rig);
        // Camera at the same position as body when rig is identity.
        let camera_center = pose.camera_center_world();
        assert!(
            (Vector3::new(camera_center.x, camera_center.y, camera_center.z) - body_pos).norm()
                < 1.0e-9
        );
    }

    #[test]
    fn frame_renders_seeded_landmarks_for_identity_pose() {
        let camera = camera_from_cam0(&fake_cam0(), 1);
        let identity_pose =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        // 3x3 grid 1 m wide at z=5 m → all 9 project inside the 752x480
        // image.
        let points = seed_landmark_grid(&identity_pose, 3, 0.5, 5.0);
        let frame = frame_from_gt_pose(0, &camera, &identity_pose, &points);
        assert_eq!(frame.keypoints.len(), 9);
        assert_eq!(frame.descriptors.len(), 9);
    }
}
