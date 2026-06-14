//! Single-binary deep stereo SLAM — the clean top-level entry point.
//!
//! Runs a full learned-front-end stereo SLAM from raw rectified images in a
//! **single Rust binary**: SuperPoint feature extraction + LightGlue matching
//! via ONNX Runtime (GPU), windowed online bundle adjustment, and VLAD→PnP→GNC
//! SE(3) loop-closure pose-graph optimization — no Python, no PyTorch, no
//! pre-exported multi-GB feature dump.
//!
//! This is the discoverable wrapper around the in-process deep front-end that
//! `stereo_vo_external_deep_files --in-process-onnx` exposes behind a thicket of
//! file-based flags. It bakes the benchmark-validated VO/BA/loop configuration
//! as defaults, so the only required input is the image directory, the two ONNX
//! models, and the stereo calibration. See `docs/inprocess_slam_benchmark.md`
//! for the EuRoC + KITTI accuracy numbers it reproduces.
//!
//! Build & run (CUDA — use the runner that sets up the provider libs + cuDNN):
//!   scripts/run_deep_stereo_slam.sh \
//!       --images-dir /tmp/MH_03_rect \
//!       --calib /tmp/MH_03_rect/calib.txt --width 752 --height 480 \
//!       --frames 2700 --loop-min-frame-gap 200 --out-dir target/deep_slam_mh03
//!
//! EuRoC aerial flights want `--loop-min-frame-gap 200` (hovering at 20 Hz makes
//! small frame gaps travel almost nothing); KITTI driving uses the default 50.
//! For a non-752×480 resolution, export a LightGlue model at that size (the
//! matcher bakes the image size; SuperPoint is resolution-dynamic).

use std::path::{Path, PathBuf};
use std::time::Instant;

use visloc_rs::io::images::read_common_image;
use visloc_rs::vision::features::deep::DeepFeatureExtractor;
use visloc_rs::vision::features::lightglue_onnx::{LightGlueMatch, LightGlueOnnxMatcher};
use visloc_rs::vision::features::superpoint_onnx::{
    OnnxBackend, SuperPointOnnxConfig, SuperPointOnnxExtractor,
};
use visloc_rs::vision::features::FeatureSet;
use visloc_rs::{
    close_loops_on_vo_trajectory, parse_kitti_calibration_txt, Camera, DescriptorMatch,
    OnlineStereoVoBa, OnlineStereoVoBaConfig, PoseTrajectory, StereoRelativePoseMode,
    StereoVoFrontend, StereoVoFrontendConfig, TrackingEvent, TrackingState, TrajectorySample,
    VoLoopClosureConfig,
};

struct Args {
    images_dir: PathBuf,
    left_subdir: String,
    right_subdir: String,
    superpoint_model: PathBuf,
    lightglue_model: PathBuf,
    out_dir: PathBuf,
    calib: Option<PathBuf>,
    projection_left: String,
    projection_right: String,
    width: u32,
    height: u32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    baseline: f64,
    frames: usize,
    max_keypoints: usize,
    onnx_cpu: bool,
    loop_min_frame_gap: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            images_dir: PathBuf::from("/tmp/MH_03_rect"),
            left_subdir: "image_0".to_string(),
            right_subdir: "image_1".to_string(),
            superpoint_model: PathBuf::from("models/superpoint_1500.onnx"),
            lightglue_model: PathBuf::from("models/lightglue.onnx"),
            out_dir: PathBuf::from("target/deep_stereo_slam"),
            calib: None,
            projection_left: "P0".to_string(),
            projection_right: "P1".to_string(),
            width: 752,
            height: 480,
            fx: 0.0,
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            baseline: 0.0,
            frames: 2700,
            max_keypoints: 1500,
            onnx_cpu: false,
            loop_min_frame_gap: 50,
        }
    }
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut next = || it.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--images-dir" => a.images_dir = PathBuf::from(next()?),
            "--left-subdir" => a.left_subdir = next()?,
            "--right-subdir" => a.right_subdir = next()?,
            "--superpoint-model" => a.superpoint_model = PathBuf::from(next()?),
            "--lightglue-model" => a.lightglue_model = PathBuf::from(next()?),
            "--out-dir" => a.out_dir = PathBuf::from(next()?),
            "--calib" => a.calib = Some(PathBuf::from(next()?)),
            "--projection-left" => a.projection_left = next()?,
            "--projection-right" => a.projection_right = next()?,
            "--width" => a.width = next()?.parse()?,
            "--height" => a.height = next()?.parse()?,
            "--fx" => a.fx = next()?.parse()?,
            "--fy" => a.fy = next()?.parse()?,
            "--cx" => a.cx = next()?.parse()?,
            "--cy" => a.cy = next()?.parse()?,
            "--baseline" => a.baseline = next()?.parse()?,
            "--frames" => a.frames = next()?.parse()?,
            "--max-keypoints" => a.max_keypoints = next()?.parse()?,
            "--onnx-cpu" => a.onnx_cpu = true,
            "--loop-min-frame-gap" => a.loop_min_frame_gap = next()?.parse()?,
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    if a.frames < 2 {
        return Err("--frames must be at least 2".into());
    }
    Ok(a)
}

/// Rectified stereo intrinsics `(fx, fy, cx, cy, baseline_m)` from a KITTI calib.
fn kitti_stereo_calibration(
    calib_path: &Path,
    projection_left: &str,
    projection_right: &str,
) -> Result<(f64, f64, f64, f64, f64), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(calib_path)?;
    let projections = parse_kitti_calibration_txt(&text)?;
    let left = projections
        .iter()
        .find(|p| p.label == projection_left)
        .ok_or_else(|| format!("calib missing {projection_left}"))?;
    let right = projections
        .iter()
        .find(|p| p.label == projection_right)
        .ok_or_else(|| format!("calib missing {projection_right}"))?;
    let baseline = right.stereo_baseline_from(left).ok_or_else(|| {
        format!("calib pair {projection_left}<->{projection_right} gave no baseline")
    })?;
    Ok((left.fx(), left.fy(), left.cx(), left.cy(), baseline))
}

/// Minimum LightGlue match confidence kept for both the stereo and temporal
/// matches. This is the benchmark-validated default (`--min-stereo-confidence` /
/// `--min-temporal-confidence` 0.5 in `stereo_vo_external_deep_files`): low-score
/// LightGlue matches are noisy correspondences that degrade the VO/BA solve, so
/// gating them is what reproduces the docs/inprocess_slam_benchmark.md accuracy.
const MIN_MATCH_CONFIDENCE: f32 = 0.5;

/// LightGlue matches → the front-end's `DescriptorMatch` (the score becomes the
/// match confidence, identical to the file-based path's contract), keeping only
/// matches at or above [`MIN_MATCH_CONFIDENCE`].
fn to_descriptor_matches(matches: Vec<LightGlueMatch>) -> Vec<DescriptorMatch> {
    matches
        .into_iter()
        .filter(|m| m.score.is_finite() && m.score >= MIN_MATCH_CONFIDENCE)
        .map(|m| DescriptorMatch {
            query_index: m.query_index,
            train_index: m.train_index,
            distance: 1.0 - m.score,
            second_best_distance: None,
            ratio: None,
            confidence: Some(m.score),
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out_dir)?;

    let (fx, fy, cx, cy, baseline) = if let Some(calib) = &args.calib {
        kitti_stereo_calibration(calib, &args.projection_left, &args.projection_right)?
    } else {
        if args.fx <= 0.0 || args.baseline <= 0.0 {
            return Err(
                "supply --calib (KITTI P0/P1) or explicit --fx/--fy/--cx/--cy/--baseline".into(),
            );
        }
        (args.fx, args.fy, args.cx, args.cy, args.baseline)
    };
    println!(
        "calibration: fx={fx:.2} fy={fy:.2} cx={cx:.2} cy={cy:.2} baseline={baseline:.4} m  \
         image {}x{}",
        args.width, args.height
    );

    // The deep front-end feeds features directly via `process_feature_pair_with_matches`,
    // so the corner extractor / matcher type parameters are never exercised.
    let camera = Camera::pinhole(0, args.width, args.height, fx, fy, cx, cy);
    let vo_config = StereoVoFrontendConfig {
        relative_pose_mode: StereoRelativePoseMode::PnpThenKabsch,
        ..StereoVoFrontendConfig::default()
    };
    let frontend = StereoVoFrontend::new(camera, baseline, vo_config);

    // Benchmark-validated online-BA config: a 10-frame sliding window extended
    // backward by a fixed 20-frame prefix so long-baseline landmarks anchor the
    // recent poses (ORB-SLAM3 fixed-keyframe pattern). The default inner BA
    // config (Huber 3 px, sparse, fix first pose) already matches the runs in
    // docs/inprocess_slam_benchmark.md.
    let online_config = OnlineStereoVoBaConfig {
        trigger_every_frames: 10,
        window_size: 10,
        local_map_history: 20,
        ..OnlineStereoVoBaConfig::default()
    };
    println!(
        "online BA: window={} history={} trigger_every={}",
        online_config.window_size,
        online_config.local_map_history,
        online_config.trigger_every_frames
    );
    let mut runner = OnlineStereoVoBa::new(frontend, online_config);

    // In-process deep front-end (SuperPoint + LightGlue, ONNX Runtime).
    let backend = if args.onnx_cpu {
        OnnxBackend::Cpu
    } else {
        OnnxBackend::CudaThenCpu
    };
    let sp_config = SuperPointOnnxConfig {
        max_keypoints: args.max_keypoints,
        ..Default::default()
    };
    let superpoint = SuperPointOnnxExtractor::load_from_path_with_backend(
        &args.superpoint_model,
        sp_config,
        backend,
    )?;
    let lightglue =
        LightGlueOnnxMatcher::load_from_path_with_backend(&args.lightglue_model, backend)?;
    let left_dir = args.images_dir.join(&args.left_subdir);
    let right_dir = args.images_dir.join(&args.right_subdir);
    println!(
        "in-process deep front-end loaded ({})",
        if args.onnx_cpu { "CPU" } else { "CUDA→CPU" }
    );

    let start = Instant::now();
    // Previous left-frame SuperPoint keypoints+descriptors, for the temporal match.
    let mut prev_left: Option<(Vec<nalgebra::Point2<f64>>, Vec<Vec<f32>>)> = None;

    for frame_id in 0..args.frames {
        let name = format!("{frame_id:06}.png");
        let left_img = read_common_image(left_dir.join(&name))?;
        let right_img = read_common_image(right_dir.join(&name))?;
        let left = superpoint.extract_deep(&left_img)?;
        let right = superpoint.extract_deep(&right_img)?;

        let stereo: Vec<DescriptorMatch> = to_descriptor_matches(lightglue.match_features(
            &left.keypoints,
            &left.descriptors,
            &right.keypoints,
            &right.descriptors,
        )?);

        let temporal: Option<Vec<DescriptorMatch>> = match prev_left.take() {
            None => None,
            Some((pk, pd)) => Some(to_descriptor_matches(lightglue.match_features(
                &pk,
                &pd,
                &left.keypoints,
                &left.descriptors,
            )?)),
        };

        let left_features = FeatureSet::new(left.keypoints.clone(), left.descriptors.clone())?;
        let right_features = FeatureSet::new(right.keypoints, right.descriptors)?;
        prev_left = Some((left.keypoints, left.descriptors));

        runner.process_pair_with_matches(
            left_features,
            right_features,
            Some(&stereo),
            temporal.as_deref(),
        )?;

        if frame_id > 0 {
            let d = runner.frontend.pair_diagnostics.last().unwrap();
            if frame_id % 100 == 0 {
                println!(
                    "frame {frame_id}/{}  src={:?} temporal={} stereo={} inliers={} t={:.3}m",
                    args.frames,
                    d.source,
                    d.temporal_match_count,
                    d.stereo_pair_correspondence_count,
                    d.inlier_count,
                    d.translation_m,
                );
            }
        }
    }
    println!(
        "VO done: {} poses in {:.1}s ({:.1} fps)",
        runner.frontend.poses.len(),
        start.elapsed().as_secs_f64(),
        args.frames as f64 / start.elapsed().as_secs_f64(),
    );

    // Loop closure: VLAD appearance retrieval → PnP verification → robust GNC
    // SE(3) pose-graph optimization, with per-loop two-view BA + anisotropic
    // loop-edge information (both positive levers on EuRoC + KITTI).
    let loop_cfg = VoLoopClosureConfig {
        min_frame_gap: args.loop_min_frame_gap,
        refine_loops_two_view: true,
        loop_edge_information: true,
        ..VoLoopClosureConfig::default()
    };
    println!(
        "loop-closure PGO: poses={} min_frame_gap={} min_similarity={:.2} vocab_k={}",
        runner.frontend.poses.len(),
        loop_cfg.min_frame_gap,
        loop_cfg.min_similarity,
        loop_cfg.vocab_k,
    );
    match close_loops_on_vo_trajectory(
        &runner.frontend.camera,
        &runner.frontend.poses,
        &runner.frontend.left_features,
        &runner.frontend.stereo_per_frame,
        &loop_cfg,
    ) {
        Ok(result) => {
            match &result.gnc {
                Some(gnc) => println!(
                    "loop-closure PGO: candidates={} verified_loops={} cost {:.3} -> {:.3} \
                     ({} outer iters, converged={})",
                    result.candidate_count,
                    result.verified_count(),
                    gnc.initial_cost,
                    gnc.final_cost,
                    gnc.outer_iterations,
                    gnc.converged,
                ),
                None => println!(
                    "loop-closure PGO: candidates={} verified_loops=0 (trajectory unchanged)",
                    result.candidate_count,
                ),
            }
            runner.frontend.poses = result.refined_poses;
        }
        Err(err) => eprintln!("loop-closure PGO skipped: {err}"),
    }

    // Outputs: KITTI 3×4 poses (for evo), trajectory centers CSV, summary.
    let frontend = &runner.frontend;
    let poses = &frontend.poses;
    let mut traj = PoseTrajectory::new();
    for (frame_id, pose) in poses.iter().enumerate() {
        traj.push_sample(TrajectorySample {
            frame_id: frame_id as u64,
            pose: pose.clone(),
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            inlier_count: 0,
            inlier_ratio: 0.0,
            reprojection_error: None,
        });
    }
    traj.write_kitti_poses(args.out_dir.join("vo_poses.txt"))?;

    let mut csv = String::from("id,x,y,z\n");
    for (i, pose) in poses.iter().enumerate() {
        let c = pose.camera_center_world();
        csv.push_str(&format!("{i},{:.6},{:.6},{:.6}\n", c.x, c.y, c.z));
    }
    std::fs::write(args.out_dir.join("vo.csv"), csv)?;
    std::fs::write(
        args.out_dir.join("summary.txt"),
        format!(
            "frames={} pairs={} trajectory_length_m={:.6}\n",
            frontend.frame_count(),
            frontend.pair_diagnostics.len(),
            frontend.trajectory_length_m(),
        ),
    )?;
    println!(
        "wrote {}/vo_poses.txt (+ vo.csv, summary.txt)",
        args.out_dir.display()
    );
    Ok(())
}
