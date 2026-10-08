//! GNSS + visual-odometry joint fusion on a synthetic drive.
//!
//! A drifting VO trajectory (yaw drift, 2 % scale error; or monocular with an
//! unknown scale that drifts 10 %) is fused with a 5 Hz noisy GNSS stream that
//! has its own clock offset, a 15 s dropout, and three multipath bursts. The
//! demo reports the absolute trajectory error (ATE) before and after fusion.
//!
//! ```text
//! cargo run --example gnss_vo_fusion_demo -- \
//!     [--scenario metric|monocular] [--window <frames>] \
//!     [--robust gnc|huber|none] [--out-dir <dir>]
//! ```

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use visloc_rs::fusion::TimedPose;
use visloc_rs::slam::gnss_fusion::{
    aligned_position_rmse, fuse_gnss_with_visual_odometry, position_rmse, transform_trajectory,
    FrameGnssStatus, GnssRobustMode, GnssVoFusionConfig,
};
use visloc_rs::slam::gnss_synthetic::{generate_gnss_vo_scenario, SyntheticGnssVoConfig};
use visloc_rs::slam::{RobustKernel, GNSS_CHI2_3DOF_999};

struct Args {
    monocular: bool,
    window: Option<usize>,
    robust: String,
    out_dir: Option<PathBuf>,
}

fn usage() -> ! {
    eprintln!(
        "usage: cargo run --example gnss_vo_fusion_demo -- [--scenario metric|monocular] \
         [--window <frames>] [--robust gnc|huber|none] [--out-dir <dir>]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut args = Args {
        monocular: false,
        window: None,
        robust: "gnc".to_owned(),
        out_dir: None,
    };
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = || iter.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--scenario" => match value().as_str() {
                "metric" => args.monocular = false,
                "monocular" => args.monocular = true,
                _ => usage(),
            },
            "--window" => args.window = Some(value().parse().unwrap_or_else(|_| usage())),
            "--robust" => args.robust = value(),
            "--out-dir" => args.out_dir = Some(PathBuf::from(value())),
            _ => usage(),
        }
    }
    args
}

fn rmse_over(estimate: &[TimedPose], truth: &[TimedPose], mask: &[bool]) -> Option<f64> {
    let (sum, count) = estimate
        .iter()
        .zip(truth)
        .zip(mask)
        .filter(|(_, selected)| **selected)
        .fold((0.0, 0usize), |(sum, count), ((a, b), _)| {
            let d = a.value.camera_center_world() - b.value.camera_center_world();
            (sum + d.norm_squared(), count + 1)
        });
    (count > 0).then(|| (sum / count as f64).sqrt())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let scenario_config = if args.monocular {
        SyntheticGnssVoConfig::monocular()
    } else {
        SyntheticGnssVoConfig::default()
    };
    let scenario = generate_gnss_vo_scenario(&scenario_config);

    let mut config = if args.monocular {
        GnssVoFusionConfig::monocular()
    } else {
        GnssVoFusionConfig::metric_gravity_aligned()
    };
    config.lever_arm = scenario.lever_arm;
    config.window = args.window;
    config.optimizer.robust = match args.robust.as_str() {
        "gnc" => GnssRobustMode::default(),
        "huber" => GnssRobustMode::Kernel(RobustKernel::Huber {
            delta: GNSS_CHI2_3DOF_999.sqrt(),
        }),
        "none" => {
            // Plain least squares with no post-hoc gating either.
            config.optimizer.refine_without_outliers = false;
            GnssRobustMode::None
        }
        _ => usage(),
    };

    let started = Instant::now();
    let fused = fuse_gnss_with_visual_odometry(&scenario.vo, &scenario.gnss, &config)?;
    let elapsed = started.elapsed();

    let truth = &scenario.ground_truth_enu;
    let vo_bootstrap = transform_trajectory(&fused.bootstrap.alignment, &scenario.vo);
    let vo_bootstrap_ate = position_rmse(&vo_bootstrap, truth).unwrap_or(f64::NAN);
    let vo_oracle_ate =
        aligned_position_rmse(&scenario.vo, truth, args.monocular).unwrap_or(f64::NAN);
    let fused_ate = position_rmse(&fused.trajectory_enu, truth).unwrap_or(f64::NAN);
    let dropout_mask: Vec<bool> = fused
        .frames
        .iter()
        .map(|frame| {
            matches!(
                frame.gnss,
                FrameGnssStatus::NoFix | FrameGnssStatus::NotOptimized
            )
        })
        .collect();
    let fused_dropout_ate = rmse_over(&fused.trajectory_enu, truth, &dropout_mask);
    let max_fused_error = fused
        .trajectory_enu
        .iter()
        .zip(truth)
        .map(|(a, b)| (a.value.camera_center_world() - b.value.camera_center_world()).norm())
        .fold(0.0_f64, f64::max);

    // Outlier bookkeeping: a frame is "near an outlier" when it lies within
    // one GNSS period of a corrupted fix (so it may interpolate from it).
    let gnss_period_ns = (1e9 / scenario_config.gnss_rate_hz) as i128;
    let near_outlier: Vec<bool> = fused
        .frames
        .iter()
        .map(|frame| {
            scenario.outlier_fix_timestamps.iter().any(|t| {
                (t.as_nanoseconds() - frame.timestamp.as_nanoseconds()).abs() < gnss_period_ns
            })
        })
        .collect();
    let flagged: Vec<bool> = fused
        .frames
        .iter()
        .map(|frame| matches!(frame.gnss, FrameGnssStatus::Outlier { .. }))
        .collect();
    let true_positive = flagged
        .iter()
        .zip(&near_outlier)
        .filter(|(f, n)| **f && **n)
        .count();
    let false_positive = flagged
        .iter()
        .zip(&near_outlier)
        .filter(|(f, n)| **f && !**n)
        .count();
    let outlier_frame_ate = rmse_over(&fused.trajectory_enu, truth, &near_outlier);

    let truth_alignment = &scenario.true_alignment;
    let yaw_error_deg = (fused.alignment.yaw() - truth_alignment.yaw()).to_degrees();
    let rotation_error_deg = fused
        .alignment
        .rotation
        .angle_to(&truth_alignment.rotation)
        .to_degrees();
    let scale_error = fused.alignment.scale / truth_alignment.scale - 1.0;
    let translation_error = (fused.alignment.translation - truth_alignment.translation).norm();

    println!(
        "scenario={} frames={} gnss_fixes={} corrupted_fixes={} window={:?} robust={}",
        if args.monocular {
            "monocular"
        } else {
            "metric"
        },
        fused.summary.frame_count,
        fused.summary.gnss_fix_count,
        scenario.outlier_fix_timestamps.len(),
        args.window,
        args.robust,
    );
    println!(
        "frames: inlier_fix={} outlier_fix={} no_fix(dropout)={} optimizer_runs={} time={:.2}s",
        fused.summary.frames_with_inlier_fix,
        fused.summary.frames_with_outlier_fix,
        fused.summary.frames_without_fix,
        fused.summary.optimizer_runs,
        elapsed.as_secs_f64(),
    );
    println!(
        "bootstrap: inliers={} rms={:.3} m; last solve: lm_iterations={} gnc_levels={} converged={}",
        fused.bootstrap.inlier_count,
        fused.bootstrap.inlier_rms,
        fused.last_update.optimizer.iterations,
        fused.last_update.optimizer.gnc_levels,
        fused.last_update.optimizer.converged,
    );
    println!(
        "alignment error{}: yaw={:+.3} deg rotation={:.3} deg scale={:+.4} translation={:.3} m",
        if args.window.is_some() {
            " (latest window; absorbs VO drift since frame 0)"
        } else {
            ""
        },
        yaw_error_deg,
        rotation_error_deg,
        scale_error,
        translation_error
    );
    println!("ATE vo_only_bootstrap_aligned={vo_bootstrap_ate:.3} m");
    println!(
        "ATE vo_only_oracle_{}_aligned={vo_oracle_ate:.3} m",
        if args.monocular { "sim3" } else { "se3" }
    );
    println!("ATE fused={fused_ate:.3} m (max {max_fused_error:.3} m)");
    println!(
        "ATE fused_dropout_frames={:.3} m fused_outlier_frames={:.3} m",
        fused_dropout_ate.unwrap_or(f64::NAN),
        outlier_frame_ate.unwrap_or(f64::NAN)
    );
    println!(
        "outlier frames flagged: {true_positive} near corrupted fixes, {false_positive} elsewhere"
    );

    if let Some(dir) = args.out_dir {
        fs::create_dir_all(&dir)?;
        let mut csv = String::from(
            "timestamp_ns,truth_x,truth_y,truth_z,vo_x,vo_y,vo_z,fused_x,fused_y,fused_z,gnss_status\n",
        );
        for (((frame, gt), vo), fused_pose) in fused
            .frames
            .iter()
            .zip(truth)
            .zip(&vo_bootstrap)
            .zip(&fused.trajectory_enu)
        {
            let (g, v, f) = (
                gt.value.camera_center_world(),
                vo.value.camera_center_world(),
                fused_pose.value.camera_center_world(),
            );
            let status = match frame.gnss {
                FrameGnssStatus::Inlier { .. } => "inlier",
                FrameGnssStatus::Outlier { .. } => "outlier",
                FrameGnssStatus::NoFix | FrameGnssStatus::NotOptimized => "no_fix",
            };
            writeln!(
                csv,
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{status}",
                frame.timestamp.as_nanoseconds(),
                g.x,
                g.y,
                g.z,
                v.x,
                v.y,
                v.z,
                f.x,
                f.y,
                f.z
            )?;
        }
        fs::write(dir.join("trajectory.csv"), csv)?;
        let summary = format!(
            concat!(
                "{{\n",
                "  \"demo\": \"gnss_vo_fusion_demo\",\n",
                "  \"scenario\": \"{}\",\n",
                "  \"frame_count\": {},\n",
                "  \"gnss_fix_count\": {},\n",
                "  \"corrupted_fix_count\": {},\n",
                "  \"frames_with_inlier_fix\": {},\n",
                "  \"frames_with_outlier_fix\": {},\n",
                "  \"frames_without_fix\": {},\n",
                "  \"ate_vo_only_bootstrap_aligned_m\": {:.6},\n",
                "  \"ate_vo_only_oracle_aligned_m\": {:.6},\n",
                "  \"ate_fused_m\": {:.6},\n",
                "  \"max_fused_error_m\": {:.6},\n",
                "  \"alignment_yaw_error_deg\": {:.6},\n",
                "  \"alignment_scale_error\": {:.6},\n",
                "  \"alignment_translation_error_m\": {:.6}\n",
                "}}\n"
            ),
            if args.monocular {
                "monocular"
            } else {
                "metric"
            },
            fused.summary.frame_count,
            fused.summary.gnss_fix_count,
            scenario.outlier_fix_timestamps.len(),
            fused.summary.frames_with_inlier_fix,
            fused.summary.frames_with_outlier_fix,
            fused.summary.frames_without_fix,
            vo_bootstrap_ate,
            vo_oracle_ate,
            fused_ate,
            max_fused_error,
            yaw_error_deg,
            scale_error,
            translation_error,
        );
        fs::write(dir.join("summary.json"), summary)?;
        println!("wrote {}", dir.display());
    }
    Ok(())
}
