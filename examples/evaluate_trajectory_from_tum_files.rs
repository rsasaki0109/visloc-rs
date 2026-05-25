use std::env;
use std::fs;
use std::path::PathBuf;

use visloc_rs::{
    PoseTrajectory, RelativePoseErrorConfig, TrajectoryAlignment, TrajectoryEvaluationConfig,
};

const DEFAULT_ESTIMATED_TUM: &str = "\
# frame_id tx ty tz qx qy qz qw
0 0.00 0.00 0.00 0 0 0 1
1 1.05 0.02 0.00 0 0 0 1
2 2.10 0.04 0.01 0 0 0 1
4 4.30 0.10 0.00 0 0 0 1
";

const DEFAULT_REFERENCE_TUM: &str = "\
# frame_id tx ty tz qx qy qz qw
0 0.0 0.0 0.0 0 0 0 1
1 1.0 0.0 0.0 0 0 0 1
2 2.0 0.0 0.0 0 0 0 1
3 3.0 0.0 0.0 0 0 0 1
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    let alignment = parse_alignment(&mut args);
    let rpe_config = parse_rpe_config(&mut args);
    let evaluation_config = parse_evaluation_config(&mut args);
    let (estimated, reference) = match args.as_slice() {
        [] => (
            PoseTrajectory::from_tum_poses_str(DEFAULT_ESTIMATED_TUM)?,
            PoseTrajectory::from_tum_poses_str(DEFAULT_REFERENCE_TUM)?,
        ),
        [estimated_path, reference_path] => {
            let estimated = PoseTrajectory::read_tum_poses(estimated_path)?;
            let reference = PoseTrajectory::read_tum_poses(reference_path)?;
            (estimated, reference)
        }
        _ => {
            eprintln!(
                "usage: cargo run --example evaluate_trajectory_from_tum_files -- [--out-dir <dir>] [--align-origin] [--rpe-delta <n>] [--rpe-step <n>] [--max-mean <m>] [--max-rmse <m>] [--max-max <m>] [--min-matched <n>] [--min-match-ratio <r>] [estimated_tum.txt reference_tum.txt]"
            );
            std::process::exit(2);
        }
    };

    let summary = estimated.translation_error_summary_against_with_alignment(&reference, alignment);
    let errors_csv = estimated.translation_errors_csv_against_with_alignment(&reference, alignment);
    let evaluation = summary.evaluate(evaluation_config);

    println!(
        "loaded trajectories: estimated={} reference={} alignment={:?}",
        estimated.len(),
        reference.len(),
        alignment
    );
    println!(
        "matched={} missing_reference={} missing_estimate={} mean={:?} rmse={:?} max={:?}",
        summary.matched_pose_count,
        summary.missing_reference_count,
        summary.missing_estimate_count,
        summary.mean_translation_error,
        summary.rmse_translation_error,
        summary.max_translation_error
    );
    println!("translation_errors_csv:\n{}", errors_csv);
    println!("summary_json:\n{}", summary.to_json());

    // Relative pose error (RPE): local drift over Δ-spaced pairs. Unlike the
    // ATE above, RPE needs no alignment, so it is reported as-is.
    let rpe = estimated.relative_pose_error_against(&reference, &rpe_config);
    println!(
        "rpe delta={} pairs={} translation_m={:?} rotation_deg={:?}",
        rpe.delta,
        rpe.pair_count,
        rpe.translation.map(|s| s.rmse),
        rpe.rotation_deg.map(|s| s.rmse)
    );
    println!("rpe_json:\n{}", rpe.to_json());
    println!(
        "evaluation passed={} match_ratio={:?} failures={:?}",
        evaluation.passed, evaluation.match_ratio, evaluation.failures
    );
    println!("evaluation_json:\n{}", evaluation.to_json());

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir)?;
        let errors_path = output_dir.join("translation_errors.csv");
        let summary_path = output_dir.join("error_summary.json");
        let evaluation_path = output_dir.join("evaluation_result.json");
        let report_path = output_dir.join("trajectory_report.html");
        let rpe_errors_path = output_dir.join("relative_pose_errors.csv");
        let rpe_summary_path = output_dir.join("relative_pose_error_summary.json");
        estimated.write_translation_errors_csv_against_with_alignment(
            &reference,
            alignment,
            &errors_path,
        )?;
        summary.write_json(&summary_path)?;
        evaluation.write_json(&evaluation_path)?;
        estimated.write_html_report_against_with_alignment(&reference, alignment, &report_path)?;
        rpe.write_errors_csv(&rpe_errors_path)?;
        rpe.write_json(&rpe_summary_path)?;
        println!(
            "wrote evaluation exports: errors={} summary={} evaluation={} report={} rpe_errors={} rpe_summary={}",
            errors_path.display(),
            summary_path.display(),
            evaluation_path.display(),
            report_path.display(),
            rpe_errors_path.display(),
            rpe_summary_path.display()
        );
    }

    if !evaluation.passed {
        std::process::exit(1);
    }

    Ok(())
}

fn parse_output_dir(args: &mut Vec<String>) -> Option<PathBuf> {
    let output_flag_index = args.iter().position(|arg| arg == "--out-dir")?;
    if output_flag_index + 1 >= args.len() {
        eprintln!("--out-dir requires a directory path");
        std::process::exit(2);
    }

    let output_dir = PathBuf::from(args.remove(output_flag_index + 1));
    args.remove(output_flag_index);
    Some(output_dir)
}

fn parse_alignment(args: &mut Vec<String>) -> TrajectoryAlignment {
    let Some(flag_index) = args.iter().position(|arg| arg == "--align-origin") else {
        return TrajectoryAlignment::None;
    };
    args.remove(flag_index);
    TrajectoryAlignment::FirstMatchedTranslation
}

fn parse_rpe_config(args: &mut Vec<String>) -> RelativePoseErrorConfig {
    RelativePoseErrorConfig {
        delta: parse_usize_flag(args, "--rpe-delta").unwrap_or(1),
        start_step: parse_usize_flag(args, "--rpe-step").unwrap_or(1),
    }
}

fn parse_evaluation_config(args: &mut Vec<String>) -> TrajectoryEvaluationConfig {
    TrajectoryEvaluationConfig {
        max_mean_translation_error: parse_f64_flag(args, "--max-mean"),
        max_rmse_translation_error: parse_f64_flag(args, "--max-rmse"),
        max_max_translation_error: parse_f64_flag(args, "--max-max"),
        min_matched_pose_count: parse_usize_flag(args, "--min-matched"),
        min_match_ratio: parse_f64_flag(args, "--min-match-ratio"),
    }
}

fn parse_f64_flag(args: &mut Vec<String>, flag: &str) -> Option<f64> {
    let flag_index = args.iter().position(|arg| arg == flag)?;
    if flag_index + 1 >= args.len() {
        eprintln!("{flag} requires a numeric value");
        std::process::exit(2);
    }

    let value = args.remove(flag_index + 1);
    args.remove(flag_index);
    Some(value.parse::<f64>().unwrap_or_else(|error| {
        eprintln!("invalid value for {flag}: {error}");
        std::process::exit(2);
    }))
}

fn parse_usize_flag(args: &mut Vec<String>, flag: &str) -> Option<usize> {
    let flag_index = args.iter().position(|arg| arg == flag)?;
    if flag_index + 1 >= args.len() {
        eprintln!("{flag} requires an integer value");
        std::process::exit(2);
    }

    let value = args.remove(flag_index + 1);
    args.remove(flag_index);
    Some(value.parse::<usize>().unwrap_or_else(|error| {
        eprintln!("invalid value for {flag}: {error}");
        std::process::exit(2);
    }))
}
