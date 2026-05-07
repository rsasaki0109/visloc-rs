use std::env;
use std::fs;
use std::path::PathBuf;

use visloc_rs::{PoseTrajectory, TrajectoryAlignment};

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
                "usage: cargo run --example evaluate_trajectory_from_tum_files -- [--out-dir <dir>] [--align-origin] [estimated_tum.txt reference_tum.txt]"
            );
            std::process::exit(2);
        }
    };

    let summary = estimated.translation_error_summary_against_with_alignment(&reference, alignment);
    let errors_csv = estimated.translation_errors_csv_against_with_alignment(&reference, alignment);

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

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir)?;
        let errors_path = output_dir.join("translation_errors.csv");
        let summary_path = output_dir.join("error_summary.json");
        estimated.write_translation_errors_csv_against_with_alignment(
            &reference,
            alignment,
            &errors_path,
        )?;
        summary.write_json(&summary_path)?;
        println!(
            "wrote evaluation exports: errors={} summary={}",
            errors_path.display(),
            summary_path.display()
        );
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
