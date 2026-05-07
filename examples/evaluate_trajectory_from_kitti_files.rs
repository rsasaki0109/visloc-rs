use std::env;
use std::fs;
use std::path::PathBuf;

use visloc_rs::PoseTrajectory;

const DEFAULT_ESTIMATED_KITTI: &str = "\
1 0 0 0.00 0 1 0 0.00 0 0 1 0.00
1 0 0 1.05 0 1 0 0.02 0 0 1 0.00
1 0 0 2.10 0 1 0 0.04 0 0 1 0.01
1 0 0 4.30 0 1 0 0.10 0 0 1 0.00
";

const DEFAULT_REFERENCE_KITTI: &str = "\
1 0 0 0.0 0 1 0 0.0 0 0 1 0.0
1 0 0 1.0 0 1 0 0.0 0 0 1 0.0
1 0 0 2.0 0 1 0 0.0 0 0 1 0.0
1 0 0 3.0 0 1 0 0.0 0 0 1 0.0
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    let (estimated_text, reference_text) = match args.as_slice() {
        [] => (
            DEFAULT_ESTIMATED_KITTI.to_string(),
            DEFAULT_REFERENCE_KITTI.to_string(),
        ),
        [estimated_path, reference_path] => (
            fs::read_to_string(estimated_path)?,
            fs::read_to_string(reference_path)?,
        ),
        _ => {
            eprintln!(
                "usage: cargo run --example evaluate_trajectory_from_kitti_files -- [--out-dir <dir>] [estimated_poses.txt reference_poses.txt]"
            );
            std::process::exit(2);
        }
    };

    let estimated = PoseTrajectory::from_kitti_poses_str(&estimated_text)?;
    let reference = PoseTrajectory::from_kitti_poses_str(&reference_text)?;
    let summary = estimated.translation_error_summary_against(&reference);
    let errors_csv = estimated.translation_errors_csv_against(&reference);

    println!(
        "loaded trajectories: estimated={} reference={}",
        estimated.len(),
        reference.len()
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
        estimated.write_translation_errors_csv_against(&reference, &errors_path)?;
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
