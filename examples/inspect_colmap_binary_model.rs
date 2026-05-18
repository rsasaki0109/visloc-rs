use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use visloc_rs::io::colmap::read_colmap_binary_model;

fn print_usage() {
    eprintln!(
        "usage: inspect_colmap_binary_model <model-dir>\n\
         \n\
         Loads the COLMAP binary model in <model-dir> (expects cameras.bin,\n\
         images.bin, points3D.bin) via read_colmap_binary_model and prints\n\
         summary counts. Intended as a writer/reader round-trip check for\n\
         the 3DGS bootstrap output of online_slam_stereo_vo_kitti_demo\n\
         --colmap-export-binary."
    );
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let model_dir = match args.next() {
        Some(arg) if arg == "-h" || arg == "--help" => {
            print_usage();
            return ExitCode::from(0);
        }
        Some(arg) => PathBuf::from(arg),
        None => {
            print_usage();
            return ExitCode::from(2);
        }
    };

    let map = match read_colmap_binary_model(&model_dir) {
        Ok(map) => map,
        Err(err) => {
            eprintln!("failed to read {}: {err}", model_dir.display());
            return ExitCode::from(1);
        }
    };

    let observations: usize = map.keyframes.values().map(|kf| kf.observations.len()).sum();

    println!("dir={}", model_dir.display());
    println!("cameras={}", map.cameras.len());
    println!("keyframes={}", map.keyframes.len());
    println!("landmarks={}", map.landmarks.len());
    println!("observations={}", observations);

    if map.cameras.is_empty() || map.keyframes.is_empty() {
        eprintln!("error: binary model is missing cameras or keyframes");
        return ExitCode::from(1);
    }

    ExitCode::from(0)
}
