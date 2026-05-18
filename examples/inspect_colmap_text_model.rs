use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use visloc_rs::io::colmap::read_colmap_text_model;

fn print_usage() {
    eprintln!(
        "usage: inspect_colmap_text_model <model-dir>\n\
         \n\
         Loads the COLMAP text model in <model-dir> (expects cameras.txt,\n\
         images.txt, points3D.txt) via read_colmap_text_model and prints\n\
         summary counts. Mirrors inspect_colmap_binary_model on the text\n\
         side so the KITTI -> COLMAP -> 3DGS smoke harness can round-trip\n\
         both writer surfaces on real driving data and assert cross-format\n\
         parity at script level."
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

    let map = match read_colmap_text_model(&model_dir) {
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
        eprintln!("error: text model is missing cameras or keyframes");
        return ExitCode::from(1);
    }

    ExitCode::from(0)
}
