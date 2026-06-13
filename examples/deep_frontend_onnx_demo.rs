//! Full in-process deep front-end (SuperPoint + LightGlue, ONNX) demo.
//!
//! Runs the entire learned front-end — SuperPoint feature extraction *and*
//! LightGlue matching — inside the process via ONNX Runtime, with no Python
//! and no pre-exported feature dump. Prints the match count on a sample pair
//! (the correctness signal) and the end-to-end front-end throughput (2×
//! extract + 1× match per pair) for the CPU and CUDA execution providers.
//!
//! Build & run (CPU):
//!   cargo run --release --example deep_frontend_onnx_demo \
//!       --features "image-io onnx-inference" -- \
//!       --superpoint-model models/superpoint_1500.onnx \
//!       --lightglue-model models/lightglue.onnx \
//!       --images-dir /tmp/MH_03_rect/image_0 --pairs 150 --backend cpu
//!
//! With CUDA: use the runner `scripts/run_deep_frontend_onnx_demo.sh`, which
//! sets up the CUDA provider libs + cuDNN LD_LIBRARY_PATH.

use std::path::{Path, PathBuf};
use std::time::Instant;

use visloc_rs::io::images::read_common_image;
use visloc_rs::vision::features::deep::DeepFeatureExtractor;
use visloc_rs::vision::features::lightglue_onnx::{LightGlueMatch, LightGlueOnnxMatcher};
use visloc_rs::vision::features::superpoint_onnx::{
    OnnxBackend, SuperPointOnnxConfig, SuperPointOnnxExtractor,
};
use visloc_rs::vision::features::{deep::DeepFeatureSet, GrayscaleImage};

struct Args {
    superpoint_model: PathBuf,
    lightglue_model: PathBuf,
    images_dir: PathBuf,
    pairs: usize,
    backend: String,
    max_keypoints: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        superpoint_model: PathBuf::from("models/superpoint_1500.onnx"),
        lightglue_model: PathBuf::from("models/lightglue.onnx"),
        images_dir: PathBuf::from("/tmp/MH_03_rect/image_0"),
        pairs: 150,
        backend: "both".to_string(),
        max_keypoints: 1500,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--superpoint-model" => a.superpoint_model = PathBuf::from(it.next().unwrap()),
            "--lightglue-model" => a.lightglue_model = PathBuf::from(it.next().unwrap()),
            "--images-dir" => a.images_dir = PathBuf::from(it.next().unwrap()),
            "--pairs" => a.pairs = it.next().unwrap().parse().expect("--pairs int"),
            "--backend" => a.backend = it.next().unwrap(),
            "--max-keypoints" => a.max_keypoints = it.next().unwrap().parse().unwrap(),
            other => panic!("unknown argument: {other}"),
        }
    }
    a
}

fn load_frames(dir: &Path, limit: usize) -> Vec<GrayscaleImage> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .map(|x| {
                    let x = x.to_ascii_lowercase();
                    x == "png" || x == "jpg" || x == "jpeg" || x == "pgm"
                })
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    paths.truncate(limit);
    paths
        .iter()
        .map(|p| read_common_image(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display())))
        .collect()
}

fn run(
    label: &str,
    sp: &SuperPointOnnxExtractor,
    lg: &LightGlueOnnxMatcher,
    frames: &[GrayscaleImage],
) {
    // Pre-extract all frames once so the matching throughput is isolated, then
    // also report the combined extract+match front-end rate.
    let feats: Vec<DeepFeatureSet> = frames
        .iter()
        .map(|f| sp.extract_deep(f).expect("extract"))
        .collect();

    // Correctness signal: matches on the first consecutive pair.
    let first: Vec<LightGlueMatch> = lg
        .match_features(
            &feats[0].keypoints,
            &feats[0].descriptors,
            &feats[1].keypoints,
            &feats[1].descriptors,
        )
        .expect("match");

    // Matching-only throughput over consecutive pairs.
    let n_pairs = frames.len() - 1;
    let t = Instant::now();
    let mut total_matches = 0usize;
    for i in 0..n_pairs {
        let m = lg
            .match_features(
                &feats[i].keypoints,
                &feats[i].descriptors,
                &feats[i + 1].keypoints,
                &feats[i + 1].descriptors,
            )
            .expect("match");
        total_matches += m.len();
    }
    let match_s = t.elapsed().as_secs_f64();

    // End-to-end front-end (2 extract + 1 match per pair), re-extracting to
    // count the real per-pair cost a streaming pipeline would pay.
    let t = Instant::now();
    for i in 0..n_pairs {
        let f0 = sp.extract_deep(&frames[i]).expect("extract");
        let f1 = sp.extract_deep(&frames[i + 1]).expect("extract");
        let _ = lg
            .match_features(
                &f0.keypoints,
                &f0.descriptors,
                &f1.keypoints,
                &f1.descriptors,
            )
            .expect("match");
    }
    let e2e_s = t.elapsed().as_secs_f64();

    println!(
        "{label:<14} pair0 matches {:>5}  | match-only {:>6.2} ms ({:>6.1} pairs/s, avg {:>5.0} m) \
         | extract+match {:>6.2} ms ({:>5.1} fps)",
        first.len(),
        match_s / n_pairs as f64 * 1000.0,
        n_pairs as f64 / match_s,
        total_matches as f64 / n_pairs as f64,
        e2e_s / n_pairs as f64 * 1000.0,
        n_pairs as f64 / e2e_s,
    );
}

fn main() {
    let args = parse_args();
    let config = SuperPointOnnxConfig {
        max_keypoints: args.max_keypoints,
        ..Default::default()
    };
    let frames = load_frames(&args.images_dir, args.pairs + 1);
    assert!(frames.len() >= 2, "need at least 2 frames");
    println!(
        "loaded {} frames ({}x{}) from {}",
        frames.len(),
        frames[0].width(),
        frames[0].height(),
        args.images_dir.display()
    );

    let backends: Vec<(&str, OnnxBackend)> = match args.backend.as_str() {
        "cpu" => vec![("cpu", OnnxBackend::Cpu)],
        "cuda" => vec![("cuda", OnnxBackend::Cuda)],
        _ => vec![("cpu", OnnxBackend::Cpu), ("cuda", OnnxBackend::Cuda)],
    };
    for (label, backend) in backends {
        let sp = SuperPointOnnxExtractor::load_from_path_with_backend(
            &args.superpoint_model,
            config.clone(),
            backend,
        )
        .expect("load SuperPoint");
        let lg = LightGlueOnnxMatcher::load_from_path_with_backend(&args.lightglue_model, backend)
            .expect("load LightGlue");
        run(label, &sp, &lg, &frames);
    }
}
