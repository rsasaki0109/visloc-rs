//! In-process SuperPoint ONNX throughput benchmark (CPU vs CUDA).
//!
//! Measures the per-frame latency / throughput of the in-Rust SuperPoint
//! front-end (`SuperPointOnnxExtractor`) over a directory of real images,
//! separately for the CPU and the CUDA execution providers. This is the
//! per-stage timing baseline the CUDA-acceleration plan calls for, and the
//! foundation of the "real-time deep-frontend stereo SLAM, pure Rust" claim:
//! it answers "can the deep frontend keep up with the camera, in-process,
//! without the multi-GB Python feature-export step?".
//!
//! Build (CPU only):
//!   cargo run --release --example superpoint_onnx_throughput \
//!       --features "image-io onnx-inference" -- \
//!       --model models/superpoint_1500.onnx --images-dir /tmp/MH_03_rect/image_0 \
//!       --frames 300 --backend cpu
//!
//! Build (with CUDA execution provider):
//!   LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
//!   cargo run --release --example superpoint_onnx_throughput \
//!       --features "image-io onnx-cuda" -- \
//!       --model models/superpoint_1500.onnx --images-dir /tmp/MH_03_rect/image_0 \
//!       --frames 300 --backend both

use std::path::{Path, PathBuf};
use std::time::Instant;

use visloc_rs::io::images::read_common_image;
use visloc_rs::vision::features::deep::DeepFeatureExtractor;
use visloc_rs::vision::features::superpoint_onnx::{
    OnnxBackend, SuperPointOnnxConfig, SuperPointOnnxExtractor,
};
use visloc_rs::vision::features::GrayscaleImage;

struct Args {
    model: PathBuf,
    images_dir: PathBuf,
    frames: usize,
    backend: String,
    max_keypoints: usize,
}

fn parse_args() -> Args {
    let mut model = PathBuf::from("models/superpoint_1500.onnx");
    let mut images_dir = PathBuf::from("/tmp/MH_03_rect/image_0");
    let mut frames = 300usize;
    let mut backend = "both".to_string();
    let mut max_keypoints = 1500usize;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(it.next().expect("--model needs a value")),
            "--images-dir" => {
                images_dir = PathBuf::from(it.next().expect("--images-dir needs a value"))
            }
            "--frames" => frames = it.next().unwrap().parse().expect("--frames int"),
            "--backend" => backend = it.next().expect("--backend cpu|cuda|both"),
            "--max-keypoints" => {
                max_keypoints = it.next().unwrap().parse().expect("--max-keypoints int")
            }
            other => panic!("unknown argument: {other}"),
        }
    }
    Args {
        model,
        images_dir,
        frames,
        backend,
        max_keypoints,
    }
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

fn bench(label: &str, extractor: &SuperPointOnnxExtractor, frames: &[GrayscaleImage]) {
    // Warm up (session allocation / first-run JIT / GPU context).
    let warm = extractor
        .extract_deep(&frames[0])
        .expect("warm-up inference");
    let dim = warm.descriptors.first().map(|d| d.len()).unwrap_or(0);

    let mut total_kp = 0usize;
    let start = Instant::now();
    for img in frames {
        let out = extractor.extract_deep(img).expect("inference");
        total_kp += out.keypoints.len();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let n = frames.len() as f64;
    let per_frame_ms = elapsed / n * 1000.0;
    let fps = n / elapsed;
    let avg_kp = total_kp as f64 / n;
    println!(
        "{label:<14} {n:>5.0} frames  {elapsed:>7.3}s  {per_frame_ms:>7.2} ms/frame  \
         {fps:>7.1} fps   avg_kp {avg_kp:>6.1}  desc_dim {dim}"
    );
}

fn main() {
    let args = parse_args();
    let config = SuperPointOnnxConfig {
        max_keypoints: args.max_keypoints,
        ..Default::default()
    };

    println!(
        "loading up to {} frames from {} ...",
        args.frames,
        args.images_dir.display()
    );
    let frames = load_frames(&args.images_dir, args.frames);
    assert!(!frames.is_empty(), "no images found");
    let (w, h) = (frames[0].width(), frames[0].height());
    println!("loaded {} frames, {}x{}", frames.len(), w, h);
    println!(
        "{:<14} {:>5} {:>11} {:>13} {:>11}   sanity",
        "backend", "n", "wall", "latency", "throughput"
    );

    let run_cpu = args.backend == "cpu" || args.backend == "both";
    let run_cuda = args.backend == "cuda" || args.backend == "both";

    if run_cpu {
        let ex = SuperPointOnnxExtractor::load_from_path_with_backend(
            &args.model,
            config.clone(),
            OnnxBackend::Cpu,
        )
        .expect("load CPU session");
        bench("cpu", &ex, &frames);
    }
    if run_cuda {
        // Strict CUDA: errors if the GPU provider cannot register, so we never
        // report CPU numbers under the "cuda" label.
        let ex = SuperPointOnnxExtractor::load_from_path_with_backend(
            &args.model,
            config.clone(),
            OnnxBackend::Cuda,
        )
        .expect("load CUDA session (GPU provider unavailable?)");
        bench("cuda", &ex, &frames);
    }
}
