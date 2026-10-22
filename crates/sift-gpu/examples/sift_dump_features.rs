//! Dump GPU SIFT keypoints (x, y, sigma, orientation, octave-est, descriptor)
//! for a handful of named frames, in a text format easy to diff against a
//! COLMAP database export. Diagnostic only.
//!
//! cargo run --release -p visloc-sift-gpu --features gpu --example sift_dump_features -- \
//!     --images <dir> --frames 0,40,55,56,57 --out <dir> \
//!     [--keypoints 4000] [--l1-root] [--magnification 3] [--max-orientations 2] [--prefer-larger-scale] [--full-pyramid]

use std::path::PathBuf;

use visloc_sift_gpu::{GpuContext, SiftGpu};
use visloc_vision::features::sift::{GrayImage, SiftConfig, SiftNormalization};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}
fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--images").expect("--images <dir>"));
    let out_dir = PathBuf::from(arg(&args, "--out").expect("--out <dir>"));
    let frames: Vec<usize> = arg(&args, "--frames")
        .expect("--frames 0,1,2")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    std::fs::create_dir_all(&out_dir).unwrap();

    let mut config = SiftConfig::default();
    if let Some(v) = arg(&args, "--keypoints") {
        config.max_keypoints = v.parse().unwrap();
    }
    if flag(&args, "--l1-root") {
        config.normalization = SiftNormalization::L1Root;
    }
    if let Some(v) = arg(&args, "--magnification") {
        config.descriptor_magnification = v.parse().unwrap();
    }
    if let Some(v) = arg(&args, "--max-orientations") {
        config.max_orientations = v.parse().unwrap();
    }
    if flag(&args, "--prefer-larger-scale") {
        config.prefer_larger_scale = true;
    }
    if flag(&args, "--full-pyramid") {
        config.full_pyramid = true;
    }
    if flag(&args, "--aligned-octave0") {
        config.aligned_octave0_upsample = true;
    }
    if flag(&args, "--subpixel") {
        config.subpixel_localization = true;
    }

    let mut gpu = SiftGpu::new(GpuContext::new().expect("gpu"));
    eprintln!("gpu: {}", gpu.context().adapter_info.name);

    for &f in &frames {
        let path = dir.join(format!("frame_{f:05}.png"));
        let gray = image::open(&path)
            .unwrap_or_else(|e| panic!("{path:?}: {e}"))
            .to_luma8();
        let (w, h) = (gray.width() as usize, gray.height() as usize);
        let pixels: Vec<f32> = gray.as_raw().iter().map(|&b| b as f32).collect();
        let img = GrayImage::new(w, h, &pixels).unwrap();
        let (kps, descs) = gpu.extract(&img, &config).unwrap();
        // Approximate octave from sigma (o=0 upsample=0.5, o=1 upsample=1, ...):
        // sigma = sigma_base * k^level * 2^(o-1), so o ~= round(log2(sigma/sigma_base)+1).
        let mut text = String::from("# x y sigma orientation_rad octave_est d0..d127\n");
        for (k, d) in kps.iter().zip(descs.iter()) {
            let o_est = (k.sigma / config.sigma_base).log2().round() as i64 + 1;
            text.push_str(&format!(
                "{:.6} {:.6} {:.6} {:.6} {}",
                k.x, k.y, k.sigma, k.orientation, o_est
            ));
            for v in d {
                text.push_str(&format!(" {v:.9}"));
            }
            text.push('\n');
        }
        let out_path = out_dir.join(format!("frame_{f:05}_ours.txt"));
        std::fs::write(&out_path, text).unwrap();
        eprintln!("frame_{f:05}: {} kps -> {out_path:?}", kps.len());
    }
}
