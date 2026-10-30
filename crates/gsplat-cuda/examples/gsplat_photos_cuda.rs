//! Photos to a COLMAP dataset (`build_photo_dataset`, `gsplat_photos
//! --steps 0` on the CPU) with the descriptor matching on the GPU
//! (`matching::CudaPairMatcher`): CPU SIFT, CUDA matching, CPU verification
//! and mapping. Train the dataset with `gsplat_train_cuda --data <out>`.
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --features photos --example gsplat_photos_cuda -- \
//!   --images <folder> --out <work dir> [--max-size 1600] [--focal F] \
//!   [--exhaustive-max 300] [--window 20] [--max-keypoints 8192] [--cpu-match]
//! ```
//!
//! `--cpu-match` matches on the CPU instead, for comparison: the matches are
//! the same, bit for bit.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use visloc_gsplat_cuda::matching::CudaPairMatcher;
use visloc_gsplat_cuda::CudaGsplat;
use visloc_gsplat_train::photos::{build_photo_dataset, PhotoSfmConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut images: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    // The CPU path of gsplat_photos (no wgpu): SIFT and verification on the CPU.
    let mut sfm = PhotoSfmConfig {
        gpu: false,
        gpu_sift: false,
        ..Default::default()
    };
    let mut cpu_match = false;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--images" => images = Some(PathBuf::from(next()?)),
            "--out" => out = Some(PathBuf::from(next()?)),
            "--max-size" => sfm.max_size = next()?.parse()?,
            "--focal" => sfm.focal_px = Some(next()?.parse()?),
            "--exhaustive-max" => sfm.exhaustive_max = next()?.parse()?,
            "--window" => sfm.window = next()?.parse()?,
            "--max-keypoints" => sfm.sift_max_keypoints = next()?.parse()?,
            "--cpu-match" => cpu_match = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let images = images.ok_or("--images <folder> is required")?;
    let out = out.ok_or("--out <work dir> is required")?;
    std::fs::create_dir_all(&out)?;
    if !cpu_match {
        let gs = CudaGsplat::from_env().ok_or(
            "set VISLOC_GSPLAT_CUDA_LIB to the library built by scripts/build_gsplat_cuda.sh",
        )??;
        sfm.matcher = Some(Arc::new(CudaPairMatcher::new(gs)));
    }
    let t0 = Instant::now();
    let (_, _, report) = build_photo_dataset(&images, &out, &sfm, &mut |m: &str| println!("{m}"))?;
    println!(
        "sfm done in {:.1}s: {}/{} registered, focal {:.1} -> {:.1} px ({}), {} points",
        t0.elapsed().as_secs_f64(),
        report.registered,
        report.images,
        report.focal_prior,
        report.focal,
        report.focal_source,
        report.points
    );
    Ok(())
}
