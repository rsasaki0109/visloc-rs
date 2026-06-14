//! Compute learned global descriptors (visual place recognition) for an image
//! sequence with the in-process EigenPlaces / CosPlace ONNX runtime, and write
//! them to a text file for use as the loop-closure / relocalisation retrieval
//! front-end.
//!
//! This is the *learned* counterpart to the hand-built k-means VLAD that
//! `close_loops_on_vo_trajectory` computes internally over local SuperPoint
//! descriptors. One pass here (GPU, in-process — no Python) produces one
//! L2-normalised descriptor per frame; the stereo-VO loop-closure benchmark
//! consumes the file via `--global-descriptor-file` and calls
//! `close_loops_on_vo_trajectory_with_globals`, so VLAD-vs-learned retrieval is a
//! clean A/B from the same trajectory and the same loop verifier.
//!
//! Output format: one line per frame in directory order, each line a
//! space-separated list of `D` float32 values (the model's descriptor
//! dimension, e.g. 2048 for EigenPlaces ResNet50/2048). Already L2-normalised,
//! so a dot product between two lines is their cosine similarity.
//!
//! Build the model once:
//!   scripts/export_vpr_onnx.py --out models/eigenplaces_r50_2048.onnx
//!
//! Run (CUDA by default; falls back to CPU if the GPU provider can't register):
//!   cargo run --release --features "image-io onnx-cuda" \
//!     --example vpr_global_descriptor_demo -- \
//!       --images-dir ~/datasets/kitti_seq02_full --subdir image_0 \
//!       --model models/eigenplaces_r50_2048.onnx \
//!       --out /tmp/seq02_eigenplaces.txt

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use visloc_rs::global_descriptor_onnx::{GlobalDescriptorOnnxExtractor, OnnxBackend};
use visloc_rs::io::images::read_common_image;

struct Args {
    images_dir: PathBuf,
    subdir: String,
    model: PathBuf,
    out: PathBuf,
    frames: Option<usize>,
    onnx_cpu: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            images_dir: PathBuf::from("/tmp/MH_03_rect"),
            subdir: "image_0".to_string(),
            model: PathBuf::from("models/eigenplaces_r50_2048.onnx"),
            out: PathBuf::from("/tmp/globals.txt"),
            frames: None,
            onnx_cpu: false,
        }
    }
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut next = || it.next().ok_or_else(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--images-dir" => args.images_dir = PathBuf::from(next()?),
            "--subdir" => args.subdir = next()?,
            "--model" => args.model = PathBuf::from(next()?),
            "--out" => args.out = PathBuf::from(next()?),
            "--frames" => args.frames = Some(next()?.parse()?),
            "--onnx-cpu" => args.onnx_cpu = true,
            "-h" | "--help" => {
                println!(
                    "vpr_global_descriptor_demo --images-dir DIR [--subdir image_0] \
                     --model M.onnx --out globals.txt [--frames N] [--onnx-cpu]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}").into()),
        }
    }
    Ok(args)
}

/// List the image files in a directory, sorted by file name (frame order).
fn list_frames(dir: &PathBuf) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("png" | "jpg" | "jpeg" | "pgm" | "bmp")
            )
        })
        .collect();
    names.sort();
    Ok(names)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let backend = if args.onnx_cpu {
        OnnxBackend::Cpu
    } else {
        OnnxBackend::CudaThenCpu
    };
    println!(
        "loading VPR model {} (backend {:?})",
        args.model.display(),
        backend
    );
    let extractor =
        GlobalDescriptorOnnxExtractor::load_from_path_with_backend(&args.model, backend)?;

    let dir = args.images_dir.join(&args.subdir);
    let mut frames = list_frames(&dir)?;
    if let Some(limit) = args.frames {
        frames.truncate(limit);
    }
    if frames.is_empty() {
        return Err(format!("no images found in {}", dir.display()).into());
    }
    println!("computing global descriptors for {} frames", frames.len());

    let out_file = File::create(&args.out)?;
    let mut writer = BufWriter::new(out_file);
    let mut descriptors: Vec<Vec<f32>> = Vec::with_capacity(frames.len());

    for (i, path) in frames.iter().enumerate() {
        let image = read_common_image(path)?;
        let descriptor = extractor.extract_global(&image)?;
        let line: Vec<String> = descriptor.iter().map(|v| format!("{v:.7}")).collect();
        writeln!(writer, "{}", line.join(" "))?;
        descriptors.push(descriptor);
        if i % 250 == 0 || i + 1 == frames.len() {
            println!("  {}/{}", i + 1, frames.len());
        }
    }
    writer.flush()?;

    let dim = descriptors[0].len();
    println!(
        "wrote {} descriptors (dim {dim}) to {}",
        descriptors.len(),
        args.out.display()
    );

    // Sanity: adjacent frames should be much more similar than a distant pair.
    if descriptors.len() >= 3 {
        let adj = cosine(&descriptors[0], &descriptors[1]);
        let far = cosine(&descriptors[0], &descriptors[descriptors.len() - 1]);
        println!("sanity: cos(frame0, frame1)={adj:.4}  cos(frame0, last)={far:.4}");
    }

    Ok(())
}
