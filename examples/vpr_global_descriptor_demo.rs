//! Stream learned VPR descriptors into a restart-safe, mmap-readable store.
//!
//! With `--rig-manifest`, one row represents one rig frame: every sensor image
//! is inferred independently, then the mean descriptor is L2-normalized. This
//! prevents a stereo rig from consuming two ANN nodes and keeps retrieval at
//! the mapper's frame granularity. Without a rig manifest, one sorted image is
//! one row.
//!
//! The output is installed only when complete. Its fixed header binds the rows
//! to the ONNX SHA-256, canonical input order, resize dimensions, and exact
//! preprocessing protocol. A stopped run resumes from `<out>.partial` after
//! validating all completed rows; descriptors are never retained as a matrix.
//!
//! Example (EigenPlaces ResNet18/512, OpenLORIS):
//!   cargo run --release --features "image-io onnx-cuda" \
//!     --example vpr_global_descriptor_demo -- \
//!       --images-dir /data/openloris/images --subdir . \
//!       --rig-manifest /data/openloris/rig-manifest.txt \
//!       --model /models/eigenplaces_resnet18_512.onnx \
//!       --out /data/openloris/eigenplaces-r18-512.vprd

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use visloc_rs::global_descriptor_onnx::{GlobalDescriptorOnnxExtractor, OnnxBackend};
use visloc_rs::global_descriptor_store::{GlobalDescriptorBinding, GlobalDescriptorWriter};
use visloc_rs::io::images::read_common_image;
use visloc_rs::GrayscaleImage;

const PREPROCESSING_PROTOCOL: &str =
    "v1:luma-f32-[0,1];resize=bilinear-half-pixel-no-antialias;rgb=repeat-luma;rig=mean-then-l2";

struct Args {
    images_dir: PathBuf,
    subdir: String,
    rig_manifest: Option<PathBuf>,
    model: PathBuf,
    out: PathBuf,
    frames: Option<usize>,
    resize_width: usize,
    resize_height: usize,
    checkpoint_every: usize,
    onnx_cpu: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            images_dir: PathBuf::from("/tmp/images"),
            subdir: ".".to_owned(),
            rig_manifest: None,
            model: PathBuf::from("models/eigenplaces_r18_512.onnx"),
            out: PathBuf::from("/tmp/globals.vprd"),
            frames: None,
            resize_width: 640,
            resize_height: 480,
            checkpoint_every: 32,
            onnx_cpu: false,
        }
    }
}

#[derive(Debug)]
struct FrameInput {
    canonical: String,
    images: Vec<PathBuf>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut next = || it.next().ok_or_else(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--images-dir" => args.images_dir = PathBuf::from(next()?),
            "--subdir" => args.subdir = next()?,
            "--rig-manifest" => args.rig_manifest = Some(PathBuf::from(next()?)),
            "--model" => args.model = PathBuf::from(next()?),
            "--out" => args.out = PathBuf::from(next()?),
            "--frames" => args.frames = Some(next()?.parse()?),
            "--resize-width" => args.resize_width = next()?.parse()?,
            "--resize-height" => args.resize_height = next()?.parse()?,
            "--checkpoint-every" => args.checkpoint_every = next()?.parse()?,
            "--onnx-cpu" => args.onnx_cpu = true,
            "-h" | "--help" => {
                println!(
                    "vpr_global_descriptor_demo --images-dir DIR [--subdir .] \
                     [--rig-manifest PATH] --model M.onnx --out globals.vprd \
                     [--frames N] [--resize-width 640] [--resize-height 480] \
                     [--checkpoint-every 32] [--onnx-cpu]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}").into()),
        }
    }
    if args.resize_width == 0 || args.resize_height == 0 || args.checkpoint_every == 0 {
        return Err("resize dimensions and checkpoint interval must be positive".into());
    }
    Ok(args)
}

fn list_frames(dir: &Path) -> Result<Vec<FrameInput>, Box<dyn Error>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| is_image(path))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .enumerate()
        .map(|(index, path)| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("non-UTF8 image filename: {}", path.display()))?;
            Ok(FrameInput {
                canonical: format!("frame {index} 0:{name}"),
                images: vec![path],
            })
        })
        .collect::<Result<_, String>>()
        .map_err(Into::into)
}

fn rig_frames(manifest: &Path, image_root: &Path) -> Result<Vec<FrameInput>, Box<dyn Error>> {
    let text = std::fs::read_to_string(manifest)?;
    let mut rows: BTreeMap<u64, BTreeMap<usize, String>> = BTreeMap::new();
    for (zero_line, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || !line.starts_with("F ") {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 4 {
            return Err(format!("rig manifest line {} has malformed F row", zero_line + 1).into());
        }
        let frame_id: u64 = fields[1].parse()?;
        let sensor: usize = fields[3].parse()?;
        if rows
            .entry(frame_id)
            .or_default()
            .insert(sensor, fields[2].to_owned())
            .is_some()
        {
            return Err(format!("rig frame {frame_id} assigns sensor {sensor} twice").into());
        }
    }
    if rows.is_empty() {
        return Err(format!("rig manifest {} contains no F rows", manifest.display()).into());
    }
    let mut result = Vec::with_capacity(rows.len());
    let expected_sensors = rows
        .first_key_value()
        .expect("non-empty rows")
        .1
        .keys()
        .copied()
        .collect::<Vec<_>>();
    for (expected, (frame_id, sensors)) in rows.into_iter().enumerate() {
        if frame_id != expected as u64 {
            return Err(format!(
                "rig frame ids must be contiguous: expected {expected}, got {frame_id}"
            )
            .into());
        }
        if sensors.keys().copied().ne(expected_sensors.iter().copied()) {
            return Err(format!(
                "rig frame {frame_id} sensor set differs from frame 0: expected {expected_sensors:?}"
            )
            .into());
        }
        let canonical = format!(
            "frame {frame_id} {}",
            sensors
                .iter()
                .map(|(sensor, name)| format!("{sensor}:{name}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let images = sensors
            .into_values()
            .map(|name| image_root.join(name))
            .collect();
        result.push(FrameInput { canonical, images });
    }
    Ok(result)
}

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "pgm" | "bmp")
    )
}

fn resize_bilinear(image: &GrayscaleImage, width: usize, height: usize) -> GrayscaleImage {
    if image.width() == width && image.height() == height {
        return image.clone();
    }
    let mut pixels = vec![0.0_f32; width * height];
    let scale_x = image.width() as f64 / width as f64;
    let scale_y = image.height() as f64 / height as f64;
    for y in 0..height {
        let source_y = ((y as f64 + 0.5) * scale_y - 0.5).clamp(0.0, image.height() as f64 - 1.0);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(image.height() - 1);
        let fy = (source_y - y0 as f64) as f32;
        for x in 0..width {
            let source_x =
                ((x as f64 + 0.5) * scale_x - 0.5).clamp(0.0, image.width() as f64 - 1.0);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(image.width() - 1);
            let fx = (source_x - x0 as f64) as f32;
            let top = image.get(x0, y0).unwrap() * (1.0 - fx) + image.get(x1, y0).unwrap() * fx;
            let bottom = image.get(x0, y1).unwrap() * (1.0 - fx) + image.get(x1, y1).unwrap() * fx;
            pixels[y * width + x] = top * (1.0 - fy) + bottom * fy;
        }
    }
    GrayscaleImage::new(width, height, pixels).expect("positive dimensions and exact pixel count")
}

fn frame_descriptor(
    extractor: &GlobalDescriptorOnnxExtractor,
    frame: &FrameInput,
    width: usize,
    height: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut mean = Vec::<f32>::new();
    for path in &frame.images {
        let resized = resize_bilinear(&read_common_image(path)?, width, height);
        let descriptor = extractor.extract_global(&resized)?;
        if mean.is_empty() {
            mean.resize(descriptor.len(), 0.0);
        } else if mean.len() != descriptor.len() {
            return Err("model returned inconsistent descriptor dimensions".into());
        }
        for (sum, value) in mean.iter_mut().zip(descriptor) {
            *sum += value;
        }
    }
    let norm = mean.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err("rig-frame mean descriptor has invalid norm".into());
    }
    for value in &mut mean {
        *value /= norm;
    }
    Ok(mean)
}

fn sha256_file(path: &Path) -> Result<[u8; 32], Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().into())
}

fn sha256_lines(frames: &[FrameInput]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for frame in frames {
        hasher.update(frame.canonical.as_bytes());
        hasher.update(b"\n");
    }
    hasher.finalize().into()
}

fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let backend = if args.onnx_cpu {
        OnnxBackend::Cpu
    } else {
        OnnxBackend::CudaThenCpu
    };
    let image_root = args.images_dir.join(&args.subdir);
    let mut frames = if let Some(manifest) = &args.rig_manifest {
        rig_frames(manifest, &image_root)?
    } else {
        list_frames(&image_root)?
    };
    if let Some(limit) = args.frames {
        frames.truncate(limit);
    }
    if frames.is_empty() {
        return Err("no input frames".into());
    }
    for frame in &frames {
        for path in &frame.images {
            if !path.is_file() {
                return Err(format!("missing input image {}", path.display()).into());
            }
        }
    }

    let model_sha256 = sha256_file(&args.model)?;
    let manifest_sha256 = sha256_lines(&frames);
    let preprocessing_sha256 = Sha256::digest(PREPROCESSING_PROTOCOL.as_bytes()).into();
    println!(
        "loading {} backend={backend:?} rows={} resize={}x{} model_sha256={} manifest_sha256={} preprocessing_sha256={}",
        args.model.display(), frames.len(), args.resize_width, args.resize_height,
        hex(&model_sha256), hex(&manifest_sha256), hex(&preprocessing_sha256)
    );
    let extractor =
        GlobalDescriptorOnnxExtractor::load_from_path_with_backend(&args.model, backend)?;

    // One inference determines D before the fixed header is created. Reusing it
    // when row zero is incomplete avoids retaining more than one descriptor.
    let first = frame_descriptor(
        &extractor,
        &frames[0],
        args.resize_width,
        args.resize_height,
    )?;
    let binding = GlobalDescriptorBinding {
        row_count: frames.len() as u64,
        dimension: first.len().try_into()?,
        resize_width: args.resize_width.try_into()?,
        resize_height: args.resize_height.try_into()?,
        model_sha256,
        manifest_sha256,
        preprocessing_sha256,
    };
    let mut writer = GlobalDescriptorWriter::begin_or_resume(&args.out, binding)?;
    let start = writer.completed() as usize;
    println!(
        "streaming descriptors from row {start} (checkpoint every {})",
        args.checkpoint_every
    );
    for index in start..frames.len() {
        let descriptor = if index == 0 {
            first.clone()
        } else {
            frame_descriptor(
                &extractor,
                &frames[index],
                args.resize_width,
                args.resize_height,
            )?
        };
        writer.append(&descriptor)?;
        if (index + 1) % args.checkpoint_every == 0 || index + 1 == frames.len() {
            writer.checkpoint()?;
            println!("  {}/{}", index + 1, frames.len());
        }
    }
    writer.finalize()?;
    println!(
        "installed {} rows, dim {} -> {}",
        frames.len(),
        first.len(),
        args.out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_resize_preserves_constant_image() {
        let image = GrayscaleImage::new(3, 2, vec![0.25; 6]).unwrap();
        let resized = resize_bilinear(&image, 7, 5);
        assert!(resized
            .pixels()
            .iter()
            .all(|value| (*value - 0.25).abs() < 1e-7));
    }

    #[test]
    fn canonical_hash_binds_frame_and_sensor_order() {
        let frame = |canonical: &str| FrameInput {
            canonical: canonical.to_owned(),
            images: Vec::new(),
        };
        assert_ne!(
            sha256_lines(&[frame("frame 0 0:a 1:b")]),
            sha256_lines(&[frame("frame 0 0:b 1:a")])
        );
        assert_ne!(
            sha256_lines(&[frame("frame 0 0:a")]),
            sha256_lines(&[frame("frame 1 0:a")])
        );
    }
}
