//! Any folder of photos -> posed dataset, with no COLMAP or Python.
//!
//! The EuRoC pipeline's SfM stages ([`crate::euroc`]) on unordered or video
//! photos of unknown calibration:
//!
//! 1. Decode every image (EXIF orientation applied), scale so the long side
//!    is at most `max_size`, keep the most common resolution (the trainer
//!    renders one camera), and write it to `out_dir/images`.
//! 2. Focal length from the EXIF 35 mm equivalent when present, else
//!    COLMAP's prior of 1.2 x the long side; principal point at the centre.
//! 3. SIFT (RootSIFT, the settings that beat COLMAP on EuRoC), GPU matching,
//!    two-view verification: exhaustive pairs for up to `exhaustive_max`
//!    images, a sliding window over the sorted names beyond.
//! 4. The COLMAP incremental-mapper port, then one global bundle adjustment
//!    that also refines the shared intrinsics (the port keeps them fixed).

use std::path::{Path, PathBuf};

use nalgebra::{Point2, Vector3};
use rayon::prelude::*;
use visloc_core::types::Camera;
use visloc_gsplat_core::colmap_scene::camera_view_from;
use visloc_gsplat_core::gaussian::Scene;
use visloc_slam::{IncrementalSfmConfig, PairwiseMatches};
use visloc_vision::features::sift::{extract_sift, GrayImage, SiftConfig};
use visloc_vision::features::FeatureSet;
use visloc_vision::matching::DescriptorMatch;

use crate::dataset::{Dataset, View};
use crate::euroc::{apply_sift_override, cpu_matches, run_colmap_port, verify_pair, EurocError};
use crate::init::ColoredPoint;

/// Settings for [`build_photo_dataset`].
#[derive(Debug, Clone)]
pub struct PhotoSfmConfig {
    /// Long side of the working images, in pixels.
    pub max_size: u32,
    /// Focal length in working-image pixels (overrides EXIF).
    pub focal_px: Option<f64>,
    /// Match every pair up to this many images, else a sliding window.
    pub exhaustive_max: usize,
    /// Sliding-window size (images) past `exhaustive_max`.
    pub window: usize,
    pub min_matches: usize,
    pub sift_max_keypoints: usize,
    /// Every n-th registered image is held out for evaluation (0: none).
    pub eval_every: usize,
    pub gpu: bool,
    /// Refine the shared intrinsics in a final global bundle adjustment.
    pub refine_intrinsics: bool,
    /// Skip that refinement above this many registered images (its global
    /// bundle adjustments dominate the SfM time on large models).
    pub refine_intrinsics_max_images: usize,
    /// Past `exhaustive_max`, also match every image with its `retrieval_k`
    /// most similar images by VLAD appearance (0: window only). Connects
    /// revisits that are far apart in file order, e.g. several passes
    /// through the same streets.
    pub retrieval_k: usize,
    /// Extra SIFT `key=value` settings (see `euroc::apply_sift_override`),
    /// e.g. `affine=1` / `domain_size_pooling=1` for wide-baseline captures.
    pub sift_overrides: Vec<String>,
}

impl Default for PhotoSfmConfig {
    fn default() -> Self {
        Self {
            max_size: 1600,
            focal_px: None,
            exhaustive_max: 300,
            window: 20,
            min_matches: 15,
            sift_max_keypoints: 4000,
            eval_every: 8,
            gpu: true,
            refine_intrinsics: true,
            refine_intrinsics_max_images: usize::MAX,
            retrieval_k: 0,
            sift_overrides: Vec::new(),
        }
    }
}

/// What [`build_photo_dataset`] did.
#[derive(Debug, Clone)]
pub struct PhotoSfmReport {
    pub images: usize,
    pub width: u32,
    pub height: u32,
    /// Initial focal length (px) and where it came from.
    pub focal_prior: f64,
    pub focal_source: &'static str,
    /// Final focal length (px) after refinement.
    pub focal: f64,
    pub pairs: usize,
    pub registered: usize,
    pub points: usize,
    pub mean_reprojection_px: f64,
}

/// Candidate pairs and their descriptor matches, awaiting verification.
type MatchedChunk<'a> = (&'a [(usize, usize)], Vec<Vec<DescriptorMatch>>);

const IMAGE_EXTENSIONS: [&str; 4] = ["jpg", "jpeg", "png", "JPG"];

/// Image pairs `(i, j)`, `i < j`, joining every image to its `k` most similar
/// images by the cosine similarity of VLAD descriptors over a 64-word
/// vocabulary built from a subsample of all descriptors.
fn retrieval_pairs(features: &[FeatureSet], k: usize) -> Vec<(usize, usize)> {
    use visloc_vision::place_recognition::{vlad, Vocabulary};
    // About 100k descriptors for the vocabulary, evenly from every image.
    let total: usize = features.iter().map(|f| f.descriptors.len()).sum();
    let step = (total / 100_000).max(1);
    let sample: Vec<&[f32]> = features
        .iter()
        .flat_map(|f| f.descriptors.iter().step_by(step).map(Vec::as_slice))
        .collect();
    let Some(vocab) = Vocabulary::build(&sample, 64, 10, 7) else {
        return Vec::new();
    };
    let global: Vec<Vec<f32>> = features
        .par_iter()
        .map(|f| vlad(&f.descriptors, &vocab))
        .collect();
    top_k_similar_pairs(&global, k)
}

/// Pairs `(i, j)`, `i < j`, joining every row of `global` to the `k` rows
/// with the largest dot product. The similarities are computed as blocks of
/// `global · globalᵀ` (a GEMM per block of rows), not row by row: a
/// row-by-row scan streams every descriptor once per image, which is
/// memory-bound past a few thousand images.
fn top_k_similar_pairs(global: &[Vec<f32>], k: usize) -> Vec<(usize, usize)> {
    const ROWS: usize = 64;
    let n = global.len();
    let dim = global.first().map_or(0, Vec::len);
    if n < 2 || dim == 0 || global.iter().any(|g| g.len() != dim) {
        return Vec::new();
    }
    // Column j of `all` is descriptor j.
    let all = nalgebra::DMatrix::<f32>::from_fn(dim, n, |r, c| global[c][r]);
    let starts: Vec<usize> = (0..n).step_by(ROWS).collect();
    let mut pairs: Vec<(usize, usize)> = starts
        .into_par_iter()
        .flat_map_iter(|r0| {
            let r1 = (r0 + ROWS).min(n);
            let rows = nalgebra::DMatrix::<f32>::from_fn(r1 - r0, dim, |r, c| global[r0 + r][c]);
            let sims = rows * &all;
            (r0..r1)
                .flat_map(|i| {
                    let mut row: Vec<(f32, usize)> = (0..n)
                        .filter(|&j| j != i)
                        .map(|j| (sims[(i - r0, j)], j))
                        .collect();
                    let kk = k.min(row.len());
                    if kk > 0 {
                        row.select_nth_unstable_by(kk - 1, |a, b| b.0.total_cmp(&a.0));
                    }
                    row.truncate(kk);
                    row.into_iter().map(move |(_, j)| (i.min(j), i.max(j)))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// EXIF `FocalLengthIn35mmFilm` of a JPEG, if present.
pub fn exif_focal_35mm(bytes: &[u8]) -> Option<f64> {
    // JPEG segments up to the first APP1 "Exif\0\0".
    let mut p = 2usize;
    if bytes.get(0..2)? != [0xFF, 0xD8] {
        return None;
    }
    let tiff = loop {
        if bytes.get(p)? != &0xFF {
            return None;
        }
        let marker = *bytes.get(p + 1)?;
        let len = u16::from_be_bytes([*bytes.get(p + 2)?, *bytes.get(p + 3)?]) as usize;
        if marker == 0xE1 && bytes.get(p + 4..p + 10)? == b"Exif\0\0" {
            break bytes.get(p + 10..p + 2 + len)?;
        }
        if marker == 0xDA {
            return None;
        }
        p += 2 + len;
    };
    let le = match tiff.get(0..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b = [*tiff.get(o)?, *tiff.get(o + 1)?];
        Some(if le {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b = [
            *tiff.get(o)?,
            *tiff.get(o + 1)?,
            *tiff.get(o + 2)?,
            *tiff.get(o + 3)?,
        ];
        Some(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    };
    // Tag -> value offset within an IFD.
    let find = |ifd: usize, tag: u16| -> Option<usize> {
        let n = u16_at(ifd)? as usize;
        (0..n)
            .map(|k| ifd + 2 + 12 * k)
            .find(|&e| u16_at(e) == Some(tag))
            .map(|e| e + 8)
    };
    let ifd0 = u32_at(4)? as usize;
    let exif_ifd = u32_at(find(ifd0, 0x8769)?)? as usize;
    let f35 = u16_at(find(exif_ifd, 0xA405)?)? as f64;
    (f35 > 0.0).then_some(f35)
}

fn load_oriented(path: &Path) -> Result<image::RgbImage, EurocError> {
    use image::ImageDecoder;
    let err = |source| EurocError::Image {
        path: path.to_path_buf(),
        source,
    };
    let mut decoder = image::ImageReader::open(path)
        .map_err(|source| EurocError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .with_guessed_format()
        .map_err(|source| EurocError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .into_decoder()
        .map_err(err)?;
    let orientation = decoder.orientation().map_err(err)?;
    let mut img = image::DynamicImage::from_decoder(decoder).map_err(err)?;
    img.apply_orientation(orientation);
    Ok(img.to_rgb8())
}

/// Run the pipeline on the images in `images_dir`, writing the working
/// images to `out_dir/images`. Returns the dataset, the coloured SfM points
/// and a report.
pub fn build_photo_dataset(
    images_dir: &Path,
    out_dir: &Path,
    cfg: &PhotoSfmConfig,
    log: &mut dyn FnMut(&str),
) -> Result<(Dataset, Vec<ColoredPoint>, PhotoSfmReport), EurocError> {
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| EurocError::Io { path, source }
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(images_dir)
        .map_err(io(images_dir))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        })
        .collect();
    files.sort();
    if files.len() < 3 {
        return Err(EurocError::Sfm(format!(
            "need at least 3 images in {}, found {}",
            images_dir.display(),
            files.len()
        )));
    }

    // Decode (oriented), scale, and read the EXIF focal in parallel.
    let decoded: Vec<(image::RgbImage, Option<f64>)> = files
        .par_iter()
        .map(|p| -> Result<_, EurocError> {
            let img = load_oriented(p)?;
            let (w, h) = img.dimensions();
            let s = (cfg.max_size as f64 / w.max(h) as f64).min(1.0);
            let img = if s < 1.0 {
                image::imageops::resize(
                    &img,
                    ((w as f64 * s).round() as u32).max(1),
                    ((h as f64 * s).round() as u32).max(1),
                    image::imageops::FilterType::Triangle,
                )
            } else {
                img
            };
            let f35 = std::fs::read(p).ok().and_then(|b| exif_focal_35mm(&b));
            Ok((img, f35))
        })
        .collect::<Result<_, _>>()?;
    // The most common resolution (one camera for the trainer).
    let mut sizes: Vec<(u32, u32)> = decoded.iter().map(|d| d.0.dimensions()).collect();
    sizes.sort_unstable();
    let (width, height) = sizes
        .chunk_by(|a, b| a == b)
        .max_by_key(|c| c.len())
        .map(|c| c[0])
        .expect("at least 3 images");
    let (files, decoded): (Vec<_>, Vec<_>) = files
        .into_iter()
        .zip(decoded)
        .filter(|(_, d)| d.0.dimensions() == (width, height))
        .unzip();
    let n = files.len();
    log(&format!(
        "{n} images at {width}x{height} (others dropped: {})",
        sizes.len() - n
    ));
    let long = width.max(height) as f64;
    let mut f35s: Vec<f64> = decoded.iter().filter_map(|d| d.1).collect();
    f35s.sort_by(f64::total_cmp);
    let (focal_prior, focal_source) = match (cfg.focal_px, f35s.get(f35s.len() / 2)) {
        (Some(f), _) => (f, "--focal"),
        (None, Some(f35)) => (f35 / 36.0 * long, "EXIF 35 mm"),
        (None, None) => (1.2 * long, "prior 1.2 x long side"),
    };
    log(&format!("focal {focal_prior:.1} px ({focal_source})"));
    let camera = Camera::pinhole(
        1,
        width,
        height,
        focal_prior,
        focal_prior,
        width as f64 / 2.0,
        height as f64 / 2.0,
    );

    let images_out = out_dir.join("images");
    std::fs::create_dir_all(&images_out).map_err(io(&images_out))?;
    let names: Vec<String> = (0..n).map(|i| format!("frame_{i:05}.png")).collect();
    decoded
        .par_iter()
        .zip(&names)
        .map(|((img, _), name)| {
            let out = images_out.join(name);
            img.save(&out)
                .map_err(|source| EurocError::Image { path: out, source })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let listing: String = names
        .iter()
        .zip(&files)
        .map(|(n, f)| format!("{n} {}\n", f.display()))
        .collect();
    let _ = std::fs::write(out_dir.join("image_names.txt"), listing);

    // SIFT: the EuRoC-vs-COLMAP settings.
    let mut sift_cfg = SiftConfig {
        max_keypoints: cfg.sift_max_keypoints,
        normalization: visloc_vision::features::sift::SiftNormalization::L1Root,
        ..SiftConfig::default()
    };
    for o in [
        "descriptor_magnification=3",
        "max_orientations=2",
        "prefer_larger_scale=1",
    ]
    .into_iter()
    .chain(cfg.sift_overrides.iter().map(String::as_str))
    {
        apply_sift_override(&mut sift_cfg, o).map_err(EurocError::Sift)?;
    }
    let grays: Vec<Vec<f32>> = decoded
        .par_iter()
        .map(|(img, _)| {
            img.pixels()
                .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
                .collect()
        })
        .collect();
    let to_features = |kps: Vec<visloc_vision::features::sift::SiftKeypoint>, desc| FeatureSet {
        keypoints: kps.iter().map(|k| Point2::new(k.x, k.y)).collect(),
        descriptors: desc,
    };
    let t0 = std::time::Instant::now();
    #[cfg(feature = "gpu")]
    let mut gpu_sift = if cfg.gpu {
        let ctx = visloc_sift_gpu::GpuContext::new()
            .map_err(|e| EurocError::Sift(format!("gpu: {e}")))?;
        Some(visloc_sift_gpu::SiftGpu::new(ctx))
    } else {
        None
    };
    // The GPU extractor implements the isotropic DoG path only; affine shape
    // estimation or domain-size pooling (wide-baseline settings) run on the
    // CPU, one image per thread. Matching stays on the GPU either way.
    #[cfg(feature = "gpu")]
    let on_gpu = gpu_sift.is_some() && visloc_sift_gpu::SiftGpu::supports(&sift_cfg);
    #[cfg(not(feature = "gpu"))]
    let on_gpu = false;
    let features: Vec<FeatureSet> = if on_gpu {
        let mut features = Vec::with_capacity(n);
        #[cfg(feature = "gpu")]
        for g in &grays {
            let gray = GrayImage::new(width as usize, height as usize, g)
                .map_err(|e| EurocError::Sift(format!("{e}")))?;
            let (kps, desc) = gpu_sift
                .as_mut()
                .expect("on_gpu")
                .extract(&gray, &sift_cfg)
                .map_err(|e| EurocError::Sift(format!("{e}")))?;
            features.push(to_features(kps, desc));
        }
        features
    } else {
        grays
            .par_iter()
            .map(|g| {
                let gray = GrayImage::new(width as usize, height as usize, g)
                    .map_err(|e| EurocError::Sift(format!("{e}")))?;
                let (kps, desc) =
                    extract_sift(&gray, &sift_cfg).map_err(|e| EurocError::Sift(format!("{e}")))?;
                Ok(to_features(kps, desc))
            })
            .collect::<Result<_, EurocError>>()?
    };
    log(&format!(
        "sift: mean {} keypoints ({:.1} s)",
        features.iter().map(|f| f.keypoints.len()).sum::<usize>() / n.max(1),
        t0.elapsed().as_secs_f64()
    ));

    // Candidate pairs.
    let candidates: Vec<(usize, usize)> = if n <= cfg.exhaustive_max {
        (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect()
    } else {
        let mut pairs: std::collections::BTreeSet<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..(i + 1 + cfg.window).min(n)).map(move |j| (i, j)))
            .collect();
        if cfg.retrieval_k > 0 {
            let t = std::time::Instant::now();
            let extra = retrieval_pairs(&features, cfg.retrieval_k);
            let before = pairs.len();
            pairs.extend(extra);
            log(&format!(
                "retrieval: VLAD top-{} added {} pairs to {} window pairs ({:.1} s)",
                cfg.retrieval_k,
                pairs.len() - before,
                before,
                t.elapsed().as_secs_f64()
            ));
        }
        pairs.into_iter().collect()
    };
    let t0 = std::time::Instant::now();
    // GPU matching binds one image group's descriptors as a single storage
    // buffer. When all of them exceed the device's binding limit (about 2 GiB:
    // ~1,000 images at 4,000 SIFT keypoints), split the images into blocks of
    // at most half the limit and match each block pair from a bank holding
    // just those two blocks.
    #[cfg(feature = "gpu")]
    let blocks: Vec<Vec<usize>> = {
        let limit = gpu_sift
            .as_ref()
            .map(|g| g.context().limits.max_storage_buffer_binding_size)
            .unwrap_or(u64::MAX);
        let bytes = |i: usize| -> u64 {
            features[i]
                .descriptors
                .iter()
                .map(|d| d.len() as u64 * 4)
                .sum()
        };
        let total: u64 = (0..n).map(bytes).sum();
        if total <= limit {
            vec![(0..n).collect()]
        } else {
            let mut blocks = vec![Vec::new()];
            let mut acc = 0u64;
            for i in 0..n {
                if acc + bytes(i) > limit / 2 && !blocks.last().expect("non-empty").is_empty() {
                    blocks.push(Vec::new());
                    acc = 0;
                }
                acc += bytes(i);
                blocks.last_mut().expect("non-empty").push(i);
            }
            log(&format!(
                "gpu matching: {:.2} GiB of descriptors over the {:.2} GiB binding limit, {} blocks",
                total as f64 / (1u64 << 30) as f64,
                limit as f64 / (1u64 << 30) as f64,
                blocks.len()
            ));
            blocks
        }
    };
    #[cfg(not(feature = "gpu"))]
    let blocks: Vec<Vec<usize>> = vec![(0..n).collect()];
    let mut block_of = vec![0usize; n];
    for (b, imgs) in blocks.iter().enumerate() {
        for &i in imgs {
            block_of[i] = b;
        }
    }
    // Candidate pairs grouped by block pair, in a deterministic order.
    let mut groups: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> =
        std::collections::BTreeMap::new();
    for &(i, j) in &candidates {
        let (bi, bj) = (block_of[i], block_of[j]);
        groups
            .entry((bi.min(bj), bi.max(bj)))
            .or_default()
            .push((i, j));
    }
    let mut pairwise: Vec<PairwiseMatches> = Vec::new();
    let (mut match_s, mut verify_s) = (0.0f64, 0.0f64);
    // Verified matches of one matched chunk, and the CPU time spent.
    let verify = |(chunk, dms): MatchedChunk| {
        let t = std::time::Instant::now();
        let v: Vec<PairwiseMatches> = chunk
            .par_iter()
            .zip(dms.into_par_iter())
            .filter_map(|(&(i, j), dm)| {
                verify_pair(
                    &camera,
                    &features[i],
                    &features[j],
                    &dm,
                    cfg.min_matches,
                    true,
                    false,
                )
                .map(|matches| PairwiseMatches {
                    image_i: i,
                    image_j: j,
                    matches,
                    two_view_config: None,
                    essential_matches: None,
                    essential_matrix: None,
                })
            })
            .collect();
        (v, t.elapsed().as_secs_f64())
    };
    let mut pending: Option<MatchedChunk> = None;
    for ((bi, bj), group) in &groups {
        // Images of this block pair, and their index in the uploaded bank.
        let mut imgs: Vec<usize> = blocks[*bi].clone();
        if bj != bi {
            imgs.extend_from_slice(&blocks[*bj]);
        }
        let mut local = vec![usize::MAX; n];
        for (k, &i) in imgs.iter().enumerate() {
            local[i] = k;
        }
        #[cfg(feature = "gpu")]
        let gpu_match = gpu_sift.as_ref().and_then(|g| {
            let ctx = g.context();
            let sets: Vec<&[Vec<f32>]> = imgs
                .iter()
                .map(|&i| features[i].descriptors.as_slice())
                .collect();
            visloc_sift_gpu::FeatureBank::upload(ctx, &sets)
                .ok()
                .map(|bank| (ctx, bank, visloc_sift_gpu::GpuMatcher::new(ctx)))
        });
        let match_pairs = |pairs: &[(usize, usize)]| -> Vec<Vec<DescriptorMatch>> {
            #[cfg(feature = "gpu")]
            if let Some((ctx, bank, m)) = &gpu_match {
                let lp: Vec<(usize, usize)> =
                    pairs.iter().map(|&(i, j)| (local[i], local[j])).collect();
                return m.match_pairs(ctx, bank, &lp, Some(0.8), true);
            }
            pairs
                .par_iter()
                .map(|&(i, j)| cpu_matches(&features[i], &features[j]))
                .collect()
        };
        // Verify the previous chunk on the CPU while the next one matches.
        for chunk in group.chunks(1024) {
            let (dms, verified) = rayon::join(
                || {
                    let t = std::time::Instant::now();
                    (match_pairs(chunk), t.elapsed().as_secs_f64())
                },
                || pending.take().map(verify),
            );
            match_s += dms.1;
            if let Some((v, secs)) = verified {
                pairwise.extend(v);
                verify_s += secs;
            }
            pending = Some((chunk, dms.0));
        }
    }
    if let Some((v, secs)) = pending.take().map(verify) {
        pairwise.extend(v);
        verify_s += secs;
    }
    log(&format!(
        "{} verified of {} candidate pairs ({:.1} s; overlapped: match {:.1} s, verify {:.1} s)",
        pairwise.len(),
        candidates.len(),
        t0.elapsed().as_secs_f64(),
        match_s,
        verify_s
    ));

    // Mapper, then a global BA that also refines the intrinsics.
    let t0 = std::time::Instant::now();
    let models = run_colmap_port(
        out_dir,
        &camera,
        width,
        height,
        &features,
        &pairwise,
        &[],
        log,
    )?;
    let (mut poses, mut tracks, mut reproj) = models.into_iter().next().expect("non-empty");
    let mut cam = camera.clone();
    let mapped = poses.iter().filter(|p| p.is_some()).count();
    if cfg.refine_intrinsics && mapped > cfg.refine_intrinsics_max_images {
        log(&format!(
            "skipping intrinsics refinement: {mapped} images > --refine-intrinsics-max {}; \
             keeping the prior focal",
            cfg.refine_intrinsics_max_images
        ));
    } else if cfg.refine_intrinsics {
        let mut sfm_cfg = IncrementalSfmConfig {
            min_seed_matches: cfg.min_matches,
            colmap_style_mapper: true,
            refine_intrinsics: true,
            ..IncrementalSfmConfig::default()
        };
        // Block-sparse joint pose + intrinsics solve: scales with the
        // covisibility pattern instead of a dense (6 * images)^2 system.
        sfm_cfg.ba_config.linear_solver = visloc_slam::LinearSolver::Sparse;
        let r = visloc_slam::incremental_sfm_with_initial_poses(
            &camera,
            &features,
            &pairwise,
            &sfm_cfg,
            Some(&poses),
        )
        .map_err(|e| EurocError::Sfm(e.to_string()))?;
        if r.poses.iter().filter(|p| p.is_some()).count()
            >= poses.iter().filter(|p| p.is_some()).count()
        {
            poses = r.poses;
            tracks = r.tracks;
            reproj = r.mean_reprojection_px;
            if let Some(c) = r.refined_camera {
                cam = c;
            }
        } else {
            log("intrinsics refinement lost images; keeping the mapper's model");
        }
    }
    let registered = poses.iter().filter(|p| p.is_some()).count();
    let focal = cam.params[0];
    log(&format!(
        "sfm: {registered}/{n} registered, {} points, {reproj:.3} px, focal {focal:.1} px ({:.1} s)",
        tracks.len(),
        t0.elapsed().as_secs_f64()
    ));

    let mut train = Vec::new();
    let mut eval = Vec::new();
    for (k, (i, pose)) in poses
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|p| (i, p)))
        .enumerate()
    {
        let view = View {
            name: names[i].clone(),
            camera: camera_view_from(&cam, pose).map_err(|e| EurocError::Camera(e.to_string()))?,
            image_path: images_out.join(&names[i]),
        };
        if cfg.eval_every > 0 && k % cfg.eval_every == 0 {
            eval.push(view);
        } else {
            train.push(view);
        }
    }
    let points: Vec<ColoredPoint> = tracks
        .iter()
        .filter_map(|t| {
            let &(img, _, px) = t.observations.first()?;
            let (x, y) = (px.x.round(), px.y.round());
            let rgb = if x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64 {
                decoded[img].0.get_pixel(x as u32, y as u32).0
            } else {
                [128; 3]
            };
            Some(ColoredPoint {
                position: Vector3::new(
                    t.position.x as f32,
                    t.position.y as f32,
                    t.position.z as f32,
                ),
                rgb,
            })
        })
        .collect();
    // COLMAP text model next to the images, so other tools (brush, Inria
    // 3DGS, gsplat_train / gsplat_mesh --data) can use the reconstruction.
    let sparse = out_dir.join("sparse").join("0");
    std::fs::create_dir_all(&sparse).map_err(io(&sparse))?;
    let p = &cam.params;
    let cameras = format!(
        "# Camera list\n1 PINHOLE {width} {height} {} {} {} {}\n",
        p[0], p[1], p[2], p[3]
    );
    let mut images_txt = String::from("# Image list\n");
    for (id, (i, pose)) in poses
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|p| (i, p)))
        .enumerate()
    {
        let q = pose.world_to_camera.rotation.quaternion();
        let t = pose.world_to_camera.translation;
        images_txt.push_str(&format!(
            "{} {} {} {} {} {} {} {} 1 {}\n\n",
            id + 1,
            q.w,
            q.i,
            q.j,
            q.k,
            t.x,
            t.y,
            t.z,
            names[i]
        ));
    }
    let points_txt: String = std::iter::once("# 3D point list\n".to_string())
        .chain(points.iter().enumerate().map(|(k, pt)| {
            format!(
                "{} {} {} {} {} {} {} 0\n",
                k + 1,
                pt.position.x,
                pt.position.y,
                pt.position.z,
                pt.rgb[0],
                pt.rgb[1],
                pt.rgb[2]
            )
        }))
        .collect();
    for (file, text) in [
        ("cameras.txt", cameras),
        ("images.txt", images_txt),
        ("points3D.txt", points_txt),
    ] {
        let path = sparse.join(file);
        std::fs::write(&path, text).map_err(io(&path))?;
    }

    let report = PhotoSfmReport {
        images: n,
        width,
        height,
        focal_prior,
        focal_source,
        focal,
        pairs: pairwise.len(),
        registered,
        points: points.len(),
        mean_reprojection_px: reproj,
    };
    Ok((
        Dataset {
            init: Scene::new(Vec::new(), 0),
            train,
            eval,
        },
        points,
        report,
    ))
}

#[cfg(test)]
mod tests {
    use super::exif_focal_35mm;

    /// A minimal little-endian EXIF APP1: IFD0 -> ExifIFD -> 0xA405 = 26.
    #[test]
    fn reads_35mm_focal_from_exif() {
        let mut tiff: Vec<u8> = b"II*\0".to_vec();
        tiff.extend(8u32.to_le_bytes()); // IFD0 at 8
        tiff.extend(1u16.to_le_bytes()); // one entry
        tiff.extend(0x8769u16.to_le_bytes());
        tiff.extend(4u16.to_le_bytes()); // LONG
        tiff.extend(1u32.to_le_bytes());
        tiff.extend(26u32.to_le_bytes()); // ExifIFD at 26
        tiff.extend(0u32.to_le_bytes()); // next IFD
        tiff.extend(1u16.to_le_bytes());
        tiff.extend(0xA405u16.to_le_bytes());
        tiff.extend(3u16.to_le_bytes()); // SHORT
        tiff.extend(1u32.to_le_bytes());
        tiff.extend(26u16.to_le_bytes());
        tiff.extend(0u16.to_le_bytes());
        tiff.extend(0u32.to_le_bytes());
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
        jpeg.extend(((tiff.len() + 8) as u16).to_be_bytes());
        jpeg.extend(b"Exif\0\0");
        jpeg.extend(&tiff);
        jpeg.extend([0xFF, 0xDA, 0, 2]);
        assert_eq!(exif_focal_35mm(&jpeg), Some(26.0));
        assert_eq!(exif_focal_35mm(&[0xFF, 0xD8, 0xFF, 0xDA, 0, 2]), None);
    }

    /// Unit vectors from a fixed LCG, so the similarities have no ties.
    fn pseudo_random_rows(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                    })
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect()
    }

    /// Row-by-row reference for `top_k_similar_pairs` (the previous
    /// implementation's scan, parallel over rows).
    fn naive_top_k(global: &[Vec<f32>], k: usize) -> Vec<(usize, usize)> {
        use rayon::prelude::*;
        let n = global.len();
        let mut pairs: Vec<(usize, usize)> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let mut sims: Vec<(f32, usize)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let s: f32 = global[i].iter().zip(&global[j]).map(|(a, b)| a * b).sum();
                        (s, j)
                    })
                    .collect();
                sims.sort_by(|a, b| b.0.total_cmp(&a.0));
                sims.truncate(k);
                sims.into_iter().map(move |(_, j)| (i.min(j), i.max(j)))
            })
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    #[test]
    fn blocked_top_k_matches_the_row_by_row_scan() {
        // 150 rows: more than two GEMM row blocks, the last one partial.
        let global = pseudo_random_rows(150, 96);
        for k in [1, 5, 30] {
            assert_eq!(
                super::top_k_similar_pairs(&global, k),
                naive_top_k(&global, k),
                "k = {k}"
            );
        }
        assert!(super::top_k_similar_pairs(&global[..1], 5).is_empty());
    }

    /// SmallCity-sized similarity search (5,822 VLAD vectors of 64 x 128).
    /// `cargo test --release -p visloc-gsplat-train --features euroc --lib
    /// retrieval_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn retrieval_bench() {
        let global = pseudo_random_rows(5822, 64 * 128);
        let t = std::time::Instant::now();
        let fast = super::top_k_similar_pairs(&global, 30);
        let tf = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let slow = naive_top_k(&global, 30);
        let ts = t.elapsed().as_secs_f64();
        println!("blocked GEMM {tf:.2} s, row-by-row {ts:.2} s");
        assert_eq!(fast, slow);
    }

    /// Vocabulary and VLAD aggregation at SmallCity scale (100k vocabulary
    /// samples; 300 of the 5,822 images of 4,000 SIFT descriptors).
    #[test]
    #[ignore]
    fn vlad_bench() {
        use rayon::prelude::*;
        use visloc_vision::place_recognition::{vlad, Vocabulary};
        let sample = pseudo_random_rows(100_000, 128);
        let refs: Vec<&[f32]> = sample.iter().map(Vec::as_slice).collect();
        let t = std::time::Instant::now();
        let vocab = Vocabulary::build(&refs, 64, 10, 7).unwrap();
        let bits = vocab
            .centroids
            .iter()
            .flatten()
            .fold(0u64, |h, x| h.rotate_left(5) ^ u64::from(x.to_bits()));
        println!(
            "vocabulary {:.2} s, centroid bits {bits:016x}",
            t.elapsed().as_secs_f64()
        );
        let images: Vec<Vec<Vec<f32>>> = (0..300)
            .map(|i| sample[i * 300..i * 300 + 4000].to_vec())
            .collect();
        let t = std::time::Instant::now();
        let g: Vec<Vec<f32>> = images.par_iter().map(|d| vlad(d, &vocab)).collect();
        println!(
            "vlad of {} images {:.2} s",
            g.len(),
            t.elapsed().as_secs_f64()
        );
    }
}
