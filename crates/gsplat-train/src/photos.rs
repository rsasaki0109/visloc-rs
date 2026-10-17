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
//! 3. SIFT (COLMAP's VLFeat detector and descriptor, RootSIFT; on the GPU
//!    with `gpu_sift`), GPU matching, two-view verification: exhaustive
//!    pairs for up to `exhaustive_max` images, a sliding window over the
//!    sorted names beyond.
//! 4. The COLMAP incremental-mapper port, then one global bundle adjustment
//!    that also refines the shared intrinsics (the port keeps them fixed).
//!
//! Equirectangular (360°) panoramas break the pinhole model of every stage:
//! they are reported in the log. With [`PhotoSfmConfig::equirect`] each one
//! is cut into its six 90° cube faces ([`crate::equirect`]) first, and the
//! faces go through the stages above as pinhole images of known intrinsics.

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
use crate::equirect;
use crate::euroc::{
    apply_sift_override, cpu_matches, read_port_camera, run_colmap_port, verify_pair, EurocError,
};
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
    /// SIFT detections kept per image, before the up-to-two orientations
    /// each (COLMAP's `max_num_features`, 8192).
    pub sift_max_keypoints: usize,
    /// Every n-th registered image is held out for evaluation (0: none);
    /// with `equirect`, every n-th panorama with a registered face, all its
    /// faces (every n-th face would hold out the same face of each).
    pub eval_every: usize,
    pub gpu: bool,
    /// Refine the shared intrinsics in a final global bundle adjustment.
    pub refine_intrinsics: bool,
    /// Skip that refinement above this many registered images (its global
    /// bundle adjustments dominate the SfM time on large models).
    pub refine_intrinsics_max_images: usize,
    /// Past `exhaustive_max`, also match every image with its `retrieval_k`
    /// most similar images by VLAD appearance (0: window only), among the
    /// images more than `window` positions away in file order. The window
    /// already pairs the near ones, which would otherwise take most of the
    /// `k` slots. Connects revisits that are far apart in file order, e.g.
    /// several passes through the same streets.
    pub retrieval_k: usize,
    /// Extra SIFT `key=value` settings (see `euroc::apply_sift_override`),
    /// e.g. `affine=1` / `domain_size_pooling=1` for wide-baseline captures.
    pub sift_overrides: Vec<String>,
    /// Match on the GPU with descriptors quantised to u8 like COLMAP (exact
    /// integer dot products): about 1.6x faster than f32 and a quarter of
    /// the descriptor memory. Off: f32 descriptors. The CPU fallback always
    /// matches f32.
    pub match_u8: bool,
    /// Treat near-black regions (every channel at most this value, larger
    /// than a few pixels) as masked out, e.g. the people and cars some
    /// datasets black out: the training images get alpha 0 there (grown by
    /// a few pixels over JPEG ringing), for `TrainConfig::alpha_mask`.
    pub mask_black: Option<u8>,
    /// Solve the large bundle adjustments' reduced camera systems on the
    /// GPU (`visloc_ba_gpu::GpuReducedSolver`; docs/gpu_bundle_adjustment.md):
    /// the result then differs from the CPU solve at the f32 level.
    pub gpu_ba: bool,
    /// Extract the VLFeat-compatible SIFT on the GPU
    /// (`visloc_sift_gpu::VlfeatSiftGpu`), an f32 port of the CPU extractor:
    /// 99.97% of its keypoints, descriptor bytes 0.0007 apart on average.
    /// Applies with `gpu` in a `gpu` build; otherwise SIFT runs on the CPU.
    /// On by default: on ETH3D the mapper scores the same with it, and
    /// extraction is 8 to 9 times faster than on the CPU.
    pub gpu_sift: bool,
    /// Once the mapper has built every model, extend the largest one with
    /// the images it left out, from 2D-2D correspondences where they see
    /// too few 3D points (the mapper's `complete_largest_model`). It can
    /// only add images to that model, then refines it. On by default.
    pub complete_model: bool,
    /// The images are 2:1 equirectangular panoramas: cut each into six 90°
    /// cube faces (at most `max_size` square) and reconstruct those, with
    /// the exact focal length `size / 2` held fixed (no `focal_px`, no
    /// intrinsics refinement). Faces of one panorama are never matched with
    /// each other (they do not overlap), and `window` counts panoramas.
    /// Each face is still its own camera: nothing ties the six to one
    /// centre, and the log reports how far apart they land.
    pub equirect: bool,
}

impl Default for PhotoSfmConfig {
    fn default() -> Self {
        Self {
            max_size: 1600,
            focal_px: None,
            exhaustive_max: 300,
            window: 20,
            min_matches: 15,
            // COLMAP's cap. 4000 left about half of COLMAP's features on
            // textured scenes and dropped photos it registers: ETH3D
            // playground 35/38 -> 38/38 (0.78 -> 0.63 cm centre RMSE),
            // facade 75/76 -> 76/76. CPU matching takes 2-3.5x longer.
            sift_max_keypoints: 8192,
            eval_every: 8,
            gpu: true,
            refine_intrinsics: true,
            refine_intrinsics_max_images: usize::MAX,
            retrieval_k: 0,
            sift_overrides: Vec::new(),
            match_u8: true,
            mask_black: None,
            gpu_ba: false,
            gpu_sift: true,
            complete_model: true,
            equirect: false,
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
/// images more than `skip_within` positions away, by the cosine similarity
/// of VLAD descriptors over a 64-word vocabulary built from a subsample of
/// all descriptors.
fn retrieval_pairs(features: &[FeatureSet], k: usize, skip_within: usize) -> Vec<(usize, usize)> {
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
    top_k_similar_pairs(&global, k, skip_within)
}

/// Pairs `(i, j)`, `i < j`, joining every row of `global` to the `k` rows
/// more than `skip_within` rows away with the largest dot product. The
/// similarities are computed as blocks of
/// `global · globalᵀ` (a GEMM per block of rows), not row by row: a
/// row-by-row scan streams every descriptor once per image, which is
/// memory-bound past a few thousand images.
fn top_k_similar_pairs(global: &[Vec<f32>], k: usize, skip_within: usize) -> Vec<(usize, usize)> {
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
                        .filter(|&j| i.abs_diff(j) > skip_within)
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

/// Pixels inside near-black regions of `img` (every channel at most `thr`):
/// the near-black set opened by a 5x5 box (drops dark specks such as
/// shadows and tyres), then grown by 2 pixels to cover JPEG ringing at the
/// region edges.
fn black_regions(img: &image::RgbImage, thr: u8) -> Vec<bool> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let dark: Vec<bool> = img
        .pixels()
        .map(|p| p.0.iter().all(|&c| c <= thr))
        .collect();
    // Separable box filter: `all` = erosion, otherwise dilation.
    let box2 = |m: &[bool], all: bool| -> Vec<bool> {
        let r = 2isize;
        let pass = |src: &[bool], horizontal: bool| -> Vec<bool> {
            (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as isize, (i / w) as isize);
                    let mut it = (-r..=r).filter_map(|d| {
                        let (xx, yy) = if horizontal { (x + d, y) } else { (x, y + d) };
                        (xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h)
                            .then(|| src[yy as usize * w + xx as usize])
                    });
                    if all {
                        it.all(|v| v)
                    } else {
                        it.any(|v| v)
                    }
                })
                .collect()
        };
        pass(&pass(m, true), false)
    };
    let opened = box2(&box2(&dark, true), false);
    box2(&opened, false)
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

/// Decode `bytes` (the file at `path`) upright, as RGB8.
fn load_oriented(path: &Path, bytes: &[u8]) -> Result<image::RgbImage, EurocError> {
    use image::ImageDecoder;
    let err = |source| EurocError::Image {
        path: path.to_path_buf(),
        source,
    };
    let mut decoder = image::ImageReader::new(std::io::Cursor::new(bytes))
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
    // Moves an RGB8 image out instead of copying it.
    Ok(img.into_rgb8())
}

/// `image::imageops::resize(img, width, height, FilterType::Triangle)` for
/// RGB8: the same weights and the same f32 sums in the same order, so the
/// same bytes. That implementation reads one pixel at a time, its second
/// pass down the columns of an f32 RGBA copy. This one sums blocks of
/// source rows in registers and walks the intermediate row by row: 108 ms
/// instead of 261 ms for a 24-megapixel photo to 1,600 px.
fn resize_triangle(img: &image::RgbImage, width: u32, height: u32) -> image::RgbImage {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return image::RgbImage::new(width, height);
    }
    if (width, height) == (w, h) {
        return img.clone();
    }
    // Per output pixel: the first input pixel and the normalised weights, as
    // `image`'s samplers compute them (the triangle kernel has support 1).
    let taps = |src: u32, dst: u32| -> Vec<(usize, Vec<f32>)> {
        let ratio = src as f32 / dst as f32;
        let sratio = if ratio < 1.0 { 1.0 } else { ratio };
        let support = sratio;
        (0..dst)
            .map(|o| {
                let input = (o as f32 + 0.5) * ratio;
                let left = ((input - support).floor() as i64).clamp(0, src as i64 - 1);
                let right = ((input + support).ceil() as i64).clamp(left + 1, src as i64);
                let input = input - 0.5;
                let mut ws: Vec<f32> = (left..right)
                    .map(|i| {
                        let x = (i as f32 - input) / sratio;
                        if x.abs() < 1.0 {
                            1.0 - x.abs()
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let mut sum = 0.0f32;
                for &wi in &ws {
                    sum += wi;
                }
                ws.iter_mut().for_each(|wi| *wi /= sum);
                (left as usize, ws)
            })
            .collect()
    };
    // Vertical pass into an f32 RGB image of the output height, a block of
    // values at a time.
    const BLOCK: usize = 64;
    let row = 3 * w as usize;
    let src = img.as_raw();
    let mut tmp: Vec<f32> = Vec::with_capacity(height as usize * row);
    for (top, ws) in taps(h, height) {
        for x0 in (0..row).step_by(BLOCK) {
            let n = BLOCK.min(row - x0);
            let mut acc = [0.0f32; BLOCK];
            for (k, &wk) in ws.iter().enumerate() {
                let line = &src[(top + k) * row + x0..(top + k) * row + x0 + n];
                for (a, &v) in acc[..n].iter_mut().zip(line) {
                    *a += v as f32 * wk;
                }
            }
            tmp.extend_from_slice(&acc[..n]);
        }
    }
    // Horizontal pass, rounded to bytes.
    let horizontal = taps(w, width);
    let mut out = image::RgbImage::new(width, height);
    for (dst, line) in out
        .chunks_exact_mut(3 * width as usize)
        .zip(tmp.chunks_exact(row))
    {
        for (px, (left, ws)) in dst.chunks_exact_mut(3).zip(&horizontal) {
            let mut acc = [0.0f32; 3];
            for (k, &wk) in ws.iter().enumerate() {
                let p = &line[3 * (left + k)..3 * (left + k) + 3];
                acc[0] += p[0] * wk;
                acc[1] += p[1] * wk;
                acc[2] += p[2] * wk;
            }
            for (d, a) in px.iter_mut().zip(acc) {
                *d = a.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
    out
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
    // Two panoramas already give faces that overlap.
    let min_images = if cfg.equirect { 2 } else { 3 };
    if files.len() < min_images {
        return Err(EurocError::Sfm(format!(
            "need at least {min_images} images in {}, found {}",
            images_dir.display(),
            files.len()
        )));
    }
    if cfg.equirect && cfg.focal_px.is_some() {
        return Err(EurocError::Sfm(
            "equirect: the cube faces' focal length is fixed; drop focal_px (--focal)".into(),
        ));
    }

    // Read, decode (oriented), scale, and read the EXIF focal and whether
    // the metadata names a 360° panorama, in parallel.
    let t_load = std::time::Instant::now();
    let loaded: Vec<((image::RgbImage, Option<f64>), bool)> = files
        .par_iter()
        .map(|p| -> Result<_, EurocError> {
            let bytes = std::fs::read(p).map_err(io(p))?;
            let img = load_oriented(p, &bytes)?;
            let (w, h) = img.dimensions();
            // A panorama is scaled for its faces (a quarter of its width) to
            // be at most `max_size`.
            let long = if cfg.equirect { w / 4 } else { w.max(h) };
            let s = (cfg.max_size as f64 / long.max(1) as f64).min(1.0);
            let img = if s < 1.0 {
                resize_triangle(
                    &img,
                    ((w as f64 * s).round() as u32).max(1),
                    ((h as f64 * s).round() as u32).max(1),
                )
            } else {
                img
            };
            Ok((
                (img, exif_focal_35mm(&bytes)),
                equirect::metadata_says_equirectangular(&bytes),
            ))
        })
        .collect::<Result<_, _>>()?;
    let (decoded, tagged_360): (Vec<_>, Vec<bool>) = loaded.into_iter().unzip();
    if !cfg.equirect {
        let panoramas = decoded
            .par_iter()
            .zip(&tagged_360)
            .filter(|((img, _), &tagged)| tagged || equirect::looks_equirectangular(img))
            .count();
        if 2 * panoramas >= decoded.len() {
            log(&format!(
                "warning: {panoramas} of {} images look like equirectangular (360°) panoramas; \
                 the pinhole reconstruction of them will not be usable. Cut them into cube \
                 faces with `equirect` (gsplat_photos --equirect)",
                decoded.len()
            ));
        }
    }
    // The most common resolution (one camera for the trainer).
    let mut sizes: Vec<(u32, u32)> = decoded.iter().map(|d| d.0.dimensions()).collect();
    sizes.sort_unstable();
    let (width, height) = sizes
        .chunk_by(|a, b| a == b)
        .max_by_key(|c| c.len())
        .map(|c| c[0])
        .expect("at least 3 images");
    // Sizes within 1% of it (undistorted crops a few pixels apart, which can
    // round to a different size after scaling) are resized to it; the rest
    // are dropped.
    let near = |(w, h): (u32, u32)| {
        w.abs_diff(width) as f64 <= 0.01 * width as f64
            && h.abs_diff(height) as f64 <= 0.01 * height as f64
    };
    let mut resized = 0usize;
    let (files, decoded): (Vec<_>, Vec<_>) = files
        .into_iter()
        .zip(decoded)
        .filter(|(_, d)| near(d.0.dimensions()))
        .map(|(f, (img, f35))| {
            if img.dimensions() == (width, height) {
                return (f, (img, f35));
            }
            resized += 1;
            let img = resize_triangle(&img, width, height);
            (f, (img, f35))
        })
        .unzip();
    log(&format!(
        "{} images at {width}x{height} ({resized} resized from within 1%; others dropped: {}; \
         decoded in {:.1} s)",
        files.len(),
        sizes.len() - files.len(),
        t_load.elapsed().as_secs_f64()
    ));
    // Each panorama -> its six cube faces, consecutive, in panorama order:
    // image `i` comes from source image `i / per_source`.
    let per_source = if cfg.equirect { 6 } else { 1 };
    let (labels, decoded, width, height) = if cfg.equirect {
        if width.abs_diff(2 * height) as f64 > 0.01 * width as f64 {
            return Err(EurocError::Sfm(format!(
                "equirect: the panoramas are {width}x{height}, not 2:1"
            )));
        }
        let panoramas = decoded;
        let size = (width / 4).max(1);
        let faces = equirect::cube_face_rotations();
        let decoded: Vec<(image::RgbImage, Option<f64>)> = panoramas
            .par_iter()
            .map(|(pano, _)| {
                faces
                    .iter()
                    .map(|(_, r)| (equirect::cube_face(pano, r, size), None))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect();
        let labels: Vec<String> = files
            .iter()
            .flat_map(|f| {
                faces
                    .iter()
                    .map(move |(face, _)| format!("{} {face}", f.display()))
            })
            .collect();
        log(&format!(
            "equirect: {} cube faces of {size}x{size} px",
            decoded.len()
        ));
        (labels, decoded, size, size)
    } else {
        let labels = files.iter().map(|f| f.display().to_string()).collect();
        (labels, decoded, width, height)
    };
    let n = decoded.len();
    let source = |i: usize| i / per_source;
    let long = width.max(height) as f64;
    let mut f35s: Vec<f64> = decoded.iter().filter_map(|d| d.1).collect();
    f35s.sort_by(f64::total_cmp);
    let (focal_prior, focal_source) = match (cfg.focal_px, f35s.get(f35s.len() / 2)) {
        _ if cfg.equirect => (width as f64 / 2.0, "cube face, 90° field of view"),
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
            match cfg.mask_black {
                None => img.save(&out),
                Some(thr) => {
                    let masked = black_regions(img, thr);
                    let mut rgba = image::RgbaImage::new(img.width(), img.height());
                    for ((o, p), &m) in rgba.pixels_mut().zip(img.pixels()).zip(&masked) {
                        *o = image::Rgba([p[0], p[1], p[2], if m { 0 } else { 255 }]);
                    }
                    rgba.save(&out)
                }
            }
            .map_err(|source| EurocError::Image { path: out, source })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let listing: String = names
        .iter()
        .zip(&labels)
        .map(|(n, l)| format!("{n} {l}\n"))
        .collect();
    let _ = std::fs::write(out_dir.join("image_names.txt"), listing);

    // SIFT: COLMAP's VLFeat detector (subpixel localization) and descriptor.
    // The default detector reports integer pixel positions: on ETH3D
    // courtyard it found 37 matches where this finds 721 for one pair, and
    // registered 15 of 38 photos instead of 38. It runs on the CPU (the GPU
    // extractor implements the default detector only); `--sift-opt
    // vlfeat_compatible_detector=0 --sift-opt vlfeat_compatible_descriptor=0`
    // restores GPU extraction.
    let mut sift_cfg = SiftConfig {
        max_keypoints: cfg.sift_max_keypoints,
        normalization: visloc_vision::features::sift::SiftNormalization::L1Root,
        ..SiftConfig::default()
    };
    for o in [
        "descriptor_magnification=3",
        "max_orientations=2",
        "prefer_larger_scale=1",
        "vlfeat_compatible_detector=1",
        "vlfeat_compatible_descriptor=1",
        "vlfeat_bilinear_orientations=1",
    ]
    .into_iter()
    .chain(cfg.sift_overrides.iter().map(String::as_str))
    {
        apply_sift_override(&mut sift_cfg, o).map_err(EurocError::Sift)?;
    }
    // f32 grey images in [0, 1] are SIFT's input (its contrast threshold
    // assumes that range), made a chunk at a time: all of them at once
    // (4 bytes a pixel) are 16.5 GB for SmallCity's 5,822 photos.
    let gray_of = |img: &image::RgbImage| -> Vec<f32> {
        img.pixels()
            .map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0)
            .collect()
    };
    let byte_scale = sift_cfg.vlfeat_compatible_descriptor;
    let to_features = |kps: Vec<visloc_vision::features::sift::SiftKeypoint>, desc| FeatureSet {
        keypoints: kps.iter().map(|k| Point2::new(k.x, k.y)).collect(),
        descriptors: if byte_scale {
            unit_scale_descriptors(desc)
        } else {
            desc
        },
    };
    #[cfg(not(feature = "gpu"))]
    if cfg.gpu_ba {
        return Err(EurocError::Sfm("gpu_ba needs the `gpu` feature".into()));
    }
    #[cfg(feature = "gpu")]
    if cfg.gpu_ba {
        let adapter = visloc_ba_gpu::register_reduced_pcg()
            .map_err(|e| EurocError::Sfm(format!("gpu: {e}")))?;
        log(&format!("gpu ba: reduced camera systems on {adapter}"));
    }
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
    // The VLFeat-compatible extractor on the GPU (with `gpu`): it borrows the
    // matcher's context and hands it back.
    #[cfg(feature = "gpu")]
    let vlfeat_gpu = cfg.gpu_sift && cfg.gpu && visloc_sift_gpu::VlfeatSiftGpu::supports(&sift_cfg);
    #[cfg(feature = "gpu")]
    if cfg.gpu_sift && cfg.gpu && !vlfeat_gpu {
        log("gpu sift: these SIFT settings are not VLFeat-compatible; extracting on the CPU");
    }
    #[cfg(not(feature = "gpu"))]
    let vlfeat_gpu = false;
    let features: Vec<FeatureSet> = if vlfeat_gpu {
        #[allow(unused_mut)]
        let mut features = Vec::with_capacity(n);
        #[cfg(feature = "gpu")]
        {
            let ctx = match gpu_sift.take() {
                Some(g) => g.into_context(),
                None => visloc_sift_gpu::GpuContext::new()
                    .map_err(|e| EurocError::Sift(format!("gpu: {e}")))?,
            };
            let mut vl = visloc_sift_gpu::VlfeatSiftGpu::new(ctx);
            log(&format!(
                "gpu sift: VLFeat-compatible extractor on {}",
                vl.context().adapter_info.name
            ));
            for chunk in decoded.chunks(256) {
                let grays: Vec<Vec<f32>> = chunk.par_iter().map(|(img, _)| gray_of(img)).collect();
                let images = grays
                    .iter()
                    .map(|g| GrayImage::new(width as usize, height as usize, g))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| EurocError::Sift(format!("{e}")))?;
                // Pipelined: the GPU detects the next photo while the host
                // handles the current one. Photos past the device's binding
                // limit go to the CPU.
                match vl.extract_batch(&images, &sift_cfg) {
                    Err(visloc_sift_gpu::SiftGpuError::Unsupported) => {
                        for gray in &images {
                            let (kps, desc) = extract_sift(gray, &sift_cfg)
                                .map_err(|e| EurocError::Sift(format!("{e}")))?;
                            features.push(to_features(kps, desc));
                        }
                    }
                    other => {
                        let results = other.map_err(|e| EurocError::Sift(format!("{e}")))?;
                        for (kps, desc) in results {
                            features.push(to_features(kps, desc));
                        }
                    }
                }
            }
            if cfg.gpu {
                gpu_sift = Some(visloc_sift_gpu::SiftGpu::new(vl.into_context()));
            }
        }
        features
    } else if on_gpu {
        #[allow(unused_mut)]
        let mut features = Vec::with_capacity(n);
        #[cfg(feature = "gpu")]
        for chunk in decoded.chunks(256) {
            let grays: Vec<Vec<f32>> = chunk.par_iter().map(|(img, _)| gray_of(img)).collect();
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
        }
        features
    } else {
        decoded
            .par_iter()
            .map(|(img, _)| {
                let g = gray_of(img);
                let gray = GrayImage::new(width as usize, height as usize, &g)
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

    // Candidate pairs, never two faces of one panorama (they do not
    // overlap). The first image of the source after image `i`'s:
    let next_source = |i: usize| (source(i) + 1) * per_source;
    let candidates: Vec<(usize, usize)> = if n <= cfg.exhaustive_max {
        (0..n)
            .flat_map(|i| (next_source(i)..n).map(move |j| (i, j)))
            .collect()
    } else {
        // The window counts source images (panoramas).
        let mut pairs: std::collections::BTreeSet<(usize, usize)> = (0..n)
            .flat_map(|i| {
                let end = (next_source(i) + cfg.window * per_source).min(n);
                (next_source(i)..end).map(move |j| (i, j))
            })
            .collect();
        if cfg.retrieval_k > 0 {
            let t = std::time::Instant::now();
            let extra = retrieval_pairs(&features, cfg.retrieval_k, cfg.window * per_source)
                .into_iter()
                .filter(|&(i, j)| source(i) != source(j));
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
        let per_value = if cfg.match_u8 { 1 } else { 4 };
        let bytes = |i: usize| -> u64 {
            features[i]
                .descriptors
                .iter()
                .map(|d| d.len() as u64 * per_value)
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
            let bank = if cfg.match_u8 {
                visloc_sift_gpu::FeatureBank::upload_u8(ctx, &sets)
            } else {
                visloc_sift_gpu::FeatureBank::upload(ctx, &sets)
            };
            bank.ok()
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
        "verified pairs beyond the {}-image window: {}",
        cfg.window,
        pairwise
            .iter()
            .filter(|p| source(p.image_j) - source(p.image_i) > cfg.window)
            .count()
    ));
    log(&format!(
        "{} verified of {} candidate pairs ({:.1} s; overlapped: match {:.1} s, verify {:.1} s)",
        pairwise.len(),
        candidates.len(),
        t0.elapsed().as_secs_f64(),
        match_s,
        verify_s
    ));

    // A cube face's intrinsics are exact.
    let refine_intrinsics = cfg.refine_intrinsics && !cfg.equirect;
    // Past matching only the intrinsics refinement reads the descriptors
    // (13 GB for SmallCity at 4,000 SIFT); without it, free them before the
    // mapper.
    let mut features = features;
    if !refine_intrinsics {
        for f in &mut features {
            f.descriptors = Vec::new();
        }
    }

    // Mapper, then a global BA that also refines the intrinsics. The mapper
    // self-calibrates the focal length and principal point in its own
    // bundle adjustments (COLMAP's `ba_refine_focal_length` /
    // `ba_refine_principal_point`): on the 13 ETH3D DSLR scenes the
    // principal point of the centred prior is ~4 px off, which fixed
    // intrinsics turned into 2-5x larger camera-centre errors.
    let t0 = std::time::Instant::now();
    let mut port_overrides: Vec<String> = if refine_intrinsics {
        vec![
            "ba_refine_focal_length=1".into(),
            "ba_refine_principal_point=1".into(),
        ]
    } else {
        Vec::new()
    };
    if cfg.complete_model {
        port_overrides.push("complete_largest_model=1".into());
    }
    let models = run_colmap_port(
        out_dir,
        &camera,
        width,
        height,
        &features,
        &pairwise,
        &port_overrides,
        log,
    )?;
    let (mut poses, mut tracks, mut reproj) = models.into_iter().next().expect("non-empty");
    let mut cam = camera.clone();
    if refine_intrinsics {
        if let Some(params) = read_port_camera(out_dir) {
            cam.params[..4].copy_from_slice(&params);
            log(&format!(
                "mapper intrinsics: fx {:.1} fy {:.1} cx {:.1} cy {:.1} px",
                params[0], params[1], params[2], params[3]
            ));
        }
    }
    let mapped = poses.iter().filter(|p| p.is_some()).count();
    if refine_intrinsics && mapped > cfg.refine_intrinsics_max_images {
        log(&format!(
            "skipping intrinsics refinement: {mapped} images > --refine-intrinsics-max {}; \
             keeping the mapper's intrinsics",
            cfg.refine_intrinsics_max_images
        ));
    } else if refine_intrinsics {
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
            &cam,
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
    if cfg.equirect {
        log(&face_centre_spread(&poses, per_source));
    }

    let mut train = Vec::new();
    let mut eval = Vec::new();
    // `k`: the rank of the image's source among the sources with a
    // registered image (one image per source without `equirect`).
    let mut k = 0usize;
    let mut last_source = None;
    for (i, pose) in poses
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|p| (i, p)))
    {
        if last_source.is_some_and(|s| s != source(i)) {
            k += 1;
        }
        last_source = Some(source(i));
        let view = View {
            name: names[i].clone(),
            camera: camera_view_from(&cam, pose).map_err(|e| EurocError::Camera(e.to_string()))?,
            image_path: images_out.join(&names[i]),
        };
        if cfg.eval_every > 0 && k.is_multiple_of(cfg.eval_every) {
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

/// How well the independently solved cube faces of each panorama agree on
/// its one optical centre: per panorama with two or more registered faces,
/// the largest distance of a face's centre from their mean, against the
/// median distance between consecutive such panoramas.
fn face_centre_spread(
    poses: &[Option<visloc_core::geometry::Pose>],
    per_panorama: usize,
) -> String {
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied()
    };
    // (mean centre, spread) of each panorama with 2+ registered faces.
    let panoramas: Vec<(Vector3<f64>, f64)> = poses
        .chunks(per_panorama)
        .filter_map(|faces| {
            let centres: Vec<Vector3<f64>> = faces
                .iter()
                .flatten()
                .map(|p| p.camera_center_world().coords)
                .collect();
            if centres.len() < 2 {
                return None;
            }
            let mean = centres.iter().sum::<Vector3<f64>>() / centres.len() as f64;
            let spread = centres
                .iter()
                .map(|c| (c - mean).norm())
                .fold(0.0, f64::max);
            Some((mean, spread))
        })
        .collect();
    let total = poses.len() / per_panorama.max(1);
    let Some(spread) = median(panoramas.iter().map(|p| p.1).collect()) else {
        return format!("equirect: no panorama of {total} has two registered faces");
    };
    let step = median(
        panoramas
            .windows(2)
            .map(|w| (w[1].0 - w[0].0).norm())
            .collect(),
    );
    format!(
        "equirect: {} of {total} panoramas have 2+ registered faces; their face centres lie a \
         median {spread:.4} from the panorama's mean centre (0 for one shared centre){}",
        panoramas.len(),
        match step {
            Some(step) if step > 0.0 => format!(
                ", {:.1}% of the median {step:.4} between panoramas",
                100.0 * spread / step
            ),
            _ => String::new(),
        }
    )
}

/// VLFeat-compatible SIFT descriptors come back already quantised like
/// COLMAP's bytes (`round(512 x)`, norm about 512). Matching expects the
/// unit-norm scale: the GPU u8 bank quantises with `round(512 x)` again,
/// which saturated every non-zero value at 255 and left near-binary
/// descriptors (SmallCity lost 70% of its 3D points). Dividing by 512 is
/// exact in f32, so the bank recovers the same bytes, and the CPU matcher's
/// comparisons are unchanged.
fn unit_scale_descriptors(mut desc: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    for d in &mut desc {
        for v in d.iter_mut() {
            *v /= 512.0;
        }
    }
    desc
}

#[cfg(test)]
mod tests {
    use super::exif_focal_35mm;
    use super::resize_triangle;
    use super::unit_scale_descriptors;

    #[test]
    fn resize_triangle_matches_the_image_crate_byte_for_byte() {
        // A noisy gradient, so every weight matters.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut img = image::RgbImage::new(1037, 691);
        for (x, y, p) in img.enumerate_pixels_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let n = (seed >> 56) as u32;
            *p = image::Rgb([
                ((x * 255 / 1036 + n) / 2) as u8,
                ((y * 255 / 690 + n) / 2) as u8,
                (n ^ (x + y)) as u8,
            ]);
        }
        // Downscales by odd ratios, a near-size resize up and down by a
        // pixel (the photo loader's case), one axis only, and a copy.
        for (w, h) in [
            (400, 267),
            (311, 207),
            (1024, 682),
            (1038, 691),
            (1036, 692),
            (1037, 100),
            (3, 2),
            (1037, 691),
        ] {
            let want = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
            let got = resize_triangle(&img, w, h);
            assert_eq!(got.dimensions(), (w, h));
            assert!(got.as_raw() == want.as_raw(), "{w}x{h} differs");
        }
    }

    #[test]
    fn vlfeat_bytes_round_trip_through_the_u8_scale() {
        let bytes: Vec<f32> = (0..128).map(|i| ((i * 37) % 256) as f32).collect();
        let unit = unit_scale_descriptors(vec![bytes.clone()]);
        for (b, u) in bytes.iter().zip(&unit[0]) {
            assert_eq!((u * 512.0).round().clamp(0.0, 255.0), *b);
        }
    }

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
                super::top_k_similar_pairs(&global, k, 0),
                naive_top_k(&global, k),
                "k = {k}"
            );
        }
        assert!(super::top_k_similar_pairs(&global[..1], 5, 0).is_empty());
    }

    #[test]
    fn top_k_skips_rows_within_the_window() {
        let global = pseudo_random_rows(150, 96);
        let pairs = super::top_k_similar_pairs(&global, 5, 20);
        assert!(pairs.iter().all(|&(i, j)| j - i > 20));
        // Every row still gets its 5 picks (each pair may serve both rows).
        for i in 0..150 {
            let own = pairs.iter().filter(|&&(a, b)| a == i || b == i).count();
            assert!(own >= 5, "row {i}: {own} pairs");
        }
    }

    /// SmallCity-sized similarity search (5,822 VLAD vectors of 64 x 128).
    /// `cargo test --release -p visloc-gsplat-train --features euroc --lib
    /// retrieval_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn retrieval_bench() {
        let global = pseudo_random_rows(5822, 64 * 128);
        let t = std::time::Instant::now();
        let fast = super::top_k_similar_pairs(&global, 30, 0);
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

    #[test]
    fn black_regions_mask_blocks_not_specks() {
        let mut img = image::RgbImage::from_pixel(40, 30, image::Rgb([120, 110, 100]));
        // A blacked-out block with JPEG-like noise, and a 2x2 dark speck.
        for y in 10..20 {
            for x in 5..17 {
                img.put_pixel(x, y, image::Rgb([(x % 3) as u8, 2, (y % 4) as u8]));
            }
        }
        for (x, y) in [(30, 5), (31, 5), (30, 6), (31, 6)] {
            img.put_pixel(x, y, image::Rgb([0, 0, 0]));
        }
        let m = super::black_regions(&img, 12);
        let at = |x: usize, y: usize| m[y * 40 + x];
        assert!(at(10, 15) && at(5, 10) && at(16, 19));
        // Grown by 2 pixels, no further.
        assert!(at(3, 15) && at(18, 15) && at(10, 8) && at(10, 21));
        assert!(!at(2, 15) && !at(19, 15) && !at(10, 7) && !at(10, 22));
        assert!(!at(30, 5) && !at(31, 6));
        assert_eq!(m.iter().filter(|&&v| v).count(), 16 * 14);
    }

    /// Smooth value noise in [0, 1]: four octaves over an integer lattice.
    fn value_noise(x: f64, y: f64, seed: u64) -> f64 {
        let hash = |i: i64, j: i64, o: u64| {
            let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                ^ (seed * 31 + o).wrapping_mul(0x1656_67B1_9E37_79F9);
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (h >> 40) as f64 / (1u64 << 24) as f64
        };
        let mut v = 0.0;
        for (o, freq) in [3.0, 7.0, 16.0, 37.0].into_iter().enumerate() {
            let (fx, fy) = (x * freq, y * freq);
            let (i, j) = (fx.floor() as i64, fy.floor() as i64);
            let (tx, ty) = (fx - fx.floor(), fy - fy.floor());
            let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
            let o = o as u64;
            let top = hash(i, j, o) * (1.0 - sx) + hash(i + 1, j, o) * sx;
            let bottom = hash(i, j + 1, o) * (1.0 - sx) + hash(i + 1, j + 1, o) * sx;
            v += (top * (1.0 - sy) + bottom * sy) / (o + 1) as f64;
        }
        v / (1.0 + 0.5 + 1.0 / 3.0 + 0.25)
    }

    /// A `w x w/2` panorama taken at `centre` in a noise-textured box room
    /// (x, z in [-4, 4] m, y in [-2.5, 2.5] m), turned by `yaw` about the
    /// vertical.
    fn room_panorama(w: u32, centre: nalgebra::Vector3<f64>, yaw: f64) -> image::RgbImage {
        use std::f64::consts::PI;
        let rot = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw);
        let half = nalgebra::Vector3::new(4.0f64, 2.5, 4.0);
        image::RgbImage::from_fn(w, w / 2, |x, y| {
            let lon = (x as f64 + 0.5) / w as f64 * 2.0 * PI - PI;
            let lat = PI / 2.0 - (y as f64 + 0.5) / (w / 2) as f64 * PI;
            let d = rot
                * nalgebra::Vector3::new(lat.cos() * lon.sin(), -lat.sin(), lat.cos() * lon.cos());
            // The nearest wall along d.
            let (t, axis) = (0..3)
                .filter(|&a| d[a].abs() > 1e-12)
                .map(|a| ((half[a].copysign(d[a]) - centre[a]) / d[a], a))
                .fold((f64::INFINITY, 0), |b, c| if c.0 < b.0 { c } else { b });
            let p = centre + t * d;
            let (u, v) = match axis {
                0 => (p.z, p.y),
                1 => (p.x, p.z),
                _ => (p.x, p.y),
            };
            let wall = (2 * axis + usize::from(d[axis] > 0.0)) as u64;
            let c = |k: u64| (255.0 * value_noise(u + 10.0, v + 10.0, 3 * wall + k)) as u8;
            image::Rgb([c(0), c(1), c(2)])
        })
    }

    /// Panoramas of a synthetic room through the whole CPU pipeline: the
    /// pinhole path warns, the cube-face path registers the faces and puts
    /// each panorama's faces at one centre.
    /// `cargo test --release -p visloc-gsplat-train --features euroc --lib
    /// equirect_end_to_end -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn equirect_end_to_end() {
        let root = std::env::temp_dir().join(format!("visloc_equirect_{}", std::process::id()));
        let images = root.join("in");
        std::fs::create_dir_all(&images).unwrap();
        // (centre, yaw) of each panorama.
        let truth: Vec<(nalgebra::Vector3<f64>, f64)> = (0..5)
            .map(|k| {
                let c = nalgebra::Vector3::new(
                    -1.0 + 0.5 * k as f64,
                    0.1 * (k % 2) as f64,
                    0.3 * k as f64,
                );
                (c, 0.4 * k as f64)
            })
            .collect();
        for (k, &(c, yaw)) in truth.iter().enumerate() {
            room_panorama(1024, c, yaw)
                .save(images.join(format!("pano_{k}.png")))
                .unwrap();
        }
        let mut cfg = super::PhotoSfmConfig {
            gpu: false,
            gpu_sift: false,
            eval_every: 0,
            ..Default::default()
        };
        let mut logs = Vec::new();
        let _ = super::build_photo_dataset(&images, &root.join("pinhole"), &cfg, &mut |m: &str| {
            logs.push(m.to_string())
        });
        assert!(logs
            .iter()
            .any(|m| m.starts_with("warning: 5 of 5 images look like")));

        cfg.equirect = true;
        let mut logs = Vec::new();
        let (dataset, _, report) =
            super::build_photo_dataset(&images, &root.join("faces"), &cfg, &mut |m: &str| {
                println!("{m}");
                logs.push(m.to_string())
            })
            .unwrap();
        assert!(!logs.iter().any(|m| m.starts_with("warning")));
        assert_eq!((report.images, report.width, report.height), (30, 256, 256));
        assert_eq!(report.focal, 128.0);
        assert!(report.registered >= 24, "{} registered", report.registered);
        assert_eq!(dataset.train.len(), report.registered);
        // Held out by panorama: every 2nd one's faces.
        cfg.eval_every = 2;
        let (dataset, _, _) =
            super::build_photo_dataset(&images, &root.join("split"), &cfg, &mut |_: &str| {})
                .unwrap();
        let panorama = |v: &crate::dataset::View| v.name[6..11].parse::<usize>().unwrap() / 6;
        assert!(dataset.eval.iter().all(|v| panorama(v) % 2 == 0));
        assert!(dataset.train.iter().all(|v| panorama(v) % 2 == 1));
        assert_eq!(dataset.eval.len() + dataset.train.len(), report.registered);

        // Against the truth, free of the gauge: face k of panorama i looks
        // along yaw_i * R_k (camera to world) from centre_i.
        let faces = crate::equirect::cube_face_rotations();
        let true_rotation = |i: usize| {
            let (_, yaw) = truth[i / 6];
            nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw).matrix()
                * faces[i % 6].1
        };
        // Recovered world-to-camera rotations and centres, by image index.
        let text = std::fs::read_to_string(root.join("faces/sparse/0/images.txt")).unwrap();
        let mut est: Vec<(usize, nalgebra::Matrix3<f64>, nalgebra::Vector3<f64>)> = Vec::new();
        for line in text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let t: Vec<&str> = line.split_whitespace().collect();
            let v: Vec<f64> = t[1..8].iter().map(|x| x.parse().unwrap()).collect();
            let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                v[0], v[1], v[2], v[3],
            ));
            let r = *q.to_rotation_matrix().matrix();
            let centre = -r.transpose() * nalgebra::Vector3::new(v[4], v[5], v[6]);
            let i: usize = t[9][6..11].parse().unwrap();
            est.push((i, r, centre));
        }
        assert_eq!(est.len(), report.registered);
        // Relative rotation of every pair of faces, recovered vs true.
        let mut worst_deg = 0.0f64;
        for (a, ra, _) in &est {
            for (b, rb, _) in &est {
                let recovered = ra * rb.transpose();
                let truth = true_rotation(*a).transpose() * true_rotation(*b);
                let angle =
                    nalgebra::Rotation3::from_matrix_unchecked(recovered.transpose() * truth)
                        .angle();
                worst_deg = worst_deg.max(angle.to_degrees());
            }
        }
        // Distances between the panoramas' mean face centres, recovered vs
        // true, after the least-squares scale.
        let mean_centre = |p: usize| {
            let c: Vec<_> = est.iter().filter(|e| e.0 / 6 == p).map(|e| e.2).collect();
            c.iter().sum::<nalgebra::Vector3<f64>>() / c.len() as f64
        };
        let (mut dt, mut de) = (Vec::new(), Vec::new());
        for p in 0..5 {
            for q in p + 1..5 {
                dt.push((truth[p].0 - truth[q].0).norm());
                de.push((mean_centre(p) - mean_centre(q)).norm());
            }
        }
        let scale = dt.iter().zip(&de).map(|(t, e)| t * e).sum::<f64>()
            / de.iter().map(|e| e * e).sum::<f64>();
        let worst_m = dt
            .iter()
            .zip(&de)
            .map(|(t, e)| (scale * e - t).abs())
            .fold(0.0, f64::max);
        println!(
            "vs truth: worst relative rotation {worst_deg:.3} deg, worst panorama distance \
             error {:.1} mm (spacing 0.58 m)",
            1000.0 * worst_m
        );
        assert!(worst_deg < 0.5, "{worst_deg} deg");
        assert!(worst_m < 0.02, "{worst_m} m");
        let _ = std::fs::remove_dir_all(&root);
    }
}
