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
//! faces go through the stages above as pinhole images of known intrinsics,
//! the six of a panorama as one rig ([`PhotoSfmConfig::equirect_rig`]).
//! [`PhotoSfmConfig::equirect_native`] instead runs the stages on the
//! panoramas themselves as equirectangular cameras and poses the faces
//! from them. Either way training sees the faces that see SfM points.
//! [`PhotoSfmConfig::insv`] takes an Insta360's dual-fisheye frames instead:
//! each lens is cut into five 90° faces with the calibration in the video's
//! metadata ([`crate::dual_fisheye`]), a frame's ten faces solved as one rig.

use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, Point2, Point3, Vector3};
use rayon::prelude::*;
use visloc_core::types::Camera;
use visloc_gsplat_core::colmap_scene::camera_view_from;
use visloc_gsplat_core::gaussian::Scene;
use visloc_slam::{
    estimate_imu_scale, FrameTiming, ImuSamples, ImuScaleConfig, IncrementalSfmConfig,
    PairwiseMatches, ScaleFrame,
};
use visloc_vision::features::sift::{extract_sift, GrayImage, SiftConfig};
use visloc_vision::features::FeatureSet;
use visloc_vision::matching::DescriptorMatch;

use crate::dataset::{Dataset, View};
use crate::dual_fisheye::{self, FisheyeLens};
use crate::equirect;
use crate::euroc::{
    apply_sift_override, cpu_matches, read_port_camera, run_colmap_port, verify_pair, EurocError,
};
use crate::init::ColoredPoint;
use crate::masks::{drop_masked_keypoints, ImageMasks};

/// Settings for [`build_photo_dataset`].
#[derive(Debug, Clone)]
pub struct PhotoSfmConfig {
    /// Long side of the working images, in pixels.
    pub max_size: u32,
    /// Focal length in working-image pixels (overrides EXIF).
    pub focal_px: Option<f64>,
    /// Principal point in working-image pixels (the image centre if unset).
    pub principal_point: Option<(f64, f64)>,
    /// The photos' calibrated camera at its own image size, any COLMAP model
    /// (a fisheye or a distorted lens included): scaled to the working
    /// images and held fixed, in place of `focal_px`, `principal_point` and
    /// the intrinsics refinement. SfM and training then follow its lens on
    /// the images as they are, without undistorting them. Not with
    /// `equirect` or `insv`.
    pub camera: Option<Camera>,
    /// Refine `camera` in the mapper's bundle adjustments instead of
    /// holding it fixed: its focal length, principal point and distortion
    /// parameters (COLMAP's `ba_refine_focal_length`,
    /// `ba_refine_principal_point`, `ba_refine_extra_params`). For a camera
    /// known only roughly, such as a fisheye model with the focal length of
    /// its field of view and no distortion.
    pub refine_camera: bool,
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
    /// Leave out the pixels whose ray is farther than this many degrees
    /// off the axis of `camera` (e.g. 100 for a 200° fisheye's image
    /// circle, whose dark rim the calibration does not cover): the training
    /// images get alpha 0 there, for `TrainConfig::alpha_mask`. Needs
    /// `camera`.
    pub max_angle: Option<f64>,
    /// A directory of per-image masks, COLMAP's `ImageReader.mask_path`: no
    /// SIFT keypoint is kept on a masked-out pixel, so matching and the
    /// mapper never see one, and the training images get alpha 0 there (for
    /// `TrainConfig::alpha_mask`; the pixels themselves are not touched, so
    /// SIFT still sees the whole image). Image `a.jpg`'s mask is
    /// `a.jpg.png` or `a.png` ([`crate::masks::mask_path`]), zero meaning
    /// masked out. Every image needs one. A mask is resized to the working
    /// images (nearest neighbour) but must have their aspect. Not with
    /// `equirect` or `insv`. Keypoints are dropped after the
    /// `sift_max_keypoints` strongest are chosen, as COLMAP does, so a large
    /// mask leaves fewer than that.
    pub mask_dir: Option<PathBuf>,
    /// With `mask_dir`: swap what the masks keep and what they hide, for
    /// masks that paint the region to remove (`255` = masked out).
    pub mask_flip: bool,
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
    /// The log reports how far apart each panorama's face centres land.
    pub equirect: bool,
    /// With `equirect`, solve each panorama's six faces as one rig: one
    /// pose per panorama, the faces at its centre with their fixed
    /// rotations. Off: every face is its own camera.
    pub equirect_rig: bool,
    /// With `equirect`, run SfM on the panoramas themselves, as
    /// equirectangular cameras (COLMAP's `EQUIRECTANGULAR`): SIFT on the
    /// whole panorama, so features across the face seams count, and one
    /// pose per panorama. The cube faces, posed from it, are still what
    /// training sees. The panoramas SfM sees are at most `2 max_size` wide.
    /// Overrides `equirect_rig`.
    pub equirect_native: bool,
    /// An IMU log of the camera the photos were taken with, on the video's
    /// clock: an Insta360 `.insv` (its trailer) or a CSV file (SI units,
    /// EuRoC, or OverMaps360's raw counts; `visloc_io::imu_csv::read_imu_file`):
    /// scale the model to metres with
    /// `visloc_slam::estimate_imu_scale` (docs/insta360_imu_scale.md). The
    /// photos must be frames of one video whose file names end in their frame
    /// number (the last run of digits, e.g. ffmpeg's `%06d.jpg`); the frame
    /// period and the offset to the IMU clock are searched from the gyro.
    /// The scale applies to everything after SfM (sparse/0, splats, mesh);
    /// `<out>/port` keeps the mapper's units. Details go to
    /// `<out>/imu_scale.txt`.
    pub imu: Option<PathBuf>,
    /// With `imu`: the frame period (seconds per frame number) the search
    /// starts from (default: an `.insv`'s video frame period, else the IMU
    /// span over the largest frame number).
    pub imu_frame_period: Option<f64>,
    /// The images are dual-fisheye frames of this Insta360 video: its two
    /// lenses side by side, front lens (first video stream) on the left
    /// (ffmpeg's `hstack` of the streams; docs/photos_to_splat.md). Each
    /// lens is cut into five 90° faces (at most `max_size` square) with the
    /// lens calibration in the video's metadata
    /// ([`crate::dual_fisheye`]). A frame's ten faces are solved as one rig
    /// in the front lens's frame, with the exact focal length held fixed (no
    /// `focal_px`, no intrinsics refinement); the back lens's pose on the
    /// rig starts from a half turn and is refined. `window` counts frames.
    /// The video's IMU gives the model metric scale unless `imu` names
    /// another log. Not with `equirect`.
    pub insv: Option<PathBuf>,
    /// With `insv`: the lenses as [`calibrate_insv_lenses`] found them (each
    /// one's optical centre and focal lengths on the factory calibration),
    /// applied when the faces are cut.
    pub insv_lenses: Option<dual_fisheye::LensCalibration>,
    /// Match the candidate pairs with this instead of the CPU (or wgpu)
    /// matcher, e.g. `visloc-gsplat-cuda`'s CUDA matcher. If it fails, the
    /// CPU matches.
    pub matcher: Option<std::sync::Arc<dyn PairMatcher>>,
}

/// Descriptor matching for [`build_photo_dataset`]'s candidate pairs, in
/// place of the CPU matcher: it must give what the CPU does, the ratio-tested
/// (0.8) mutual nearest neighbours of
/// `BruteForceMatcher::match_descriptors_cross_checked`.
pub trait PairMatcher: Send + Sync + std::fmt::Debug {
    /// Every image's descriptors, before any [`PairMatcher::match_pairs`].
    fn set_features(&self, descriptors: &[&[Vec<f32>]]) -> Result<(), String>;
    /// The matches of each pair `(i, j)` (`query_index` in image i,
    /// `train_index` in image j).
    fn match_pairs(&self, pairs: &[(usize, usize)]) -> Result<Vec<Vec<DescriptorMatch>>, String>;
}

impl Default for PhotoSfmConfig {
    fn default() -> Self {
        Self {
            max_size: 1600,
            focal_px: None,
            principal_point: None,
            camera: None,
            refine_camera: false,
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
            max_angle: None,
            mask_dir: None,
            mask_flip: false,
            gpu_ba: false,
            gpu_sift: true,
            complete_model: true,
            equirect: false,
            equirect_rig: true,
            equirect_native: false,
            imu: None,
            imu_frame_period: None,
            insv: None,
            insv_lenses: None,
            matcher: None,
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
    /// Metres per SfM unit applied to the model, with `PhotoSfmConfig::imu`.
    pub metric_scale: Option<f64>,
}

/// Candidate pairs and their descriptor matches, awaiting verification.
type MatchedChunk<'a> = (&'a [(usize, usize)], Vec<Vec<DescriptorMatch>>);

const IMAGE_EXTENSIONS: [&str; 4] = ["jpg", "jpeg", "png", "JPG"];

/// Image pairs `(i, j)`, `i < j`, joining every image to its `k` most similar
/// images more than `skip_within` positions away, by the cosine similarity
/// of VLAD descriptors over a 64-word vocabulary built from a subsample of
/// all descriptors.
fn retrieval_pairs<'a>(
    features: &[FeatureSet],
    descriptors: &(dyn Fn(usize) -> std::borrow::Cow<'a, [Vec<f32>]> + Sync),
    k: usize,
    skip_within: usize,
) -> Vec<(usize, usize)> {
    use visloc_vision::place_recognition::{vlad, Vocabulary};
    // About 100k descriptors for the vocabulary, evenly from every image
    // (one descriptor per keypoint).
    let total: usize = features.iter().map(|f| f.keypoints.len()).sum();
    let step = (total / 100_000).max(1);
    let sample: Vec<Vec<f32>> = (0..features.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            descriptors(i)
                .iter()
                .step_by(step)
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect();
    let sample: Vec<&[f32]> = sample.iter().map(Vec::as_slice).collect();
    let Some(vocab) = Vocabulary::build(&sample, 64, 10, 7) else {
        return Vec::new();
    };
    let global: Vec<Vec<f32>> = (0..features.len())
        .into_par_iter()
        .map(|i| vlad(&descriptors(i), &vocab))
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

/// The frame number in a file name: its last run of digits.
fn frame_number(path: &Path) -> Option<i64> {
    let stem = path.file_stem()?.to_str()?;
    let digits: String = stem
        .chars()
        .rev()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.chars().rev().collect::<String>().parse().ok()
}

/// The median spacing of an `.insv`'s video frames (its trailer), seconds;
/// `None` for other files.
fn insv_frame_period(path: &Path) -> Option<f64> {
    let insv = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("insv"));
    if !insv {
        return None;
    }
    let times = visloc_io::insv::read_insv_telemetry(path).ok()?.frame_times;
    let mut steps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    steps.sort_by(f64::total_cmp);
    steps.get(steps.len() / 2).copied()
}

/// Metres per SfM unit from `imu` ([`PhotoSfmConfig::imu`]). Image `i` comes
/// from `sources[i / per_source]`, `per_source` being the number of
/// `face_rotations` (face frame to source frame; none: one image per
/// source); each source's first registered face, turned back by its face
/// rotation, is the source's camera.
fn imu_metric_scale(
    imu: &Path,
    sources: &[PathBuf],
    poses: &[Option<visloc_core::geometry::Pose>],
    face_rotations: &[Matrix3<f64>],
    cfg: &PhotoSfmConfig,
    out_dir: &Path,
    log: &mut dyn FnMut(&str),
) -> Result<f64, EurocError> {
    let (csv, source) =
        visloc_io::imu_csv::read_imu_file(imu).map_err(|e| EurocError::Sfm(format!("imu: {e}")))?;
    let per_source = face_rotations.len().max(1);
    let mut frames = Vec::new();
    let mut unnumbered = 0;
    for (s, source) in sources.iter().enumerate() {
        let Some((k, pose)) =
            (0..per_source).find_map(|k| poses.get(s * per_source + k)?.as_ref().map(|p| (k, p)))
        else {
            continue;
        };
        let Some(number) = frame_number(source) else {
            unnumbered += 1;
            continue;
        };
        let face_cw = *pose.world_to_camera.rotation.to_rotation_matrix().matrix();
        // A face's world-to-camera rotation is its rotation's transpose times
        // the source's (`equirect::face_pose`).
        let r_cw = face_rotations.get(k).map_or(face_cw, |r| r * face_cw);
        frames.push(ScaleFrame {
            number,
            rotation: r_cw.transpose(),
            centre: pose.camera_center_world().coords,
        });
    }
    if unnumbered > 0 {
        log(&format!(
            "imu: {unnumbered} registered photos have no frame number in their file name; left out"
        ));
    }
    let samples = ImuSamples {
        time: csv.time,
        gyro: csv.gyro,
        accel: csv.accel,
    };
    let span =
        samples.time.last().copied().unwrap_or(0.0) - samples.time.first().copied().unwrap_or(0.0);
    // An `.insv` times its video frames in the trailer: unless given, the
    // search starts from their period, for frames named by their number in
    // the video (ffmpeg's `-frame_pts 1`).
    let nominal_period = cfg.imu_frame_period.or_else(|| insv_frame_period(imu));
    let est_cfg = ImuScaleConfig {
        timing: FrameTiming::Search { nominal_period },
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let e = estimate_imu_scale(&samples, &frames, &est_cfg)
        .map_err(|e| EurocError::Sfm(format!("imu: {e}")))?;
    log(&format!(
        "imu: {} samples over {span:.1} s ({source}), {} frames; frames every {:.6} s from \
         {:+.4} s (drift {:+.1} / {:+.1} ms), rotation residual {:.3} deg, gyro bias {:.2} deg/s",
        samples.time.len(),
        frames.len(),
        e.period,
        e.offset,
        e.drift_ms[0],
        e.drift_ms[1],
        e.rotation_residual_deg,
        e.gyro_bias.norm().to_degrees()
    ));
    log(&format!(
        "imu: scale {:.4} m per SfM unit (SD {:.4} over {}/{} runs, |g| {:.3} m/s2, {:.1} s)",
        e.scale,
        e.scale_std,
        e.runs_kept(),
        e.runs.len(),
        e.gravity_median,
        t.elapsed().as_secs_f64()
    ));
    let mut text = format!(
        "# Metric scale from {} (docs/insta360_imu_scale.md), applied to the model\n\
         scale {:.6}\nscale_sd {:.6}\nruns {} {}\ngravity {:.4}\nframe_period {:.9}\n\
         frame_offset {:.6}\nrotation_residual_deg {:.4}\ngyro_bias_dps {:.4} {:.4} {:.4}\n\
         drift_ms {:.2} {:.2}\nspeed_jumps {}\n# run: start_s end_s intervals scale gravity kept\n",
        imu.display(),
        e.scale,
        e.scale_std,
        e.runs_kept(),
        e.runs.len(),
        e.gravity_median,
        e.period,
        e.offset,
        e.rotation_residual_deg,
        e.gyro_bias.x.to_degrees(),
        e.gyro_bias.y.to_degrees(),
        e.gyro_bias.z.to_degrees(),
        e.drift_ms[0],
        e.drift_ms[1],
        e.speed_jumps
    );
    for r in &e.runs {
        text += &format!(
            "run {:.3} {:.3} {} {:.6} {:.4} {}\n",
            r.start, r.end, r.intervals, r.scale, r.gravity, r.kept as u8
        );
    }
    let path = out_dir.join("imu_scale.txt");
    std::fs::write(&path, text).map_err(|source| EurocError::Io { path, source })?;
    Ok(e.scale)
}

/// The back lens's pose on the camera as the mapper refined it, from the
/// front faces of each registered dual-fisheye frame (images `10 i` and
/// `10 i + 5`, the two lenses' axes): the mean rotation taking a direction
/// in the back lens's frame to the front lens's, and the back lens's mean
/// centre in the front lens's frame. Written to `<out>/back_lens.txt`,
/// which `insv_stitch --back-lens` reads.
fn write_back_lens(
    poses: &[Option<visloc_core::geometry::Pose>],
    metric: bool,
    out_dir: &Path,
    log: &mut dyn FnMut(&str),
) -> Result<(), EurocError> {
    let mut quats: Vec<nalgebra::UnitQuaternion<f64>> = Vec::new();
    let mut centre = Vector3::zeros();
    for frame in poses.chunks_exact(10) {
        let (Some(front), Some(back)) = (&frame[0], &frame[5]) else {
            continue;
        };
        let (f, b) = (&front.world_to_camera, &back.world_to_camera);
        // Back lens to front lens: camera-to-world of the back, then
        // world-to-camera of the front.
        quats.push(f.rotation * b.rotation.inverse());
        centre += f.rotation * back.camera_center_world().coords + f.translation;
    }
    let Some(first) = quats.first().copied() else {
        return Ok(());
    };
    let n = quats.len() as f64;
    let sum = quats.iter().fold(nalgebra::Vector4::zeros(), |acc, q| {
        let v = q.as_ref().coords;
        if v.dot(&first.as_ref().coords) < 0.0 {
            acc - v
        } else {
            acc + v
        }
    });
    let rotation = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::from(sum))
        .to_rotation_matrix()
        .into_inner();
    let centre = centre / n;
    let off = rotation * dual_fisheye::back_lens_rotation().transpose();
    let beyond = ((off.trace() - 1.0) / 2.0)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    let unit = if metric { "m" } else { "model units" };
    log(&format!(
        "insv: back lens {beyond:.3}° past the half turn, centre ({:.4}, {:.4}, {:.4}) {unit} \
         in the front lens's frame ({} frames)",
        centre.x,
        centre.y,
        centre.z,
        quats.len()
    ));
    let r = &rotation;
    let text = format!(
        "# The back lens on the camera as gsplat_photos --insv refined it ({} frames)\n\
         # rotation: a direction in the back lens's frame to the front lens's, row by row\n\
         rotation {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}\n\
         # the back lens's centre in the front lens's frame ({unit})\n\
         centre {:.6} {:.6} {:.6}\n\
         beyond_half_turn_deg {beyond:.4}\n",
        quats.len(),
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
        centre.x,
        centre.y,
        centre.z
    );
    let path = out_dir.join("back_lens.txt");
    std::fs::write(&path, text).map_err(|source| EurocError::Io { path, source })
}

/// Calibrates an Insta360's two lenses from a [`build_photo_dataset`] run
/// `run` on its dual-fisheye frames (`cfg.insv`, as that run had it): every
/// observation in the run's model (`sparse/0`) goes back to its lens image
/// through the model the faces were cut with, and
/// [`dual_fisheye::calibrate_lenses`] refines each lens's optical centre
/// and focal lengths, the back lens's pose, the frames and the points, on
/// at most 200 frames spread over the run. The result is on the factory
/// calibration (`cfg.insv_lenses` included), for `insv_lenses`
/// (docs/insta360_lens.md).
pub fn calibrate_insv_lenses(
    run: &Path,
    cfg: &PhotoSfmConfig,
    log: &mut dyn FnMut(&str),
) -> Result<dual_fisheye::LensCalibration, EurocError> {
    use std::collections::HashMap;
    let fail = |m: String| EurocError::Sfm(format!("insv lens calibration: {m}"));
    let read = |name: &str| {
        let path = run.join(name);
        std::fs::read_to_string(&path).map_err(|source| EurocError::Io { path, source })
    };
    let video = cfg
        .insv
        .as_ref()
        .ok_or_else(|| fail("no insv video".into()))?;
    let telemetry = visloc_io::insv::read_insv_telemetry(video).map_err(|e| fail(e.to_string()))?;
    let [front, back] = telemetry.lenses.as_slice() else {
        return Err(fail(format!("no lens calibration in {}", video.display())));
    };
    // The lens images' size: the frames' height.
    let names = read("image_names.txt")?;
    let source = names
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .ok_or_else(|| fail("empty image_names.txt".into()))?;
    let (_, size) = image::image_dimensions(source).map_err(|e| fail(format!("{source}: {e}")))?;
    let mut models = [
        FisheyeLens::from_insv(front, size),
        FisheyeLens::from_insv(back, size),
    ];
    if let Some(prior) = &cfg.insv_lenses {
        models = prior.apply(&models, size);
    }
    let numbers = |l: &str| -> Vec<f64> {
        l.split_whitespace()
            .filter_map(|x| x.parse().ok())
            .collect()
    };
    let cameras: HashMap<u64, [f64; 4]> = read("sparse/0/cameras.txt")?
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let v: Vec<&str> = l.split_whitespace().collect();
            let p: Vec<f64> = v.get(4..8)?.iter().filter_map(|x| x.parse().ok()).collect();
            (v.get(1) == Some(&"PINHOLE") && p.len() == 4)
                .then(|| (v[0].parse().ok().unwrap_or(0), [p[0], p[1], p[2], p[3]]))
        })
        .collect();
    let points: HashMap<i64, Vector3<f64>> = read("sparse/0/points3D.txt")?
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let v = numbers(l);
            (v.len() >= 4).then(|| (v[0] as i64, Vector3::new(v[1], v[2], v[3])))
        })
        .collect();
    // Faces: image `i` is face `i % 5` of lens `(i % 10) / 5` of frame
    // `i / 10`; its lens's pose is the face's turned by the face's rotation.
    struct Face {
        frame: usize,
        lens: usize,
        rotation: Matrix3<f64>,
        camera: [f64; 4],
        observations: Vec<(f64, f64, i64)>,
    }
    let rotations = dual_fisheye::lens_face_rotations();
    let images = read("sparse/0/images.txt")?;
    let lines: Vec<&str> = images.lines().filter(|l| !l.starts_with('#')).collect();
    let mut faces = Vec::new();
    let mut lens_poses: HashMap<(usize, usize), (Matrix3<f64>, Vector3<f64>)> = HashMap::new();
    for pair in lines.chunks(2) {
        let head: Vec<&str> = pair[0].split_whitespace().collect();
        let (Some(name), Some(v)) = (head.get(9), head.get(1..9)) else {
            continue;
        };
        let v: Vec<f64> = v.iter().filter_map(|x| x.parse().ok()).collect();
        let index: Option<usize> = name
            .trim_start_matches("frame_")
            .trim_end_matches(".png")
            .parse()
            .ok();
        let (Some(i), true) = (index, v.len() == 8) else {
            continue;
        };
        let Some(&camera) = cameras.get(&(v[7] as u64)) else {
            continue;
        };
        let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            v[0], v[1], v[2], v[3],
        ));
        let rotation = rotations[i % 5].1;
        let (frame, lens) = (i / 10, (i % 10) / 5);
        lens_poses.entry((frame, lens)).or_insert((
            rotation * q.to_rotation_matrix().into_inner(),
            rotation * Vector3::new(v[4], v[5], v[6]),
        ));
        let observations = pair
            .get(1)
            .map(|l| {
                numbers(l)
                    .chunks_exact(3)
                    .filter(|o| o[2] >= 0.0)
                    .map(|o| (o[0], o[1], o[2] as i64))
                    .collect()
            })
            .unwrap_or_default();
        faces.push(Face {
            frame,
            lens,
            rotation,
            camera,
            observations,
        });
    }
    // At most 200 frames with the front lens posed, spread over the run.
    let mut posed: Vec<usize> = lens_poses
        .keys()
        .filter(|k| k.1 == 0)
        .map(|k| k.0)
        .collect();
    posed.sort_unstable();
    let keep = posed.len().min(200);
    if keep < 3 {
        return Err(fail(format!(
            "{} frames posed in {}",
            posed.len(),
            run.display()
        )));
    }
    let chosen: Vec<usize> = (0..keep).map(|k| posed[k * posed.len() / keep]).collect();
    let slot: HashMap<usize, usize> = chosen.iter().enumerate().map(|(k, &f)| (f, k)).collect();
    let frames: Vec<(Matrix3<f64>, Vector3<f64>)> =
        chosen.iter().map(|f| lens_poses[&(*f, 0)]).collect();
    // The back lens on the front one: the mean over the chosen frames.
    let pairs: Vec<(nalgebra::UnitQuaternion<f64>, Vector3<f64>)> = chosen
        .iter()
        .filter_map(|f| {
            let (rf, tf) = lens_poses.get(&(*f, 0))?;
            let (rb, tb) = lens_poses.get(&(*f, 1))?;
            let r = rb * rf.transpose();
            Some((nalgebra::UnitQuaternion::from_matrix(&r), tb - r * tf))
        })
        .collect();
    let Some((first, _)) = pairs.first().copied() else {
        return Err(fail("no frame with both lenses posed".into()));
    };
    let q = pairs
        .iter()
        .fold(nalgebra::Vector4::zeros(), |acc, (q, _)| {
            let v = q.as_ref().coords;
            if v.dot(&first.as_ref().coords) < 0.0 {
                acc - v
            } else {
                acc + v
            }
        });
    let back_pose = (
        nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::from(q))
            .to_rotation_matrix()
            .into_inner(),
        pairs.iter().map(|p| p.1).sum::<Vector3<f64>>() / pairs.len() as f64,
    );
    // Points seen at least three times in the chosen frames, at most 20,000.
    let mut seen: HashMap<i64, usize> = HashMap::new();
    for face in faces.iter().filter(|f| slot.contains_key(&f.frame)) {
        for o in &face.observations {
            *seen.entry(o.2).or_default() += 1;
        }
    }
    let mut ids: Vec<i64> = seen
        .iter()
        .filter(|(id, n)| **n >= 3 && points.contains_key(id))
        .map(|(id, _)| *id)
        .collect();
    ids.sort_unstable();
    let stride = ids.len().div_ceil(20_000).max(1);
    let ids: Vec<i64> = ids.into_iter().step_by(stride).collect();
    let index: HashMap<i64, usize> = ids.iter().enumerate().map(|(k, id)| (*id, k)).collect();
    let mut xyz: Vec<Vector3<f64>> = ids.iter().map(|id| points[id]).collect();
    // Each observation where the lens image shows it: back through the
    // face's pinhole and rotation, then the model the face was cut with.
    let mut observations = Vec::new();
    for face in &faces {
        let Some(&frame) = slot.get(&face.frame) else {
            continue;
        };
        let [fx, fy, cx, cy] = face.camera;
        for &(x, y, id) in &face.observations {
            let Some(&point) = index.get(&id) else {
                continue;
            };
            let ray = face.rotation * Vector3::new((x - cx) / fx, (y - cy) / fy, 1.0);
            if let Some((u, v)) = models[face.lens].project(&ray) {
                observations.push(dual_fisheye::LensObservation {
                    frame,
                    lens: face.lens,
                    point,
                    pixel: [u, v],
                });
            }
        }
    }
    let t0 = std::time::Instant::now();
    // The faces see each lens within 90° of its axis only, which leaves the
    // two centres free to move together a little. Where the lenses see the
    // same things (SIFT along the seam, `dual_fisheye::seam_pairs`, in up to
    // 16 frames) pins them to each other: each match is a point both lenses
    // of its frame see, started at the scene's typical distance.
    let sources: HashMap<usize, &str> = names
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let i: usize = it
                .next()?
                .trim_start_matches("frame_")
                .trim_end_matches(".png")
                .parse()
                .ok()?;
            Some((i / 10, it.next()?))
        })
        .collect();
    let mut depths: Vec<f64> = observations
        .iter()
        .filter(|o| o.lens == 0)
        .step_by(97)
        .map(|o| (frames[o.frame].0 * xyz[o.point] + frames[o.frame].1).norm())
        .collect();
    depths.sort_by(f64::total_cmp);
    let depth = depths.get(depths.len() / 2).copied().unwrap_or(1.0);
    let seam_frames = keep.min(16);
    let mut seam = 0;
    for k in 0..seam_frames {
        let slot = k * keep / seam_frames;
        let Some(path) = sources.get(&chosen[slot]) else {
            continue;
        };
        let Ok(img) = image::open(path) else {
            continue;
        };
        let (r, t) = frames[slot];
        for (front_ray, back_ray) in
            dual_fisheye::seam_pairs(&img.to_rgb8(), &models, &back_pose.0.transpose())
        {
            let (Some(a), Some(b)) = (models[0].project(&front_ray), models[1].project(&back_ray))
            else {
                continue;
            };
            let point = xyz.len();
            xyz.push(r.transpose() * (front_ray * depth - t));
            for (lens, pixel) in [(0, [a.0, a.1]), (1, [b.0, b.1])] {
                observations.push(dual_fisheye::LensObservation {
                    frame: slot,
                    lens,
                    point,
                    pixel,
                });
            }
            seam += 1;
        }
    }
    let mut cal =
        dual_fisheye::calibrate_lenses(&models, size, &observations, &frames, &back_pose, &xyz);
    if let Some(prior) = &cfg.insv_lenses {
        let s = size as f64 / prior.size.max(1) as f64;
        for k in 0..2 {
            for a in 0..2 {
                cal.centre_offsets[k][a] += prior.centre_offsets[k][a] * s;
            }
            cal.focal_scales[k] *= prior.focal_scales[k];
        }
    }
    let to_video = 3840.0 / size as f64;
    let past = cal.back_rotation * dual_fisheye::back_lens_rotation().transpose();
    log(&format!(
        "insv: lenses calibrated on {} frames, {} points, {} observations, {seam} seam matches \
         ({:.1} s): optical \
         centres ({:+.1}, {:+.1}) and ({:+.1}, {:+.1}) px from the factory model's (of a \
         3840 px lens image), focal lengths x{:.4} / x{:.4}; the back lens {:.2}° past the \
         half turn; reprojection {:.3} -> {:.3} px",
        frames.len(),
        xyz.len(),
        observations.len(),
        t0.elapsed().as_secs_f64(),
        cal.centre_offsets[0][0] * to_video,
        cal.centre_offsets[0][1] * to_video,
        cal.centre_offsets[1][0] * to_video,
        cal.centre_offsets[1][1] * to_video,
        cal.focal_scales[0],
        cal.focal_scales[1],
        ((past.trace() - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees(),
        cal.reprojection[0],
        cal.reprojection[1],
    ));
    Ok(cal)
}

/// Read, decode (oriented) and scale the photos (or panoramas), and read
/// their EXIF focal lengths and whether the metadata names a 360°
/// panorama, in parallel. Keeps the most common resolution (one camera for
/// the trainer): returns the files kept, their images and EXIF focal
/// lengths (35 mm equivalent), and the resolution.
#[allow(clippy::type_complexity)]
fn load_photos(
    files: Vec<PathBuf>,
    cfg: &PhotoSfmConfig,
    log: &mut dyn FnMut(&str),
) -> Result<(Vec<PathBuf>, Vec<(image::RgbImage, Option<f64>)>, u32, u32), EurocError> {
    let t_load = std::time::Instant::now();
    let loaded: Vec<((image::RgbImage, Option<f64>), bool)> = files
        .par_iter()
        .map(|p| -> Result<_, EurocError> {
            let bytes = std::fs::read(p).map_err(|source| EurocError::Io {
                path: p.clone(),
                source,
            })?;
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
    Ok((files, decoded, width, height))
}

/// Cut dual-fisheye frames into their lens faces (the front lens's five,
/// then the back lens's) and write them to `dir` as `frame_{i}.png`, a
/// frame at a time: only the frames being cut are in memory, not every
/// frame and face. The frames are scaled for the faces to be at most
/// `max_size`, at most the lenses' resolution on their axes; frames of
/// another size than the most common are dropped. Returns the frames kept,
/// the faces' labels and paths, and the faces' side.
#[allow(clippy::type_complexity)]
fn write_lens_faces(
    files: Vec<PathBuf>,
    lenses: &[visloc_io::insv::InsvLens],
    cfg: &PhotoSfmConfig,
    dir: &Path,
    log: &mut dyn FnMut(&str),
) -> Result<(Vec<PathBuf>, Vec<String>, Vec<PathBuf>, u32), EurocError> {
    let t0 = std::time::Instant::now();
    // The side of the faces at a lens image's resolution on its axis, for a
    // frame `height` pixels high.
    let natural_face = |height: u32| -> u32 {
        lenses
            .iter()
            .map(|lens| dual_fisheye::natural_face_size(&FisheyeLens::from_insv(lens, height)))
            .max()
            .unwrap_or(height)
    };
    let dims: Vec<(u32, u32)> = files
        .par_iter()
        .map(|p| {
            image::image_dimensions(p).map_err(|source| EurocError::Image {
                path: p.clone(),
                source,
            })
        })
        .collect::<Result<_, _>>()?;
    let mut sizes = dims.clone();
    sizes.sort_unstable();
    let (w, h) = sizes
        .chunk_by(|a, b| a == b)
        .max_by_key(|c| c.len())
        .map(|c| c[0])
        .expect("at least 2 frames");
    let near = |(a, b): (u32, u32)| {
        a.abs_diff(w) as f64 <= 0.01 * w as f64 && b.abs_diff(h) as f64 <= 0.01 * h as f64
    };
    let files: Vec<PathBuf> = files
        .into_iter()
        .zip(&dims)
        .filter(|&(_, &d)| near(d))
        .map(|(f, _)| f)
        .collect();
    let s = (cfg.max_size as f64 / natural_face(h).max(1) as f64).min(1.0);
    let (width, height) = if s < 1.0 {
        (
            ((w as f64 * s).round() as u32).max(1),
            ((h as f64 * s).round() as u32).max(1),
        )
    } else {
        (w, h)
    };
    if width.abs_diff(2 * height) as f64 > 0.01 * width as f64 {
        return Err(EurocError::Sfm(format!(
            "insv: the frames are {width}x{height}, not two square lenses side by side"
        )));
    }
    let size = natural_face(height).min(cfg.max_size).max(1);
    let faces = dual_fisheye::lens_face_rotations();
    let mut lens_models: Vec<FisheyeLens> = lenses
        .iter()
        .map(|l| FisheyeLens::from_insv(l, height))
        .collect();
    if let (Some(cal), [front, back]) = (&cfg.insv_lenses, lens_models.as_slice()) {
        lens_models = cal.apply(&[front.clone(), back.clone()], height).to_vec();
    }
    let per_frame = lens_models.len() * faces.len();
    let path = |i: usize| dir.join(format!("frame_{i:05}.png"));
    files
        .par_iter()
        .enumerate()
        .try_for_each(|(f, p)| -> Result<(), EurocError> {
            let bytes = std::fs::read(p).map_err(|source| EurocError::Io {
                path: p.clone(),
                source,
            })?;
            let frame = load_oriented(p, &bytes)?;
            let frame = if frame.dimensions() == (width, height) {
                frame
            } else {
                resize_triangle(&frame, width, height)
            };
            for (l, (lens, x0)) in lens_models.iter().zip([0, width - height]).enumerate() {
                let half = image::imageops::crop_imm(&frame, x0, 0, height, height).to_image();
                for (k, (_, r)) in faces.iter().enumerate() {
                    let face = dual_fisheye::lens_face(&half, lens, r, size);
                    let i = per_frame * f + faces.len() * l + k;
                    save_working_image(&face, &path(i), cfg.mask_black, None)?;
                }
            }
            Ok(())
        })?;
    let mut labels = Vec::with_capacity(per_frame * files.len());
    for f in &files {
        for lens in ["front lens", "back lens"] {
            for (face, _) in &faces {
                labels.push(format!("{} {lens} {face}", f.display()));
            }
        }
    }
    log(&format!(
        "insv: {} lens faces of {size}x{size} px from {} dual-fisheye frames at {width}x{height} \
         (others dropped: {}; {:.1} s)",
        labels.len(),
        files.len(),
        dims.len() - files.len(),
        t0.elapsed().as_secs_f64()
    ));
    let paths = (0..labels.len()).map(path).collect();
    Ok((files, labels, paths, size))
}

/// A [`PhotoSfmConfig::camera`] from a `cameras.txt` row without its id:
/// `MODEL WIDTH HEIGHT PARAMS...`, with exactly the model's COLMAP
/// parameter count.
pub fn parse_camera_row(row: &str) -> Result<Camera, String> {
    let usage = "a camera is \"MODEL WIDTH HEIGHT PARAMS...\" (a cameras.txt row without its id)";
    let camera = visloc_io::colmap::parse_cameras_txt(&format!("1 {row}"))
        .map_err(|e| format!("{usage}: {e}"))?
        .into_iter()
        .next()
        .ok_or(usage)?;
    match camera.model.colmap_param_count() {
        Some(n) if n == camera.params.len() && camera.intrinsics().is_some() => Ok(camera),
        Some(n) => Err(format!(
            "{}: {n} parameters, {} given",
            row.split_whitespace().next().unwrap_or_default(),
            camera.params.len()
        )),
        None => Err(format!("{usage}: unknown model {:?}", camera.model)),
    }
}

/// `camera` on a `width x height` copy of its images, as camera 1: the
/// focal lengths and principal point scale with the image, the distortion
/// (on normalized coordinates) stays. The copy must keep the aspect ratio.
fn scaled_camera(camera: &Camera, width: u32, height: u32) -> Result<Camera, EurocError> {
    let (fx, fy, cx, cy) = camera.intrinsics().ok_or_else(|| {
        EurocError::Camera(format!(
            "camera model {:?} has no focal length",
            camera.model
        ))
    })?;
    let sx = width as f64 / camera.width as f64;
    let sy = height as f64 / camera.height as f64;
    if (sx / sy - 1.0).abs() > 0.01 {
        return Err(EurocError::Camera(format!(
            "the photos are {width}x{height}, the camera {}x{}: another aspect ratio",
            camera.width, camera.height
        )));
    }
    let mut scaled = Camera {
        id: 1,
        width,
        height,
        ..camera.clone()
    };
    scaled.set_intrinsics(fx * sx, fy * sy, cx * sx, cy * sy);
    Ok(scaled)
}

/// Write a working image as PNG; with `mask_black` or `outside` (pixels to
/// leave out), RGBA with alpha 0 on its near-black regions and there.
fn save_working_image(
    img: &image::RgbImage,
    path: &Path,
    mask_black: Option<u8>,
    outside: Option<&[bool]>,
) -> Result<(), EurocError> {
    let masked = match (mask_black.map(|thr| black_regions(img, thr)), outside) {
        (None, None) => None,
        (Some(black), None) => Some(black),
        (None, Some(out)) => Some(out.to_vec()),
        (Some(black), Some(out)) => Some(black.iter().zip(out).map(|(a, b)| a | b).collect()),
    };
    match masked {
        None => img.save(path),
        Some(masked) => {
            let mut rgba = image::RgbaImage::new(img.width(), img.height());
            for ((o, p), &m) in rgba.pixels_mut().zip(img.pixels()).zip(&masked) {
                *o = image::Rgba([p[0], p[1], p[2], if m { 0 } else { 255 }]);
            }
            rgba.save(path)
        }
    }
    .map_err(|source| EurocError::Image {
        path: path.to_path_buf(),
        source,
    })
}

/// A chunk of `count` images' features from `vl.extract_batch`'s `batch`, the
/// chunk's first image being image `start`: image `k`'s keypoints less those
/// on its mask, `load_mask(start + k)`, decoded when they are filtered (a
/// thread's worth at a time, not the chunk's at once). A batch the device
/// cannot take (`Unsupported`, for photos past its binding limit) is
/// extracted on the CPU, one image at a time, by `cpu_extract(k)`.
#[cfg(feature = "gpu")]
#[allow(clippy::type_complexity)]
fn chunk_features(
    start: usize,
    count: usize,
    batch: Result<
        Vec<(
            Vec<visloc_vision::features::sift::SiftKeypoint>,
            Vec<Vec<f32>>,
        )>,
        visloc_sift_gpu::SiftGpuError,
    >,
    cpu_extract: &mut dyn FnMut(
        usize,
    ) -> Result<
        (
            Vec<visloc_vision::features::sift::SiftKeypoint>,
            Vec<Vec<f32>>,
        ),
        EurocError,
    >,
    features_of: &(dyn Fn(
        Vec<visloc_vision::features::sift::SiftKeypoint>,
        Vec<Vec<f32>>,
        Option<&[bool]>,
    ) -> (FeatureSet, Vec<u8>)
          + Sync),
    load_mask: &(dyn Fn(usize) -> Result<Option<Vec<bool>>, EurocError> + Sync),
) -> Result<Vec<(FeatureSet, Vec<u8>)>, EurocError> {
    match batch {
        Err(visloc_sift_gpu::SiftGpuError::Unsupported) => {
            extract_chunk(start, count, cpu_extract, features_of, load_mask)
        }
        other => other
            .map_err(|e| EurocError::Sift(format!("{e}")))?
            .into_par_iter()
            .enumerate()
            .map(|(k, (kps, desc))| Ok(features_of(kps, desc, load_mask(start + k)?.as_deref())))
            .collect(),
    }
}

/// The features of a chunk of `count` images one at a time, the chunk's first
/// being image `start`: `extract(k)` gives chunk image `k`'s keypoints and
/// descriptors, and `load_mask(start + k)` its mask, decoded when they are
/// filtered.
#[cfg(feature = "gpu")]
#[allow(clippy::type_complexity)]
fn extract_chunk(
    start: usize,
    count: usize,
    extract: &mut dyn FnMut(
        usize,
    ) -> Result<
        (
            Vec<visloc_vision::features::sift::SiftKeypoint>,
            Vec<Vec<f32>>,
        ),
        EurocError,
    >,
    features_of: &(dyn Fn(
        Vec<visloc_vision::features::sift::SiftKeypoint>,
        Vec<Vec<f32>>,
        Option<&[bool]>,
    ) -> (FeatureSet, Vec<u8>)
          + Sync),
    load_mask: &(dyn Fn(usize) -> Result<Option<Vec<bool>>, EurocError> + Sync),
) -> Result<Vec<(FeatureSet, Vec<u8>)>, EurocError> {
    (0..count)
        .map(|k| {
            let (kps, desc) = extract(k)?;
            Ok(features_of(kps, desc, load_mask(start + k)?.as_deref()))
        })
        .collect()
}

/// Images extracted one at a time in chunks of `chunk_size`: for each chunk,
/// `prepare(start, count)` readies the chunk's images (`count` from image
/// `start`), `extract(&chunk, k)` extracts the chunk's image `k`
/// ([`extract_chunk`] masks it with `load_mask(start + k)`), and
/// `done(&chunk, &features)` sees the chunk's features. Returns every
/// image's features.
#[cfg(feature = "gpu")]
#[allow(clippy::type_complexity)]
fn extract_chunked<P>(
    n: usize,
    chunk_size: usize,
    mut prepare: impl FnMut(usize, usize) -> Result<P, EurocError>,
    mut extract: impl FnMut(
        &P,
        usize,
    ) -> Result<
        (
            Vec<visloc_vision::features::sift::SiftKeypoint>,
            Vec<Vec<f32>>,
        ),
        EurocError,
    >,
    features_of: &(dyn Fn(
        Vec<visloc_vision::features::sift::SiftKeypoint>,
        Vec<Vec<f32>>,
        Option<&[bool]>,
    ) -> (FeatureSet, Vec<u8>)
          + Sync),
    load_mask: &(dyn Fn(usize) -> Result<Option<Vec<bool>>, EurocError> + Sync),
    mut done: impl FnMut(&P, &[(FeatureSet, Vec<u8>)]),
) -> Result<Vec<(FeatureSet, Vec<u8>)>, EurocError> {
    let mut all = Vec::with_capacity(n);
    for start in (0..n).step_by(chunk_size) {
        let count = chunk_size.min(n - start);
        let chunk = prepare(start, count)?;
        let features = extract_chunk(
            start,
            count,
            &mut |k| extract(&chunk, k),
            features_of,
            load_mask,
        )?;
        done(&chunk, &features);
        all.extend(features);
    }
    Ok(all)
}

/// Images extracted in batches, in chunks of `chunk_size`: for each chunk,
/// `prepare(start, count)` readies the chunk's images (`count` from image
/// `start`), `batch(&chunk)` extracts them all at once, `cpu_extract(&chunk,
/// k)` extracts one on the CPU if the device cannot take the batch (see
/// [`chunk_features`], which masks the keypoints with `load_mask(start + k)`),
/// and `done(&chunk, &features)` sees the chunk's features. Returns every
/// image's features.
#[cfg(feature = "gpu")]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn extract_batched<P>(
    n: usize,
    chunk_size: usize,
    mut prepare: impl FnMut(usize, usize) -> Result<P, EurocError>,
    mut batch: impl FnMut(
        &P,
    ) -> Result<
        Result<
            Vec<(
                Vec<visloc_vision::features::sift::SiftKeypoint>,
                Vec<Vec<f32>>,
            )>,
            visloc_sift_gpu::SiftGpuError,
        >,
        EurocError,
    >,
    mut cpu_extract: impl FnMut(
        &P,
        usize,
    ) -> Result<
        (
            Vec<visloc_vision::features::sift::SiftKeypoint>,
            Vec<Vec<f32>>,
        ),
        EurocError,
    >,
    features_of: &(dyn Fn(
        Vec<visloc_vision::features::sift::SiftKeypoint>,
        Vec<Vec<f32>>,
        Option<&[bool]>,
    ) -> (FeatureSet, Vec<u8>)
          + Sync),
    load_mask: &(dyn Fn(usize) -> Result<Option<Vec<bool>>, EurocError> + Sync),
    mut done: impl FnMut(&P, &[(FeatureSet, Vec<u8>)]),
) -> Result<Vec<(FeatureSet, Vec<u8>)>, EurocError> {
    let mut all = Vec::with_capacity(n);
    for start in (0..n).step_by(chunk_size) {
        let count = chunk_size.min(n - start);
        let chunk = prepare(start, count)?;
        let results = batch(&chunk)?;
        let features = chunk_features(
            start,
            count,
            results,
            &mut |k| cpu_extract(&chunk, k),
            features_of,
            load_mask,
        )?;
        done(&chunk, &features);
        all.extend(features);
    }
    Ok(all)
}

/// The images SfM extracts features from: decoded in memory, or written
/// already (lens faces) and read back, an image at a time. PNG is lossless,
/// so the features are the same.
enum WorkImages {
    Memory(Vec<(image::RgbImage, Option<f64>)>),
    Disk(Vec<PathBuf>),
}

impl WorkImages {
    fn len(&self) -> usize {
        match self {
            Self::Memory(images) => images.len(),
            Self::Disk(paths) => paths.len(),
        }
    }

    fn get(&self, i: usize) -> Result<std::borrow::Cow<'_, image::RgbImage>, EurocError> {
        match self {
            Self::Memory(images) => Ok(std::borrow::Cow::Borrowed(&images[i].0)),
            Self::Disk(paths) => image::open(&paths[i])
                .map(|img| std::borrow::Cow::Owned(img.into_rgb8()))
                .map_err(|source| EurocError::Image {
                    path: paths[i].clone(),
                    source,
                }),
        }
    }
}

/// The colour at each keypoint (its nearest pixel; grey off the image). An
/// SfM point takes its first observation's, so the images need not be kept
/// past feature extraction.
fn keypoint_colours(img: &image::RgbImage, keypoints: &[Point2<f64>]) -> Vec<[u8; 3]> {
    let (w, h) = img.dimensions();
    keypoints
        .iter()
        .map(|p| {
            let (x, y) = (p.x.round(), p.y.round());
            if x >= 0.0 && y >= 0.0 && x < w as f64 && y < h as f64 {
                img.get_pixel(x as u32, y as u32).0
            } else {
                [128; 3]
            }
        })
        .collect()
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
    let min_images = if cfg.equirect || cfg.insv.is_some() {
        2
    } else {
        3
    };
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
    if cfg.mask_flip && cfg.mask_dir.is_none() {
        return Err(EurocError::Sfm("masks: mask_flip needs mask_dir".into()));
    }
    if let Some(dir) = &cfg.mask_dir {
        if cfg.equirect || cfg.insv.is_some() {
            return Err(EurocError::Sfm(
                "masks: not supported with equirect or insv, whose images are cut from the \
                 photos; drop mask_dir (--masks)"
                    .into(),
            ));
        }
        // Fail on a missing mask, or one of another aspect, before decoding
        // anything (the headers are enough).
        ImageMasks::find(dir, &files, cfg.mask_flip)?.check_aspect(&files)?;
    }
    // The lenses of dual-fisheye frames, from the video's metadata.
    let lenses: Option<Vec<visloc_io::insv::InsvLens>> = match &cfg.insv {
        None => None,
        Some(_) if cfg.equirect => {
            return Err(EurocError::Sfm(
                "insv: the frames are dual fisheye, not equirectangular; drop equirect".into(),
            ))
        }
        Some(_) if cfg.focal_px.is_some() => {
            return Err(EurocError::Sfm(
                "insv: the lens faces' focal length is fixed; drop focal_px (--focal)".into(),
            ))
        }
        Some(path) => {
            let t = visloc_io::insv::read_insv_telemetry(path)
                .map_err(|e| EurocError::Sfm(format!("insv: {e}")))?;
            if t.lenses.len() != 2 {
                return Err(EurocError::Sfm(format!(
                    "insv: no lens calibration in {}'s metadata (decoded for the X5's layout; \
                     docs/insta360_lens.md)",
                    path.display()
                )));
            }
            log(&format!(
                "insv: {} lenses, xi {:.2}, fx {:.1} / {:.1}, image circles {:.1} / {:.1} px \
                 (calibration pixels)",
                t.camera.as_deref().unwrap_or("Insta360"),
                t.lenses[0].xi,
                t.lenses[0].fx,
                t.lenses[1].fx,
                t.lenses[0].circle_radius,
                t.lenses[1].circle_radius
            ));
            Some(t.lenses)
        }
    };
    // The working images, `out_dir/images/frame_{i}.png`. Image `i` comes
    // from source image `i / per_source`: a panorama gives its six cube
    // faces, consecutive, in panorama order, and a dual-fisheye frame its
    // front lens's five faces, then its back lens's.
    let images_out = out_dir.join("images");
    std::fs::create_dir_all(&images_out).map_err(io(&images_out))?;
    let per_source = if cfg.equirect {
        6
    } else if lenses.is_some() {
        10
    } else {
        1
    };
    let (files, labels, work, width, height, panoramas) = if let Some(lenses) = &lenses {
        let (files, labels, paths, size) = write_lens_faces(files, lenses, cfg, &images_out, log)?;
        (files, labels, WorkImages::Disk(paths), size, size, None)
    } else if cfg.equirect {
        let (files, decoded, width, height) = load_photos(files, cfg, log)?;
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
        // SfM on the panoramas: at most `2 max_size` wide (COLMAP's 3200 px
        // SIFT cap at the default), half the faces' angular resolution; a
        // full-size panorama's SIFT pyramid takes gigabytes.
        let panoramas = cfg.equirect_native.then(|| {
            let w = width.min(2 * cfg.max_size).max(2);
            let h = w / 2;
            let scaled: Vec<(image::RgbImage, Option<f64>)> = if w == width {
                panoramas
            } else {
                panoramas
                    .into_par_iter()
                    .map(|(pano, f)| (resize_triangle(&pano, w, h), f))
                    .collect()
            };
            (scaled, w, h)
        });
        (
            files,
            labels,
            WorkImages::Memory(decoded),
            size,
            size,
            panoramas,
        )
    } else {
        let (files, decoded, width, height) = load_photos(files, cfg, log)?;
        let labels = files.iter().map(|f| f.display().to_string()).collect();
        (
            files,
            labels,
            WorkImages::Memory(decoded),
            width,
            height,
            None,
        )
    };
    // The masks of the images kept (`load_photos` drops some), one decoded
    // when its image is processed.
    let masks = match &cfg.mask_dir {
        Some(dir) => {
            let masks = ImageMasks::find(dir, &files, cfg.mask_flip)?;
            log(&format!(
                "masks: {} from {}{}",
                files.len(),
                dir.display(),
                if cfg.mask_flip { " (flipped)" } else { "" }
            ));
            Some(masks)
        }
        None => None,
    };
    let n = work.len();
    let long = width.max(height) as f64;
    let mut f35s: Vec<f64> = match &work {
        WorkImages::Memory(images) => images.iter().filter_map(|d| d.1).collect(),
        WorkImages::Disk(_) => Vec::new(),
    };
    f35s.sort_by(f64::total_cmp);
    let calibrated = match &cfg.camera {
        Some(_) if cfg.equirect || lenses.is_some() => {
            return Err(EurocError::Camera(
                "a calibrated camera does not combine with equirect or insv".into(),
            ))
        }
        Some(camera) => Some(scaled_camera(camera, width, height)?),
        None if cfg.refine_camera => {
            return Err(EurocError::Camera(
                "refining the camera needs a camera to start from".into(),
            ))
        }
        None => None,
    };
    let (focal_prior, focal_source) = match (cfg.focal_px, f35s.get(f35s.len() / 2)) {
        _ if cfg.equirect => (width as f64 / 2.0, "cube face, 90° field of view"),
        _ if lenses.is_some() => (width as f64 / 2.0, "lens face, 90° field of view"),
        _ if calibrated.is_some() => (
            calibrated
                .as_ref()
                .and_then(|c| c.intrinsics())
                .map_or(0.0, |k| k.0),
            "--camera",
        ),
        (Some(f), _) => (f, "--focal"),
        (None, Some(f35)) => (f35 / 36.0 * long, "EXIF 35 mm"),
        (None, None) => (1.2 * long, "prior 1.2 x long side"),
    };
    log(&format!("focal {focal_prior:.1} px ({focal_source})"));
    let (cx, cy) = cfg
        .principal_point
        .unwrap_or((width as f64 / 2.0, height as f64 / 2.0));
    let camera = match calibrated {
        Some(camera) => {
            log(&format!(
                "camera: {} {:?} ({})",
                camera.model.colmap_name().unwrap_or("?"),
                camera.params,
                if cfg.refine_camera {
                    "refined by the mapper"
                } else {
                    "held fixed"
                }
            ));
            camera
        }
        None => Camera::pinhole(1, width, height, focal_prior, focal_prior, cx, cy),
    };
    // The pixels past `max_angle` off the lens's axis.
    let outside: Option<Vec<bool>> = match cfg.max_angle {
        None => None,
        Some(_) if cfg.camera.is_none() => {
            return Err(EurocError::Camera(
                "max_angle needs a calibrated camera".into(),
            ))
        }
        Some(deg) => {
            let cos_max = deg.to_radians().cos();
            let past = |x: u32, y: u32| {
                camera
                    .unit_ray_from_pixel(&Point2::new(x as f64 + 0.5, y as f64 + 0.5))
                    .is_none_or(|ray| ray.z < cos_max)
            };
            let mask: Vec<bool> = (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .map(|(x, y)| past(x, y))
                .collect();
            log(&format!(
                "leaving out {:.1}% of each image, past {deg}° off the axis",
                100.0 * mask.iter().filter(|&&m| m).count() as f64 / mask.len() as f64
            ));
            Some(mask)
        }
    };

    let names: Vec<String> = (0..n).map(|i| format!("frame_{i:05}.png")).collect();
    if let WorkImages::Memory(images) = &work {
        images
            .par_iter()
            .zip(&names)
            .enumerate()
            .map(|(i, ((img, _), name))| {
                // The masks' alpha joins the other regions left out.
                let from_file = masks
                    .as_ref()
                    .map(|m| m.load(i, width, height))
                    .transpose()?;
                let left_out: Option<std::borrow::Cow<[bool]>> =
                    match (outside.as_deref(), from_file) {
                        (None, None) => None,
                        (Some(o), None) => Some(std::borrow::Cow::Borrowed(o)),
                        (None, Some(m)) => Some(std::borrow::Cow::Owned(m)),
                        (Some(o), Some(mut m)) => {
                            m.iter_mut().zip(o).for_each(|(a, b)| *a |= b);
                            Some(std::borrow::Cow::Owned(m))
                        }
                    };
                save_working_image(
                    img,
                    &images_out.join(name),
                    cfg.mask_black,
                    left_out.as_deref(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
    }
    let listing: String = names
        .iter()
        .zip(&labels)
        .map(|(n, l)| format!("{n} {l}\n"))
        .collect();
    let _ = std::fs::write(out_dir.join("image_names.txt"), listing);

    // With `equirect_native` SfM runs on the panoramas, one equirectangular
    // camera; the faces written above (`faces`: their names, size and
    // pinhole camera) are posed from the panoramas afterwards.
    let (work, n, per_source, width, height, camera, names, faces) = match panoramas {
        Some((panoramas, w, h)) => {
            log(&format!(
                "equirect: SfM on the {} panoramas ({w}x{h} px, equirectangular camera)",
                panoramas.len()
            ));
            let n = panoramas.len();
            let pano_names = (0..n).map(|i| format!("panorama_{i:05}")).collect();
            let camera_360 = Camera::equirectangular(1, w, h);
            let faces = Some((names, width, camera));
            (
                WorkImages::Memory(panoramas),
                n,
                1,
                w,
                h,
                camera_360,
                pano_names,
                faces,
            )
        }
        None => (work, n, per_source, width, height, camera, names, None),
    };
    let source = |i: usize| i / per_source;

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
    // VLFeat-compatible descriptors are kept as their bytes
    // (`unit_scale_descriptors`), the others as f32 in the features.
    let byte_scale = sift_cfg.vlfeat_compatible_descriptor;
    let to_features = |kps: Vec<visloc_vision::features::sift::SiftKeypoint>,
                       desc: Vec<Vec<f32>>|
     -> (FeatureSet, Vec<u8>) {
        let keypoints = kps.iter().map(|k| Point2::new(k.x, k.y)).collect();
        if byte_scale {
            debug_assert!(desc.iter().all(|d| d.len() == SIFT_DIM));
            let bytes = desc.iter().flatten().map(|&v| v as u8).collect();
            let features = FeatureSet {
                keypoints,
                descriptors: Vec::new(),
            };
            (features, bytes)
        } else {
            let features = FeatureSet {
                keypoints,
                descriptors: desc,
            };
            (features, Vec::new())
        }
    };
    // Image `i`'s mask (`true` = masked out), if there are masks: decoded
    // when its keypoints are extracted, not kept.
    let load_mask = |i: usize| -> Result<Option<Vec<bool>>, EurocError> {
        masks.as_ref().map(|m| m.load(i, width, height)).transpose()
    };
    // An image's features, less the keypoints on its masked-out pixels, which
    // the matcher and the mapper then never see.
    let dropped = std::sync::atomic::AtomicUsize::new(0);
    let features_of = |kps: Vec<visloc_vision::features::sift::SiftKeypoint>,
                       desc: Vec<Vec<f32>>,
                       masked: Option<&[bool]>|
     -> (FeatureSet, Vec<u8>) {
        let (kps, desc) = match masked {
            None => (kps, desc),
            Some(masked) => {
                let before = kps.len();
                let kept = drop_masked_keypoints(
                    kps,
                    desc,
                    masked,
                    width as usize,
                    sift_cfg.vlfeat_compatible_detector,
                );
                dropped.fetch_add(before - kept.0.len(), std::sync::atomic::Ordering::Relaxed);
                kept
            }
        };
        to_features(kps, desc)
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
    // With each image's keypoints, the colour at each (`keypoint_colours`).
    #[allow(clippy::type_complexity)]
    let (sift, keypoint_rgb): (Vec<(FeatureSet, Vec<u8>)>, Vec<Vec<[u8; 3]>>) = if vlfeat_gpu {
        #[allow(unused_mut)]
        let mut features = Vec::with_capacity(n);
        #[allow(unused_mut)]
        let mut colours = Vec::with_capacity(n);
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
            // Pipelined: the GPU detects the next photo while the host
            // handles the current one. Photos past the device's binding
            // limit go to the CPU.
            features.extend(extract_batched(
                n,
                256,
                |start, count| {
                    let chunk = (start..start + count)
                        .into_par_iter()
                        .map(|i| work.get(i))
                        .collect::<Result<Vec<_>, _>>()?;
                    let grays: Vec<Vec<f32>> = chunk.par_iter().map(|img| gray_of(img)).collect();
                    Ok((chunk, grays))
                },
                |(_, grays)| {
                    let images = grays
                        .iter()
                        .map(|g| GrayImage::new(width as usize, height as usize, g))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| EurocError::Sift(format!("{e}")))?;
                    Ok(vl.extract_batch(&images, &sift_cfg))
                },
                |(_, grays), k| {
                    let gray = GrayImage::new(width as usize, height as usize, &grays[k])
                        .map_err(|e| EurocError::Sift(format!("{e}")))?;
                    extract_sift(&gray, &sift_cfg).map_err(|e| EurocError::Sift(format!("{e}")))
                },
                &features_of,
                &load_mask,
                |(chunk, _), features| {
                    colours.extend(
                        chunk
                            .iter()
                            .zip(features)
                            .map(|(img, f)| keypoint_colours(img, &f.0.keypoints)),
                    )
                },
            )?);
            if cfg.gpu {
                gpu_sift = Some(visloc_sift_gpu::SiftGpu::new(vl.into_context()));
            }
        }
        (features, colours)
    } else if on_gpu {
        #[allow(unused_mut)]
        let mut features = Vec::with_capacity(n);
        #[allow(unused_mut)]
        let mut colours = Vec::with_capacity(n);
        #[cfg(feature = "gpu")]
        features.extend(extract_chunked(
            n,
            256,
            |start, count| {
                let chunk = (start..start + count)
                    .into_par_iter()
                    .map(|i| work.get(i))
                    .collect::<Result<Vec<_>, _>>()?;
                let grays: Vec<Vec<f32>> = chunk.par_iter().map(|img| gray_of(img)).collect();
                Ok((chunk, grays))
            },
            |(_, grays), k| {
                let gray = GrayImage::new(width as usize, height as usize, &grays[k])
                    .map_err(|e| EurocError::Sift(format!("{e}")))?;
                gpu_sift
                    .as_mut()
                    .expect("on_gpu")
                    .extract(&gray, &sift_cfg)
                    .map_err(|e| EurocError::Sift(format!("{e}")))
            },
            &features_of,
            &load_mask,
            |(chunk, _), features| {
                colours.extend(
                    chunk
                        .iter()
                        .zip(features)
                        .map(|(img, f)| keypoint_colours(img, &f.0.keypoints)),
                )
            },
        )?);
        (features, colours)
    } else {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let img = work.get(i)?;
                let g = gray_of(&img);
                let gray = GrayImage::new(width as usize, height as usize, &g)
                    .map_err(|e| EurocError::Sift(format!("{e}")))?;
                let (kps, desc) =
                    extract_sift(&gray, &sift_cfg).map_err(|e| EurocError::Sift(format!("{e}")))?;
                let f = features_of(kps, desc, load_mask(i)?.as_deref());
                let colours = keypoint_colours(&img, &f.0.keypoints);
                Ok((f, colours))
            })
            .collect::<Result<Vec<_>, EurocError>>()?
            .into_iter()
            .unzip()
    };
    drop(work);
    let (features, desc_bytes): (Vec<FeatureSet>, Vec<Vec<u8>>) = sift.into_iter().unzip();
    // Image `i`'s descriptors as the matchers take them.
    let descriptors = |i: usize| -> std::borrow::Cow<'_, [Vec<f32>]> {
        if byte_scale {
            std::borrow::Cow::Owned(unit_scale_descriptors(&desc_bytes[i]))
        } else {
            std::borrow::Cow::Borrowed(&features[i].descriptors)
        }
    };
    log(&format!(
        "sift: mean {} keypoints ({:.1} s)",
        features.iter().map(|f| f.keypoints.len()).sum::<usize>() / n.max(1),
        t0.elapsed().as_secs_f64()
    ));
    if masks.is_some() {
        let kept: usize = features.iter().map(|f| f.keypoints.len()).sum();
        let dropped = dropped.load(std::sync::atomic::Ordering::Relaxed);
        log(&format!(
            "masks: dropped {dropped} of {} keypoints ({:.1}%) on masked-out pixels",
            kept + dropped,
            100.0 * dropped as f64 / (kept + dropped).max(1) as f64
        ));
    }

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
            let extra = retrieval_pairs(
                &features,
                &descriptors,
                cfg.retrieval_k,
                cfg.window * per_source,
            )
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
        let bytes =
            |i: usize| -> u64 { (features[i].keypoints.len() * SIFT_DIM) as u64 * per_value };
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
    // An external matcher takes every pair once it has the descriptors (as
    // the CPU matcher takes them: VLFeat-compatible ones live in
    // `desc_bytes`, not in the features).
    let external = cfg.matcher.as_ref().and_then(|m| {
        let owned: Vec<_> = (0..n).map(descriptors).collect();
        let sets: Vec<&[Vec<f32>]> = owned.iter().map(|d| d.as_ref()).collect();
        match m.set_features(&sets) {
            Ok(()) => {
                log(&format!("matching with {m:?}"));
                Some(m.clone())
            }
            Err(e) => {
                log(&format!("matcher failed ({e}); matching on the CPU"));
                None
            }
        }
    });
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
            let owned: Vec<std::borrow::Cow<'_, [Vec<f32>]>> =
                imgs.par_iter().map(|&i| descriptors(i)).collect();
            let sets: Vec<&[Vec<f32>]> = owned.iter().map(|d| d.as_ref()).collect();
            let bank = if cfg.match_u8 {
                visloc_sift_gpu::FeatureBank::upload_u8(ctx, &sets)
            } else {
                visloc_sift_gpu::FeatureBank::upload(ctx, &sets)
            };
            bank.ok()
                .map(|bank| (ctx, bank, visloc_sift_gpu::GpuMatcher::new(ctx)))
        });
        let match_pairs = |pairs: &[(usize, usize)]| -> Vec<Vec<DescriptorMatch>> {
            if let Some(m) = &external {
                match m.match_pairs(pairs) {
                    Ok(dms) => return dms,
                    Err(e) => eprintln!("matcher failed ({e}); this chunk on the CPU"),
                }
            }
            #[cfg(feature = "gpu")]
            if let Some((ctx, bank, m)) = &gpu_match {
                let lp: Vec<(usize, usize)> =
                    pairs.iter().map(|&(i, j)| (local[i], local[j])).collect();
                return m.match_pairs(ctx, bank, &lp, Some(0.8), true);
            }
            // The descriptors of the chunk's images, as f32 for this chunk.
            let mut used: Vec<usize> = pairs.iter().flat_map(|&(i, j)| [i, j]).collect();
            used.sort_unstable();
            used.dedup();
            let sets: std::collections::HashMap<usize, std::borrow::Cow<'_, [Vec<f32>]>> =
                used.par_iter().map(|&i| (i, descriptors(i))).collect();
            pairs
                .par_iter()
                .map(|&(i, j)| cpu_matches(&sets[&i], &sets[&j]))
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

    // A cube or lens face's intrinsics are exact, a `--camera` calibrated.
    let refine_intrinsics =
        cfg.refine_intrinsics && !cfg.equirect && lenses.is_none() && cfg.camera.is_none();
    // Past matching only the intrinsics refinement reads the descriptors
    // (13 GB of f32 for SmallCity at 4,000 SIFT); without it, free them
    // before the mapper, with it give the features their f32 descriptors.
    let mut features = features;
    if !refine_intrinsics {
        for f in &mut features {
            f.descriptors = Vec::new();
        }
    } else if byte_scale {
        features
            .par_iter_mut()
            .zip(&desc_bytes)
            .for_each(|(f, b)| f.descriptors = unit_scale_descriptors(b));
    }
    drop(desc_bytes);

    // Mapper, then a global BA that also refines the intrinsics. The mapper
    // self-calibrates the focal length and principal point in its own
    // bundle adjustments (COLMAP's `ba_refine_focal_length` /
    // `ba_refine_principal_point`): on the 13 ETH3D DSLR scenes the
    // principal point of the centred prior is ~4 px off, which fixed
    // intrinsics turned into 2-5x larger camera-centre errors.
    let t0 = std::time::Instant::now();
    // A `--camera` to refine: the mapper's bundle adjustments refine its
    // distortion too, and their result is final (the global refinement
    // below holds a lens's distortion fixed).
    let mut port_overrides: Vec<String> = if cfg.refine_camera {
        vec![
            "ba_refine_focal_length=1".into(),
            "ba_refine_principal_point=1".into(),
            "ba_refine_extra_params=1".into(),
        ]
    } else if refine_intrinsics {
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
    // The converged final global BA on every model, not only from 1,000
    // frames: images registered from few tracks (structure-less, next to
    // views panned from one spot) stay a few tenths of a degree off after
    // the incremental loop's capped BAs. On ETH3D electro it takes three
    // such images from 0.17-0.27° to 0.02-0.05°.
    port_overrides.push("ba_global_final_converge_min_frames=0".into());
    // A dual-fisheye frame's ten faces are one rig in the front lens's
    // frame. The back lens's pose on it starts from a half turn with no
    // offset and is refined, its faces' with it: on an X5 it settles about
    // 1.1° from the half turn, 3.6 cm behind, while the front lens's faces
    // move by hundredths of a degree (docs/insta360_lens.md).
    if lenses.is_some() {
        port_overrides.push("ba_refine_sensor_from_rig=1".into());
    }
    // Face `k`'s `sensor_from_rig`: the panorama's or front lens's (rig)
    // frame to the face's.
    let rig: Option<Vec<nalgebra::Matrix3<f64>>> = if lenses.is_some() {
        Some(
            dual_fisheye::rig_face_rotations()
                .iter()
                .map(|r| r.transpose())
                .collect(),
        )
    } else {
        (cfg.equirect && cfg.equirect_rig && faces.is_none()).then(|| {
            equirect::cube_face_rotations()
                .iter()
                .map(|(_, r)| r.transpose())
                .collect()
        })
    };
    let models = run_colmap_port(
        out_dir,
        &camera,
        width,
        height,
        &features,
        &pairwise,
        rig.as_deref(),
        &port_overrides,
        log,
    )?;
    let (mut poses, mut tracks, mut reproj) = models.into_iter().next().expect("non-empty");
    let mut cam = camera.clone();
    if cfg.refine_camera {
        match read_port_camera(out_dir) {
            Some(c) if c.model == cam.model && c.params.len() == cam.params.len() => {
                log(&format!(
                    "mapper camera: {} {:?}",
                    c.model.colmap_name().unwrap_or("?"),
                    c.params
                ));
                cam.params = c.params;
            }
            _ => log("the mapper wrote no camera of the --camera model; keeping it"),
        }
    } else if refine_intrinsics {
        if let Some((fx, fy, cx, cy)) = read_port_camera(out_dir).and_then(|c| c.intrinsics()) {
            cam.set_intrinsics(fx, fy, cx, cy);
            log(&format!(
                "mapper intrinsics: fx {fx:.1} fy {fy:.1} cx {cx:.1} cy {cy:.1} px"
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
        // Start the Levenberg–Marquardt damping at 1e4 instead of 1e-4. The
        // damping is absolute and a photo set's Hessian diagonals are
        // 1e4–1e8, so 1e-4 is no damping: the first steps swing cameras by up
        // to a radian and are rejected until λ reaches the Hessian's scale.
        // On three 500-image SmallCity subsets this keeps a segment the
        // mapper left bent from staying 1–3 m off (max error 282 → 23 cm on
        // one, within about a centimetre on the other two).
        sfm_cfg.ba_config.initial_lambda = Some(1e4);
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
    // The training cameras' focal length (the cube faces' with
    // `equirect_native`).
    let focal = match &faces {
        Some((_, size, _)) => *size as f64 / 2.0,
        None => cam.params[0],
    };
    log(&format!(
        "sfm: {registered}/{n} registered, {} points, {reproj:.3} px, focal {focal:.1} px ({:.1} s)",
        tracks.len(),
        t0.elapsed().as_secs_f64()
    ));
    if cfg.equirect && faces.is_none() {
        log(&face_centre_spread(&poses, per_source));
    }
    // Metric scale from an IMU log, applied before anything is built from
    // the model (points, cube-face poses, training views, sparse/0).
    let mut metric_scale = None;
    if let Some(imu) = cfg.imu.as_ref().or(cfg.insv.as_ref()) {
        // The faces' rotations (face frame to source frame).
        let face_rotations: Vec<Matrix3<f64>> = match per_source {
            10 => dual_fisheye::rig_face_rotations().to_vec(),
            6 => equirect::cube_face_rotations()
                .iter()
                .map(|(_, r)| *r)
                .collect(),
            _ => Vec::new(),
        };
        let scale = imu_metric_scale(imu, &files, &poses, &face_rotations, cfg, out_dir, log)?;
        for pose in poses.iter_mut().flatten() {
            pose.world_to_camera.translation *= scale;
        }
        for track in &mut tracks {
            track.position = Point3::from(track.position.coords * scale);
        }
        metric_scale = Some(scale);
    }
    if per_source == 10 {
        write_back_lens(&poses, metric_scale.is_some(), out_dir, log)?;
    }

    let points: Vec<ColoredPoint> = tracks
        .iter()
        .filter_map(|t| {
            let &(img, kp, _) = t.observations.first()?;
            let rgb = keypoint_rgb[img][kp];
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
    // SfM points seen per cube face (image `6 i + k`), with a rig or
    // `equirect_native`, where every face of a registered panorama has a
    // pose whether or not anything in it was matched.
    let face_support: Option<Vec<usize>> = if faces.is_some() {
        let axes = equirect::face_axes();
        let mut counts = vec![0usize; 6 * n];
        for &(pano, _, px) in tracks.iter().flat_map(|t| &t.observations) {
            if let Some(ray) = camera.unit_ray_from_pixel(&px) {
                counts[6 * pano + equirect::nearest_face(&axes, &ray)] += 1;
            }
        }
        Some(counts)
    } else if rig.is_some() {
        let mut counts = vec![0usize; n];
        for &(image, _, _) in tracks.iter().flat_map(|t| &t.observations) {
            counts[image] += 1;
        }
        Some(counts)
    } else {
        None
    };
    // With `equirect_native` the tracks observe the panoramas, while the
    // model written below has their cube faces, this many pixels square.
    let face_size = faces.as_ref().map(|(_, size, _)| *size as f64);
    // Each registered panorama -> its six cube faces, posed by the face's
    // fixed rotation behind the panorama's pose (face `k` of panorama `i`
    // is image `6 i + k`, as written above).
    let (poses, names, cam, width, height, per_source) = match faces {
        Some((face_names, size, face_camera)) => {
            let rotations = equirect::cube_face_rotations();
            let face_poses: Vec<Option<visloc_core::geometry::Pose>> = poses
                .iter()
                .flat_map(|pose| {
                    rotations
                        .iter()
                        .map(move |(_, r)| pose.as_ref().map(|pose| equirect::face_pose(pose, r)))
                })
                .collect();
            (face_poses, face_names, face_camera, size, size, 6)
        }
        None => (poses, names, cam, width, height, per_source),
    };
    let source = |i: usize| i / per_source;
    // Faces that see fewer than `min_matches` SfM points (sky, or a camera
    // mount or selfie stick that moves with the camera) are left out of
    // training.
    let mut poses = poses;
    if let Some(counts) = face_support {
        let posed = poses.iter().filter(|p| p.is_some()).count();
        for (pose, &count) in poses.iter_mut().zip(&counts) {
            if count < cfg.min_matches {
                *pose = None;
            }
        }
        let kept = poses.iter().filter(|p| p.is_some()).count();
        log(&format!(
            "equirect: {kept} of {posed} posed faces see at least {} SfM points and are kept \
             for training",
            cfg.min_matches
        ));
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
    // COLMAP text model next to the images, so other tools (brush, Inria
    // 3DGS, gsplat_train / gsplat_mesh --data) can use the reconstruction.
    let sparse = out_dir.join("sparse").join("0");
    std::fs::create_dir_all(&sparse).map_err(io(&sparse))?;
    let (model, params) = visloc_io::colmap::colmap_camera_record(&cam);
    let cameras = params.iter().fold(
        format!(
            "# Camera list\n1 {} {width} {height}",
            model.colmap_name().unwrap_or("PINHOLE")
        ),
        |row, p| format!("{row} {p}"),
    ) + "\n";
    // Image ids follow the registered images in order; point ids are the
    // track indices plus one. Each image lists its observations (x, y,
    // point id) and each point its track (image id, observation index), as
    // COLMAP writes them, so tools that read the observations (visloc-mvs's
    // source views and depth ranges) can use the model.
    let mut image_ids = vec![0usize; poses.len()];
    let mut observations: Vec<Vec<(f64, f64, usize)>> = vec![Vec::new(); poses.len()];
    for (id, (i, _)) in poses
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|p| (i, p)))
        .enumerate()
    {
        image_ids[i] = id + 1;
    }
    // An observation of a panorama (`equirect_native`) goes to the face
    // nearest its ray, at the pixel where the ray crosses the face.
    let face_axes = equirect::face_axes();
    let face_rotations = equirect::cube_face_rotations();
    let model_observation = |img: usize, px: &Point2<f64>| -> Option<(usize, f64, f64)> {
        let Some(size) = face_size else {
            return Some((img, px.x, px.y));
        };
        let ray = camera.unit_ray_from_pixel(px)?;
        let k = equirect::nearest_face(&face_axes, &ray);
        let d = face_rotations[k].1.transpose() * ray;
        let half = size / 2.0;
        Some((
            6 * img + k,
            half * d.x / d.z + half,
            half * d.y / d.z + half,
        ))
    };
    let mut point_tracks: Vec<Vec<(usize, usize)>> = vec![Vec::new(); tracks.len()];
    for (k, t) in tracks.iter().enumerate() {
        for &(img, _, px) in &t.observations {
            let Some((img, x, y)) = model_observation(img, &px) else {
                continue;
            };
            if image_ids[img] == 0 {
                continue;
            }
            point_tracks[k].push((image_ids[img], observations[img].len()));
            observations[img].push((x, y, k + 1));
        }
    }
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
            "{} {} {} {} {} {} {} {} 1 {}\n",
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
        let line: Vec<String> = observations[i]
            .iter()
            .map(|&(x, y, point)| format!("{x} {y} {point}"))
            .collect();
        images_txt.push_str(&line.join(" "));
        images_txt.push('\n');
    }
    let points_txt: String = std::iter::once("# 3D point list\n".to_string())
        .chain(tracks.iter().enumerate().filter_map(|(k, t)| {
            let &(img, kp, _) = t.observations.first()?;
            let rgb = keypoint_rgb[img][kp];
            let track: Vec<String> = point_tracks[k]
                .iter()
                .map(|&(image, index)| format!("{image} {index}"))
                .collect();
            Some(format!(
                "{} {} {} {} {} {} {} 0 {}\n",
                k + 1,
                t.position.x,
                t.position.y,
                t.position.z,
                rgb[0],
                rgb[1],
                rgb[2],
                track.join(" ")
            ))
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
        metric_scale,
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

/// SIFT descriptor length.
const SIFT_DIM: usize = 128;

/// VLFeat-compatible SIFT descriptors come back already quantised like
/// COLMAP's bytes (`round(512 x)`, norm about 512), so they are kept as
/// bytes, a quarter of the memory of f32. Matching expects the unit-norm
/// scale: the GPU u8 bank quantises with `round(512 x)` again, which
/// saturated every non-zero value at 255 and left near-binary descriptors
/// (SmallCity lost 70% of its 3D points). `byte / 512` is exact in f32, so
/// the bank recovers the same bytes, and the CPU matcher's comparisons are
/// unchanged. `bytes` holds the descriptors one after another.
fn unit_scale_descriptors(bytes: &[u8]) -> Vec<Vec<f32>> {
    bytes
        .chunks_exact(SIFT_DIM)
        .map(|d| d.iter().map(|&b| b as f32 / 512.0).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::exif_focal_35mm;
    use super::resize_triangle;
    use super::unit_scale_descriptors;
    use std::path::{Path, PathBuf};

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
        let bytes: Vec<u8> = (0..256).map(|i| ((i * 37) % 256) as u8).collect();
        let unit = unit_scale_descriptors(&bytes);
        assert_eq!(unit.len(), 2);
        for (b, u) in bytes.iter().zip(unit.iter().flatten()) {
            assert_eq!(*u, *b as f32 / 512.0);
            assert_eq!((u * 512.0).round().clamp(0.0, 255.0), *b as f32);
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

    /// The colour of the noise-textured box room (x, z in [-4, 4] m, y in
    /// [-2.5, 2.5] m) along the ray from `centre` in the direction `d`.
    fn room_colour(centre: nalgebra::Vector3<f64>, d: nalgebra::Vector3<f64>) -> image::Rgb<u8> {
        let half = nalgebra::Vector3::new(4.0f64, 2.5, 4.0);
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
    }

    /// A `w x w/2` panorama taken at `centre` in the room, turned by `yaw`
    /// about the vertical.
    fn room_panorama(w: u32, centre: nalgebra::Vector3<f64>, yaw: f64) -> image::RgbImage {
        use std::f64::consts::PI;
        let rot = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw);
        image::RgbImage::from_fn(w, w / 2, |x, y| {
            let lon = (x as f64 + 0.5) / w as f64 * 2.0 * PI - PI;
            let lat = PI / 2.0 - (y as f64 + 0.5) / (w / 2) as f64 * PI;
            let d = rot
                * nalgebra::Vector3::new(lat.cos() * lon.sin(), -lat.sin(), lat.cos() * lon.cos());
            room_colour(centre, d)
        })
    }

    /// A `w x h` pinhole photo of the room (focal length `focal` px, the
    /// principal point at the centre) taken at `centre`, turned by `yaw`.
    fn room_photo(
        w: u32,
        h: u32,
        focal: f64,
        centre: nalgebra::Vector3<f64>,
        yaw: f64,
    ) -> image::RgbImage {
        let rot = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw);
        image::RgbImage::from_fn(w, h, |x, y| {
            let ray = nalgebra::Vector3::new(
                (x as f64 + 0.5 - w as f64 / 2.0) / focal,
                (y as f64 + 0.5 - h as f64 / 2.0) / focal,
                1.0,
            );
            room_colour(centre, rot * ray.normalize())
        })
    }

    /// Five panoramas of the synthetic room in `root/in`, and the `(centre,
    /// yaw)` each was taken at.
    fn room_panoramas(root: &Path) -> (PathBuf, Vec<(nalgebra::Vector3<f64>, f64)>) {
        let images = root.join("in");
        std::fs::create_dir_all(&images).unwrap();
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
        (images, truth)
    }

    /// The cube-face poses of `sparse/0/images.txt` under `out` against
    /// [`room_panoramas`]' truth, free of the gauge: how many there are, the
    /// worst error of a relative rotation between two faces (degrees) and
    /// of a distance between two panoramas' mean face centres (metres,
    /// after the least-squares scale).
    fn face_pose_errors(out: &Path, truth: &[(nalgebra::Vector3<f64>, f64)]) -> (usize, f64, f64) {
        // Face k of panorama i looks along yaw_i * R_k (camera to world).
        let faces = crate::equirect::cube_face_rotations();
        let true_rotation = |i: usize| {
            let (_, yaw) = truth[i / 6];
            nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw).matrix()
                * faces[i % 6].1
        };
        // Recovered world-to-camera rotations and centres, by image index
        // (each image's line; its observations' line follows).
        let text = std::fs::read_to_string(out.join("sparse/0/images.txt")).unwrap();
        let mut est: Vec<(usize, nalgebra::Matrix3<f64>, nalgebra::Vector3<f64>)> = Vec::new();
        for line in text.lines().filter(|l| !l.starts_with('#')).step_by(2) {
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
        let panoramas: Vec<usize> = (0..truth.len())
            .filter(|&p| est.iter().any(|e| e.0 / 6 == p))
            .collect();
        let (mut dt, mut de) = (Vec::new(), Vec::new());
        for (k, &p) in panoramas.iter().enumerate() {
            for &q in &panoramas[k + 1..] {
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
            "vs truth: {} faces, worst relative rotation {worst_deg:.3} deg, worst panorama \
             distance error {:.1} mm (spacing 0.58 m)",
            est.len(),
            1000.0 * worst_m
        );
        (est.len(), worst_deg, worst_m)
    }

    /// The observations of the model in `sparse/0` under `out` (a PINHOLE
    /// camera): how many there are, and their median reprojection error
    /// (pixels) through the written poses and points.
    fn model_reprojection(out: &Path) -> (usize, f64) {
        let read = |name: &str| std::fs::read_to_string(out.join("sparse/0").join(name)).unwrap();
        let numbers = |line: &str| -> Vec<f64> {
            line.split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect()
        };
        let cameras = read("cameras.txt");
        let camera = cameras.lines().find(|l| !l.starts_with('#')).unwrap();
        let c = numbers(camera);
        let (fx, fy, cx, cy) = (c[3], c[4], c[5], c[6]);
        let points: std::collections::HashMap<usize, nalgebra::Point3<f64>> = read("points3D.txt")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| {
                let v = numbers(l);
                (v[0] as usize, nalgebra::Point3::new(v[1], v[2], v[3]))
            })
            .collect();
        let images = read("images.txt");
        let lines: Vec<&str> = images.lines().filter(|l| !l.starts_with('#')).collect();
        let mut errors = Vec::new();
        for pair in lines.chunks(2) {
            let v = numbers(pair[0]);
            let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                v[1], v[2], v[3], v[4],
            ));
            let t = nalgebra::Vector3::new(v[5], v[6], v[7]);
            for o in numbers(pair[1]).chunks(3) {
                let p = q * points[&(o[2] as usize)].coords + t;
                errors.push((fx * p.x / p.z + cx - o[0]).hypot(fy * p.y / p.z + cy - o[1]));
            }
        }
        errors.sort_by(f64::total_cmp);
        (errors.len(), errors[errors.len() / 2])
    }

    /// Panoramas of a synthetic room through the whole CPU pipeline: the
    /// pinhole path warns, the cube-face path registers the faces as one
    /// rig per panorama.
    /// `cargo test --release -p visloc-gsplat-train --features euroc --lib
    /// equirect_end_to_end -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn equirect_end_to_end() {
        let root = std::env::temp_dir().join(format!("visloc_equirect_{}", std::process::id()));
        let (images, truth) = room_panoramas(&root);
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
        // One rig per panorama: its faces share a centre exactly.
        assert!(logs.iter().any(|m| m.contains("lie a median 0.0000 from")));
        assert_eq!((report.images, report.width, report.height), (30, 256, 256));
        assert_eq!(report.focal, 128.0);
        assert!(report.registered >= 24, "{} registered", report.registered);
        // Every face of the room sees SfM points: all are kept for training.
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

        let (faces, worst_deg, worst_m) = face_pose_errors(&root.join("faces"), &truth);
        let (observations, median_px) = model_reprojection(&root.join("faces"));
        println!("{observations} observations, median reprojection error {median_px:.3} px");
        assert!(observations > 10_000, "{observations} observations");
        assert!(median_px < 0.25, "{median_px} px");
        assert_eq!(faces, report.registered);
        assert!(worst_deg < 0.05, "{worst_deg} deg");
        assert!(worst_m < 0.005, "{worst_m} m");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The same room with `equirect_native`: SfM on the panoramas as
    /// equirectangular cameras registers all five, and the cube faces posed
    /// from them match the truth.
    /// `cargo test --release -p visloc-gsplat-train --features euroc --lib
    /// equirect_native_end_to_end -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn equirect_native_end_to_end() {
        let root = std::env::temp_dir().join(format!("visloc_native_{}", std::process::id()));
        let (images, truth) = room_panoramas(&root);
        let cfg = super::PhotoSfmConfig {
            gpu: false,
            gpu_sift: false,
            eval_every: 0,
            equirect: true,
            equirect_native: true,
            ..Default::default()
        };
        let mut logs = Vec::new();
        let (dataset, points, report) =
            super::build_photo_dataset(&images, &root.join("native"), &cfg, &mut |m: &str| {
                println!("{m}");
                logs.push(m.to_string())
            })
            .unwrap();
        assert!(logs
            .iter()
            .any(|m| m.contains("SfM on the 5 panoramas (1024x512")));
        assert_eq!((report.images, report.registered), (5, 5));
        assert_eq!(report.focal, 128.0);
        // Training sees the cube faces, posed from their panoramas.
        assert_eq!(dataset.train.len(), 30);
        assert!(points.len() > 1000, "{} points", points.len());
        let (faces, worst_deg, worst_m) = face_pose_errors(&root.join("native"), &truth);
        let (observations, median_px) = model_reprojection(&root.join("native"));
        println!("{observations} observations, median reprojection error {median_px:.3} px");
        assert!(observations > 10_000, "{observations} observations");
        assert!(median_px < 0.25, "{median_px} px");
        assert_eq!(faces, 30);
        assert!(worst_deg < 0.05, "{worst_deg} deg");
        assert!(worst_m < 0.005, "{worst_m} m");
        let _ = std::fs::remove_dir_all(&root);
    }

    const PHOTO_W: u32 = 480;
    const PHOTO_H: u32 = 360;

    /// Six photos of the synthetic room in `root/in` (`view_0.png` ...), a
    /// sideways walk with a small turn.
    fn room_photos(root: &Path) -> PathBuf {
        let images = root.join("in");
        std::fs::create_dir_all(&images).unwrap();
        for k in 0..6 {
            let centre = nalgebra::Vector3::new(-1.0 + 0.4 * k as f64, 0.05 * (k % 2) as f64, 0.0);
            room_photo(PHOTO_W, PHOTO_H, 420.0, centre, 0.04 * k as f64)
                .save(images.join(format!("view_{k}.png")))
                .unwrap();
        }
        images
    }

    /// `edges.len()` masks `view_{k}.png` in `dir`, `mw x mh`, each with its
    /// own edge: 255 left of column `edges[k]` and 0 from it on (`invert`:
    /// the other way round). Distinct edges, so a mask paired with the wrong
    /// image shows.
    fn edge_masks(dir: &Path, mw: u32, mh: u32, edges: &[u32], invert: bool) {
        std::fs::create_dir_all(dir).unwrap();
        for (k, &edge) in edges.iter().enumerate() {
            image::GrayImage::from_fn(mw, mh, |x, _| {
                image::Luma([if (x < edge) != invert { 255 } else { 0 }])
            })
            .save(dir.join(format!("view_{k}.png")))
            .unwrap();
        }
    }

    /// Image `i`'s keypoints as the matcher and mapper got them
    /// (`out/port/features`, in extraction order), if it took part in a
    /// verified pair.
    fn keypoints_of(out: &Path, i: usize) -> Option<Vec<(f64, f64)>> {
        let text =
            std::fs::read_to_string(out.join(format!("port/features/frame_{i:05}_features.txt")))
                .ok()?;
        Some(
            text.lines()
                .map(|l| {
                    let mut v = l.split_whitespace().map(|x| x.parse().unwrap());
                    (v.next().unwrap(), v.next().unwrap())
                })
                .collect(),
        )
    }

    /// The pixel column a keypoint at `x` lies on, by the detector's
    /// convention (see `drop_masked_keypoints`), written out independently.
    fn pixel_of(x: f64, half_pixel: bool) -> i64 {
        if half_pixel {
            x.floor() as i64
        } else {
            (x + 0.5).floor() as i64
        }
    }

    /// The mean keypoint count per image from a run's log, and the count the
    /// masks dropped (0 without masks).
    fn keypoint_counts(logs: &[String]) -> (usize, usize) {
        let mean = logs
            .iter()
            .find_map(|m| m.strip_prefix("sift: mean "))
            .map(|m| m.split_whitespace().next().unwrap().parse().unwrap())
            .unwrap();
        let dropped = logs
            .iter()
            .find_map(|m| m.strip_prefix("masks: dropped "))
            .map_or(0, |m| m.split_whitespace().next().unwrap().parse().unwrap());
        (mean, dropped)
    }

    fn run_photos(
        images: &Path,
        out: &Path,
        cfg: &super::PhotoSfmConfig,
    ) -> (crate::dataset::Dataset, super::PhotoSfmReport, Vec<String>) {
        let mut logs = Vec::new();
        let (dataset, _, report) = super::build_photo_dataset(images, out, cfg, &mut |m: &str| {
            println!("{m}");
            logs.push(m.to_string())
        })
        .unwrap();
        (dataset, report, logs)
    }

    /// The working image `k` of a run, as RGBA.
    fn working_image(out: &Path, k: usize) -> image::RgbaImage {
        image::open(out.join(format!("images/frame_{k:05}.png")))
            .unwrap()
            .into_rgba8()
    }

    /// Photos of the room through SfM with and without a mask over the right
    /// of each, each view with its own edge (masks twice the photos' size,
    /// and `max_size` scaling the photos down, so the nearest-neighbour
    /// resize is in play). Against the run without masks: each view's
    /// keypoints are exactly the unmasked run's that lie left of its own
    /// edge, by the detector's pixel convention `half_pixel`, and there are
    /// some within half a pixel of an edge, so the other convention would
    /// give other keypoints. The training images keep their pixels under the
    /// mask and take alpha 0 from each view's own edge. `sift` adjusts the
    /// SIFT path.
    fn masked_sfm_keeps_keypoints_off_the_mask(
        tag: &str,
        sift: impl Fn(&mut super::PhotoSfmConfig),
        min_registered: usize,
        half_pixel: bool,
    ) {
        let root = std::env::temp_dir().join(format!("visloc_masks_{tag}_{}", std::process::id()));
        let images = room_photos(&root);
        // Edges in the 240 x 180 working images; the masks are 960 x 720.
        let edges: Vec<u32> = (0..6).map(|k| 96 + 12 * k).collect();
        let native: Vec<u32> = edges.iter().map(|e| 4 * e).collect();
        edge_masks(&root.join("masks"), 960, 720, &native, false);
        let mut cfg = super::PhotoSfmConfig {
            max_size: 240,
            focal_px: Some(420.0 * 240.0 / 480.0),
            eval_every: 0,
            ..Default::default()
        };
        sift(&mut cfg);

        let (_, control, control_logs) = run_photos(&images, &root.join("control"), &cfg);
        assert!(control.registered >= min_registered, "{control:?}");
        let control_kps: Vec<Vec<(f64, f64)>> = (0..6)
            .map(|k| keypoints_of(&root.join("control"), k).expect("control keypoints"))
            .collect();
        let right_of_edge: usize = control_kps
            .iter()
            .zip(&edges)
            .map(|(kps, &e)| {
                kps.iter()
                    .filter(|p| pixel_of(p.0, half_pixel) >= e as i64)
                    .count()
            })
            .sum();
        // Keypoints whose pixel the two conventions disagree on, at an edge.
        let at_edge: usize = control_kps
            .iter()
            .zip(&edges)
            .map(|(kps, &e)| {
                kps.iter()
                    .filter(|p| pixel_of(p.0, true) < e as i64 && pixel_of(p.0, false) >= e as i64)
                    .count()
            })
            .sum();
        assert!(
            right_of_edge > 500 && at_edge > 0,
            "{right_of_edge} {at_edge}"
        );

        cfg.mask_dir = Some(root.join("masks"));
        let (dataset, masked, logs) = run_photos(&images, &root.join("masked"), &cfg);
        assert!(masked.registered >= min_registered, "{masked:?}");
        let mut kept = 0;
        for (k, &e) in edges.iter().enumerate() {
            let want: Vec<(f64, f64)> = control_kps[k]
                .iter()
                .copied()
                .filter(|p| pixel_of(p.0, half_pixel) < e as i64)
                .collect();
            let got = keypoints_of(&root.join("masked"), k).expect("masked keypoints");
            assert!(
                got == want,
                "view {k}: {} keypoints, want {}",
                got.len(),
                want.len()
            );
            kept += got.len();
        }
        // The counts agree with the log.
        let (mean_without, _) = keypoint_counts(&control_logs);
        let (mean_with, dropped) = keypoint_counts(&logs);
        assert_eq!(dropped, right_of_edge);
        assert!(
            mean_with < mean_without && kept > 500,
            "{mean_with} {mean_without}"
        );

        // Training images: the photo's pixels, alpha 0 from the view's edge.
        for (k, &e) in edges.iter().enumerate() {
            let source = image::open(root.join(format!("in/view_{k}.png")))
                .unwrap()
                .into_rgb8();
            let scaled = super::resize_triangle(&source, 240, 180);
            let written = working_image(&root.join("masked"), k);
            for y in 0..180 {
                for x in 0..240 {
                    let p = written.get_pixel(x, y);
                    assert_eq!(
                        p[3],
                        if x < e { 255 } else { 0 },
                        "view {k}: alpha at {x},{y}"
                    );
                    assert_eq!(
                        p.0[..3],
                        scaled.get_pixel(x, y).0,
                        "view {k}: {x},{y} altered"
                    );
                }
            }
        }
        // And the trainer reads that alpha as its mask.
        let (_, valid) = crate::dataset::load_view_rgb_mask(&dataset.train[0]).unwrap();
        let valid = valid.expect("the view has a mask");
        assert!(!valid[100 * 240 + 239] && valid[100 * 240 + 50]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn masked_sfm_cpu() {
        masked_sfm_keeps_keypoints_off_the_mask(
            "cpu",
            |c| {
                c.gpu = false;
                c.gpu_sift = false;
            },
            5,
            true,
        );
    }

    /// The default detector on the CPU: pixel `i` is centred on `x = i`.
    #[test]
    fn masked_sfm_cpu_plain_detector() {
        masked_sfm_keeps_keypoints_off_the_mask(
            "cpu_plain",
            |c| {
                c.gpu = false;
                c.gpu_sift = false;
                plain_detector(c);
            },
            3,
            false,
        );
    }

    /// SIFT settings for the default detector, with the contrast threshold
    /// lowered for the smooth synthetic photos.
    fn plain_detector(c: &mut super::PhotoSfmConfig) {
        c.sift_overrides = vec![
            "vlfeat_compatible_detector=0".into(),
            "vlfeat_compatible_descriptor=0".into(),
            "contrast_threshold=0.004".into(),
        ];
    }

    /// The CPU matcher behind `PhotoSfmConfig::matcher`, so a test of the GPU
    /// extractors does not also depend on the GPU matcher.
    #[cfg(feature = "gpu")]
    #[derive(Debug, Default)]
    struct CpuPairs(std::sync::Mutex<Vec<Vec<Vec<f32>>>>);

    #[cfg(feature = "gpu")]
    impl super::PairMatcher for CpuPairs {
        fn set_features(&self, descriptors: &[&[Vec<f32>]]) -> Result<(), String> {
            *self.0.lock().unwrap() = descriptors.iter().map(|d| d.to_vec()).collect();
            Ok(())
        }
        fn match_pairs(
            &self,
            pairs: &[(usize, usize)],
        ) -> Result<Vec<Vec<visloc_vision::matching::DescriptorMatch>>, String> {
            let d = self.0.lock().unwrap();
            Ok(pairs
                .iter()
                .map(|&(i, j)| crate::euroc::cpu_matches(&d[i], &d[j]))
                .collect())
        }
    }

    #[cfg(feature = "gpu")]
    fn cpu_pairs(c: &mut super::PhotoSfmConfig) {
        c.matcher = Some(std::sync::Arc::new(CpuPairs::default()));
    }

    #[cfg(feature = "gpu")]
    fn have_gpu() -> bool {
        let ok = visloc_sift_gpu::GpuContext::new().is_ok();
        if !ok {
            eprintln!("no GPU adapter; skipping");
        }
        ok
    }

    /// The VLFeat-compatible extractor on the GPU (the default).
    #[cfg(feature = "gpu")]
    #[test]
    fn masked_sfm_gpu_vlfeat() {
        if have_gpu() {
            masked_sfm_keeps_keypoints_off_the_mask("gpu_vlfeat", cpu_pairs, 5, true);
        }
    }

    /// The GPU extractor of the default detector (pixel `i` centred on
    /// `x = i`).
    #[cfg(feature = "gpu")]
    #[test]
    fn masked_sfm_gpu_plain() {
        if have_gpu() {
            masked_sfm_keeps_keypoints_off_the_mask(
                "gpu_plain",
                |c| {
                    cpu_pairs(c);
                    plain_detector(c);
                },
                3,
                false,
            );
        }
    }

    /// `extract_chunked`, the plain GPU extractor's loop, over 260 images in
    /// chunks of 256, without extracting anything real: image `g` has 100
    /// keypoints on `x = 0.5 .. 99.5` and a mask from column `edges[g]`, so
    /// it must keep exactly `edges[g]` of them. Image `g` is the chunk's
    /// image `g % 256`, so a mask looked up by the chunk's index, or by the
    /// chunk's first image, keeps another number.
    #[cfg(feature = "gpu")]
    #[test]
    fn chunked_extraction_masks_each_image_by_its_global_index() {
        const N: usize = 260;
        let edges: Vec<usize> = (0..N).map(|g| 10 + (37 * g) % 80).collect();
        let load_mask = |g: usize| -> Result<Option<Vec<bool>>, super::EurocError> {
            Ok(Some((0..100).map(|x| x >= edges[g]).collect()))
        };
        let features_of = |kps: Vec<_>, desc: Vec<_>, masked: Option<&[bool]>| {
            let (kps, desc) =
                crate::masks::drop_masked_keypoints(kps, desc, masked.unwrap(), 100, true);
            (
                super::FeatureSet {
                    keypoints: kps
                        .iter()
                        .map(|k| nalgebra::Point2::new(k.x, k.y))
                        .collect(),
                    descriptors: desc,
                },
                Vec::new(),
            )
        };
        let mut chunk_sizes = Vec::new();
        let got = super::extract_chunked(
            N,
            256,
            |start, count| Ok((start, count)),
            |&(_, _), _| {
                let kps = (0..100)
                    .map(|x| {
                        visloc_vision::features::sift::SiftKeypoint::from_location_scale_orientation(
                            x as f64 + 0.5,
                            0.5,
                            1.0,
                            0.0,
                        )
                    })
                    .collect();
                Ok((kps, vec![Vec::new(); 100]))
            },
            &features_of,
            &load_mask,
            |&(start, count), features| {
                assert_eq!(features.len(), count);
                chunk_sizes.push((start, count));
            },
        )
        .unwrap();
        assert_eq!(chunk_sizes, [(0, 256), (256, 4)]);
        assert_eq!(got.len(), N);
        for (g, (f, &e)) in got.iter().zip(&edges).enumerate() {
            let xs: Vec<f64> = f.0.keypoints.iter().map(|p| p.x).collect();
            assert_eq!(xs.len(), e, "image {g}");
            assert!(xs.iter().all(|&x| x < e as f64), "image {g}");
        }
        // Every image has a different edge from the one of its chunk's first
        // image, and from its chunk index (g % 256) neighbours.
        assert!((256..N).all(|g| edges[g] != edges[0] && edges[g] != edges[g - 256]));
    }

    /// `extract_batched`, the VLFeat GPU extractor's loop, over 260 images in
    /// chunks of 256, with a fake batch extractor: image `g` has 100
    /// keypoints on `x = 0.5 .. 99.5` and a mask from column `edges[g]`, so
    /// it must keep exactly `edges[g]` of them, whether the batch comes from
    /// the device or is refused (`Unsupported`) and the CPU extracts each
    /// image. A mask looked up by the chunk's index, or with the chunk's
    /// first image, keeps another number.
    #[cfg(feature = "gpu")]
    #[test]
    fn batched_extraction_masks_each_image_by_its_global_index() {
        use visloc_vision::features::sift::SiftKeypoint;
        const N: usize = 260;
        let edges: Vec<usize> = (0..N).map(|g| 10 + (37 * g) % 80).collect();
        let load_mask = |g: usize| -> Result<Option<Vec<bool>>, super::EurocError> {
            Ok(Some((0..100).map(|x| x >= edges[g]).collect()))
        };
        let features_of = |kps: Vec<_>, desc: Vec<_>, masked: Option<&[bool]>| {
            let (kps, desc) =
                crate::masks::drop_masked_keypoints(kps, desc, masked.unwrap(), 100, true);
            (
                super::FeatureSet {
                    keypoints: kps
                        .iter()
                        .map(|k| nalgebra::Point2::new(k.x, k.y))
                        .collect(),
                    descriptors: desc,
                },
                Vec::new(),
            )
        };
        let one_image = || {
            let kps: Vec<SiftKeypoint> = (0..100)
                .map(|x| {
                    SiftKeypoint::from_location_scale_orientation(x as f64 + 0.5, 0.5, 1.0, 0.0)
                })
                .collect();
            (kps, vec![Vec::new(); 100])
        };
        for refused in [false, true] {
            let mut chunks = Vec::new();
            let got = super::extract_batched(
                N,
                256,
                |start, count| Ok((start, count)),
                |&(_, count)| {
                    Ok(if refused {
                        Err(visloc_sift_gpu::SiftGpuError::Unsupported)
                    } else {
                        Ok((0..count).map(|_| one_image()).collect())
                    })
                },
                |_, _| Ok(one_image()),
                &features_of,
                &load_mask,
                |&(start, count), features| {
                    assert_eq!(features.len(), count);
                    chunks.push(start);
                },
            )
            .unwrap();
            assert_eq!(chunks, [0, 256]);
            assert_eq!(got.len(), N);
            for (g, (f, &e)) in got.iter().zip(&edges).enumerate() {
                let xs: Vec<f64> = f.0.keypoints.iter().map(|p| p.x).collect();
                assert_eq!(xs.len(), e, "refused {refused}: image {g}");
                assert!(xs.iter().all(|&x| x < e as f64), "image {g}");
            }
        }
        assert!((256..N).all(|g| edges[g] != edges[0] && edges[g] != edges[g - 256]));
    }

    /// 260 small photos of the room (more than one 256-image extraction
    /// chunk) in `root/in`, each a step further along a sideways walk, and
    /// for each its own mask `view_k.png` in `root/masks`, `edges[k]`
    /// columns wide. SfM then runs on them with a window of neighbours, and
    /// every view's keypoints must lie left of its own edge, and reach
    /// close to it. A mask paired with the wrong image, or an image
    /// processed with the mask of its chunk's first, would put keypoints
    /// past the edge or leave a gap before it.
    ///
    /// The `masks_pair_with_their_images_*` tests that call this are ignored
    /// by default: SfM on 260 photos takes about ten seconds in a release
    /// build and far longer in a debug one. The chunk indexing itself is
    /// covered by default in `chunk_features_use_each_images_own_mask_past_the_first_chunk`;
    /// these add the call sites in `build_photo_dataset`. Run them with
    /// `cargo test --release -p visloc-gsplat-train --features gpu,euroc --lib
    /// masks_pair_with_their_images -- --ignored`.
    fn masks_pair_with_their_images_across_chunks(
        tag: &str,
        sift: impl Fn(&mut super::PhotoSfmConfig),
        half_pixel: bool,
    ) {
        const N: usize = 260;
        let root = std::env::temp_dir().join(format!("visloc_masks_{tag}_{}", std::process::id()));
        let images = root.join("in");
        std::fs::create_dir_all(&images).unwrap();
        for k in 0..N {
            let centre = nalgebra::Vector3::new(-3.2 + 0.025 * k as f64, 0.0, 0.0);
            room_photo(160, 120, 140.0, centre, 0.0005 * k as f64)
                .save(images.join(format!("view_{k:03}.png")))
                .unwrap();
        }
        // Edges 30..=129 of 160 columns, every neighbour's different.
        let edges: Vec<u32> = (0..N as u32).map(|k| 30 + (37 * k) % 100).collect();
        let masks = root.join("masks");
        std::fs::create_dir_all(&masks).unwrap();
        for (k, &e) in edges.iter().enumerate() {
            image::GrayImage::from_fn(160, 120, |x, _| image::Luma([if x < e { 255 } else { 0 }]))
                .save(masks.join(format!("view_{k:03}.png")))
                .unwrap();
        }
        let mut cfg = super::PhotoSfmConfig {
            focal_px: Some(140.0),
            eval_every: 0,
            exhaustive_max: 0,
            window: 8,
            mask_dir: Some(masks),
            ..Default::default()
        };
        sift(&mut cfg);
        let out = root.join("out");
        let _ = run_photos(&images, &out, &cfg);
        let mut present = Vec::new();
        for (k, &e) in edges.iter().enumerate() {
            let Some(kps) = keypoints_of(&out, k) else {
                continue;
            };
            present.push(k);
            let columns: Vec<i64> = kps.iter().map(|p| pixel_of(p.0, half_pixel)).collect();
            let furthest = *columns.iter().max().expect("keypoints");
            assert!(
                furthest < e as i64,
                "view {k}: a keypoint on column {furthest}, edge {e}"
            );
            assert!(
                furthest >= e as i64 - 25,
                "view {k}: keypoints stop at {furthest}, edge {e}"
            );
        }
        // Every view across the chunk boundary took part, and so was checked.
        assert!(present.len() > N * 9 / 10, "{} of {N} views", present.len());
        assert!((248..N).all(|k| present.contains(&k)), "{present:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore = "260 photos through SfM: slow in a debug build"]
    fn masks_pair_with_their_images_cpu() {
        masks_pair_with_their_images_across_chunks(
            "pair_cpu",
            |c| {
                c.gpu = false;
                c.gpu_sift = false;
            },
            true,
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "260 photos through SfM: slow in a debug build"]
    fn masks_pair_with_their_images_gpu_vlfeat() {
        if have_gpu() {
            masks_pair_with_their_images_across_chunks("pair_gpu", cpu_pairs, true);
        }
    }

    /// `chunk_features` for a chunk starting at image 256, past the first
    /// chunk's boundary: with the GPU's results, and with the CPU fallback
    /// for a batch the device cannot take (`Unsupported`, which needs photos
    /// past its binding limit, so is given here). Image `256 + k` has
    /// the mask with edge `edges[k]`; its keypoints must lie left of it.
    #[cfg(feature = "gpu")]
    #[test]
    fn chunk_features_use_each_images_own_mask_past_the_first_chunk() {
        use visloc_vision::features::sift::{extract_sift, GrayImage, SiftConfig};
        const START: usize = 256;
        let edges = [40usize, 70, 55, 85];
        let (w, h) = (128usize, 96usize);
        let grays: Vec<Vec<f32>> = (0..edges.len())
            .map(|k| {
                let centre = nalgebra::Vector3::new(-1.0 + 0.3 * k as f64, 0.0, 0.0);
                room_photo(w as u32, h as u32, 110.0, centre, 0.05 * k as f64)
                    .pixels()
                    .map(|p| {
                        (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0
                    })
                    .collect()
            })
            .collect();
        let images: Vec<GrayImage> = grays
            .iter()
            .map(|g| GrayImage::new(w, h, g).unwrap())
            .collect();
        let sift_cfg = SiftConfig {
            contrast_threshold: 0.004,
            vlfeat_compatible_detector: true,
            vlfeat_compatible_descriptor: true,
            ..SiftConfig::default()
        };
        let raw: Vec<_> = images
            .iter()
            .map(|g| extract_sift(g, &sift_cfg).unwrap())
            .collect();
        // Only the chunk's own images have masks: asking for image `k` of the
        // chunk instead of `256 + k` would hit one of these.
        let load_mask = |i: usize| -> Result<Option<Vec<bool>>, super::EurocError> {
            let edge = edges.get(i.wrapping_sub(START)).copied();
            Ok(Some(
                (0..w * h)
                    .map(|p| edge.is_none_or(|e| p % w >= e))
                    .collect(),
            ))
        };
        let features_of = |kps: Vec<_>, desc: Vec<_>, masked: Option<&[bool]>| {
            let (kps, desc) =
                crate::masks::drop_masked_keypoints(kps, desc, masked.unwrap(), w, true);
            (
                super::FeatureSet {
                    keypoints: kps
                        .iter()
                        .map(|k| nalgebra::Point2::new(k.x, k.y))
                        .collect(),
                    descriptors: desc,
                },
                Vec::new(),
            )
        };
        let by_chunk = |batch| {
            super::chunk_features(
                START,
                images.len(),
                batch,
                &mut |k| {
                    extract_sift(&images[k], &sift_cfg)
                        .map_err(|e| super::EurocError::Sift(format!("{e}")))
                },
                &features_of,
                &load_mask,
            )
            .unwrap()
        };
        // The plain GPU extractor's path, one image at a time.
        let one_at_a_time = super::extract_chunk(
            START,
            images.len(),
            &mut |k| Ok(raw[k].clone()),
            &features_of,
            &load_mask,
        )
        .unwrap();
        for (what, got) in [
            ("gpu results", by_chunk(Ok(raw.clone()))),
            (
                "cpu fallback",
                by_chunk(Err(visloc_sift_gpu::SiftGpuError::Unsupported)),
            ),
            ("one at a time", one_at_a_time),
        ] {
            assert_eq!(got.len(), edges.len());
            for (k, (f, &e)) in got.iter().zip(&edges).enumerate() {
                let xs: Vec<f64> = f.0.keypoints.iter().map(|p| p.x).collect();
                let want = raw[k]
                    .0
                    .iter()
                    .filter(|p| (p.x.floor() as usize) < e)
                    .count();
                // Some keypoints of the image are past its edge, so the mask
                // of a neighbour (or none) would give other counts.
                assert!(
                    raw[k].0.iter().any(|p| (p.x.floor() as usize) >= e),
                    "{what} {k}"
                );
                assert!(
                    xs.iter().all(|&x| (x.floor() as usize) < e),
                    "{what}: image {k}"
                );
                assert_eq!(xs.len(), want, "{what}: image {k}");
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "260 photos through SfM: slow in a debug build"]
    fn masks_pair_with_their_images_gpu_plain() {
        if have_gpu() {
            masks_pair_with_their_images_across_chunks(
                "pair_plain",
                |c| {
                    cpu_pairs(c);
                    plain_detector(c);
                },
                false,
            );
        }
    }

    /// A mask that keeps everything is no mask: the same model, byte for
    /// byte, and the same keypoints.
    fn all_keep_mask_is_no_mask(tag: &str, sift: impl Fn(&mut super::PhotoSfmConfig)) {
        let root = std::env::temp_dir().join(format!("visloc_masks_{tag}_{}", std::process::id()));
        let images = room_photos(&root);
        edge_masks(
            &root.join("masks"),
            PHOTO_W,
            PHOTO_H,
            &[2 * PHOTO_W; 6],
            false,
        );
        let mut cfg = super::PhotoSfmConfig {
            max_size: 240,
            focal_px: Some(210.0),
            eval_every: 0,
            ..Default::default()
        };
        sift(&mut cfg);
        let (_, a, logs_a) = run_photos(&images, &root.join("a"), &cfg);
        let (_, _, _) = run_photos(&images, &root.join("again"), &cfg);
        cfg.mask_dir = Some(root.join("masks"));
        let (_, b, logs_b) = run_photos(&images, &root.join("b"), &cfg);
        assert!(a.registered >= 5);
        assert_eq!(keypoint_counts(&logs_b), (keypoint_counts(&logs_a).0, 0));
        for file in ["cameras.txt", "images.txt", "points3D.txt"] {
            let read = |d: &str| std::fs::read(root.join(d).join("sparse/0").join(file)).unwrap();
            // The run is deterministic, so a difference below is the mask's.
            assert!(
                read("a") == read("again"),
                "{file}: two runs without masks differ"
            );
            assert!(
                read("a") == read("b"),
                "{file}: an all-keep mask changed the model"
            );
        }
        assert_eq!((a.registered, a.points), (b.registered, b.points));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn all_keep_mask_matches_no_mask_cpu() {
        all_keep_mask_is_no_mask("keep_cpu", |c| {
            c.gpu = false;
            c.gpu_sift = false;
        });
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn all_keep_mask_matches_no_mask_gpu() {
        if have_gpu() {
            all_keep_mask_is_no_mask("keep_gpu", cpu_pairs);
        }
    }

    /// `mask_flip` through SfM: masks painted the other way round (255 =
    /// masked out) with the flip give the keypoints and the training alpha of
    /// the masks as they are.
    #[test]
    fn flipped_masks_give_the_same_run_as_unflipped() {
        let root = std::env::temp_dir().join(format!("visloc_masks_flip_{}", std::process::id()));
        let images = room_photos(&root);
        // Edges in the 240 x 180 working images; the masks are 480 x 360.
        let edges: Vec<u32> = (0..6).map(|k| 100 + 15 * k).collect();
        let native: Vec<u32> = edges.iter().map(|e| 2 * e).collect();
        edge_masks(&root.join("keep"), PHOTO_W, PHOTO_H, &native, false);
        edge_masks(&root.join("drop"), PHOTO_W, PHOTO_H, &native, true);
        let mut cfg = super::PhotoSfmConfig {
            gpu: false,
            gpu_sift: false,
            max_size: 240,
            focal_px: Some(210.0),
            eval_every: 0,
            mask_dir: Some(root.join("keep")),
            ..Default::default()
        };
        let (_, _, _) = run_photos(&images, &root.join("plain"), &cfg);
        cfg.mask_dir = Some(root.join("drop"));
        cfg.mask_flip = true;
        let (_, _, logs) = run_photos(&images, &root.join("flipped"), &cfg);
        assert!(logs.iter().any(|m| m.ends_with("(flipped)")), "{logs:?}");
        let mut dropped = 0;
        for (k, &e) in edges.iter().enumerate() {
            let a = keypoints_of(&root.join("plain"), k).unwrap();
            let b = keypoints_of(&root.join("flipped"), k).unwrap();
            assert!(
                a == b && !a.is_empty(),
                "view {k}: {} vs {}",
                a.len(),
                b.len()
            );
            assert!(a.iter().all(|p| p.0 < e as f64), "view {k}");
            let alpha = |out: &str| working_image(&root.join(out), k);
            assert!(
                alpha("plain") == alpha("flipped"),
                "view {k}: alpha differs"
            );
            assert_eq!(alpha("flipped").get_pixel(e, 5)[3], 0);
            assert_eq!(alpha("flipped").get_pixel(e - 1, 5)[3], 255);
            dropped += 1;
        }
        assert_eq!(dropped, 6);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A mask and `max_angle` together: the training alpha is 0 wherever
    /// either leaves a pixel out, and each of them leaves some out alone.
    #[test]
    fn mask_and_max_angle_both_leave_pixels_out_of_the_training_alpha() {
        let root = std::env::temp_dir().join(format!("visloc_masks_angle_{}", std::process::id()));
        let images = room_photos(&root);
        // 320 x 240 working images, masks of that size dropping from column 256.
        edge_masks(&root.join("masks"), 320, 240, &[256; 6], false);
        let cfg = super::PhotoSfmConfig {
            gpu: false,
            gpu_sift: false,
            max_size: 320,
            eval_every: 0,
            camera: Some(super::parse_camera_row("PINHOLE 480 360 420 420 240 180").unwrap()),
            max_angle: Some(25.0),
            mask_dir: Some(root.join("masks")),
            ..Default::default()
        };
        let (_, _, _) = run_photos(&images, &root.join("out"), &cfg);
        let alpha = working_image(&root.join("out"), 0);
        // The working camera: focal 280, principal point (160, 120).
        let tan_max = 25.0f64.to_radians().tan();
        let (mut by_mask, mut by_angle, mut by_both) = (0, 0, 0);
        for y in 0..240u32 {
            for x in 0..320u32 {
                let r = (x as f64 + 0.5 - 160.0).hypot(y as f64 + 0.5 - 120.0) / 280.0;
                if (r - tan_max).abs() < 1e-3 {
                    continue;
                }
                let (masked, past) = (x >= 256, r > tan_max);
                let want = if masked || past { 0 } else { 255 };
                assert_eq!(alpha.get_pixel(x, y)[3], want, "alpha at {x},{y}");
                match (masked, past) {
                    (true, false) => by_mask += 1,
                    (false, true) => by_angle += 1,
                    (true, true) => by_both += 1,
                    _ => {}
                }
            }
        }
        assert!(by_mask > 1000 && by_angle > 1000 && by_both > 100);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn masks_are_validated_before_anything_is_decoded() {
        let root = std::env::temp_dir().join(format!("visloc_masks_cfg_{}", std::process::id()));
        let images = room_photos(&root);
        let masks = root.join("masks");
        edge_masks(&masks, PHOTO_W, PHOTO_H, &[300; 6], false);
        let cpu = super::PhotoSfmConfig {
            gpu: false,
            gpu_sift: false,
            eval_every: 0,
            ..Default::default()
        };
        // The error, and what the run did before it.
        let fails = |cfg: &super::PhotoSfmConfig, out: &str| {
            let mut logs = Vec::new();
            match super::build_photo_dataset(&images, &root.join(out), cfg, &mut |m: &str| {
                logs.push(m.to_string())
            }) {
                Err(e) => {
                    // Nothing decoded: no log line about the images, and
                    // no working images directory.
                    assert!(logs.is_empty(), "{out}: {logs:?}");
                    assert!(!root.join(out).join("images").exists(), "{out}");
                    e.to_string()
                }
                Ok(_) => panic!("{out}: should fail"),
            }
        };
        // A missing mask stops the run before any image is decoded.
        std::fs::remove_file(masks.join("view_3.png")).unwrap();
        let cfg = super::PhotoSfmConfig {
            mask_dir: Some(masks.clone()),
            ..cpu.clone()
        };
        let err = fails(&cfg, "missing");
        assert!(
            err.contains("1 of 6 images have no mask") && err.contains("view_3.png"),
            "{err}"
        );
        // A mask of another aspect, the same.
        edge_masks(&masks, PHOTO_W, PHOTO_W, &[300; 6], false);
        let err = fails(&cfg, "aspect");
        assert!(err.contains("not the aspect"), "{err}");
        // Flip without masks, and masks with equirect or insv.
        let err = fails(
            &super::PhotoSfmConfig {
                mask_flip: true,
                ..cpu.clone()
            },
            "flip",
        );
        assert!(err.contains("mask_flip needs mask_dir"), "{err}");
        let err = fails(
            &super::PhotoSfmConfig {
                equirect: true,
                ..cfg.clone()
            },
            "equirect",
        );
        assert!(err.contains("not supported with equirect or insv"), "{err}");
        let err = fails(
            &super::PhotoSfmConfig {
                insv: Some(PathBuf::from("/no/such/video.insv")),
                ..cfg.clone()
            },
            "insv",
        );
        assert!(err.contains("not supported with equirect or insv"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
