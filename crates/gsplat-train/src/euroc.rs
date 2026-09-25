//! Raw EuRoC sequence -> training dataset, entirely in Rust (M3): no COLMAP,
//! no Python.
//!
//! 1. Read the sequence (`mav0/`) and take every `stride`-th cam0 frame.
//! 2. Undistort each frame (radial-tangential) to a pinhole camera with the
//!    same intrinsics (bilinear remap), write it as PNG for the trainer.
//! 3. Native SIFT features, matched to temporal neighbours (cross-checked,
//!    geometrically verified).
//! 4. visloc-rs incremental SfM (COLMAP-style mapper) -> camera poses and
//!    triangulated tracks. Monocular, so the scale is arbitrary; the trainer
//!    normalises learning rates by the scene extent.
//! 5. Views for the registered frames (every 8th held out, the usual split)
//!    and the tracks as coloured points (grey sampled from the frame).

use std::path::{Path, PathBuf};

use nalgebra::{Point2, Vector3};
use rayon::prelude::*;
use visloc_core::types::Camera;
use visloc_gsplat_core::colmap_scene::camera_view_from;
use visloc_gsplat_core::gaussian::Scene;
use visloc_io::euroc::read_euroc_dataset_dir;
use visloc_slam::{incremental_sfm, IncrementalSfmConfig, PairwiseMatches};
use visloc_vision::distortion::RadialTangential;
use visloc_vision::features::sift::{extract_sift, GrayImage, SiftConfig};
use visloc_vision::features::FeatureSet;
use visloc_vision::matching::{BruteForceMatcher, CrossCheckMatcher, DescriptorMatch, Matcher};
use visloc_vision::two_view::{
    ConfigurationType, TwoViewCorrespondence, TwoViewGeometryOptions, TwoViewGeometryVerifier,
};

use crate::dataset::{Dataset, View};
use crate::init::ColoredPoint;

/// Settings for [`build_euroc_dataset`].
#[derive(Debug, Clone)]
pub struct EurocSfmConfig {
    /// Use every `stride`-th cam0 frame.
    pub stride: usize,
    /// Cap on the number of frames used (after the stride).
    pub max_frames: usize,
    /// Match each frame to the next `window` frames...
    pub window: usize,
    /// ...plus these longer offsets (loop-ish constraints).
    pub skip_offsets: Vec<usize>,
    pub min_matches: usize,
    pub sift_max_keypoints: usize,
    /// Hold out every `eval_every`-th registered view.
    pub eval_every: usize,
    /// Extract SIFT on the GPU (`visloc-sift-gpu`, needs the `gpu` feature);
    /// same detector/descriptor path as the CPU extractor, validated by
    /// keypoint agreement rather than bytes.
    pub gpu_sift: bool,
    /// Run the SfM's global bundle adjustments on the GPU (`visloc-ba-gpu`,
    /// needs the `gpu` feature). Local windows stay on the CPU unless
    /// `VISLOC_BA_GPU_LOCAL` is set (latency-bound on small systems).
    pub gpu_ba: bool,
    /// Override the SfM's LM iteration budget per bundle adjustment.
    pub ba_max_iterations: Option<usize>,
    /// Relative cost tolerance that ends local BA windows early.
    pub local_ba_relative_tolerance: Option<f64>,
    /// Keyframe gate: a frame joins the SfM only once the accumulated median
    /// feature motion since the last kept frame reaches this many pixels.
    /// Near-static frames (e.g. before take-off) have no parallax and were
    /// registered metres off. `0` keeps every frame.
    pub min_keyframe_motion_px: f64,
}

impl Default for EurocSfmConfig {
    fn default() -> Self {
        Self {
            stride: 4,
            max_frames: 200,
            // Denser than the original 5 / [8, 12] / 4000 (GPU matching and
            // BA make it cheap): on EuRoC V1_01, MH_01, V1_02 and V2_01
            // (200 frames, stride 4) it registers 139-200 frames instead of
            // 75-183, and removes a monocular scale break on V1_01 (73.6 cm
            // -> 4.9 cm ATE).
            window: 10,
            skip_offsets: vec![15, 20, 30, 45, 60, 90, 120],
            min_matches: 30,
            sift_max_keypoints: 8000,
            eval_every: 8,
            gpu_sift: false,
            gpu_ba: false,
            ba_max_iterations: None,
            local_ba_relative_tolerance: None,
            min_keyframe_motion_px: 2.0,
        }
    }
}

/// Errors from the EuRoC pipeline.
#[derive(Debug, thiserror::Error)]
pub enum EurocError {
    #[error("euroc: {0}")]
    Euroc(String),
    #[error("image {path}: {source}")]
    Image {
        path: PathBuf,
        source: image::ImageError,
    },
    #[error("io {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("sift: {0}")]
    Sift(String),
    #[error("sfm: {0}")]
    Sfm(String),
    #[error("camera: {0}")]
    Camera(String),
}

/// What the pipeline produced (for logging).
#[derive(Debug, Clone, Copy)]
pub struct EurocSfmReport {
    pub frames: usize,
    pub pairs: usize,
    pub registered: usize,
    pub points: usize,
    pub mean_reprojection_px: f64,
    /// Sim(3)-aligned ATE RMSE of the registered camera centres against the
    /// Vicon/Leica ground truth (metres), when the sequence has one.
    pub ate_rmse_m: Option<f64>,
}

/// Sim(3) (Umeyama) alignment of `est` onto `gt`; returns the RMSE of the
/// aligned residuals, or `None` for fewer than three pairs.
pub fn sim3_ate_rmse(est: &[Vector3<f64>], gt: &[Vector3<f64>]) -> Option<f64> {
    let n = est.len();
    if n < 3 || gt.len() != n {
        return None;
    }
    let mu_e = est.iter().sum::<Vector3<f64>>() / n as f64;
    let mu_g = gt.iter().sum::<Vector3<f64>>() / n as f64;
    let mut cov = nalgebra::Matrix3::<f64>::zeros();
    let mut var_e = 0.0;
    for (e, g) in est.iter().zip(gt) {
        let (de, dg) = (e - mu_e, g - mu_g);
        cov += dg * de.transpose();
        var_e += de.norm_squared();
    }
    cov /= n as f64;
    var_e /= n as f64;
    let svd = cov.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let mut d = nalgebra::Matrix3::<f64>::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = u * d * vt;
    let scale = (svd.singular_values.component_mul(&d.diagonal())).sum() / var_e.max(1e-300);
    let t = mu_g - scale * r * mu_e;
    let sq: f64 = est
        .iter()
        .zip(gt)
        .map(|(e, g)| (scale * r * e + t - g).norm_squared())
        .sum();
    Some((sq / n as f64).sqrt())
}

/// Bilinear undistortion of a grey image to a pinhole camera with the same
/// intrinsics: for each output pixel, distort its normalised coordinate and
/// sample the source. Pixels that map outside the source are 0.
pub fn undistort_gray(
    src: &[u8],
    width: u32,
    height: u32,
    k: [f64; 4],
    dist: &RadialTangential,
) -> Vec<u8> {
    let (fx, fy, cx, cy) = (k[0], k[1], k[2], k[3]);
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0u8; w * h];
    for v in 0..h {
        for u in 0..w {
            let n = Point2::new((u as f64 - cx) / fx, (v as f64 - cy) / fy);
            let d = dist.distort_normalized(n);
            let (sx, sy) = (d.x * fx + cx, d.y * fy + cy);
            if sx < 0.0 || sy < 0.0 || sx > (w - 1) as f64 || sy > (h - 1) as f64 {
                continue;
            }
            let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            let (ax, ay) = (sx - x0 as f64, sy - y0 as f64);
            let p = |x: usize, y: usize| src[y * w + x] as f64;
            let val = p(x0, y0) * (1.0 - ax) * (1.0 - ay)
                + p(x1, y0) * ax * (1.0 - ay)
                + p(x0, y1) * (1.0 - ax) * ay
                + p(x1, y1) * ax * ay;
            out[v * w + u] = val.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn cpu_matches(fi: &FeatureSet, fj: &FeatureSet) -> Vec<DescriptorMatch> {
    CrossCheckMatcher::new(BruteForceMatcher { ratio: Some(0.8) })
        .match_descriptors(&fi.descriptors, &fj.descriptors)
}

fn verify_pair(
    camera: &Camera,
    fi: &FeatureSet,
    fj: &FeatureSet,
    dm: &[DescriptorMatch],
    min_matches: usize,
) -> Option<Vec<(usize, usize)>> {
    if dm.len() < min_matches {
        return None;
    }
    let corrs: Vec<TwoViewCorrespondence> = dm
        .iter()
        .map(|m| {
            TwoViewCorrespondence::new(fi.keypoints[m.query_index], fj.keypoints[m.train_index])
        })
        .collect();
    let report = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0))
        .classify(&corrs, camera);
    let keep = matches!(
        report.config,
        ConfigurationType::Calibrated | ConfigurationType::Uncalibrated
    );
    if !keep || report.inliers.len() < min_matches {
        return None;
    }
    Some(
        report
            .inliers
            .iter()
            .map(|&i| (dm[i].query_index, dm[i].train_index))
            .collect(),
    )
}

/// Run the pipeline on `sequence_dir` (the directory containing `mav0/`),
/// writing undistorted frames to `out_dir/images`. Returns the dataset (SfM
/// points as the init scene too), the coloured points and a report.
pub fn build_euroc_dataset(
    sequence_dir: &Path,
    out_dir: &Path,
    cfg: &EurocSfmConfig,
    log: &mut dyn FnMut(&str),
) -> Result<(Dataset, Vec<ColoredPoint>, EurocSfmReport), EurocError> {
    let seq = read_euroc_dataset_dir(sequence_dir).map_err(|e| EurocError::Euroc(e.to_string()))?;
    let calib = &seq.cam0_calibration;
    let (width, height) = calib.resolution;
    let k = calib.intrinsics;
    let dist = RadialTangential::from_euroc_coefficients(&calib.distortion_coefficients)
        .ok_or_else(|| EurocError::Euroc("unsupported cam0 distortion".into()))?;
    let camera = Camera::pinhole(1, width, height, k[0], k[1], k[2], k[3]);

    let images_dir = out_dir.join("images");
    std::fs::create_dir_all(&images_dir).map_err(|source| EurocError::Io {
        path: images_dir.clone(),
        source,
    })?;

    let frames: Vec<_> = seq
        .cam0_images
        .iter()
        .step_by(cfg.stride.max(1))
        .take(cfg.max_frames)
        .collect();
    log(&format!(
        "{} frames (stride {}) of {}x{}",
        frames.len(),
        cfg.stride,
        width,
        height
    ));

    // Undistort, save, extract SIFT.
    let sift_cfg = SiftConfig {
        max_keypoints: cfg.sift_max_keypoints,
        ..SiftConfig::default()
    };
    #[cfg(feature = "gpu")]
    let mut gpu_sift = if cfg.gpu_sift {
        let ctx = visloc_sift_gpu::GpuContext::new()
            .map_err(|e| EurocError::Sift(format!("gpu: {e}")))?;
        Some(visloc_sift_gpu::SiftGpu::new(ctx))
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    if cfg.gpu_sift || cfg.gpu_ba {
        return Err(EurocError::Sift(
            "gpu_sift / gpu_ba need the `gpu` feature".into(),
        ));
    }
    #[cfg(feature = "gpu")]
    if cfg.gpu_ba {
        let ctx =
            visloc_ba_gpu::GpuContext::new().map_err(|e| EurocError::Sfm(format!("gpu: {e}")))?;
        // Process-wide: the first registration wins.
        visloc_slam::set_ba_accelerator(Box::new(visloc_ba_gpu::GpuBundleAdjuster::new(ctx)));
    }
    let mut grays: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    let mut names: Vec<String> = Vec::with_capacity(frames.len());
    let mut features: Vec<FeatureSet> = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        let path = seq.cam0_image_dir.join(&f.filename);
        let img = image::open(&path)
            .map_err(|source| EurocError::Image {
                path: path.clone(),
                source,
            })?
            .to_luma8();
        let und = undistort_gray(img.as_raw(), width, height, k, &dist);
        let name = format!("frame_{i:05}.png");
        let out = images_dir.join(&name);
        let rgb: Vec<u8> = und.iter().flat_map(|&g| [g, g, g]).collect();
        image::save_buffer(&out, &rgb, width, height, image::ColorType::Rgb8).map_err(
            |source| EurocError::Image {
                path: out.clone(),
                source,
            },
        )?;
        let pixels: Vec<f32> = und.iter().map(|&b| b as f32).collect();
        let gray = GrayImage::new(width as usize, height as usize, &pixels)
            .map_err(|e| EurocError::Sift(format!("{e}")))?;
        #[cfg(feature = "gpu")]
        let extracted = match gpu_sift.as_mut() {
            Some(g) => g
                .extract(&gray, &sift_cfg)
                .map_err(|e| EurocError::Sift(format!("{e}"))),
            None => extract_sift(&gray, &sift_cfg).map_err(|e| EurocError::Sift(format!("{e}"))),
        };
        #[cfg(not(feature = "gpu"))]
        let extracted =
            extract_sift(&gray, &sift_cfg).map_err(|e| EurocError::Sift(format!("{e}")));
        let (kps, desc) = extracted?;
        features.push(FeatureSet {
            keypoints: kps.iter().map(|k| Point2::new(k.x, k.y)).collect(),
            descriptors: desc,
        });
        grays.push(und);
        names.push(name);
    }
    log(&format!(
        "sift: mean {} keypoints",
        features.iter().map(|f| f.keypoints.len()).sum::<usize>() / features.len().max(1)
    ));

    // Temporal-neighbour pairs, verified.
    let n = features.len();
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        let mut offsets: Vec<usize> = (1..=cfg.window).collect();
        offsets.extend(cfg.skip_offsets.iter().copied());
        for d in offsets {
            if i + d < n {
                candidates.push((i, i + d));
            }
        }
    }
    // Cross-checked ratio matches for every candidate pair: one batched GPU
    // pass over a device-resident descriptor bank, or per pair on the CPU.
    #[cfg(feature = "gpu")]
    let all_matches: Option<Vec<Vec<DescriptorMatch>>> = gpu_sift.as_ref().map(|g| {
        let ctx = g.context();
        let sets: Vec<&[Vec<f32>]> = features.iter().map(|f| f.descriptors.as_slice()).collect();
        let bank = visloc_sift_gpu::FeatureBank::upload(ctx, &sets);
        match bank {
            Ok(bank) => visloc_sift_gpu::GpuMatcher::new(ctx).match_pairs(
                ctx,
                &bank,
                &candidates,
                Some(0.8),
                true,
            ),
            Err(_) => candidates
                .iter()
                .map(|&(i, j)| cpu_matches(&features[i], &features[j]))
                .collect(),
        }
    });
    #[cfg(not(feature = "gpu"))]
    let all_matches: Option<Vec<Vec<DescriptorMatch>>> = None;
    log(&format!("matched {} candidate pairs", candidates.len()));

    // Keyframe gate from the consecutive-pair matches: median pixel motion
    // of the cross-checked matches, accumulated since the last kept frame.
    let consecutive_motion = |c: usize, i: usize, j: usize| -> Option<f64> {
        let dm = match &all_matches {
            Some(all) => std::borrow::Cow::Borrowed(&all[c]),
            None => std::borrow::Cow::Owned(cpu_matches(&features[i], &features[j])),
        };
        if dm.len() < cfg.min_matches {
            return None; // weak overlap: treat as motion
        }
        let mut d: Vec<f64> = dm
            .iter()
            .map(|m| {
                (features[i].keypoints[m.query_index] - features[j].keypoints[m.train_index]).norm()
            })
            .collect();
        d.sort_by(|a, b| a.total_cmp(b));
        Some(d[d.len() / 2])
    };
    let mut keep = vec![true; n];
    if cfg.min_keyframe_motion_px > 0.0 {
        let mut accumulated = 0.0;
        for (c, &(i, j)) in candidates.iter().enumerate() {
            if j != i + 1 {
                continue;
            }
            match consecutive_motion(c, i, j) {
                Some(px) => accumulated += px,
                None => accumulated = f64::INFINITY,
            }
            if accumulated >= cfg.min_keyframe_motion_px {
                accumulated = 0.0;
            } else {
                keep[j] = false;
            }
        }
    }
    let dropped = keep.iter().filter(|k| !**k).count();
    if dropped > 0 {
        log(&format!(
            "keyframe gate: {dropped} near-static frames left out (< {} px)",
            cfg.min_keyframe_motion_px
        ));
    }
    // Geometric verification is independent per pair (seeded RANSAC), so it
    // runs in parallel; results keep the candidate order.
    let pairwise: Vec<PairwiseMatches> = candidates
        .par_iter()
        .enumerate()
        .filter(|(_, &(i, j))| keep[i] && keep[j])
        .filter_map(|(c, &(i, j))| {
            let dm = match &all_matches {
                Some(all) => std::borrow::Cow::Borrowed(&all[c]),
                None => std::borrow::Cow::Owned(cpu_matches(&features[i], &features[j])),
            };
            verify_pair(&camera, &features[i], &features[j], &dm, cfg.min_matches).map(|matches| {
                PairwiseMatches {
                    image_i: i,
                    image_j: j,
                    matches,
                    two_view_config: None,
                    essential_matches: None,
                    essential_matrix: None,
                }
            })
        })
        .collect();
    log(&format!("{} verified pairs", pairwise.len()));

    let mut sfm_cfg = IncrementalSfmConfig {
        min_seed_matches: cfg.min_matches,
        colmap_style_mapper: true,
        ..IncrementalSfmConfig::default()
    };
    if let Some(it) = cfg.ba_max_iterations {
        sfm_cfg.ba_config.max_iterations = it;
    }
    sfm_cfg.local_ba_relative_cost_tolerance = cfg.local_ba_relative_tolerance;
    let result = incremental_sfm(&camera, &features, &pairwise, &sfm_cfg)
        .map_err(|e| EurocError::Sfm(e.to_string()))?;
    let registered = result.poses.iter().filter(|p| p.is_some()).count();
    // ATE against ground truth: nearest GT sample within 10 ms of each
    // registered frame, cam0 centre = p_WB + R_WB t_BS.
    let ate_rmse_m = {
        let t_bs = seq.cam0_calibration.t_body_sensor;
        let t_bs = Vector3::new(t_bs[(0, 3)], t_bs[(1, 3)], t_bs[(2, 3)]);
        let gt = &seq.ground_truth;
        let mut est_c = Vec::new();
        let mut gt_c = Vec::new();
        for (i, pose) in result.poses.iter().enumerate() {
            let Some(pose) = pose else { continue };
            let ts = frames[i].timestamp_nanoseconds;
            let k = gt.partition_point(|s| s.timestamp_nanoseconds < ts);
            let near = [k.checked_sub(1), Some(k)]
                .into_iter()
                .flatten()
                .filter_map(|j| gt.get(j))
                .min_by_key(|s| (s.timestamp_nanoseconds - ts).abs());
            if let Some(s) = near.filter(|s| (s.timestamp_nanoseconds - ts).abs() <= 10_000_000) {
                est_c.push(pose.camera_to_world().translation);
                gt_c.push(s.position_world + s.orientation_world * t_bs);
            }
        }
        // Associated centres for offline inspection / other aligners.
        let csv: String = std::iter::once(
            "frame,est_x,est_y,est_z,gt_x,gt_y,gt_z
"
            .to_string(),
        )
        .chain(est_c.iter().zip(&gt_c).enumerate().map(|(i, (e, g))| {
            format!(
                "{i},{},{},{},{},{},{}
",
                e.x, e.y, e.z, g.x, g.y, g.z
            )
        }))
        .collect();
        let _ = std::fs::write(out_dir.join("trajectory_vs_gt.csv"), csv);
        sim3_ate_rmse(&est_c, &gt_c)
    };
    if let Some(ate) = ate_rmse_m {
        log(&format!("sfm: Sim(3) ATE RMSE {:.4} m", ate));
    }
    log(&format!(
        "sfm: {registered}/{n} registered, {} tracks, reprojection {:.3} px",
        result.tracks.len(),
        result.mean_reprojection_px
    ));

    // Views for registered frames (in frame order = name order).
    let cam = result.refined_camera.as_ref().unwrap_or(&camera);
    let mut views = Vec::new();
    for (i, pose) in result.poses.iter().enumerate() {
        let Some(pose) = pose else { continue };
        let view = camera_view_from(cam, pose).map_err(|e| EurocError::Camera(e.to_string()))?;
        views.push(View {
            name: names[i].clone(),
            camera: view,
            image_path: images_dir.join(&names[i]),
        });
    }
    let mut train = Vec::new();
    let mut eval = Vec::new();
    for (i, v) in views.into_iter().enumerate() {
        if cfg.eval_every > 0 && i % cfg.eval_every == 0 {
            eval.push(v);
        } else {
            train.push(v);
        }
    }

    // Tracks -> grey-coloured points (colour at the first observation).
    let w = width as usize;
    let points: Vec<ColoredPoint> = result
        .tracks
        .iter()
        .filter_map(|t| {
            let &(img, _, px) = t.observations.first()?;
            let (x, y) = (px.x.round() as isize, px.y.round() as isize);
            let g = if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < height as usize {
                grays[img][y as usize * w + x as usize]
            } else {
                128
            };
            Some(ColoredPoint {
                position: Vector3::new(
                    t.position.x as f32,
                    t.position.y as f32,
                    t.position.z as f32,
                ),
                rgb: [g, g, g],
            })
        })
        .collect();

    let report = EurocSfmReport {
        frames: n,
        pairs: pairwise.len(),
        registered,
        points: points.len(),
        mean_reprojection_px: result.mean_reprojection_px,
        ate_rmse_m,
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
    use super::sim3_ate_rmse;
    use nalgebra::{Rotation3, Vector3};

    #[test]
    fn sim3_ate_is_zero_for_a_similar_trajectory_and_positive_otherwise() {
        let gt: Vec<Vector3<f64>> = (0..20)
            .map(|i| {
                let t = i as f64 * 0.3;
                Vector3::new(t.cos() * 2.0, t.sin() * 1.5, 0.1 * t)
            })
            .collect();
        let r = Rotation3::from_euler_angles(0.3, -0.7, 1.1);
        let est: Vec<Vector3<f64>> = gt
            .iter()
            .map(|g| 0.37 * (r * g) + Vector3::new(4.0, -2.0, 0.5))
            .collect();
        assert!(sim3_ate_rmse(&est, &gt).unwrap() < 1e-9);
        let mut noisy = est.clone();
        noisy[5].x += 0.37;
        assert!(sim3_ate_rmse(&noisy, &gt).unwrap() > 1e-3);
        assert!(sim3_ate_rmse(&est[..2], &gt[..2]).is_none());
    }
}
