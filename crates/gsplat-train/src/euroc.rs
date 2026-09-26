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
use visloc_slam::{IncrementalSfmConfig, PairwiseMatches};
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
    /// `key=value` overrides of the incremental mapper configuration (see
    /// [`apply_sfm_override`]), applied last.
    pub sfm_overrides: Vec<String>,
    /// Keep planar / panoramic / multi-model verified pairs too (COLMAP's
    /// `UseInlierMatchesCheck`), not only calibrated/uncalibrated ones.
    pub keep_planar: bool,
    /// With `keep_planar`: keep planar / multi-model pairs but still drop
    /// pure-rotation (panoramic) ones, which carry no baseline.
    pub keep_planar_no_panoramic: bool,
    /// Inlier floor for a verified pair; `None` = `min_matches`.
    pub verify_min_inliers: Option<usize>,
    /// Diagnostic: replace the SIFT features and verified pairs with an
    /// export of a COLMAP database (`export_colmap_db.py`), so the mapper
    /// can be compared on identical correspondences.
    pub import_colmap: Option<PathBuf>,
    /// Diagnostic: seed the mapper with fixed initial poses from a text file
    /// of `frame qw qx qy qz tx ty tz` (world-to-camera, COLMAP convention).
    pub init_poses: Option<PathBuf>,
    /// Run the faithful COLMAP incremental-mapper port
    /// (`visloc_slam::colmap_incremental`) instead of `incremental_sfm`.
    pub colmap_port_mapper: bool,
    /// RootSIFT (L1-root) descriptor normalisation, COLMAP's default.
    pub sift_l1_root: bool,
    /// `key=value` overrides of the SIFT configuration (see `apply_sift_override`).
    pub sift_overrides: Vec<String>,
    /// Diagnostic: replace the extracted keypoints/descriptors with
    /// `export_colmap_features.py` output (matching and verification stay ours).
    pub import_features: Option<PathBuf>,
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
            sfm_overrides: Vec::new(),
            keep_planar: false,
            keep_planar_no_panoramic: false,
            verify_min_inliers: None,
            import_colmap: None,
            init_poses: None,
            colmap_port_mapper: false,
            sift_l1_root: false,
            sift_overrides: Vec::new(),
            import_features: None,
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

/// Poses, tracks and mean reprojection error read back from the port.
type PortModel = (
    Vec<Option<visloc_core::geometry::Pose>>,
    Vec<visloc_slam::SfmTrack>,
    f64,
);
/// A chunk of candidate pairs and their raw matches, awaiting verification.
type MatchedChunk<'a> = (&'a [(usize, usize)], Vec<Vec<DescriptorMatch>>);
/// Features and verified pairs imported from a COLMAP database export.
type ImportedMatches = (Vec<FeatureSet>, Vec<PairwiseMatches>);

/// The parts of an SfM result the dataset builder consumes.
struct SfmOutcome {
    poses: Vec<Option<visloc_core::geometry::Pose>>,
    tracks: Vec<visloc_slam::SfmTrack>,
    mean_reprojection_px: f64,
    refined_camera: Option<Camera>,
}

/// Run the COLMAP incremental-mapper port on our features/verified pairs:
/// write its inputs (single-sensor rig manifest, keypoints, VISLOC-COLMAP-1
/// pairs) under `out_dir/port`, run `colmap_incremental::pipeline::run`,
/// and read the largest model back as poses + tracks.
fn run_colmap_port(
    out_dir: &Path,
    camera: &Camera,
    width: u32,
    height: u32,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    log: &mut dyn FnMut(&str),
) -> Result<PortModel, EurocError> {
    use std::io::Write;
    use visloc_slam::colmap_incremental::{pipeline, DatabaseCache, PipelineOptions};
    let err = |e: String| EurocError::Sfm(format!("colmap-port: {e}"));
    let dir = out_dir.join("port");
    let feat_dir = dir.join("features");
    std::fs::create_dir_all(&feat_dir).map_err(|e| err(e.to_string()))?;
    let [fx, fy, cx, cy] = [
        camera.params[0],
        camera.params[1],
        camera.params[2],
        camera.params[3],
    ];
    let name = |i: usize| format!("frame_{i:05}.png");
    // Only images that take part in a verified pair (the port's
    // correspondence graph has no node for an isolated image); names keep
    // the original frame index, the export order is the port's image id.
    let mut used = vec![false; features.len()];
    for p in pairwise {
        used[p.image_i] = true;
        used[p.image_j] = true;
    }
    let order: Vec<usize> = (0..features.len()).filter(|&i| used[i]).collect();
    let mut slot = vec![usize::MAX; features.len()];
    for (k, &i) in order.iter().enumerate() {
        slot[i] = k;
    }
    let mut manifest = format!(
        "# generalized-rig-manifest-v1\nS 0 1 {width} {height} {fx} {fy} {cx} {cy} 1 0 0 0 0 0 0\n"
    );
    for &i in &order {
        manifest.push_str(&format!("F {i} {} 0\n", name(i)));
        let text: String = features[i]
            .keypoints
            .iter()
            .map(|k| format!("{} {}\n", k.x, k.y))
            .collect();
        std::fs::write(feat_dir.join(format!("frame_{i:05}_features.txt")), text)
            .map_err(|e| err(e.to_string()))?;
    }
    std::fs::write(dir.join("manifest.txt"), manifest).map_err(|e| err(e.to_string()))?;
    let mut bin: Vec<u8> = b"VISLOC-COLMAP-1\0".to_vec();
    bin.extend((order.len() as u64).to_le_bytes());
    for &i in &order {
        let f = &features[i];
        let n = name(i);
        bin.extend((n.len() as u64).to_le_bytes());
        bin.extend(n.as_bytes());
        bin.extend((f.keypoints.len() as u64).to_le_bytes());
    }
    bin.extend((pairwise.len() as u64).to_le_bytes());
    for p in pairwise {
        bin.extend((slot[p.image_i] as u64).to_le_bytes());
        bin.extend((slot[p.image_j] as u64).to_le_bytes());
        bin.extend((p.matches.len() as u64).to_le_bytes());
        for &(a, b) in &p.matches {
            bin.extend((a as u32).to_le_bytes());
            bin.extend((b as u32).to_le_bytes());
        }
    }
    std::fs::File::create(dir.join("pairs.bin"))
        .and_then(|mut f| f.write_all(&bin))
        .map_err(|e| err(e.to_string()))?;

    let db = DatabaseCache::from_generalized_rig_export(
        &dir.join("manifest.txt"),
        &feat_dir,
        &dir.join("pairs.bin"),
    )
    .map_err(|e| err(e.to_string()))?;
    let mut options = PipelineOptions::default();
    // The port's default (8) is a rig-benchmark control override; use
    // COLMAP's own `Mapper.abs_pose_min_num_inliers` for single cameras.
    options.mapper.abs_pose_min_num_inliers = 30;
    let run = pipeline::run(&options, &db);
    let _ = std::fs::write(
        dir.join("mapper.log"),
        run.log.join(
            "
",
        ),
    );
    let best = run
        .models
        .iter()
        .max_by_key(|m| m.reconstruction.num_reg_images())
        .ok_or_else(|| err("no model reconstructed".into()))?;
    log(&format!(
        "colmap-port: {} model(s), largest {} registered",
        run.models.len(),
        best.reconstruction.num_reg_images()
    ));
    let model_dir = dir.join("model");
    best.reconstruction
        .export_colmap_text(&model_dir)
        .map_err(|e| err(e.to_string()))?;

    // Read the text model back: images -> poses, points -> tracks.
    let images_txt =
        std::fs::read_to_string(model_dir.join("images.txt")).map_err(|e| err(e.to_string()))?;
    let mut poses = vec![None; features.len()];
    let mut frame_of_image: std::collections::HashMap<u64, usize> = Default::default();
    let mut lines = images_txt.lines().filter(|l| !l.starts_with('#'));
    while let Some(header) = lines.next() {
        let _points = lines.next();
        let t: Vec<&str> = header.split_whitespace().collect();
        if t.len() < 10 {
            continue;
        }
        let v: Vec<f64> = t[1..8].iter().filter_map(|x| x.parse().ok()).collect();
        let (Ok(image_id), Some(frame)) = (
            t[0].parse::<u64>(),
            t[9].strip_prefix("frame_")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok()),
        ) else {
            continue;
        };
        if v.len() != 7 || frame >= poses.len() {
            continue;
        }
        let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            v[0], v[1], v[2], v[3],
        ));
        poses[frame] = Some(visloc_core::geometry::Pose::from_world_to_camera(
            q,
            Vector3::new(v[4], v[5], v[6]),
        ));
        frame_of_image.insert(image_id, frame);
    }
    let points_txt =
        std::fs::read_to_string(model_dir.join("points3D.txt")).map_err(|e| err(e.to_string()))?;
    let mut tracks = Vec::new();
    let (mut err_sum, mut err_n) = (0.0, 0usize);
    for line in points_txt.lines().filter(|l| !l.starts_with('#')) {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 8 {
            continue;
        }
        let xyz: Vec<f64> = t[1..4].iter().filter_map(|x| x.parse().ok()).collect();
        if xyz.len() != 3 {
            continue;
        }
        if let Ok(e) = t[7].parse::<f64>() {
            err_sum += e;
            err_n += 1;
        }
        let observations = t[8..]
            .chunks_exact(2)
            .filter_map(|c| {
                let frame = *frame_of_image.get(&c[0].parse::<u64>().ok()?)?;
                let kp: usize = c[1].parse().ok()?;
                Some((frame, kp, *features[frame].keypoints.get(kp)?))
            })
            .collect();
        tracks.push(visloc_slam::SfmTrack {
            position: nalgebra::Point3::new(xyz[0], xyz[1], xyz[2]),
            observations,
        });
    }
    Ok((poses, tracks, err_sum / err_n.max(1) as f64))
}

/// Load `export_colmap_db.py` output: per-frame keypoints (no descriptors)
/// and verified inlier matches.
fn import_colmap_export(dir: &Path, n: usize) -> Result<ImportedMatches, EurocError> {
    let bad = |what: String| EurocError::Sfm(format!("import-colmap: {what}"));
    let mut features = Vec::with_capacity(n);
    for i in 0..n {
        let path = dir.join("keypoints").join(format!("frame_{i:05}.txt"));
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let keypoints: Vec<Point2<f64>> = text
            .lines()
            .filter_map(|l| {
                let mut it = l.split_whitespace().map(|v| v.parse::<f64>());
                Some(Point2::new(it.next()?.ok()?, it.next()?.ok()?))
            })
            .collect();
        let descriptors = vec![Vec::new(); keypoints.len()];
        features.push(FeatureSet {
            keypoints,
            descriptors,
        });
    }
    let text = std::fs::read_to_string(dir.join("pairs.txt")).map_err(|e| bad(e.to_string()))?;
    let mut pairwise = Vec::new();
    for line in text.lines() {
        let v: Vec<usize> = line
            .split_whitespace()
            .map(|t| t.parse::<usize>())
            .collect::<Result<_, _>>()
            .map_err(|e| bad(e.to_string()))?;
        if v.len() < 2 || v[0] >= n || v[1] >= n {
            continue;
        }
        let matches: Vec<(usize, usize)> = v[2..].chunks_exact(2).map(|c| (c[0], c[1])).collect();
        pairwise.push(PairwiseMatches {
            image_i: v[0],
            image_j: v[1],
            matches,
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        });
    }
    Ok((features, pairwise))
}

#[cfg(feature = "gpu")]
macro_rules! visloc_sift_gpu_or_unit {
    () => {
        visloc_sift_gpu::SiftGpu
    };
}
#[cfg(not(feature = "gpu"))]
macro_rules! visloc_sift_gpu_or_unit {
    () => {
        ()
    };
}

/// Apply one `key=value` override to the SIFT configuration.
pub fn apply_sift_override(cfg: &mut SiftConfig, kv: &str) -> Result<(), String> {
    let (key, value) = kv
        .split_once('=')
        .ok_or_else(|| format!("--sift-opt {kv}: expected key=value"))?;
    let num = || {
        value
            .parse::<f64>()
            .map_err(|e| format!("--sift-opt {key}: {e}"))
    };
    let flag = || matches!(value, "1" | "true");
    match key {
        "descriptor_magnification" => cfg.descriptor_magnification = num()?,
        "max_orientations" => cfg.max_orientations = num()? as usize,
        "prefer_larger_scale" => cfg.prefer_larger_scale = flag(),
        "full_pyramid" => cfg.full_pyramid = flag(),
        "contrast_threshold" => cfg.contrast_threshold = num()?,
        "edge_threshold" => cfg.edge_threshold = num()?,
        "octaves" => cfg.octaves = num()? as usize,
        "sigma_base" => cfg.sigma_base = num()?,
        _ => return Err(format!("--sift-opt: unknown key {key}")),
    }
    Ok(())
}

/// Apply one `key=value` override to the incremental mapper configuration.
pub fn apply_sfm_override(cfg: &mut IncrementalSfmConfig, kv: &str) -> Result<(), String> {
    let (key, value) = kv
        .split_once('=')
        .ok_or_else(|| format!("--sfm-opt {kv}: expected key=value"))?;
    let flag = || -> Result<bool, String> {
        match value {
            "1" | "true" => Ok(true),
            "0" | "false" => Ok(false),
            _ => Err(format!("--sfm-opt {key}: expected a bool, got {value}")),
        }
    };
    let num = || -> Result<f64, String> {
        value
            .parse::<f64>()
            .map_err(|e| format!("--sfm-opt {key}: {e}"))
    };
    match key {
        "retriangulate" => cfg.retriangulate = flag()?,
        "incremental_correspondence_triangulation" => {
            cfg.incremental_correspondence_triangulation = flag()?
        }
        "colmap_style_mapper" => cfg.colmap_style_mapper = flag()?,
        "final_iterative_global_refinement" => cfg.final_iterative_global_refinement = flag()?,
        "ba_every" => cfg.ba_every = num()? as usize,
        "seed_min_median_tri_angle_deg" => cfg.seed_min_median_tri_angle_deg = Some(num()?),
        "min_seed_matches" => cfg.min_seed_matches = num()? as usize,
        "seed_trials" => cfg.seed_trials = num()? as usize,
        "post_refinement_registration" => cfg.post_refinement_registration = flag()?,
        "filter_images" => cfg.filter_images = flag()?,
        "final_global_ba" => cfg.final_global_ba = flag()?,
        "pnp_max_iterations" => cfg.pnp_max_iterations = num()? as usize,
        "min_pnp_inliers" => cfg.min_pnp_inliers = num()? as usize,
        "max_registration_trials" => cfg.max_registration_trials = num()? as usize,
        "local_ba_num_images" => cfg.local_ba_num_images = num()? as usize,
        "global_ba_images_ratio" => cfg.global_ba_images_ratio = num()?,
        "global_ba_max_refinements" => cfg.global_ba_max_refinements = num()? as usize,
        "max_reprojection_error_px" => cfg.max_reprojection_error_px = num()?,
        "min_triangulation_angle_deg" => cfg.min_triangulation_angle_deg = num()?,
        "huber_delta" => {
            cfg.ba_config.robust_kernel = visloc_slam::RobustKernel::Huber { delta: num()? }
        }
        "next_image_policy" => {
            cfg.next_image_policy = match value {
                "auto" => visloc_slam::NextImagePolicy::Auto,
                "visibility" => visloc_slam::NextImagePolicy::VisibilityPyramid,
                "count" => visloc_slam::NextImagePolicy::CorrespondenceCount,
                _ => return Err(format!("--sfm-opt next_image_policy: {value}")),
            }
        }
        _ => return Err(format!("--sfm-opt: unknown key {key}")),
    }
    Ok(())
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
    keep_planar: bool,
    no_panoramic: bool,
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
    ) || (keep_planar
        && match report.config {
            ConfigurationType::Planar | ConfigurationType::Multiple => true,
            ConfigurationType::Panoramic | ConfigurationType::PlanarOrPanoramic => !no_panoramic,
            _ => false,
        });
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
    let mut sift_cfg = SiftConfig {
        max_keypoints: cfg.sift_max_keypoints,
        normalization: if cfg.sift_l1_root {
            visloc_vision::features::sift::SiftNormalization::L1Root
        } else {
            visloc_vision::features::sift::SiftNormalization::L2
        },
        ..SiftConfig::default()
    };
    for o in &cfg.sift_overrides {
        apply_sift_override(&mut sift_cfg, o).map_err(EurocError::Sift)?;
    }
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
    // Decode + undistort + write the training PNG in parallel; SIFT then
    // runs on the GPU one frame at a time (one device), or in parallel on
    // the CPU.
    let grays: Vec<Vec<u8>> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| -> Result<Vec<u8>, EurocError> {
            let path = seq.cam0_image_dir.join(&f.filename);
            let img = image::open(&path)
                .map_err(|source| EurocError::Image {
                    path: path.clone(),
                    source,
                })?
                .to_luma8();
            let und = undistort_gray(img.as_raw(), width, height, k, &dist);
            let out = images_dir.join(format!("frame_{i:05}.png"));
            let rgb: Vec<u8> = und.iter().flat_map(|&g| [g, g, g]).collect();
            image::save_buffer(&out, &rgb, width, height, image::ColorType::Rgb8).map_err(
                |source| EurocError::Image {
                    path: out.clone(),
                    source,
                },
            )?;
            Ok(und)
        })
        .collect::<Result<_, _>>()?;
    let names: Vec<String> = (0..frames.len())
        .map(|i| format!("frame_{i:05}.png"))
        .collect();
    let extract_one = |und: &[u8], gpu: Option<&mut visloc_sift_gpu_or_unit!()>| {
        let _ = &gpu;
        let pixels: Vec<f32> = und.iter().map(|&b| b as f32).collect();
        let gray = GrayImage::new(width as usize, height as usize, &pixels)
            .map_err(|e| EurocError::Sift(format!("{e}")))?;
        #[cfg(feature = "gpu")]
        let extracted = match gpu {
            Some(g) => g
                .extract(&gray, &sift_cfg)
                .map_err(|e| EurocError::Sift(format!("{e}"))),
            None => extract_sift(&gray, &sift_cfg).map_err(|e| EurocError::Sift(format!("{e}"))),
        };
        #[cfg(not(feature = "gpu"))]
        let extracted =
            extract_sift(&gray, &sift_cfg).map_err(|e| EurocError::Sift(format!("{e}")));
        let (kps, desc) = extracted?;
        Ok::<_, EurocError>(FeatureSet {
            keypoints: kps.iter().map(|k| Point2::new(k.x, k.y)).collect(),
            descriptors: desc,
        })
    };
    #[cfg(feature = "gpu")]
    let mut features: Vec<FeatureSet> = match gpu_sift.as_mut() {
        Some(g) => grays
            .iter()
            .map(|und| extract_one(und, Some(&mut *g)))
            .collect::<Result<_, _>>()?,
        None => grays
            .par_iter()
            .map(|und| extract_one(und, None))
            .collect::<Result<_, _>>()?,
    };
    #[cfg(not(feature = "gpu"))]
    let mut features: Vec<FeatureSet> = grays
        .par_iter()
        .map(|und| extract_one(und, None))
        .collect::<Result<_, _>>()?;
    if let Some(dir) = &cfg.import_features {
        for (i, f) in features.iter_mut().enumerate() {
            let path = dir.join(format!("frame_{i:05}.bin"));
            let bytes =
                std::fs::read(&path).map_err(|e| EurocError::Sfm(format!("{path:?}: {e}")))?;
            let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
            let f32_at =
                |k: usize| f32::from_le_bytes(bytes[4 + 4 * k..8 + 4 * k].try_into().unwrap());
            f.keypoints = (0..n)
                .map(|j| Point2::new(f32_at(2 * j) as f64, f32_at(2 * j + 1) as f64))
                .collect();
            f.descriptors = (0..n)
                .map(|j| (0..128).map(|c| f32_at(2 * n + 128 * j + c)).collect())
                .collect();
        }
        log("import-features: replaced keypoints/descriptors");
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
    // Cross-checked ratio matches: one batched GPU pass over a
    // device-resident descriptor bank, or per pair on the CPU.
    #[cfg(feature = "gpu")]
    let gpu_match = gpu_sift.as_ref().and_then(|g| {
        let ctx = g.context();
        let sets: Vec<&[Vec<f32>]> = features.iter().map(|f| f.descriptors.as_slice()).collect();
        visloc_sift_gpu::FeatureBank::upload(ctx, &sets)
            .ok()
            .map(|bank| (ctx, bank, visloc_sift_gpu::GpuMatcher::new(ctx)))
    });
    let match_pairs = |pairs: &[(usize, usize)]| -> Vec<Vec<DescriptorMatch>> {
        #[cfg(feature = "gpu")]
        if let Some((ctx, bank, m)) = &gpu_match {
            return m.match_pairs(ctx, bank, pairs, Some(0.8), true);
        }
        pairs
            .par_iter()
            .map(|&(i, j)| cpu_matches(&features[i], &features[j]))
            .collect()
    };

    // Keyframe gate from the consecutive-pair matches: median pixel motion
    // of the cross-checked matches, accumulated since the last kept frame.
    let consecutive: Vec<(usize, usize)> = (1..n).map(|j| (j - 1, j)).collect();
    let consecutive_matches = match_pairs(&consecutive);
    let consecutive_motion = |dm: &[DescriptorMatch], i: usize, j: usize| -> Option<f64> {
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
    if cfg.min_keyframe_motion_px > 0.0 && cfg.window >= 1 {
        let mut accumulated = 0.0;
        for (&(i, j), dm) in consecutive.iter().zip(&consecutive_matches) {
            match consecutive_motion(dm, i, j) {
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

    // Match and verify only the pairs between kept frames, as a two-stage
    // software pipeline: the GPU matches chunk k while the CPU verifies
    // chunk k-1 (seeded RANSAC, so the output is order-deterministic).
    let kept: Vec<(usize, usize)> = candidates
        .iter()
        .copied()
        .filter(|&(i, j)| keep[i] && keep[j])
        .collect();
    let verify_chunk = |chunk: &[(usize, usize)], dms: Vec<Vec<DescriptorMatch>>| {
        chunk
            .par_iter()
            .zip(dms.into_par_iter())
            .filter_map(|(&(i, j), dm)| {
                verify_pair(
                    &camera,
                    &features[i],
                    &features[j],
                    &dm,
                    cfg.verify_min_inliers.unwrap_or(cfg.min_matches),
                    cfg.keep_planar,
                    cfg.keep_planar_no_panoramic,
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
            .collect::<Vec<_>>()
    };
    let mut pairwise: Vec<PairwiseMatches> = Vec::new();
    let mut pending: Option<MatchedChunk> = None;
    let (mut match_s, mut verify_s) = (0.0f64, 0.0f64);
    for chunk in kept.chunks(512) {
        let ((dms, ms), (verified, vs)) = rayon::join(
            || {
                let t = std::time::Instant::now();
                (match_pairs(chunk), t.elapsed().as_secs_f64())
            },
            || {
                let t = std::time::Instant::now();
                (
                    pending.take().map(|(c, d)| verify_chunk(c, d)),
                    t.elapsed().as_secs_f64(),
                )
            },
        );
        match_s += ms;
        verify_s += vs;
        pairwise.extend(verified.into_iter().flatten());
        pending = Some((chunk, dms));
    }
    if let Some((c, d)) = pending {
        let t = std::time::Instant::now();
        pairwise.extend(verify_chunk(c, d));
        verify_s += t.elapsed().as_secs_f64();
    }
    log(&format!(
        "matched {} of {} candidate pairs (gpu/cpu match {match_s:.1}s, verify {verify_s:.1}s)",
        kept.len(),
        candidates.len()
    ));
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
    for o in &cfg.sfm_overrides {
        apply_sfm_override(&mut sfm_cfg, o).map_err(EurocError::Sfm)?;
    }
    let (features, pairwise) = match &cfg.import_colmap {
        Some(dir) => {
            let imported = import_colmap_export(dir, features.len())?;
            log(&format!(
                "import-colmap: {} verified pairs from {}",
                imported.1.len(),
                dir.display()
            ));
            imported
        }
        None => (features, pairwise),
    };
    let init_poses: Option<Vec<Option<visloc_core::geometry::Pose>>> = match &cfg.init_poses {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| EurocError::Sfm(format!("init-poses: {e}")))?;
            let mut poses = vec![None; features.len()];
            for line in text.lines() {
                let v: Vec<f64> = line
                    .split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect();
                if v.len() == 8 && (v[0] as usize) < poses.len() {
                    let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        v[1], v[2], v[3], v[4],
                    ));
                    poses[v[0] as usize] = Some(visloc_core::geometry::Pose::from_world_to_camera(
                        q,
                        Vector3::new(v[5], v[6], v[7]),
                    ));
                }
            }
            log(&format!(
                "init-poses: {} poses from {}",
                poses.iter().filter(|p| p.is_some()).count(),
                path.display()
            ));
            Some(poses)
        }
        None => None,
    };
    let result = if cfg.colmap_port_mapper {
        let (poses, tracks, mean_reprojection_px) =
            run_colmap_port(out_dir, &camera, width, height, &features, &pairwise, log)?;
        SfmOutcome {
            poses,
            tracks,
            mean_reprojection_px,
            refined_camera: None,
        }
    } else {
        let r = visloc_slam::incremental_sfm_with_initial_poses(
            &camera,
            &features,
            &pairwise,
            &sfm_cfg,
            init_poses.as_deref(),
        )
        .map_err(|e| EurocError::Sfm(e.to_string()))?;
        SfmOutcome {
            poses: r.poses,
            tracks: r.tracks,
            mean_reprojection_px: r.mean_reprojection_px,
            refined_camera: r.refined_camera,
        }
    };
    let registered = result.poses.iter().filter(|p| p.is_some()).count();
    if std::env::var_os("VISLOC_EUROC_DUMP_TRACKS").is_some() {
        // Diagnostic: one line per track, its observing image indices.
        let text: String = result
            .tracks
            .iter()
            .map(|t| {
                let ids: Vec<String> = t.observations.iter().map(|o| o.0.to_string()).collect();
                ids.join(" ")
                    + "
"
            })
            .collect();
        let _ = std::fs::write(out_dir.join("tracks.txt"), text);
    }
    // ATE against ground truth: nearest GT sample within 10 ms of each
    // registered frame, cam0 centre = p_WB + R_WB t_BS.
    let ate_rmse_m = {
        let t_bs = seq.cam0_calibration.t_body_sensor;
        let t_bs = Vector3::new(t_bs[(0, 3)], t_bs[(1, 3)], t_bs[(2, 3)]);
        let gt = &seq.ground_truth;
        let mut est_c = Vec::new();
        let mut gt_c = Vec::new();
        let mut frame_ids = Vec::new();
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
                frame_ids.push(i);
            }
        }
        // Associated centres for offline inspection / other aligners.
        let csv: String = std::iter::once(
            "frame,est_x,est_y,est_z,gt_x,gt_y,gt_z
"
            .to_string(),
        )
        .chain(est_c.iter().zip(&gt_c).zip(&frame_ids).map(|((e, g), &i)| {
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
