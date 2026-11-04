//! visloc-rs's photo SfM in the browser: photos to camera poses and a sparse
//! point cloud with the pieces `gsplat_photos --cpu` runs, on data in memory:
//! VLFeat-compatible SIFT, ratio-tested mutual nearest-neighbour matching,
//! two-view geometric verification, and the COLMAP-port incremental mapper
//! ([`DatabaseCache::from_memory`], then `pipeline::run`).
//!
//! A page drives a `Session` from a Web Worker: `add_photo` per photo,
//! `plan_pairs` and `match_some` in steps (so it can report progress), then
//! `reconstruct`. With WebGPU (`init_gpu`), SIFT and matching run on the GPU
//! (`visloc-sift-gpu`, the ports `gsplat_photos` runs); verification and the
//! mapper stay on the CPU. The same functions build natively, where the tests
//! run them on a synthetic scene.

use image::RgbImage;
use nalgebra::{Point2, Vector3};
use visloc_core::types::Camera;
use visloc_slam::colmap_incremental::{
    pipeline, DatabaseCache, MemoryPair, PipelineOptions, SensorT,
};
use visloc_vision::features::sift::{
    extract_sift, GrayImage, SiftConfig, SiftKeypoint, SiftNormalization,
};
use visloc_vision::features::FeatureSet;
use visloc_vision::matching::{BruteForceMatcher, DescriptorMatch};
use visloc_vision::two_view::{
    ConfigurationType, TwoViewCorrespondence, TwoViewGeometryOptions, TwoViewGeometryVerifier,
};

/// One photo's features: keypoints and their SIFT descriptors (the VLFeat
/// bytes over 512, as `gsplat_photos` matches them), and each keypoint's
/// colour in the photo.
pub struct Photo {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub features: FeatureSet,
    pub colours: Vec<[u8; 3]>,
}

/// Decode a JPEG or PNG and shrink it so that its long side is at most
/// `max_size` pixels.
pub fn decode(bytes: &[u8], max_size: u32) -> Result<RgbImage, String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| e.to_string())?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let long = w.max(h);
    if long <= max_size {
        return Ok(img);
    }
    let s = max_size as f64 / long as f64;
    let nw = ((w as f64 * s).round() as u32).max(1);
    let nh = ((h as f64 * s).round() as u32).max(1);
    Ok(image::imageops::resize(
        &img,
        nw,
        nh,
        image::imageops::FilterType::Triangle,
    ))
}

/// `gsplat_photos`' SIFT: VLFeat-compatible detector and descriptor, RootSIFT
/// normalisation, up to `max_keypoints`.
pub fn sift_config(max_keypoints: usize) -> SiftConfig {
    SiftConfig {
        max_keypoints,
        normalization: SiftNormalization::L1Root,
        descriptor_magnification: 3.0,
        max_orientations: 2,
        prefer_larger_scale: true,
        vlfeat_compatible_detector: true,
        vlfeat_compatible_descriptor: true,
        vlfeat_bilinear_orientations: true,
        ..SiftConfig::default()
    }
}

/// The photo in grey, in [0, 1] as the detector's thresholds assume.
pub fn gray(img: &RgbImage) -> Vec<f32> {
    img.pixels()
        .map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0)
        .collect()
}

/// SIFT on a photo, on the CPU.
pub fn extract(name: &str, img: &RgbImage, cfg: &SiftConfig) -> Result<Photo, String> {
    let (w, h) = img.dimensions();
    let pixels = gray(img);
    let (kps, desc) = extract_sift(
        &GrayImage {
            width: w as usize,
            height: h as usize,
            pixels: &pixels,
        },
        cfg,
    )
    .map_err(|e| e.to_string())?;
    Ok(photo(name, img, &kps, &desc))
}

/// A photo from its SIFT keypoints and VLFeat byte descriptors (0..=255, as
/// both the CPU and the GPU extractor give them).
pub fn photo(name: &str, img: &RgbImage, kps: &[SiftKeypoint], desc: &[Vec<f32>]) -> Photo {
    let (w, h) = img.dimensions();
    let keypoints: Vec<Point2<f64>> = kps.iter().map(|k| Point2::new(k.x, k.y)).collect();
    let colours = keypoints
        .iter()
        .map(|p| {
            let x = (p.x.round().max(0.0) as u32).min(w - 1);
            let y = (p.y.round().max(0.0) as u32).min(h - 1);
            img.get_pixel(x, y).0
        })
        .collect();
    let descriptors = desc
        .iter()
        .map(|d| d.iter().map(|&v| (v as u8) as f32 / 512.0).collect())
        .collect();
    Photo {
        name: name.to_string(),
        width: w,
        height: h,
        features: FeatureSet {
            keypoints,
            descriptors,
        },
        colours,
    }
}

/// The image pairs to match: all of them up to `exhaustive_max` photos, else
/// each photo with the next `window` (photos in capture order).
pub fn candidate_pairs(n: usize, window: usize, exhaustive_max: usize) -> Vec<(usize, usize)> {
    let reach = if n <= exhaustive_max {
        n
    } else {
        window.max(1)
    };
    (0..n)
        .flat_map(|i| (i + 1..(i + 1 + reach).min(n)).map(move |j| (i, j)))
        .collect()
}

/// Match two photos (ratio 0.8, mutual) and keep the matches a two-view
/// geometry explains (`gsplat_photos`' `verify_pair`, 4 px), as point2D
/// index pairs; `None` below `min_matches`.
pub fn verify_pair(
    camera: &Camera,
    a: &FeatureSet,
    b: &FeatureSet,
    min_matches: usize,
) -> Option<Vec<(u32, u32)>> {
    let dm = BruteForceMatcher { ratio: Some(0.8) }
        .match_descriptors_cross_checked(&a.descriptors, &b.descriptors);
    verify_matches(camera, a, b, &dm, min_matches)
}

/// The matches `dm` (from `a` to `b`) that a two-view geometry explains
/// (`gsplat_photos`' `verify_pair`, 4 px), as point2D index pairs; `None`
/// below `min_matches`.
pub fn verify_matches(
    camera: &Camera,
    a: &FeatureSet,
    b: &FeatureSet,
    dm: &[DescriptorMatch],
    min_matches: usize,
) -> Option<Vec<(u32, u32)>> {
    if dm.len() < min_matches {
        return None;
    }
    let corrs: Vec<TwoViewCorrespondence> = dm
        .iter()
        .map(|m| TwoViewCorrespondence::new(a.keypoints[m.query_index], b.keypoints[m.train_index]))
        .collect();
    let report = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0))
        .classify(&corrs, camera);
    let keep = matches!(
        report.config,
        ConfigurationType::Calibrated
            | ConfigurationType::Uncalibrated
            | ConfigurationType::Planar
            | ConfigurationType::Multiple
            | ConfigurationType::Panoramic
            | ConfigurationType::PlanarOrPanoramic
    );
    if !keep || report.inliers.len() < min_matches {
        return None;
    }
    Some(
        report
            .inliers
            .iter()
            .map(|&i| (dm[i].query_index as u32, dm[i].train_index as u32))
            .collect(),
    )
}

/// A registered photo: its camera centre and world-to-camera rotation (rows).
pub struct Placed {
    pub index: usize,
    pub centre: [f64; 3],
    pub rotation: [[f64; 3]; 3],
}

/// The largest model the mapper built.
pub struct Model {
    pub cameras: Vec<Placed>,
    pub points: Vec<([f32; 3], [u8; 3])>,
    /// The mapper's refined pinhole intrinsics (fx, fy, cx, cy).
    pub intrinsics: (f64, f64, f64, f64),
}

/// The COLMAP-port mapper on the photos and their verified pairs, with
/// `gsplat_photos`' options (`run_colmap_port`): COLMAP's absolute-pose
/// thresholds for single cameras, the structure-less registration fallback,
/// the focal length and principal point refined, and a converged final
/// global bundle adjustment.
pub fn reconstruct(
    camera: &Camera,
    photos: &[Photo],
    pairs: &[MemoryPair],
) -> Result<Model, String> {
    // Only photos in a verified pair (the correspondence graph has no node
    // for an isolated one), renumbered.
    let mut used = vec![false; photos.len()];
    for (i, j, _) in pairs {
        used[*i] = true;
        used[*j] = true;
    }
    let order: Vec<usize> = (0..photos.len()).filter(|&i| used[i]).collect();
    if order.len() < 2 {
        return Err("no two photos share enough verified matches".into());
    }
    let mut local = vec![usize::MAX; photos.len()];
    for (k, &i) in order.iter().enumerate() {
        local[i] = k;
    }
    let mut port_camera = camera.clone();
    port_camera.id = 1;
    let images = order
        .iter()
        .map(|&i| (photos[i].name.clone(), photos[i].features.keypoints.clone()))
        .collect();
    let local_pairs: Vec<MemoryPair> = pairs
        .iter()
        .map(|(i, j, m)| (local[*i], local[*j], m.clone()))
        .collect();
    let db =
        DatabaseCache::from_memory(port_camera, images, &local_pairs).map_err(|e| e.to_string())?;
    let (w, h) = (camera.width, camera.height);
    let mut options = PipelineOptions::default();
    for kv in [
        format!(
            "abs_pose_max_error={}",
            (12.0 * w.max(h) as f64 / 1600.0).clamp(4.0, 12.0)
        ),
        "abs_pose_min_num_inliers=30".into(),
        "structure_less_registration_fallback=1".into(),
        "structure_less_min_verified=10".into(),
        "ba_refine_focal_length=1".into(),
        "ba_refine_principal_point=1".into(),
        "ba_global_final_converge_min_frames=0".into(),
    ] {
        options
            .apply_override(&kv)
            .map_err(|e| format!("mapper option {kv}: {e}"))?;
    }
    let run = pipeline::run(&options, &db);
    let best = run
        .models
        .iter()
        .max_by_key(|m| m.reconstruction.num_reg_images())
        .ok_or("the mapper built no model")?;
    let rec = &best.reconstruction;
    let mut cameras = Vec::new();
    for (k, &i) in order.iter().enumerate() {
        let id = k as u64;
        if !rec.exists_image(id) || !rec.is_image_registered(id) {
            continue;
        }
        let image = rec.image(id);
        let frame = rec.frame(image.frame_id);
        let rig = rec.rig(frame.rig_id());
        let pose = frame.sensor_from_world(rig, SensorT::camera(image.camera_id));
        let r = pose.rotation.to_rotation_matrix();
        let c: Vector3<f64> = -(r.transpose() * pose.translation);
        let m = r.matrix();
        cameras.push(Placed {
            index: i,
            centre: [c.x, c.y, c.z],
            rotation: [
                [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
                [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
                [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
            ],
        });
    }
    let points = rec
        .point3d_ids()
        .into_iter()
        .map(|id| {
            let p = rec.point3d(id);
            // the mean colour of the point's keypoints
            let mut sum = [0u32; 3];
            for t in &p.track {
                let c = photos[order[t.image_id as usize]].colours[t.point2d_idx];
                for (s, v) in sum.iter_mut().zip(c) {
                    *s += v as u32;
                }
            }
            let n = p.track.len().max(1) as u32;
            (
                [p.xyz.x as f32, p.xyz.y as f32, p.xyz.z as f32],
                [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8],
            )
        })
        .collect();
    let intrinsics = rec.camera(1).intrinsics().unwrap_or((0.0, 0.0, 0.0, 0.0));
    Ok(Model {
        cameras,
        points,
        intrinsics,
    })
}

/// The model as JSON for the page: `{intrinsics: [fx, fy, cx, cy], width,
/// height, cameras: [{index, name, centre, rotation}], points: [x, y, z, ...],
/// colours: [r, g, b, ...]}`.
pub fn model_json(model: &Model, photos: &[Photo]) -> String {
    let (fx, fy, cx, cy) = model.intrinsics;
    let cameras: Vec<serde_json::Value> = model
        .cameras
        .iter()
        .map(|c| {
            serde_json::json!({
                "index": c.index,
                "name": photos[c.index].name,
                "centre": c.centre,
                "rotation": c.rotation,
            })
        })
        .collect();
    let points: Vec<f32> = model.points.iter().flat_map(|p| p.0).collect();
    let colours: Vec<u8> = model.points.iter().flat_map(|p| p.1).collect();
    serde_json::json!({
        "intrinsics": [fx, fy, cx, cy],
        "width": photos.first().map_or(0, |p| p.width),
        "height": photos.first().map_or(0, |p| p.height),
        "cameras": cameras,
        "points": points,
        "colours": colours,
    })
    .to_string()
}

#[cfg(target_arch = "wasm32")]
mod web {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use visloc_sift_gpu::{FeatureBank, GpuContext, GpuMatcher, VlfeatSiftGpu};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::future_to_promise;

    #[wasm_bindgen(start)]
    fn start() {
        console_error_panic_hook::set_once();
    }

    fn err(message: impl std::fmt::Display) -> JsValue {
        JsValue::from_str(&message.to_string())
    }

    /// SIFT and matching on WebGPU.
    struct Gpu {
        sift: VlfeatSiftGpu,
        matcher: GpuMatcher,
        /// Every photo's descriptors, uploaded by `plan_pairs`.
        bank: Option<FeatureBank>,
    }

    struct State {
        sift: SiftConfig,
        /// The focal length at the working size, or 0 for the prior of 1.2
        /// times the long side.
        focal_px: f64,
        min_matches: usize,
        photos: Vec<Photo>,
        todo: Vec<(usize, usize)>,
        verified: Vec<MemoryPair>,
        /// Taken out of the state while a GPU call is awaited.
        gpu: Option<Gpu>,
    }

    impl State {
        fn camera(&self) -> Camera {
            let (w, h) = self.photos.first().map_or((1, 1), |p| (p.width, p.height));
            let f = if self.focal_px > 0.0 {
                self.focal_px
            } else {
                1.2 * w.max(h) as f64
            };
            Camera::pinhole(1, w, h, f, f, w as f64 / 2.0, h as f64 / 2.0)
        }
    }

    /// One reconstruction, driven step by step from a Web Worker. The methods
    /// that return promises must be awaited one at a time.
    #[wasm_bindgen]
    pub struct Session {
        state: Rc<RefCell<State>>,
    }

    #[wasm_bindgen]
    impl Session {
        #[wasm_bindgen(constructor)]
        pub fn new(max_keypoints: u32, focal_px: f64) -> Session {
            Session {
                state: Rc::new(RefCell::new(State {
                    sift: sift_config(max_keypoints as usize),
                    focal_px,
                    min_matches: 15,
                    photos: Vec::new(),
                    todo: Vec::new(),
                    verified: Vec::new(),
                    gpu: None,
                })),
            }
        }

        /// Run SIFT and matching on WebGPU from now on. Resolves to the
        /// adapter's name (browsers may hide it: empty); rejects (no WebGPU,
        /// no adapter), leaving the CPU.
        pub fn init_gpu(&self) -> js_sys::Promise {
            let state = Rc::clone(&self.state);
            future_to_promise(async move {
                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::BROWSER_WEBGPU,
                    ..wgpu::InstanceDescriptor::new_without_display_handle()
                });
                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                        apply_limit_buckets: false,
                    })
                    .await
                    .map_err(|e| err(format!("no WebGPU adapter: {e}")))?;
                let ctx = GpuContext::from_adapter(instance, &adapter)
                    .await
                    .map_err(err)?;
                let name = ctx.adapter_info.name.clone();
                let matcher = GpuMatcher::new(&ctx);
                let sift = VlfeatSiftGpu::new(ctx);
                state.borrow_mut().gpu = Some(Gpu {
                    sift,
                    matcher,
                    bank: None,
                });
                Ok(JsValue::from_str(&name))
            })
        }

        /// Extract a photo's features, on the GPU after `init_gpu` (on the
        /// CPU without it, or should the GPU fail); resolves to its keypoint
        /// count. The page decodes and shrinks the photo (the browser's
        /// decoder, which also applies its EXIF orientation) and passes its
        /// RGBA pixels.
        pub fn add_photo(
            &self,
            name: String,
            width: u32,
            height: u32,
            rgba: Vec<u8>,
        ) -> js_sys::Promise {
            let state = Rc::clone(&self.state);
            future_to_promise(async move {
                let (cfg, first) = {
                    let s = state.borrow();
                    let first = s.photos.first().map(|p| (p.width, p.height));
                    (s.sift.clone(), first)
                };
                let rgb: Vec<u8> = rgba
                    .chunks_exact(4)
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect();
                let img = RgbImage::from_raw(width, height, rgb)
                    .ok_or_else(|| err(format!("{name}: {width}x{height} needs more pixels")))?;
                if let Some((w, h)) = first {
                    if (w, h) != img.dimensions() {
                        return Err(err(format!(
                            "{name} is {}x{}, the first photo {w}x{h}: one camera, one size",
                            img.width(),
                            img.height(),
                        )));
                    }
                }
                let gpu = state.borrow_mut().gpu.take();
                let (found, gpu) = match gpu {
                    Some(mut gpu) => {
                        let (w, h) = img.dimensions();
                        let pixels = gray(&img);
                        let image = GrayImage {
                            width: w as usize,
                            height: h as usize,
                            pixels: &pixels,
                        };
                        let found = match gpu.sift.extract_async(&image, &cfg).await {
                            Ok((kps, desc)) => Ok(photo(&name, &img, &kps, &desc)),
                            Err(_) => extract(&name, &img, &cfg),
                        };
                        (found, Some(gpu))
                    }
                    None => (extract(&name, &img, &cfg), None),
                };
                let mut s = state.borrow_mut();
                s.gpu = gpu;
                let p = found.map_err(err)?;
                let n = p.features.keypoints.len() as u32;
                s.photos.push(p);
                Ok(JsValue::from(n))
            })
        }

        /// List the pairs to match (see [`candidate_pairs`]) and, on the
        /// GPU, upload every photo's descriptors; returns the pair count.
        pub fn plan_pairs(&self, window: u32, exhaustive_max: u32) -> u32 {
            let mut s = self.state.borrow_mut();
            let s = &mut *s;
            s.todo = candidate_pairs(s.photos.len(), window as usize, exhaustive_max as usize);
            s.todo.reverse();
            s.verified.clear();
            if let Some(gpu) = s.gpu.as_mut() {
                let sets: Vec<&[Vec<f32>]> = s
                    .photos
                    .iter()
                    .map(|p| p.features.descriptors.as_slice())
                    .collect();
                // f32, not u8: in browsers, `dot4U8Packed` can be slower
                // than f32 dot products (2.4x on an AMD RDNA 2 iGPU)
                gpu.bank = FeatureBank::upload(gpu.sift.context(), &sets).ok();
            }
            s.todo.len() as u32
        }

        /// Match up to `count` planned pairs (on the GPU, in one dispatch:
        /// keep it short of the OS's GPU watchdog) and verify them; resolves
        /// to how many remain.
        pub fn match_some(&self, count: u32) -> js_sys::Promise {
            let state = Rc::clone(&self.state);
            future_to_promise(async move {
                let (batch, gpu) = {
                    let mut s = state.borrow_mut();
                    let at = s.todo.len() - (count as usize).min(s.todo.len());
                    let mut batch = s.todo.split_off(at);
                    batch.reverse();
                    (batch, s.gpu.take())
                };
                let gpu_matches = match &gpu {
                    Some(Gpu {
                        sift,
                        matcher,
                        bank: Some(bank),
                    }) => Some(
                        matcher
                            .match_pairs_async(sift.context(), bank, &batch, Some(0.8), true)
                            .await,
                    ),
                    _ => None,
                };
                let mut s = state.borrow_mut();
                s.gpu = gpu;
                let camera = s.camera();
                let State {
                    photos,
                    verified,
                    min_matches,
                    todo,
                    ..
                } = &mut *s;
                for (k, &(i, j)) in batch.iter().enumerate() {
                    let (a, b) = (&photos[i].features, &photos[j].features);
                    let cpu;
                    let dm = match &gpu_matches {
                        Some(all) => &all[k],
                        None => {
                            cpu = BruteForceMatcher { ratio: Some(0.8) }
                                .match_descriptors_cross_checked(&a.descriptors, &b.descriptors);
                            &cpu
                        }
                    };
                    if let Some(m) = verify_matches(&camera, a, b, dm, *min_matches) {
                        verified.push((i, j, m));
                    }
                }
                Ok(JsValue::from(todo.len() as u32))
            })
        }

        /// The pairs verified so far.
        pub fn verified_pairs(&self) -> u32 {
            self.state.borrow().verified.len() as u32
        }

        /// Run the mapper; returns the model as JSON (see [`model_json`]).
        pub fn reconstruct(&self) -> Result<String, JsValue> {
            let s = self.state.borrow();
            let model = reconstruct(&s.camera(), &s.photos, &s.verified).map_err(err)?;
            Ok(model_json(&model, &s.photos))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::Session;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Rotation3};

    /// An arc of 8 cameras looking at 400 points in a box, each point's
    /// descriptor the same in every view (a little noise), its keypoints the
    /// exact projections: matching, verification and the mapper must register
    /// every camera, where they were up to a similarity.
    #[test]
    fn synthetic_arc_is_reconstructed() {
        let (w, h, f) = (640u32, 480u32, 500.0);
        let camera = Camera::pinhole(1, w, h, f, f, 320.0, 240.0);
        let mut seed = 7u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f64 / u32::MAX as f64
        };
        let points: Vec<Vector3<f64>> = (0..400)
            .map(|_| Vector3::new(rnd() * 4.0 - 2.0, rnd() * 3.0 - 1.5, rnd() * 4.0 - 2.0))
            .collect();
        let descriptors: Vec<Vec<f32>> = (0..points.len())
            .map(|_| (0..128).map(|_| rnd() as f32).collect())
            .collect();
        let n_cams = 8;
        let mut truth = Vec::new();
        let mut photos = Vec::new();
        for c in 0..n_cams {
            let a = c as f64 * 0.35; // a 140 degree arc, 8 m out
            let centre = Vector3::new(8.0 * a.sin(), 0.3, -8.0 * a.cos());
            let forward = (-centre).normalize();
            let right = forward.cross(&Vector3::new(0.0, -1.0, 0.0)).normalize();
            let down = forward.cross(&right);
            let r = Rotation3::from_matrix_unchecked(Matrix3::from_rows(&[
                right.transpose(),
                down.transpose(),
                forward.transpose(),
            ]));
            truth.push(centre);
            let mut keypoints = Vec::new();
            let mut desc = Vec::new();
            for (p, d) in points.iter().zip(&descriptors) {
                let q = r * (p - centre);
                let (u, v) = (f * q.x / q.z + 320.0, f * q.y / q.z + 240.0);
                if q.z > 0.1 && (0.0..w as f64).contains(&u) && (0.0..h as f64).contains(&v) {
                    keypoints.push(Point2::new(u, v));
                    desc.push(d.iter().map(|x| x + 0.01 * rnd() as f32).collect());
                }
            }
            let n = keypoints.len();
            photos.push(Photo {
                name: format!("cam{c}.png"),
                width: w,
                height: h,
                features: FeatureSet {
                    keypoints,
                    descriptors: desc,
                },
                colours: vec![[200, 100, 50]; n],
            });
        }
        let pairs: Vec<MemoryPair> = candidate_pairs(n_cams, 3, 0)
            .into_iter()
            .filter_map(|(i, j)| {
                verify_pair(&camera, &photos[i].features, &photos[j].features, 15)
                    .map(|m| (i, j, m))
            })
            .collect();
        assert!(pairs.len() >= n_cams - 1, "{} verified pairs", pairs.len());
        let model = reconstruct(&camera, &photos, &pairs).expect("a model");
        assert_eq!(model.cameras.len(), n_cams, "every camera registered");
        assert!(model.points.len() > 300, "{} points", model.points.len());
        // the centres match up to a similarity: the distances to the first
        // camera, scaled by the farthest one's
        let got: Vec<Vector3<f64>> = model
            .cameras
            .iter()
            .map(|c| Vector3::from(c.centre))
            .collect();
        let d_true = |k: usize| (truth[k] - truth[0]).norm();
        let d_got = |k: usize| (got[k] - got[0]).norm();
        let s = d_got(n_cams - 1) / d_true(n_cams - 1);
        for k in 1..n_cams {
            let rel = (d_got(k) / s - d_true(k)).abs() / d_true(k);
            assert!(
                rel < 0.01,
                "camera {k}: distance off by {:.2}%",
                100.0 * rel
            );
        }
        let json = model_json(&model, &photos);
        assert!(json.contains("\"cameras\"") && json.contains("cam7.png"));
    }

    #[test]
    fn candidate_pairs_are_exhaustive_then_windowed() {
        assert_eq!(candidate_pairs(4, 1, 10).len(), 6);
        assert_eq!(
            candidate_pairs(5, 2, 3),
            vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
        );
    }
}
