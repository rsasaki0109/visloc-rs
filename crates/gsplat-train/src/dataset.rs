//! A COLMAP scene with its images, split into train and eval views.
//!
//! Expected layout (what `colmap image_undistorter` writes, plus a text model):
//!
//! ```text
//! <root>/images/<name>          one image per registered view
//! <root>/sparse/0/cameras.txt   (or sparse/, text or binary)
//! ```
//!
//! Image names come from `images.txt` or `images.bin` (the visloc map keeps
//! only ids). Views are sorted by image name and every `eval_every`-th one
//! (index 0, N, 2N, ...) is held out, matching brush's and the Inria code's
//! split so scores are comparable. Cameras must be pinhole: undistort first.
//!
//! The exception is COLMAP's `EQUIRECTANGULAR` camera (model 17, parameters
//! `width, height`), as COLMAP 4.1 and other SfM tools write it for 360°
//! panoramas. Each such image is cut into the six 90° cube faces of
//! [`crate::equirect`], the ones `gsplat_photos --equirect` trains on, and
//! the faces become the views: a face's pose is the panorama's with the
//! face's fixed rotation behind it, and its image is written as a PNG under
//! [`EquirectOptions::faces_dir`]. A panorama's faces are held out together.
//! Faces that see fewer than [`EquirectOptions::min_face_points`] of the
//! model's points (sky, or a camera mount) are left out, as in
//! `gsplat_photos`. An alpha channel in a panorama is cut with it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use visloc_core::types::{Camera, CameraModel, VisualMap};
use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_core::colmap_scene::{
    camera_view_from, read_visual_map, scene_from_visual_map, seed_scene_from_landmarks,
    ColmapSceneError, DEFAULT_SEED_LOG_SCALE,
};
use visloc_gsplat_core::gaussian::Scene;

use crate::equirect;

/// Errors from loading a dataset.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("no COLMAP model under {0} (looked in sparse/0 and sparse)")]
    NoModel(PathBuf),
    #[error("reading {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("image id {0} has no name in images.txt")]
    UnnamedImage(u64),
    #[error(transparent)]
    Colmap(#[from] ColmapSceneError),
    #[error("image {path}: {source}")]
    Image {
        path: PathBuf,
        source: image::ImageError,
    },
    #[error("image {path} is {got_w}x{got_h}, camera expects {want_w}x{want_h}")]
    SizeMismatch {
        path: PathBuf,
        got_w: u32,
        got_h: u32,
        want_w: u32,
        want_h: u32,
    },
}

/// One posed image.
#[derive(Debug, Clone)]
pub struct View {
    pub name: String,
    pub camera: CameraView,
    pub image_path: PathBuf,
}

/// A loaded dataset: SfM points as the initial scene, plus the view split.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Seed gaussians from the COLMAP points (isotropic, default log-scale).
    pub init: Scene,
    pub train: Vec<View>,
    pub eval: Vec<View>,
}

/// How the panoramas of an `EQUIRECTANGULAR` model are cut into views (see
/// the module docs).
#[derive(Debug, Clone)]
pub struct EquirectOptions {
    /// At most this many pixels square per face. A panorama is scaled down
    /// first if a quarter of its width is more (the default is
    /// `PhotoSfmConfig::max_size`'s, so the two paths cut the same faces).
    pub max_face_size: u32,
    /// Faces that see fewer model points than this are left out (the default
    /// is `PhotoSfmConfig::min_matches`'s).
    pub min_face_points: usize,
    /// Where the face images are written; `<root>/equirect_faces` if `None`.
    pub faces_dir: Option<PathBuf>,
}

impl Default for EquirectOptions {
    fn default() -> Self {
        Self {
            max_face_size: 1600,
            min_face_points: 15,
            faces_dir: None,
        }
    }
}

/// The model directory of `root`: `sparse/0`, else `sparse`.
pub fn find_model_dir(root: impl AsRef<Path>) -> Result<PathBuf, DatasetError> {
    let root = root.as_ref();
    [root.join("sparse").join("0"), root.join("sparse")]
        .into_iter()
        .find(|d| d.join("cameras.txt").exists() || d.join("cameras.bin").exists())
        .ok_or_else(|| DatasetError::NoModel(root.to_path_buf()))
}

/// Load `<root>` (see module docs). `eval_every = None` keeps every view for
/// training.
pub fn load_colmap_dataset(
    root: impl AsRef<Path>,
    eval_every: Option<usize>,
) -> Result<Dataset, DatasetError> {
    load_colmap_dataset_with(root, eval_every, &EquirectOptions::default(), &mut |_| {})
}

/// [`load_colmap_dataset`] with the options for cutting panoramas, and a
/// `log` for what that does.
pub fn load_colmap_dataset_with(
    root: impl AsRef<Path>,
    eval_every: Option<usize>,
    equirect: &EquirectOptions,
    log: &mut dyn FnMut(&str),
) -> Result<Dataset, DatasetError> {
    let root = root.as_ref();
    let model_dir = find_model_dir(root)?;
    let map = read_visual_map(&model_dir).map_err(ColmapSceneError::from)?;
    let names = read_image_names(&model_dir)?;
    if map.keyframes.values().any(|kf| {
        map.cameras
            .get(&kf.frame.camera_id)
            .is_some_and(|c| c.model == CameraModel::Equirectangular)
    }) {
        return load_equirect_dataset(root, &map, &names, eval_every, equirect, log);
    }
    let colmap = scene_from_visual_map(&map, DEFAULT_SEED_LOG_SCALE)?;

    let mut views = Vec::with_capacity(colmap.views.len());
    for (id, camera) in colmap.image_ids.iter().zip(colmap.views) {
        let name = names
            .get(id)
            .ok_or(DatasetError::UnnamedImage(*id))?
            .clone();
        views.push(View {
            image_path: root.join("images").join(&name),
            name,
            camera,
        });
    }
    views.sort_by(|a, b| a.name.cmp(&b.name));

    let mut train = Vec::new();
    let mut eval = Vec::new();
    for (i, v) in views.into_iter().enumerate() {
        match eval_every {
            Some(n) if n > 0 && i % n == 0 => eval.push(v),
            _ => train.push(v),
        }
    }
    Ok(Dataset {
        init: colmap.scene,
        train,
        eval,
    })
}

/// The views of a model with `EQUIRECTANGULAR` cameras (see the module docs):
/// every panorama's kept cube faces, and any pinhole image as itself.
fn load_equirect_dataset(
    root: &Path,
    map: &VisualMap,
    names: &HashMap<u64, String>,
    eval_every: Option<usize>,
    opts: &EquirectOptions,
    log: &mut dyn FnMut(&str),
) -> Result<Dataset, DatasetError> {
    let init = seed_scene_from_landmarks(map, DEFAULT_SEED_LOG_SCALE)?;
    // Registered images by name, as for pinhole models.
    let mut registered = Vec::new();
    for (&id, kf) in &map.keyframes {
        if let (Some(pose), Some(camera)) = (&kf.frame.pose, map.cameras.get(&kf.frame.camera_id)) {
            let name = names.get(&id).ok_or(DatasetError::UnnamedImage(id))?;
            registered.push((name, id, kf, pose, camera));
        }
    }
    registered.sort_by(|a, b| a.0.cmp(b.0));
    if registered.is_empty() {
        return Err(ColmapSceneError::NoImages.into());
    }
    // Without any observation (a model exported without POINTS2D) there is
    // nothing to tell the faces apart by: keep them all.
    let judge_faces = registered
        .iter()
        .any(|(_, _, kf, _, _)| !kf.observations.is_empty());
    if !judge_faces {
        log("equirect: the model has no observations; keeping every cube face");
    }
    let faces_dir = opts
        .faces_dir
        .clone()
        .unwrap_or_else(|| root.join("equirect_faces"));
    let axes = equirect::face_axes();
    let rotations = equirect::cube_face_rotations();

    let (mut train, mut eval) = (Vec::new(), Vec::new());
    let (mut posed, mut kept) = (0usize, 0usize);
    // `k`: the rank of the image among those that gave a view.
    let mut k = 0usize;
    for (name, id, kf, pose, camera) in registered {
        let mut views = Vec::new();
        if camera.model != CameraModel::Equirectangular {
            views.push(View {
                image_path: root.join("images").join(name),
                name: name.clone(),
                camera: camera_view_from(camera, pose)?,
            });
        } else {
            let mut support = [0usize; 6];
            for o in &kf.observations {
                if let Some(ray) = camera.unit_ray_from_pixel(&o.xy) {
                    support[equirect::nearest_face(&axes, &ray)] += 1;
                }
            }
            let keep: Vec<bool> = support
                .iter()
                .map(|&n| !judge_faces || n >= opts.min_face_points)
                .collect();
            posed += 6;
            kept += keep.iter().filter(|&&keep| keep).count();
            if keep.contains(&true) {
                let (pw, ph, size) = scaled_size(camera, opts.max_face_size);
                let faces = cut_faces(
                    &root.join("images").join(name),
                    camera,
                    (pw, ph, size),
                    &keep,
                )?;
                let stem = Path::new(name).file_stem().unwrap_or_default();
                let half = size as f64 / 2.0;
                let face_camera = Camera::pinhole(1, size, size, half, half, half, half);
                std::fs::create_dir_all(&faces_dir).map_err(|source| DatasetError::Io {
                    path: faces_dir.clone(),
                    source,
                })?;
                for (face, image) in faces {
                    let (face_name, rotation) = &rotations[face];
                    let name = format!("{id:06}_{}_{face_name}.png", stem.to_string_lossy());
                    let image_path = faces_dir.join(&name);
                    image
                        .save(&image_path)
                        .map_err(|source| DatasetError::Image {
                            path: image_path.clone(),
                            source,
                        })?;
                    let pose = equirect::face_pose(pose, rotation);
                    views.push(View {
                        name,
                        camera: camera_view_from(&face_camera, &pose)?,
                        image_path,
                    });
                }
            }
        }
        if views.is_empty() {
            continue;
        }
        let held_out = matches!(eval_every, Some(n) if n > 0 && k.is_multiple_of(n));
        k += 1;
        let split = if held_out { &mut eval } else { &mut train };
        split.extend(views);
    }
    if posed > 0 {
        log(&format!(
            "equirect: {kept} of {posed} posed faces see at least {} SfM points and are kept \
             for training",
            opts.min_face_points
        ));
    }
    Ok(Dataset { init, train, eval })
}

/// `(panorama width, height, face size)` for `camera`: the panorama is
/// scaled down first so that a face, a quarter of its width, is at most
/// `max_face_size` (`gsplat_photos`'s rule).
fn scaled_size(camera: &Camera, max_face_size: u32) -> (u32, u32, u32) {
    let (w, h) = (camera.width, camera.height);
    let s = (max_face_size as f64 / (w / 4).max(1) as f64).min(1.0);
    let (pw, ph) = if s < 1.0 {
        (
            ((w as f64 * s).round() as u32).max(1),
            ((h as f64 * s).round() as u32).max(1),
        )
    } else {
        (w, h)
    };
    (pw, ph, (pw / 4).max(1))
}

/// The cube faces that `keep` selects, as `(index into
/// [`equirect::cube_face_rotations`], image)`, cut from the panorama file at
/// `path`, which must be `camera`'s size. The panorama is scaled to
/// `(pw, ph)` first; an alpha channel is cut with it.
fn cut_faces(
    path: &Path,
    camera: &Camera,
    (pw, ph, size): (u32, u32, u32),
    keep: &[bool],
) -> Result<Vec<(usize, image::DynamicImage)>, DatasetError> {
    let img = image::open(path).map_err(|source| DatasetError::Image {
        path: path.to_path_buf(),
        source,
    })?;
    if (img.width(), img.height()) != (camera.width, camera.height) {
        return Err(DatasetError::SizeMismatch {
            path: path.to_path_buf(),
            got_w: img.width(),
            got_h: img.height(),
            want_w: camera.width,
            want_h: camera.height,
        });
    }
    use image::imageops::{resize, FilterType::Triangle};
    let rgb = img.to_rgb8();
    let rgb = if (pw, ph) == rgb.dimensions() {
        rgb
    } else {
        resize(&rgb, pw, ph, Triangle)
    };
    // The alpha plane goes through the same cut as a grey image.
    let alpha = img.color().has_alpha().then(|| {
        let rgba = img.to_rgba8();
        let plane = image::GrayImage::from_fn(rgba.width(), rgba.height(), |x, y| {
            image::Luma([rgba.get_pixel(x, y)[3]])
        });
        let plane = if (pw, ph) == plane.dimensions() {
            plane
        } else {
            resize(&plane, pw, ph, Triangle)
        };
        image::DynamicImage::ImageLuma8(plane).to_rgb8()
    });
    Ok(equirect::cube_face_rotations()
        .iter()
        .enumerate()
        .filter(|(k, _)| keep[*k])
        .map(|(k, (_, rotation))| {
            let face = equirect::cube_face(&rgb, rotation, size);
            let face = match &alpha {
                None => image::DynamicImage::ImageRgb8(face),
                Some(alpha) => {
                    let a = equirect::cube_face(alpha, rotation, size);
                    image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(
                        size,
                        size,
                        |x, y| {
                            let p = face.get_pixel(x, y).0;
                            image::Rgba([p[0], p[1], p[2], a.get_pixel(x, y).0[0]])
                        },
                    ))
                }
            };
            (k, face)
        })
        .collect())
}

/// `IMAGE_ID -> NAME` of the model in `model_dir`: from a COLMAP text
/// `images.txt` (pose lines are `ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`, each
/// followed by a points line), else from `images.bin`.
fn read_image_names(model_dir: &Path) -> Result<HashMap<u64, String>, DatasetError> {
    let path = &model_dir.join("images.txt");
    if !path.exists() {
        let path = model_dir.join("images.bin");
        let bytes = std::fs::read(&path).map_err(|source| DatasetError::Io {
            path: path.clone(),
            source,
        })?;
        return Ok(visloc_io::colmap::parse_images_bin_names(&bytes)
            .map_err(ColmapSceneError::from)?
            .into_iter()
            .collect());
    }
    let text = std::fs::read_to_string(path).map_err(|source| DatasetError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut names = HashMap::new();
    let mut pose_line = true;
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        if pose_line {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.len() >= 10 {
                if let Ok(id) = t[0].parse::<u64>() {
                    names.insert(id, t[9..].join(" "));
                }
            }
        }
        pose_line = !pose_line;
    }
    Ok(names)
}

/// Load a view's image as row-major RGB in `[0, 1]`, checking it matches the
/// camera's resolution.
pub fn load_view_rgb(view: &View) -> Result<Vec<[f32; 3]>, DatasetError> {
    Ok(load_view_rgb_mask(view)?.0)
}

/// Row-major RGB in `[0, 1]` and, if the image has alpha, the valid pixels.
pub type RgbAndMask = (Vec<[f32; 3]>, Option<Vec<bool>>);

/// [`load_view_rgb`] plus, when the image file has an alpha channel, which
/// pixels are valid (alpha >= 128). Images without alpha give `None`.
pub fn load_view_rgb_mask(view: &View) -> Result<RgbAndMask, DatasetError> {
    let img = image::open(&view.image_path).map_err(|source| DatasetError::Image {
        path: view.image_path.clone(),
        source,
    })?;
    let (w, h) = (img.width(), img.height());
    let (want_w, want_h) = (view.camera.camera.width, view.camera.camera.height);
    if (w, h) != (want_w, want_h) {
        return Err(DatasetError::SizeMismatch {
            path: view.image_path.clone(),
            got_w: w,
            got_h: h,
            want_w,
            want_h,
        });
    }
    let valid = img
        .color()
        .has_alpha()
        .then(|| img.to_rgba8().pixels().map(|p| p[3] >= 128).collect());
    let rgb = img
        .to_rgb8()
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();
    Ok((rgb, valid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equirect::{cube_face_rotations, face_axes};
    use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
    use std::f64::consts::PI;
    use visloc_core::types::Camera;

    /// Panoramas are 512 x 256: their cube faces 128 px square.
    const PANO_W: u32 = 512;
    const FACE: u32 = PANO_W / 4;

    /// A removed-on-drop scratch directory.
    struct Tmp(PathBuf);

    impl Tmp {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("visloc_dataset_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Tmp(dir)
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A smooth, non-symmetric brightness over the sphere of WORLD
    /// directions.
    fn shade(d: &Vector3<f64>) -> f64 {
        let d = d.normalize();
        128.0 + 60.0 * (3.0 * d.x).sin() + 40.0 * (2.0 * d.y + 1.0).cos() + 25.0 * (4.0 * d.z).sin()
    }

    /// A pose that is not special in any axis: no face's rotation commutes
    /// with it.
    fn pano_rotation() -> UnitQuaternion<f64> {
        UnitQuaternion::from_scaled_axis(Vector3::new(0.3, -0.5, 0.2) * 1.7)
    }

    /// The panorama a camera with world-to-camera rotation `rot` takes of a
    /// world painted by [`shade`]; `alpha` gives each pixel's alpha.
    fn panorama(
        rot: &UnitQuaternion<f64>,
        alpha: Option<&dyn Fn(u32, u32) -> u8>,
    ) -> image::DynamicImage {
        let (w, h) = (PANO_W, PANO_W / 2);
        let pixel = |x: u32, y: u32| {
            let lon = (x as f64 + 0.5) / w as f64 * 2.0 * PI - PI;
            let lat = PI / 2.0 - (y as f64 + 0.5) / h as f64 * PI;
            let d = Vector3::new(lat.cos() * lon.sin(), -lat.sin(), lat.cos() * lon.cos());
            let v = shade(&(rot.inverse() * d)).round() as u8;
            [v, v / 2, 255 - v]
        };
        match alpha {
            None => image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
                image::Rgb(pixel(x, y))
            })),
            Some(a) => image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(w, h, |x, y| {
                let [r, g, b] = pixel(x, y);
                image::Rgba([r, g, b, a(x, y)])
            })),
        }
    }

    struct Img {
        id: u32,
        name: String,
        rot: UnitQuaternion<f64>,
        t: Vector3<f64>,
        /// `(x, y, point id)`; -1 for a keypoint with no 3D point.
        obs: Vec<(f64, f64, i64)>,
    }

    struct Pt {
        id: u64,
        xyz: [f64; 3],
        rgb: [u8; 3],
    }

    /// A COLMAP model, writable as text or binary.
    struct Model {
        /// `(name, binary model id, width, height, params)`.
        camera: (&'static str, i32, u32, u32, Vec<f64>),
        images: Vec<Img>,
        points: Vec<Pt>,
    }

    impl Model {
        /// `n` equirectangular panoramas, 0.5 m apart, each seeing
        /// `per_face[k]` matched points around cube face `k`'s axis and
        /// `unmatched_left` keypoints with no 3D point around the left
        /// face's.
        fn equirect(n: usize, per_face: [usize; 6], unmatched_left: usize) -> Self {
            let cam = Camera::equirectangular(1, PANO_W, PANO_W / 2);
            let rotations = cube_face_rotations();
            let axes = face_axes();
            let (mut images, mut points) = (Vec::new(), Vec::new());
            for i in 0..n {
                let (rot, t) = (
                    pano_rotation(),
                    Vector3::new(0.4 + 0.5 * i as f64, -1.3, 2.1),
                );
                let centre = -(rot.inverse() * t);
                let pixel = |dir: Vector3<f64>| {
                    let p = cam.project(&Point3::from(dir)).unwrap();
                    (p.x, p.y)
                };
                // Rays within ~9 degrees of a face's axis, spread around it.
                let around = |k: usize, j: usize, of: usize| {
                    let a = 2.0 * PI * j as f64 / of.max(1) as f64;
                    let (x, y) = (rotations[k].1.column(0), rotations[k].1.column(1));
                    (axes[k] + 0.15 * (a.cos() * x + a.sin() * y)).normalize()
                };
                let mut obs = Vec::new();
                for (k, &count) in per_face.iter().enumerate() {
                    for j in 0..count {
                        let dir = around(k, j, count);
                        let id = points.len() as u64 + 1;
                        let w = centre + rot.inverse() * (dir * 3.0);
                        points.push(Pt {
                            id,
                            xyz: [w.x, w.y, w.z],
                            rgb: [10 + (id % 200) as u8, 200, 30],
                        });
                        let (x, y) = pixel(dir);
                        obs.push((x, y, id as i64));
                    }
                }
                for j in 0..unmatched_left {
                    let (x, y) = pixel(around(3, j, unmatched_left));
                    obs.push((x, y, -1));
                }
                images.push(Img {
                    // Ids run against the names, so sorting by id is not sorting by name.
                    id: (n - i) as u32,
                    name: format!("pano_{}.png", (b'a' + i as u8) as char),
                    rot,
                    t,
                    obs,
                });
            }
            Model {
                camera: (
                    "EQUIRECTANGULAR",
                    17,
                    PANO_W,
                    PANO_W / 2,
                    vec![PANO_W as f64, (PANO_W / 2) as f64],
                ),
                images,
                points,
            }
        }

        /// Three 64 x 48 pinhole images, named out of id order.
        fn pinhole() -> Self {
            let img = |id: u32, name: &str, angle: f64| Img {
                id,
                name: name.into(),
                rot: UnitQuaternion::from_scaled_axis(Vector3::new(0.1, angle, -0.2)),
                t: Vector3::new(angle, 0.5, -1.0),
                obs: vec![(10.0, 12.0, 1), (30.0, 5.0, -1)],
            };
            Model {
                camera: ("PINHOLE", 1, 64, 48, vec![50.0, 51.0, 32.0, 24.0]),
                images: vec![
                    img(1, "c.png", 0.3),
                    img(2, "a.png", -0.4),
                    img(3, "b.png", 0.9),
                ],
                points: vec![Pt {
                    id: 1,
                    xyz: [0.5, 0.25, 4.0],
                    rgb: [1, 2, 3],
                }],
            }
        }

        /// `<root>/sparse/0` as text or binary files.
        fn write(&self, root: &Path, binary: bool) {
            let dir = root.join("sparse").join("0");
            std::fs::create_dir_all(&dir).unwrap();
            let (name, model_id, w, h, params) = &self.camera;
            let q = |img: &Img| {
                let q = img.rot.quaternion();
                [q.w, q.i, q.j, q.k]
            };
            if binary {
                let mut cams = 1u64.to_le_bytes().to_vec();
                cams.extend(1u32.to_le_bytes());
                cams.extend(model_id.to_le_bytes());
                cams.extend((*w as u64).to_le_bytes());
                cams.extend((*h as u64).to_le_bytes());
                params.iter().for_each(|p| cams.extend(p.to_le_bytes()));
                std::fs::write(dir.join("cameras.bin"), cams).unwrap();

                let mut imgs = (self.images.len() as u64).to_le_bytes().to_vec();
                for img in &self.images {
                    imgs.extend(img.id.to_le_bytes());
                    q(img).iter().for_each(|v| imgs.extend(v.to_le_bytes()));
                    img.t.iter().for_each(|v| imgs.extend(v.to_le_bytes()));
                    imgs.extend(1u32.to_le_bytes());
                    imgs.extend(img.name.as_bytes());
                    imgs.push(0);
                    imgs.extend((img.obs.len() as u64).to_le_bytes());
                    for (x, y, id) in &img.obs {
                        imgs.extend(x.to_le_bytes());
                        imgs.extend(y.to_le_bytes());
                        imgs.extend(id.to_le_bytes());
                    }
                }
                std::fs::write(dir.join("images.bin"), imgs).unwrap();

                let mut pts = (self.points.len() as u64).to_le_bytes().to_vec();
                for p in &self.points {
                    pts.extend(p.id.to_le_bytes());
                    p.xyz.iter().for_each(|v| pts.extend(v.to_le_bytes()));
                    pts.extend(p.rgb);
                    pts.extend(0.5f64.to_le_bytes());
                    pts.extend(0u64.to_le_bytes());
                }
                std::fs::write(dir.join("points3D.bin"), pts).unwrap();
            } else {
                let params: Vec<String> = params.iter().map(|p| p.to_string()).collect();
                std::fs::write(
                    dir.join("cameras.txt"),
                    format!("1 {name} {w} {h} {}\n", params.join(" ")),
                )
                .unwrap();
                let mut imgs = String::new();
                for img in &self.images {
                    let (q, t) = (q(img), img.t);
                    imgs += &format!(
                        "{} {} {} {} {} {} {} {} 1 {}\n",
                        img.id, q[0], q[1], q[2], q[3], t.x, t.y, t.z, img.name
                    );
                    let line: Vec<String> = img
                        .obs
                        .iter()
                        .map(|(x, y, id)| format!("{x} {y} {id}"))
                        .collect();
                    imgs += &line.join(" ");
                    imgs.push('\n');
                }
                std::fs::write(dir.join("images.txt"), imgs).unwrap();
                let pts: String = self
                    .points
                    .iter()
                    .map(|p| {
                        format!(
                            "{} {} {} {} {} {} {} 0.5\n",
                            p.id, p.xyz[0], p.xyz[1], p.xyz[2], p.rgb[0], p.rgb[1], p.rgb[2]
                        )
                    })
                    .collect();
                std::fs::write(dir.join("points3D.txt"), pts).unwrap();
            }
        }
    }

    /// `<root>/images/<name>` for every image of `model`.
    fn write_images(root: &Path, model: &Model, pano: &image::DynamicImage) {
        std::fs::create_dir_all(root.join("images")).unwrap();
        for img in &model.images {
            let (w, h) = (model.camera.2, model.camera.3);
            let image = if (pano.width(), pano.height()) == (w, h) {
                pano.clone()
            } else {
                image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
                    image::Rgb([(x * 4) as u8, (y * 5) as u8, 9])
                }))
            };
            image.save(root.join("images").join(&img.name)).unwrap();
        }
    }

    fn load(root: &Path, eval_every: Option<usize>, log: &mut Vec<String>) -> Dataset {
        load_colmap_dataset_with(root, eval_every, &EquirectOptions::default(), &mut |m| {
            log.push(m.to_string())
        })
        .unwrap()
    }

    /// Which cube face a view is, from its file name (`..._<face>.png`).
    fn face_of(view: &View) -> &'static str {
        let tag = view
            .name
            .trim_end_matches(".png")
            .rsplit('_')
            .next()
            .unwrap();
        cube_face_rotations()
            .iter()
            .map(|(n, _)| *n)
            .find(|n| *n == tag)
            .unwrap_or_else(|| panic!("{} is no cube face", view.name))
    }

    fn rotation_f64(view: &View) -> Matrix3<f64> {
        view.camera.rotation.map(|v| v as f64)
    }

    #[test]
    fn equirect_faces_see_the_world_through_the_pose_they_are_given() {
        // The panorama paints a world direction's brightness; each face image
        // must show, at every pixel, the brightness of the world direction
        // its CameraView says that pixel looks along. Catches a face
        // rotation composed transposed, on the wrong side, or dropped
        // (every one of those lights up a different wrong rotation below),
        // a translation left unrotated (the centre check), and a face image
        // paired with another face's pose (the error check).
        let tmp = Tmp::new("faces");
        let model = Model::equirect(1, [20; 6], 0);
        let rot = pano_rotation();
        write_images(&tmp.0, &model, &panorama(&rot, None));
        model.write(&tmp.0, false);
        let ds = load(&tmp.0, None, &mut Vec::new());
        assert_eq!(ds.train.len(), 6);
        assert!(ds.eval.is_empty());

        let centre = -(rot.inverse() * model.images[0].t);
        let r_pano = rot.to_rotation_matrix().into_inner();
        let face_error = |view: &View, w2c: &Matrix3<f64>| {
            let img = image::open(&view.image_path).unwrap().to_rgb8();
            assert_eq!(img.dimensions(), (FACE, FACE));
            let half = FACE as f64 / 2.0;
            let mut err = 0.0;
            for (u, v, p) in img.enumerate_pixels() {
                let ray = Vector3::new(
                    (u as f64 + 0.5 - half) / half,
                    (v as f64 + 0.5 - half) / half,
                    1.0,
                );
                err += (p.0[0] as f64 - shade(&(w2c.transpose() * ray)).round()).abs();
            }
            err / (FACE * FACE) as f64
        };
        let rotations = cube_face_rotations();
        // Mean error of the pose the dataset gives, and of three wrong
        // composites (face rotation dropped, applied transposed the wrong
        // way, applied on the wrong side).
        let (mut right, mut wrong) = (0.0, [0.0; 3]);
        for view in &ds.train {
            let r_k = rotations
                .iter()
                .find(|(n, _)| *n == face_of(view))
                .unwrap()
                .1;
            right += face_error(view, &rotation_f64(view)) / 6.0;
            let hypotheses = [r_pano, r_k * r_pano, r_pano * r_k.transpose()];
            for (w, h) in wrong.iter_mut().zip(&hypotheses) {
                *w += face_error(view, h) / 6.0;
            }
            // A face looks out of the panorama's own centre.
            let c = view.camera.camera_center();
            assert!(
                (c.map(|v| v as f64) - centre).norm() < 1e-4,
                "{}: centre {c:?} vs {centre:?}",
                view.name
            );
        }
        assert!(right < 1.5, "faces disagree with their poses: {right:.2}");
        // The fixture can tell the right pose from the wrong ones.
        for (i, w) in wrong.iter().enumerate() {
            assert!(
                *w > 8.0,
                "wrong composite {i} scores {w:.2}: fixture too weak"
            );
        }
        // Faces land beside the model by default and are 90 degree pinholes.
        for v in &ds.train {
            assert_eq!(v.image_path.parent().unwrap(), tmp.0.join("equirect_faces"));
            let c = &v.camera.camera;
            assert_eq!((c.width, c.height), (FACE, FACE));
            assert_eq!((c.fx, c.fy, c.cx, c.cy), (64.0, 64.0, 64.0, 64.0));
        }
    }

    #[test]
    fn faces_that_see_too_few_matched_points_are_left_out() {
        // Matched points per face, in cube_face_rotations order: front 16,
        // right 15 (exactly the default minimum), back 14, left 0 but with 30
        // keypoints that match nothing, up 40, down 3. Catches `<` turned
        // into `<=` (right lost), unmatched keypoints counted (left kept), a
        // face credited with another's points (the sets differ), and dropped
        // faces still being cut (the file count).
        let tmp = Tmp::new("support");
        let model = Model::equirect(1, [16, 15, 14, 0, 40, 3], 30);
        write_images(&tmp.0, &model, &panorama(&pano_rotation(), None));
        model.write(&tmp.0, false);
        let faces_dir = tmp.0.join("cut");
        let opts = EquirectOptions {
            faces_dir: Some(faces_dir.clone()),
            ..EquirectOptions::default()
        };
        let mut log = Vec::new();
        let ds = load_colmap_dataset_with(&tmp.0, None, &opts, &mut |m| log.push(m.to_string()))
            .unwrap();
        let mut kept: Vec<_> = ds.train.iter().map(face_of).collect();
        kept.sort_unstable();
        assert_eq!(kept, ["front", "right", "up"]);
        assert_eq!(std::fs::read_dir(&faces_dir).unwrap().count(), 3);
        assert!(ds
            .train
            .iter()
            .all(|v| v.image_path.starts_with(&faces_dir)));
        assert!(
            log.iter().any(|l| l.contains("3 of 6 posed faces")),
            "log: {log:?}"
        );
    }

    #[test]
    fn a_model_without_observations_keeps_every_face() {
        // No POINTS2D at all (a model exported without them) says nothing
        // about which faces see anything; dropping all of them would leave
        // nothing to train on.
        let tmp = Tmp::new("noobs");
        let model = Model::equirect(1, [0; 6], 0);
        let mut model = model;
        model.points.push(Pt {
            id: 1,
            xyz: [0.0, 0.0, 1.0],
            rgb: [0; 3],
        });
        write_images(&tmp.0, &model, &panorama(&pano_rotation(), None));
        model.write(&tmp.0, false);
        let mut log = Vec::new();
        let ds = load(&tmp.0, None, &mut log);
        assert_eq!(ds.train.len(), 6);
        assert!(log.iter().any(|l| l.contains("no observations")), "{log:?}");
    }

    #[test]
    fn held_out_views_are_whole_panoramas() {
        // Four panoramas by name, every 2nd held out: the first and third,
        // with all six of their faces. Splitting the 24 faces instead would
        // hold out one face of each panorama and train on the rest of that
        // view; ranking by id would hold out the wrong two.
        let tmp = Tmp::new("split");
        let model = Model::equirect(4, [20; 6], 0);
        write_images(&tmp.0, &model, &panorama(&pano_rotation(), None));
        model.write(&tmp.0, false);
        let ds = load(&tmp.0, Some(2), &mut Vec::new());
        assert_eq!((ds.train.len(), ds.eval.len()), (12, 12));
        assert!(ds
            .train
            .iter()
            .all(|v| v.name.contains("pano_b") || v.name.contains("pano_d")));
        for pano in ["pano_a", "pano_c"] {
            let mut faces: Vec<_> = ds
                .eval
                .iter()
                .filter(|v| v.name.contains(pano))
                .map(face_of)
                .collect();
            faces.sort_unstable();
            assert_eq!(faces, ["back", "down", "front", "left", "right", "up"]);
        }
    }

    #[test]
    fn a_panorama_with_no_kept_face_takes_no_split_slot() {
        // pano_b sees nothing (no observations of its own, while the model has
        // some), so none of its faces is kept. The held-out split counts the
        // panoramas that give views, as gsplat_photos does: pano_a is held
        // out and pano_c, the second to give views, trains.
        let tmp = Tmp::new("empty_slot");
        let mut model = Model::equirect(3, [20; 6], 0);
        model
            .images
            .iter_mut()
            .find(|i| i.name == "pano_b.png")
            .unwrap()
            .obs
            .clear();
        write_images(&tmp.0, &model, &panorama(&pano_rotation(), None));
        model.write(&tmp.0, false);
        let ds = load(&tmp.0, Some(2), &mut Vec::new());
        assert_eq!((ds.eval.len(), ds.train.len()), (6, 6));
        assert!(ds.eval.iter().all(|v| v.name.contains("pano_a")));
        assert!(ds.train.iter().all(|v| v.name.contains("pano_c")));
    }

    #[test]
    fn a_binary_model_loads_like_its_text_twin() {
        // The same model as sparse/0/*.txt and sparse/0/*.bin: the same
        // views (pose, image name, pixels) and the same seed points.
        // Equirectangular (camera model 17 in the binary file) and pinhole.
        for model in [Model::equirect(2, [20; 6], 4), Model::pinhole()] {
            let (txt, bin) = (Tmp::new("twin_txt"), Tmp::new("twin_bin"));
            let pano = panorama(&pano_rotation(), None);
            for (root, binary) in [(&txt, false), (&bin, true)] {
                write_images(&root.0, &model, &pano);
                model.write(&root.0, binary);
            }
            let a = load(&txt.0, Some(2), &mut Vec::new());
            let b = load(&bin.0, Some(2), &mut Vec::new());
            assert!(!a.train.is_empty() && !a.eval.is_empty());
            for (set_a, set_b) in [(&a.train, &b.train), (&a.eval, &b.eval)] {
                assert_eq!(set_a.len(), set_b.len());
                for (va, vb) in set_a.iter().zip(set_b) {
                    assert_eq!(va.name, vb.name);
                    assert_eq!(va.image_path.file_name(), vb.image_path.file_name());
                    assert_eq!(va.camera.rotation, vb.camera.rotation);
                    assert_eq!(va.camera.translation, vb.camera.translation);
                    assert_eq!(
                        std::fs::read(&va.image_path).unwrap(),
                        std::fs::read(&vb.image_path).unwrap()
                    );
                }
            }
            assert_eq!(a.init.gaussians.len(), model.points.len());
            assert_eq!(a.init.gaussians.len(), b.init.gaussians.len());
            for (ga, gb) in a.init.gaussians.iter().zip(&b.init.gaussians) {
                assert_eq!(ga.mean, gb.mean);
            }
        }
    }

    #[test]
    fn a_panorama_scaled_down_is_filtered_before_its_faces_are_cut() {
        // One-pixel stripes at twice the resolution the faces are cut at. The
        // faces of the scaled-down panorama (as `gsplat_photos` cuts them)
        // show the stripes' mean; sampling the full-size panorama instead
        // would alias them into black and white.
        let tmp = Tmp::new("alias");
        let model = Model::equirect(1, [20; 6], 0);
        let stripes =
            image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(PANO_W, PANO_W / 2, |x, _| {
                image::Rgb([if x % 2 == 0 { 0 } else { 255 }; 3])
            }));
        write_images(&tmp.0, &model, &stripes);
        model.write(&tmp.0, false);
        let opts = EquirectOptions {
            max_face_size: FACE / 2,
            ..EquirectOptions::default()
        };
        let ds = load_colmap_dataset_with(&tmp.0, None, &opts, &mut |_| {}).unwrap();
        assert_eq!(ds.train.len(), 6);
        for v in &ds.train {
            let face = image::open(&v.image_path).unwrap().to_rgb8();
            let (lo, hi) = face
                .pixels()
                .fold((255u8, 0u8), |(lo, hi), p| (lo.min(p.0[0]), hi.max(p.0[0])));
            assert!(
                lo >= 100 && hi <= 155,
                "{}: stripes alias to {lo}..{hi}",
                v.name
            );
        }
    }

    #[test]
    fn a_pinhole_image_beside_panoramas_is_one_view_of_its_own() {
        // pano_a keeps camera 1 (equirectangular); pano_b's file is read as a
        // photo from a second, pinhole camera. Sorted by name, every 2nd
        // held out: the panorama's six faces, then the photo to train on.
        let tmp = Tmp::new("mixed");
        let model = Model::equirect(2, [20; 6], 0);
        write_images(&tmp.0, &model, &panorama(&pano_rotation(), None));
        model.write(&tmp.0, false);
        let sparse = tmp.0.join("sparse").join("0");
        let cameras = std::fs::read_to_string(sparse.join("cameras.txt")).unwrap();
        std::fs::write(
            sparse.join("cameras.txt"),
            cameras + &format!("2 PINHOLE {PANO_W} {} 300 300 256 128\n", PANO_W / 2),
        )
        .unwrap();
        let images = std::fs::read_to_string(sparse.join("images.txt")).unwrap();
        assert_eq!(images.matches(" 1 pano_b.png").count(), 1);
        std::fs::write(
            sparse.join("images.txt"),
            images.replace(" 1 pano_b.png", " 2 pano_b.png"),
        )
        .unwrap();
        let ds = load(&tmp.0, Some(2), &mut Vec::new());
        assert_eq!((ds.eval.len(), ds.train.len()), (6, 1));
        assert!(ds.eval.iter().all(|v| v.name.contains("pano_a")));
        let photo = &ds.train[0];
        assert_eq!(photo.name, "pano_b.png");
        assert_eq!(photo.image_path, tmp.0.join("images").join("pano_b.png"));
        assert_eq!(photo.camera.camera.fx, 300.0);
    }

    #[test]
    fn pinhole_models_load_as_they_always_did() {
        // Sorted by image name, every 2nd held out, images under images/,
        // the camera straight from the quaternion: no face is cut.
        let tmp = Tmp::new("pinhole");
        let model = Model::pinhole();
        write_images(&tmp.0, &model, &panorama(&pano_rotation(), None));
        model.write(&tmp.0, false);
        let ds = load(&tmp.0, Some(2), &mut Vec::new());
        let names = |vs: &[View]| vs.iter().map(|v| v.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&ds.eval), ["a.png", "c.png"]);
        assert_eq!(names(&ds.train), ["b.png"]);
        assert_eq!(ds.train[0].image_path, tmp.0.join("images").join("b.png"));
        let want = model.images[2].rot.to_rotation_matrix().into_inner();
        assert!((rotation_f64(&ds.train[0]) - want).norm() < 1e-6);
        let c = &ds.train[0].camera.camera;
        assert_eq!(
            (c.width, c.height, c.fx, c.fy, c.cx, c.cy),
            (64, 48, 50.0, 51.0, 32.0, 24.0)
        );
        assert!(!tmp.0.join("equirect_faces").exists());
    }

    #[test]
    fn faces_carry_the_panoramas_alpha() {
        // A hole in the panorama's alpha, centred on longitude +90 degrees
        // and within 34 degrees of the equator (the right face's centre; the
        // up and down faces start at 35), is a hole in that face and nowhere
        // else. A panorama without alpha gives faces without any.
        let tmp = Tmp::new("alpha");
        let model = Model::equirect(1, [20; 6], 0);
        let (h, w) = (PANO_W / 2, PANO_W);
        let hole = |x: u32, y: u32| {
            let inside = x.abs_diff(3 * w / 4) < 40 && y.abs_diff(h / 2) < h * 34 / 180;
            if inside {
                0
            } else {
                255
            }
        };
        write_images(&tmp.0, &model, &panorama(&pano_rotation(), Some(&hole)));
        model.write(&tmp.0, false);
        let ds = load(&tmp.0, None, &mut Vec::new());
        assert_eq!(ds.train.len(), 6);
        for v in &ds.train {
            let (_, valid) = load_view_rgb_mask(v).unwrap();
            let valid = valid.unwrap_or_else(|| panic!("{} lost its alpha", v.name));
            let at = |x: u32, y: u32| valid[(y * FACE + x) as usize];
            if face_of(v) == "right" {
                assert!(!at(FACE / 2, FACE / 2));
                assert!(at(2, FACE / 2) && at(FACE - 3, FACE / 2));
                let holes = valid.iter().filter(|&&ok| !ok).count();
                assert!(holes > (FACE * FACE) as usize / 5, "{holes} hole pixels");
            } else {
                assert!(valid.iter().all(|&ok| ok), "{} has a hole", v.name);
            }
        }
        // And without alpha in the panorama.
        let plain = Tmp::new("alpha_none");
        write_images(&plain.0, &model, &panorama(&pano_rotation(), None));
        model.write(&plain.0, false);
        for v in &load(&plain.0, None, &mut Vec::new()).train {
            assert!(load_view_rgb_mask(v).unwrap().1.is_none());
        }
    }

    #[test]
    fn faces_are_cut_at_most_max_face_size_and_the_panorama_must_fit_its_camera() {
        let tmp = Tmp::new("size");
        let model = Model::equirect(1, [20; 6], 0);
        write_images(&tmp.0, &model, &panorama(&pano_rotation(), None));
        model.write(&tmp.0, false);
        let opts = EquirectOptions {
            max_face_size: 64,
            ..EquirectOptions::default()
        };
        let ds = load_colmap_dataset_with(&tmp.0, None, &opts, &mut |_| {}).unwrap();
        assert_eq!(ds.train.len(), 6);
        for v in &ds.train {
            assert_eq!((v.camera.camera.width, v.camera.camera.height), (64, 64));
            let face = image::open(&v.image_path).unwrap();
            assert_eq!((face.width(), face.height()), (64, 64));
            assert_eq!(v.camera.camera.fx, 32.0);
        }
        // A panorama file that is not the camera's size is an error.
        image::RgbImage::new(PANO_W / 2, PANO_W / 4)
            .save(tmp.0.join("images").join("pano_a.png"))
            .unwrap();
        assert!(matches!(
            load_colmap_dataset_with(&tmp.0, None, &EquirectOptions::default(), &mut |_| {}),
            Err(DatasetError::SizeMismatch { .. })
        ));
    }
}
