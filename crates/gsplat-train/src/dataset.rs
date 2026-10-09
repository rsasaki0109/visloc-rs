//! A COLMAP scene with its images, split into train and eval views.
//!
//! Expected layout (what `colmap image_undistorter` writes, plus a text model):
//!
//! ```text
//! <root>/images/<name>          one image per registered view
//! <root>/sparse/0/cameras.txt   (or sparse/, text or binary)
//! ```
//!
//! Image names come from the text `images.txt` (the visloc map keeps only
//! ids). Views are sorted by image name and every `eval_every`-th one (index 0, N,
//! 2N, ...) is held out, matching brush's and the Inria code's split so scores
//! are comparable. Cameras must be pinhole: undistort first.

use std::path::{Path, PathBuf};

use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_core::colmap_scene::{
    load_colmap_scene, ColmapSceneError, DEFAULT_SEED_LOG_SCALE,
};
use visloc_gsplat_core::gaussian::Scene;

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

/// Load `<root>` (see module docs). `eval_every = None` keeps every view for
/// training.
pub fn load_colmap_dataset(
    root: impl AsRef<Path>,
    eval_every: Option<usize>,
) -> Result<Dataset, DatasetError> {
    let root = root.as_ref();
    let model_dir = [root.join("sparse").join("0"), root.join("sparse")]
        .into_iter()
        .find(|d| d.join("cameras.txt").exists() || d.join("cameras.bin").exists())
        .ok_or_else(|| DatasetError::NoModel(root.to_path_buf()))?;
    let colmap = load_colmap_scene(&model_dir, DEFAULT_SEED_LOG_SCALE)?;
    let names = read_image_names(&model_dir.join("images.txt"))?;

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

/// `IMAGE_ID -> NAME` from a COLMAP text `images.txt` (pose lines are
/// `ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`, each followed by a points line).
fn read_image_names(path: &Path) -> Result<std::collections::HashMap<u64, String>, DatasetError> {
    let text = std::fs::read_to_string(path).map_err(|source| DatasetError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut names = std::collections::HashMap::new();
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
