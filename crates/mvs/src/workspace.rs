//! The views multi-view stereo works on: each image at the working
//! resolution with its pinhole camera, pose, depth range and source views.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use thiserror::Error;

use crate::model::SparseModel;

/// Why a workspace could not be built.
#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error("image {name}: camera {camera_id} is not PINHOLE or SIMPLE_PINHOLE (undistort first)")]
    UnsupportedCamera { name: String, camera_id: u32 },
    #[error("image {name}: unknown camera {camera_id}")]
    UnknownCamera { name: String, camera_id: u32 },
    #[error("loading {path}: {source}")]
    Image {
        path: PathBuf,
        source: image::ImageError,
    },
    #[error("image {name}: observes no 3D point in front of it, so it has no depth range")]
    NoDepthRange { name: String },
}

/// How a workspace is built.
#[derive(Debug, Clone)]
pub struct WorkspaceOptions {
    /// Images are resized so neither side exceeds this many pixels (COLMAP's
    /// `max_image_size`); `None` keeps the full resolution.
    pub max_image_size: Option<u32>,
    /// Source views per reference view, the best by [`view_pair_score`].
    pub num_sources: usize,
    /// Depth range per view: these quantiles of its sparse points' depths,
    /// widened by `depth_stretch` on either side (COLMAP's
    /// `ComputeDepthRanges`: 1%, 99%, 25%).
    pub depth_quantiles: (f64, f64),
    pub depth_stretch: f64,
}

impl Default for WorkspaceOptions {
    fn default() -> Self {
        Self {
            max_image_size: Some(3200),
            num_sources: 10,
            depth_quantiles: (0.01, 0.99),
            depth_stretch: 0.25,
        }
    }
}

/// One image as multi-view stereo sees it.
#[derive(Debug, Clone)]
pub struct View {
    pub name: String,
    pub width: usize,
    pub height: usize,
    /// Pinhole intrinsics at the working resolution (the upper-left pixel's
    /// centre is at `(0.5, 0.5)`).
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// World-to-camera rotation and translation.
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    /// Grey levels in `[0, 255]`, row-major; empty when they are the
    /// colours' luma, which [`View::gray_levels`] then computes on demand
    /// ([`Workspace::load`] keeps none, to save memory).
    pub gray: Vec<f32>,
    /// Colours, row-major.
    pub rgb: Vec<[u8; 3]>,
    /// Depths searched for this view.
    pub depth_range: (f32, f32),
    /// The median depth of the sparse points the view observes.
    pub median_depth: f32,
    /// Indices of the source views, best first.
    pub sources: Vec<usize>,
}

impl View {
    /// The grey levels: [`View::gray`], or the colours' luma
    /// `0.299 r + 0.587 g + 0.114 b` when it is empty.
    pub fn gray_levels(&self) -> Cow<'_, [f32]> {
        if self.gray.is_empty() {
            Cow::Owned(self.rgb.par_iter().map(|&c| luma(c)).collect())
        } else {
            Cow::Borrowed(&self.gray)
        }
    }

    /// The camera centre in world coordinates.
    pub fn center(&self) -> Vector3<f64> {
        -(self.rotation.transpose() * self.translation)
    }

    /// `(x, y, depth)` of world point `point` in this view's pixel
    /// coordinates.
    pub fn project(&self, point: &Vector3<f64>) -> (f64, f64, f64) {
        let p = self.rotation * point + self.translation;
        (
            self.fx * p.x / p.z + self.cx,
            self.fy * p.y / p.z + self.cy,
            p.z,
        )
    }

    /// The world point at pixel coordinates `(x, y)` and `depth`.
    pub fn unproject(&self, x: f64, y: f64, depth: f64) -> Vector3<f64> {
        let camera = Vector3::new(
            (x - self.cx) / self.fx * depth,
            (y - self.cy) / self.fy * depth,
            depth,
        );
        self.rotation.transpose() * (camera - self.translation)
    }
}

/// The views of a sparse model.
#[derive(Debug, Clone)]
pub struct Workspace {
    pub views: Vec<View>,
}

impl Workspace {
    /// The model's images, loaded from `image_dir` (where their names are
    /// relative to), resized and paired with source views.
    pub fn load(
        model: &SparseModel,
        image_dir: &Path,
        options: &WorkspaceOptions,
    ) -> Result<Self, WorkspaceError> {
        let tracks = point_tracks(model);
        let scores = view_pair_scores(model, &tracks);
        let views = model
            .images
            .par_iter()
            .enumerate()
            .map(|(index, image)| {
                let camera = model.cameras.get(&image.camera_id).ok_or_else(|| {
                    WorkspaceError::UnknownCamera {
                        name: image.name.clone(),
                        camera_id: image.camera_id,
                    }
                })?;
                let (fx, fy, cx, cy) =
                    camera
                        .pinhole()
                        .ok_or_else(|| WorkspaceError::UnsupportedCamera {
                            name: image.name.clone(),
                            camera_id: image.camera_id,
                        })?;
                let path = image_dir.join(&image.name);
                let rgb = image::open(&path)
                    .map_err(|source| WorkspaceError::Image {
                        path: path.clone(),
                        source,
                    })?
                    .into_rgb8();
                let rgb = match options.max_image_size {
                    Some(max) if rgb.width().max(rgb.height()) > max => {
                        let scale = f64::from(max) / f64::from(rgb.width().max(rgb.height()));
                        let width = (f64::from(rgb.width()) * scale).round().max(1.0) as u32;
                        let height = (f64::from(rgb.height()) * scale).round().max(1.0) as u32;
                        image::imageops::resize(
                            &rgb,
                            width,
                            height,
                            image::imageops::FilterType::Triangle,
                        )
                    }
                    _ => rgb,
                };
                // The camera scaled to the working image (also when the file
                // is smaller than the model's camera).
                let sx = f64::from(rgb.width()) / f64::from(camera.width);
                let sy = f64::from(rgb.height()) / f64::from(camera.height);
                let rotation = image.rotation();
                let translation = image.translation();
                let (depth_range, median_depth) =
                    depth_range(model, image, &rotation, &translation, options).ok_or_else(
                        || WorkspaceError::NoDepthRange {
                            name: image.name.clone(),
                        },
                    )?;
                let mut candidates: Vec<(usize, f64)> = scores
                    .iter()
                    .filter_map(|(&(a, b), &score)| {
                        if a == index {
                            Some((b, score))
                        } else if b == index {
                            Some((a, score))
                        } else {
                            None
                        }
                    })
                    .filter(|&(_, score)| score > 0.0)
                    .collect();
                // Best score first; ties by index, so the order is fixed.
                candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                candidates.truncate(options.num_sources);
                Ok(View {
                    name: image.name.clone(),
                    width: rgb.width() as usize,
                    height: rgb.height() as usize,
                    fx: fx * sx,
                    fy: fy * sy,
                    cx: cx * sx,
                    cy: cy * sy,
                    rotation,
                    translation,
                    gray: Vec::new(),
                    rgb: rgb.pixels().map(|p| p.0).collect(),
                    depth_range,
                    median_depth,
                    sources: candidates.into_iter().map(|(view, _)| view).collect(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { views })
    }
}

impl Workspace {
    /// Per view `v`, the views that `v` is the last to read when the views
    /// are visited in order, each reading its own data and its sources'.
    pub(crate) fn last_read_by(&self) -> Vec<Vec<usize>> {
        let n = self.views.len();
        let mut last_reader: Vec<usize> = (0..n).collect();
        for (v, view) in self.views.iter().enumerate() {
            for &s in &view.sources {
                last_reader[s] = last_reader[s].max(v);
            }
        }
        let mut last_read_by = vec![Vec::new(); n];
        for (s, &v) in last_reader.iter().enumerate() {
            last_read_by[v].push(s);
        }
        last_read_by
    }
}

/// The grey level of a colour.
pub(crate) fn luma([r, g, b]: [u8; 3]) -> f32 {
    0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b)
}

/// Per 3D point, the model's image indices observing it.
fn point_tracks(model: &SparseModel) -> HashMap<u64, Vec<usize>> {
    let mut tracks: HashMap<u64, Vec<usize>> = HashMap::new();
    for (index, image) in model.images.iter().enumerate() {
        for id in image.point3d_ids.iter().flatten() {
            let track = tracks.entry(*id).or_default();
            if track.last() != Some(&index) {
                track.push(index);
            }
        }
    }
    tracks
}

/// MVSNet's view-pair score for one shared point seen at triangulation angle
/// `theta_deg`: a Gaussian around 5° (σ 1° below, 10° above), so pairs with
/// a moderate baseline score best.
pub fn view_pair_score(theta_deg: f64) -> f64 {
    const THETA0: f64 = 5.0;
    let sigma = if theta_deg <= THETA0 { 1.0 } else { 10.0 };
    (-(theta_deg - THETA0).powi(2) / (2.0 * sigma * sigma)).exp()
}

/// The summed [`view_pair_score`] of every image pair over their shared
/// points, keyed `(a, b)` with `a < b`.
fn view_pair_scores(
    model: &SparseModel,
    tracks: &HashMap<u64, Vec<usize>>,
) -> HashMap<(usize, usize), f64> {
    let centers: Vec<Vector3<f64>> = model.images.iter().map(|image| image.center()).collect();
    let mut scores: HashMap<(usize, usize), f64> = HashMap::new();
    let mut ids: Vec<&u64> = tracks.keys().collect();
    // A fixed order, so the sums do not depend on the hash order.
    ids.sort_unstable();
    for id in ids {
        let Some(point) = model.points.get(id) else {
            continue;
        };
        let track = &tracks[id];
        for (i, &a) in track.iter().enumerate() {
            for &b in &track[i + 1..] {
                let ray_a = (point.xyz - centers[a]).normalize();
                let ray_b = (point.xyz - centers[b]).normalize();
                let theta = ray_a.dot(&ray_b).clamp(-1.0, 1.0).acos().to_degrees();
                *scores.entry((a.min(b), a.max(b))).or_default() += view_pair_score(theta);
            }
        }
    }
    scores
}

/// The depth range of `image` from the depths of the points it observes,
/// and their median.
fn depth_range(
    model: &SparseModel,
    image: &crate::model::Image,
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
    options: &WorkspaceOptions,
) -> Option<((f32, f32), f32)> {
    let mut depths: Vec<f64> = image
        .point3d_ids
        .iter()
        .flatten()
        .filter_map(|id| model.points.get(id))
        .map(|point| (rotation * point.xyz + translation).z)
        .filter(|&z| z > 0.0)
        .collect();
    if depths.is_empty() {
        return None;
    }
    depths.sort_by(f64::total_cmp);
    let at = |q: f64| depths[((depths.len() - 1) as f64 * q).round() as usize];
    let (lo, hi) = options.depth_quantiles;
    Some((
        (
            (at(lo) * (1.0 - options.depth_stretch)) as f32,
            (at(hi) * (1.0 + options.depth_stretch)) as f32,
        ),
        at(0.5) as f32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_score_peaks_at_five_degrees() {
        assert!((view_pair_score(5.0) - 1.0).abs() < 1e-12);
        assert!(view_pair_score(4.0) < view_pair_score(6.0));
        assert!(view_pair_score(0.0) < 1e-5);
    }
}
