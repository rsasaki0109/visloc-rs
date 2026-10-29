//! Mesh extraction from a trained splat: its median depth rendered from
//! every view, fused by [`visloc_gsplat_core::mesh`]'s TSDF and extracted
//! with surface nets.

use nalgebra::Vector3;
use rayon::prelude::*;
use visloc_gsplat_core::camera::CameraView;
pub use visloc_gsplat_core::mesh::{rig_scale, DepthFrame, Mesh, Tsdf};

/// Settings for [`fuse_views`]; lengths in voxels unless noted.
#[derive(Debug, Clone)]
pub struct MeshOptions {
    /// Voxel edge (world units); `None`: [`scene_scale`] / 256.
    pub voxel: Option<f32>,
    pub trunc_voxels: f32,
    /// Ignore depths beyond this (world units); `None`: 2 x [`scene_scale`].
    pub max_depth: Option<f32>,
    pub carve: bool,
    pub min_weight: f32,
    /// Drop triangles farther than this from every support point (0: off).
    pub support_voxels: f32,
    pub min_component: usize,
}

impl Default for MeshOptions {
    fn default() -> Self {
        Self {
            voxel: None,
            trunc_voxels: 4.0,
            max_depth: None,
            carve: true,
            min_weight: 3.0,
            support_voxels: 10.0,
            min_component: 500,
        }
    }
}

/// The larger of [`rig_scale`] and half the 90th percentile depth of the
/// `support` points the views see. Cameras around an object see it at about
/// the rig scale; cameras close together looking out (a 360 capture, a short
/// walk) see the scene far beyond it, where 2 x the rig scale would drop
/// almost every depth. Samples at most ~500 views and ~20k points.
pub fn scene_scale(views: &[&CameraView], support: &[[f32; 3]]) -> f32 {
    let rig = rig_scale(views);
    if views.is_empty() || support.is_empty() {
        return rig;
    }
    let points: Vec<Vector3<f32>> = support
        .iter()
        .step_by((support.len() / 20_000).max(1))
        .map(|&p| Vector3::from(p))
        .collect();
    let mut depths: Vec<f32> = views
        .par_iter()
        .step_by((views.len() / 500).max(1))
        .flat_map_iter(|view| {
            let cam = &view.camera;
            points.iter().filter_map(move |p| {
                let x = view.rotation * p + view.translation;
                if x.z <= 1e-6 {
                    return None;
                }
                let u = cam.fx * x.x / x.z + cam.cx;
                let v = cam.fy * x.y / x.z + cam.cy;
                (u >= 0.0 && u < cam.width as f32 && v >= 0.0 && v < cam.height as f32)
                    .then_some(x.z)
            })
        })
        .collect();
    if depths.is_empty() {
        return rig;
    }
    let k = (depths.len() * 9 / 10).min(depths.len() - 1);
    let (_, p90, _) = depths.select_nth_unstable_by(k, f32::total_cmp);
    rig.max(0.5 * *p90)
}

/// Fuse the median depth of every view into a TSDF, extract the mesh, filter
/// it by `support` (e.g. SfM points) and drop small components. `render`
/// gives a view's row-major RGB and median depth (camera z, 0 = no surface),
/// from whichever renderer holds the splat. Returns the mesh and a one-line
/// summary.
pub fn fuse_views<E>(
    views: &[&CameraView],
    support: &[[f32; 3]],
    opts: &MeshOptions,
    mut render: impl FnMut(&CameraView) -> Result<(Vec<[f32; 3]>, Vec<f32>), E>,
) -> Result<(Mesh, String), E> {
    let scale = scene_scale(views, support);
    let voxel = opts.voxel.unwrap_or(scale / 256.0);
    let max_depth = opts.max_depth.unwrap_or(2.0 * scale);
    let mut tsdf = Tsdf::new(voxel, voxel * opts.trunc_voxels, max_depth);
    tsdf.carve = opts.carve;
    tsdf.min_weight = opts.min_weight;
    let t0 = std::time::Instant::now();
    let (mut render_s, mut integrate_s) = (0.0f64, 0.0f64);
    for view in views {
        let t = std::time::Instant::now();
        let (rgb, depth) = render(view)?;
        render_s += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        tsdf.integrate(&DepthFrame {
            view,
            depth: &depth,
            rgb: &rgb,
        });
        integrate_s += t.elapsed().as_secs_f64();
    }
    let fused = t0.elapsed().as_secs_f64();
    let mut mesh = tsdf.extract();
    let raw = mesh.triangles.len();
    if opts.support_voxels > 0.0 && !support.is_empty() {
        mesh.keep_supported(support, opts.support_voxels * voxel);
    }
    mesh.remove_small_components(opts.min_component);
    let summary = format!(
        "mesh: {} views fused in {fused:.1} s (render {render_s:.1} s, integrate {integrate_s:.1} s; voxel {voxel:.4}, {} blocks), {raw} -> {} triangles",
        views.len(),
        tsdf.num_blocks(),
        mesh.triangles.len()
    );
    Ok((mesh, summary))
}

/// [`fuse_views`] with the wgpu renderer: render `scene`'s median depth from
/// every view. Returns the mesh, a one-line summary and the context.
#[cfg(feature = "gpu")]
pub fn extract_from_splat(
    ctx: visloc_gsplat_render::GpuContext,
    scene: &visloc_gsplat_core::gaussian::Scene,
    views: &[&CameraView],
    support: &[[f32; 3]],
    opts: &MeshOptions,
) -> Result<(Mesh, String, visloc_gsplat_render::GpuContext), visloc_gsplat_render::GpuError> {
    let mut ctx = Some(ctx);
    let mut renderer: Option<(u32, u32, visloc_gsplat_render::Renderer)> = None;
    let fused = fuse_views(
        views,
        support,
        opts,
        |view| -> Result<_, visloc_gsplat_render::GpuError> {
            let (w, h) = (view.camera.width, view.camera.height);
            if renderer.as_ref().map(|r| (r.0, r.1)) != Some((w, h)) {
                let c = match renderer.take() {
                    Some((_, _, r)) => r.into_context(),
                    None => ctx.take().expect("context"),
                };
                renderer = Some((w, h, visloc_gsplat_render::Renderer::new(c, scene, w, h)?));
            }
            let (_, _, r) = renderer.as_mut().expect("renderer set above");
            let (image, depth, _) = r.render_depth(view, [0.0, 0.0, 0.0]);
            Ok((image.rgb, depth))
        },
    );
    let (mesh, summary) = fused?;
    let ctx = match renderer {
        Some((_, _, r)) => r.into_context(),
        None => ctx.take().expect("context"),
    };
    Ok((mesh, summary, ctx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;
    use visloc_gsplat_core::camera::PinholeCamera;

    /// Six axis cameras at `dist * d` looking along `sign * d` (-1: inward).
    fn axis_views(dist: f32, sign: f32) -> Vec<CameraView> {
        let cam = PinholeCamera::new(128, 128, 64.0, 64.0, 64.0, 64.0);
        let dirs = [
            Vector3::x(),
            -Vector3::x(),
            Vector3::y(),
            -Vector3::y(),
            Vector3::z(),
            -Vector3::z(),
        ];
        dirs.iter()
            .map(|&d: &Vector3<f32>| {
                let fwd = sign * d;
                let up = if d.y.abs() > 0.5 {
                    Vector3::z()
                } else {
                    Vector3::y()
                };
                let right = up.cross(&fwd).normalize();
                let down = fwd.cross(&right);
                let r = Matrix3::from_rows(&[right.transpose(), down.transpose(), fwd.transpose()]);
                CameraView::new(r, -(r * (d * dist)), cam)
            })
            .collect()
    }

    /// Points on a sphere of `radius` about the origin.
    fn sphere_points(radius: f32) -> Vec<[f32; 3]> {
        let n = 2000;
        (0..n)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
                let phi = i as f32 * 2.399_963;
                let s = (1.0 - z * z).sqrt();
                [radius * s * phi.cos(), radius * s * phi.sin(), radius * z]
            })
            .collect()
    }

    /// Cameras around an object keep the rig scale; cameras close together
    /// looking out at far walls take half the far depths.
    #[test]
    fn scene_scale_covers_outward_captures() {
        let inward = axis_views(3.0, -1.0);
        let refs: Vec<&CameraView> = inward.iter().collect();
        let scale = scene_scale(&refs, &sphere_points(1.0));
        assert_eq!(scale, rig_scale(&refs));
        assert!((scale - 3.0).abs() < 1e-5, "{scale}");

        let outward = axis_views(0.1, 1.0);
        let refs: Vec<&CameraView> = outward.iter().collect();
        assert!((rig_scale(&refs) - 0.1).abs() < 1e-5);
        let scale = scene_scale(&refs, &sphere_points(10.0));
        // Depths along each axis run from ~5.7 (frustum edge) to ~9.9.
        assert!(scale > 4.0 && scale < 5.0, "{scale}");
        assert_eq!(scene_scale(&refs, &[]), rig_scale(&refs));
    }
}
