//! Host-side uniform computation shared by the GPU kernels.
//!
//! Everything here is pure `f32`/`u32` math on the camera and image size, so it
//! is unit-tested without a GPU device.

use visloc_gsplat_core::camera::CameraView;

/// Tile edge length in pixels (one workgroup rasterizes one tile).
pub const TILE_WIDTH: u32 = 16;
/// Tile area / rasterizer workgroup size (one thread per pixel).
pub const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

/// The uniform block consumed by `project_forward` / `project_visible`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProjectUniforms {
    /// World-to-camera rotation, row-major 3x3.
    pub view_rot: [[f32; 4]; 3],
    /// World-to-camera translation.
    pub view_t: [f32; 4],
    /// Camera centre in world coordinates.
    pub camera_center: [f32; 4],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub img_w: u32,
    pub img_h: u32,
    pub tile_bw: u32,
    pub tile_bh: u32,
    pub sh_degree: u32,
    pub total_splats: u32,
    /// Number of visible (compacted) gaussians this frame.
    pub num_visible: u32,
    /// Number of `(tile, compact)` intersections this frame.
    pub num_intersections: u32,
    /// SH degree actually evaluated (<= `sh_degree`, which sets the buffer
    /// stride); trainers raise it progressively. Higher bands read as 0.
    pub sh_active_degree: u32,
    /// Padding to keep the struct 16-byte aligned for storage/uniform use.
    pub _pad: [u32; 3],
}

/// The uniform block consumed by the `rasterize` kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RasterUniforms {
    pub tile_bw: u32,
    pub img_w: u32,
    pub img_h: u32,
    pub tile_bh: u32,
    pub bg_r: f32,
    pub bg_g: f32,
    pub bg_b: f32,
    pub _pad: f32,
}

impl RasterUniforms {
    pub fn new(u: &ProjectUniforms, bg: [f32; 3]) -> Self {
        Self {
            tile_bw: u.tile_bw,
            img_w: u.img_w,
            img_h: u.img_h,
            tile_bh: u.tile_bh,
            bg_r: bg[0],
            bg_g: bg[1],
            bg_b: bg[2],
            _pad: 0.0,
        }
    }
}

/// Tile grid dimensions for an image of `w x h` pixels.
pub fn tile_bounds(w: u32, h: u32) -> (u32, u32) {
    (w.div_ceil(TILE_WIDTH), h.div_ceil(TILE_WIDTH))
}

/// Number of tiles covering the image.
pub fn num_tiles(w: u32, h: u32) -> u32 {
    let (bw, bh) = tile_bounds(w, h);
    bw * bh
}

impl ProjectUniforms {
    /// Build the projection uniforms for `view`.
    pub fn from_view(view: &CameraView, sh_degree: u32, total_splats: u32) -> Self {
        let r = view.rotation;
        let t = view.translation;
        let c = view.camera_center();
        let (bw, bh) = tile_bounds(view.camera.width, view.camera.height);
        Self {
            view_rot: [
                [r[(0, 0)], r[(0, 1)], r[(0, 2)], 0.0],
                [r[(1, 0)], r[(1, 1)], r[(1, 2)], 0.0],
                [r[(2, 0)], r[(2, 1)], r[(2, 2)], 0.0],
            ],
            view_t: [t.x, t.y, t.z, 0.0],
            camera_center: [c.x, c.y, c.z, 0.0],
            fx: view.camera.fx,
            fy: view.camera.fy,
            cx: view.camera.cx,
            cy: view.camera.cy,
            img_w: view.camera.width,
            img_h: view.camera.height,
            tile_bw: bw,
            tile_bh: bh,
            sh_degree,
            total_splats,
            num_visible: 0,
            num_intersections: 0,
            sh_active_degree: sh_degree,
            _pad: [0; 3],
        }
    }

    pub fn num_tiles(&self) -> u32 {
        self.tile_bw * self.tile_bh
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};
    use visloc_gsplat_core::camera::PinholeCamera;

    #[test]
    fn tile_bounds_rounds_up() {
        assert_eq!(tile_bounds(16, 16), (1, 1));
        assert_eq!(tile_bounds(17, 1), (2, 1));
        assert_eq!(tile_bounds(64, 48), (4, 3));
        assert_eq!(num_tiles(64, 48), 12);
    }

    #[test]
    fn uniforms_carry_camera_fields() {
        let view = CameraView::new(
            Matrix3::identity(),
            Vector3::new(0.0, 0.0, -5.0),
            PinholeCamera::new(64, 48, 50.0, 50.0, 32.0, 24.0),
        );
        let u = ProjectUniforms::from_view(&view, 3, 10);
        assert_eq!(u.fx, 50.0);
        assert_eq!(u.img_w, 64);
        assert_eq!((u.tile_bw, u.tile_bh), (4, 3));
        assert_eq!(u.sh_degree, 3);
        assert_eq!(u.total_splats, 10);
        // camera centre is -R^T t = -t for identity rotation.
        assert_eq!(u.camera_center, [0.0, 0.0, 5.0, 0.0]);
    }
}
