//! Minimal camera and camera-to-world transform for the reference renderer.
//!
//! The COLMAP / 3DGS convention stores the **world-to-camera** rotation and
//! translation in `images.txt` (quaternion `(w, x, y, z)`, then `t`), with the
//! camera looking down its local `+Z`. This module keeps that convention and
//! exposes the pieces the rasterizer needs: world-to-camera, the Jacobian of
//! the perspective projection, and pixel intrinsics.

use nalgebra::{Matrix3, Vector3};

use crate::lens::{Lens, Projection};

/// Pixel intrinsics (the COLMAP `PINHOLE` / `SIMPLE_PINHOLE` shape) and the
/// lens in front of them: an undistorted pinhole unless
/// [`PinholeCamera::with_lens`] sets a distortion or a fisheye projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeCamera {
    pub width: u32,
    pub height: u32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub lens: Lens,
}

impl PinholeCamera {
    pub fn new(width: u32, height: u32, fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            lens: Lens::PINHOLE,
        }
    }

    /// This camera behind `lens`, its [`Lens::max_theta`] fitted to the
    /// image ([`Lens::fit_max_theta`]).
    pub fn with_lens(self, lens: Lens) -> Self {
        let lens = if lens.is_pinhole() {
            Lens::PINHOLE
        } else {
            lens.fit_max_theta(self.fx, self.fy, self.cx, self.cy, self.width, self.height)
        };
        Self { lens, ..self }
    }

    /// Project a camera-space point to pixels through an undistorted
    /// pinhole, returning `(u, v, depth)`. Depth is the camera-space `z`
    /// (positive in front). The lens is not applied.
    pub fn project(&self, p_cam: Vector3<f32>) -> (f32, f32, f32) {
        let z = p_cam.z;
        let inv = if z.abs() > f32::EPSILON { 1.0 / z } else { 0.0 };
        (
            self.fx * p_cam.x * inv + self.cx,
            self.fy * p_cam.y * inv + self.cy,
            z,
        )
    }
}

/// A camera pose as a world-to-camera rigid transform with its intrinsics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraView {
    /// World-to-camera rotation `R` (world points map as `R * p + t`).
    pub rotation: Matrix3<f32>,
    /// World-to-camera translation.
    pub translation: Vector3<f32>,
    pub camera: PinholeCamera,
}

impl CameraView {
    pub fn new(rotation: Matrix3<f32>, translation: Vector3<f32>, camera: PinholeCamera) -> Self {
        Self {
            rotation,
            translation,
            camera,
        }
    }

    /// Transform a world point into the camera frame.
    pub fn world_to_camera_point(&self, p_world: Vector3<f32>) -> Vector3<f32> {
        self.rotation * p_world + self.translation
    }

    /// This view for the code that still back-projects pixels through a
    /// pinhole (TSDF meshing, the planar median depth, the depth-normal
    /// loss): a distorted pinhole without its distortion, as those paths
    /// treated every camera before lenses were rendered, or `None` for an
    /// equidistant (fisheye) lens, whose rays they cannot follow.
    pub fn pinhole_geometry(&self) -> Option<Self> {
        match self.camera.lens.projection {
            Projection::Equidistant => None,
            Projection::Pinhole => Some(Self {
                camera: PinholeCamera {
                    lens: Lens::PINHOLE,
                    ..self.camera
                },
                ..*self
            }),
        }
    }

    /// Camera centre in world coordinates (`C = -R^T t`).
    pub fn camera_center(&self) -> Vector3<f32> {
        -(self.rotation.transpose() * self.translation)
    }

    /// Unit direction from the camera centre toward a world point. This is the
    /// vector the SH evaluation uses (Inria normalizes `mean - camera_center`).
    pub fn view_direction(&self, p_world: Vector3<f32>) -> Vector3<f32> {
        let d = p_world - self.camera_center();
        let n = d.norm();
        if n > f32::EPSILON {
            d / n
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    #[test]
    fn project_center_ray() {
        let cam = PinholeCamera::new(640, 480, 500.0, 500.0, 320.0, 240.0);
        let (u, v, z) = cam.project(Vector3::new(0.0, 0.0, 2.0));
        assert!((u - 320.0).abs() < 1e-4);
        assert!((v - 240.0).abs() < 1e-4);
        assert!((z - 2.0).abs() < 1e-6);
    }

    #[test]
    fn project_offset() {
        let cam = PinholeCamera::new(640, 480, 500.0, 500.0, 320.0, 240.0);
        let (u, v, _) = cam.project(Vector3::new(1.0, 1.0, 2.0));
        assert!((u - (500.0 * 0.5 + 320.0)).abs() < 1e-4);
        assert!((v - (500.0 * 0.5 + 240.0)).abs() < 1e-4);
    }

    #[test]
    fn camera_center_inverts_transform() {
        let r = Matrix3::identity();
        let t = Vector3::new(1.0, 2.0, 3.0);
        let view = CameraView::new(r, t, PinholeCamera::new(1, 1, 1.0, 1.0, 0.0, 0.0));
        let c = view.camera_center();
        assert!((c - Vector3::new(-1.0, -2.0, -3.0)).norm() < 1e-6);
        // A point at the camera centre maps to the origin in camera space.
        assert!(view.world_to_camera_point(c).norm() < 1e-5);
    }

    #[test]
    fn view_direction_is_unit() {
        let view = CameraView::new(
            Matrix3::identity(),
            Vector3::zeros(),
            PinholeCamera::new(1, 1, 1.0, 1.0, 0.0, 0.0),
        );
        let d = view.view_direction(Vector3::new(3.0, 4.0, 0.0));
        assert!((d.norm() - 1.0).abs() < 1e-6);
    }
}
