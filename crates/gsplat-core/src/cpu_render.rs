//! A slow, exact CPU reference rasterizer for 3D Gaussian Splatting.
//!
//! This is deliberately simple: it projects every Gaussian into the image,
//! computes the screen-space 2D covariance, and alpha-composites front-to-back
//! over the pixels its 3-sigma ellipse covers. It is *not* tile-sorted or
//! optimized; it exists to be an unambiguous reference for the future GPU
//! kernels (correctness before speed) and to render a frame without a GPU.
//!
//! The math follows the Inria `gaussian-splatting` CUDA rasterizer:
//!
//! - world → camera: `p_cam = R * mean + t`
//! - 3D covariance: `Σ = R_cov S Sᵀ R_covᵀ` where `S = diag(exp(scale_log))`
//!   and `R_cov` is the rotation of the Gaussian
//! - 2D covariance (the EWA splatting approximation, dropping the Jacobian's
//!   third column): `Σ' = J W Σ Wᵀ Jᵀ` with `W` the world-to-camera rotation
//!   and `J` the affine part of the projective Jacobian at the projected mean
//!   (of the camera's [`crate::lens::Lens`] for a distorted or fisheye one)
//! - screen-space extent: `x = mean2d + Σ'^(1/2) * z`, and a pixel is covered
//!   when `zᵀz <= 1`

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

use crate::camera::CameraView;
use crate::gaussian::Scene;
use crate::lens::{extended_lens_coords, lens_coords, Projection};
use crate::sh::eval_sh_color;

/// Rendered image (linear RGB, row-major, top-left origin).
#[derive(Debug, Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<[f32; 3]>,
}

impl Image {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            rgb: vec![[0.0, 0.0, 0.0]; (width * height) as usize],
        }
    }

    /// Convert to 8-bit RGB bytes (clamped to `[0, 1]`).
    ///
    /// Splat colours (`0.5 + SH_C0 * dc` plus higher SH) live in the same
    /// display space as the training images, as in the Inria and brush
    /// trainers, so no transfer curve is applied. (An earlier gamma-2.2
    /// encode here washed every render out.)
    pub fn to_rgb8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.rgb.len() * 3);
        for px in &self.rgb {
            for c in px {
                let v = c.clamp(0.0, 1.0);
                out.push((v * 255.0).round() as u8);
            }
        }
        out
    }

    pub fn pixel(&self, x: u32, y: u32) -> [f32; 3] {
        self.rgb[(y * self.width + x) as usize]
    }
}

/// A projected Gaussian ready for rasterization.
struct Projected {
    depth: f32,
    u: f32,
    v: f32,
    inv_cov: Matrix2<f32>,
    /// Half-extent in pixels along x / y of the blendable footprint.
    ext_x: f32,
    ext_y: f32,
    opacity: f32,
    color: [f32; 3],
}

/// The screen position, the EWA Jacobian (rows in pixels per camera-space
/// unit, third row zero) and the sort depth of camera-space point `p_cam`,
/// or `None` where the camera does not draw it.
///
/// A pinhole, distorted or not, draws what lies 0.1 in front of it, with
/// depth `z`, and linearises at a point clamped to 1.3x the frustum, as the
/// Inria rasterizer does: outside it the Jacobian blows up and near-camera
/// primitives would otherwise explode to huge footprints. An equidistant
/// (fisheye) lens draws what lies 0.1 from its centre within the wider of
/// [`crate::lens::Lens::max_theta`] and 90°, even behind the image plane,
/// at the point itself, with the distance as the depth. Past `max_theta` a
/// lens is its first-order extension there ([`extended_lens_coords`]).
fn project_point(view: &CameraView, p_cam: Vector3<f32>) -> Option<(f32, f32, Matrix3<f32>, f32)> {
    let cam = &view.camera;
    let (fx, fy) = (cam.fx, cam.fy);
    let lens = &cam.lens;
    let jacobian = |rows: [[f32; 3]; 2]| {
        Matrix3::new(
            fx * rows[0][0],
            fx * rows[0][1],
            fx * rows[0][2],
            fy * rows[1][0],
            fy * rows[1][1],
            fy * rows[1][2],
            0.0,
            0.0,
            0.0,
        )
    };
    let clamped = |p: [f32; 3]| {
        let inv_z = 1.0 / p[2];
        let lim_x = 1.3 * 0.5 * cam.width as f32 / fx;
        let lim_y = 1.3 * 0.5 * cam.height as f32 / fy;
        [
            (p[0] * inv_z).clamp(-lim_x, lim_x) * p[2],
            (p[1] * inv_z).clamp(-lim_y, lim_y) * p[2],
            p[2],
        ]
    };
    let p = [p_cam.x, p_cam.y, p_cam.z];
    if lens.is_pinhole() {
        if p_cam.z <= 0.1 {
            return None;
        }
        let (u, v, _z) = cam.project(p_cam);
        let inv_z = 1.0 / p_cam.z;
        let inv_z2 = inv_z * inv_z;
        let [tx, ty, _] = clamped(p);
        let j = jacobian([[inv_z, 0.0, -tx * inv_z2], [0.0, inv_z, -ty * inv_z2]]);
        return Some((u, v, j, p_cam.z));
    }
    let depth = match lens.projection {
        Projection::Pinhole => {
            if p_cam.z <= 0.1 {
                return None;
            }
            p_cam.z
        }
        Projection::Equidistant => {
            let distance = p_cam.norm();
            let widest = (lens.max_theta as f64).max(std::f64::consts::FRAC_PI_2);
            if distance <= 0.1 || crate::lens::ray_angle(p.map(f64::from)) > widest {
                return None;
            }
            distance
        }
    };
    let ([mu, mv], rows, q) = extended_lens_coords(lens, p);
    let rows = match lens.projection {
        Projection::Pinhole => lens_coords(lens, clamped(q)).1,
        Projection::Equidistant => rows,
    };
    Some((fx * mu + cam.cx, fy * mv + cam.cy, jacobian(rows), depth))
}

/// Render `scene` from `view` on the CPU.
///
/// Gaussians are composited **front-to-back** (nearest first) with per-pixel
/// transmittance, matching the Inria CUDA rasterizer and the GPU renderer in
/// `visloc-gsplat-render`. `bg` is the background colour in linear RGB.
pub fn render(scene: &Scene, view: &CameraView, bg: [f32; 3]) -> Image {
    let w = view.camera.width;
    let h = view.camera.height;
    let mut image = Image::new(w, h); // zero-initialized; background added at the end

    // World-to-camera rotation, reused for every Gaussian.
    let rot_wc = view.rotation;

    // Phase 1: project every Gaussian and discard the ones that cannot
    // contribute.
    let mut projected: Vec<Projected> = Vec::with_capacity(scene.len());
    for g in &scene.gaussians {
        let scale = g.scale();
        let opacity = g.opacity();
        if opacity < 1e-4 {
            continue;
        }
        let p_cam = view.world_to_camera_point(g.mean);
        let Some((u, v, j, depth)) = project_point(view, p_cam) else {
            continue;
        };

        use nalgebra::UnitQuaternion;
        let r_mat = UnitQuaternion::from_quaternion(g.unit_rotation())
            .to_rotation_matrix()
            .into_inner();
        let s2 = Vector3::new(scale.x * scale.x, scale.y * scale.y, scale.z * scale.z);
        let sigma = r_mat * Matrix3::from_diagonal(&s2) * r_mat.transpose();

        let cov_cam = rot_wc * sigma * rot_wc.transpose();
        let cov2 = j * cov_cam * j.transpose();
        let a = cov2[(0, 0)] + 0.3;
        let b = cov2[(0, 1)];
        let c = cov2[(1, 1)] + 0.3;
        let cov = Matrix2::new(a, b, b, c);
        let det = cov.determinant();
        if !(det.is_finite()) || det <= 0.0 {
            continue;
        }
        let inv_cov = cov.try_inverse().unwrap_or_else(Matrix2::zeros);

        let mid = 0.5 * (a + c);
        let rad = (mid * mid - det).max(0.0).sqrt();
        let lambda2 = mid - rad;
        if lambda2 <= 0.0 {
            continue;
        }
        // Only pixels with opacity * exp(-power) >= 1/255 are blended, i.e.
        // d^T C^-1 d <= k = 2 ln(255 * opacity); bound that ellipse per axis
        // (same extent as the GPU `compute_projected`).
        let k = 2.0 * (255.0 * opacity).ln();
        if !k.is_finite() || k <= 0.0 {
            continue;
        }
        let ext_x = (k * a).sqrt();
        let ext_y = (k * c).sqrt();
        if !ext_x.is_finite() || !ext_y.is_finite() {
            continue;
        }

        let dir = view.view_direction(g.mean);
        let mut color = [0.0f32; 3];
        eval_sh_color(g.sh_degree, dir, &g.sh_dc, &g.sh_rest, &mut color);

        if u + ext_x < 0.0
            || v + ext_y < 0.0
            || u - ext_x > (w - 1) as f32
            || v - ext_y > (h - 1) as f32
        {
            continue;
        }

        projected.push(Projected {
            depth,
            u,
            v,
            inv_cov,
            ext_x,
            ext_y,
            opacity,
            color,
        });
    }

    // Phase 2: composite front-to-back (nearest first) with transmittance.
    projected.sort_by(|a, b| {
        a.depth
            .partial_cmp(&b.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut trans = vec![1.0f32; (w * h) as usize];
    for p in &projected {
        let x0 = ((p.u - p.ext_x).floor().max(0.0)) as u32;
        let y0 = ((p.v - p.ext_y).floor().max(0.0)) as u32;
        let x1 = ((p.u + p.ext_x).ceil().min((w - 1) as f32).max(0.0)) as u32;
        let y1 = ((p.v + p.ext_y).ceil().min((h - 1) as f32).max(0.0)) as u32;

        for py in y0..=y1 {
            for pxi in x0..=x1 {
                let dx = pxi as f32 + 0.5 - p.u;
                let dy = py as f32 + 0.5 - p.v;
                let d = Vector2::new(dx, dy);
                let power = -0.5 * (d.transpose() * p.inv_cov * d)[(0, 0)];
                if power > 0.0 {
                    continue;
                }
                let alpha = (p.opacity * power.exp()).min(0.99);
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let idx = (py * w + pxi) as usize;
                let t = trans[idx];
                if t <= 1e-4 {
                    continue;
                }
                let contrib = t * alpha;
                let px = &mut image.rgb[idx];
                for (dst, src) in px.iter_mut().zip(p.color.iter()) {
                    *dst += contrib * src.max(0.0);
                }
                trans[idx] = t * (1.0 - alpha);
            }
        }
    }

    // Add the background through the remaining transmittance.
    for (i, px) in image.rgb.iter_mut().enumerate() {
        let t = trans[i];
        for (dst, bg_c) in px.iter_mut().zip(bg.iter()) {
            *dst += t * bg_c;
        }
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::PinholeCamera;
    use crate::gaussian::Gaussian;
    use nalgebra::{Matrix3, Quaternion};

    fn front_camera() -> CameraView {
        // Camera at origin looking down +Z with identity rotation.
        CameraView::new(
            Matrix3::identity(),
            Vector3::zeros(),
            PinholeCamera::new(64, 64, 50.0, 50.0, 32.0, 32.0),
        )
    }

    fn solid_gaussian(mean: Vector3<f32>, scale: f32, color: [f32; 3]) -> Gaussian {
        Gaussian {
            mean,
            scale_log: Vector3::new(scale.ln(), scale.ln(), scale.ln()),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            opacity_logit: 10.0, // nearly opaque
            sh_dc: [
                (color[0] - 0.5) / crate::sh::SH_C0,
                (color[1] - 0.5) / crate::sh::SH_C0,
                (color[2] - 0.5) / crate::sh::SH_C0,
            ],
            sh_rest: Vec::new(),
            sh_degree: 0,
        }
    }

    #[test]
    fn empty_scene_is_background() {
        let scene = Scene::new(vec![], 0);
        let img = render(&scene, &front_camera(), [0.1, 0.2, 0.3]);
        assert_eq!(img.width, 64);
        assert_eq!(img.pixel(0, 0), [0.1, 0.2, 0.3]);
    }

    #[test]
    fn centered_gaussian_paints_center() {
        // A small red blob at (0, 0, 5) projects to the image centre.
        let g = solid_gaussian(Vector3::new(0.0, 0.0, 5.0), 0.2, [1.0, 0.0, 0.0]);
        let scene = Scene::new(vec![g], 0);
        let img = render(&scene, &front_camera(), [0.0, 0.0, 0.0]);
        let center = img.pixel(32, 32);
        // Red channel should dominate; green/blue stay near zero.
        assert!(center[0] > 0.5, "center red was {}", center[0]);
        assert!(center[1] < 0.1, "center green was {}", center[1]);
        // A corner stays background.
        assert!(img.pixel(0, 0)[0] < 0.05);
    }

    #[test]
    fn behind_camera_is_skipped() {
        let g = solid_gaussian(Vector3::new(0.0, 0.0, -5.0), 0.2, [1.0, 1.0, 1.0]);
        let scene = Scene::new(vec![g], 0);
        let img = render(&scene, &front_camera(), [0.0, 0.0, 0.0]);
        assert!(img.pixel(32, 32)[0] < 1e-6);
    }

    #[test]
    fn output_is_deterministic() {
        let g = solid_gaussian(Vector3::new(0.05, -0.03, 5.0), 0.15, [0.2, 0.7, 0.4]);
        let scene = Scene::new(vec![g], 0);
        let a = render(&scene, &front_camera(), [0.0; 3]);
        let b = render(&scene, &front_camera(), [0.0; 3]);
        assert_eq!(a.to_rgb8(), b.to_rgb8());
    }
}
