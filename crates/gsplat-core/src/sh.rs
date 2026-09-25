//! Spherical-harmonics colour evaluation for view-dependent appearance.
//!
//! The 3DGS convention stores colours as real spherical-harmonics coefficients
//! up to degree 3 (16 coefficients per channel). The observed colour for a
//! viewing direction `d` (pointing from the camera toward the Gaussian, in the
//! world frame) is
//!
//! ```text
//! color = SH_C0 * f_dc + sum_k SH_basis_k(d) * f_rest_k + 0.5
//! ```
//!
//! where `SH_C0 = 0.28209479177387814` and the `+0.5` offset maps the DC term
//! into `[0, 1]` the way the Inria renderer does. The basis functions below are
//! the standard real SH polynomials in the same ordering as `graphdeco`'s
//! `SHRotation`/`eval_sh` implementation.

use nalgebra::Vector3;

/// Degree-0 SH normalization constant (`1 / (2 * sqrt(pi))`).
pub const SH_C0: f32 = 0.282_094_8;

/// Real SH basis constants from the Inria `gaussian-splatting` reference
/// (`utils/sh_utils.py`), indexed by the flattened coefficient order.
pub const SH_BASIS_COEFFS: [f32; 15] = [
    0.488_602_5, // 1, -1
    0.488_602_5, // 1, 0
    0.488_602_5, // 1, 1
    1.092_548_5, // 2, -2
    1.092_548_5, // 2, -1
    0.315_391_6, // 2, 0
    1.092_548_5, // 2, 1
    0.546_274_2, // 2, 2
    0.590_043_6, // 3, -3
    2.890_611_4, // 3, -2
    0.457_045_8, // 3, -1
    0.373_176_3, // 3, 0
    0.457_045_8, // 3, 1
    1.445_305_7, // 3, 2
    0.590_043_6, // 3, 3
];

/// Evaluate the real SH basis up to `degree` (0..=3) for unit direction `dir`.
///
/// The returned slice has `(degree + 1)^2` entries, ordered as `(l, m)` with
/// `l = 0` first. `dir` is expected to be a unit vector; it is not renormalized
/// (callers pass a normalize of the view direction).
pub fn eval_sh_basis(degree: u32, dir: Vector3<f32>) -> Vec<f32> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    let mut out = Vec::with_capacity(((degree + 1) * (degree + 1)) as usize);
    out.push(SH_C0);
    if degree >= 1 {
        let c = &SH_BASIS_COEFFS;
        out.push(-c[0] * y);
        out.push(c[1] * z);
        out.push(-c[2] * x);
    }
    if degree >= 2 {
        let c = &SH_BASIS_COEFFS;
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, yz, xz) = (x * y, y * z, x * z);
        out.push(c[3] * xy);
        out.push(-c[4] * yz);
        out.push(c[5] * (2.0 * zz - xx - yy));
        out.push(-c[6] * xz);
        out.push(c[7] * (xx - yy));
    }
    if degree >= 3 {
        let c = &SH_BASIS_COEFFS;
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let xy = x * y;
        out.push(-c[8] * y * (3.0 * xx - yy));
        out.push(c[9] * xy * z);
        out.push(-c[10] * y * (4.0 * zz - xx - yy));
        out.push(c[11] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy));
        out.push(-c[12] * x * (4.0 * zz - xx - yy));
        out.push(c[13] * z * (xx - yy));
        out.push(-c[14] * x * (xx - 3.0 * yy));
    }
    out
}

/// Evaluate the view-dependent RGB colour of a Gaussian.
///
/// `sh_dc` is the degree-0 coefficient per channel and `sh_rest` is
/// channel-major higher-order coefficients (`channel * coeffs_per_channel + k`).
/// The result is the raw SH colour (the Inria `+0.5` DC offset is applied so the
/// value is roughly in `[0, 1]`).
pub fn eval_sh_color(
    degree: u32,
    dir: Vector3<f32>,
    sh_dc: &[f32; 3],
    sh_rest: &[f32],
    out: &mut [f32; 3],
) {
    let basis = eval_sh_basis(degree, dir);
    let coeffs_per_channel = basis.len() - 1;
    for c in 0..3 {
        let mut acc = SH_C0 * sh_dc[c];
        let base = c * coeffs_per_channel;
        for k in 0..coeffs_per_channel {
            if let Some(v) = sh_rest.get(base + k) {
                acc += basis[k + 1] * v;
            }
        }
        out[c] = acc + 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_only_is_constant() {
        // With no higher-order terms the colour must not depend on direction.
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        eval_sh_color(
            3,
            Vector3::new(1.0, 0.0, 0.0),
            &[0.2, 0.4, 0.6],
            &[],
            &mut a,
        );
        eval_sh_color(
            3,
            Vector3::new(0.0, 1.0, 0.0),
            &[0.2, 0.4, 0.6],
            &[],
            &mut b,
        );
        assert_eq!(a, b);
        assert!((a[0] - (SH_C0 * 0.2 + 0.5)).abs() < 1e-6);
    }

    #[test]
    fn basis_constant_at_identity_direction() {
        // z = 1: basis[1..4] = (0, c0, 0) for the degree-1 block.
        let b = eval_sh_basis(1, Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(b.len(), 4);
        assert!((b[0] - SH_C0).abs() < 1e-6);
        assert!(b[1].abs() < 1e-6);
        assert!((b[2] - SH_BASIS_COEFFS[1]).abs() < 1e-6);
        assert!(b[3].abs() < 1e-6);
    }

    #[test]
    fn basis_is_orthonormal_on_the_sphere() {
        // Real SH with the Inria / brush normalisation are orthonormal:
        // (4 pi) * mean over a uniform sphere sampling of b_i b_j = delta_ij.
        // Catches wrong per-band constants (they once were swapped).
        let n = 40_000;
        let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        let mut gram = [[0.0f64; 16]; 16];
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).sqrt();
            let phi = golden * i as f32;
            let b = eval_sh_basis(3, Vector3::new(r * phi.cos(), r * phi.sin(), z));
            for a in 0..16 {
                for c in 0..16 {
                    gram[a][c] += (b[a] * b[c]) as f64;
                }
            }
        }
        let scale = 4.0 * std::f64::consts::PI / n as f64;
        for (a, row) in gram.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                let want = if a == c { 1.0 } else { 0.0 };
                assert!(
                    (v * scale - want).abs() < 2e-3,
                    "<b{a}, b{c}> = {}",
                    v * scale
                );
            }
        }
    }

    #[test]
    fn basis_sizes() {
        assert_eq!(eval_sh_basis(0, Vector3::new(0.0, 0.0, 1.0)).len(), 1);
        assert_eq!(eval_sh_basis(1, Vector3::new(0.0, 0.0, 1.0)).len(), 4);
        assert_eq!(eval_sh_basis(2, Vector3::new(0.0, 0.0, 1.0)).len(), 9);
        assert_eq!(eval_sh_basis(3, Vector3::new(0.0, 0.0, 1.0)).len(), 16);
    }

    #[test]
    fn direction_changes_colour_with_rest() {
        let rest = vec![0.5f32; 45];
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        eval_sh_color(3, Vector3::new(1.0, 0.0, 0.0), &[0.0; 3], &rest, &mut a);
        eval_sh_color(3, Vector3::new(0.0, 1.0, 0.0), &[0.0; 3], &rest, &mut b);
        assert!((a[0] - b[0]).abs() > 1e-4);
    }
}
