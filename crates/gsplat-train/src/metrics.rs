//! Image-quality metrics on display-space RGB in `[0, 1]`.

/// Peak signal-to-noise ratio (dB) between a render and its ground truth, both
/// row-major RGB in `[0, 1]`. The render is clamped to `[0, 1]` first, as it
/// would be when written to an 8-bit image. Returns `f64::INFINITY` for an
/// exact match.
///
/// # Panics
/// If the two images have different lengths.
pub fn psnr(render: &[[f32; 3]], ground_truth: &[[f32; 3]]) -> f64 {
    psnr_masked(render, ground_truth, None)
}

/// [`psnr`] over the pixels where `valid` is true (all of them for `None`),
/// e.g. leaving out masked-out content. Infinite when no pixel is valid.
///
/// # Panics
/// If the images (or the mask) have different lengths.
pub fn psnr_masked(render: &[[f32; 3]], ground_truth: &[[f32; 3]], valid: Option<&[bool]>) -> f64 {
    assert_eq!(render.len(), ground_truth.len(), "image sizes differ");
    if let Some(v) = valid {
        assert_eq!(v.len(), render.len(), "mask size differs");
    }
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (i, (r, g)) in render.iter().zip(ground_truth).enumerate() {
        if valid.is_some_and(|v| !v[i]) {
            continue;
        }
        n += 1;
        for c in 0..3 {
            let d = (r[c].clamp(0.0, 1.0) - g[c]) as f64;
            sum += d * d;
        }
    }
    let mse = sum / (n * 3).max(1) as f64;
    if n == 0 || mse == 0.0 {
        f64::INFINITY
    } else {
        -10.0 * mse.log10()
    }
}

/// 11-tap gaussian window (sigma 1.5), normalised; the SSIM window of the
/// Inria code and of the GPU loss (`shaders/ssim.wgsl`).
fn gauss_window() -> [f64; 11] {
    let mut w = [0.0; 11];
    for (k, x) in w.iter_mut().enumerate() {
        let d = k as f64 - 5.0;
        *x = (-d * d / (2.0 * 1.5 * 1.5)).exp();
    }
    let s: f64 = w.iter().sum();
    w.map(|x| x / s)
}

/// Separable zero-padded blur of a single-channel `w x h` image.
fn blur(img: &[f64], w: usize, h: usize) -> Vec<f64> {
    let g = gauss_window();
    let mut tmp = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, gk) in g.iter().enumerate() {
                let xx = x as i64 + k as i64 - 5;
                if xx >= 0 && (xx as usize) < w {
                    acc += gk * img[y * w + xx as usize];
                }
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, gk) in g.iter().enumerate() {
                let yy = y as i64 + k as i64 - 5;
                if yy >= 0 && (yy as usize) < h {
                    acc += gk * tmp[yy as usize * w + x];
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Mean SSIM over pixels and channels (11x11 gaussian window, sigma 1.5,
/// zero padding, C1 = 0.01^2, C2 = 0.03^2 -- the Inria training SSIM), and
/// optionally its gradient w.r.t. `render`. Images are row-major RGB, `w x h`;
/// the render is not clamped (this is the training loss, not a display metric).
pub fn ssim_with_grad(
    render: &[[f64; 3]],
    ground_truth: &[[f64; 3]],
    w: usize,
    h: usize,
    want_grad: bool,
) -> (f64, Option<Vec<[f64; 3]>>) {
    assert_eq!(render.len(), w * h, "render size");
    assert_eq!(ground_truth.len(), w * h, "ground truth size");
    const C1: f64 = 0.01 * 0.01;
    const C2: f64 = 0.03 * 0.03;
    let n = (w * h) as f64;
    let mut total = 0.0;
    let mut grad = want_grad.then(|| vec![[0.0; 3]; w * h]);
    for c in 0..3 {
        let x: Vec<f64> = render.iter().map(|p| p[c]).collect();
        let y: Vec<f64> = ground_truth.iter().map(|p| p[c]).collect();
        let sq = |v: &[f64]| v.iter().map(|a| a * a).collect::<Vec<_>>();
        let xy: Vec<f64> = x.iter().zip(&y).map(|(a, b)| a * b).collect();
        let (mx, my) = (blur(&x, w, h), blur(&y, w, h));
        let (sxx, syy, sxy) = (blur(&sq(&x), w, h), blur(&sq(&y), w, h), blur(&xy, w, h));
        let mut a_map = vec![0.0; w * h];
        let mut b_map = vec![0.0; w * h];
        let mut c_map = vec![0.0; w * h];
        for q in 0..w * h {
            let a = 2.0 * mx[q] * my[q] + C1;
            let b = 2.0 * (sxy[q] - mx[q] * my[q]) + C2;
            let cc = mx[q] * mx[q] + my[q] * my[q] + C1;
            let d = (sxx[q] - mx[q] * mx[q]) + (syy[q] - my[q] * my[q]) + C2;
            let s = a * b / (cc * d);
            total += s;
            a_map[q] = (2.0 * my[q] * b - 2.0 * my[q] * a) / (cc * d)
                - s * (2.0 * mx[q] / cc - 2.0 * mx[q] / d);
            b_map[q] = -s / d;
            c_map[q] = 2.0 * a / (cc * d);
        }
        if let Some(gr) = grad.as_mut() {
            let (ga, gb, gc) = (blur(&a_map, w, h), blur(&b_map, w, h), blur(&c_map, w, h));
            for p in 0..w * h {
                gr[p][c] = (ga[p] + 2.0 * x[p] * gb[p] + y[p] * gc[p]) / (3.0 * n);
            }
        }
    }
    (total / (3.0 * n), grad)
}

/// Mean SSIM of a render (clamped to `[0, 1]`, as displayed) against its
/// ground truth, both `[0, 1]` RGB, `w x h`.
pub fn ssim(render: &[[f32; 3]], ground_truth: &[[f32; 3]], w: usize, h: usize) -> f64 {
    let r: Vec<[f64; 3]> = render
        .iter()
        .map(|p| p.map(|v| v.clamp(0.0, 1.0) as f64))
        .collect();
    let g: Vec<[f64; 3]> = ground_truth.iter().map(|p| p.map(|v| v as f64)).collect();
    ssim_with_grad(&r, &g, w, h, false).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let mut v = [0.0; 3];
                for x in &mut v {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    *x = (s >> 11) as f64 / (1u64 << 53) as f64;
                }
                v
            })
            .collect()
    }

    #[test]
    fn ssim_is_one_for_identical_images() {
        let img = noise(20 * 14, 3);
        let (s, _) = ssim_with_grad(&img, &img, 20, 14, false);
        assert!((s - 1.0).abs() < 1e-9, "ssim {s}");
    }

    #[test]
    fn ssim_gradient_matches_finite_differences() {
        let (w, h) = (17, 13);
        let x = noise(w * h, 5);
        let y = noise(w * h, 9);
        let (_, g) = ssim_with_grad(&x, &y, w, h, true);
        let g = g.unwrap();
        for &p in &[0usize, 5, w + 3, 6 * w + 8, w * h - 1] {
            for c in 0..3 {
                let eps = 1e-6;
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp[p][c] += eps;
                xm[p][c] -= eps;
                let fd = (ssim_with_grad(&xp, &y, w, h, false).0
                    - ssim_with_grad(&xm, &y, w, h, false).0)
                    / (2.0 * eps);
                assert!(
                    (fd - g[p][c]).abs() < 1e-6 * fd.abs().max(1e-3),
                    "pixel {p} channel {c}: analytic {} vs FD {fd}",
                    g[p][c]
                );
            }
        }
    }

    #[test]
    fn psnr_of_uniform_error() {
        // Every channel off by 0.1 -> MSE 0.01 -> 20 dB.
        let gt = vec![[0.5f32; 3]; 16];
        let r = vec![[0.6f32; 3]; 16];
        assert!((psnr(&r, &gt) - 20.0).abs() < 1e-4);
    }

    #[test]
    fn psnr_masked_skips_invalid_pixels() {
        let r = vec![[0.5f32; 3], [0.0; 3]];
        let gt = vec![[0.6f32; 3], [1.0; 3]];
        let full = psnr(&r, &gt);
        let masked = psnr_masked(&r, &gt, Some(&[true, false]));
        assert!((masked - 20.0).abs() < 1e-4, "{masked}");
        assert!(full < masked);
        assert!(psnr_masked(&r, &gt, Some(&[false, false])).is_infinite());
    }

    #[test]
    fn psnr_clamps_render() {
        // A render overshooting to 2.0 scores as 1.0.
        let gt = vec![[1.0f32; 3]; 4];
        let r = vec![[2.0f32; 3]; 4];
        assert!(psnr(&r, &gt).is_infinite());
    }
}
