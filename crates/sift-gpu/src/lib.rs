//! wgpu compute SIFT for visloc-rs.
//!
//! `SiftGpu` runs visloc-vision's default SIFT path (DoG detector on a
//! nearest-doubled first octave, legacy orientation histogram, isotropic
//! 128-D descriptor with L2 or L1-root normalization) on the GPU in f32.
//! It is a quality-validated port, not a bit-exact one: compare it with
//! `visloc_vision::features::sift::extract_sift` by keypoint repeatability
//! and descriptor similarity (see `examples/sift_gpu_bench.rs`).
//!
//! Configurations outside that path (affine shapes, DSP, Hessian-Laplace,
//! the VLFeat-compatible detector/descriptor, scale-adaptive gradients,
//! standard orientation peaks) are rejected by `SiftGpu::supports` so
//! callers can fall back to the CPU extractor.

#[cfg(feature = "gpu")]
mod extractor;
#[cfg(feature = "gpu")]
mod matcher;

#[cfg(feature = "gpu")]
pub use extractor::{SiftGpu, SiftGpuError};
#[cfg(feature = "gpu")]
pub use matcher::{FeatureBank, GpuMatcher};
#[cfg(feature = "gpu")]
pub use visloc_gsplat_render::{try_context, GpuContext, GpuError};

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use visloc_vision::features::sift::{extract_sift, GrayImage, SiftConfig, SiftNormalization};

    /// Smooth deterministic texture with blobs at several scales (0..255).
    fn texture(w: usize, h: usize) -> Vec<f32> {
        let mut px = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64, y as f64);
                let mut v = 128.0
                    + 40.0 * (fx * 0.21).sin() * (fy * 0.17).cos()
                    + 25.0 * (fx * 0.05 + fy * 0.07).sin();
                for (i, &(cx, cy, s)) in [(30.0, 40.0, 3.0), (100.0, 60.0, 6.0), (150.0, 20.0, 2.0)]
                    .iter()
                    .enumerate()
                {
                    let d2 = (fx - cx).powi(2) + (fy - cy).powi(2);
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    v += sign * 60.0 * (-d2 / (2.0 * s * s)).exp();
                }
                px[y * w + x] = v.clamp(0.0, 255.0) as f32;
            }
        }
        px
    }

    fn compare(config: &SiftConfig) {
        let Some(ctx) = try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mut gpu = SiftGpu::new(ctx);
        let (w, h) = (192, 128);
        let px = texture(w, h);
        let img = GrayImage::new(w, h, &px).unwrap();
        let (ck, cd) = extract_sift(&img, config).unwrap();
        let (gk, gd) = gpu.extract(&img, config).unwrap();
        assert!(!ck.is_empty());
        let mut matched = 0;
        let mut worst = 1.0f32;
        for (ci, k) in ck.iter().enumerate() {
            let hit = gk.iter().position(|g| {
                (g.x - k.x).abs() < 1e-6
                    && (g.y - k.y).abs() < 1e-6
                    && (g.sigma - k.sigma).abs() < 1e-9
                    && {
                        let d = (g.orientation - k.orientation).rem_euclid(std::f64::consts::TAU);
                        d.min(std::f64::consts::TAU - d) < 0.02
                    }
            });
            if let Some(gi) = hit {
                matched += 1;
                let dot: f32 = cd[ci].iter().zip(&gd[gi]).map(|(a, b)| a * b).sum();
                worst = worst.min(dot);
            }
        }
        let ratio = matched as f64 / ck.len() as f64;
        assert!(
            ratio >= 0.98,
            "matched {matched}/{} (gpu {})",
            ck.len(),
            gk.len()
        );
        assert!(worst > 0.999, "worst descriptor dot {worst}");
    }

    #[test]
    fn matches_cpu_default() {
        compare(&SiftConfig::default());
    }

    #[test]
    fn matches_cpu_capped_l1root() {
        compare(&SiftConfig {
            max_keypoints: 40,
            max_orientations: 2,
            normalization: SiftNormalization::L1Root,
            ..SiftConfig::default()
        });
    }

    #[test]
    fn matcher_matches_cpu_cross_check() {
        use visloc_vision::matching::{BruteForceMatcher, CrossCheckMatcher, Matcher};
        let Some(ctx) = try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        // Deterministic pseudo-random unit descriptors; image b is a noisy
        // permuted copy of a, so there are true matches plus distractors.
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let unit = |v: Vec<f32>| {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / n).collect::<Vec<f32>>()
        };
        let a: Vec<Vec<f32>> = (0..300)
            .map(|_| unit((0..128).map(|_| rnd()).collect()))
            .collect();
        let mut b: Vec<Vec<f32>> = a
            .iter()
            .rev()
            .take(200)
            .map(|d| unit(d.iter().map(|x| x + 0.05 * rnd()).collect()))
            .collect();
        b.extend((0..77).map(|_| unit((0..128).map(|_| rnd()).collect())));
        let c: Vec<Vec<f32>> = (0..5)
            .map(|_| unit((0..128).map(|_| rnd()).collect()))
            .collect();

        // Edge cases: a single descriptor and an empty image.
        let d: Vec<Vec<f32>> = vec![a[7].clone()];
        let empty: Vec<Vec<f32>> = Vec::new();

        let bank = FeatureBank::upload(&ctx, &[&a, &b, &c, &d, &empty]).unwrap();
        let matcher = GpuMatcher::new(&ctx);
        let pairs = [
            (0usize, 1usize),
            (1, 0),
            (0, 2),
            (2, 1),
            (0, 3),
            (3, 0),
            (4, 0),
            (0, 4),
        ];
        let sets = [&a, &b, &c, &d, &empty];
        for cross in [false, true] {
            for ratio in [None, Some(0.8)] {
                let gpu = matcher.match_pairs(&ctx, &bank, &pairs, ratio, cross);
                for (p, &(i, j)) in pairs.iter().enumerate() {
                    let bf = BruteForceMatcher { ratio };
                    let cpu = if cross {
                        CrossCheckMatcher::new(bf).match_descriptors(sets[i], sets[j])
                    } else {
                        bf.match_descriptors(sets[i], sets[j])
                    };
                    let key = |m: &visloc_vision::matching::DescriptorMatch| {
                        (m.query_index, m.train_index)
                    };
                    let g: Vec<_> = gpu[p].iter().map(key).collect();
                    let c: Vec<_> = cpu.iter().map(key).collect();
                    assert_eq!(g, c, "pair {i}->{j} ratio {ratio:?} cross {cross}");
                    for (gm, cm) in gpu[p].iter().zip(&cpu) {
                        assert!((gm.distance - cm.distance).abs() < 1e-3);
                    }
                }
            }
        }
    }
}
