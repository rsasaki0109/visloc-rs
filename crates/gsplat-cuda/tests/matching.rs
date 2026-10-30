//! The CUDA matcher (`matching::CudaPairMatcher`) against the CPU matcher
//! `build_photo_dataset` uses, `BruteForceMatcher::match_descriptors_cross_checked`
//! with ratio 0.8: the same matches, bit for bit. Needs `--features photos`;
//! skipped unless `VISLOC_GSPLAT_CUDA_LIB` points at the library built by
//! `scripts/build_gsplat_cuda.sh`.
#![cfg(feature = "photos")]

use visloc_gsplat_cuda::matching::CudaPairMatcher;
use visloc_gsplat_cuda::CudaGsplat;
use visloc_gsplat_train::photos::PairMatcher;
use visloc_vision::matching::BruteForceMatcher;

fn gsplat() -> Option<CudaGsplat> {
    match CudaGsplat::from_env() {
        None => {
            eprintln!("skipping: {} is not set", visloc_gsplat_cuda::LIBRARY_ENV);
            None
        }
        Some(r) => Some(r.expect("load the CUDA library")),
    }
}

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }
}

/// `n` descriptors of 128 bytes / `scale`: the first `shared` are noisy
/// copies of `base`'s (so they match), a few are exact duplicates of
/// earlier rows (ties), the rest random.
fn image(
    rng: &mut Rng,
    n: usize,
    base: Option<&[Vec<f32>]>,
    shared: usize,
    scale: f32,
) -> Vec<Vec<f32>> {
    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(n);
    for k in 0..n {
        let row: Vec<u8> = match base {
            Some(b) if k < shared.min(b.len()) => b[k]
                .iter()
                .map(|&v| {
                    let noise = (rng.next() % 9) as i32 - 4;
                    ((v * scale) as i32 + noise).clamp(0, 255) as u8
                })
                .collect(),
            _ if k > 8 && rng.next().is_multiple_of(16) => rows[(rng.next() as usize) % k].clone(),
            _ => (0..128).map(|_| (rng.next() % 256) as u8).collect(),
        };
        rows.push(row);
    }
    rows.into_iter()
        .map(|r| r.into_iter().map(|b| f32::from(b) / scale).collect())
        .collect()
}

fn check(scale: f32) {
    let Some(gs) = gsplat() else { return };
    let mut rng = Rng(7);
    // Counts off the 128-row tiles, an empty image, and a one-row image.
    let first = image(&mut rng, 700, None, 0, scale);
    let images = [
        first.clone(),
        image(&mut rng, 333, Some(&first), 250, scale),
        image(&mut rng, 129, Some(&first), 100, scale),
        Vec::new(),
        image(&mut rng, 1, Some(&first), 1, scale),
        image(&mut rng, 1000, Some(&first), 600, scale),
    ];
    let sets: Vec<&[Vec<f32>]> = images.iter().map(|v| v.as_slice()).collect();
    let matcher = CudaPairMatcher::new(gs);
    matcher.set_features(&sets).expect("set features");
    let mut pairs = Vec::new();
    for i in 0..images.len() {
        for j in 0..images.len() {
            if i != j {
                pairs.push((i, j));
            }
        }
    }
    let gpu = matcher.match_pairs(&pairs).expect("match");
    let cpu = BruteForceMatcher { ratio: Some(0.8) };
    let mut total = 0;
    for (k, &(i, j)) in pairs.iter().enumerate() {
        let want = cpu.match_descriptors_cross_checked(&images[i], &images[j]);
        assert_eq!(gpu[k], want, "pair ({i}, {j})");
        total += want.len();
    }
    eprintln!(
        "scale {scale}: {} pairs, {total} matches, identical",
        pairs.len()
    );
    assert!(total > 1000, "too few matches to compare");
}

#[test]
fn matches_equal_the_cpu_rootsift_bytes() {
    check(512.0);
}

#[test]
fn matches_equal_the_cpu_raw_bytes() {
    check(1.0);
}
