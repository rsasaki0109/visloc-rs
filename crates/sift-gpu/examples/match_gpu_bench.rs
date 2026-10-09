//! Batched GPU matching vs the CPU cross-checked ratio matcher on real
//! frames: SIFT (GPU) on the first N images of a directory, then every
//! pair (i, i + d) for d in 1..=window.
//!
//! cargo run --release -p visloc-sift-gpu --features gpu --example match_gpu_bench -- \
//!     --images <dir> [--max-images 40] [--window 5] [--max-keypoints 4000] [--cpu-pairs 20] [--u8]
//!
//! `--u8` also times a u8 bank (`FeatureBank::upload_u8`) and reports how
//! many of the f32 matches it keeps.

use std::path::PathBuf;
use std::time::Instant;

use visloc_sift_gpu::{FeatureBank, GpuContext, GpuMatcher, SiftGpu};
use visloc_vision::features::sift::{GrayImage, SiftConfig};
use visloc_vision::matching::{BruteForceMatcher, CrossCheckMatcher, Matcher};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--images").expect("--images <dir>"));
    let max_images: usize = arg(&args, "--max-images").map_or(40, |v| v.parse().unwrap());
    let window: usize = arg(&args, "--window").map_or(5, |v| v.parse().unwrap());
    let cpu_pairs: usize = arg(&args, "--cpu-pairs").map_or(20, |v| v.parse().unwrap());
    let config = SiftConfig {
        max_keypoints: arg(&args, "--max-keypoints").map_or(4000, |v| v.parse().unwrap()),
        ..SiftConfig::default()
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e == "png" || e == "jpg" || e == "JPG")
        })
        .collect();
    paths.sort();
    paths.truncate(max_images);

    let mut sift = SiftGpu::new(GpuContext::new().expect("gpu"));
    let t = Instant::now();
    let descs: Vec<Vec<Vec<f32>>> = paths
        .iter()
        .map(|p| {
            let g = image::open(p).unwrap().to_luma8();
            let px: Vec<f32> = g.as_raw().iter().map(|&b| b as f32).collect();
            let img = GrayImage::new(g.width() as usize, g.height() as usize, &px).unwrap();
            sift.extract(&img, &config).unwrap().1
        })
        .collect();
    println!(
        "sift: {} images, mean {} kps, {:.2}s",
        descs.len(),
        descs.iter().map(Vec::len).sum::<usize>() / descs.len().max(1),
        t.elapsed().as_secs_f64()
    );
    let pairs: Vec<(usize, usize)> = (0..descs.len())
        .flat_map(|i| (1..=window).map(move |d| (i, i + d)))
        .filter(|&(_, j)| j < descs.len())
        .collect();

    let ctx = sift.context();
    let t = Instant::now();
    let sets: Vec<&[Vec<f32>]> = descs.iter().map(Vec::as_slice).collect();
    let bank = FeatureBank::upload(ctx, &sets).unwrap();
    let matcher = GpuMatcher::new(ctx);
    let t_up = t.elapsed().as_secs_f64();
    // Warm-up, then timed.
    let _ = matcher.match_pairs(ctx, &bank, &pairs[..1], Some(0.8), true);
    let t = Instant::now();
    let gpu = matcher.match_pairs(ctx, &bank, &pairs, Some(0.8), true);
    let tg = t.elapsed().as_secs_f64();
    println!(
        "gpu: {} pairs in {:.3}s ({:.2} ms/pair, bank upload {:.3}s), mean {} matches",
        pairs.len(),
        tg,
        1e3 * tg / pairs.len() as f64,
        t_up,
        gpu.iter().map(Vec::len).sum::<usize>() / pairs.len().max(1)
    );
    // FNV-1a over every match (indices + distance bits): a kernel change
    // that keeps the per-element summation order must keep this digest.
    let mut digest = 0xcbf29ce484222325u64;
    for m in gpu.iter().flatten() {
        for v in [
            m.query_index as u64,
            m.train_index as u64,
            u64::from(m.distance.to_bits()),
        ] {
            for b in v.to_le_bytes() {
                digest = (digest ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
    }
    println!("gpu match digest {digest:016x}");

    if args.iter().any(|a| a == "--u8") {
        let t = Instant::now();
        let bank8 = FeatureBank::upload_u8(ctx, &sets).unwrap();
        let t_up8 = t.elapsed().as_secs_f64();
        let _ = matcher.match_pairs(ctx, &bank8, &pairs[..1], Some(0.8), true);
        let t = Instant::now();
        let g8 = matcher.match_pairs(ctx, &bank8, &pairs, Some(0.8), true);
        let t8 = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let f8 = matcher.match_pairs(ctx, &bank8, &pairs, Some(0.8), false);
        let tf8 = t.elapsed().as_secs_f64();
        let (mut same, mut n32, mut n8) = (0usize, 0usize, 0usize);
        for (a, b) in gpu.iter().zip(&g8) {
            let s: std::collections::HashSet<(usize, usize)> =
                a.iter().map(|m| (m.query_index, m.train_index)).collect();
            n32 += a.len();
            n8 += b.len();
            same += b
                .iter()
                .filter(|m| s.contains(&(m.query_index, m.train_index)))
                .count();
        }
        println!(
            "u8: {} pairs in {:.3}s ({:.2} ms/pair, bank upload {:.3}s), forward-only {:.2} ms/pair, mean {} matches ({} forward)",
            pairs.len(), t8, 1e3 * t8 / pairs.len() as f64, t_up8, 1e3 * tf8 / pairs.len() as f64,
            n8 / pairs.len().max(1), f8.iter().map(Vec::len).sum::<usize>() / pairs.len().max(1)
        );
        println!(
            "u8 vs f32: {same} shared of {n32} f32 / {n8} u8 matches ({:.2}% of f32 kept, {:.2}% of u8 new)",
            100.0 * same as f64 / n32.max(1) as f64,
            100.0 * (n8 - same) as f64 / n8.max(1) as f64
        );
    }
    let m = CrossCheckMatcher::new(BruteForceMatcher { ratio: Some(0.8) });
    let n = cpu_pairs.min(pairs.len());
    let t = Instant::now();
    let mut same = 0usize;
    let mut total = 0usize;
    for (p, &(i, j)) in pairs.iter().take(n).enumerate() {
        let cpu = m.match_descriptors(&descs[i], &descs[j]);
        let c: std::collections::HashSet<(usize, usize)> =
            cpu.iter().map(|m| (m.query_index, m.train_index)).collect();
        total += c.len().max(gpu[p].len());
        same += gpu[p]
            .iter()
            .filter(|m| c.contains(&(m.query_index, m.train_index)))
            .count();
    }
    let tc = t.elapsed().as_secs_f64();
    println!(
        "cpu: {n} pairs in {tc:.2}s ({:.1} ms/pair) -> gpu speedup {:.0}x | agreement {:.3}%",
        1e3 * tc / n.max(1) as f64,
        (tc / n.max(1) as f64) / (tg / pairs.len() as f64),
        100.0 * same as f64 / total.max(1) as f64
    );
}
