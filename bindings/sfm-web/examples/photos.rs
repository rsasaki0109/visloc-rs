//! The browser SfM's functions natively on a folder of photos, one thread as
//! in the page, with the time each stage takes:
//!
//! ```text
//! cargo run --release --example photos -- <folder> [--max-size 800] [--max-keypoints 2000]
//!     [--focal F] [--window 8] [--exhaustive-max 40] [--out model.json]
//! ```

use std::time::Instant;

use visloc_core::types::Camera;
use visloc_sfm_web::{
    candidate_pairs, decode, extract, model_json, reconstruct, sift_config, verify_matches,
};
use visloc_vision::matching::BruteForceMatcher;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().ok_or("usage: photos <folder> [options]")?;
    let (mut max_size, mut max_keypoints, mut focal, mut window, mut exhaustive_max) =
        (800u32, 2000usize, 0.0f64, 8usize, 40usize);
    let mut out: Option<String> = None;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--max-size" => max_size = next()?.parse()?,
            "--max-keypoints" => max_keypoints = next()?.parse()?,
            "--focal" => focal = next()?.parse()?,
            "--window" => window = next()?.parse()?,
            "--exhaustive-max" => exhaustive_max = next()?.parse()?,
            "--out" => out = Some(next()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let mut files: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
        })
        .collect();
    files.sort();
    let cfg = sift_config(max_keypoints);
    let t = Instant::now();
    let mut photos = Vec::new();
    for f in &files {
        let img = decode(&std::fs::read(f)?, max_size)?;
        photos.push(extract(
            &f.file_name().unwrap().to_string_lossy(),
            &img,
            &cfg,
        )?);
    }
    let mean = photos
        .iter()
        .map(|p| p.features.keypoints.len())
        .sum::<usize>()
        / photos.len().max(1);
    println!(
        "{} photos at {}x{}, {mean} keypoints each: decode + SIFT {:.1} s",
        photos.len(),
        photos[0].width,
        photos[0].height,
        t.elapsed().as_secs_f64()
    );
    let (w, h) = (photos[0].width, photos[0].height);
    let f = if focal > 0.0 {
        focal
    } else {
        1.2 * w.max(h) as f64
    };
    let camera = Camera::pinhole(1, w, h, f, f, w as f64 / 2.0, h as f64 / 2.0);
    let t = Instant::now();
    let candidates = candidate_pairs(photos.len(), window, exhaustive_max);
    let matches: Vec<_> = candidates
        .iter()
        .map(|&(i, j)| {
            let (a, b) = (&photos[i].features, &photos[j].features);
            BruteForceMatcher { ratio: Some(0.8) }
                .match_descriptors_cross_checked(&a.descriptors, &b.descriptors)
        })
        .collect();
    let t_match = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let pairs: Vec<_> = candidates
        .iter()
        .zip(&matches)
        .filter_map(|(&(i, j), dm)| {
            let (a, b) = (&photos[i].features, &photos[j].features);
            verify_matches(&camera, a, b, dm, 15).map(|m| (i, j, m))
        })
        .collect();
    println!(
        "{} of {} pairs verified: matching {t_match:.1} s, verification {:.1} s",
        pairs.len(),
        candidates.len(),
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let model = reconstruct(&camera, &photos, &pairs)?;
    println!(
        "{} of {} photos registered, {} points, focal {:.1} px: mapper {:.1} s",
        model.cameras.len(),
        photos.len(),
        model.points.len(),
        model.intrinsics.0,
        t.elapsed().as_secs_f64()
    );
    if let Some(out) = out {
        std::fs::write(&out, model_json(&model, &photos))?;
        println!("wrote {out}");
    }
    Ok(())
}
