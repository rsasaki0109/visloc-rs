//! Export a trained splat for web viewers (antimatter15 `.splat`).
//!
//! ```text
//! cargo run --release -p visloc-gsplat-core --example gsplat_export_web -- \
//!     --ply scene.ply --out scene.splat [--max 400000] [--cameras sparse/0/images.txt]
//! ```
//!
//! Keeps the `--max` most important gaussians (opacity x volume, the order
//! antimatter15's viewer sorts by) and writes them largest first, so a
//! viewer that streams the file shows the coarse scene early. With
//! `--cameras`, the COLMAP poses are also written next to the output as
//! `<out>.cameras.json` (intrinsics; per view the camera centre and the
//! world-to-camera rotation) so the
//! viewer can start from, and step through, the capture viewpoints.

use std::path::PathBuf;

use visloc_gsplat_core::gaussian::Scene;
use visloc_gsplat_core::ply::load_ply;
use visloc_gsplat_core::splat::save_splat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut ply: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut max = 400_000usize;
    let mut cameras: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--ply" => ply = Some(PathBuf::from(next()?)),
            "--out" => out = Some(PathBuf::from(next()?)),
            "--max" => max = next()?.parse()?,
            "--cameras" => cameras = Some(PathBuf::from(next()?)),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply is required")?;
    let out = out.ok_or("--out is required")?;
    let scene = load_ply(&ply)?;
    let n = scene.len();
    let importance = |g: &visloc_gsplat_core::gaussian::Gaussian| {
        let opacity = 1.0 / (1.0 + (-g.opacity_logit).exp());
        opacity * (g.scale_log.x + g.scale_log.y + g.scale_log.z).exp()
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        importance(&scene.gaussians[b]).total_cmp(&importance(&scene.gaussians[a]))
    });
    order.truncate(max);
    let kept = Scene::new(
        order.iter().map(|&i| scene.gaussians[i].clone()).collect(),
        scene.sh_degree,
    );
    save_splat(&kept, &out)?;
    println!(
        "{} of {n} gaussians -> {} ({:.1} MB)",
        kept.len(),
        out.display(),
        std::fs::metadata(&out)?.len() as f64 / 1e6
    );

    if let Some(path) = cameras {
        let text = std::fs::read_to_string(&path)?;
        let mut rows = Vec::new();
        // Per image a pose line, then its 2D points: a line of its own, empty
        // when it has none (a long one must not be read as a pose).
        let lines: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).collect();
        for line in lines.iter().step_by(2) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 10 {
                continue;
            }
            let v: Vec<f64> = f[1..8].iter().map(|x| x.parse().unwrap_or(0.0)).collect();
            let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                v[0], v[1], v[2], v[3],
            ));
            let r = q.to_rotation_matrix();
            let t = nalgebra::Vector3::new(v[4], v[5], v[6]);
            let c = -(r.transpose() * t);
            let m = r.matrix();
            rows.push((
                f[9].to_string(),
                format!(
                    "{{\"name\":\"{}\",\"center\":[{},{},{}],\"rotation\":[[{},{},{}],[{},{},{}],[{},{},{}]]}}",
                    f[9],
                    c.x,
                    c.y,
                    c.z,
                    m[(0, 0)],
                    m[(0, 1)],
                    m[(0, 2)],
                    m[(1, 0)],
                    m[(1, 1)],
                    m[(1, 2)],
                    m[(2, 0)],
                    m[(2, 1)],
                    m[(2, 2)]
                ),
            ));
        }
        rows.sort();
        // Intrinsics from the sibling cameras.txt (first camera).
        let intr = std::fs::read_to_string(path.with_file_name("cameras.txt"))
            .ok()
            .and_then(|t| {
                let l = t
                    .lines()
                    .find(|l| !l.starts_with('#') && !l.trim().is_empty())?;
                let f: Vec<&str> = l.split_whitespace().collect();
                let n = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok());
                let (w, h) = (n(2)?, n(3)?);
                // PINHOLE / OPENCV: fx fy cx cy; SIMPLE_*: f cx cy.
                let (fx, fy) = if f[1] == "PINHOLE" || f[1] == "OPENCV" {
                    (n(4)?, n(5)?)
                } else {
                    (n(4)?, n(4)?)
                };
                Some(format!(
                    "\"width\":{w},\"height\":{h},\"fx\":{fx},\"fy\":{fy},"
                ))
            })
            .unwrap_or_default();
        let json = format!(
            "{{{intr}\"views\":[{}]}}",
            rows.into_iter()
                .map(|r| r.1)
                .collect::<Vec<_>>()
                .join(",\n")
        );
        let cam_out = out.with_extension("cameras.json");
        std::fs::write(&cam_out, json)?;
        println!("cameras -> {}", cam_out.display());
    }
    Ok(())
}
