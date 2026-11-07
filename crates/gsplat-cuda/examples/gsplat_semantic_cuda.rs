//! Label a trained splat's gaussians by class from per-pixel labels, with the
//! CUDA renderer.
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --example gsplat_semantic_cuda -- \
//!   --ply scene.ply --data <colmap_root> --labels <dir> --out semantic.ply \
//!   [--web <dir> [--max 400000]]
//! ```
//!
//! `<dir>` holds `classes.txt` (`index r g b name` per line) and a label
//! image per view, `<image name>.png` with the extension replaced: 8-bit grey,
//! each pixel's class index, 255 for none (`scripts/sam3_labels.py` writes
//! them). A label image of another size is resized to the view's, nearest
//! pixel. Every view of the model (training and held out) votes: at each
//! labelled pixel, each splat's blending weight there goes to the pixel's
//! class, through the view's lens. A gaussian takes the class with the most
//! weight (`visloc_gsplat_core::semantic`).
//!
//! `--out` gets the splat painted by class (degree-0 SH, the geometry
//! unchanged) with two extra float properties per gaussian: `class` (255:
//! no labelled pixel saw it) and `class_confidence`, that class's share of
//! the gaussian's weight. `--web <dir>` also writes, for the web viewer
//! (`docs/euroc_splat`, `?url=scene.splat&semantic=semantic.splat&cameras=
//! cameras.json`), the RGB and the class splats with the same `--max`
//! gaussians in the same order (`scene.splat`, `semantic.splat`), the legend
//! (`semantic.json`) and the capture viewpoints (`cameras.json`).
//!
//! `--check` holds out every `--holdout`-th view (8, as `gsplat_train_cuda`)
//! and lets only the others vote, then renders the painted splat at each
//! held-out view and scores it against that view's labels:
//! pixel accuracy and IoU per class (a pixel's class: the class colour
//! nearest its render).

use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use visloc_gsplat_core::gaussian::Scene;
use visloc_gsplat_core::ply::{load_ply, save_ply_with};
use visloc_gsplat_core::semantic::{classes_json, colour_by_class, label_gaussians, parse_classes};
use visloc_gsplat_core::splat::{importance_order, save_splat};
use visloc_gsplat_cuda::CudaGsplat;
use visloc_gsplat_render::packing::PackedScene;
use visloc_gsplat_train::dataset::{load_colmap_dataset, View};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut ply: Option<PathBuf> = None;
    let mut data: Option<PathBuf> = None;
    let mut labels: Option<PathBuf> = None;
    let mut out = PathBuf::from("semantic.ply");
    let mut web: Option<PathBuf> = None;
    let mut max = 400_000usize;
    let mut check = false;
    let mut holdout = 8usize;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--ply" => ply = Some(PathBuf::from(next()?)),
            "--data" => data = Some(PathBuf::from(next()?)),
            "--labels" => labels = Some(PathBuf::from(next()?)),
            "--out" => out = PathBuf::from(next()?),
            "--web" => web = Some(PathBuf::from(next()?)),
            "--max" => max = next()?.parse()?,
            "--check" => check = true,
            "--holdout" => holdout = next()?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply <scene.ply> is required")?;
    let data = data.ok_or("--data <colmap root> is required")?;
    let labels = labels.ok_or("--labels <dir> is required")?;

    let classes = parse_classes(&std::fs::read_to_string(labels.join("classes.txt"))?)?;
    let k = classes.len();
    let scene = load_ply(&ply)?;
    let dataset = load_colmap_dataset(&data, check.then_some(holdout))?;
    let views: Vec<_> = dataset.train.iter().chain(&dataset.eval).collect();
    println!(
        "{} gaussians, {} views, {k} classes",
        scene.len(),
        views.len()
    );
    // A view's label image at its size, if there is one.
    let view_labels = |v: &View| -> Result<Option<Vec<u8>>, image::ImageError> {
        let path = labels.join(Path::new(&v.name).with_extension("png"));
        if !path.exists() {
            return Ok(None);
        }
        let (w, h) = (v.camera.camera.width, v.camera.camera.height);
        let mut img = image::open(&path)?.to_luma8();
        if img.dimensions() != (w, h) {
            img = image::imageops::resize(&img, w, h, FilterType::Nearest);
        }
        Ok(Some(img.into_raw()))
    };

    let mut gs = CudaGsplat::from_env().ok_or(
        "set VISLOC_GSPLAT_CUDA_LIB to the library built by scripts/build_gsplat_cuda.sh",
    )??;
    gs.set_scene(&PackedScene::from_scene(&scene))?;
    let t0 = std::time::Instant::now();
    let mut voted = 0usize;
    // With --check only the training views vote.
    let voters = if check {
        dataset.train.len()
    } else {
        views.len()
    };
    for v in &views[..voters] {
        if let Some(l) = view_labels(v)? {
            gs.vote_labels(&v.camera, &l, k, voted == 0)?;
            voted += 1;
        }
    }
    if voted == 0 {
        return Err(format!("no label image in {} for any view", labels.display()).into());
    }
    let votes = gs.votes(k)?;
    let per = label_gaussians(&votes, k);
    println!(
        "{voted} of {} views voted in {:.1} s",
        views.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut counts = vec![0usize; k];
    let mut unseen = 0usize;
    let mut confidence = 0.0f64;
    for l in &per {
        match counts.get_mut(l.class as usize) {
            Some(c) => {
                *c += 1;
                confidence += l.confidence as f64;
            }
            None => unseen += 1,
        }
    }
    let n = scene.len().max(1) as f64;
    for (c, &count) in classes.iter().zip(&counts) {
        println!("  {:>6.2}% {}", 100.0 * count as f64 / n, c.name);
    }
    println!(
        "  {:>6.2}% seen by no labelled pixel; mean confidence {:.3}",
        100.0 * unseen as f64 / n,
        confidence / (scene.len() - unseen).max(1) as f64
    );

    let painted = colour_by_class(&scene, &per, &classes);
    let class: Vec<f32> = per.iter().map(|l| l.class as f32).collect();
    let conf: Vec<f32> = per.iter().map(|l| l.confidence).collect();
    save_ply_with(
        &painted,
        &[("class", &class), ("class_confidence", &conf)],
        &out,
    )?;
    println!("wrote {}", out.display());

    if check {
        // Each held-out pixel's class in the painted splat: the class colour
        // nearest its render (black, the background, for none).
        gs.set_scene(&PackedScene::from_scene(&painted))?;
        let colors: Vec<[f32; 3]> = classes
            .iter()
            .map(|c| c.color.map(|v| v as f32 / 255.0))
            .chain([[0.0; 3]])
            .collect();
        let (mut inter, mut union) = (vec![0u64; k + 1], vec![0u64; k + 1]);
        let (mut right, mut total, mut checked) = (0u64, 0u64, 0usize);
        for v in &dataset.eval {
            let Some(truth) = view_labels(v)? else {
                continue;
            };
            checked += 1;
            let frame = gs.render(&v.camera, [0.0; 3], 0)?;
            for (&t, rgb) in truth.iter().zip(frame.rgb.chunks_exact(3)) {
                let t = t as usize;
                if t >= k {
                    continue;
                }
                let dist = |c: &[f32; 3]| (0..3).map(|i| (c[i] - rgb[i]).powi(2)).sum::<f32>();
                let pred = (0..=k)
                    .min_by(|&a, &b| dist(&colors[a]).total_cmp(&dist(&colors[b])))
                    .unwrap_or(k);
                total += 1;
                union[t] += 1;
                if pred == t {
                    right += 1;
                    inter[t] += 1;
                } else {
                    union[pred] += 1;
                }
            }
        }
        let ious: Vec<f64> = (0..k)
            .filter(|&c| union[c] > 0)
            .map(|c| inter[c] as f64 / union[c] as f64)
            .collect();
        println!(
            "check on {checked} held-out views (voted by the {voted} others): pixel accuracy {:.1}%, mIoU {:.1}% over {} classes",
            100.0 * right as f64 / total.max(1) as f64,
            100.0 * ious.iter().sum::<f64>() / ious.len().max(1) as f64,
            ious.len()
        );
        for (c, cls) in classes.iter().enumerate() {
            if union[c] > 0 {
                println!(
                    "  IoU {:>5.1}% {}",
                    100.0 * inter[c] as f64 / union[c] as f64,
                    cls.name
                );
            }
        }
    }

    if let Some(dir) = web {
        std::fs::create_dir_all(&dir)?;
        let order = importance_order(&scene, max);
        let pick = |s: &Scene| {
            Scene::new(
                order.iter().map(|&i| s.gaussians[i].clone()).collect(),
                s.sh_degree,
            )
        };
        save_splat(&pick(&scene), dir.join("scene.splat"))?;
        save_splat(&pick(&painted), dir.join("semantic.splat"))?;
        std::fs::write(dir.join("semantic.json"), classes_json(&classes, &counts))?;
        // The capture viewpoints in antimatter15's cameras.json: per view the
        // centre and the camera-to-world rotation, row by row.
        let mut sorted = views.clone();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        let rows: Vec<String> = sorted
            .iter()
            .enumerate()
            .map(|(id, v)| {
                let (r, c, cam) = (
                    &v.camera.rotation,
                    v.camera.camera_center(),
                    &v.camera.camera,
                );
                format!(
                    "{{\"id\":{id},\"img_name\":\"{}\",\"width\":{},\"height\":{},\
                     \"position\":[{},{},{}],\"rotation\":[[{},{},{}],[{},{},{}],[{},{},{}]],\
                     \"fx\":{},\"fy\":{}}}",
                    v.name.replace(['"', '\\'], "_"),
                    cam.width,
                    cam.height,
                    c.x,
                    c.y,
                    c.z,
                    r[(0, 0)],
                    r[(1, 0)],
                    r[(2, 0)],
                    r[(0, 1)],
                    r[(1, 1)],
                    r[(2, 1)],
                    r[(0, 2)],
                    r[(1, 2)],
                    r[(2, 2)],
                    cam.fx,
                    cam.fy
                )
            })
            .collect();
        std::fs::write(
            dir.join("cameras.json"),
            format!("[\n{}\n]\n", rows.join(",\n")),
        )?;
        println!(
            "{} of {} gaussians -> {}/{{scene,semantic}}.splat, semantic.json, cameras.json",
            order.len(),
            scene.len(),
            dir.display()
        );
    }
    Ok(())
}
