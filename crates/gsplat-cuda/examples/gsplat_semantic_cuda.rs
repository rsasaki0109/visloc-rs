//! Label a trained splat's gaussians by class from per-pixel labels, with the
//! CUDA renderer.
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --example gsplat_semantic_cuda -- \
//!   --ply scene.ply --data <colmap_root> --labels <dir> --out semantic.ply \
//!   [--web <dir> [--max 400000]] [--instances [--instance-overlap 0.5] [--min-views 3]] \
//!   [--check [--holdout 8] [--check-images <dir>]] [--logits]
//! ```
//!
//! `<dir>` holds `classes.txt` (`index r g b name` per line) and a label
//! image per view, `<image name>.png` with the extension replaced: 8-bit grey,
//! each pixel's class index, 255 for none (`scripts/sam3_labels.py` writes
//! them). A label image of another size is resized to the view's, nearest
//! pixel. Every view of the model (training and held out) votes: at each
//! labelled pixel, each splat's blending weight there goes to the pixel's
//! class, through the view's lens. A gaussian takes the class with the most
//! weight (`visloc_gsplat_core::semantic`). With `--logits` the classes come
//! instead from a splat trained with the labels (`gsplat_train_cuda
//! --labels`): its `class_logit_<c>` properties, the largest's class.
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
//! `--instances` also gives each gaussian a 3D object, from the per-view
//! instance images in `<dir>/instances/` (`sam3_labels.py --instances`):
//! each view's instances vote on their own, those of one class that share
//! their gaussians across views (`--instance-overlap`, 0.5) merge, and
//! objects seen in fewer than `--min-views` views (3) are dropped
//! (`semantic::merge_instances`). `--out` then has an `instance` property
//! (-1: none), and `--web` writes `instances.splat` (each object its own
//! colour), `instances.json` (the legend, and per object its class and
//! number as `gsplat_extract` takes them) and `instances.bin` (each splat's
//! object, u32 little-endian, 4294967295 for none), with which the viewer
//! picks the object under a click.
//!
//! `--check` holds out every `--holdout`-th view (8, as `gsplat_train_cuda`)
//! and lets only the others vote, then renders the painted splat at each
//! held-out view and scores it against that view's labels: pixel accuracy
//! and per class its IoU, recall and precision, each with the classes that
//! took the rest (a pixel's class: the class colour nearest its render).
//! `--check-images <dir>` writes each held-out view's classes as rendered
//! (`<image stem>.png`) and as labelled (`<image stem>_labels.png`), in the
//! class colours (black: none). With `--instances` it also matches the
//! objects against each held-out view's instances, by a soft pixel IoU in
//! blending weight.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use visloc_gsplat_core::gaussian::Scene;
use visloc_gsplat_core::ply::{ply_properties, save_ply_with, scene_from_ply};
use visloc_gsplat_core::semantic::{
    classes_json, colour_by_class, instance_color, instances_json, label_from_logits,
    label_gaussians, merge_instances, paint, parse_classes, view_instance_members, Instance,
    ViewInstance, NO_INSTANCE,
};
use visloc_gsplat_core::splat::{importance_order, save_splat};
use visloc_gsplat_cuda::CudaGsplat;
use visloc_gsplat_render::packing::PackedScene;
use visloc_gsplat_train::dataset::{load_colmap_dataset, View};

/// An instance image, row-major, and its instances' classes (instance i at i - 1).
type InstanceImage = (Vec<u8>, Vec<u8>);

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
    let mut instances = false;
    let mut overlap = 0.5f32;
    let mut min_views = 3usize;
    let mut from_logits = false;
    let mut check_images: Option<PathBuf> = None;
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
            "--check-images" => check_images = Some(PathBuf::from(next()?)),
            "--instances" => instances = true,
            "--instance-overlap" => overlap = next()?.parse()?,
            "--min-views" => min_views = next()?.parse()?,
            "--logits" => from_logits = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply <scene.ply> is required")?;
    let data = data.ok_or("--data <colmap root> is required")?;
    let labels = labels.ok_or("--labels <dir> is required")?;

    let classes = parse_classes(&std::fs::read_to_string(labels.join("classes.txt"))?)?;
    let k = classes.len();
    let ply_bytes = std::fs::read(&ply)?;
    let scene = scene_from_ply(&ply_bytes)?;
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
    // A view's instance image (`<labels>/instances/`) and its instances'
    // classes, if there is one.
    let view_instance_image =
        |v: &View| -> Result<Option<InstanceImage>, Box<dyn std::error::Error>> {
            let base = labels.join("instances").join(&v.name);
            let (png, txt) = (base.with_extension("png"), base.with_extension("txt"));
            if !png.exists() || !txt.exists() {
                return Ok(None);
            }
            let mut found = Vec::new();
            for line in std::fs::read_to_string(&txt)?.lines() {
                let f: Vec<&str> = line.split_whitespace().collect();
                if f.len() < 2 || f[0].starts_with('#') {
                    continue;
                }
                if f[0].parse::<usize>()? != found.len() + 1 {
                    return Err(format!("{}: ids must run 1, 2, ...", txt.display()).into());
                }
                found.push(f[1].parse::<u8>()?);
            }
            let (w, h) = (v.camera.camera.width, v.camera.camera.height);
            let mut img = image::open(&png)?.to_luma8();
            if img.dimensions() != (w, h) {
                img = image::imageops::resize(&img, w, h, FilterType::Nearest);
            }
            Ok(Some((img.into_raw(), found)))
        };

    let mut gs = CudaGsplat::from_env().ok_or(
        "set VISLOC_GSPLAT_CUDA_LIB to the library built by scripts/build_gsplat_cuda.sh",
    )??;
    gs.set_scene(&PackedScene::from_scene(&scene))?;
    // With --check only the training views vote.
    let voters = if check {
        dataset.train.len()
    } else {
        views.len()
    };
    let per = if from_logits {
        let names: Vec<String> = (0..k).map(|c| format!("class_logit_{c}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut logits = vec![0.0f32; scene.len() * k];
        for (c, column) in ply_properties(&ply_bytes, &refs)?.into_iter().enumerate() {
            let column = column.ok_or_else(|| {
                format!(
                    "{} has no {} (train it with gsplat_train_cuda --labels)",
                    ply.display(),
                    names[c]
                )
            })?;
            for (i, z) in column.into_iter().enumerate() {
                logits[i * k + c] = z;
            }
        }
        println!("classes from the splat's {k} trained logits per gaussian");
        label_from_logits(&logits, k)
    } else {
        let t0 = std::time::Instant::now();
        let mut voted = 0usize;
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
        println!(
            "{voted} of {} views voted in {:.1} s",
            views.len(),
            t0.elapsed().as_secs_f64()
        );
        label_gaussians(&votes, k)
    };

    // Per class its share of the gaussians and their mean confidence (the
    // class's share of a gaussian's votes, or its probability from logits).
    let mut counts = vec![0usize; k];
    let mut sure = vec![0.0f64; k];
    let mut unseen = 0usize;
    for l in &per {
        match counts.get_mut(l.class as usize) {
            Some(c) => {
                *c += 1;
                sure[l.class as usize] += l.confidence as f64;
            }
            None => unseen += 1,
        }
    }
    let n = scene.len().max(1) as f64;
    for (c, cls) in classes.iter().enumerate() {
        println!(
            "  {:>6.2}% {}, confidence {:.3}",
            100.0 * counts[c] as f64 / n,
            cls.name,
            sure[c] / counts[c].max(1) as f64
        );
    }
    println!(
        "  {:>6.2}% seen by no labelled pixel; mean confidence {:.3}",
        100.0 * unseen as f64 / n,
        sure.iter().sum::<f64>() / (scene.len() - unseen).max(1) as f64
    );

    // 3D instances: each view's instance image votes on its own, then the
    // view instances merge across views.
    let mut instance_of: Vec<u32> = Vec::new();
    let mut found: Vec<Instance> = Vec::new();
    let mut per_class_instances = vec![0usize; k];
    if instances {
        let t1 = std::time::Instant::now();
        let mut view_insts = Vec::new();
        for (vi, v) in views[..voters].iter().enumerate() {
            let Some((map, list)) = view_instance_image(v)? else {
                continue;
            };
            if list.is_empty() {
                continue;
            }
            gs.vote_labels(&v.camera, &map, list.len() + 1, true)?;
            let votes = gs.votes(list.len() + 1)?;
            for (i, members) in view_instance_members(&votes, list.len(), 0.5)
                .into_iter()
                .enumerate()
            {
                if !members.is_empty() {
                    view_insts.push(ViewInstance {
                        view: vi as u32,
                        class: list[i],
                        members,
                    });
                }
            }
        }
        let gaussian_class: Vec<u8> = per.iter().map(|l| l.class).collect();
        (instance_of, found) = merge_instances(&view_insts, &gaussian_class, overlap, min_views);
        for f in &found {
            if let Some(c) = per_class_instances.get_mut(f.class as usize) {
                *c += 1;
            }
        }
        println!(
            "{} view instances -> {} instances (overlap {overlap}, {min_views} views or more) in {:.1} s; {:.1}% of the gaussians in one",
            view_insts.len(),
            found.len(),
            t1.elapsed().as_secs_f64(),
            100.0 * instance_of.iter().filter(|&&i| i != NO_INSTANCE).count() as f64 / n
        );
        for (c, &count) in classes.iter().zip(&per_class_instances) {
            if count > 0 {
                println!("  {count:>5} {}", c.name);
            }
        }
    }

    let painted = colour_by_class(&scene, &per, &classes);
    let class: Vec<f32> = per.iter().map(|l| l.class as f32).collect();
    let conf: Vec<f32> = per.iter().map(|l| l.confidence).collect();
    let instance: Vec<f32> = instance_of
        .iter()
        .map(|&i| if i == NO_INSTANCE { -1.0 } else { i as f32 })
        .collect();
    let mut extra: Vec<(&str, &[f32])> = vec![("class", &class), ("class_confidence", &conf)];
    if instances {
        extra.push(("instance", &instance));
    }
    save_ply_with(&painted, &extra, &out)?;
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
        // confusion[t * (k + 1) + p]: labelled t, rendered as p (k: none).
        let mut confusion = vec![0u64; k * (k + 1)];
        let (mut right, mut total, mut checked) = (0u64, 0u64, 0usize);
        if let Some(dir) = &check_images {
            std::fs::create_dir_all(dir)?;
        }
        for v in &dataset.eval {
            let Some(truth) = view_labels(v)? else {
                continue;
            };
            checked += 1;
            let frame = gs.render(&v.camera, [0.0; 3], 0)?;
            let pred: Vec<usize> = frame
                .rgb
                .chunks_exact(3)
                .map(|rgb| {
                    let dist = |c: &[f32; 3]| (0..3).map(|i| (c[i] - rgb[i]).powi(2)).sum::<f32>();
                    (0..=k)
                        .min_by(|&a, &b| dist(&colors[a]).total_cmp(&dist(&colors[b])))
                        .unwrap_or(k)
                })
                .collect();
            for (&t, &pred) in truth.iter().zip(&pred) {
                let t = t as usize;
                if t >= k {
                    continue;
                }
                total += 1;
                union[t] += 1;
                confusion[t * (k + 1) + pred] += 1;
                if pred == t {
                    right += 1;
                    inter[t] += 1;
                } else {
                    union[pred] += 1;
                }
            }
            if let Some(dir) = &check_images {
                // The view's classes as rendered and as labelled, in the
                // class colours (black: none).
                let (w, h) = (v.camera.camera.width, v.camera.camera.height);
                let paint = |c: usize| colors[c.min(k)].map(|x| (x * 255.0).round() as u8);
                let stem = Path::new(&v.name).with_extension("");
                let stem = stem.file_name().unwrap_or_default().to_string_lossy();
                let rendered: Vec<u8> = pred.iter().flat_map(|&c| paint(c)).collect();
                let labelled: Vec<u8> = truth.iter().flat_map(|&t| paint(t as usize)).collect();
                image::RgbImage::from_raw(w, h, rendered)
                    .ok_or("render size")?
                    .save(dir.join(format!("{stem}.png")))?;
                image::RgbImage::from_raw(w, h, labelled)
                    .ok_or("label size")?
                    .save(dir.join(format!("{stem}_labels.png")))?;
            }
        }
        let ious: Vec<f64> = (0..k)
            .filter(|&c| union[c] > 0)
            .map(|c| inter[c] as f64 / union[c] as f64)
            .collect();
        println!(
            "check on {checked} held-out views ({} by the {voters} others): pixel accuracy {:.1}%, mIoU {:.1}% over {} classes",
            if from_logits { "trained" } else { "voted" },
            100.0 * right as f64 / total.max(1) as f64,
            100.0 * ious.iter().sum::<f64>() / ious.len().max(1) as f64,
            ious.len()
        );
        // Per class its recall and precision and, after each, what took the
        // rest (the three largest shares of 0.5% or more).
        let name = |c: usize| classes.get(c).map_or("none", |cls| cls.name.as_str());
        let shares = |pairs: Vec<(usize, u64)>, of: u64| {
            let mut pairs: Vec<(usize, u64)> = pairs
                .into_iter()
                .filter(|&(_, n)| n * 200 >= of.max(1))
                .collect();
            pairs.sort_by_key(|&(c, n)| (std::cmp::Reverse(n), c));
            let top: Vec<String> = pairs
                .iter()
                .take(3)
                .map(|&(c, n)| format!("{} {:.1}%", name(c), 100.0 * n as f64 / of as f64))
                .collect();
            if top.is_empty() {
                String::new()
            } else {
                format!(" (else {})", top.join(", "))
            }
        };
        let at = |t: usize, p: usize| confusion[t * (k + 1) + p];
        for c in (0..k).filter(|&c| union[c] > 0) {
            let labelled: u64 = (0..=k).map(|p| at(c, p)).sum();
            let rendered: u64 = (0..k).map(|t| at(t, c)).sum();
            let missed = (0..=k).filter(|&p| p != c).map(|p| (p, at(c, p))).collect();
            let taken = (0..k).filter(|&t| t != c).map(|t| (t, at(t, c))).collect();
            println!(
                "  IoU {:>5.1}% {}: recall {:.1}%{}, precision {:.1}%{}",
                100.0 * inter[c] as f64 / union[c] as f64,
                name(c),
                100.0 * at(c, c) as f64 / labelled.max(1) as f64,
                shares(missed, labelled),
                100.0 * at(c, c) as f64 / rendered.max(1) as f64,
                shares(taken, rendered),
            );
        }
    }

    if check && instances {
        // Each held-out view's instance image votes; in that view, an
        // instance image's instance i and a 3D instance j overlap by the
        // blending weight i's pixels give j's gaussians, against the weight
        // of i's pixels and of all pixels on j's gaussians (a soft pixel IoU).
        let mut best_iou: Vec<(u8, f64)> = Vec::new();
        let (mut shown, mut matched) = (0usize, 0usize);
        for v in &dataset.eval {
            let Some((map, list)) = view_instance_image(v)? else {
                continue;
            };
            let m = list.len();
            if m == 0 {
                continue;
            }
            gs.vote_labels(&v.camera, &map, m + 1, true)?;
            let votes = gs.votes(m + 1)?;
            let mut w2 = vec![0f64; m + 1];
            let mut w3: HashMap<u32, f64> = HashMap::new();
            let mut both: HashMap<(usize, u32), f64> = HashMap::new();
            for (g, vv) in votes.chunks_exact(m + 1).enumerate() {
                for (i, &w) in vv.iter().enumerate() {
                    w2[i] += w as f64;
                }
                let j = instance_of[g];
                let total: f64 = vv.iter().map(|&w| w as f64).sum();
                if j == NO_INSTANCE || total == 0.0 {
                    continue;
                }
                *w3.entry(j).or_default() += total;
                for (i, &w) in vv.iter().enumerate().skip(1) {
                    if w > 0.0 {
                        *both.entry((i, j)).or_default() += w as f64;
                    }
                }
            }
            let iou = |i: usize, j: u32| {
                let x = both.get(&(i, j)).copied().unwrap_or(0.0);
                x / (w2[i] + w3[&j] - x)
            };
            for i in 1..=m {
                if w2[i] > 0.0 {
                    let best = w3.keys().map(|&j| iou(i, j)).fold(0.0, f64::max);
                    best_iou.push((list[i - 1], best));
                }
            }
            // A 3D instance shows in the view when its gaussians take the
            // weight of 50 pixels or more.
            for (&j, &wj) in &w3 {
                if wj >= 50.0 {
                    shown += 1;
                    if (1..=m).any(|i| w2[i] > 0.0 && iou(i, j) >= 0.5) {
                        matched += 1;
                    }
                }
            }
        }
        let stats = |ious: &[f64]| {
            (
                ious.len(),
                100.0 * ious.iter().sum::<f64>() / ious.len().max(1) as f64,
                100.0 * ious.iter().filter(|&&x| x >= 0.5).count() as f64
                    / ious.len().max(1) as f64,
            )
        };
        let all: Vec<f64> = best_iou.iter().map(|b| b.1).collect();
        let (count, mean, recall) = stats(&all);
        println!(
            "instance check: {count} held-out view instances, best IoU with a 3D instance {mean:.1}% on average, {recall:.1}% at 0.5 or more; {matched} of the {shown} 3D instances shown match one at 0.5 ({:.1}%)",
            100.0 * matched as f64 / shown.max(1) as f64
        );
        for (c, cls) in classes.iter().enumerate() {
            let ious: Vec<f64> = best_iou
                .iter()
                .filter(|b| b.0 as usize == c)
                .map(|b| b.1)
                .collect();
            if !ious.is_empty() {
                let (count, mean, recall) = stats(&ious);
                println!(
                    "  {count:>4} {}: mean {mean:.1}%, {recall:.1}% at 0.5",
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
        if instances {
            let colors: Vec<[u8; 3]> = instance_of.iter().map(|&i| instance_color(i)).collect();
            save_splat(&pick(&paint(&scene, &colors)), dir.join("instances.splat"))?;
            std::fs::write(dir.join("instances.json"), instances_json(&classes, &found))?;
            // Each splat's object, in the splats' order, for picking.
            let ids: Vec<u8> = order
                .iter()
                .flat_map(|&i| instance_of[i].to_le_bytes())
                .collect();
            std::fs::write(dir.join("instances.bin"), ids)?;
        }
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
