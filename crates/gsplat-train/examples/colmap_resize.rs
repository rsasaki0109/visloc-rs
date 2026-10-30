//! Scale a COLMAP text model of pinhole images down for training.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-train --example colmap_resize -- \
//!     --model <dir with cameras.txt, images.txt, points3D.txt> \
//!     --images <image dir> --out <dir> [--max-size 1600 | --size 1600x1067]
//! ```
//!
//! Every camera (`PINHOLE` or `SIMPLE_PINHOLE`) is scaled so that its
//! longer side is at most `--max-size` pixels, its size rounded, or with
//! `--size` to exactly that size (cameras of slightly different sizes, as in
//! some ETH3D scenes, then share one, which the trainers need). The focal
//! lengths and principal point are scaled per axis by the new size over the
//! original. A `SIMPLE_PINHOLE` whose two scales differ becomes a
//! `PINHOLE`. The images are resampled to their camera's size (Lanczos3) and
//! written as PNG to `<out>/images/<name>.png`; `images.txt` gets those names
//! and its 2D observations scaled, and `points3D.txt` is copied. The result,
//! `<out>/images` and `<out>/sparse/0`, is what `visloc_gsplat_train::dataset`
//! loads (e.g. an ETH3D scene's `dslr_calibration_undistorted` with its
//! `images` directory).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use image::imageops::FilterType;

struct Scale {
    sx: f64,
    sy: f64,
    width: u32,
    height: u32,
}

fn data_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
}

/// Rewrite cameras.txt; returns each camera's scale.
fn resize_cameras(
    text: &str,
    max_size: u32,
    size: Option<(u32, u32)>,
) -> Result<(String, HashMap<u32, Scale>), String> {
    let mut out = String::from("# Camera list with one line of data per camera:\n");
    out += "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    let mut scales = HashMap::new();
    for line in data_lines(text) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 4 {
            return Err(format!("bad camera line: {line}"));
        }
        let id: u32 = f[0].parse().map_err(|_| format!("bad camera id: {line}"))?;
        let (w, h): (u32, u32) = (
            f[2].parse().map_err(|_| format!("bad width: {line}"))?,
            f[3].parse().map_err(|_| format!("bad height: {line}"))?,
        );
        let p: Vec<f64> = f[4..]
            .iter()
            .map(|v| v.parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|_| format!("bad parameters: {line}"))?;
        let s = (max_size as f64 / w.max(h) as f64).min(1.0);
        let (nw, nh) = size.unwrap_or((
            ((w as f64 * s).round() as u32).max(1),
            ((h as f64 * s).round() as u32).max(1),
        ));
        let (sx, sy) = (nw as f64 / w as f64, nh as f64 / h as f64);
        let [fx, fy, cx, cy] = match (f[1], p.as_slice()) {
            ("PINHOLE", &[fx, fy, cx, cy]) => [fx, fy, cx, cy],
            ("SIMPLE_PINHOLE", &[f, cx, cy]) => [f, f, cx, cy],
            _ => return Err(format!("only PINHOLE / SIMPLE_PINHOLE cameras: {line}")),
        };
        out += &format!(
            "{id} PINHOLE {nw} {nh} {} {} {} {}\n",
            fx * sx,
            fy * sy,
            cx * sx,
            cy * sy
        );
        scales.insert(
            id,
            Scale {
                sx,
                sy,
                width: nw,
                height: nh,
            },
        );
    }
    Ok((out, scales))
}

/// One image of images.txt: its file name, the PNG it becomes, and its camera.
struct ImageEntry {
    name: String,
    new_name: String,
    camera: u32,
}

fn png_name(name: &str) -> String {
    Path::new(name)
        .with_extension("png")
        .to_string_lossy()
        .into_owned()
}

/// Rewrite images.txt; returns it and its images.
fn resize_images_txt(
    text: &str,
    scales: &HashMap<u32, Scale>,
) -> Result<(String, Vec<ImageEntry>), String> {
    let mut out = String::from(
        "# Image list with two lines of data per image:\n\
         #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\
         #   POINTS2D[] as (X, Y, POINT3D_ID)\n",
    );
    let mut images = Vec::new();
    // Data lines come in pairs; an image without observations has an empty
    // second line, which `data_lines` would drop, so walk the raw lines.
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    while let Some(header) = lines.next() {
        if header.trim().is_empty() {
            continue;
        }
        let points = lines.next().unwrap_or("");
        let f: Vec<&str> = header.split_whitespace().collect();
        if f.len() < 10 {
            return Err(format!("bad image line: {header}"));
        }
        let cam: u32 = f[8]
            .parse()
            .map_err(|_| format!("bad camera id: {header}"))?;
        let s = scales
            .get(&cam)
            .ok_or_else(|| format!("unknown camera {cam}"))?;
        let name = f[9..].join(" ");
        let new_name = png_name(&name);
        out += &format!("{} {new_name}\n", f[..9].join(" "));
        let obs: Vec<&str> = points.split_whitespace().collect();
        let mut scaled = Vec::with_capacity(obs.len());
        for o in obs.chunks(3) {
            if o.len() != 3 {
                return Err(format!("bad observations of {name}"));
            }
            let x: f64 = o[0].parse().map_err(|_| format!("bad x of {name}"))?;
            let y: f64 = o[1].parse().map_err(|_| format!("bad y of {name}"))?;
            scaled.push(format!("{} {} {}", x * s.sx, y * s.sy, o[2]));
        }
        out += &scaled.join(" ");
        out += "\n";
        images.push(ImageEntry {
            name,
            new_name,
            camera: cam,
        });
    }
    Ok((out, images))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut model: Option<PathBuf> = None;
    let mut image_dir: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut max_size = 1600u32;
    let mut size: Option<(u32, u32)> = None;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(next()?)),
            "--images" => image_dir = Some(PathBuf::from(next()?)),
            "--out" => out = Some(PathBuf::from(next()?)),
            "--max-size" => max_size = next()?.parse()?,
            "--size" => {
                let v = next()?;
                let (w, h) = v.split_once('x').ok_or("--size takes WxH")?;
                size = Some((w.parse()?, h.parse()?));
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let model = model.ok_or("--model is required")?;
    let image_dir = image_dir.ok_or("--images is required")?;
    let out = out.ok_or("--out is required")?;

    let (cameras, scales) = resize_cameras(
        &fs::read_to_string(model.join("cameras.txt"))?,
        max_size,
        size,
    )?;
    let (images_txt, images) =
        resize_images_txt(&fs::read_to_string(model.join("images.txt"))?, &scales)?;
    let sparse = out.join("sparse").join("0");
    fs::create_dir_all(&sparse)?;
    fs::write(sparse.join("cameras.txt"), cameras)?;
    fs::write(sparse.join("images.txt"), images_txt)?;
    fs::copy(model.join("points3D.txt"), sparse.join("points3D.txt"))?;

    for (
        k,
        ImageEntry {
            name,
            new_name,
            camera,
        },
    ) in images.iter().enumerate()
    {
        let s = &scales[camera];
        let img = image::open(image_dir.join(name))?.to_rgb8();
        let resized = image::imageops::resize(&img, s.width, s.height, FilterType::Lanczos3);
        let path = out.join("images").join(new_name);
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        resized.save(&path)?;
        println!(
            "[{}/{}] {name} {}x{} -> {}x{}",
            k + 1,
            images.len(),
            img.width(),
            img.height(),
            s.width,
            s.height
        );
    }
    println!("wrote {}", out.display());
    Ok(())
}
