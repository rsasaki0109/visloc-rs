//! YOLOX detection and ByteTrack tracking on an image sequence.
//!
//! ```text
//! curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
//! ORT_DYLIB_PATH=/path/to/libonnxruntime.so cargo run --release \
//!   --features image-io,onnx-inference --example yolox_track_demo -- \
//!   --model yolox_s.onnx --images <dir of .png/.jpg frames> --out-dir target/yolox_track [--draw]
//! ```
//!
//! The frames are taken in file-name order. `--input-size 416` is needed for
//! `yolox_nano` and `yolox_tiny`. The tracks go to `tracks.txt` in the
//! MOTChallenge text format (`frame,id,x,y,w,h,score,class,-1,-1`, frames from
//! 1). With `--draw`, each frame is also written with its track boxes and IDs.
//!
//! `--boxes-csv FILE` also writes the tracks of movable classes (people,
//! vehicles, animals) in the format `basalt_euroc_vio_demo --object-boxes`
//! reads: `timestamp_ns,object_id,class_id,score,x_min,y_min,x_max,y_max`.
//! The timestamp is the frame's file name, as in the EuRoC layout
//! (`mav0/cam0/data/<timestamp_ns>.png`).
//!
//! `--rotate 90|180|270` turns each frame clockwise before detection, for
//! cameras mounted sideways (the SLAM cameras of Project Aria glasses, as in
//! LaMAria, need `--rotate 90`). Every output is in the original frame's
//! pixels.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use visloc_rs::objects::{
    ByteTracker, ObjectDetector, RgbImage, TrackedObject, YoloxConfig, YoloxOnnxDetector,
    COCO_CLASSES, MOVABLE_COCO_CLASSES,
};

struct Args {
    model: PathBuf,
    images: PathBuf,
    out_dir: PathBuf,
    input_size: usize,
    /// Clockwise rotation before detection: 0, 90, 180 or 270 degrees.
    rotate: u32,
    max_frames: Option<usize>,
    draw: bool,
    boxes_csv: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let mut frames: Vec<PathBuf> = fs::read_dir(&args.images)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
        })
        .collect();
    frames.sort();
    if let Some(max) = args.max_frames {
        frames.truncate(max);
    }
    if frames.is_empty() {
        return Err(format!("no .png or .jpg images in {}", args.images.display()).into());
    }

    let config = YoloxConfig {
        input_width: args.input_size,
        input_height: args.input_size,
        ..YoloxConfig::default()
    };
    let mut detector = YoloxOnnxDetector::load_from_path(&args.model, config)?;
    let mut tracker = ByteTracker::default();
    fs::create_dir_all(&args.out_dir)?;
    let drawn_dir = args.out_dir.join("frames");
    if args.draw {
        fs::create_dir_all(&drawn_dir)?;
    }

    let mut mot = String::new();
    let mut boxes_csv =
        String::from("timestamp_ns,object_id,class_id,score,x_min,y_min,x_max,y_max\n");
    let mut detect_seconds = 0.0;
    let mut track_seconds = 0.0;
    let mut track_classes = BTreeMap::new();
    for path in &frames {
        let decoded = image::open(path)?.to_rgb8();
        let (width, height) = (decoded.width(), decoded.height());
        let upright = match args.rotate {
            90 => image::imageops::rotate90(&decoded),
            180 => image::imageops::rotate180(&decoded),
            270 => image::imageops::rotate270(&decoded),
            _ => decoded.clone(),
        };
        let (upright_width, upright_height) = (upright.width(), upright.height());
        let image = RgbImage::new(
            upright_width as usize,
            upright_height as usize,
            upright.into_raw(),
        )?;
        let started = Instant::now();
        let detections = detector.detect(&image)?;
        detect_seconds += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut tracks = tracker.update(&detections);
        track_seconds += started.elapsed().as_secs_f64();
        for t in &mut tracks {
            t.bbox = t
                .bbox
                .unrotated(args.rotate, f64::from(width), f64::from(height));
        }
        for t in &tracks {
            let b = t.bbox;
            writeln!(
                mot,
                "{},{},{:.2},{:.2},{:.2},{:.2},{:.3},{},-1,-1",
                tracker.frame(),
                t.id.0,
                b.x_min,
                b.y_min,
                b.width(),
                b.height(),
                t.score,
                t.class_id
            )?;
            track_classes.insert(t.id.0, t.class_id);
        }
        if args.boxes_csv.is_some() {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            let timestamp_ns: i64 = stem.parse().map_err(|_| {
                format!(
                    "--boxes-csv needs frames named <timestamp_ns>.png, got {}",
                    path.display()
                )
            })?;
            for t in tracks
                .iter()
                .filter(|t| MOVABLE_COCO_CLASSES.contains(&t.class_id))
            {
                let b = t.bbox;
                writeln!(
                    boxes_csv,
                    "{timestamp_ns},{},{},{:.3},{:.2},{:.2},{:.2},{:.2}",
                    t.id.0, t.class_id, t.score, b.x_min, b.y_min, b.x_max, b.y_max
                )?;
            }
        }
        if args.draw {
            let mut canvas = decoded;
            draw_tracks(&mut canvas, &tracks);
            let name = Path::new(path.file_name().ok_or("file name")?).with_extension("png");
            canvas.save(drawn_dir.join(name))?;
        }
    }
    fs::write(args.out_dir.join("tracks.txt"), mot)?;
    if let Some(path) = &args.boxes_csv {
        fs::write(path, boxes_csv)?;
        println!("object boxes: {}", path.display());
    }

    let n = frames.len() as f64;
    println!(
        "{} frames: detection {:.1} ms/frame, tracking {:.3} ms/frame",
        frames.len(),
        1e3 * detect_seconds / n,
        1e3 * track_seconds / n
    );
    let mut per_class: BTreeMap<&str, usize> = BTreeMap::new();
    for class in track_classes.values() {
        *per_class
            .entry(COCO_CLASSES.get(*class as usize).copied().unwrap_or("?"))
            .or_default() += 1;
    }
    println!("{} tracks: {per_class:?}", track_classes.len());
    println!("tracks: {}", args.out_dir.join("tracks.txt").display());
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let usage = "usage: yolox_track_demo --model <onnx> --images <dir> [--out-dir <dir>] \
                 [--input-size 640] [--rotate 0|90|180|270] [--max-frames N] [--draw] \
                 [--boxes-csv FILE]";
    let mut model = None;
    let mut images = None;
    let mut out_dir = PathBuf::from("target/yolox_track");
    let mut input_size = 640;
    let mut rotate = 0;
    let mut max_frames = None;
    let mut draw = false;
    let mut boxes_csv = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("{arg} needs a value\n{usage}"))
        };
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--images" => images = Some(PathBuf::from(value()?)),
            "--out-dir" => out_dir = PathBuf::from(value()?),
            "--input-size" => input_size = value()?.parse().map_err(|_| usage.to_string())?,
            "--rotate" => {
                rotate = value()?.parse().map_err(|_| usage.to_string())?;
                if ![0, 90, 180, 270].contains(&rotate) {
                    return Err(format!("--rotate takes 0, 90, 180 or 270\n{usage}"));
                }
            }
            "--max-frames" => max_frames = Some(value()?.parse().map_err(|_| usage.to_string())?),
            "--draw" => draw = true,
            "--boxes-csv" => boxes_csv = Some(PathBuf::from(value()?)),
            _ => return Err(format!("unknown argument {arg}\n{usage}")),
        }
    }
    Ok(Args {
        model: model.ok_or(usage)?,
        images: images.ok_or(usage)?,
        out_dir,
        input_size,
        rotate,
        max_frames,
        draw,
        boxes_csv,
    })
}

/// Box outlines in a colour per ID, with the ID written above each box.
fn draw_tracks(canvas: &mut image::RgbImage, tracks: &[TrackedObject]) {
    for t in tracks {
        let hue = (t.id.0 as f64 * 0.618_034).fract();
        let color = hsv_to_rgb(hue, 0.9, 1.0);
        let (w, h) = (canvas.width() as i64, canvas.height() as i64);
        let x0 = (t.bbox.x_min.round() as i64).clamp(0, w - 1);
        let y0 = (t.bbox.y_min.round() as i64).clamp(0, h - 1);
        let x1 = (t.bbox.x_max.round() as i64).clamp(0, w - 1);
        let y1 = (t.bbox.y_max.round() as i64).clamp(0, h - 1);
        for thickness in 0..3 {
            for x in x0..=x1 {
                put(canvas, x, y0 + thickness, color);
                put(canvas, x, y1 - thickness, color);
            }
            for y in y0..=y1 {
                put(canvas, x0 + thickness, y, color);
                put(canvas, x1 - thickness, y, color);
            }
        }
        draw_number(canvas, x0 + 2, (y0 - 18).max(0), t.id.0, color);
    }
}

fn put(canvas: &mut image::RgbImage, x: i64, y: i64, color: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < canvas.width() && (y as u32) < canvas.height() {
        canvas.put_pixel(x as u32, y as u32, image::Rgb(color));
    }
}

/// Digits from a 3×5 bitmap font, drawn at three times the size.
fn draw_number(canvas: &mut image::RgbImage, x: i64, y: i64, value: u64, color: [u8; 3]) {
    const DIGITS: [[u8; 5]; 10] = [
        [7, 5, 5, 5, 7],
        [2, 6, 2, 2, 7],
        [7, 1, 7, 4, 7],
        [7, 1, 7, 1, 7],
        [5, 5, 7, 1, 1],
        [7, 4, 7, 1, 7],
        [7, 4, 7, 5, 7],
        [7, 1, 1, 1, 1],
        [7, 5, 7, 5, 7],
        [7, 5, 7, 1, 7],
    ];
    for (i, digit) in value.to_string().bytes().enumerate() {
        let rows = DIGITS[(digit - b'0') as usize];
        let left = x + 12 * i as i64;
        for (row, bits) in rows.iter().enumerate() {
            for col in 0..3 {
                if bits & (4 >> col) == 0 {
                    continue;
                }
                for dy in 0..3 {
                    for dx in 0..3 {
                        put(canvas, left + 3 * col + dx, y + 3 * row as i64 + dy, color);
                    }
                }
            }
        }
    }
}

fn hsv_to_rgb(hue: f64, saturation: f64, value: f64) -> [u8; 3] {
    let h = hue * 6.0;
    let c = value * saturation;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = value - c;
    [r, g, b].map(|v| ((v + m) * 255.0).round() as u8)
}
