//! 2D multi-object tracking on synthetic detections: a `ReplayDetector`
//! feeds `ByteTracker`, and CLEAR MOT scores the tracks against the truth.
//!
//! Three people walk across a 1280×720 frame. Two of them cross, and the
//! farther one is partly hidden while they overlap, so its detector score
//! drops to 0.3; ByteTrack's low-score association keeps it tracked. The
//! tracks are written in the MOTChallenge text format.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use visloc_rs::objects::{
    BBox, ByteTracker, Detection, MotAccumulator, ObjectDetector, ReplayDetector, RgbImage,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir =
        parse_output_dir(&mut args).unwrap_or_else(|| PathBuf::from("target/track_objects_dummy"));
    if !args.is_empty() {
        eprintln!("usage: cargo run --example track_objects_dummy -- [--out-dir <dir>]");
        std::process::exit(2);
    }

    // (id, start centre, velocity per frame, width, height)
    let people = [
        (1, (200.0, 420.0), (4.0, 0.0), 70.0, 180.0),
        (2, (1000.0, 380.0), (-4.0, 0.0), 45.0, 115.0),
        (3, (640.0, 600.0), (1.0, -1.5), 60.0, 150.0),
    ];
    let frames = 150;
    let mut truth = Vec::new();
    let mut detections = Vec::new();
    for k in 0..frames {
        let boxes: Vec<(u64, BBox)> = people
            .iter()
            .map(|&(id, (x, y), (vx, vy), w, h)| {
                let t = k as f64;
                (id, BBox::from_center_size(x + vx * t, y + vy * t, w, h))
            })
            .collect();
        let hidden = boxes[0].1.iou(&boxes[1].1) > 0.0;
        detections.push(
            boxes
                .iter()
                .map(|&(id, bbox)| {
                    let score = if id == 2 && hidden { 0.3 } else { 0.9 };
                    Detection::new(bbox, score, 0)
                })
                .collect::<Vec<_>>(),
        );
        truth.push(boxes);
    }

    let mut detector = ReplayDetector::new(detections);
    let mut tracker = ByteTracker::default();
    let mut accumulator = MotAccumulator::new(0.5);
    // The replay ignores the image; a real detector would get the camera frame.
    let image = RgbImage::from_gray(1, 1, &[0])?;
    let mut mot = String::new();
    for frame_truth in &truth {
        let tracks = tracker.update(&detector.detect(&image)?);
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
        }
        let hypotheses: Vec<(u64, BBox)> = tracks.iter().map(|t| (t.id.0, t.bbox)).collect();
        accumulator.add_frame(frame_truth, &hypotheses);
    }

    fs::create_dir_all(&output_dir)?;
    let path = output_dir.join("tracks.txt");
    fs::write(&path, mot)?;
    let s = accumulator.summary();
    println!(
        "{} frames: MOTA {:.3}, mean IoU {:.3}, {} ID switches, {} misses, {} false positives",
        s.frames,
        s.mota(),
        s.motp_iou(),
        s.id_switches,
        s.misses,
        s.false_positives
    );
    println!("tracks: {}", path.display());
    if s.id_switches != 0 || s.misses != 0 {
        return Err("the synthetic people should be tracked without errors".into());
    }
    Ok(())
}

fn parse_output_dir(args: &mut Vec<String>) -> Option<PathBuf> {
    let index = args.iter().position(|arg| arg == "--out-dir")?;
    if index + 1 >= args.len() {
        eprintln!("--out-dir requires a directory path");
        std::process::exit(2);
    }
    let dir = PathBuf::from(args.remove(index + 1));
    args.remove(index);
    Some(dir)
}
