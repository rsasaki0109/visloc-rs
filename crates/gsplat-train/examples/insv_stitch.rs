//! An Insta360 video's two lenses stitched into equirectangular (360°)
//! frames, with the lens calibration in the video's metadata; no Insta360
//! Studio involved.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-train --features euroc --example insv_stitch -- \
//!     VIDEO.insv (--out <folder> | --video OUT.mp4) [--step 1] [--width PX] [--blend 2] \
//!     [--back-lens auto|factory|back_lens.txt] [--distance M] [--frame SIDE_BY_SIDE.jpg]
//! ```
//!
//! Frames are decoded with ffmpeg (on `PATH`), both lenses side by side,
//! and every `--step`-th one is stitched
//! (`visloc_gsplat_train::dual_fisheye::stitch_equirect`) into
//! `<out>/NNNNNN.jpg` (named by the frame's number in the video, as
//! `gsplat_photos --insv` names its frames) or, with `--video`, encoded by
//! ffmpeg into one equirectangular video. `--frame` (repeatable) stitches
//! images of the two lenses side by side instead (front lens on the left,
//! e.g. frames `gsplat_photos --insv` cut), still taking the calibration
//! from the video.
//!
//! The back lens's rotation on the camera is fitted to where the two lenses
//! see the same things along the seam, in 16 frames spread over the video
//! (`--back-lens auto`); `factory` takes it from the recorded angles alone,
//! and a file from the `back_lens.txt` a `gsplat_photos --insv` run wrote
//! (docs/insta360_lens.md). The lenses are 3.2 cm apart, so where they meet
//! things at different distances cannot all line up: the seam is stitched
//! for the distance the fit finds typical along it (`--distance` sets it).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use nalgebra::{Matrix3, Vector3};
use visloc_gsplat_train::dual_fisheye::{
    back_lens_rotation, factory_back_lens_rotation, fit_back_lens, seam_pairs, stitch_equirect,
    FisheyeLens, SeamFit, SeamPair,
};
use visloc_io::insv::read_insv_telemetry;

const USAGE: &str = "\
usage: insv_stitch VIDEO.insv (--out <folder> | --video OUT.mp4) [options]

An Insta360 video's dual-fisheye frames -> equirectangular (360°) frames.

  --out <folder>     write every stitched frame as <folder>/NNNNNN.jpg (frame number)
  --video OUT.mp4    encode the stitched frames into one video with ffmpeg instead
  --step 1           stitch every N-th frame
  --width PX         panorama width (height = width / 2; default: the lenses' resolution)
  --blend 2          degrees over which the two lenses are mixed at the seam
  --back-lens auto   the back lens's rotation: auto (fitted along the seam in 16 frames),
                     factory (the recorded angles alone), or a gsplat_photos --insv run's
                     back_lens.txt
  --distance M       stitch the seam for things this far away (metres; inf: far away;
                     default: the scene's typical distance along the seam with
                     --back-lens auto, far away otherwise)
  --frame IMG        stitch this side-by-side image (front lens on the left) instead of
                     the video's frames, written as <out>/<IMG's stem>.jpg; repeatable
                     (--back-lens auto then fits on all of them)
  --quality 92       JPEG quality of --out frames
";

/// The back lens's rotation, its optical centre's offset (pixels) and the
/// distance the seam is stitched at (none: far away).
type BackLens = (Matrix3<f64>, [f64; 2], Option<f64>);

/// Frames the automatic back-lens fit samples from a video.
const FIT_FRAMES: u64 = 16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut video: Option<PathBuf> = None;
    let (mut out, mut encode): (Option<PathBuf>, Option<PathBuf>) = (None, None);
    let mut frames: Vec<PathBuf> = Vec::new();
    let (mut step, mut width, mut blend, mut quality) = (1u32, None, 2.0f64, 92u8);
    let mut back_lens = String::from("auto");
    let mut distance: Option<f64> = None;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(next()?)),
            "--video" => encode = Some(PathBuf::from(next()?)),
            "--frame" => frames.push(PathBuf::from(next()?)),
            "--step" => step = next()?.parse::<u32>()?.max(1),
            "--width" => width = Some(next()?.parse::<u32>()?),
            "--blend" => blend = next()?.parse()?,
            "--quality" => quality = next()?.parse()?,
            "--back-lens" => back_lens = next()?,
            "--distance" => distance = Some(next()?.parse()?),
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            s if s.starts_with('-') => return Err(format!("unknown option {s}\n{USAGE}").into()),
            s => video = Some(PathBuf::from(s)),
        }
    }
    let video = video.ok_or(format!("no video given\n{USAGE}"))?;
    if out.is_none() && encode.is_none() {
        return Err(format!("give --out or --video\n{USAGE}").into());
    }
    let telemetry = read_insv_telemetry(&video)?;
    let [front, back] = telemetry.lenses.as_slice() else {
        return Err(format!(
            "{}: no lens calibration in the metadata (decoded for the X5's layout; \
             docs/insta360_lens.md)",
            video.display()
        )
        .into());
    };
    let factory = factory_back_lens_rotation(front, back);
    // The back lens's centre in the front lens's frame, as recorded.
    let baseline = Vector3::from(back.translation);
    let blend = blend.max(0.0).to_radians();
    // The back lens's rotation and optical centre, and the distance its
    // seam is stitched at.
    let fit = |pairs: &[SeamPair], lens: &FisheyeLens| -> BackLens {
        let SeamFit {
            rotation,
            centre_offset,
            parallax,
            across,
            pairs: kept,
        } = fit_back_lens(pairs, lens, &factory, &baseline);
        // The baseline over the shared shift; none for a shift too small to
        // tell from far away (beyond 300 m).
        let distance = (parallax > baseline.norm() / 300.0).then(|| baseline.norm() / parallax);
        // Past the half turn as a rotation vector in the front lens's frame.
        let past =
            nalgebra::Rotation3::from_matrix_unchecked(back_lens_rotation() * rotation.transpose())
                .scaled_axis()
                .map(f64::to_degrees);
        eprintln!(
            "back lens: {:.3}° past the half turn ({:+.3}, {:+.3}, {:+.3}), {:.3}° from the \
             recorded angles, optical centre ({:+.1}, {:+.1}) px; {kept} of {} matches along \
             the seam, median {:.3}° across their parallax; the scene along the seam about {} \
             away",
            angle_between(&rotation, &back_lens_rotation()),
            past.x,
            past.y,
            past.z,
            angle_between(&rotation, &factory),
            centre_offset[0],
            centre_offset[1],
            pairs.len(),
            across.to_degrees(),
            distance.map_or("far".to_string(), |d| format!("{d:.1} m"))
        );
        (rotation, centre_offset, distance)
    };
    // The back lens's centre over the stitching distance (`stitch_equirect`).
    let parallax = |fitted: Option<f64>| -> Vector3<f64> {
        match distance.or(fitted) {
            Some(d) if d.is_finite() && d > 0.0 => baseline / d,
            _ => Vector3::zeros(),
        }
    };

    if !frames.is_empty() {
        let dir = out.ok_or("--frame writes into --out")?;
        std::fs::create_dir_all(&dir)?;
        let images: Vec<image::RgbImage> = frames
            .iter()
            .map(|p| image::open(p).map(|i| i.to_rgb8()))
            .collect::<Result<_, _>>()?;
        let size = images[0].height();
        let mut lenses = [
            FisheyeLens::from_insv(front, size),
            FisheyeLens::from_insv(back, size),
        ];
        let (rotation, centre, fitted) = match back_lens.as_str() {
            "auto" => fit(
                &images
                    .iter()
                    .flat_map(|img| seam_pairs(img, &lenses, &factory))
                    .collect::<Vec<_>>(),
                &lenses[1],
            ),
            "factory" => (factory, [0.0; 2], None),
            file => (read_back_lens(Path::new(file))?, [0.0; 2], None),
        };
        lenses[1] = lenses[1].shifted(centre);
        let parallax = parallax(fitted);
        let w = width.unwrap_or_else(|| natural_width(&lenses));
        for (path, img) in frames.iter().zip(&images) {
            if img.height() != size {
                return Err(
                    format!("{}: not {}x{size} like the first", path.display(), 2 * size).into(),
                );
            }
            let t = Instant::now();
            let pano = stitch_equirect(img, &lenses, &rotation, &parallax, w, blend);
            let name = dir
                .join(path.file_stem().unwrap_or_default())
                .with_extension("jpg");
            save_jpeg(&pano, &name, quality)?;
            println!(
                "{} ({}x{}, {:.1} s)",
                name.display(),
                pano.width(),
                pano.height(),
                t.elapsed().as_secs_f64()
            );
        }
        return Ok(());
    }

    let (size, stack) = lens_streams(&video)?;
    let mut lenses = [
        FisheyeLens::from_insv(front, size),
        FisheyeLens::from_insv(back, size),
    ];
    let (rotation, centre, fitted) = match back_lens.as_str() {
        "auto" => {
            let every = frame_count(&video).map_or(30, |n| (n / FIT_FRAMES).max(1));
            let t = Instant::now();
            let mut pairs = Vec::new();
            let mut taken = 0;
            decode_frames(&video, size, stack, every, |_, img| {
                pairs.extend(seam_pairs(&img, &lenses, &factory));
                taken += 1;
                Ok(taken < FIT_FRAMES)
            })?;
            eprintln!(
                "fitting the back lens on {taken} frames ({:.1} s)",
                t.elapsed().as_secs_f64()
            );
            fit(&pairs, &lenses[1])
        }
        "factory" => (factory, [0.0; 2], None),
        file => (read_back_lens(Path::new(file))?, [0.0; 2], None),
    };
    lenses[1] = lenses[1].shifted(centre);
    let parallax = parallax(fitted);
    let w = width.unwrap_or_else(|| natural_width(&lenses)).max(2) & !1;
    let h = w / 2;
    let mut encoder = match &encode {
        Some(path) => {
            let rate = frame_rate(&video)? / step as f64;
            Some(
                Command::new("ffmpeg")
                    .args(["-hide_banner", "-loglevel", "error", "-y"])
                    .args([
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        &format!("{w}x{h}"),
                    ])
                    .args(["-r", &format!("{rate}"), "-i", "-"])
                    .args(["-c:v", "libx264", "-preset", "medium", "-crf", "18"])
                    .args(["-pix_fmt", "yuv420p"])
                    .arg(path)
                    .stdin(Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("ffmpeg: {e}"))?,
            )
        }
        None => None,
    };
    if let Some(dir) = &out {
        std::fs::create_dir_all(dir)?;
    }
    let t = Instant::now();
    let mut n = 0u64;
    decode_frames(&video, size, stack, step as u64, |index, img| {
        let pano = stitch_equirect(&img, &lenses, &rotation, &parallax, w, blend);
        if let Some(dir) = &out {
            save_jpeg(&pano, &dir.join(format!("{index:06}.jpg")), quality)?;
        }
        if let Some(enc) = encoder.as_mut() {
            enc.stdin
                .as_mut()
                .ok_or("ffmpeg: no input")?
                .write_all(pano.as_raw())?;
        }
        n += 1;
        if n.is_multiple_of(10) {
            eprintln!(
                "{n} frames ({:.2} s each)",
                t.elapsed().as_secs_f64() / n as f64
            );
        }
        Ok(true)
    })?;
    if let Some(mut enc) = encoder {
        drop(enc.stdin.take());
        if !enc.wait()?.success() {
            return Err("ffmpeg could not encode the video".into());
        }
    }
    println!(
        "{n} frames stitched to {w}x{h} in {:.1} s",
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

type Error = Box<dyn std::error::Error>;

/// Decodes every `every`-th frame of `video`, both lenses side by side
/// (`2 size x size`), and hands each with its number in the video to `f`
/// until `f` returns false.
fn decode_frames(
    video: &Path,
    size: u32,
    stack: bool,
    every: u64,
    mut f: impl FnMut(u64, image::RgbImage) -> Result<bool, Error>,
) -> Result<(), Error> {
    let select = format!("select='not(mod(n\\,{every}))'");
    let filter = if stack {
        format!("[0:v:0][0:v:1]hstack=inputs=2,{select}")
    } else {
        format!("[0:v:0]{select}")
    };
    let mut decoder = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(video)
        .args(["-filter_complex", &filter, "-fps_mode", "passthrough"])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e} (is it on PATH?)"))?;
    let mut stdout = decoder.stdout.take().ok_or("ffmpeg: no output")?;
    let mut buf = vec![0u8; 2 * size as usize * size as usize * 3];
    let mut k = 0u64;
    let finished = loop {
        match stdout.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break true,
            Err(e) => return Err(e.into()),
        }
        let img = image::RgbImage::from_raw(2 * size, size, std::mem::take(&mut buf))
            .ok_or("frame size")?;
        let more = f(k * every, img.clone())?;
        buf = img.into_raw();
        k += 1;
        if !more {
            break false;
        }
    };
    drop(stdout);
    if finished {
        if !decoder.wait()?.success() {
            return Err("ffmpeg could not decode the video".into());
        }
    } else {
        let _ = decoder.kill();
        let _ = decoder.wait();
    }
    Ok(())
}

/// The angle of the rotation taking `b` to `a`, in degrees.
fn angle_between(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
    (((a * b.transpose()).trace() - 1.0) / 2.0)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// The `rotation` row of a `back_lens.txt` (`gsplat_photos --insv`).
fn read_back_lens(path: &Path) -> Result<Matrix3<f64>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = text
        .lines()
        .find_map(|l| l.strip_prefix("rotation "))
        .ok_or(format!("{}: no rotation line", path.display()))?
        .split_whitespace()
        .map(|v| v.parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if values.len() != 9 {
        return Err(format!("{}: the rotation needs 9 numbers", path.display()));
    }
    Ok(Matrix3::from_row_slice(&values))
}

/// The panorama width that matches the lenses' resolution on their axes.
fn natural_width(lenses: &[FisheyeLens; 2]) -> u32 {
    let r = lenses[0]
        .axial_resolution()
        .max(lenses[1].axial_resolution());
    ((2.0 * std::f64::consts::PI * r / 16.0).round() as u32 * 16).max(16)
}

/// The lens image size and whether the lenses are two streams (an X5's
/// `.insv`) rather than side by side in one.
fn lens_streams(video: &Path) -> Result<(u32, bool), String> {
    let text = probe(
        video,
        &[
            "-select_streams",
            "v",
            "-show_entries",
            "stream=width,height",
        ],
    )?;
    let streams: Vec<(u32, u32)> = text
        .lines()
        .filter_map(|l| {
            let (w, h) = l.trim().split_once(',')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .collect();
    match streams.as_slice() {
        [(w0, h0), (w1, h1), ..] if w0 == h0 && (w0, h0) == (w1, h1) => Ok((*h0, true)),
        [(w, h)] if *w == 2 * *h => Ok((*h, false)),
        _ => Err(format!(
            "{}: video streams {streams:?}; expected two square lens streams or one with the \
             lenses side by side",
            video.display()
        )),
    }
}

/// The video's frame rate (frames per second).
fn frame_rate(video: &Path) -> Result<f64, String> {
    let text = probe(
        video,
        &[
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
        ],
    )?;
    let rate = text.lines().next().unwrap_or("").trim();
    let value = match rate.split_once('/') {
        Some((a, b)) => a
            .parse::<f64>()
            .ok()
            .zip(b.parse::<f64>().ok())
            .map(|(a, b)| a / b),
        None => rate.parse().ok(),
    };
    value
        .filter(|r| r.is_finite() && *r > 0.0)
        .ok_or(format!("{}: no frame rate ({rate:?})", video.display()))
}

/// The number of frames in the video's first stream, if its header says.
fn frame_count(video: &Path) -> Option<u64> {
    probe(
        video,
        &[
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
        ],
    )
    .ok()?
    .lines()
    .next()?
    .trim()
    .parse()
    .ok()
}

/// ffprobe's CSV output for `args`.
fn probe(video: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("ffprobe")
        .args(["-v", "error"])
        .args(args)
        .args(["-of", "csv=p=0"])
        .arg(video)
        .output()
        .map_err(|e| format!("ffprobe: {e} (is it on PATH?)"))?;
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn save_jpeg(img: &image::RgbImage, path: &Path, quality: u8) -> Result<(), Error> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    image::codecs::jpeg::JpegEncoder::new_with_quality(file, quality).encode_image(img)?;
    Ok(())
}
