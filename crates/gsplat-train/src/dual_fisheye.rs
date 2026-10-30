//! Dual-fisheye 360° frames (an Insta360's two lenses side by side) cut into
//! pinhole faces with the lenses' factory calibration, for the pinhole
//! pipeline.
//!
//! A frame is `2S x S`: the front lens (the video's first stream) on the
//! left and the back lens on the right, each the `S x S` square that spans
//! the lens's 200° image circle (ffmpeg's `hstack` of an X5 video's two
//! streams; docs/insta360_lens.md). Directions are in the camera convention
//! the rest of the crate uses: x right, y down, z forward (along the lens's
//! axis). [`extract_dual_fisheye_frames`] cuts such frames from the video
//! with ffmpeg.

use std::f64::consts::FRAC_1_SQRT_2;
use std::path::Path;
use std::process::Command;

use nalgebra::{Matrix3, Vector3};
use visloc_io::insv::InsvLens;

/// A lens of a dual-fisheye frame: the unified (Mei) model of
/// [`InsvLens`] in the pixels of the frame's `S x S` half (pixel centres at
/// integers).
#[derive(Debug, Clone, PartialEq)]
pub struct FisheyeLens {
    xi: f64,
    fx: f64,
    fy: f64,
    radial: [f64; 3],
    tangential: [f64; 2],
    centre: f64,
}

impl FisheyeLens {
    /// `lens` in an `size x size` lens image: the calibration's pixels
    /// scaled by `size / (2 circle_radius)`, the optical centre in the
    /// middle.
    pub fn from_insv(lens: &InsvLens, size: u32) -> Self {
        let s = size as f64 / (2.0 * lens.circle_radius);
        Self {
            xi: lens.xi,
            fx: lens.fx * s,
            fy: lens.fy * s,
            radial: lens.radial,
            tangential: lens.tangential,
            centre: size as f64 / 2.0 - 0.5,
        }
    }

    /// Where `ray` (lens frame) lands in the lens image. The model holds
    /// within the image circle, 100° off the axis; `None` far behind it,
    /// where the unified model has no image.
    pub fn project(&self, ray: &Vector3<f64>) -> Option<(f64, f64)> {
        let zs = ray.z + self.xi * ray.norm();
        if zs <= 1e-9 * ray.norm() {
            return None;
        }
        let (x, y) = (ray.x / zs, ray.y / zs);
        let r2 = x * x + y * y;
        let [k1, k2, k3] = self.radial;
        let [p1, p2] = self.tangential;
        let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
        let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        Some((self.fx * xd + self.centre, self.fy * yd + self.centre))
    }

    /// Pixels per radian along the optical axis.
    pub fn axial_resolution(&self) -> f64 {
        self.fx.min(self.fy) / (1.0 + self.xi)
    }
}

/// The side of the `size x size` faces [`lens_face`] cuts at the lens's
/// resolution on its axis (a 90° face has `size / 2` pixels per radian at
/// its centre).
pub fn natural_face_size(lens: &FisheyeLens) -> u32 {
    (2.0 * lens.axial_resolution()).round() as u32
}

/// The five faces cut from each lens: a name and the rotation taking a
/// direction in the face camera's frame to the lens's frame (its columns
/// are the face camera's axes). Every face is a 90° pinhole (`f = size / 2`,
/// centred): one along the lens's axis and four tilted 45° from it, whose
/// far corners reach 90° off the axis. All of it lies inside the image
/// circle (the camera blacks out beyond about 98°), so no face has an edge
/// of the circle in it; beyond 90° the other lens sees.
pub fn lens_face_rotations() -> [(&'static str, Matrix3<f64>); 5] {
    let r = |x: [f64; 3], y: [f64; 3], z: [f64; 3]| {
        Matrix3::from_columns(&[Vector3::from(x), Vector3::from(y), Vector3::from(z)])
    };
    let h = FRAC_1_SQRT_2;
    [
        ("front", Matrix3::identity()),
        ("right", r([h, 0., -h], [0., 1., 0.], [h, 0., h])),
        ("left", r([h, 0., h], [0., 1., 0.], [-h, 0., h])),
        ("up", r([1., 0., 0.], [0., h, h], [0., -h, h])),
        ("down", r([1., 0., 0.], [0., h, -h], [0., h, h])),
    ]
}

/// The back lens's nominal rotation on the camera, taking a direction in
/// its frame to the front lens's: half a turn about the vertical (y) axis.
/// The X5's actual one is about 1° off (docs/insta360_lens.md); the
/// mapper refines it.
pub fn back_lens_rotation() -> Matrix3<f64> {
    Matrix3::from_diagonal(&Vector3::new(-1.0, 1.0, -1.0))
}

/// The ten faces of a frame as one rig whose frame is the front lens's: the
/// rotation taking a direction in each face's frame to the front lens's,
/// the front lens's [`lens_face_rotations`] then the back lens's (behind
/// [`back_lens_rotation`]).
pub fn rig_face_rotations() -> [Matrix3<f64>; 10] {
    let faces = lens_face_rotations();
    let back = back_lens_rotation();
    std::array::from_fn(|k| {
        let face = faces[k % 5].1;
        if k < 5 {
            face
        } else {
            back * face
        }
    })
}

/// Cuts every `step`-th frame of a dual-fisheye video into `dir` with ffmpeg
/// and ffprobe (on `PATH`): both lenses side by side, front lens on the
/// left, as `NNNNNN.jpg` named by the frame's number in the video (the
/// numbering `PhotoSfmConfig::imu` times from an `.insv`). The video has
/// the two lenses as two streams (an X5's `.insv`) or side by side in one.
/// The frames are scaled so that faces of `face_size` pixels see the
/// lenses' resolution on their axes ([`natural_face_size`]; never
/// enlarged). Returns how many frames `dir` then holds.
pub fn extract_dual_fisheye_frames(
    video: &Path,
    dir: &Path,
    step: u32,
    lenses: &[InsvLens],
    face_size: u32,
) -> Result<usize, String> {
    let run = |cmd: &mut Command| {
        let name = cmd.get_program().to_string_lossy().into_owned();
        cmd.output().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!("{name} is not on PATH: install ffmpeg, or cut the frames yourself")
            } else {
                format!("{name}: {e}")
            }
        })
    };
    let probe = run(Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v"])
        .args(["-show_entries", "stream=width,height", "-of", "csv=p=0"])
        .arg(video))?;
    if !probe.status.success() {
        return Err(format!(
            "ffprobe {}: {}",
            video.display(),
            String::from_utf8_lossy(&probe.stderr).trim()
        ));
    }
    let streams: Vec<(u32, u32)> = String::from_utf8_lossy(&probe.stdout)
        .lines()
        .filter_map(|l| {
            let (w, h) = l.trim().split_once(',')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .collect();
    let (lens_size, stack) = match streams.as_slice() {
        [(w0, h0), (w1, h1), ..] if w0 == h0 && (w0, h0) == (w1, h1) => (*h0, true),
        [(w, h)] if *w == 2 * *h => (*h, false),
        _ => {
            return Err(format!(
                "{}: video streams {streams:?}; expected two square lens streams or one with \
                 the lenses side by side",
                video.display()
            ))
        }
    };
    let natural = lenses
        .iter()
        .map(|l| natural_face_size(&FisheyeLens::from_insv(l, lens_size)))
        .max()
        .unwrap_or(lens_size)
        .max(1);
    let height = (face_size as u64 * lens_size as u64)
        .div_ceil(natural as u64)
        .min(lens_size as u64) as u32
        & !1;
    let select = format!("select='not(mod(n\\,{}))'", step.max(1));
    let scale = format!("scale={}:{height}", 2 * height);
    let filter = if stack {
        format!("[0:v:0][0:v:1]hstack=inputs=2,{select},{scale}")
    } else {
        format!("[0:v:0]{select},{scale}")
    };
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let out = run(Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(video)
        .args(["-filter_complex", &filter])
        .args(["-fps_mode", "passthrough", "-frame_pts", "1", "-q:v", "2"])
        .arg(dir.join("%06d.jpg")))?;
    if !out.status.success() {
        return Err(format!(
            "ffmpeg {}: {}",
            video.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let frames = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "jpg"))
        .count();
    Ok(frames)
}

/// The `size x size`, 90° pinhole view of the lens image `img` (`lens`'s
/// pixels) through a face rotated by `rotation` (face frame to lens frame,
/// as in [`lens_face_rotations`]): bilinear samples, black off the image.
/// Downscale the lens image first to about the face's resolution
/// ([`natural_face_size`]): the samples do no filtering of their own.
pub fn lens_face(
    img: &image::RgbImage,
    lens: &FisheyeLens,
    rotation: &Matrix3<f64>,
    size: u32,
) -> image::RgbImage {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let half = size as f64 / 2.0;
    let mut face = image::RgbImage::new(size, size);
    for (u, v, out) in face.enumerate_pixels_mut() {
        let ray = Vector3::new(
            (u as f64 + 0.5 - half) / half,
            (v as f64 + 0.5 - half) / half,
            1.0,
        );
        let Some((px, py)) = lens.project(&(rotation * ray)) else {
            continue;
        };
        let (x0, y0) = (px.floor(), py.floor());
        let (fx, fy) = (px - x0, py - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        if x0 < -1 || y0 < -1 || x0 >= w || y0 >= h {
            continue;
        }
        let at = |x: i64, y: i64| {
            if (0..w).contains(&x) && (0..h).contains(&y) {
                img.get_pixel(x as u32, y as u32).0
            } else {
                [0; 3]
            }
        };
        let (p00, p10, p01, p11) = (
            at(x0, y0),
            at(x0 + 1, y0),
            at(x0, y0 + 1),
            at(x0 + 1, y0 + 1),
        );
        for c in 0..3 {
            let top = p00[c] as f64 * (1.0 - fx) + p10[c] as f64 * fx;
            let bottom = p01[c] as f64 * (1.0 - fx) + p11[c] as f64 * fx;
            out.0[c] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
        }
    }
    face
}

#[cfg(test)]
mod tests {
    use super::*;

    /// About the X5's front lens (docs/insta360_lens.md), rounded.
    fn x5_like() -> InsvLens {
        InsvLens {
            xi: 2.0,
            fx: 4270.0,
            fy: 4270.0,
            cx: 2700.0,
            cy: 2680.0,
            radial: [0.18, 2.08, -3.2],
            tangential: [-0.0003, -0.0016],
            circle_radius: 2646.5,
            angles: [0.4, -0.2, 89.8],
            translation: [0.0; 3],
        }
    }

    #[test]
    fn the_image_circle_spans_the_lens_image() {
        // The calibration's 200° circle reaches the square's edges.
        let size = 3840;
        let lens = FisheyeLens::from_insv(&x5_like(), size);
        let c = size as f64 / 2.0 - 0.5;
        for phi in [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0] {
            let t = 100f64.to_radians();
            let ray = Vector3::new(t.sin() * phi.cos(), t.sin() * phi.sin(), t.cos());
            let (x, y) = lens.project(&ray).unwrap();
            let r = (x - c).hypot(y - c);
            assert!((r - size as f64 / 2.0).abs() < 0.01 * size as f64, "{r}");
        }
        assert_eq!(lens.project(&Vector3::z()), Some((c, c)));
        // About 2,065 px faces match the axis's resolution at 3840 px.
        assert!((natural_face_size(&lens) as i64 - 2065).abs() < 5);
    }

    #[test]
    fn extracts_side_by_side_frames_named_by_frame_number() {
        // Needs ffmpeg; skipped without it.
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("dual_fisheye_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let video = dir.join("two_lenses.mp4");
        let made = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc=size=64x64:rate=30:duration=1"])
            .args([
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30:duration=1",
            ])
            .args(["-map", "0:v", "-map", "1:v"])
            .arg(&video)
            .status()
            .unwrap();
        assert!(made.success());
        let frames = dir.join("frames");
        let lenses = [x5_like(), x5_like()];
        let n = extract_dual_fisheye_frames(&video, &frames, 10, &lenses, 4000).unwrap();
        let mut names: Vec<String> = std::fs::read_dir(&frames)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        let first = image::open(frames.join("000000.jpg")).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(n, 3);
        assert_eq!(names, ["000000.jpg", "000010.jpg", "000020.jpg"]);
        // Two 64 px lenses side by side, not enlarged for the large faces.
        assert_eq!((first.width(), first.height()), (128, 64));
    }

    #[test]
    fn faces_are_rotations_within_90_degrees_of_the_axis() {
        for (name, r) in lens_face_rotations() {
            assert!((r.determinant() - 1.0).abs() < 1e-12, "{name}");
            assert!((r.transpose() * r - Matrix3::identity()).norm() < 1e-12);
            for (u, v) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let corner = r * Vector3::new(u, v, 1.0).normalize();
                assert!(corner.z > -1e-12, "{name} corner beyond 90°");
            }
        }
        // Up tilts towards -y (the image's top), right towards +x.
        let faces = lens_face_rotations();
        assert!(faces[3].1.column(2).y < 0.0 && faces[1].1.column(2).x > 0.0);
        // The back lens's front face looks the other way, its up face still up.
        let rig = rig_face_rotations();
        assert_eq!(rig[0], Matrix3::identity());
        assert!((rig[5].column(2) + Vector3::z()).norm() < 1e-12);
        assert!(rig[8].column(2).y < 0.0);
        assert!(rig.iter().all(|r| (r.determinant() - 1.0).abs() < 1e-12));
    }

    #[test]
    fn a_face_samples_where_the_lens_sees() {
        // A lens image with one bright pixel block where a ray 30° to the
        // right of the axis lands; the right face sees it 15° left of its
        // centre, the front face 30° right of its centre.
        let size = 1024u32;
        let lens = FisheyeLens::from_insv(&x5_like(), size);
        let t = 30f64.to_radians();
        let (x, y) = lens.project(&Vector3::new(t.sin(), 0.0, t.cos())).unwrap();
        let mut img = image::RgbImage::new(size, size);
        for dy in -3..=3 {
            for dx in -3..=3 {
                img.put_pixel(
                    (x.round() as i64 + dx) as u32,
                    (y.round() as i64 + dy) as u32,
                    image::Rgb([255; 3]),
                );
            }
        }
        let faces = lens_face_rotations();
        let face = 512u32;
        let half = face as f64 / 2.0;
        let front = lens_face(&img, &lens, &faces[0].1, face);
        let right = lens_face(&img, &lens, &faces[1].1, face);
        let at = |f: &image::RgbImage, deg: f64| {
            let u = half + half * deg.to_radians().tan() - 0.5;
            f.get_pixel(u.round() as u32, face / 2).0[0]
        };
        assert!(at(&front, 30.0) > 200);
        assert!(at(&right, -15.0) > 200);
        assert_eq!(at(&front, -30.0), 0);
        // The far corners of a tilted face are still inside the image.
        assert!(lens
            .project(&(faces[1].1 * Vector3::new(1.0, 1.0, 1.0)))
            .is_some());
    }
}
