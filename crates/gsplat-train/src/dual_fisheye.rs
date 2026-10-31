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
    centre: [f64; 2],
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
            centre: [size as f64 / 2.0 - 0.5; 2],
        }
    }

    /// The same lens with its optical centre `offset` pixels away (x right,
    /// y down): [`fit_back_lens`] measures how far the back lens's sits
    /// from where the front lens's puts it.
    pub fn shifted(&self, offset: [f64; 2]) -> Self {
        Self {
            centre: [self.centre[0] + offset[0], self.centre[1] + offset[1]],
            ..self.clone()
        }
    }

    /// The same lens with its focal lengths `scale` times as long
    /// ([`calibrate_lenses`]).
    pub fn with_focal_scale(&self, scale: f64) -> Self {
        Self {
            fx: self.fx * scale,
            fy: self.fy * scale,
            ..self.clone()
        }
    }

    /// Where `ray` (lens frame) lands in the lens image. The model holds
    /// within the image circle, 100° off the axis; `None` far behind it,
    /// where the unified model has no image.
    pub fn project(&self, ray: &Vector3<f64>) -> Option<(f64, f64)> {
        self.project_adjusted(ray, 1.0, [0.0; 2])
    }

    /// [`FisheyeLens::project`] with the focal lengths `scale` times as long
    /// and the optical centre `offset` pixels away.
    fn project_adjusted(
        &self,
        ray: &Vector3<f64>,
        scale: f64,
        offset: [f64; 2],
    ) -> Option<(f64, f64)> {
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
        Some((
            scale * self.fx * xd + self.centre[0] + offset[0],
            scale * self.fy * yd + self.centre[1] + offset[1],
        ))
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

/// The back lens's rotation on the camera (as [`back_lens_rotation`]) from
/// the lenses' recorded angles: the half turn, then the turn about the
/// optical axis that the third angles hold beyond 90° each
/// (docs/insta360_lens.md).
pub fn factory_back_lens_rotation(front: &InsvLens, back: &InsvLens) -> Matrix3<f64> {
    let roll = (front.angles[2] - 90.0 + back.angles[2] - 90.0).to_radians();
    nalgebra::Rotation3::from_axis_angle(&Vector3::z_axis(), -roll).into_inner()
        * back_lens_rotation()
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

/// The `width x width / 2` equirectangular panorama of a dual-fisheye frame
/// (`2S x S`, the front lens on the left; `lenses` in the pixels of an
/// `S x S` half), in [`crate::equirect`]'s convention with the front lens's
/// axis at the centre column. `back` is the back lens's rotation on the
/// camera (a direction in its frame to the front lens's, as
/// [`back_lens_rotation`]). The panorama is stitched on a sphere around
/// the front lens: `parallax` is the back lens's centre in the front
/// lens's frame divided by the sphere's radius (zero for a sphere at
/// infinity), and the back lens samples where it sees that sphere. Each
/// direction takes the lens whose axis is nearer; within `blend` radians of
/// where both are as far, the two are mixed, linearly across the band.
/// With a `disparity` measured on the same frame (with the same `back` and
/// `parallax`), the band also morphs one lens into the other: a thing the
/// two see apart moves from where the front lens sees it to where the back
/// lens does, so it shows once, wherever it is. Bilinear samples, no
/// filtering: make `width` about the lenses' resolution (`2π` times
/// [`FisheyeLens::axial_resolution`]) or downscale the frame first.
pub fn stitch_equirect(
    frame: &image::RgbImage,
    lenses: &[FisheyeLens; 2],
    back: &Matrix3<f64>,
    parallax: &Vector3<f64>,
    width: u32,
    blend: f64,
    disparity: Option<&SeamDisparity>,
) -> image::RgbImage {
    use std::f64::consts::PI;
    let (w, h) = (width.max(2) & !1, width.max(2) / 2);
    let size = frame.height();
    let mut pano = image::RgbImage::new(w, h);
    if frame.width() != 2 * size || size == 0 {
        return pano;
    }
    let to_back = back.transpose();
    let seen = SEEN_DEG.to_radians().cos();
    let row_bytes = 3 * w as usize;
    let rows_per = (h as usize).div_ceil(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    );
    std::thread::scope(|scope| {
        for (chunk, rows) in pano.chunks_mut(rows_per * row_bytes).enumerate() {
            let to_back = &to_back;
            scope.spawn(move || {
                for (r, row) in rows.chunks_mut(row_bytes).enumerate() {
                    let y = chunk * rows_per + r;
                    let lat = PI / 2.0 - (y as f64 + 0.5) / h as f64 * PI;
                    for (x, out) in row.chunks_mut(3).enumerate() {
                        let lon = (x as f64 + 0.5) / w as f64 * 2.0 * PI - PI;
                        let ray =
                            Vector3::new(lat.cos() * lon.sin(), -lat.sin(), lat.cos() * lon.cos());
                        let ray_back = (to_back * (ray - parallax)).normalize();
                        let off_front = ray.z.clamp(-1.0, 1.0).acos();
                        let off_back = ray_back.z.clamp(-1.0, 1.0).acos();
                        let w_front = if blend > 0.0 {
                            ((off_back - off_front) / (2.0 * blend) + 0.5).clamp(0.0, 1.0)
                        } else if off_front <= off_back {
                            1.0
                        } else {
                            0.0
                        };
                        // Across the band, each lens samples where it sees
                        // what shows here, part of the way to the other's view.
                        let (ray_front, ray_back) = match disparity {
                            Some(seam) if w_front > 0.0 && w_front < 1.0 => {
                                let (u, v) = seam.coordinates(&ray);
                                let d = seam.disparity(u, v);
                                let to = seam.at(u, v + w_front * d);
                                (
                                    seam.at(u, v - (1.0 - w_front) * d),
                                    (to_back * (to - parallax)).normalize(),
                                )
                            }
                            _ => (ray, ray_back),
                        };
                        // A lens that does not see there (beyond its image
                        // circle) leaves it to the other.
                        let samples = [
                            (&lenses[0], w_front, &ray_front, 0),
                            (&lenses[1], 1.0 - w_front, &ray_back, size),
                        ]
                        .map(|(lens, weight, dir, x0)| {
                            let c = (weight > 0.0)
                                .then(|| lens.project(dir))
                                .flatten()
                                .and_then(|(px, py)| sample_half(frame, x0, size, px, py));
                            (weight, c, dir.z >= seen)
                        });
                        let any_seen = samples.iter().any(|(_, c, s)| c.is_some() && *s);
                        let (mut rgb, mut total) = ([0.0; 3], 0.0);
                        for (weight, c, s) in samples {
                            if let Some(c) = c.filter(|_| s || !any_seen) {
                                for k in 0..3 {
                                    rgb[k] += weight * c[k];
                                }
                                total += weight;
                            }
                        }
                        let total = if total > 0.0 { total } else { 1.0 };
                        for k in 0..3 {
                            out[k] = (rgb[k] / total).round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            });
        }
    });
    pano
}

/// How far apart the two lenses see things along their seam, for
/// [`stitch_equirect`] to morph one lens into the other across it.
///
/// The seam's frame turns about the lenses' baseline: a direction is `u`
/// around it and `v` from the plane across it, towards the front lens's
/// axis. Where the front lens sees `(u, v)`, the back lens, as
/// [`stitch_equirect`] samples it, sees the same thing at `(u, v + D)`. The
/// lenses sit apart along the baseline, so a thing nearer or farther than
/// the stitching distance moves along `v` only: `D` is the disparity of a
/// stereo pair rectified about the baseline, which
/// [`SeamDisparity::measure`] matches.
#[derive(Debug, Clone, PartialEq)]
pub struct SeamDisparity {
    /// The seam's frame in the front lens's (columns: `u = 0`, `u = π/2`,
    /// and the baseline towards the front lens).
    frame: Matrix3<f64>,
    /// Samples per radian, along and across the seam.
    per_rad: f64,
    /// `v` at the top of the first row (radians); rows go towards the back
    /// lens.
    top: f64,
    width: usize,
    height: usize,
    /// `D` per sample (radians).
    values: Vec<f32>,
}

/// How far across the seam [`SeamDisparity::measure`] looks, either side
/// (degrees): the lenses both see about 8° either side.
const SEAM_HALF_DEG: f64 = 7.0;
/// The largest disparity [`SeamDisparity::measure`] looks for (degrees):
/// the 3.2 cm baseline's parallax at half a metre.
const MAX_DISPARITY_DEG: f64 = 3.5;
/// How far off its axis a lens's samples count as seen (degrees); the
/// camera blacks out beyond about 98°.
const SEEN_DEG: f64 = 96.0;

impl SeamDisparity {
    /// Matches the two lenses of `frame` along their seam, as
    /// [`stitch_equirect`] takes `frame`, `lenses`, `back` and `parallax`;
    /// `baseline` is the back lens's centre in the front lens's frame.
    ///
    /// Both lenses are rendered into the seam's frame, about 7° either
    /// side of it at half the lenses' axial resolution, matched by 7 x 7
    /// census costs aggregated along four paths (semi-global matching) for
    /// disparities up to 3.5°, and the result smoothed. Where a lens does
    /// not see, or nothing matches, `D` comes from the neighbours, else 0.
    pub fn measure(
        frame: &image::RgbImage,
        lenses: &[FisheyeLens; 2],
        back: &Matrix3<f64>,
        parallax: &Vector3<f64>,
        baseline: &Vector3<f64>,
    ) -> Self {
        let per_rad = 0.5
            * lenses[0]
                .axial_resolution()
                .max(lenses[1].axial_resolution());
        let top = SEAM_HALF_DEG.to_radians();
        let width = (2.0 * std::f64::consts::PI * per_rad).round().max(8.0) as usize;
        let height = (2.0 * top * per_rad).round().max(8.0) as usize;
        let mut seam = Self {
            frame: seam_frame(baseline),
            per_rad,
            top,
            width,
            height,
            values: vec![0.0; width * height],
        };
        let size = frame.height();
        if frame.width() != 2 * size || size == 0 {
            return seam;
        }
        let [front, back_strip] = seam.render(frame, lenses, back, parallax);
        let range = (MAX_DISPARITY_DEG.to_radians() * per_rad).round() as usize;
        let raw = sgm_disparity(&front, &back_strip, width, height, range);
        seam.values = smooth_disparity(&raw, width, height)
            .into_iter()
            .map(|d| (d / per_rad) as f32)
            .collect();
        seam
    }

    /// The front lens's and the back lens's grey seam strips (`NaN` where
    /// a lens does not see), each sample the mean of 2 x 2 rays.
    fn render(
        &self,
        frame: &image::RgbImage,
        lenses: &[FisheyeLens; 2],
        back: &Matrix3<f64>,
        parallax: &Vector3<f64>,
    ) -> [Vec<f32>; 2] {
        let size = frame.height();
        let to_back = back.transpose();
        let seen = SEEN_DEG.to_radians().cos();
        let mut strips = [
            vec![0f32; self.width * self.height],
            vec![0f32; self.width * self.height],
        ];
        for (k, strip) in strips.iter_mut().enumerate() {
            par_rows(strip, self.width, |first, rows| {
                for (r, row) in rows.chunks_mut(self.width).enumerate() {
                    let j = first + r;
                    for (i, out) in row.iter_mut().enumerate() {
                        let mut sum = 0.0;
                        for (di, dj) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                            let d = self.direction(i as f64 + di, j as f64 + dj);
                            let ray = if k == 0 {
                                d
                            } else {
                                (to_back * (d - parallax)).normalize()
                            };
                            match (ray.z > seen)
                                .then(|| lenses[k].project(&ray))
                                .flatten()
                                .and_then(|(px, py)| {
                                    sample_half(frame, k as u32 * size, size, px, py)
                                }) {
                                Some(c) => sum += 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2],
                                None => {
                                    sum = f64::NAN;
                                    break;
                                }
                            }
                        }
                        *out = (sum / 4.0) as f32;
                    }
                }
            });
        }
        strips
    }

    /// The direction (front lens's frame) at column `x` and row `y` of the
    /// grid (pixel centres at `+ 0.5`).
    fn direction(&self, x: f64, y: f64) -> Vector3<f64> {
        self.at(x / self.per_rad, self.top - y / self.per_rad)
    }

    /// `ray`'s `(u, v)` in the seam's frame (radians).
    fn coordinates(&self, ray: &Vector3<f64>) -> (f64, f64) {
        let l = self.frame.transpose() * ray;
        (l.y.atan2(l.x), (l.z / l.norm()).clamp(-1.0, 1.0).asin())
    }

    /// The direction at `(u, v)` (front lens's frame).
    fn at(&self, u: f64, v: f64) -> Vector3<f64> {
        self.frame * Vector3::new(v.cos() * u.cos(), v.cos() * u.sin(), v.sin())
    }

    /// `D` at `(u, v)` (radians), bilinear; the nearest row beyond the
    /// grid.
    fn disparity(&self, u: f64, v: f64) -> f64 {
        let x = (u * self.per_rad - 0.5).rem_euclid(self.width as f64);
        let y = ((self.top - v) * self.per_rad - 0.5).clamp(0.0, (self.height - 1) as f64);
        let (x0, y0) = (x.floor() as usize % self.width, y.floor() as usize);
        let (x1, y1) = ((x0 + 1) % self.width, (y0 + 1).min(self.height - 1));
        let (fx, fy) = (x - x.floor(), y - y.floor());
        let p = |x: usize, y: usize| self.values[y * self.width + x] as f64;
        (p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx) * (1.0 - fy)
            + (p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx) * fy
    }

    /// `D` at the direction `ray` (front lens's frame; radians): how much
    /// farther towards the front lens's axis the back lens sees what the
    /// front lens sees along `ray`.
    pub fn disparity_at(&self, ray: &Vector3<f64>) -> f64 {
        let (u, v) = self.coordinates(ray);
        self.disparity(u, v)
    }
}

/// The seam's frame about `baseline` (columns: `u = 0` towards the front
/// lens's x, `u = π/2`, and the baseline towards the front lens).
fn seam_frame(baseline: &Vector3<f64>) -> Matrix3<f64> {
    let e3 = if baseline.norm() > 0.0 {
        -baseline.normalize()
    } else {
        Vector3::z()
    };
    let e1 = (Vector3::x() - e3 * e3.x).normalize();
    Matrix3::from_columns(&[e1, e3.cross(&e1), e3])
}

/// `f` of every index below `width * height`, a block of rows per thread.
fn par_map<T: Send + Clone + Default>(
    width: usize,
    height: usize,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let mut out = vec![T::default(); width * height];
    par_rows(&mut out, width, |first, rows| {
        for (k, o) in rows.iter_mut().enumerate() {
            *o = f(first * width + k);
        }
    });
    out
}

/// Hands `f` blocks of whole rows of `data` (`unit` items each), one per
/// thread, with the index of each block's first row.
fn par_rows<T: Send>(data: &mut [T], unit: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = data.len() / unit.max(1);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (k, block) in data.chunks_mut(per * unit.max(1)).enumerate() {
            let f = &f;
            scope.spawn(move || f(k * per, block));
        }
    });
}

/// Census window of [`sgm_disparity`] (odd).
const CENSUS: usize = 7;
/// Semi-global matching's penalties for a disparity step of one and of
/// more, in census bits.
const SGM_P1: u16 = 8;
const SGM_P2: u16 = 64;
/// How many samples before its first a path around the strip starts.
const SGM_WARM: usize = 128;
/// How much more (percent) the next best disparity, not next to the best,
/// must cost for the best to count.
const SGM_UNIQUE: u32 = 5;

/// The disparity (samples, sub-sample) at each sample of `a` against `b`
/// (`width x height`, `NaN` where unseen): `a(x, y)` matches
/// `b(x, y - d)` for `d` within `±range`, the strips wrapping around in
/// `x`. `NaN` where `a`'s census is incomplete.
fn sgm_disparity(a: &[f32], b: &[f32], width: usize, height: usize, range: usize) -> Vec<f64> {
    let (ca, cb) = (census(a, width, height), census(b, width, height));
    let nd = 2 * range + 1;
    let max_cost = (CENSUS * CENSUS - 1) as u8;
    // Matching costs, `nd` per sample. A disparity that leaves what `b`
    // sees costs the sample's mean, favouring none.
    let mut cost = vec![max_cost / 2; width * height * nd];
    par_rows(&mut cost, width * nd, |first, rows| {
        for (r, row) in rows.chunks_mut(width * nd).enumerate() {
            let y = first + r;
            for x in 0..width {
                let Some(ca) = ca[y * width + x] else {
                    continue;
                };
                let costs = &mut row[x * nd..(x + 1) * nd];
                let (mut sum, mut seen) = (0u32, 0u32);
                for (k, c) in costs.iter_mut().enumerate() {
                    let yb = y as isize + range as isize - k as isize;
                    *c = match (0..height as isize)
                        .contains(&yb)
                        .then(|| cb[yb as usize * width + x])
                        .flatten()
                    {
                        Some(cb) => {
                            let c = (ca ^ cb).count_ones() as u8;
                            sum += c as u32;
                            seen += 1;
                            c
                        }
                        None => u8::MAX,
                    };
                }
                if seen > 0 {
                    let mean = (sum / seen) as u8;
                    for c in costs.iter_mut().filter(|c| **c == u8::MAX) {
                        *c = mean;
                    }
                }
            }
        }
    });
    // Aggregated along the rows both ways, then the columns both ways.
    let mut total = vec![0u16; width * height * nd];
    par_rows(&mut total, width * nd, |first, rows| {
        let mut prev = vec![0u16; nd];
        let mut here = vec![0u16; nd];
        for (r, row) in rows.chunks_mut(width * nd).enumerate() {
            let y = first + r;
            let costs = &cost[y * width * nd..(y + 1) * width * nd];
            for forward in [true, false] {
                prev.iter_mut().for_each(|p| *p = 0);
                // The strip wraps around: each path starts a stretch early,
                // so it has history where it starts counting.
                let warm = width.min(SGM_WARM);
                for step in 0..width + warm {
                    let i = (width - warm + step) % width;
                    let x = if forward { i } else { width - 1 - i };
                    sgm_step(&costs[x * nd..(x + 1) * nd], &prev, &mut here);
                    if step >= warm {
                        for (t, h) in row[x * nd..(x + 1) * nd].iter_mut().zip(&here) {
                            *t += h;
                        }
                    }
                    std::mem::swap(&mut prev, &mut here);
                }
            }
        }
    });
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let block = width.div_ceil(threads);
    let columns: Vec<Vec<u16>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..width)
            .step_by(block)
            .map(|x0| {
                let cost = &cost;
                scope.spawn(move || {
                    let x1 = (x0 + block).min(width);
                    let n = (x1 - x0) * nd;
                    let mut sums = vec![0u16; height * n];
                    let mut prev = vec![0u16; n];
                    let mut here = vec![0u16; n];
                    for down in [true, false] {
                        prev.iter_mut().for_each(|p| *p = 0);
                        for step in 0..height {
                            let y = if down { step } else { height - 1 - step };
                            let costs = &cost[(y * width + x0) * nd..(y * width + x1) * nd];
                            for c in 0..x1 - x0 {
                                let s = c * nd..(c + 1) * nd;
                                sgm_step(&costs[s.clone()], &prev[s.clone()], &mut here[s]);
                            }
                            for (t, h) in sums[y * n..(y + 1) * n].iter_mut().zip(&here) {
                                *t += h;
                            }
                            std::mem::swap(&mut prev, &mut here);
                        }
                    }
                    sums
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("sgm thread"))
            .collect()
    });
    for (b, sums) in columns.iter().enumerate() {
        let x0 = b * block;
        let n = sums.len() / height.max(1);
        for y in 0..height {
            let dst = &mut total[(y * width + x0) * nd..(y * width + x0) * nd + n];
            for (t, s) in dst.iter_mut().zip(&sums[y * n..(y + 1) * n]) {
                *t += s;
            }
        }
    }
    // The cheapest disparity, refined by a parabola through its neighbours;
    // none where another, not next to it, costs about as little (nothing to
    // match, as in a blown-out sky).
    par_map(width, height, |p| {
        if ca[p].is_none() {
            return f64::NAN;
        }
        let s = &total[p * nd..(p + 1) * nd];
        let (k, best) = s
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| **v)
            .map_or((range, 0), |(k, v)| (k, *v as u32));
        let second = s
            .iter()
            .enumerate()
            .filter(|(j, _)| j.abs_diff(k) > 1)
            .map(|(_, v)| *v as u32)
            .min()
            .unwrap_or(u32::MAX);
        if second * 100 <= best * (100 + SGM_UNIQUE) {
            return f64::NAN;
        }
        let mut d = k as f64;
        if k > 0 && k + 1 < nd {
            let (l, c, r) = (s[k - 1] as f64, s[k] as f64, s[k + 1] as f64);
            let curve = l - 2.0 * c + r;
            if curve > 0.0 {
                d += 0.5 * (l - r) / curve;
            }
        }
        d - range as f64
    })
}

/// One step of a semi-global matching path: `here` from this sample's
/// `costs` and the path's previous sample.
fn sgm_step(costs: &[u8], prev: &[u16], here: &mut [u16]) {
    let best = prev.iter().copied().min().unwrap_or(0);
    let n = costs.len();
    for k in 0..n {
        let mut m = prev[k].min(best + SGM_P2);
        if k > 0 {
            m = m.min(prev[k - 1] + SGM_P1);
        }
        if k + 1 < n {
            m = m.min(prev[k + 1] + SGM_P1);
        }
        here[k] = costs[k] as u16 + m - best;
    }
}

/// The `CENSUS x CENSUS` census transform of `img` (`width x height`): a
/// bit per neighbour, set when it is darker than the centre; `None` where
/// the window leaves the image or meets `NaN`. Wraps around in `x`.
fn census(img: &[f32], width: usize, height: usize) -> Vec<Option<u64>> {
    let r = CENSUS / 2;
    par_map(width, height, |p| {
        let (x, y) = (p % width, p / width);
        let centre = img[p];
        if centre.is_nan() || y < r || y + r >= height {
            return None;
        }
        let mut bits = 0u64;
        for dy in 0..CENSUS {
            for dx in 0..CENSUS {
                if dx == r && dy == r {
                    continue;
                }
                let v = img[(y + dy - r) * width + (x + width + dx - r) % width];
                if v.is_nan() {
                    return None;
                }
                bits = (bits << 1) | (v < centre) as u64;
            }
        }
        Some(bits)
    })
}

/// `raw` disparities (`NaN` where none) median-filtered, holes filled
/// from their neighbours (else 0), and smoothed.
fn smooth_disparity(raw: &[f64], width: usize, height: usize) -> Vec<f64> {
    // 5 x 5 median of the known values.
    let r = 2isize;
    let median: Vec<f64> = par_map(width, height, |p| {
        let (x, y) = ((p % width) as isize, (p / width) as isize);
        let (mut window, mut n) = ([0.0; 25], 0);
        for dy in -r..=r {
            let yy = y + dy;
            if !(0..height as isize).contains(&yy) {
                continue;
            }
            for dx in -r..=r {
                let xx = (x + dx).rem_euclid(width as isize);
                let v = raw[yy as usize * width + xx as usize];
                if v.is_finite() {
                    window[n] = v;
                    n += 1;
                }
            }
        }
        if n < 5 {
            return f64::NAN;
        }
        let window = &mut window[..n];
        window.sort_by(f64::total_cmp);
        window[n / 2]
    });
    // Normalised box blurs: a light one where most is known, a wide one to
    // fill the holes.
    let (mut value, mut weight): (Vec<f64>, Vec<f64>) = median
        .iter()
        .map(|&v| if v.is_finite() { (v, 1.0) } else { (0.0, 0.0) })
        .unzip();
    for radius in [2, 2] {
        value = box_blur(&value, width, height, radius);
        weight = box_blur(&weight, width, height, radius);
    }
    let (mut wide, mut wide_weight) = (value.clone(), weight.clone());
    for radius in [8, 8, 8] {
        wide = box_blur(&wide, width, height, radius);
        wide_weight = box_blur(&wide_weight, width, height, radius);
    }
    (0..width * height)
        .map(|p| {
            if weight[p] > 0.5 {
                value[p] / weight[p]
            } else if wide_weight[p] > 1e-3 {
                wide[p] / wide_weight[p]
            } else {
                0.0
            }
        })
        .collect()
}

/// A `(2 radius + 1)`-wide box blur of `img`, wrapping in `x` and clamped
/// in `y`.
fn box_blur(img: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let n = (2 * radius + 1) as f64;
    let mut rows = vec![0.0; img.len()];
    for y in 0..height {
        let line = &img[y * width..(y + 1) * width];
        let mut sum: f64 = (0..=2 * radius)
            .map(|k| line[(k + width - radius) % width])
            .sum();
        for x in 0..width {
            rows[y * width + x] = sum / n;
            sum += line[(x + radius + 1) % width] - line[(x + width - radius) % width];
        }
    }
    let mut out = vec![0.0; img.len()];
    for x in 0..width {
        for y in 0..height {
            let mut sum = 0.0;
            for k in 0..=2 * radius {
                let yy = (y + k).saturating_sub(radius).min(height - 1);
                sum += rows[yy * width + x];
            }
            out[y * width + x] = sum / n;
        }
    }
    out
}

/// A direction seen by both lenses: its ray in the front lens's frame and
/// in the back lens's.
pub type SeamPair = (Vector3<f64>, Vector3<f64>);

/// The pairs of rays SIFT matches between the two lenses along their seam in
/// a dual-fisheye frame (as [`stitch_equirect`] takes it). The seam is cut
/// into eight pinhole views straddling it, 50° along it and 24° across,
/// each rendered from both lenses (the back lens's placed by `back`, as in
/// [`back_lens_rotation`]); a match's rays are exact whatever `back` is,
/// which only aims the views.
#[cfg(feature = "euroc")]
pub fn seam_pairs(
    frame: &image::RgbImage,
    lenses: &[FisheyeLens; 2],
    back: &Matrix3<f64>,
) -> Vec<SeamPair> {
    use visloc_vision::features::sift::{extract_sift, GrayImage, SiftConfig};
    let size = frame.height();
    if frame.width() != 2 * size || size == 0 {
        return Vec::new();
    }
    // About the lenses' resolution along the seam (half their axial one).
    let px_per_rad = 0.5
        * 2.0
        * lenses[0]
            .axial_resolution()
            .max(lenses[1].axial_resolution());
    let (along, across) = (50f64.to_radians(), 24f64.to_radians());
    let f = px_per_rad.max(50.0);
    let w = (2.0 * f * (along / 2.0).tan()).round() as usize;
    let h = (2.0 * f * (across / 2.0).tan()).round() as usize;
    let to_back = back.transpose();
    let sift = SiftConfig::default();
    use rayon::prelude::*;
    (0..8)
        .into_par_iter()
        .flat_map_iter(|k| {
            let mut pairs = Vec::new();
            let phi = k as f64 * std::f64::consts::FRAC_PI_4;
            let (s, c) = phi.sin_cos();
            // View frame to front frame: x along the seam, y towards the front
            // lens's axis, z out through the seam.
            let view = Matrix3::new(-s, 0.0, c, c, 0.0, s, 0.0, 1.0, 0.0);
            let ray = |u: f64, v: f64| {
                view * Vector3::new((u - w as f64 / 2.0) / f, (v - h as f64 / 2.0) / f, 1.0)
            };
            let render = |lens: &FisheyeLens, x0: u32, to_lens: &Matrix3<f64>| -> Vec<f32> {
                let mut out = vec![0f32; w * h];
                for v in 0..h {
                    for u in 0..w {
                        let d = to_lens * ray(u as f64 + 0.5, v as f64 + 0.5);
                        if let Some(c) = lens
                            .project(&d)
                            .and_then(|(px, py)| sample_half(frame, x0, size, px, py))
                        {
                            out[v * w + u] =
                                ((0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0) as f32;
                        }
                    }
                }
                out
            };
            let g0 = render(&lenses[0], 0, &Matrix3::identity());
            let g1 = render(&lenses[1], size, &to_back);
            let (Ok(i0), Ok(i1)) = (GrayImage::new(w, h, &g0), GrayImage::new(w, h, &g1)) else {
                return pairs;
            };
            let (Ok((k0, d0)), Ok((k1, d1))) = (extract_sift(&i0, &sift), extract_sift(&i1, &sift))
            else {
                return pairs;
            };
            // Within 95° of each lens's axis: clear of the black beyond the
            // image circle (about 98°), whose edge SIFT also finds.
            let max_off = 95f64.to_radians().cos();
            for m in crate::euroc::cpu_matches(&d0, &d1) {
                let (a, b) = (&k0[m.query_index], &k1[m.train_index]);
                let ra = ray(a.x + 0.5, a.y + 0.5).normalize();
                let rb = to_back * ray(b.x + 0.5, b.y + 0.5).normalize();
                if ra.z > max_off && rb.z > max_off {
                    pairs.push((ra, rb));
                }
            }
            pairs
        })
        .collect()
}

/// The back lens on the camera as [`fit_back_lens`] finds it along the
/// seam.
#[derive(Debug, Clone, PartialEq)]
pub struct SeamFit {
    /// The back lens's rotation (as [`back_lens_rotation`]).
    pub rotation: Matrix3<f64>,
    /// How far the back lens's optical centre sits from its model's, in its
    /// image's pixels ([`FisheyeLens::shifted`]).
    pub centre_offset: [f64; 2],
    /// The shift towards the lenses' axes the pairs share (radians): the
    /// baseline over the scene's typical distance along the seam.
    pub parallax: f64,
    /// The pairs' median offset across the parallax after the fit
    /// (radians).
    pub across: f64,
    /// How many pairs the fit kept.
    pub pairs: usize,
}

/// The back lens's rotation (as [`back_lens_rotation`]) and optical centre
/// that best align `pairs` (`seam_pairs` (feature `euroc`), whose back-lens rays `back`
/// gave), refined from `initial`. The lenses sit apart along `baseline` (the
/// back lens's centre in the front lens's frame), so a point at a finite
/// distance shows the back lens a direction shifted towards the front
/// lens's axis: the fit takes each pair's offset across that shift exactly
/// (parallax cannot cause it) and along it up to one offset shared by all
/// pairs (the scene's typical parallax), robustly.
pub fn fit_back_lens(
    pairs: &[SeamPair],
    back: &FisheyeLens,
    initial: &Matrix3<f64>,
    baseline: &Vector3<f64>,
) -> SeamFit {
    use nalgebra::{SMatrix, SVector};
    let e = if baseline.norm() > 0.0 {
        -baseline.normalize()
    } else {
        -Vector3::z()
    };
    // Per pair: the directions across and along its parallax, and how the
    // back lens's ray turns as its optical centre moves (radians per pixel):
    // the pixel stays, so the ray is the one a pixel `offset` away saw.
    type PairFrame = (Vector3<f64>, Vector3<f64>, SMatrix<f64, 3, 2>);
    let frames: Vec<Option<PairFrame>> = pairs
        .iter()
        .map(|(a, b)| {
            let n = a.cross(&e);
            if n.norm() < 1e-6 {
                return None;
            }
            let n = n.normalize();
            let side = if b.y.abs() < 0.9 {
                Vector3::y()
            } else {
                Vector3::x()
            };
            let t1 = b.cross(&side).try_normalize(1e-9)?;
            let t2 = b.cross(&t1);
            let h = 1e-4;
            let p0 = back.project(b)?;
            let p1 = back.project(&(b + h * t1))?;
            let p2 = back.project(&(b + h * t2))?;
            let m = nalgebra::Matrix2::new(
                (p1.0 - p0.0) / h,
                (p2.0 - p0.0) / h,
                (p1.1 - p0.1) / h,
                (p2.1 - p0.1) / h,
            );
            let turn = SMatrix::<f64, 3, 2>::from_columns(&[t1, t2]) * m.try_inverse()?;
            Some((n, n.cross(a), -turn))
        })
        .collect();
    let back_ray = |b: &Vector3<f64>, turn: &SMatrix<f64, 3, 2>, offset: &SVector<f64, 2>| {
        (b + turn * offset).normalize()
    };
    let mut rot = *initial;
    let mut offset = SVector::<f64, 2>::zeros();
    let mut along_offset = 0.0;
    // Pairs more than 2° off the current fit are mismatches.
    let gate = |rot: &Matrix3<f64>, offset: &SVector<f64, 2>| -> Vec<bool> {
        pairs
            .iter()
            .zip(&frames)
            .map(|((a, b), f)| {
                f.as_ref().is_some_and(|(_, _, turn)| {
                    a.dot(&(rot * back_ray(b, turn, offset))) > 2f64.to_radians().cos()
                })
            })
            .collect()
    };
    let mut keep = gate(initial, &offset);
    let scale = 0.1f64.to_radians();
    for _ in 0..20 {
        let mut jtj = SMatrix::<f64, 6, 6>::zeros();
        let mut jtr = SVector::<f64, 6>::zeros();
        for (((_, b), f), &k) in pairs.iter().zip(&frames).zip(&keep) {
            let (Some((n, ep, turn)), true) = (f, k) else {
                continue;
            };
            let q = rot * back_ray(b, turn, &offset);
            let dq = rot * turn; // d q / d offset
            for (dir, c, dc) in [(n, 0.0, 0.0), (ep, along_offset, -1.0)] {
                let r = q.dot(dir) - c;
                // Soft-L1 weight for iteratively reweighted least squares.
                let w = 1.0 / (1.0 + (r / scale).powi(2)).sqrt();
                let g = q.cross(dir);
                let d = dq.transpose() * dir;
                let j = SVector::<f64, 6>::from_column_slice(&[g.x, g.y, g.z, d[0], d[1], dc]);
                jtj += w * j * j.transpose();
                jtr += w * j * r;
            }
        }
        // Keep the centre within reach of the linearisation (a few pixels
        // per step) with a light prior.
        jtj += SMatrix::<f64, 6, 6>::from_diagonal(&SVector::from_column_slice(&[
            1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-9,
        ]));
        let Some(step) = jtj.lu().solve(&(-jtr)) else {
            break;
        };
        let omega = Vector3::new(step[0], step[1], step[2]);
        rot = nalgebra::Rotation3::new(omega).into_inner() * rot;
        offset += SVector::<f64, 2>::new(step[3], step[4]);
        along_offset += step[5];
        keep = gate(&rot, &offset);
        if omega.norm() < 1e-9 && step.fixed_rows::<2>(3).norm() < 1e-4 {
            break;
        }
    }
    let mut across: Vec<f64> = pairs
        .iter()
        .zip(&frames)
        .zip(&keep)
        .filter_map(|(((_, b), f), &k)| {
            let (n, _, turn) = f.as_ref().filter(|_| k)?;
            Some((rot * back_ray(b, turn, &offset)).dot(n).abs())
        })
        .collect();
    across.sort_by(f64::total_cmp);
    SeamFit {
        rotation: rot,
        centre_offset: [offset[0], offset[1]],
        parallax: along_offset,
        across: across.get(across.len() / 2).copied().unwrap_or(f64::NAN),
        pairs: across.len(),
    }
}

/// A point seen by one lens in one frame, for [`calibrate_lenses`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LensObservation {
    /// The frame (an index into the frames' poses).
    pub frame: usize,
    /// The lens: 0 the front one, 1 the back one.
    pub lens: usize,
    /// The point (an index into the points).
    pub point: usize,
    /// Where the lens image shows the point (pixels).
    pub pixel: [f64; 2],
}

/// The two lenses of a dual-fisheye camera as [`calibrate_lenses`] finds
/// them, relative to the models it started from.
#[derive(Debug, Clone, PartialEq)]
pub struct LensCalibration {
    /// The lens image size (pixels) the centre offsets are in.
    pub size: u32,
    /// Each lens's optical centre from its model's (pixels, x right, y
    /// down; the front lens first), as [`FisheyeLens::shifted`] takes it.
    pub centre_offsets: [[f64; 2]; 2],
    /// Each lens's focal lengths over its model's
    /// ([`FisheyeLens::with_focal_scale`]).
    pub focal_scales: [f64; 2],
    /// The back lens's rotation on the camera, as [`back_lens_rotation`].
    pub back_rotation: Matrix3<f64>,
    /// The back lens's centre in the front lens's frame (the points'
    /// units).
    pub back_centre: Vector3<f64>,
    /// The observations' median reprojection error before and after
    /// (pixels).
    pub reprojection: [f64; 2],
}

impl LensCalibration {
    /// `lenses`, models of `size` px lens images, calibrated.
    pub fn apply(&self, lenses: &[FisheyeLens; 2], size: u32) -> [FisheyeLens; 2] {
        let s = size as f64 / self.size.max(1) as f64;
        std::array::from_fn(|k| {
            lenses[k]
                .with_focal_scale(self.focal_scales[k])
                .shifted([self.centre_offsets[k][0] * s, self.centre_offsets[k][1] * s])
        })
    }

    /// The calibration as `lenses.txt` holds it: a `key values…` line each.
    pub fn to_text(&self) -> String {
        let [f, b] = self.centre_offsets;
        let r = &self.back_rotation;
        format!(
            "# Insta360 lens calibration (gsplat_photos --insv-calibrate; docs/insta360_lens.md)\n\
             size {}\n\
             front_centre_offset {:.4} {:.4}\n\
             back_centre_offset {:.4} {:.4}\n\
             focal_scales {:.6} {:.6}\n\
             back_rotation {}\n\
             back_centre {:.6} {:.6} {:.6}\n\
             reprojection_px {:.4} {:.4}\n",
            self.size,
            f[0],
            f[1],
            b[0],
            b[1],
            self.focal_scales[0],
            self.focal_scales[1],
            (0..9)
                .map(|k| format!("{:.9}", r[(k / 3, k % 3)]))
                .collect::<Vec<_>>()
                .join(" "),
            self.back_centre.x,
            self.back_centre.y,
            self.back_centre.z,
            self.reprojection[0],
            self.reprojection[1],
        )
    }

    /// Reads [`LensCalibration::to_text`]'s text.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let row = |key: &str, n: usize| -> Result<Vec<f64>, String> {
            let values: Vec<f64> = text
                .lines()
                .find_map(|l| l.strip_prefix(key)?.strip_prefix(' '))
                .ok_or(format!("no {key} line"))?
                .split_whitespace()
                .map(|v| v.parse::<f64>().map_err(|e| format!("{key}: {e}")))
                .collect::<Result<_, _>>()?;
            if values.len() == n {
                Ok(values)
            } else {
                Err(format!("{key}: {} numbers, not {n}", values.len()))
            }
        };
        let (f, b) = (
            row("front_centre_offset", 2)?,
            row("back_centre_offset", 2)?,
        );
        let scales = row("focal_scales", 2)?;
        let c = row("back_centre", 3)?;
        Ok(Self {
            size: row("size", 1)?[0] as u32,
            centre_offsets: [[f[0], f[1]], [b[0], b[1]]],
            focal_scales: [scales[0], scales[1]],
            back_rotation: Matrix3::from_row_slice(&row("back_rotation", 9)?),
            back_centre: Vector3::new(c[0], c[1], c[2]),
            reprojection: row("reprojection_px", 2).map_or([f64::NAN; 2], |r| [r[0], r[1]]),
        })
    }
}

/// Calibrates both lenses of a dual-fisheye camera from points they see:
/// each lens's optical centre and focal lengths, with the back lens's pose
/// on the camera, every frame's pose and the points refined along, in the
/// lens images' pixels (a bundle adjustment through the unified model; the
/// first frame held, soft-L1 at 1 px, the points eliminated).
///
/// `lenses` are the models the pixels are measured against, in `size` px
/// lens images; `frames` the front lens's poses (world to lens, `x = R X +
/// t`); `back` the back lens's from the front lens's (`x_back = R x + t`).
/// Near the axis an optical centre's offset looks like a turn of the lens,
/// but not towards the image circle, where the unified model packs the
/// pixels differently: the offsets stand out from the poses where the
/// lenses see widely. The cost grows with the cube of the frames: a hundred
/// or two are plenty.
pub fn calibrate_lenses(
    lenses: &[FisheyeLens; 2],
    size: u32,
    observations: &[LensObservation],
    frames: &[(Matrix3<f64>, Vector3<f64>)],
    back: &(Matrix3<f64>, Vector3<f64>),
    points: &[Vector3<f64>],
) -> LensCalibration {
    use nalgebra::{DMatrix, DVector, Rotation3, SMatrix, Vector2};
    // The camera-side parameters: 6 per frame but the first (rotation, then
    // translation), then the back lens's rotation and translation, the two
    // centre offsets and the two log focal scales.
    const NG: usize = 12;
    const M: usize = 15; // at most 6 frame + 9 global parameters per observation
    let nf = frames.len();
    let base = 6 * nf.saturating_sub(1);
    let nc = base + NG;
    let mut rot: Vec<Matrix3<f64>> = frames.iter().map(|f| f.0).collect();
    let mut tr: Vec<Vector3<f64>> = frames.iter().map(|f| f.1).collect();
    let (mut rb, mut tb) = *back;
    let mut intr = [0.0f64; 6]; // offsets (front x, y, back x, y), log scales
    let mut pts = points.to_vec();
    let mut lengths: Vec<f64> = pts.iter().map(|p| p.norm()).collect();
    lengths.sort_by(f64::total_cmp);
    let scene = lengths
        .get(lengths.len() / 2)
        .copied()
        .unwrap_or(1.0)
        .max(1e-3);
    let exp = |w: &Vector3<f64>| Rotation3::new(*w).into_inner();
    // An observation's residual (pixels) with the given parameters.
    let residual = |o: &LensObservation,
                    r: &Matrix3<f64>,
                    t: &Vector3<f64>,
                    rb: &Matrix3<f64>,
                    tb: &Vector3<f64>,
                    intr: &[f64; 6],
                    x: &Vector3<f64>|
     -> Option<Vector2<f64>> {
        let mut p = r * x + t;
        if o.lens == 1 {
            p = rb * p + tb;
        }
        let l = o.lens;
        let (u, v) =
            lenses[l].project_adjusted(&p, intr[4 + l].exp(), [intr[2 * l], intr[2 * l + 1]])?;
        Some(Vector2::new(u - o.pixel[0], v - o.pixel[1]))
    };
    let usable: Vec<LensObservation> = observations
        .iter()
        .copied()
        .filter(|o| o.frame < nf && o.point < pts.len() && o.lens < 2)
        .filter(|o| {
            residual(
                o,
                &rot[o.frame],
                &tr[o.frame],
                &rb,
                &tb,
                &intr,
                &pts[o.point],
            )
            .is_some_and(|r| r.norm() < 50.0)
        })
        .collect();
    let mut by_point: Vec<Vec<usize>> = vec![Vec::new(); pts.len()];
    for (i, o) in usable.iter().enumerate() {
        by_point[o.point].push(i);
    }
    let errors = |rot: &[Matrix3<f64>],
                  tr: &[Vector3<f64>],
                  rb: &Matrix3<f64>,
                  tb: &Vector3<f64>,
                  intr: &[f64; 6],
                  pts: &[Vector3<f64>]|
     -> Vec<Option<Vector2<f64>>> {
        usable
            .iter()
            .map(|o| residual(o, &rot[o.frame], &tr[o.frame], rb, tb, intr, &pts[o.point]))
            .collect()
    };
    let cost = |res: &[Option<Vector2<f64>>]| -> f64 {
        res.iter()
            .map(|r| r.map_or(100.0, |r| 2.0 * ((1.0 + r.norm_squared()).sqrt() - 1.0)))
            .sum()
    };
    let median = |res: &[Option<Vector2<f64>>]| -> f64 {
        let mut e: Vec<f64> = res.iter().flatten().map(|r| r.norm()).collect();
        e.sort_by(f64::total_cmp);
        e.get(e.len() / 2).copied().unwrap_or(f64::NAN)
    };
    let mut res = errors(&rot, &tr, &rb, &tb, &intr, &pts);
    let before = median(&res);
    let mut current = cost(&res);
    let mut lambda = 1e-3;
    for _ in 0..50 {
        // Weighted Jacobians per observation (soft-L1 as reweighting).
        struct Lin {
            cols: [usize; M],
            m: usize,
            jc: SMatrix<f64, 2, M>,
            jp: SMatrix<f64, 2, 3>,
            r: Vector2<f64>,
        }
        let lin: Vec<Option<Lin>> = usable
            .iter()
            .zip(&res)
            .map(|(o, r)| {
                let r = (*r)?;
                let w = (1.0 + r.norm_squared()).powf(-0.25);
                let f = o.frame;
                let at = |dr: Vector3<f64>,
                          dt: Vector3<f64>,
                          dbr: Vector3<f64>,
                          dbt: Vector3<f64>,
                          di: [f64; 6],
                          dx: Vector3<f64>| {
                    let mut i2 = intr;
                    for k in 0..6 {
                        i2[k] += di[k];
                    }
                    residual(
                        o,
                        &(exp(&dr) * rot[f]),
                        &(tr[f] + dt),
                        &(exp(&dbr) * rb),
                        &(tb + dbt),
                        &i2,
                        &(pts[o.point] + dx),
                    )
                };
                let z = Vector3::zeros();
                let unit = |k: usize, h: f64| {
                    let mut v = Vector3::zeros();
                    v[k] = h;
                    v
                };
                let diff = |plus: Option<Vector2<f64>>, minus: Option<Vector2<f64>>, h: f64| match (
                    plus, minus,
                ) {
                    (Some(a), Some(b)) => (a - b) / (2.0 * h),
                    _ => Vector2::zeros(),
                };
                let (ha, ht, hi, hk) = (1e-6, 1e-6 * scene, 1e-3, 1e-6);
                let mut cols = [0usize; M];
                let mut jc = SMatrix::<f64, 2, M>::zeros();
                let mut m = 0;
                if f > 0 {
                    for k in 0..6 {
                        let (h, d) = if k < 3 {
                            (ha, unit(k, ha))
                        } else {
                            (ht, unit(k - 3, ht))
                        };
                        let (p, n) = if k < 3 {
                            (at(d, z, z, z, [0.0; 6], z), at(-d, z, z, z, [0.0; 6], z))
                        } else {
                            (at(z, d, z, z, [0.0; 6], z), at(z, -d, z, z, [0.0; 6], z))
                        };
                        jc.set_column(m, &(diff(p, n, h) * w));
                        cols[m] = 6 * (f - 1) + k;
                        m += 1;
                    }
                }
                if o.lens == 1 {
                    for k in 0..6 {
                        let (h, d) = if k < 3 {
                            (ha, unit(k, ha))
                        } else {
                            (ht, unit(k - 3, ht))
                        };
                        let (p, n) = if k < 3 {
                            (at(z, z, d, z, [0.0; 6], z), at(z, z, -d, z, [0.0; 6], z))
                        } else {
                            (at(z, z, z, d, [0.0; 6], z), at(z, z, z, -d, [0.0; 6], z))
                        };
                        jc.set_column(m, &(diff(p, n, h) * w));
                        cols[m] = base + k;
                        m += 1;
                    }
                }
                let l = o.lens;
                for (k, h) in [(2 * l, hi), (2 * l + 1, hi), (4 + l, hk)] {
                    let mut d = [0.0; 6];
                    d[k] = h;
                    let mut dn = [0.0; 6];
                    dn[k] = -h;
                    jc.set_column(
                        m,
                        &(diff(at(z, z, z, z, d, z), at(z, z, z, z, dn, z), h) * w),
                    );
                    cols[m] = base + 6 + k;
                    m += 1;
                }
                let mut jp = SMatrix::<f64, 2, 3>::zeros();
                for k in 0..3 {
                    let d = unit(k, ht);
                    jp.set_column(
                        k,
                        &(diff(
                            at(z, z, z, z, [0.0; 6], d),
                            at(z, z, z, z, [0.0; 6], -d),
                            ht,
                        ) * w),
                    );
                }
                Some(Lin {
                    cols,
                    m,
                    jc,
                    jp,
                    r: r * w,
                })
            })
            .collect();
        // The reduced camera system, the points eliminated, for each damping
        // tried until the cost falls.
        let mut improved = false;
        while lambda < 1e8 {
            let mut s = DMatrix::<f64>::zeros(nc, nc);
            let mut g = DVector::<f64>::zeros(nc);
            for l in lin.iter().flatten() {
                for a in 0..l.m {
                    g[l.cols[a]] += l.jc.column(a).dot(&l.r);
                    for b in 0..l.m {
                        s[(l.cols[a], l.cols[b])] += l.jc.column(a).dot(&l.jc.column(b));
                    }
                }
            }
            let diag: Vec<f64> = (0..nc).map(|k| s[(k, k)]).collect();
            let mut point_solve: Vec<Option<(nalgebra::Matrix3<f64>, Vector3<f64>)>> =
                vec![None; pts.len()];
            for (p, obs) in by_point.iter().enumerate() {
                let mut v = nalgebra::Matrix3::<f64>::zeros();
                let mut bp = Vector3::<f64>::zeros();
                let members: Vec<&Lin> = obs.iter().filter_map(|&i| lin[i].as_ref()).collect();
                if members.len() < 2 {
                    continue;
                }
                for l in &members {
                    v += l.jp.transpose() * l.jp;
                    bp += l.jp.transpose() * l.r;
                }
                for k in 0..3 {
                    v[(k, k)] *= 1.0 + lambda;
                    v[(k, k)] += 1e-12;
                }
                let Some(vi) = v.try_inverse() else { continue };
                // W_i = Jc_i^T Jp_i; S -= W_i V^-1 W_j^T, g -= W_i V^-1 bp.
                let ws: Vec<SMatrix<f64, M, 3>> =
                    members.iter().map(|l| l.jc.transpose() * l.jp).collect();
                for (li, wi) in members.iter().zip(&ws) {
                    let y = wi * vi;
                    let gy = y * bp;
                    for a in 0..li.m {
                        g[li.cols[a]] -= gy[a];
                    }
                    for (lj, wj) in members.iter().zip(&ws) {
                        let block = y * wj.transpose();
                        for a in 0..li.m {
                            for b in 0..lj.m {
                                s[(li.cols[a], lj.cols[b])] -= block[(a, b)];
                            }
                        }
                    }
                }
                point_solve[p] = Some((vi, bp));
            }
            for (k, d) in diag.iter().enumerate() {
                s[(k, k)] += lambda * d + 1e-12;
            }
            let Some(chol) = s.clone().cholesky() else {
                lambda *= 5.0;
                continue;
            };
            let dc = chol.solve(&(-&g));
            // The step: frames, back lens, intrinsics, then the points.
            let mut rot2 = rot.clone();
            let mut tr2 = tr.clone();
            for f in 1..nf {
                let k = 6 * (f - 1);
                rot2[f] = exp(&Vector3::new(dc[k], dc[k + 1], dc[k + 2])) * rot[f];
                tr2[f] = tr[f] + Vector3::new(dc[k + 3], dc[k + 4], dc[k + 5]);
            }
            let rb2 = exp(&Vector3::new(dc[base], dc[base + 1], dc[base + 2])) * rb;
            let tb2 = tb + Vector3::new(dc[base + 3], dc[base + 4], dc[base + 5]);
            let mut intr2 = intr;
            for k in 0..6 {
                intr2[k] += dc[base + 6 + k];
            }
            let mut pts2 = pts.clone();
            for (p, obs) in by_point.iter().enumerate() {
                let Some((vi, bp)) = point_solve[p] else {
                    continue;
                };
                let mut rhs = bp;
                for l in obs.iter().filter_map(|&i| lin[i].as_ref()) {
                    let mut wdc = Vector3::zeros();
                    for a in 0..l.m {
                        wdc += l.jp.transpose() * l.jc.column(a) * dc[l.cols[a]];
                    }
                    rhs += wdc;
                }
                pts2[p] = pts[p] - vi * rhs;
            }
            let res2 = errors(&rot2, &tr2, &rb2, &tb2, &intr2, &pts2);
            let c2 = cost(&res2);
            if c2 < current {
                let gain = (current - c2) / current.max(1e-12);
                (rot, tr, rb, tb, intr, pts, res, current) =
                    (rot2, tr2, rb2, tb2, intr2, pts2, res2, c2);
                lambda = (lambda / 3.0).max(1e-9);
                improved = gain > 1e-8;

                break;
            }
            lambda *= 5.0;
        }
        if !improved {
            break;
        }
    }
    LensCalibration {
        size,
        centre_offsets: [[intr[0], intr[1]], [intr[2], intr[3]]],
        focal_scales: [intr[4].exp(), intr[5].exp()],
        back_rotation: rb.transpose(),
        back_centre: -(rb.transpose() * tb),
        reprojection: [before, median(&res)],
    }
}

/// The bilinear sample at `(px, py)` of the `size x size` half of `frame`
/// starting at column `x0`; `None` outside it.
fn sample_half(frame: &image::RgbImage, x0: u32, size: u32, px: f64, py: f64) -> Option<[f64; 3]> {
    let max = size as f64 - 1.0;
    if !(0.0..=max).contains(&px) || !(0.0..=max).contains(&py) {
        return None;
    }
    let (xa, ya) = (
        px.floor().min(max - 1.0).max(0.0),
        py.floor().min(max - 1.0).max(0.0),
    );
    let (fx, fy) = (px - xa, py - ya);
    let (xa, ya) = (xa as u32 + x0, ya as u32);
    let p = |x: u32, y: u32| frame.get_pixel(x, y).0;
    let (p00, p10, p01, p11) = (p(xa, ya), p(xa + 1, ya), p(xa, ya + 1), p(xa + 1, ya + 1));
    Some(std::array::from_fn(|c| {
        let top = p00[c] as f64 * (1.0 - fx) + p10[c] as f64 * fx;
        let bottom = p01[c] as f64 * (1.0 - fx) + p11[c] as f64 * fx;
        top * (1.0 - fy) + bottom * fy
    }))
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
    fn stitching_puts_each_direction_where_both_lenses_saw_it() {
        // A dual-fisheye frame of a world whose colour depends only on the
        // direction, the back lens 2° past the half turn; stitched with that
        // rotation, the panorama shows the same world, seams included.
        use std::f64::consts::PI;
        let size = 512u32;
        let lens = FisheyeLens::from_insv(&x5_like(), size);
        let back = nalgebra::Rotation3::from_euler_angles(0.01, 0.02, -0.03).into_inner()
            * back_lens_rotation();
        let world = |d: &Vector3<f64>| {
            let d = d.normalize();
            [
                (127.5 + 120.0 * (9.0 * d.x).sin()) as u8,
                (127.5 + 120.0 * (7.0 * d.y + 3.0 * d.z).cos()) as u8,
                (127.5 + 120.0 * (8.0 * d.z).sin()) as u8,
            ]
        };
        // Each lens image: a dense grid of directions within 100° of the
        // axis, splatted to their pixels.
        let mut frame = image::RgbImage::new(2 * size, size);
        let n = 1400;
        for i in 0..n {
            for j in 0..n {
                // Directions within 100° of the axis, densely.
                let th = (i as f64 + 0.5) / n as f64 * 100f64.to_radians();
                let ph = (j as f64 + 0.5) / n as f64 * 2.0 * PI;
                let d = Vector3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos());
                for (k, rot) in [Matrix3::identity(), back].iter().enumerate() {
                    if let Some((x, y)) = lens.project(&d) {
                        let (x, y) = (x.round(), y.round());
                        if x >= 0.0 && y >= 0.0 && x < size as f64 && y < size as f64 {
                            let c = world(&(rot * d));
                            frame.put_pixel(x as u32 + k as u32 * size, y as u32, image::Rgb(c));
                        }
                    }
                }
            }
        }
        let lenses = [lens.clone(), lens];
        // Mean colour error over the panorama (poles aside).
        let error = |pano: &image::RgbImage| {
            let (w, h) = (pano.width(), pano.height());
            let (mut sum, mut n) = (0.0, 0.0);
            for y in (h / 8..7 * h / 8).step_by(5) {
                for x in (0..w).step_by(5) {
                    let lon = (x as f64 + 0.5) / w as f64 * 2.0 * PI - PI;
                    let lat = PI / 2.0 - (y as f64 + 0.5) / h as f64 * PI;
                    let d = Vector3::new(lat.cos() * lon.sin(), -lat.sin(), lat.cos() * lon.cos());
                    let (want, got) = (world(&d), pano.get_pixel(x, y).0);
                    for c in 0..3 {
                        sum += (want[c] as f64 - got[c] as f64).abs();
                        n += 1.0;
                    }
                }
            }
            sum / n
        };
        let none = Vector3::zeros();
        let pano = stitch_equirect(&frame, &lenses, &back, &none, 1024, 2f64.to_radians(), None);
        assert_eq!((pano.width(), pano.height()), (1024, 512));
        let mean = error(&pano);
        // About a level of interpolation error, nothing like a misplaced seam.
        assert!(mean < 1.5, "{mean}");
        // With the plain half turn, the back lens's half is 2° off.
        let plain = error(&stitch_equirect(
            &frame,
            &lenses,
            &back_lens_rotation(),
            &none,
            1024,
            0.0,
            None,
        ));
        assert!(plain > 3.0 * mean, "{plain} vs {mean}");
    }

    #[test]
    fn stitching_at_a_distance_takes_the_lenses_offset_into_account() {
        // The world painted on a sphere 1.5 m around the front lens, the
        // back lens 3.2 cm behind it: stitched on that sphere, the seam
        // matches; stitched at infinity, the back lens's view is up to 1.2°
        // off.
        use std::f64::consts::PI;
        let size = 512u32;
        let lens = FisheyeLens::from_insv(&x5_like(), size);
        let back = back_lens_rotation();
        let centre = Vector3::new(0.0, 0.0, -0.032);
        let radius = 1.5;
        let world = |d: &Vector3<f64>| {
            let d = d.normalize();
            [
                (127.5 + 120.0 * (9.0 * d.x).sin()) as u8,
                (127.5 + 120.0 * (7.0 * d.y + 3.0 * d.z).cos()) as u8,
                (127.5 + 120.0 * (8.0 * d.z).sin()) as u8,
            ]
        };
        let mut frame = image::RgbImage::new(2 * size, size);
        let n = 1400;
        for i in 0..n {
            for j in 0..n {
                let th = (i as f64 + 0.5) / n as f64 * 100f64.to_radians();
                let ph = (j as f64 + 0.5) / n as f64 * 2.0 * PI;
                let d = Vector3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos());
                let Some((x, y)) = lens.project(&d) else {
                    continue;
                };
                let (x, y) = (x.round(), y.round());
                if x < 0.0 || y < 0.0 || x >= size as f64 || y >= size as f64 {
                    continue;
                }
                frame.put_pixel(x as u32, y as u32, image::Rgb(world(&d)));
                // The back lens sees, along `back d`, the sphere's point
                // from its own centre.
                let dir = back * d;
                let b = centre.dot(&dir);
                let along = -b + (b * b - centre.norm_squared() + radius * radius).sqrt();
                let point = centre + along * dir;
                frame.put_pixel(x as u32 + size, y as u32, image::Rgb(world(&point)));
            }
        }
        let lenses = [lens.clone(), lens];
        // The seam's band, 80-100° off the front lens's axis.
        let seam_error = |pano: &image::RgbImage| {
            let (w, h) = (pano.width(), pano.height());
            let (mut sum, mut count) = (0.0, 0.0);
            for y in (h / 8..7 * h / 8).step_by(3) {
                for x in 0..w {
                    let lon = (x as f64 + 0.5) / w as f64 * 2.0 * PI - PI;
                    let lat = PI / 2.0 - (y as f64 + 0.5) / h as f64 * PI;
                    let d = Vector3::new(lat.cos() * lon.sin(), -lat.sin(), lat.cos() * lon.cos());
                    if d.z.abs() > 10f64.to_radians().sin() {
                        continue;
                    }
                    let (want, got) = (world(&d), pano.get_pixel(x, y).0);
                    for c in 0..3 {
                        sum += (want[c] as f64 - got[c] as f64).abs();
                        count += 1.0;
                    }
                }
            }
            sum / count
        };
        let blend = 2f64.to_radians();
        let near = seam_error(&stitch_equirect(
            &frame,
            &lenses,
            &back,
            &(centre / radius),
            1024,
            blend,
            None,
        ));
        let far = seam_error(&stitch_equirect(
            &frame,
            &lenses,
            &back,
            &Vector3::zeros(),
            1024,
            blend,
            None,
        ));
        assert!(near < 2.0, "{near}");
        assert!(far > 3.0 * near, "{far} vs {near}");
    }

    #[test]
    fn semi_global_matching_finds_a_shift_across_the_strip() {
        // A random texture, the second strip shifted by 3 samples on the
        // left half and by -1.5 on the right.
        let (w, h) = (160usize, 48usize);
        let mut seed = 7u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let texture: Vec<f32> = (0..w * (h + 20)).map(|_| 255.0 * noise()).collect();
        // Bilinear along y, the texture's row 10 at y = 0.
        let at = |x: usize, y: f64| {
            let y = y + 10.0;
            let (y0, f) = (y.floor() as usize, y - y.floor());
            texture[y0 * w + x] * (1.0 - f as f32) + texture[(y0 + 1) * w + x] * f as f32
        };
        let shift = |x: usize| if x < w / 2 { 3.0 } else { -1.5 };
        let a: Vec<f32> = (0..w * h).map(|p| at(p % w, (p / w) as f64)).collect();
        // `a(x, y)` is `b(x, y - d)`.
        let b: Vec<f32> = (0..w * h)
            .map(|p| at(p % w, (p / w) as f64 + shift(p % w)))
            .collect();
        let d = sgm_disparity(&a, &b, w, h, 6);
        let mut errors: Vec<f64> = (0..w * h)
            .filter(|p| {
                let (x, y) = (p % w, p / w);
                (8..h - 8).contains(&y) && x % (w / 2) > 4 && x % (w / 2) < w / 2 - 4
            })
            .map(|p| (d[p] - shift(p % w)).abs())
            .collect();
        errors.sort_by(f64::total_cmp);
        let median = errors[errors.len() / 2];
        let p90 = errors[errors.len() * 9 / 10];
        assert!(median < 0.2 && p90 < 0.5, "{median} {p90}");
    }

    #[test]
    fn the_seam_morph_shows_things_once_whatever_their_distance() {
        // A textured world 1 m away above the lenses' horizontal plane and
        // 4 m below it, the back lens 3.2 cm behind the front one: no one
        // stitching distance fits both, the measured disparity does.
        use std::f64::consts::PI;
        let size = 768u32;
        let lens = FisheyeLens::from_insv(&x5_like(), size);
        let back = back_lens_rotation();
        let centre = Vector3::new(0.0, 0.0, -0.032);
        let radius = |d: &Vector3<f64>| if d.y < 0.0 { 1.0 } else { 4.0 };
        // Random cells about 1-2° wide, by direction from the front lens.
        let colour = |d: &Vector3<f64>| {
            let d = d.normalize();
            let cell = |k: f64, salt: u64| {
                let q = (d * k).map(|c| c.floor() as i64 as u64);
                let h = (q.x.wrapping_mul(73856093) ^ q.y.wrapping_mul(19349663))
                    ^ q.z.wrapping_mul(83492791)
                    ^ salt;
                (h.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8
            };
            [cell(40.0, 1), cell(57.0, 2), cell(31.0, 3)]
        };
        // What the back lens sees along `dir` (front lens's frame): the
        // nearer of the two surfaces its ray meets where they are.
        let seen_from_back = |dir: &Vector3<f64>| {
            [1.0, 4.0]
                .into_iter()
                .filter_map(|r: f64| {
                    let b = centre.dot(dir);
                    let t = -b + (b * b - centre.norm_squared() + r * r).sqrt();
                    let p = centre + t * dir;
                    (radius(&p) == r).then_some((t, p))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, p)| p)
        };
        // Each pixel shows the direction of a dense grid nearest its centre.
        let mut frame = image::RgbImage::new(2 * size, size);
        let mut nearest = vec![f64::MAX; (size * size) as usize];
        let (nt, np) = (1100, 3200);
        for i in 0..nt {
            for j in 0..np {
                let th = (i as f64 + 0.5) / nt as f64 * 100f64.to_radians();
                let ph = (j as f64 + 0.5) / np as f64 * 2.0 * PI;
                let d = Vector3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos());
                let Some((x, y)) = lens.project(&d) else {
                    continue;
                };
                let (xr, yr) = (x.round(), y.round());
                if xr < 0.0 || yr < 0.0 || xr >= size as f64 || yr >= size as f64 {
                    continue;
                }
                let (xr, yr) = (xr as u32, yr as u32);
                let off = (x - xr as f64).hypot(y - yr as f64);
                let slot = &mut nearest[(yr * size + xr) as usize];
                if off >= *slot {
                    continue;
                }
                *slot = off;
                frame.put_pixel(xr, yr, image::Rgb(colour(&d)));
                if let Some(p) = seen_from_back(&(back * d)) {
                    frame.put_pixel(xr + size, yr, image::Rgb(colour(&p)));
                }
            }
        }
        let lenses = [lens.clone(), lens];
        let none = Vector3::zeros();
        let seam = SeamDisparity::measure(&frame, &lenses, &back, &none, &centre);
        // The measured disparity against the true one, 3° either side.
        let mut errors = Vec::new();
        for y in 0..seam.height {
            for x in 0..seam.width {
                let d = seam.direction(x as f64 + 0.5, y as f64 + 0.5);
                let (u, v) = seam.coordinates(&d);
                let near_step = (u.abs() - PI / 2.0).abs() > 0.3;
                if v.abs() > 3f64.to_radians() || !near_step {
                    continue;
                }
                let (_, seen) = seam.coordinates(&(radius(&d) * d - centre));
                errors.push((seam.values[y * seam.width + x] as f64 - (seen - v)).abs());
            }
        }
        errors.sort_by(f64::total_cmp);
        let median = errors[errors.len() / 2].to_degrees();
        assert!(median < 0.2, "{median}°");
        // The seam's band in the panorama against a single view of the
        // world: the front lens's for a fixed distance, which leaves the
        // back lens's view of things at other distances beside it; where
        // the morph puts each thing, between the two lenses' views, for the
        // morph (the true disparity, at the front lens's share of the band,
        // as `stitch_equirect` mixes them).
        let blend = 6f64.to_radians();
        let seam_error = |pano: &image::RgbImage, morphed: bool| {
            let (w, h) = (pano.width(), pano.height());
            let (mut sum, mut count) = (0.0, 0.0);
            for y in (h / 8..7 * h / 8).step_by(2) {
                for x in 0..w {
                    let lon = (x as f64 + 0.5) / w as f64 * 2.0 * PI - PI;
                    let lat = PI / 2.0 - (y as f64 + 0.5) / h as f64 * PI;
                    let d = Vector3::new(lat.cos() * lon.sin(), -lat.sin(), lat.cos() * lon.cos());
                    if d.z.abs() > 4f64.to_radians().sin() || d.y.abs() < 0.2 {
                        continue;
                    }
                    let shown = if morphed {
                        let (u, v) = seam.coordinates(&d);
                        let (_, seen) = seam.coordinates(&(radius(&d) * d - centre));
                        let off_front = d.z.clamp(-1.0, 1.0).acos();
                        let off_back = (back.transpose() * d).z.clamp(-1.0, 1.0).acos();
                        let w_front =
                            ((off_back - off_front) / (2.0 * blend) + 0.5).clamp(0.0, 1.0);
                        seam.at(u, v - (1.0 - w_front) * (seen - v))
                    } else {
                        d
                    };
                    let (want, got) = (colour(&shown), pano.get_pixel(x, y).0);
                    for c in 0..3 {
                        sum += (want[c] as f64 - got[c] as f64).abs();
                        count += 1.0;
                    }
                }
            }
            sum / count
        };
        let stitch = |parallax: &Vector3<f64>, seam: Option<&SeamDisparity>| {
            stitch_equirect(&frame, &lenses, &back, parallax, 1024, blend, seam)
        };
        let morphed = seam_error(&stitch(&none, Some(&seam)), true);
        for distance in [1.0, 2.0, 4.0, f64::INFINITY] {
            let fixed = seam_error(&stitch(&(centre / distance), None), false);
            assert!(
                morphed < 0.6 * fixed,
                "{morphed} vs {fixed} at {distance} m"
            );
        }
    }

    #[test]
    fn the_lens_calibration_finds_both_optical_centres() {
        // Points around a walk of 10 frames seen by two lenses whose optical
        // centres sit 4 and 6 px from their models' (1190 px lens images),
        // focal lengths 0.2% apart, the back lens 1° past the half turn.
        use nalgebra::Rotation3;
        let size = 1190u32;
        let model = FisheyeLens::from_insv(&x5_like(), size);
        let lenses = [model.clone(), model.clone()];
        let truth = [
            model.shifted([4.0, -3.0]).with_focal_scale(1.002),
            model.shifted([2.0, 5.5]).with_focal_scale(0.999),
        ];
        let back_rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), 1f64.to_radians())
            .into_inner()
            * back_lens_rotation();
        let back_centre = Vector3::new(0.0, 0.0, -0.032);
        let back = (
            back_rotation.transpose(),
            -(back_rotation.transpose() * back_centre),
        );
        let mut seed = 11u64;
        let mut uniform = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let frames: Vec<(Matrix3<f64>, Vector3<f64>)> = (0..10)
            .map(|f| {
                let r = Rotation3::from_euler_angles(0.03 * (f as f64).sin(), 0.3 * f as f64, 0.02)
                    .into_inner();
                let c = Vector3::new(0.8 * f as f64, 0.05 * (f as f64 * 0.7).sin(), 0.0);
                (r, -(r * c))
            })
            .collect();
        let points: Vec<Vector3<f64>> = (0..700)
            .map(|_| {
                let (z, a) = (
                    2.0 * uniform() - 1.0,
                    2.0 * std::f64::consts::PI * uniform(),
                );
                let d = Vector3::new(
                    (1.0 - z * z).sqrt() * a.cos(),
                    z,
                    (1.0 - z * z).sqrt() * a.sin(),
                );
                Vector3::new(4.0 + 4.0 * uniform(), 0.0, 0.0) + d * (2.0 + 6.0 * uniform())
            })
            .collect();
        let mut observations = Vec::new();
        for (f, (r, t)) in frames.iter().enumerate() {
            for (p, x) in points.iter().enumerate() {
                let front = r * x + t;
                for (lens, ray) in [(0, front), (1, back.0 * front + back.1)] {
                    // Within 90° of the axis, as the lens faces see.
                    if ray.z <= 0.05 * ray.norm() {
                        continue;
                    }
                    let (u, v) = truth[lens].project(&ray).unwrap();
                    let noise = [0.3 * (uniform() - 0.5), 0.3 * (uniform() - 0.5)];
                    observations.push(LensObservation {
                        frame: f,
                        lens,
                        point: p,
                        pixel: [u + noise[0], v + noise[1]],
                    });
                }
            }
        }
        // Started from the models, the half turn, slightly moved poses and
        // points.
        let start: Vec<(Matrix3<f64>, Vector3<f64>)> = frames
            .iter()
            .enumerate()
            .map(|(f, (r, t))| {
                let turn = if f == 0 { 0.0 } else { 0.002 };
                (
                    Rotation3::from_euler_angles(turn, -turn, turn).into_inner() * r,
                    t + Vector3::repeat(0.01 * (f > 0) as u8 as f64),
                )
            })
            .collect();
        let half_turn = back_lens_rotation();
        let start_back = (
            half_turn.transpose(),
            -(half_turn.transpose() * Vector3::new(0.0, 0.0, -0.03)),
        );
        let start_points: Vec<Vector3<f64>> = points
            .iter()
            .map(|p| p + Vector3::new(0.01, -0.01, 0.01))
            .collect();
        let cal = calibrate_lenses(
            &lenses,
            size,
            &observations,
            &start,
            &start_back,
            &start_points,
        );
        for (lens, want) in [[4.0, -3.0], [2.0, 5.5]].iter().enumerate() {
            let got = cal.centre_offsets[lens];
            assert!(
                (got[0] - want[0]).hypot(got[1] - want[1]) < 0.3,
                "lens {lens}: {got:?}"
            );
        }
        assert!(
            (cal.focal_scales[0] - 1.002).abs() < 3e-4,
            "{:?}",
            cal.focal_scales
        );
        assert!(
            (cal.focal_scales[1] - 0.999).abs() < 3e-4,
            "{:?}",
            cal.focal_scales
        );
        let angle = ((cal.back_rotation * back_rotation.transpose()).trace() - 1.0) / 2.0;
        assert!(
            angle.clamp(-1.0, 1.0).acos().to_degrees() < 0.05,
            "{}",
            cal.back_rotation
        );
        assert!(
            cal.reprojection[1] < 0.15 && cal.reprojection[0] > 1.0,
            "{:?}",
            cal.reprojection
        );
        // Through the text file and back.
        let again = LensCalibration::from_text(&cal.to_text()).unwrap();
        assert!((again.centre_offsets[1][1] - cal.centre_offsets[1][1]).abs() < 1e-3);
        assert_eq!(again.size, size);
        // Applied to the models, the calibration gives the true lenses.
        let ray = Vector3::new(0.3, -0.2, 0.9);
        let fixed = cal.apply(&lenses, size);
        let (a, b) = (
            fixed[0].project(&ray).unwrap(),
            truth[0].project(&ray).unwrap(),
        );
        assert!((a.0 - b.0).hypot(a.1 - b.1) < 0.3);
    }

    #[test]
    fn the_seam_fit_finds_the_back_lens_rotation_and_centre() {
        // Directions along the seam seen by the front lens and, at 2 m from
        // the camera, by a back lens 1.3° past the half turn whose optical
        // centre is 9 px right and 4 px up of its model's.
        let size = 3840u32;
        let lens = FisheyeLens::from_insv(&x5_like(), size);
        let truth = nalgebra::Rotation3::from_euler_angles(0.012, -0.018, 0.009).into_inner()
            * back_lens_rotation();
        let offset = [9.0, -4.0];
        let shifted = lens.shifted(offset);
        let baseline = Vector3::new(-0.001, 0.0004, -0.032);
        let distance = 2.0;
        // The model's ray through a pixel, by Newton steps on `project`.
        let unproject = |lens: &FisheyeLens, px: (f64, f64), guess: Vector3<f64>| {
            let mut r = guess.normalize();
            for _ in 0..20 {
                let side = if r.y.abs() < 0.9 {
                    Vector3::y()
                } else {
                    Vector3::x()
                };
                let t1 = r.cross(&side).normalize();
                let t2 = r.cross(&t1);
                let p0 = lens.project(&r).unwrap();
                let h = 1e-6;
                let p1 = lens.project(&(r + h * t1)).unwrap();
                let p2 = lens.project(&(r + h * t2)).unwrap();
                let m = nalgebra::Matrix2::new(
                    (p1.0 - p0.0) / h,
                    (p2.0 - p0.0) / h,
                    (p1.1 - p0.1) / h,
                    (p2.1 - p0.1) / h,
                );
                let d = m.try_inverse().unwrap() * nalgebra::Vector2::new(px.0 - p0.0, px.1 - p0.1);
                r = (r + d[0] * t1 + d[1] * t2).normalize();
            }
            r
        };
        let mut pairs = Vec::new();
        for i in 0..400 {
            let phi = i as f64 * 0.0157 * 4.0;
            let theta = (86.0 + 8.0 * ((i * 7) % 11) as f64 / 10.0).to_radians();
            let front = Vector3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            );
            // The back lens sees the point 2 m away from its own centre.
            let seen = truth.transpose() * (distance * front - baseline);
            let px = shifted.project(&seen).unwrap();
            pairs.push((front, unproject(&lens, px, seen)));
        }
        let fit = fit_back_lens(&pairs, &lens, &back_lens_rotation(), &baseline);
        let angle = ((fit.rotation * truth.transpose()).trace() - 1.0) / 2.0;
        assert!(
            angle.clamp(-1.0, 1.0).acos().to_degrees() < 0.02,
            "{:?}",
            fit.rotation
        );
        assert!(
            (fit.centre_offset[0] - 9.0).abs() < 0.5 && (fit.centre_offset[1] + 4.0).abs() < 0.5,
            "{:?}",
            fit.centre_offset
        );
        // The shared shift: the baseline over the distance.
        assert!(
            (fit.parallax - baseline.norm() / distance).abs() < 0.1 * baseline.norm() / distance,
            "{}",
            fit.parallax
        );
        assert!(
            fit.across < 0.005f64.to_radians() && fit.pairs == 400,
            "{} {}",
            fit.across,
            fit.pairs
        );
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
