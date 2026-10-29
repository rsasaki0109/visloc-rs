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

/// Calibration pixels per pixel of an X5 video's 3840 px lens image: the
/// video's square spans about 5,322 of the calibration's 5,376 px
/// (docs/insta360_lens.md).
pub const X5_CALIBRATION_PX_PER_VIDEO_PX: f64 = 1.386;

impl FisheyeLens {
    /// `lens` in a `size x size` lens image of the video. The video holds
    /// each lens upside down relative to its calibration: the square around
    /// the middle of the lens's calibration square (not around `(cx, cy)`)
    /// that spans [`X5_CALIBRATION_PX_PER_VIDEO_PX`] x 3840 calibration
    /// pixels, scaled to `size` (docs/insta360_lens.md).
    pub fn from_insv(lens: &InsvLens, size: u32) -> Self {
        let s = size as f64 / (3840.0 * X5_CALIBRATION_PX_PER_VIDEO_PX);
        let mid = size as f64 / 2.0 - 0.5;
        // The middle of the lens's own square in the side-by-side
        // calibration image (pixel centres at integers).
        let side = lens.calibration_size[0] / 2.0;
        let middle_x = (lens.cx / side).floor() * side + side / 2.0 - 0.5;
        let middle_y = lens.calibration_size[1] / 2.0 - 0.5;
        Self {
            xi: lens.xi,
            fx: lens.fx * s,
            fy: lens.fy * s,
            radial: lens.radial,
            // Upside down, the model's y and with it p1 change sign.
            tangential: [-lens.tangential[0], lens.tangential[1]],
            centre: [
                mid + (lens.cx - middle_x) * s,
                mid - (lens.cy - middle_y) * s,
            ],
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
        Some((self.fx * xd + self.centre[0], self.fy * yd + self.centre[1]))
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
/// Bilinear samples, no filtering: make `width` about the lenses'
/// resolution (`2π` times [`FisheyeLens::axial_resolution`]) or downscale
/// the frame first.
pub fn stitch_equirect(
    frame: &image::RgbImage,
    lenses: &[FisheyeLens; 2],
    back: &Matrix3<f64>,
    parallax: &Vector3<f64>,
    width: u32,
    blend: f64,
) -> image::RgbImage {
    use std::f64::consts::PI;
    let (w, h) = (width.max(2) & !1, width.max(2) / 2);
    let size = frame.height();
    let mut pano = image::RgbImage::new(w, h);
    if frame.width() != 2 * size || size == 0 {
        return pano;
    }
    let to_back = back.transpose();
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
                        let mut rgb = [0.0; 3];
                        for (lens, weight, dir, x0) in [
                            (&lenses[0], w_front, &ray, 0),
                            (&lenses[1], 1.0 - w_front, &ray_back, size),
                        ] {
                            if weight <= 0.0 {
                                continue;
                            }
                            if let Some(c) = lens
                                .project(dir)
                                .and_then(|(px, py)| sample_half(frame, x0, size, px, py))
                            {
                                for k in 0..3 {
                                    rgb[k] += weight * c[k];
                                }
                            }
                        }
                        for k in 0..3 {
                            out[k] = rgb[k].round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            });
        }
    });
    pano
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

/// The back lens on the camera as [`fit_back_lens_rotation`] finds it along
/// the seam.
#[derive(Debug, Clone, PartialEq)]
pub struct SeamFit {
    /// The back lens's rotation (as [`back_lens_rotation`]).
    pub rotation: Matrix3<f64>,
    /// The shift towards the lenses' axes the pairs share (radians): the
    /// baseline over the scene's typical distance along the seam.
    pub parallax: f64,
    /// The pairs' median offset across the parallax after the fit
    /// (radians).
    pub across: f64,
    /// How many pairs the fit kept.
    pub pairs: usize,
}

/// The back lens's rotation (as [`back_lens_rotation`]) that best aligns
/// `pairs` ([`seam_pairs`]), refined from `initial`. The lenses sit apart
/// along `baseline` (the back lens's centre in the front lens's frame), so a
/// point at a finite distance shows the back lens a direction shifted towards
/// the front lens's axis: the fit takes each pair's offset across that
/// shift exactly (parallax cannot cause it) and along it up to one offset
/// shared by all pairs (the scene's typical parallax), robustly.
pub fn fit_back_lens_rotation(
    pairs: &[SeamPair],
    initial: &Matrix3<f64>,
    baseline: &Vector3<f64>,
) -> SeamFit {
    let e = if baseline.norm() > 0.0 {
        -baseline.normalize()
    } else {
        -Vector3::z()
    };
    // Each pair's directions across and along its parallax (none on the
    // baseline itself, far from any seam).
    let frames: Vec<Option<(Vector3<f64>, Vector3<f64>)>> = pairs
        .iter()
        .map(|(a, _)| {
            let n = a.cross(&e);
            (n.norm() > 1e-6).then(|| {
                let n = n.normalize();
                (n, n.cross(a))
            })
        })
        .collect();
    let mut rot = *initial;
    let mut along_offset = 0.0;
    // Pairs more than 2° off the initial rotation are mismatches.
    let gate = |rot: &Matrix3<f64>| -> Vec<bool> {
        pairs
            .iter()
            .zip(&frames)
            .map(|((a, b), f)| f.is_some() && a.dot(&(rot * b)) > 2f64.to_radians().cos())
            .collect()
    };
    let mut keep = gate(initial);
    let scale = 0.1f64.to_radians();
    for _ in 0..15 {
        let mut jtj = nalgebra::Matrix4::<f64>::zeros();
        let mut jtr = nalgebra::Vector4::<f64>::zeros();
        for (((_, b), f), &k) in pairs.iter().zip(&frames).zip(&keep) {
            let (Some((n, ep)), true) = (f, k) else {
                continue;
            };
            let q = rot * b;
            for (dir, c, dc) in [(n, 0.0, 0.0), (ep, along_offset, -1.0)] {
                let r = q.dot(dir) - c;
                // Soft-L1 weight for iteratively reweighted least squares.
                let w = 1.0 / (1.0 + (r / scale).powi(2)).sqrt();
                let g = q.cross(dir);
                let j = nalgebra::Vector4::new(g.x, g.y, g.z, dc);
                jtj += w * j * j.transpose();
                jtr += w * j * r;
            }
        }
        let Some(step) = (jtj + nalgebra::Matrix4::identity() * 1e-9)
            .lu()
            .solve(&(-jtr))
        else {
            break;
        };
        let omega = Vector3::new(step[0], step[1], step[2]);
        rot = nalgebra::Rotation3::new(omega).into_inner() * rot;
        along_offset += step[3];
        if omega.norm() < 1e-9 {
            break;
        }
        keep = gate(&rot);
    }
    let mut across: Vec<f64> = pairs
        .iter()
        .zip(&frames)
        .zip(&keep)
        .filter_map(|(((_, b), f), &k)| Some((rot * b).dot(&f.filter(|_| k)?.0).abs()))
        .collect();
    across.sort_by(f64::total_cmp);
    SeamFit {
        rotation: rot,
        parallax: along_offset,
        across: across.get(across.len() / 2).copied().unwrap_or(f64::NAN),
        pairs: across.len(),
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
            calibration_size: [10752.0, 5376.0],
        }
    }

    #[test]
    fn the_video_holds_the_calibration_upside_down_around_its_middle() {
        // A ray lands where the calibration puts its mirror image (y
        // flipped), mapped from the middle of the lens's calibration square
        // upside down and scaled; for the back lens, the right half's.
        let size = 3840;
        let s = X5_CALIBRATION_PX_PER_VIDEO_PX;
        let mid = size as f64 / 2.0 - 0.5;
        for (lens, middle_x) in [
            (x5_like(), 2687.5),
            (
                InsvLens {
                    cx: 8080.0,
                    ..x5_like()
                },
                8063.5,
            ),
        ] {
            let model = FisheyeLens::from_insv(&lens, size);
            for ray in [
                Vector3::new(0.0, 0.0, 1.0),
                Vector3::new(0.3, -0.2, 0.9),
                Vector3::new(-0.8, 0.5, 0.1),
            ] {
                // The calibration's model of the mirrored ray, in its pixels.
                let m = Vector3::new(ray.x, -ray.y, ray.z);
                let zs = m.z + lens.xi * m.norm();
                let (x, y) = (m.x / zs, m.y / zs);
                let r2 = x * x + y * y;
                let [k1, k2, k3] = lens.radial;
                let [p1, p2] = lens.tangential;
                let rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
                let xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
                let yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
                let (cu, cv) = (lens.fx * xd + lens.cx, lens.fy * yd + lens.cy);
                let want = (mid + (cu - middle_x) / s, mid - (cv - 2687.5) / s);
                let got = model.project(&ray).unwrap();
                assert!(
                    (got.0 - want.0).abs() < 1e-9 && (got.1 - want.1).abs() < 1e-9,
                    "{got:?} {want:?}"
                );
            }
        }
        // The axis lands (cx, cy) from the middle, upside down.
        let lens = FisheyeLens::from_insv(&x5_like(), size);
        let (x, y) = lens.project(&Vector3::z()).unwrap();
        assert!((x - (mid + 12.5 / s)).abs() < 1e-9 && (y - (mid + 7.5 / s)).abs() < 1e-9);
        // The 200° circle nearly spans the square: 2646.5 / 1.386 = 1909 px.
        let t = 100f64.to_radians();
        for phi in [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0] {
            let ray = Vector3::new(t.sin() * phi.cos(), t.sin() * phi.sin(), t.cos());
            let (u, v) = lens.project(&ray).unwrap();
            let r = (u - x).hypot(v - y);
            assert!((r - 2646.5 / s).abs() < 0.01 * r, "{r}");
        }
        // About 2,066 px faces match the axis's resolution at 3840 px.
        assert!((natural_face_size(&lens) as i64 - 2054).abs() < 5);
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
        let pano = stitch_equirect(&frame, &lenses, &back, &none, 1024, 2f64.to_radians());
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
        ));
        let far = seam_error(&stitch_equirect(
            &frame,
            &lenses,
            &back,
            &Vector3::zeros(),
            1024,
            blend,
        ));
        assert!(near < 2.0, "{near}");
        assert!(far > 3.0 * near, "{far} vs {near}");
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
