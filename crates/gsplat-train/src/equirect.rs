//! Equirectangular (360°) panoramas: detection, and cube faces cut from
//! them for the pinhole pipeline.
//!
//! Panorama convention: column `x` of a `W x H` image is longitude
//! `(x + 0.5) / W * 2π - π` (zero, straight ahead, at the centre column;
//! positive to the right), row `y` is latitude `π/2 - (y + 0.5) / H * π`
//! (up at the top). Directions are in the camera convention the rest of the
//! crate uses: x right, y down, z forward.

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;

/// The six cube faces: a name and the rotation taking a direction in the
/// face camera's frame to the panorama's frame (its columns are the face
/// camera's x, y and z axes in the panorama frame). Every face is a pinhole
/// camera with a 90° field of view, so `f = size / 2`, centred.
pub fn cube_face_rotations() -> [(&'static str, Matrix3<f64>); 6] {
    let r = |x: [f64; 3], y: [f64; 3], z: [f64; 3]| {
        Matrix3::from_columns(&[Vector3::from(x), Vector3::from(y), Vector3::from(z)])
    };
    [
        ("front", r([1., 0., 0.], [0., 1., 0.], [0., 0., 1.])),
        ("right", r([0., 0., -1.], [0., 1., 0.], [1., 0., 0.])),
        ("back", r([-1., 0., 0.], [0., 1., 0.], [0., 0., -1.])),
        ("left", r([0., 0., 1.], [0., 1., 0.], [-1., 0., 0.])),
        // Image top towards the back, so the up face joins the front face
        // along its bottom edge.
        ("up", r([1., 0., 0.], [0., 0., 1.], [0., -1., 0.])),
        // Image top towards the front.
        ("down", r([1., 0., 0.], [0., 0., -1.], [0., 1., 0.])),
    ]
}

/// The `size x size`, 90° pinhole view of `pano` through a camera rotated
/// by `rotation` (face frame to panorama frame, as in
/// [`cube_face_rotations`]): bilinear samples, wrapping around in longitude.
/// Downscale the panorama to about `4 * size` wide first: the samples do no
/// filtering of their own.
pub fn cube_face(pano: &image::RgbImage, rotation: &Matrix3<f64>, size: u32) -> image::RgbImage {
    use std::f64::consts::PI;
    let (w, h) = (pano.width() as i64, pano.height() as i64);
    let half = size as f64 / 2.0;
    let mut face = image::RgbImage::new(size, size);
    if w == 0 || h == 0 {
        return face;
    }
    for (u, v, out) in face.enumerate_pixels_mut() {
        let ray = Vector3::new(
            (u as f64 + 0.5 - half) / half,
            (v as f64 + 0.5 - half) / half,
            1.0,
        );
        let d = rotation * ray;
        let lon = d.x.atan2(d.z);
        let lat = (-d.y).atan2(d.x.hypot(d.z));
        let px = (lon / (2.0 * PI) + 0.5) * w as f64 - 0.5;
        let py = (0.5 - lat / PI) * h as f64 - 0.5;
        let (x0, y0) = (px.floor(), py.floor());
        let (fx, fy) = (px - x0, py - y0);
        let at = |x: i64, y: i64| {
            pano.get_pixel(x.rem_euclid(w) as u32, y.clamp(0, h - 1) as u32)
                .0
        };
        let (x0, y0) = (x0 as i64, y0 as i64);
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

/// The optical axis of each cube face in the panorama's frame, in
/// [`cube_face_rotations`] order.
pub fn face_axes() -> [Vector3<f64>; 6] {
    cube_face_rotations().map(|(_, r)| r.column(2).into_owned())
}

/// The cube face (an index into [`cube_face_rotations`]) whose axis lies
/// closest to `ray`, a direction in the panorama's frame; `axes` is
/// [`face_axes`].
pub fn nearest_face(axes: &[Vector3<f64>; 6], ray: &Vector3<f64>) -> usize {
    (0..6)
        .max_by(|&a, &b| ray.dot(&axes[a]).total_cmp(&ray.dot(&axes[b])))
        .expect("six faces")
}

/// The pose of the cube face with rotation `face_rotation` (as in
/// [`cube_face_rotations`]) of a panorama posed `pano`: the same centre, the
/// face's fixed rotation behind the panorama's.
pub fn face_pose(pano: &Pose, face_rotation: &Matrix3<f64>) -> Pose {
    let face_from_pano = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
        face_rotation.transpose(),
    ));
    let w2c = &pano.world_to_camera;
    Pose::from_world_to_camera(
        face_from_pano * w2c.rotation,
        face_from_pano * w2c.translation,
    )
}

/// Whether `img` looks like an equirectangular panorama: 2:1 within 1%, its
/// left and right edges continue into each other (they are neighbours on
/// the sphere), and its top and bottom rows are smooth (each is one pole
/// stretched across the width). A heuristic for a warning, not a proof.
pub fn looks_equirectangular(img: &image::RgbImage) -> bool {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 8 || h < 4 || w.abs_diff(2 * h) as f64 > 0.01 * w as f64 {
        return false;
    }
    let g = |x: usize, y: usize| {
        let p = img.get_pixel(x as u32, y as u32).0;
        0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64
    };
    // Mean luma step across the wrap-around seam, and next to it inside.
    let (mut seam, mut inside) = (0.0, 0.0);
    for y in 0..h {
        seam += (g(0, y) - g(w - 1, y)).abs();
        inside += ((g(0, y) - g(1, y)).abs() + (g(w - 2, y) - g(w - 1, y)).abs()) / 2.0;
    }
    let (seam, inside) = (seam / h as f64, inside / h as f64);
    // Mean luma step along a row.
    let along = |y: usize| (1..w).map(|x| (g(x, y) - g(x - 1, y)).abs()).sum::<f64>() / w as f64;
    let poles = (along(0) + along(h - 1)) / 2.0;
    seam <= 2.0 * inside + 2.0 && poles <= 0.25 * along(h / 2) + 0.5
}

/// Whether the metadata near the start of an image file names the
/// equirectangular projection, as the Photo Sphere XMP
/// (`GPano:ProjectionType`) of most 360° cameras and stitchers does.
pub fn metadata_says_equirectangular(bytes: &[u8]) -> bool {
    // XMP sits in the header segments, after an EXIF block of up to 64 KiB.
    const NEEDLE: &[u8] = b"equirectangular";
    bytes[..bytes.len().min(256 << 10)]
        .windows(NEEDLE.len())
        .any(|w| w.eq_ignore_ascii_case(NEEDLE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A smooth, non-symmetric brightness over the sphere.
    fn shade(d: &Vector3<f64>) -> f64 {
        let d = d.normalize();
        128.0 + 60.0 * (3.0 * d.x).sin() + 40.0 * (2.0 * d.y + 1.0).cos() + 25.0 * (4.0 * d.z).sin()
    }

    /// `shade` drawn as a `w x w/2` panorama.
    fn panorama(w: u32) -> image::RgbImage {
        let h = w / 2;
        image::RgbImage::from_fn(w, h, |x, y| {
            let lon = (x as f64 + 0.5) / w as f64 * 2.0 * PI - PI;
            let lat = PI / 2.0 - (y as f64 + 0.5) / h as f64 * PI;
            let d = Vector3::new(lat.cos() * lon.sin(), -lat.sin(), lat.cos() * lon.cos());
            let v = shade(&d).round() as u8;
            image::Rgb([v, v / 2, 255 - v])
        })
    }

    #[test]
    fn face_rotations_are_right_handed_and_cover_the_six_axes() {
        let mut axes: Vec<[i64; 3]> = Vec::new();
        for (name, r) in cube_face_rotations() {
            assert!((r.determinant() - 1.0).abs() < 1e-12, "{name}");
            assert!(
                (r.transpose() * r - Matrix3::identity()).norm() < 1e-12,
                "{name}"
            );
            let z = r.column(2);
            axes.push([z.x as i64, z.y as i64, z.z as i64]);
        }
        axes.sort_unstable();
        axes.dedup();
        assert_eq!(axes.len(), 6);
    }

    #[test]
    fn faces_sample_the_panorama_along_their_rays() {
        let size = 64u32;
        let pano = panorama(4 * size);
        for (name, r) in cube_face_rotations() {
            let face = cube_face(&pano, &r, size);
            let half = size as f64 / 2.0;
            let mut err = 0.0;
            for (u, v, p) in face.enumerate_pixels() {
                let ray = Vector3::new(
                    (u as f64 + 0.5 - half) / half,
                    (v as f64 + 0.5 - half) / half,
                    1.0,
                );
                err += (p.0[0] as f64 - shade(&(r * ray)).round()).abs();
            }
            let mean = err / (size * size) as f64;
            assert!(mean < 1.5, "{name}: mean error {mean:.2}");
        }
    }

    #[test]
    fn face_centres_land_where_the_convention_says() {
        // One bright column at longitude +90° (x = 3W/4): the right face's
        // centre column; bright top rows: the up face's centre.
        let (w, size) = (256u32, 64u32);
        let mut pano = image::RgbImage::new(w, w / 2);
        for y in 0..w / 2 {
            pano.put_pixel(3 * w / 4, y, image::Rgb([255, 255, 255]));
            pano.put_pixel(3 * w / 4 - 1, y, image::Rgb([255, 255, 255]));
        }
        for (x, y) in (0..w).flat_map(|x| (0..3).map(move |y| (x, y))) {
            pano.put_pixel(x, y, image::Rgb([0, 255, 0]));
        }
        let faces = cube_face_rotations();
        let right = cube_face(&pano, &faces[1].1, size);
        assert!(right.get_pixel(size / 2, size / 2).0[0] > 100);
        assert_eq!(right.get_pixel(5, size / 2).0[0], 0);
        let up = cube_face(&pano, &faces[4].1, size);
        assert!(up.get_pixel(size / 2, size / 2).0[1] > 200);
        let front = cube_face(&pano, &faces[0].1, size);
        assert_eq!(front.get_pixel(size / 2, size / 2).0, [0, 0, 0]);
    }

    #[test]
    fn detects_panoramas_not_ordinary_wide_photos() {
        assert!(looks_equirectangular(&panorama(512)));
        // The same panorama shifted by a quarter turn is still one.
        let p = panorama(512);
        let shifted = image::RgbImage::from_fn(512, 256, |x, y| *p.get_pixel((x + 128) % 512, y));
        assert!(looks_equirectangular(&shifted));
        // A 2:1 crop of a planar texture: the edges do not meet, the top and
        // bottom rows are as busy as the rest.
        let photo = image::RgbImage::from_fn(512, 256, |x, y| {
            let v = 128.0
                + 60.0 * (x as f64 * 0.05).sin() * (y as f64 * 0.07).cos()
                + 40.0 * (x as f64 * 0.013 + y as f64 * 0.02).sin();
            image::Rgb([v as u8; 3])
        });
        assert!(!looks_equirectangular(&photo));
        // Not 2:1.
        let tall = image::RgbImage::from_fn(384, 256, |x, y| *p.get_pixel(x, y));
        assert!(!looks_equirectangular(&tall));
    }

    #[test]
    fn reads_the_photo_sphere_projection_type() {
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1, 0, 0];
        jpeg.extend_from_slice(
            b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta GPano:ProjectionType=\"Equirectangular\"/>",
        );
        assert!(metadata_says_equirectangular(&jpeg));
        assert!(!metadata_says_equirectangular(&[0xFF, 0xD8, 0xFF, 0xDA]));
    }

    /// A pose that is not special in any axis: no face's rotation commutes
    /// with it, and its translation is not along a face axis.
    fn awkward_pose() -> Pose {
        Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.3, -0.5, 0.2) * 1.7),
            Vector3::new(0.4, -1.3, 2.1),
        )
    }

    #[test]
    fn a_point_on_a_face_axis_lands_at_that_faces_centre() {
        // The world point at distance 3 along face k's axis, seen from the
        // panorama at `pano`, projects to the middle of face k's image.
        // Catches a face rotation applied transposed (R for R^T), applied on
        // the wrong side of the panorama's rotation, or a translation left
        // unrotated: each moves some face's point off its centre pixel.
        let pano = awkward_pose();
        let w2c = &pano.world_to_camera;
        let centre = -(w2c.rotation.inverse() * w2c.translation);
        for ((name, r), axis) in cube_face_rotations().iter().zip(face_axes()) {
            let world = centre + w2c.rotation.inverse() * (axis * 3.0);
            let face = face_pose(&pano, r).world_to_camera;
            let p = face.rotation * world + face.translation;
            assert!(p.z > 0.0, "{name}: behind the face camera");
            assert!(p.x.abs() < 1e-9 && p.y.abs() < 1e-9, "{name}: {p:?}");
            assert!((p.z - 3.0).abs() < 1e-9, "{name}: {p:?}");
            // Same optical centre as the panorama.
            let c = -(face.rotation.inverse() * face.translation);
            assert!((c - centre).norm() < 1e-9, "{name}: centre moved");
        }
    }

    #[test]
    fn the_nearest_face_of_an_axis_is_that_face() {
        let axes = face_axes();
        for (k, a) in axes.iter().enumerate() {
            assert_eq!(nearest_face(&axes, a), k);
            // Tilted a little toward another face, still this one.
            let tilted = a + axes[(k + 1) % 6] * 0.3;
            assert_eq!(nearest_face(&axes, &tilted), k, "face {k}");
        }
        // Up is image -y; a ray mostly up and a little ahead is the up face.
        let up = cube_face_rotations()
            .iter()
            .position(|(n, _)| *n == "up")
            .unwrap();
        assert_eq!(nearest_face(&axes, &Vector3::new(0.1, -0.9, 0.3)), up);
    }
}
