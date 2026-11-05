//! Per-image masks for SfM: the pixels SIFT keypoints must stay off, such as
//! an operator or a monopod in every frame (COLMAP's
//! `ImageReader.mask_path`).
//!
//! A mask is a PNG named after its image, in a directory of its own (see
//! [`mask_path`]). Zero (black) pixels are masked out and any other value is
//! kept, in the convention of COLMAP and of the Spirula SfM; a mask with an
//! alpha channel is read through its alpha. A mask may be smaller or larger
//! than the image so long as it has the same aspect: it is resized to the
//! working image with nearest-neighbour sampling, so it stays binary.
//!
//! [`ImageMasks`] only holds paths. A mask is decoded when its image is
//! processed ([`ImageMasks::load`]), never all at once.

use std::path::{Path, PathBuf};

use image::DynamicImage;
use visloc_vision::features::sift::SiftKeypoint;

use crate::euroc::EurocError;

/// The mask of `image` in `dir`: `<file name>.png` (COLMAP's own naming,
/// `IMG_1.jpg.png`) or else `<file stem>.png` (`IMG_1.png`, the Spirula
/// layout), whichever exists. `None` if neither does.
pub fn mask_path(dir: &Path, image: &Path) -> Option<PathBuf> {
    let name = image.file_name()?.to_str()?;
    let stem = image.file_stem()?.to_str()?;
    [format!("{name}.png"), format!("{stem}.png")]
        .into_iter()
        .map(|file| dir.join(file))
        .find(|p| p.is_file())
}

/// The masks of a list of images, found by [`ImageMasks::find`].
#[derive(Debug, Clone)]
pub struct ImageMasks {
    /// Image `i`'s mask file.
    paths: Vec<PathBuf>,
    /// Swap what the masks keep and what they hide.
    flip: bool,
}

impl ImageMasks {
    /// The mask of every image in `images`, looked up in `dir` by
    /// [`mask_path`]. An image without one is an error naming it, not an
    /// image left unmasked: a mask that is silently missing for some frames
    /// puts the masked object back into exactly those frames.
    pub fn find(dir: &Path, images: &[PathBuf], flip: bool) -> Result<Self, EurocError> {
        if !dir.is_dir() {
            return Err(EurocError::Sfm(format!(
                "masks: {} is not a directory",
                dir.display()
            )));
        }
        let found: Vec<Option<PathBuf>> = images.iter().map(|i| mask_path(dir, i)).collect();
        if let Some(first) = found.iter().position(Option::is_none) {
            let missing = found.iter().filter(|p| p.is_none()).count();
            let image = &images[first];
            let name = image.file_name().unwrap_or_default().to_string_lossy();
            let stem = image.file_stem().unwrap_or_default().to_string_lossy();
            return Err(EurocError::Sfm(format!(
                "masks: {missing} of {} images have no mask in {}; the first is {} (looked for \
                 {name}.png and {stem}.png there)",
                images.len(),
                dir.display(),
                image.display(),
            )));
        }
        Ok(Self {
            paths: found.into_iter().flatten().collect(),
            flip,
        })
    }

    /// Fail if a mask does not have the aspect of the `width` x `height`
    /// images (to 1%, like the photo loader's size tolerance): a mask of
    /// another shape cannot be resized onto them without being stretched.
    /// Reads the PNG headers only.
    pub fn check_aspect(&self, width: u32, height: u32) -> Result<(), EurocError> {
        for path in &self.paths {
            let (w, h) = image::image_dimensions(path).map_err(|source| EurocError::Image {
                path: path.clone(),
                source,
            })?;
            if !same_aspect((w, h), (width, height)) {
                return Err(EurocError::Sfm(format!(
                    "masks: {} is {w}x{h}, not the aspect of the {width}x{height} images",
                    path.display()
                )));
            }
        }
        Ok(())
    }

    /// Image `i`'s mask, resized to `width` x `height` (row-major): `true`
    /// where the pixel is masked out.
    pub fn load(&self, i: usize, width: u32, height: u32) -> Result<Vec<bool>, EurocError> {
        let path = &self.paths[i];
        let img = image::open(path).map_err(|source| EurocError::Image {
            path: path.clone(),
            source,
        })?;
        if !same_aspect(
            (
                image::GenericImageView::width(&img),
                image::GenericImageView::height(&img),
            ),
            (width, height),
        ) {
            return Err(EurocError::Sfm(format!(
                "masks: {} is {}x{}, not the aspect of the {width}x{height} image",
                path.display(),
                image::GenericImageView::width(&img),
                image::GenericImageView::height(&img)
            )));
        }
        masked_out(&img, self.flip, width, height)
            .map_err(|e| EurocError::Sfm(format!("masks: {}: {e}", path.display())))
    }
}

/// Whether `a` and `b` (width, height) have the same aspect within 1%.
fn same_aspect(a: (u32, u32), b: (u32, u32)) -> bool {
    let (cross_a, cross_b) = (a.0 as f64 * b.1 as f64, b.0 as f64 * a.1 as f64);
    a.0 > 0 && a.1 > 0 && (cross_a - cross_b).abs() <= 0.01 * cross_b
}

/// `mask` resized to `width` x `height` by nearest-neighbour sampling at the
/// pixel centres, as `true` where it is masked out: a pixel is kept if it is
/// nonzero (its alpha, if it has one; any of its channels otherwise), and
/// `flip` swaps the two.
fn masked_out(
    mask: &DynamicImage,
    flip: bool,
    width: u32,
    height: u32,
) -> Result<Vec<bool>, String> {
    use image::GenericImageView;
    let kept: Box<dyn Fn(u32, u32) -> bool + '_> = match mask {
        DynamicImage::ImageLuma8(m) => Box::new(|x, y| m.get_pixel(x, y).0[0] != 0),
        DynamicImage::ImageLuma16(m) => Box::new(|x, y| m.get_pixel(x, y).0[0] != 0),
        DynamicImage::ImageLumaA8(m) => Box::new(|x, y| m.get_pixel(x, y).0[1] != 0),
        DynamicImage::ImageLumaA16(m) => Box::new(|x, y| m.get_pixel(x, y).0[1] != 0),
        DynamicImage::ImageRgb8(m) => Box::new(|x, y| m.get_pixel(x, y).0.iter().any(|&c| c != 0)),
        DynamicImage::ImageRgb16(m) => Box::new(|x, y| m.get_pixel(x, y).0.iter().any(|&c| c != 0)),
        DynamicImage::ImageRgba8(m) => Box::new(|x, y| m.get_pixel(x, y).0[3] != 0),
        DynamicImage::ImageRgba16(m) => Box::new(|x, y| m.get_pixel(x, y).0[3] != 0),
        other => return Err(format!("unsupported pixel format {:?}", other.color())),
    };
    let (mw, mh) = mask.dimensions();
    let nearest = |v: u32, from: u32, to: u32| -> u32 {
        (((v as f64 + 0.5) * from as f64 / to as f64) as u32).min(from - 1)
    };
    let columns: Vec<u32> = (0..width).map(|x| nearest(x, mw, width)).collect();
    Ok((0..height)
        .flat_map(|y| {
            let sy = nearest(y, mh, height);
            let columns = &columns;
            let kept = &kept;
            columns.iter().map(move |&sx| kept(sx, sy) == flip)
        })
        .collect())
}

/// `keypoints` and their `descriptors` without the keypoints inside the
/// masked-out pixels of `masked` (`width` wide, `true` = masked out), in
/// their order. A keypoint is in the pixel its position falls in: positions
/// are in COLMAP's convention, where pixel `(i, j)` covers `[i, i+1) x
/// [j, j+1)`.
pub fn drop_masked_keypoints(
    keypoints: Vec<SiftKeypoint>,
    descriptors: Vec<Vec<f32>>,
    masked: &[bool],
    width: usize,
) -> (Vec<SiftKeypoint>, Vec<Vec<f32>>) {
    debug_assert_eq!(keypoints.len(), descriptors.len());
    let height = masked.len() / width.max(1);
    let outside = |k: &SiftKeypoint| {
        let x = (k.x.floor().max(0.0) as usize).min(width.saturating_sub(1));
        let y = (k.y.floor().max(0.0) as usize).min(height.saturating_sub(1));
        masked[y * width + x]
    };
    keypoints
        .into_iter()
        .zip(descriptors)
        .filter(|(k, _)| !outside(k))
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayAlphaImage, GrayImage, Luma, LumaA, Rgb, RgbImage, Rgba, RgbaImage};

    /// A scratch directory removed on drop.
    struct Tmp(PathBuf);

    impl Tmp {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("visloc_masks_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn at(m: &[bool], width: u32, x: u32, y: u32) -> bool {
        m[(y * width + x) as usize]
    }

    #[test]
    fn zero_is_masked_out_and_anything_else_kept() {
        // Left half 0, right half 255 (Spirula's masks), and a faint 1.
        let g = GrayImage::from_fn(8, 2, |x, y| {
            Luma([if x < 4 {
                0
            } else if y == 0 {
                255
            } else {
                1
            }])
        });
        let m = masked_out(&DynamicImage::ImageLuma8(g), false, 8, 2).unwrap();
        assert!((0..4).all(|x| at(&m, 8, x, 0) && at(&m, 8, x, 1)));
        assert!((4..8).all(|x| !at(&m, 8, x, 0) && !at(&m, 8, x, 1)));
    }

    #[test]
    fn flip_swaps_keep_and_drop() {
        let g = GrayImage::from_fn(4, 1, |x, _| Luma([if x < 2 { 0 } else { 255 }]));
        let img = DynamicImage::ImageLuma8(g);
        let m = masked_out(&img, false, 4, 1).unwrap();
        let f = masked_out(&img, true, 4, 1).unwrap();
        assert_eq!(m, [true, true, false, false]);
        assert_eq!(f, [false, false, true, true]);
    }

    #[test]
    fn sixteen_bit_masks_are_not_rounded_to_eight() {
        // 255/65535 would round to 0 in 8 bits.
        let g = image::ImageBuffer::<Luma<u16>, _>::from_fn(2, 1, |x, _| {
            Luma([if x == 0 { 0u16 } else { 255 }])
        });
        let m = masked_out(&DynamicImage::ImageLuma16(g), false, 2, 1).unwrap();
        assert_eq!(m, [true, false]);
    }

    #[test]
    fn alpha_is_the_mask_when_there_is_one() {
        // Colour says the opposite of alpha: alpha wins.
        let rgba = RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgba([255, 255, 255, 0])
            } else {
                Rgba([0, 0, 0, 255])
            }
        });
        let m = masked_out(&DynamicImage::ImageRgba8(rgba), false, 2, 1).unwrap();
        assert_eq!(m, [true, false]);
        let la = GrayAlphaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                LumaA([255, 0])
            } else {
                LumaA([0, 255])
            }
        });
        let m = masked_out(&DynamicImage::ImageLumaA8(la), false, 2, 1).unwrap();
        assert_eq!(m, [true, false]);
    }

    #[test]
    fn rgb_is_masked_only_where_every_channel_is_zero() {
        let rgb = RgbImage::from_fn(3, 1, |x, _| match x {
            0 => Rgb([0, 0, 0]),
            1 => Rgb([0, 0, 1]),
            _ => Rgb([255, 255, 255]),
        });
        let m = masked_out(&DynamicImage::ImageRgb8(rgb), false, 3, 1).unwrap();
        assert_eq!(m, [true, false, false]);
    }

    #[test]
    fn resizing_is_nearest_neighbour_and_stays_binary() {
        // A 4x4 mask with a distinct pixel each, down to 2x2 and up to 8x8.
        let g = GrayImage::from_fn(4, 4, |x, y| Luma([if (x + y) % 2 == 0 { 0 } else { 255 }]));
        let img = DynamicImage::ImageLuma8(g.clone());
        // Down: output pixel (i, j) samples the source at the centre of its
        // 2x2 block, rounded down: (2i + 1, 2j + 1).
        let down = masked_out(&img, false, 2, 2).unwrap();
        for j in 0..2 {
            for i in 0..2 {
                assert_eq!(
                    at(&down, 2, i, j),
                    g.get_pixel(2 * i + 1, 2 * j + 1).0[0] == 0
                );
            }
        }
        // Up: every output pixel copies the source pixel it lies in.
        let up = masked_out(&img, false, 8, 8).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(at(&up, 8, x, y), g.get_pixel(x / 2, y / 2).0[0] == 0);
            }
        }
        // A non-integer ratio picks the source pixel under the output
        // pixel's centre.
        let odd = masked_out(&img, false, 3, 3).unwrap();
        for j in 0..3u32 {
            for i in 0..3u32 {
                let (sx, sy) = (
                    (((i as f64 + 0.5) * 4.0 / 3.0) as u32),
                    (((j as f64 + 0.5) * 4.0 / 3.0) as u32),
                );
                assert_eq!(at(&odd, 3, i, j), g.get_pixel(sx, sy).0[0] == 0);
            }
        }
    }

    #[test]
    fn mask_path_takes_the_colmap_name_or_the_spirula_stem() {
        let tmp = Tmp::new("names");
        let dir = &tmp.0;
        let touch = |name: &str| std::fs::write(dir.join(name), b"x").unwrap();
        touch("a.png"); // Spirula: a.jpg -> a.png
        touch("b.jpg.png"); // COLMAP: b.jpg -> b.jpg.png
        touch("c.png");
        touch("c.jpg.png");
        assert_eq!(
            mask_path(dir, Path::new("/x/a.jpg")),
            Some(dir.join("a.png"))
        );
        assert_eq!(
            mask_path(dir, Path::new("/x/b.jpg")),
            Some(dir.join("b.jpg.png"))
        );
        // Both present: COLMAP's name wins.
        assert_eq!(
            mask_path(dir, Path::new("/x/c.jpg")),
            Some(dir.join("c.jpg.png"))
        );
        assert_eq!(mask_path(dir, Path::new("/x/d.jpg")), None);
        // A PNG photo's mask is the same stem.
        assert_eq!(
            mask_path(dir, Path::new("/x/a.png")),
            Some(dir.join("a.png"))
        );
    }

    #[test]
    fn a_missing_mask_is_an_error_naming_the_image() {
        let tmp = Tmp::new("missing");
        GrayImage::new(4, 4).save(tmp.0.join("a.png")).unwrap();
        let images = [PathBuf::from("/x/a.jpg"), PathBuf::from("/x/b.jpg")];
        let err = ImageMasks::find(&tmp.0, &images, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 of 2 images have no mask"), "{err}");
        assert!(err.contains("/x/b.jpg"), "{err}");
        assert!(ImageMasks::find(&tmp.0, &images[..1], false).is_ok());
        let err = ImageMasks::find(&tmp.0.join("nope"), &images, false).unwrap_err();
        assert!(err.to_string().contains("not a directory"));
    }

    #[test]
    fn a_mask_of_another_aspect_is_an_error_and_another_size_is_not() {
        let tmp = Tmp::new("aspect");
        // Four-thirds masks of two sizes, and a square one.
        GrayImage::from_pixel(8, 6, Luma([255]))
            .save(tmp.0.join("small.png"))
            .unwrap();
        GrayImage::from_pixel(40, 30, Luma([255]))
            .save(tmp.0.join("big.png"))
            .unwrap();
        GrayImage::from_pixel(8, 8, Luma([255]))
            .save(tmp.0.join("square.png"))
            .unwrap();
        let masks = |names: &[&str]| {
            let images: Vec<PathBuf> = names
                .iter()
                .map(|n| PathBuf::from(format!("/x/{n}.jpg")))
                .collect();
            ImageMasks::find(&tmp.0, &images, false).unwrap()
        };
        let ok = masks(&["small", "big"]);
        ok.check_aspect(16, 12).unwrap();
        assert_eq!(ok.load(0, 16, 12).unwrap().len(), 16 * 12);
        assert_eq!(ok.load(1, 4, 3).unwrap().len(), 4 * 3);
        let bad = masks(&["small", "square"]);
        let err = bad.check_aspect(16, 12).unwrap_err().to_string();
        assert!(err.contains("square.png") && err.contains("8x8"), "{err}");
        // `load` checks too, for a caller that skipped `check_aspect`.
        assert!(bad.load(1, 16, 12).is_err());
        assert!(bad.load(0, 16, 12).is_ok());
    }

    #[test]
    fn keypoints_in_masked_pixels_are_dropped_with_their_descriptors() {
        let kp = |x: f64, y: f64| SiftKeypoint::from_location_scale_orientation(x, y, 1.0, 0.0);
        // 4x2 image, the right half (x >= 2) masked out.
        let masked = [false, false, true, true, false, false, true, true];
        let kps = vec![
            kp(0.5, 0.5),
            kp(1.99, 1.99),
            kp(2.0, 0.5),
            kp(3.9, 1.5),
            kp(2.5, 0.2),
        ];
        let desc: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32]).collect();
        let (k, d) = drop_masked_keypoints(kps, desc, &masked, 4);
        // 1.99 is still pixel 1; 2.0 is pixel 2.
        assert_eq!(
            k.iter().map(|k| (k.x, k.y)).collect::<Vec<_>>(),
            [(0.5, 0.5), (1.99, 1.99)]
        );
        assert_eq!(d, [vec![0.0], vec![1.0]]);
    }
}
