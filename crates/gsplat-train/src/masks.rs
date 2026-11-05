//! Per-image masks for SfM: the pixels SIFT keypoints must stay off, such as
//! an operator or a monopod in every frame (COLMAP's
//! `ImageReader.mask_path`).
//!
//! A mask is a PNG named after its image, in a directory of its own (see
//! [`mask_path`]). Zero (black) pixels are masked out and any other value is
//! kept, in the convention of COLMAP; a mask with an
//! alpha channel is read through its alpha, unless the alpha is opaque
//! everywhere (an RGBA export of a black-and-white mask): it then says
//! nothing, and the mask is read through its colour, as one without alpha
//! is. A mask may be smaller or larger
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
/// `IMG_1.jpg.png`) or else `<file stem>.png` (`IMG_1.png`, the stem
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

    /// Fail if a mask does not have the aspect of its image (to 1%, like
    /// the photo loader's size tolerance, and as the image is oriented by its
    /// EXIF tag): a mask of another shape cannot be resized onto the image
    /// without being stretched. `images` are the images the masks were found
    /// for. Reads the headers only, so it can run before any image is decoded.
    pub fn check_aspect(&self, images: &[PathBuf]) -> Result<(), EurocError> {
        for (path, image) in self.paths.iter().zip(images) {
            let (w, h) = image::image_dimensions(path).map_err(|source| EurocError::Image {
                path: path.clone(),
                source,
            })?;
            let (iw, ih) = oriented_dimensions(image)?;
            if !same_aspect((w, h), (iw, ih)) {
                return Err(EurocError::Sfm(format!(
                    "masks: {} is {w}x{h}, not the aspect of {} ({iw}x{ih})",
                    path.display(),
                    image.display()
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

/// The size of `path`'s image after its EXIF orientation is applied, from its
/// header.
fn oriented_dimensions(path: &Path) -> Result<(u32, u32), EurocError> {
    use image::metadata::Orientation;
    use image::ImageDecoder;
    let io = |source| EurocError::Io {
        path: path.to_path_buf(),
        source,
    };
    let err = |source| EurocError::Image {
        path: path.to_path_buf(),
        source,
    };
    let mut decoder = image::ImageReader::open(path)
        .map_err(io)?
        .with_guessed_format()
        .map_err(io)?
        .into_decoder()
        .map_err(err)?;
    let (w, h) = decoder.dimensions();
    Ok(match decoder.orientation().map_err(err)? {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Rotate90FlipH
        | Orientation::Rotate270FlipH => (h, w),
        _ => (w, h),
    })
}

/// Whether `a` and `b` (width, height) have the same aspect within 1%.
fn same_aspect(a: (u32, u32), b: (u32, u32)) -> bool {
    let (cross_a, cross_b) = (a.0 as f64 * b.1 as f64, b.0 as f64 * a.1 as f64);
    a.0 > 0 && a.1 > 0 && (cross_a - cross_b).abs() <= 0.01 * cross_b
}

/// `mask` resized to `width` x `height` by nearest-neighbour sampling at the
/// pixel centres, as `true` where it is masked out: a pixel is kept if it is
/// nonzero (its alpha, if it has one that is not opaque everywhere; any of
/// its colour channels otherwise), and `flip` swaps the two.
fn masked_out(
    mask: &DynamicImage,
    flip: bool,
    width: u32,
    height: u32,
) -> Result<Vec<bool>, String> {
    use image::GenericImageView;
    // An alpha that is opaque everywhere masks nothing: the mask is then in
    // the colour, as written by an editor that adds an alpha channel to a
    // black-and-white mask.
    let opaque = match mask {
        DynamicImage::ImageLumaA8(m) => m.pixels().all(|p| p.0[1] == u8::MAX),
        DynamicImage::ImageLumaA16(m) => m.pixels().all(|p| p.0[1] == u16::MAX),
        DynamicImage::ImageRgba8(m) => m.pixels().all(|p| p.0[3] == u8::MAX),
        DynamicImage::ImageRgba16(m) => m.pixels().all(|p| p.0[3] == u16::MAX),
        _ => false,
    };
    let kept: Box<dyn Fn(u32, u32) -> bool + '_> = match mask {
        DynamicImage::ImageLuma8(m) => Box::new(|x, y| m.get_pixel(x, y).0[0] != 0),
        DynamicImage::ImageLuma16(m) => Box::new(|x, y| m.get_pixel(x, y).0[0] != 0),
        DynamicImage::ImageLumaA8(m) if opaque => Box::new(|x, y| m.get_pixel(x, y).0[0] != 0),
        DynamicImage::ImageLumaA16(m) if opaque => Box::new(|x, y| m.get_pixel(x, y).0[0] != 0),
        DynamicImage::ImageLumaA8(m) => Box::new(|x, y| m.get_pixel(x, y).0[1] != 0),
        DynamicImage::ImageLumaA16(m) => Box::new(|x, y| m.get_pixel(x, y).0[1] != 0),
        DynamicImage::ImageRgb8(m) => Box::new(|x, y| m.get_pixel(x, y).0.iter().any(|&c| c != 0)),
        DynamicImage::ImageRgb16(m) => Box::new(|x, y| m.get_pixel(x, y).0.iter().any(|&c| c != 0)),
        DynamicImage::ImageRgba8(m) if opaque => {
            Box::new(|x, y| m.get_pixel(x, y).0[..3].iter().any(|&c| c != 0))
        }
        DynamicImage::ImageRgba16(m) if opaque => {
            Box::new(|x, y| m.get_pixel(x, y).0[..3].iter().any(|&c| c != 0))
        }
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
/// their order. A keypoint is in the pixel it lies on, which depends on the
/// detector's convention for positions. With `half_pixel_centres` (the
/// VLFeat-compatible detector, COLMAP's convention) pixel `i` covers
/// `[i, i+1)`, so the pixel is `floor(x)`; without it (the default
/// detector, on the CPU and the GPU) pixel `i` is centred on `x = i`, so the
/// pixel is `round(x)`, as in `keypoint_colours`.
pub fn drop_masked_keypoints(
    keypoints: Vec<SiftKeypoint>,
    descriptors: Vec<Vec<f32>>,
    masked: &[bool],
    width: usize,
    half_pixel_centres: bool,
) -> (Vec<SiftKeypoint>, Vec<Vec<f32>>) {
    debug_assert_eq!(keypoints.len(), descriptors.len());
    let height = masked.len() / width.max(1);
    let pixel = |v: f64| {
        let v = if half_pixel_centres { v } else { v + 0.5 };
        v.floor().max(0.0) as usize
    };
    let outside = |k: &SiftKeypoint| {
        let x = pixel(k.x).min(width.saturating_sub(1));
        let y = pixel(k.y).min(height.saturating_sub(1));
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
        // Left half 0, right half 255 (a typical 0/255 mask), and a faint 1.
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

    /// A black-and-white mask saved with an alpha channel that is opaque
    /// everywhere: the alpha masks nothing, so the colour is the mask. Each
    /// image's colour has a black pixel and a nonzero one.
    #[test]
    fn an_opaque_alpha_leaves_the_mask_to_the_colour() {
        let rgba = RgbaImage::from_fn(3, 1, |x, _| match x {
            0 => Rgba([0, 0, 0, 255]),
            1 => Rgba([0, 0, 1, 255]),
            _ => Rgba([255, 255, 255, 255]),
        });
        let rgba16 = image::ImageBuffer::<Rgba<u16>, _>::from_fn(3, 1, |x, _| {
            Rgba([0, 0, if x == 0 { 0 } else { 255 }, u16::MAX])
        });
        let la = GrayAlphaImage::from_fn(3, 1, |x, _| LumaA([if x == 0 { 0 } else { 255 }, 255]));
        let la16 = image::ImageBuffer::<LumaA<u16>, _>::from_fn(3, 1, |x, _| {
            LumaA([if x == 0 { 0 } else { 255 }, u16::MAX])
        });
        for img in [
            DynamicImage::ImageRgba8(rgba.clone()),
            DynamicImage::ImageRgba16(rgba16),
            DynamicImage::ImageLumaA8(la),
            DynamicImage::ImageLumaA16(la16),
        ] {
            assert_eq!(
                masked_out(&img, false, 3, 1).unwrap(),
                [true, false, false],
                "{:?}",
                img.color()
            );
            assert_eq!(
                masked_out(&img, true, 3, 1).unwrap(),
                [false, true, true],
                "{:?} flipped",
                img.color()
            );
        }
        // One pixel less than opaque, and the alpha is the mask again.
        let mut faded = rgba;
        faded.put_pixel(2, 0, Rgba([255, 255, 255, 254]));
        let m = masked_out(&DynamicImage::ImageRgba8(faded), false, 3, 1).unwrap();
        assert_eq!(m, [false, false, false]);
    }

    /// The same through a PNG on disk: an RGBA file of a black-and-white
    /// mask with opaque alpha masks its black half.
    #[test]
    fn an_opaque_rgba_png_masks_its_black_pixels() {
        let tmp = Tmp::new("opaque_rgba");
        RgbaImage::from_fn(8, 6, |x, _| {
            if x < 4 {
                Rgba([0, 0, 0, 255])
            } else {
                Rgba([255, 255, 255, 255])
            }
        })
        .save(tmp.0.join("a.png"))
        .unwrap();
        let masks = ImageMasks::find(&tmp.0, &[PathBuf::from("/x/a.jpg")], false).unwrap();
        let m = masks.load(0, 8, 6).unwrap();
        assert!((0..6).all(|y| (0..8).all(|x| at(&m, 8, x, y) == (x < 4))));
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
    fn mask_path_takes_the_colmap_name_or_the_file_stem() {
        let tmp = Tmp::new("names");
        let dir = &tmp.0;
        let touch = |name: &str| std::fs::write(dir.join(name), b"x").unwrap();
        touch("a.png"); // stem: a.jpg -> a.png
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

    /// A `w x h` JPEG whose EXIF orientation tag says `orientation`.
    fn jpeg_with_orientation(w: u32, h: u32, orientation: u16) -> Vec<u8> {
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
            .encode_image(&RgbImage::new(w, h))
            .unwrap();
        let mut tiff: Vec<u8> = b"II*\0".to_vec();
        tiff.extend(8u32.to_le_bytes());
        tiff.extend(1u16.to_le_bytes());
        tiff.extend(0x0112u16.to_le_bytes());
        tiff.extend(3u16.to_le_bytes());
        tiff.extend(1u32.to_le_bytes());
        tiff.extend(orientation.to_le_bytes());
        tiff.extend(0u16.to_le_bytes());
        tiff.extend(0u32.to_le_bytes());
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xE1];
        out.extend(((tiff.len() + 8) as u16).to_be_bytes());
        out.extend(b"Exif\0\0");
        out.extend(&tiff);
        out.extend(&jpeg[2..]);
        out
    }

    #[test]
    fn a_mask_of_another_aspect_is_an_error_and_another_size_is_not() {
        let tmp = Tmp::new("aspect");
        let (images_dir, masks_dir) = (tmp.0.join("images"), tmp.0.join("masks"));
        std::fs::create_dir_all(&images_dir).unwrap();
        std::fs::create_dir_all(&masks_dir).unwrap();
        // Four-thirds photos; masks of two sizes (same aspect), a square one,
        // and a portrait one for a photo stored landscape and turned by EXIF.
        let png = |dir: &Path, name: &str, w: u32, h: u32| {
            GrayImage::from_pixel(w, h, Luma([255]))
                .save(dir.join(name))
                .unwrap();
        };
        for name in ["small", "big", "square", "turned"] {
            png(&images_dir, &format!("{name}.png"), 16, 12);
        }
        std::fs::write(
            images_dir.join("turned.jpg"),
            jpeg_with_orientation(16, 12, 6),
        )
        .unwrap();
        std::fs::remove_file(images_dir.join("turned.png")).unwrap();
        png(&masks_dir, "small.png", 8, 6);
        png(&masks_dir, "big.png", 40, 30);
        png(&masks_dir, "square.png", 8, 8);
        png(&masks_dir, "turned.png", 12, 16);
        let masks = |names: &[&str]| {
            let images: Vec<PathBuf> = names
                .iter()
                .map(|n| {
                    let ext = if *n == "turned" { "jpg" } else { "png" };
                    images_dir.join(format!("{n}.{ext}"))
                })
                .collect();
            (
                ImageMasks::find(&masks_dir, &images, false).unwrap(),
                images,
            )
        };
        let (ok, images) = masks(&["small", "big"]);
        ok.check_aspect(&images).unwrap();
        assert_eq!(ok.load(0, 16, 12).unwrap().len(), 16 * 12);
        assert_eq!(ok.load(1, 4, 3).unwrap().len(), 4 * 3);
        let (bad, images) = masks(&["small", "square"]);
        let err = bad.check_aspect(&images).unwrap_err().to_string();
        assert!(err.contains("square.png") && err.contains("8x8"), "{err}");
        // `load` checks too, for a caller that skipped `check_aspect`.
        assert!(bad.load(1, 16, 12).is_err());
        assert!(bad.load(0, 16, 12).is_ok());
        // The image is 12x16 once its EXIF orientation is applied.
        let (turned, images) = masks(&["turned"]);
        turned.check_aspect(&images).unwrap();
        // The same photo without the tag would not match the portrait mask.
        std::fs::write(
            images_dir.join("turned.jpg"),
            jpeg_with_orientation(16, 12, 1),
        )
        .unwrap();
        assert!(turned.check_aspect(&images).is_err());
    }

    #[test]
    fn keypoints_in_masked_pixels_are_dropped_with_their_descriptors() {
        let kp = |x: f64, y: f64| SiftKeypoint::from_location_scale_orientation(x, y, 1.0, 0.0);
        // 4x2 image, the right half (x >= 2) masked out.
        let masked = [false, false, true, true, false, false, true, true];
        let positions = [(0.5, 0.5), (1.99, 1.99), (2.0, 0.5), (3.9, 1.5), (2.5, 0.2)];
        let run = |half_pixel: bool| {
            let kps = positions.iter().map(|&(x, y)| kp(x, y)).collect();
            let desc: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32]).collect();
            let (k, d) = drop_masked_keypoints(kps, desc, &masked, 4, half_pixel);
            (k.iter().map(|k| (k.x, k.y)).collect::<Vec<_>>(), d)
        };
        // Pixel i covers [i, i+1): 1.99 is still pixel 1; 2.0 is pixel 2.
        let (k, d) = run(true);
        assert_eq!(k, [(0.5, 0.5), (1.99, 1.99)]);
        assert_eq!(d, [vec![0.0], vec![1.0]]);
        // Pixel i is centred on i: 1.99 is already pixel 2 (so is 1.5); 0.5
        // rounds up to pixel 1 and stays.
        let (k, d) = run(false);
        assert_eq!(k, [(0.5, 0.5)]);
        assert_eq!(d, [vec![0.0]]);
    }
}
