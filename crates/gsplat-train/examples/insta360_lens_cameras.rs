//! The `--camera` rows of an Insta360 photo's two lenses, from the factory
//! calibration in its trailer.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-train --example insta360_lens_cameras -- PHOTO.insp
//! ```
//!
//! An `.insp` holds both lenses side by side, each a square as tall as the
//! photo, with each lens's unified (Mei) model in its trailer
//! (`visloc_io::insv::read_insta360_lenses`). Each lens is printed as the
//! `OPENCV_FISHEYE` camera of its half of the photo that matches it out to
//! 100° off the axis (`FisheyeLens::kannala_brandt`), as `gsplat_photos
//! --camera` takes it: SfM and training then follow the factory lens on the
//! halves as they are, scaled to the working images. The fit's error and
//! the shift of the tangential terms it leaves out go to stderr.

use std::path::PathBuf;

use visloc_gsplat_train::dual_fisheye::FisheyeLens;
use visloc_io::insv::read_insta360_lenses;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: insta360_lens_cameras PHOTO.insp")?;
    let lenses = read_insta360_lenses(&path)?;
    if lenses.is_empty() {
        return Err(format!("{}: no lens calibration found", path.display()).into());
    }
    let max = 100f64.to_radians();
    for (k, lens) in lenses.iter().enumerate() {
        let side = lens.image_size[1];
        let fisheye = FisheyeLens::in_photo(lens);
        let (p, error) = fisheye.kannala_brandt(max);
        eprintln!(
            "lens {k}: fitted to 100° within {error:.3} px; the tangential terms left out move \
             pixels by up to {:.2} px",
            fisheye.tangential_shift(max)
        );
        let params: Vec<String> = p.iter().map(|v| format!("{v:.8}")).collect();
        println!("OPENCV_FISHEYE {side} {side} {}", params.join(" "));
    }
    Ok(())
}
