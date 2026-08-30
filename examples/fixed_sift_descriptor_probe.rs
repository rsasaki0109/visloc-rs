//! Fixed-keypoint descriptor probe for the opt-in VLFeat-compatible path.
//!
//! This intentionally stays tiny: it re-describes the six-column COLMAP
//! keypoints exported by the accompanying SQLite probe and writes the same
//! `x y score descriptor...` text consumed by the existing matcher.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use visloc_rs::vision::features::sift::{describe_sift_keypoints, GrayImage, SiftConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let keypoint_path = PathBuf::from(
        args.next()
            .ok_or("usage: fixed_sift_descriptor_probe <keypoints.tsv> <images-dir> <out-dir>")?,
    );
    let images_dir = PathBuf::from(
        args.next()
            .ok_or("usage: fixed_sift_descriptor_probe <keypoints.tsv> <images-dir> <out-dir>")?,
    );
    let output_dir = PathBuf::from(
        args.next()
            .ok_or("usage: fixed_sift_descriptor_probe <keypoints.tsv> <images-dir> <out-dir>")?,
    );

    let mut keypoints = BTreeMap::<String, Vec<(f64, f64, f64, f64)>>::new();
    for line in fs::read_to_string(&keypoint_path)?.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 5 {
            continue;
        }
        keypoints.entry(fields[0].to_string()).or_default().push((
            fields[1].parse()?,
            fields[2].parse()?,
            fields[3].parse()?,
            fields[4].parse()?,
        ));
    }
    fs::create_dir_all(&output_dir)?;
    let config = SiftConfig {
        vlfeat_compatible_descriptor: true,
        ..SiftConfig::default()
    };
    for (stem, rows) in keypoints {
        let image_path = images_dir.join(format!("{stem}.png"));
        let image = visloc_io::images::read_common_image(&image_path)?;
        let gray = GrayImage::new(image.width(), image.height(), image.pixels())?;
        let descriptors = describe_sift_keypoints(
            &gray,
            &rows
                .iter()
                .map(|&(x, y, sigma, orientation)| {
                    visloc_rs::vision::features::sift::SiftKeypoint::from_location_scale_orientation(
                        x, y, sigma, orientation,
                    )
                })
                .collect::<Vec<_>>(),
            &config,
        );
        let mut text = String::from("# X Y SCORE D0 D1 ...\n");
        for ((x, y, _, _), descriptor) in rows.iter().zip(descriptors.iter()) {
            text.push_str(&format!("{x:.6} {y:.6} 1.0"));
            for value in descriptor {
                text.push_str(&format!(" {value:.9}"));
            }
            text.push('\n');
        }
        fs::write(output_dir.join(format!("{stem}_features.txt")), text)?;
    }
    Ok(())
}
