//! Compares the classical CornerFeatureExtractor + BruteForceMatcher pipeline
//! against the deep-style HogLikeFeatureExtractor + MutualSoftmaxMatcher
//! pipeline on a synthetic two-view scene.
//!
//! Both frontends feed into the same essential-matrix RANSAC + relative pose
//! recovery, so the comparison isolates the impact of the frontend itself
//! (descriptor + matcher) on inlier ratio and pose error.
//!
//! Run with:
//!   cargo run --example deep_frontend_two_view_demo
//! Optional: --out-dir <dir> writes a summary.txt for documentation/CI.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::Camera;
use visloc_rs::vision::two_view::{RelativePose, RelativePoseEstimator, TwoViewCorrespondence};
use visloc_rs::{
    BruteForceMatcher, CornerFeatureConfig, CornerFeatureExtractor, DeepFeatureExtractor,
    DescriptorMatch, FeatureExtractor, FeatureSet, GrayscaleImage, HogLikeFeatureConfig,
    HogLikeFeatureExtractor, Matcher, MutualSoftmaxConfig, MutualSoftmaxMatcher,
};

const IMAGE_WIDTH: usize = 320;
const IMAGE_HEIGHT: usize = 240;
const FOCAL: f64 = 320.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    if !args.is_empty() {
        eprintln!("usage: cargo run --example deep_frontend_two_view_demo -- [--out-dir <dir>]");
        std::process::exit(2);
    }

    let camera = Camera::pinhole(
        1,
        IMAGE_WIDTH as u32,
        IMAGE_HEIGHT as u32,
        FOCAL,
        FOCAL,
        IMAGE_WIDTH as f64 / 2.0,
        IMAGE_HEIGHT as f64 / 2.0,
    );

    // Pose A: world origin. Pose B: small forward translation + slight yaw.
    // Translation is along -X (right-ward camera motion ~ left-ward map shift)
    // so the parallax exercises the essential matrix estimator.
    let pose_a = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let yaw_b = 0.06_f64;
    let pose_b = Pose::from_world_to_camera(
        UnitQuaternion::from_euler_angles(0.0, yaw_b, 0.0),
        Vector3::new(-0.30, 0.0, 0.0),
    );

    let landmarks = build_synthetic_landmarks();

    // Render greyscale views with a textured ground plane + landmark blobs.
    let image_a = render_view(&camera, &pose_a, &landmarks);
    let image_b = render_view(&camera, &pose_b, &landmarks);

    let truth_relative_translation = pose_b.camera_center_world() - pose_a.camera_center_world();
    let truth_relative_translation_norm = truth_relative_translation.norm();

    println!("== Synthetic two-view scene ==");
    println!(
        "image       : {}x{}, focal {}",
        IMAGE_WIDTH, IMAGE_HEIGHT, FOCAL
    );
    println!("landmarks   : {}", landmarks.len());
    println!(
        "truth t (B-A): [{:.3}, {:.3}, {:.3}] |t|={:.3}",
        truth_relative_translation.x,
        truth_relative_translation.y,
        truth_relative_translation.z,
        truth_relative_translation_norm
    );
    println!("truth yaw  : {:.4} rad", yaw_b);

    let classical = run_pipeline(
        "Classical (Corner + BF + ratio 0.8)",
        &image_a,
        &image_b,
        &camera,
        truth_relative_translation_norm,
        |image| {
            let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
                max_features: 512,
                min_score: 0.05,
                descriptor_radius: 2,
            });
            extractor.extract(image).unwrap()
        },
        |query, train| {
            let matcher = BruteForceMatcher { ratio: Some(0.8) };
            matcher.match_descriptors(query, train)
        },
    );

    let deep = run_pipeline(
        "Deep-style (HOG-like + MutualSoftmax)",
        &image_a,
        &image_b,
        &camera,
        truth_relative_translation_norm,
        |image| {
            let extractor = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
                max_features: 512,
                min_corner_score: 0.05,
                descriptor_clip: 0.2,
                orient: false,
            });
            let deep_features = extractor.extract_deep(image).unwrap();
            deep_features.into_feature_set()
        },
        |query, train| {
            let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
                temperature: 25.0,
                min_confidence: 0.15,
                emit_ratio_metadata: false,
            });
            matcher.match_descriptors(query, train)
        },
    );

    println!();
    println!("== Summary ==");
    print_diagnostics(&classical);
    print_diagnostics(&deep);

    if let Some(dir) = output_dir.as_ref() {
        write_summary(
            dir,
            truth_relative_translation_norm,
            yaw_b,
            &classical,
            &deep,
        )?;
        println!("wrote {}/summary.txt", dir.display());
    }
    Ok(())
}

#[derive(Debug)]
struct Diagnostics {
    label: String,
    keypoints_a: usize,
    keypoints_b: usize,
    matches: usize,
    inliers: usize,
    inlier_ratio: f64,
    sampson_mean: f64,
    rotation_error_rad: f64,
    translation_direction_error_rad: f64,
    estimated_translation_unit: Vector3<f64>,
}

fn run_pipeline<E, M>(
    label: &str,
    image_a: &GrayscaleImage,
    image_b: &GrayscaleImage,
    camera: &Camera,
    truth_translation_norm: f64,
    extract_fn: E,
    match_fn: M,
) -> Diagnostics
where
    E: Fn(&GrayscaleImage) -> FeatureSet,
    M: Fn(&[Vec<f32>], &[Vec<f32>]) -> Vec<DescriptorMatch>,
{
    let features_a = extract_fn(image_a);
    let features_b = extract_fn(image_b);

    let descriptor_matches = match_fn(&features_a.descriptors, &features_b.descriptors);

    let correspondences: Vec<TwoViewCorrespondence> = descriptor_matches
        .iter()
        .map(|m| {
            TwoViewCorrespondence::new(
                features_a.keypoints[m.query_index],
                features_b.keypoints[m.train_index],
            )
        })
        .collect();

    let estimator = RelativePoseEstimator::default();
    let relative_pose = estimator.estimate_with_scale(&correspondences, camera, 1.0);

    let (rotation_error_rad, translation_direction_error_rad, inliers, sampson_mean, t_unit) =
        match relative_pose {
            Some(RelativePose {
                previous_to_current,
                translation_unit,
                inliers,
                mean_sampson_error,
                ..
            }) => {
                let truth_yaw = 0.06_f64;
                let truth_rotation = UnitQuaternion::from_euler_angles(0.0, truth_yaw, 0.0);
                let r_err = previous_to_current
                    .rotation
                    .rotation_to(&truth_rotation)
                    .angle();
                let truth_translation_dir = Vector3::new(-0.30_f64, 0.0, 0.0).normalize();
                let dot = translation_unit.dot(&truth_translation_dir);
                let dir_err = dot.abs().min(1.0).acos();
                (
                    r_err,
                    dir_err,
                    inliers.len(),
                    mean_sampson_error,
                    translation_unit,
                )
            }
            None => (f64::NAN, f64::NAN, 0, f64::NAN, Vector3::zeros()),
        };

    let inlier_ratio = if descriptor_matches.is_empty() {
        0.0
    } else {
        inliers as f64 / descriptor_matches.len() as f64
    };

    let _ = truth_translation_norm;

    Diagnostics {
        label: label.to_string(),
        keypoints_a: features_a.len(),
        keypoints_b: features_b.len(),
        matches: descriptor_matches.len(),
        inliers,
        inlier_ratio,
        sampson_mean,
        rotation_error_rad,
        translation_direction_error_rad,
        estimated_translation_unit: t_unit,
    }
}

fn print_diagnostics(diag: &Diagnostics) {
    println!("-- {} --", diag.label);
    println!(
        "  keypoints     : A={} B={}",
        diag.keypoints_a, diag.keypoints_b
    );
    println!("  putative match: {}", diag.matches);
    println!(
        "  ransac inliers: {} ({:.3} of putatives)",
        diag.inliers, diag.inlier_ratio
    );
    if diag.sampson_mean.is_finite() {
        println!("  mean Sampson  : {:.5}", diag.sampson_mean);
    } else {
        println!("  mean Sampson  : NA (RANSAC failed)");
    }
    if diag.rotation_error_rad.is_finite() {
        println!(
            "  rotation err  : {:.4} rad ({:.2} deg)",
            diag.rotation_error_rad,
            diag.rotation_error_rad.to_degrees()
        );
        println!(
            "  t direction err: {:.4} rad ({:.2} deg)",
            diag.translation_direction_error_rad,
            diag.translation_direction_error_rad.to_degrees()
        );
        println!(
            "  t unit (est)  : [{:.3}, {:.3}, {:.3}]",
            diag.estimated_translation_unit.x,
            diag.estimated_translation_unit.y,
            diag.estimated_translation_unit.z
        );
    } else {
        println!("  pose         : NA (RANSAC failed)");
    }
}

fn build_synthetic_landmarks() -> Vec<Point3<f64>> {
    let mut points = Vec::new();
    // Cube of points spread over depth so the geometry is well conditioned.
    for ix in -2..=2 {
        for iy in -1..=1 {
            for iz in 0..=2 {
                let x = ix as f64 * 0.6;
                let y = iy as f64 * 0.5;
                let z = 4.0 + iz as f64 * 1.2;
                points.push(Point3::new(x, y, z));
            }
        }
    }
    points
}

fn render_view(camera: &Camera, pose: &Pose, landmarks: &[Point3<f64>]) -> GrayscaleImage {
    let mut pixels = vec![25_u8; IMAGE_WIDTH * IMAGE_HEIGHT];
    // 1) procedurally textured background derived from the camera ray (so
    //    motion induces parallax) — checker-modulated low-amplitude noise.
    for y in 0..IMAGE_HEIGHT {
        for x in 0..IMAGE_WIDTH {
            let nx = (x as f64 - IMAGE_WIDTH as f64 / 2.0) / FOCAL;
            let ny = (y as f64 - IMAGE_HEIGHT as f64 / 2.0) / FOCAL;
            // Project the pixel onto a virtual texture plane at z=8 in world.
            let ray_camera = Vector3::new(nx, ny, 1.0);
            let world_ray = pose.camera_to_world().rotation * ray_camera;
            let cam_origin = pose.camera_center_world();
            let depth_plane = 8.0_f64;
            // Solve cam_origin.z + t * world_ray.z = depth_plane.
            let denom = world_ray.z;
            if denom.abs() < 1e-6 {
                continue;
            }
            let t = (depth_plane - cam_origin.z) / denom;
            if t <= 0.0 {
                continue;
            }
            let world_x = cam_origin.x + t * world_ray.x;
            let world_y = cam_origin.y + t * world_ray.y;
            // Multi-scale checker so HOG/Corner detectors find rich corners.
            let checker_a = ((world_x * 4.0).sin() * (world_y * 4.0).sin()).abs();
            let checker_b = ((world_x * 1.7).cos() * (world_y * 2.3).cos()).abs();
            let stripe = ((world_x + world_y) * 6.0).sin().abs();
            let value = (60.0 + 130.0 * (0.55 * checker_a + 0.30 * checker_b + 0.15 * stripe))
                .clamp(0.0, 255.0) as u8;
            pixels[y * IMAGE_WIDTH + x] = value;
        }
    }

    // 2) Bright dots at landmark projections (helps both detectors agree on
    //    salient points).
    for landmark in landmarks {
        let camera_point = pose.transform_world_point(landmark);
        if camera_point.z <= 0.1 {
            continue;
        }
        let Some(projected) = camera.project(&camera_point) else {
            continue;
        };
        splash_blob(&mut pixels, projected, 240);
    }

    GrayscaleImage::from_luma_u8(IMAGE_WIDTH, IMAGE_HEIGHT, pixels).unwrap()
}

fn splash_blob(pixels: &mut [u8], center: Point2<f64>, intensity: u8) {
    let radius: i32 = 2;
    let cx = center.x.round() as i32;
    let cy = center.y.round() as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let x = cx + dx;
            let y = cy + dy;
            if x < 0 || y < 0 || x >= IMAGE_WIDTH as i32 || y >= IMAGE_HEIGHT as i32 {
                continue;
            }
            let r2 = (dx * dx + dy * dy) as f64;
            if r2 > (radius as f64).powi(2) {
                continue;
            }
            let alpha = (1.0 - r2 / (radius as f64).powi(2)).clamp(0.0, 1.0);
            let index = (y as usize) * IMAGE_WIDTH + x as usize;
            let blended = (pixels[index] as f64) * (1.0 - alpha) + (intensity as f64) * alpha;
            pixels[index] = blended.clamp(0.0, 255.0) as u8;
        }
    }
}

fn parse_output_dir(args: &mut Vec<String>) -> Option<PathBuf> {
    let mut output_dir = None;
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--out-dir" {
            if index + 1 >= args.len() {
                eprintln!("--out-dir requires a path argument");
                std::process::exit(2);
            }
            output_dir = Some(PathBuf::from(args.remove(index + 1)));
            args.remove(index);
        } else {
            index += 1;
        }
    }
    output_dir
}

fn write_summary(
    dir: &Path,
    truth_translation_norm: f64,
    truth_yaw: f64,
    classical: &Diagnostics,
    deep: &Diagnostics,
) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join("summary.txt");
    let mut body = String::new();
    body.push_str(&format!(
        "truth |t|={:.4}  yaw={:.4} rad\n\n",
        truth_translation_norm, truth_yaw
    ));
    for diag in [classical, deep] {
        body.push_str(&format!("[{}]\n", diag.label));
        body.push_str(&format!(
            "  keypoints A={} B={}\n  matches={} inliers={} ratio={:.3}\n",
            diag.keypoints_a, diag.keypoints_b, diag.matches, diag.inliers, diag.inlier_ratio
        ));
        if diag.rotation_error_rad.is_finite() {
            body.push_str(&format!(
                "  sampson_mean={:.5} rot_err_rad={:.4} t_dir_err_rad={:.4}\n\n",
                diag.sampson_mean, diag.rotation_error_rad, diag.translation_direction_error_rad
            ));
        } else {
            body.push_str("  pose: NA (RANSAC failed)\n\n");
        }
    }
    fs::write(path, body)
}
