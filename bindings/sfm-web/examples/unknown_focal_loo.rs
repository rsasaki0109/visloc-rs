//! Leave-one-out check of localizing a photo whose focal length is unknown.
//!
//! For each photo: build the map from the other photos (known focal length),
//! match the held-out photo's SIFT descriptors to the map points, then
//! localize it three ways and compare its camera centre with the full model's:
//!
//! - `prior`: P3P RANSAC with the focal length a photo without EXIF gets
//!   (1.2 x the long side, the page's default);
//! - `p4pf`: P4Pf RANSAC, focal length estimated;
//! - `oracle`: P3P RANSAC with the true focal length.
//!
//! ```text
//! cargo run --release --example unknown_focal_loo -- <folder> --focal F [--max-size 800]
//! ```

use nalgebra::{Matrix3, Point2, Point3, Vector3};
use visloc_core::types::Camera;
use visloc_sfm_web::{candidate_pairs, decode, extract, reconstruct, sift_config, verify_matches};
use visloc_vision::matching::BruteForceMatcher;
use visloc_vision::pnp::{
    estimate_pose_with_focal, Correspondence2D3D, GaussNewtonPoseRefiner, P3PGrunert,
    P4PfRansacConfig,
};
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};

const THRESHOLD_PX: f64 = 4.0;
const MIN_INLIERS: usize = 30;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: unknown_focal_loo <folder> --focal F")?;
    let (mut max_size, mut focal) = (800u32, 0.0f64);
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--max-size" => max_size = next()?.parse()?,
            "--focal" => focal = next()?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if focal <= 0.0 {
        return Err("--focal (the true focal length at the working size) is required".into());
    }
    let mut files: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
        })
        .collect();
    files.sort();
    let cfg = sift_config(2000);
    let mut photos = Vec::new();
    for f in &files {
        let img = decode(&std::fs::read(f)?, max_size)?;
        photos.push(extract(
            &f.file_name().unwrap().to_string_lossy(),
            &img,
            &cfg,
        )?);
    }
    let (w, h) = (photos[0].width, photos[0].height);
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let camera = Camera::pinhole(1, w, h, focal, focal, cx, cy);
    let prior_focal = 1.2 * w.max(h) as f64;
    let prior_camera = Camera::pinhole(1, w, h, prior_focal, prior_focal, cx, cy);

    let pairs: Vec<_> = candidate_pairs(photos.len(), 8, 40)
        .into_iter()
        .filter_map(|(i, j)| {
            let (a, b) = (&photos[i].features, &photos[j].features);
            let dm = BruteForceMatcher { ratio: Some(0.8) }
                .match_descriptors_cross_checked(&a.descriptors, &b.descriptors);
            verify_matches(&camera, a, b, &dm, 15).map(|m| (i, j, m))
        })
        .collect();
    let full = reconstruct(&camera, &photos, &pairs)?;
    let full_centre = |i: usize| {
        full.cameras
            .iter()
            .find(|c| c.index == i)
            .map(|c| Vector3::from(c.centre))
    };
    println!(
        "{} photos, full model {} registered; true focal {focal:.0} px, no-EXIF prior {prior_focal:.0} px",
        photos.len(),
        full.cameras.len()
    );
    println!("photo        corrs | prior: inl  err | p4pf: inl  focal   err | oracle: inl  err   (err: % of scene radius)");

    let mut tally = [(0usize, Vec::new()), (0, Vec::new()), (0, Vec::new())];
    let mut focal_errors = Vec::new();
    for k in 0..photos.len() {
        let Some(truth) = full_centre(k) else {
            continue;
        };
        let loo_pairs: Vec<_> = pairs
            .iter()
            .filter(|(i, j, _)| *i != k && *j != k)
            .cloned()
            .collect();
        let Ok(map) = reconstruct(&camera, &photos, &loo_pairs) else {
            continue;
        };
        // Similarity from the held-out map's frame to the full model's, on the
        // photos both registered.
        let common: Vec<(Vector3<f64>, Vector3<f64>)> = map
            .cameras
            .iter()
            .filter_map(|c| full_centre(c.index).map(|f| (Vector3::from(c.centre), f)))
            .collect();
        let Some((scale, rotation, translation)) = umeyama(&common) else {
            continue;
        };
        let centroid = common.iter().map(|(_, f)| f).sum::<Vector3<f64>>() / common.len() as f64;
        let radius = common
            .iter()
            .map(|(_, f)| (f - centroid).norm())
            .sum::<f64>()
            / common.len() as f64;

        // Map points' descriptors: the first observation's.
        let point_descriptors: Vec<Vec<f32>> = map
            .points
            .iter()
            .map(|p| {
                let (photo, kp) = p.track[0];
                photos[photo].features.descriptors[kp].clone()
            })
            .collect();
        let query = &photos[k].features;
        let matches = BruteForceMatcher { ratio: Some(0.8) }
            .match_descriptors_cross_checked(&query.descriptors, &point_descriptors);
        let corrs: Vec<Correspondence2D3D> = matches
            .iter()
            .map(|m| Correspondence2D3D {
                point2d: query.keypoints[m.query_index],
                point3d: Point3::from(map.points[m.train_index].xyz),
                confidence: None,
            })
            .collect();

        let to_full = |c: Vector3<f64>| scale * (rotation * c) + translation;
        let err = |centre: Vector3<f64>| 100.0 * (to_full(centre) - truth).norm() / radius;
        let p3p = |cam: &Camera| {
            PnPRansac {
                pose_estimator: P3PGrunert,
                pose_refiner: Some(GaussNewtonPoseRefiner::default()),
                iterations: 1000,
                reprojection_threshold: THRESHOLD_PX,
                seed: 7,
                early_stop_min_iterations: 0,
                early_stop_inlier_ratio: None,
                confidence: None,
            }
            .estimate(&corrs, cam)
            .filter(|r| r.inliers.len() >= MIN_INLIERS)
            .map(|r| (r.inliers.len(), err(r.pose.camera_center_world().coords)))
        };
        let prior = p3p(&prior_camera);
        let oracle = p3p(&camera);
        let p4pf = estimate_pose_with_focal(
            &corrs,
            Point2::new(cx, cy),
            &P4PfRansacConfig {
                max_error_px: THRESHOLD_PX,
                min_inliers: MIN_INLIERS,
                ..P4PfRansacConfig::default()
            },
        )
        .map(|r| {
            (
                r.inliers.len(),
                r.estimate.focal_px,
                err(r.pose().camera_center_world().coords),
            )
        });

        let show = |r: Option<(usize, f64)>| {
            r.map_or("   -      -".to_string(), |(n, e)| format!("{n:4} {e:6.2}"))
        };
        println!(
            "{:12} {:5} | {} | {} | {}",
            photos[k].name,
            corrs.len(),
            show(prior),
            p4pf.map_or("   -        -      -".to_string(), |(n, f, e)| format!(
                "{n:4} {f:7.1} {e:6.2}"
            )),
            show(oracle)
        );
        for (slot, r) in [prior, p4pf.map(|(n, _, e)| (n, e)), oracle]
            .into_iter()
            .enumerate()
        {
            if let Some((_, e)) = r {
                tally[slot].0 += 1;
                tally[slot].1.push(e);
            }
        }
        if let Some((_, f, _)) = p4pf {
            focal_errors.push(100.0 * (f - focal).abs() / focal);
        }
    }
    let median = |v: &mut Vec<f64>| {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    for (name, (n, errs)) in ["prior", "p4pf", "oracle"].iter().zip(tally.iter_mut()) {
        println!(
            "{name:6}: localized {n}, median centre error {:.2}% of scene radius",
            median(errs)
        );
    }
    println!(
        "p4pf focal: median error {:.2}% of the true focal length",
        median(&mut focal_errors)
    );
    Ok(())
}

/// Similarity `(s, R, t)` with `full ~ s R loo + t`, least squares (Umeyama).
fn umeyama(pairs: &[(Vector3<f64>, Vector3<f64>)]) -> Option<(f64, Matrix3<f64>, Vector3<f64>)> {
    if pairs.len() < 3 {
        return None;
    }
    let n = pairs.len() as f64;
    let mu_a = pairs.iter().map(|(a, _)| a).sum::<Vector3<f64>>() / n;
    let mu_b = pairs.iter().map(|(_, b)| b).sum::<Vector3<f64>>() / n;
    let mut cov = Matrix3::zeros();
    let mut var_a = 0.0;
    for (a, b) in pairs {
        cov += (b - mu_b) * (a - mu_a).transpose();
        var_a += (a - mu_a).norm_squared();
    }
    cov /= n;
    var_a /= n;
    let svd = cov.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut d = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let rotation = u * d * v_t;
    let scale = (svd.singular_values.component_mul(&d.diagonal())).sum() / var_a;
    let translation = mu_b - scale * rotation * mu_a;
    Some((scale, rotation, translation))
}
