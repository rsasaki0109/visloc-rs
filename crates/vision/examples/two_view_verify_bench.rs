//! Time COLMAP-style two-view verification (`TwoViewGeometryVerifier`) and
//! its E / F / H RANSACs on synthetic pairs, and print a digest of every
//! report (configuration, inliers, model bits) to check that a speed-up
//! leaves the results unchanged.
//!
//! cargo run --release -p visloc-vision --example two_view_verify_bench -- \
//!     [correspondences 400] [outlier % 40] [pairs 50]
use nalgebra::{Point2, Rotation3, Vector3};
use std::time::Instant;
use visloc_core::types::Camera;
use visloc_vision::two_view::{
    fundamental_ransac, homography_ransac, EssentialRansac, EssentialRansacConfig,
    FivePointEssentialMatrixEstimator, FundamentalRansacConfig, HomographyRansacConfig,
    TwoViewCorrespondence, TwoViewGeometryOptions, TwoViewGeometryVerifier,
};
fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (n, outlier_pct, pairs) = (
        args.first().copied().unwrap_or(400),
        args.get(1).copied().unwrap_or(40),
        args.get(2).copied().unwrap_or(50),
    );
    let cam = Camera::pinhole(1, 1024, 690, 487.0, 487.0, 512.0, 345.0);
    let mut s = 0x1234_5678u64;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut sets = Vec::new();
    for _ in 0..pairs {
        let r = Rotation3::from_euler_angles(0.05 * rnd(), 0.2 * (rnd() - 0.5), 0.05 * rnd());
        let t = Vector3::new(rnd() - 0.5, 0.1 * (rnd() - 0.5), rnd() - 0.5).normalize() * 0.5;
        let mut c = Vec::new();
        while c.len() < n {
            let p = Vector3::new(8.0 * (rnd() - 0.5), 5.0 * (rnd() - 0.5), 4.0 + 10.0 * rnd());
            let q = r * p + t;
            if q.z < 0.5 {
                continue;
            }
            let n1 = (rnd() - 0.5, rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
            let a = Point2::new(
                487.0 * p.x / p.z + 512.0 + 0.5 * n1.0,
                487.0 * p.y / p.z + 345.0 + 0.5 * n1.1,
            );
            let mut b = Point2::new(
                487.0 * q.x / q.z + 512.0 + 0.5 * n1.2,
                487.0 * q.y / q.z + 345.0 + 0.5 * n1.3,
            );
            if rnd() * 100.0 < outlier_pct as f64 {
                b = Point2::new(1024.0 * rnd(), 690.0 * rnd());
            }
            c.push(TwoViewCorrespondence::new(a, b));
        }
        sets.push(c);
    }
    let opts = TwoViewGeometryOptions::for_camera(&cam, 4.0);
    let v = TwoViewGeometryVerifier::new(opts);
    let t = Instant::now();
    let mut inl = 0;
    for c in &sets {
        inl += v.classify(c, &cam).inliers.len();
    }
    let tc = t.elapsed().as_secs_f64() / pairs as f64;
    let er = EssentialRansac {
        estimator: FivePointEssentialMatrixEstimator::default(),
        config: EssentialRansacConfig {
            iterations: opts.ransac_iterations,
            sampson_threshold: opts.essential_sampson_threshold,
            seed: opts.seed,
        },
    };
    let t = Instant::now();
    for c in &sets {
        er.estimate(c, &cam);
    }
    let te = t.elapsed().as_secs_f64() / pairs as f64;
    let t = Instant::now();
    for c in &sets {
        fundamental_ransac(
            c,
            &FundamentalRansacConfig {
                iterations: opts.ransac_iterations,
                max_error_px: 4.0,
                seed: 7,
            },
        );
    }
    let tf = t.elapsed().as_secs_f64() / pairs as f64;
    let t = Instant::now();
    for c in &sets {
        homography_ransac(
            c,
            &HomographyRansacConfig {
                iterations: opts.ransac_iterations,
                max_error_px: 4.0,
                seed: 7,
            },
        );
    }
    let th = t.elapsed().as_secs_f64() / pairs as f64;
    // Digest of every report: config, inliers, E/F/H bits.
    let mut h = 0xcbf29ce484222325u64;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    };
    for c in &sets {
        let r = v.classify(c, &cam);
        eat(r.config as u64);
        for &i in &r.inliers {
            eat(i as u64);
        }
        for m in [r.essential, r.fundamental, r.homography]
            .into_iter()
            .flatten()
        {
            for x in m.iter() {
                eat(x.to_bits());
            }
        }
    }
    println!("digest {h:016x}");
    println!("n={n} outliers={outlier_pct}%: classify {:.2} ms/pair (E {:.2}, F {:.2}, H {:.2}), mean inliers {}", 1e3*tc, 1e3*te, 1e3*tf, 1e3*th, inl / pairs);
}
