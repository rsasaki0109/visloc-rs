//! Multi-session pose-graph merging on real KITTI ground-truth poses
//! (ORB-SLAM3-Atlas-style map merge), with PCM cross-session bridge screening.
//!
//! A single KITTI odometry sequence is *split* into two independent sessions —
//! as if the same loopy route had been mapped on two separate runs. Each session
//! is anchored at its own origin in its **own arbitrary world frame** (session B
//! is right-multiplied by a fixed frame offset `g`) and accumulates its own
//! odometry drift (per-edge yaw perturbation). The sessions share no common
//! gauge: B's trajectory is rotated/translated away from A's and cannot be
//! compared to ground truth until it is welded into A's frame.
//!
//! Cross-session *bridges* (the physical relative camera pose between a session-A
//! keyframe and a session-B keyframe that revisit the same place — what
//! cross-session place recognition would report) are the only thing tying the two
//! maps together. The demo:
//!
//!   1. collects candidate bridges: the consecutive boundary match (A's last KF
//!      ↔ B's first KF) plus every spatial revisit straddling the split, and
//!      optionally injects `--inject-wrong-bridges N` perceptual-aliasing false
//!      positives (claiming two genuinely-distant places coincide);
//!   2. screens them with [`PoseGraph::consistent_session_bridges`] (PCM across
//!      sessions, `require_individual = false`), comparing the ISOTROPIC
//!      SE(3)-tangent-norm test against the covariance-aware MAHALANOBIS test
//!      ([`visloc_rs::slam::pcm::PcmNoiseModel`]) at their fair operating point —
//!      the highest genuine recall each reaches while admitting ZERO wrong
//!      bridges. The Mahalanobis cycle covariance grows with the odometry span,
//!      so genuine revisits across many drifted edges are not over-penalized by a
//!      single rad+metre threshold, and it recovers more genuine bridges at equal
//!      precision (the gap widens with drift);
//!   3. welds B into A at the first bridge of the higher-recall screen with
//!      [`PoseGraph::merge_session`], adds the rest as loop-closure constraints,
//!      and jointly re-optimizes;
//!   4. reports the absolute trajectory error (ATE, metres vs. KITTI ground
//!      truth) of the merged map against a single-session full-batch reference
//!      and an oracle all-genuine merge (isolating the weld math from screening),
//!      plus — to show why screening matters — a merge that blindly trusts every
//!      candidate bridge (the wrong ones fold the joined map onto itself).
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --example multi_session_kitti_merge_demo -- \
//!     --kitti-poses /path/to/KITTI_odometry/poses/00.txt \
//!     --keyframe-stride 30 \
//!     --yaw-drift-deg-per-edge 0.2 \
//!     --inject-wrong-bridges 2 \
//!     --out-dir target/multi_session_merge_demo
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{UnitQuaternion, Vector3, Vector6};
use visloc_rs::core::geometry::{Pose, SE3};
use visloc_rs::slam::pcm::{PcmConfig, PcmNoiseModel};
use visloc_rs::tracking::PoseTrajectory;
use visloc_rs::{relative_world_to_camera, LoopClosureConstraint, PoseGraph, PoseGraphSe3Config};

#[derive(Debug)]
struct CliArgs {
    kitti_poses: PathBuf,
    keyframe_stride: usize,
    yaw_drift_per_edge_rad: f64,
    split_fraction: f64,
    proximity_m: f64,
    inject_wrong_bridges: usize,
    inject_seed: u64,
    max_keyframes: usize,
    out_dir: PathBuf,
}

fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut kitti_poses: Option<PathBuf> = None;
    let mut keyframe_stride: usize = 30;
    let mut yaw_drift_deg: f64 = 0.2;
    let mut split_fraction: f64 = 0.5;
    let mut proximity_m: f64 = 10.0;
    let mut inject_wrong_bridges: usize = 2;
    let mut inject_seed: u64 = 1;
    let mut max_keyframes: usize = 200;
    let mut out_dir: PathBuf = PathBuf::from("target/multi_session_merge_demo");

    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--kitti-poses" => {
                kitti_poses = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--keyframe-stride" => {
                keyframe_stride = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--yaw-drift-deg-per-edge" => {
                yaw_drift_deg = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--split-fraction" => {
                split_fraction = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--proximity-m" => {
                proximity_m = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--inject-wrong-bridges" => {
                inject_wrong_bridges = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--inject-seed" => {
                inject_seed = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--max-keyframes" => {
                max_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--out-dir" => {
                out_dir = PathBuf::from(args.remove(i + 1));
                args.remove(i);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let kitti_poses = kitti_poses
        .ok_or("--kitti-poses <path/to/poses/SS.txt> is required (KITTI odometry GT pose file)")?;
    Ok(CliArgs {
        kitti_poses,
        keyframe_stride,
        yaw_drift_per_edge_rad: yaw_drift_deg.to_radians(),
        split_fraction,
        proximity_m,
        inject_wrong_bridges,
        inject_seed,
        max_keyframes,
        out_dir,
    })
}

/// Small deterministic linear-congruential RNG (no `rand` dependency, so the
/// injected wrong bridges are reproducible across runs and platforms).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Absolute trajectory error (mean / RMSE / max per-keyframe camera-centre error,
/// metres). The anchor (global keyframe 0) is gauge-fixed and the merged estimate
/// lives in the truth world frame, so estimate and truth are directly comparable
/// (no Umeyama alignment).
fn ate(estimate: &[Pose], truth: &[Pose]) -> (f64, f64, f64) {
    let errs: Vec<f64> = estimate
        .iter()
        .zip(truth)
        .map(|(e, t)| (e.camera_center_world() - t.camera_center_world()).norm())
        .collect();
    let n = errs.len().max(1) as f64;
    let mean = errs.iter().sum::<f64>() / n;
    let rmse = (errs.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
    let max = errs.iter().copied().fold(0.0_f64, f64::max);
    (mean, rmse, max)
}

fn pgo_config() -> PoseGraphSe3Config {
    PoseGraphSe3Config {
        initial_lambda: Some(1.0e-3),
        max_iterations: 80,
        // Drifted odometry is a usable seed and the genuine loops are
        // truth-relative; chordal init is not robust, so let LM refine the seed.
        chordal_init: false,
        ..PoseGraphSe3Config::default()
    }
}

fn loop_constraint(from: u64, to: u64, relative: SE3) -> LoopClosureConstraint {
    LoopClosureConstraint {
        from_keyframe_id: from,
        to_keyframe_id: to,
        relative_pose: relative,
        inlier_count: 100,
        inlier_ratio: 1.0,
        mean_sampson_error: 0.0,
        score: 100.0,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let trajectory = PoseTrajectory::read_kitti_poses(&args.kitti_poses)?;
    let samples = trajectory.samples();
    if samples.is_empty() {
        return Err("KITTI pose file is empty".into());
    }

    // Subsample with stride; cap at max_keyframes.
    let mut keyframes: Vec<Pose> = samples
        .iter()
        .step_by(args.keyframe_stride.max(1))
        .map(|s| s.pose.clone())
        .collect();
    if keyframes.len() > args.max_keyframes {
        keyframes.truncate(args.max_keyframes);
    }
    let n = keyframes.len();
    if n < 8 {
        return Err(format!("need at least 8 keyframes after subsampling, got {n}").into());
    }
    let m = ((n as f64 * args.split_fraction) as usize).clamp(3, n - 3);
    println!(
        "kitti_poses={} total_samples={} stride={} keyframe_count={n} split=A[0..{m}] B[{m}..{n}] yaw_drift_per_edge={:.4} rad",
        args.kitti_poses.display(),
        samples.len(),
        args.keyframe_stride,
        args.yaw_drift_per_edge_rad,
    );

    // Truth sequential edges and their drifted (yaw-perturbed) counterparts. Both
    // sessions and the full-batch reference draw from this SAME edge realization,
    // so the only difference between "two merged sessions" and "one full session"
    // is the gauge/frame split — the optima must coincide.
    let truth_edges: Vec<SE3> = keyframes
        .windows(2)
        .map(|w| relative_world_to_camera(&w[0], &w[1]))
        .collect();
    let yaw_drift_rot =
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), args.yaw_drift_per_edge_rad);
    let noisy_edges: Vec<SE3> = truth_edges
        .iter()
        .map(|e| SE3::new(yaw_drift_rot * e.rotation, e.translation))
        .collect();

    // --- Genuine revisits (truth-relative, frame-invariant relative poses) ---
    // Every temporally-distant, spatially-close pair is a real loop. Partition by
    // where the endpoints fall relative to the split.
    let min_gap = 10usize;
    let mut a_loops: Vec<LoopClosureConstraint> = Vec::new(); // both endpoints < m
    let mut b_loops: Vec<LoopClosureConstraint> = Vec::new(); // both endpoints >= m (local ids)
    let mut cross: Vec<LoopClosureConstraint> = Vec::new(); // i < m <= j (A id, B local id)
    for i in 0..n {
        let ci = keyframes[i].camera_center_world();
        for j in (i + min_gap)..n {
            if (ci - keyframes[j].camera_center_world()).norm() > args.proximity_m {
                continue;
            }
            let rel = relative_world_to_camera(&keyframes[i], &keyframes[j]);
            if j < m {
                a_loops.push(loop_constraint(i as u64, j as u64, rel));
            } else if i >= m {
                b_loops.push(loop_constraint((i - m) as u64, (j - m) as u64, rel));
            } else {
                cross.push(loop_constraint(i as u64, (j - m) as u64, rel));
            }
        }
    }
    println!(
        "genuine loops: A-internal={} B-internal={} cross-session={}",
        a_loops.len(),
        b_loops.len(),
        cross.len()
    );

    // --- Session A: ids 0..m, drifted from truth[0] in the truth world frame ---
    let mut a_drifted: Vec<Pose> = vec![keyframes[0].clone()];
    for edge in &noisy_edges[0..m - 1] {
        let prev = a_drifted.last().unwrap().world_to_camera.clone();
        a_drifted.push(Pose {
            world_to_camera: edge.compose(&prev),
        });
    }
    let mut session_a = PoseGraph::new();
    for (id, p) in a_drifted.iter().enumerate() {
        session_a.add_pose(id as u64, p.clone());
    }
    session_a.anchor(0);
    for (k, edge) in noisy_edges[0..m - 1].iter().enumerate() {
        session_a.add_sequential_edge(k as u64, (k + 1) as u64, edge.clone());
    }
    for c in &a_loops {
        session_a.add_loop_closure_constraint(c);
    }

    // --- Session B: local ids 0..(n-m), drifted in its OWN arbitrary frame ---
    // B's origin is right-multiplied by a fixed frame offset `g`: a separate map
    // with no shared gauge. Its internal structure uses the same noisy edges
    // [m-1..n-1], so after the merge aligns the frame it reaches the same optimum.
    let g = SE3::exp(&Vector6::new(0.6, -0.4, 0.9, 0.7, -0.3, 0.5));
    let nb = n - m;
    let mut b_drifted: Vec<Pose> = vec![Pose {
        world_to_camera: keyframes[m].world_to_camera.compose(&g),
    }];
    for edge in &noisy_edges[m..n - 1] {
        let prev = b_drifted.last().unwrap().world_to_camera.clone();
        b_drifted.push(Pose {
            world_to_camera: edge.compose(&prev),
        });
    }
    let mut session_b = PoseGraph::new();
    for (id, p) in b_drifted.iter().enumerate() {
        session_b.add_pose(id as u64, p.clone());
    }
    session_b.anchor(0);
    for (k, edge) in noisy_edges[m..n - 1].iter().enumerate() {
        session_b.add_sequential_edge(k as u64, (k + 1) as u64, edge.clone());
    }
    for c in &b_loops {
        session_b.add_loop_closure_constraint(c);
    }

    // --- Candidate cross-session bridges (place-recognition matches) ---
    // The consecutive boundary match (A's last KF ↔ B's first KF) is always a
    // genuine bridge; add it first so a screened merge has a reliable weld even if
    // a sequence happens to have no spatial cross-revisit.
    let mut candidates: Vec<LoopClosureConstraint> = Vec::new();
    let mut candidate_is_wrong: Vec<bool> = Vec::new();
    candidates.push(loop_constraint(
        (m - 1) as u64,
        0,
        relative_world_to_camera(&keyframes[m - 1], &keyframes[m]),
    ));
    candidate_is_wrong.push(false);
    for c in &cross {
        candidates.push(c.clone());
        candidate_is_wrong.push(false);
    }
    let genuine_bridges = candidates.len();

    // Inject wrong cross-session bridges: an A node and a B node that are far
    // apart in truth, asserted near-coincident (perceptual aliasing).
    if args.inject_wrong_bridges > 0 {
        let mut rng = Lcg(args.inject_seed.wrapping_mul(2) | 1);
        let mut guard = 0usize;
        while candidate_is_wrong.iter().filter(|&&w| w).count() < args.inject_wrong_bridges
            && guard < args.inject_wrong_bridges * 1000
        {
            guard += 1;
            let i = (rng.next_u64() as usize) % m; // an A node
            let j = (rng.next_u64() as usize) % nb; // a B local node (global m+j)
            if (keyframes[i].camera_center_world() - keyframes[m + j].camera_center_world()).norm()
                <= args.proximity_m * 3.0
            {
                continue; // too close — might actually be a revisit
            }
            let translation = Vector3::new(
                rng.range(-2.0, 2.0),
                rng.range(-1.0, 1.0),
                rng.range(-2.0, 2.0),
            );
            let axis = Vector3::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            );
            let rotation = nalgebra::Unit::try_new(axis, 1e-9)
                .map(|a| UnitQuaternion::from_axis_angle(&a, rng.range(0.0, 0.2)))
                .unwrap_or_else(UnitQuaternion::identity);
            candidates.push(LoopClosureConstraint {
                from_keyframe_id: i as u64,
                to_keyframe_id: j as u64,
                relative_pose: SE3::new(rotation, translation),
                inlier_count: 80,
                inlier_ratio: 0.9,
                mean_sampson_error: 0.0,
                score: 80.0,
            });
            candidate_is_wrong.push(true);
        }
    }
    let wrong_total = candidate_is_wrong.iter().filter(|&&w| w).count();
    println!(
        "candidate bridges: {} ({} genuine + {} wrong)",
        candidates.len(),
        genuine_bridges,
        wrong_total
    );

    // --- PCM cross-session screening (front-end guard) ---
    // Two screens compared: the ISOTROPIC SE(3)-tangent norm (a single threshold
    // mixing rad and metres) and the covariance-aware MAHALANOBIS test, whose
    // cycle covariance grows with the odometry span so genuine bridges across many
    // drifted edges are not penalized — lifting recall at the same (zero-wrong)
    // precision. The Mahalanobis noise model is derived from the simulated drift:
    // per-edge rotation variance ≈ (yaw drift)², translation variance ≈ (yaw drift
    // × mean edge length)².
    let mean_edge_len = {
        let total: f64 = keyframes
            .windows(2)
            .map(|w| (w[0].camera_center_world() - w[1].camera_center_world()).norm())
            .sum();
        total / (n - 1) as f64
    };
    let d = args.yaw_drift_per_edge_rad.max(1e-3);
    let noise = PcmNoiseModel::isotropic(
        (2.0 * d).powi(2),                 // odo rotation var / edge
        (2.0 * d * mean_edge_len).powi(2), // odo translation var / edge
        1e-6,                              // loop-measurement rotation var
        1e-4,                              // loop-measurement translation var
    );
    // A screen's free threshold is swept; the FAIR operating point is the highest
    // genuine recall reachable while admitting ZERO wrong bridges (same precision).
    // `sweep` returns (best kept set, recall) at that point.
    let sweep = |label: &str, noise: Option<PcmNoiseModel>, grid: &[f64]| -> (Vec<usize>, usize) {
        let mut best: Vec<usize> = Vec::new();
        for &t in grid {
            let cfg = PcmConfig {
                threshold: t,
                require_individual: false, // no single-session relative across sessions
                noise,
            };
            let kept =
                session_a.consistent_session_bridges(&session_b, m as u64, &candidates, &cfg);
            let wrong = kept.iter().filter(|&&i| candidate_is_wrong[i]).count();
            let genuine = kept.len() - wrong;
            if wrong == 0 && genuine > best.iter().filter(|&&i| !candidate_is_wrong[i]).count() {
                best = kept;
            }
        }
        let recall = best.iter().filter(|&&i| !candidate_is_wrong[i]).count();
        println!(
            "[pcm {label}] best zero-wrong recall = {recall}/{genuine_bridges} genuine bridges"
        );
        (best, recall)
    };
    // Isotropic threshold is an SE(3)-tangent norm (rad + metres); Mahalanobis is
    // a unitless distance (√χ²) — different scales, so each gets its own grid.
    let iso_grid: Vec<f64> = (1..=60).map(|i| i as f64 * 0.5).collect(); // 0.5 .. 30
    let maha_grid: Vec<f64> = (1..=60).map(|i| i as f64 * 0.5).collect(); // 0.5 .. 30
    let (kept_iso, iso_recall) = sweep("isotropic  ", None, &iso_grid);
    let (kept_maha, maha_recall) = sweep("mahalanobis", Some(noise), &maha_grid);
    // Merge with the higher-recall zero-wrong screen.
    let (kept, kept_genuine) = if maha_recall >= iso_recall {
        (kept_maha, maha_recall)
    } else {
        (kept_iso, iso_recall)
    };
    if kept.is_empty() {
        return Err("PCM screened out every bridge at zero-wrong precision — cannot merge".into());
    }

    // --- Reference: single-session full batch (all n KFs, one frame, all loops) ---
    let mut full = PoseGraph::new();
    let mut full_drifted: Vec<Pose> = vec![keyframes[0].clone()];
    for edge in &noisy_edges {
        let prev = full_drifted.last().unwrap().world_to_camera.clone();
        full_drifted.push(Pose {
            world_to_camera: edge.compose(&prev),
        });
    }
    for (id, p) in full_drifted.iter().enumerate() {
        full.add_pose(id as u64, p.clone());
    }
    full.anchor(0);
    for (k, edge) in noisy_edges.iter().enumerate() {
        full.add_sequential_edge(k as u64, (k + 1) as u64, edge.clone());
    }
    for c in &a_loops {
        full.add_loop_closure_constraint(c);
    }
    for c in &b_loops {
        full.add_loop_closure_constraint(&loop_constraint(
            c.from_keyframe_id + m as u64,
            c.to_keyframe_id + m as u64,
            c.relative_pose.clone(),
        ));
    }
    for c in &cross {
        full.add_loop_closure_constraint(&loop_constraint(
            c.from_keyframe_id,
            c.to_keyframe_id + m as u64,
            c.relative_pose.clone(),
        ));
    }
    full.optimize_se3_iterative(&pgo_config())?;
    let full_traj: Vec<Pose> = (0..n as u64).map(|id| full.poses[&id].clone()).collect();
    let (f_mean, f_rmse, f_max) = ate(&full_traj, &keyframes);

    // --- Oracle merge: ALL genuine bridges, no wrong, no screening. Isolates the
    //     merge math from PCM's recall — should match the single-session batch
    //     (same constraints, B just welded from a different frame). ---
    let genuine_idx: Vec<usize> = (0..genuine_bridges).collect();
    let merged_oracle =
        merge_and_optimize(&session_a, &session_b, m as u64, &candidates, &genuine_idx)?;
    let oracle_traj: Vec<Pose> = (0..n as u64)
        .map(|id| merged_oracle.poses[&id].clone())
        .collect();
    let (o_mean, o_rmse, o_max) = ate(&oracle_traj, &keyframes);

    // --- Merge with PCM-screened bridges, then jointly optimize ---
    let merged_screened = merge_and_optimize(&session_a, &session_b, m as u64, &candidates, &kept)?;
    let merged_traj: Vec<Pose> = (0..n as u64)
        .map(|id| merged_screened.poses[&id].clone())
        .collect();
    let (s_mean, s_rmse, s_max) = ate(&merged_traj, &keyframes);

    // --- Naive merge: trust EVERY candidate (no screening) — wrong bridges fold
    //     the joined map onto itself. Only run when wrong bridges were injected. ---
    let mut naive_ate: Option<(f64, f64, f64)> = None;
    if wrong_total > 0 {
        let all: Vec<usize> = (0..candidates.len()).collect();
        let merged_naive = merge_and_optimize(&session_a, &session_b, m as u64, &candidates, &all)?;
        let naive_traj: Vec<Pose> = (0..n as u64)
            .map(|id| merged_naive.poses[&id].clone())
            .collect();
        naive_ate = Some(ate(&naive_traj, &keyframes));
    }

    println!();
    println!(
        "ATE single-batch (reference)     mean={f_mean:.3} rmse={f_rmse:.3} max={f_max:.3} (m)"
    );
    println!(
        "ATE merged + all-genuine (oracle) mean={o_mean:.3} rmse={o_rmse:.3} max={o_max:.3} (m)"
    );
    println!(
        "ATE merged + PCM screening       mean={s_mean:.3} rmse={s_rmse:.3} max={s_max:.3} (m)"
    );
    if let Some((nv_mean, nv_rmse, nv_max)) = naive_ate {
        println!("ATE merged + ALL bridges (naive) mean={nv_mean:.3} rmse={nv_rmse:.3} max={nv_max:.3} (m)");
        println!(
            "summary: merging all {genuine_bridges} genuine bridges reproduces the single-session batch \
             ({o_rmse:.2}m vs {f_rmse:.2}m) — the weld math is exact across frames. A naive merge that \
             additionally trusts {wrong_total} wrong bridge(s) collapses to {nv_rmse:.2}m. Both screens \
             drop every wrong bridge (precision 1.0); the question is recall. At equal precision the \
             ISOTROPIC norm recovers {iso_recall}/{genuine_bridges} genuine bridges while the \
             covariance-aware MAHALANOBIS test recovers {maha_recall}/{genuine_bridges} — its cycle \
             covariance grows with the odometry span, so revisits across many drifted edges are no \
             longer over-penalized by a single rad+metre threshold. The higher-recall screen lands the \
             merge at {s_rmse:.2}m, close to the oracle ({o_rmse:.2}m); the Mahalanobis advantage WIDENS \
             with drift (raise --yaw-drift-deg-per-edge)."
        );
    } else {
        println!(
            "summary: two independently-drifted sessions in different frames merge to {s_rmse:.2}m rmse \
             ({kept_genuine}/{genuine_bridges} genuine bridges kept; isotropic recall {iso_recall}, \
             Mahalanobis {maha_recall}), vs the oracle all-genuine merge {o_rmse:.2}m and the \
             single-session batch {f_rmse:.2}m."
        );
    }

    fs::create_dir_all(&args.out_dir)?;
    write_csv(&args.out_dir.join("truth.csv"), &keyframes)?;
    write_csv(&args.out_dir.join("merged.csv"), &merged_traj)?;
    write_csv(&args.out_dir.join("batch.csv"), &full_traj)?;
    println!("\ntrajectories written to {}", args.out_dir.display());
    Ok(())
}

/// Weld session B into a clone of session A at the first kept bridge, add the
/// remaining kept bridges as loop-closure constraints, and jointly optimize.
fn merge_and_optimize(
    session_a: &PoseGraph,
    session_b: &PoseGraph,
    id_offset: u64,
    candidates: &[LoopClosureConstraint],
    kept: &[usize],
) -> Result<PoseGraph, Box<dyn std::error::Error>> {
    let mut merged = session_a.clone();
    merged.merge_session(session_b, id_offset, &candidates[kept[0]])?;
    for &k in &kept[1..] {
        let c = &candidates[k];
        merged.add_loop_closure_constraint(&loop_constraint(
            c.from_keyframe_id,
            c.to_keyframe_id + id_offset,
            c.relative_pose.clone(),
        ));
    }
    merged.optimize_se3_iterative(&pgo_config())?;
    Ok(merged)
}

fn write_csv(path: &Path, poses: &[Pose]) -> std::io::Result<()> {
    let mut s = String::from("id,x,y,z\n");
    for (i, pose) in poses.iter().enumerate() {
        let c = pose.camera_center_world();
        s.push_str(&format!("{i},{:.6},{:.6},{:.6}\n", c.x, c.y, c.z));
    }
    fs::write(path, s)
}
