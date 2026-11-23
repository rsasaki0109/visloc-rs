//! Closed-loop drive in the simulated apartment without ROS: the robot
//! follows a patrol route steering on the pose it *believes* (`map -> odom`
//! composed with its wheel odometry), and the same drive is run with each
//! source of `map -> odom`:
//!
//! - `ground_truth`: perfect localization (the upper bound);
//! - `odometry`: the start pose, then wheel odometry alone (drifts);
//! - `visloc`: map-based localization in the map of `nav_sim_build_map`
//!   (built in-process here), correcting the wheel odometry.
//!
//! ```text
//! cargo run --release --example nav_sim_drive -- [--laps 2] [--out runs/nav_sim/drive] \
//!     [--p3p-solver grunert|lambda-twist] [--inlier-refits N]
//! ```
//!
//! Prints per mode: waypoints reached (true pose within 0.3 m when the robot
//! believes it is there), collisions, and the believed-pose error. With
//! `--out`, writes `<mode>.csv` (time, true and believed pose) per mode.
//! Navigation proper (planning around obstacles) is Nav2's job; see
//! `docs/navigation.md` for the same robot driven by Nav2 over ROS 2.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use visloc_rs::navigation::sim::{
    apartment_mapping_route, build_map, waypoint_command, CameraRig, MappingConfig, Pose2, World,
};
use visloc_rs::navigation::sim_robot::{LocalizationMode, SimRobot, SimRobotConfig};
use visloc_rs::types::LandmarkDescriptorStore;
use visloc_rs::vision::pnp::P3PSolver;

/// A patrol through every room of the apartment, from and back to the
/// start: living room, bedroom 1, bedroom 2, kitchen, living room.
fn patrol() -> Vec<Pose2> {
    [
        (2.5, 3.5),
        (2.5, 4.5),
        (1.5, 4.5),
        (1.5, 5.6),
        (3.0, 6.0),
        (1.5, 5.6),
        (1.5, 4.5),
        (8.5, 4.5),
        (8.5, 6.0),
        (8.5, 4.5),
        (7.5, 4.5),
        (7.5, 3.0),
        (8.0, 1.0),
        (6.6, 2.0),
        (5.0, 1.8),
        (3.0, 1.7),
        (1.5, 2.5),
    ]
    .iter()
    .map(|&(x, y)| Pose2::new(x, y, 0.0))
    .collect()
}

struct ModeResult {
    reached: usize,
    waypoints: usize,
    collisions: usize,
    rmse: f64,
    max_error: f64,
    time_s: f64,
    csv: String,
    attempts: usize,
    accepted: usize,
    latency_ms: f64,
}

fn drive(
    mode: LocalizationMode,
    laps: usize,
    p3p_solver: P3PSolver,
    inlier_refits: usize,
) -> ModeResult {
    let world = World::apartment(1);
    let rig = CameraRig::forward(0.5);
    let run = build_map(
        &world,
        &rig,
        &apartment_mapping_route(),
        &MappingConfig::default(),
    );
    let descriptors = LandmarkDescriptorStore::from_visual_map(&run.map);
    let start = Pose2::new(1.5, 2.5, 0.0);
    let config = SimRobotConfig {
        mode,
        p3p_solver,
        inlier_refits,
        ..SimRobotConfig::default()
    };
    let mut robot = SimRobot::new(world, rig, run.map, descriptors, start, config);
    let route: Vec<Pose2> = (0..laps).flat_map(|_| patrol()).collect();
    let dt = 0.05;
    let mut next = 0;
    let (mut reached, mut sum_sq, mut samples, mut max_error) = (0, 0.0, 0, 0.0f64);
    let mut csv = String::from("time_s,true_x,true_y,true_yaw,est_x,est_y,est_yaw\n");
    let mut output = robot.step(0.0, 0.0, dt);
    while robot.time_s() < 900.0 {
        let Some(believed) = output.estimate() else {
            output = robot.step(0.0, 0.0, dt);
            continue;
        };
        let before = next;
        let (v, w, done) = waypoint_command(&believed, &route, &mut next, &robot.robot.config);
        if next > before {
            let target = route[before];
            let truth = output.ground_truth;
            if (truth.x - target.x).hypot(truth.y - target.y) < 0.3 {
                reached += 1;
            }
        }
        if done {
            break;
        }
        output = robot.step(v, w, dt);
        if let Some(estimate) = output.estimate() {
            let truth = output.ground_truth;
            let error = (estimate.x - truth.x).hypot(estimate.y - truth.y);
            sum_sq += error * error;
            samples += 1;
            max_error = max_error.max(error);
            writeln!(
                csv,
                "{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                output.time_s, truth.x, truth.y, truth.yaw, estimate.x, estimate.y, estimate.yaw
            )
            .expect("string write");
        }
    }
    let counts = robot.counts;
    ModeResult {
        reached,
        waypoints: route.len(),
        collisions: robot.robot.collisions,
        rmse: (sum_sq / samples.max(1) as f64).sqrt(),
        max_error,
        time_s: robot.time_s(),
        csv,
        attempts: counts.attempts,
        accepted: counts.accepted,
        latency_ms: 1e3 * counts.latency_sum_s / counts.attempts.max(1) as f64,
    }
}

fn run() -> Result<(), String> {
    let mut laps = 1;
    let mut out: Option<PathBuf> = None;
    let mut p3p_solver = P3PSolver::default();
    let mut inlier_refits = 0;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--laps" => laps = value.parse().map_err(|_| format!("bad --laps {value}"))?,
            "--out" => out = Some(PathBuf::from(value)),
            "--inlier-refits" => {
                inlier_refits = value
                    .parse()
                    .map_err(|_| format!("bad --inlier-refits {value}"))?;
            }
            "--p3p-solver" => {
                p3p_solver = value
                    .parse()
                    .map_err(|error| format!("--p3p-solver: {error}"))?;
            }
            other => {
                return Err(format!(
                    "unknown argument {other}; usage: [--laps N] [--out DIR] \
                     [--p3p-solver grunert|lambda-twist] [--inlier-refits N]"
                ))
            }
        }
    }
    println!(
        "{:<13} {:>9} {:>10} {:>9} {:>9} {:>7} {:>16} {:>11}",
        "mode",
        "reached",
        "collisions",
        "rmse_m",
        "max_m",
        "time_s",
        "fixes_accepted",
        "latency_ms"
    );
    for (name, mode) in [
        ("ground_truth", LocalizationMode::GroundTruth),
        ("odometry", LocalizationMode::Odometry),
        ("visloc", LocalizationMode::Visloc),
    ] {
        let r = drive(mode, laps, p3p_solver, inlier_refits);
        let fixes = if mode == LocalizationMode::Visloc {
            format!("{}/{}", r.accepted, r.attempts)
        } else {
            "-".into()
        };
        let latency = if r.attempts > 0 {
            format!("{:.1}", r.latency_ms)
        } else {
            "-".into()
        };
        println!(
            "{name:<13} {:>9} {:>10} {:>9.3} {:>9.3} {:>7.1} {fixes:>16} {latency:>11}",
            format!("{}/{}", r.reached, r.waypoints),
            r.collisions,
            r.rmse,
            r.max_error,
            r.time_s
        );
        if let Some(dir) = &out {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            std::fs::write(dir.join(format!("{name}.csv")), &r.csv)
                .map_err(|e| format!("writing {name}.csv: {e}"))?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
