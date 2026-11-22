//! The simulated robot localized in its map, closed loop: it steers on the
//! pose it believes (`map -> odom` and its wheel odometry).

use visloc_core::types::LandmarkDescriptorStore;
use visloc_navigation::sim::{
    apartment_mapping_route, build_map, waypoint_command, CameraRig, MappingConfig, Pose2, World,
};
use visloc_navigation::sim_robot::{LocalizationMode, SimRobot, SimRobotConfig};

struct Drive {
    reached: usize,
    collisions: usize,
    max_error: f64,
    localized_at_s: Option<f64>,
}

/// Living room, through the doorway into the kitchen, and back.
fn route() -> Vec<Pose2> {
    [
        (3.0, 1.7),
        (5.0, 1.8),
        (6.6, 2.0),
        (7.5, 3.0),
        (8.0, 1.0),
        (6.6, 2.0),
        (5.0, 1.8),
        (3.0, 2.0),
    ]
    .iter()
    .map(|&(x, y)| Pose2::new(x, y, 0.0))
    .collect()
}

fn drive(mode: LocalizationMode) -> Drive {
    let world = World::apartment(1);
    let rig = CameraRig::forward(0.5);
    let run = build_map(
        &world,
        &rig,
        &apartment_mapping_route(),
        &MappingConfig::default(),
    );
    let descriptors = LandmarkDescriptorStore::from_visual_map(&run.map);
    let config = SimRobotConfig {
        mode,
        ..SimRobotConfig::default()
    };
    let start = Pose2::new(1.5, 2.5, 0.0);
    let mut robot = SimRobot::new(world, rig, run.map, descriptors, start, config);
    let route = route();
    let (mut next, mut reached, mut max_error) = (0, 0, 0.0f64);
    let mut localized_at_s = None;
    let mut output = robot.step(0.0, 0.0, 0.05);
    while robot.time_s() < 120.0 {
        let Some(believed) = output.estimate() else {
            output = robot.step(0.0, 0.0, 0.05);
            continue;
        };
        localized_at_s.get_or_insert(output.time_s);
        let error = (believed.x - output.ground_truth.x).hypot(believed.y - output.ground_truth.y);
        max_error = max_error.max(error);
        let before = next;
        let (v, w, done) = waypoint_command(&believed, &route, &mut next, &robot.robot.config);
        if next > before {
            let (target, truth) = (route[before], output.ground_truth);
            reached += usize::from((truth.x - target.x).hypot(truth.y - target.y) < 0.3);
        }
        if done {
            break;
        }
        output = robot.step(v, w, 0.05);
    }
    Drive {
        reached,
        collisions: robot.robot.collisions,
        max_error,
        localized_at_s,
    }
}

#[test]
fn visloc_localization_keeps_the_robot_on_its_route() {
    let drive = drive(LocalizationMode::Visloc);
    let localized = drive.localized_at_s.expect("localized");
    assert!(localized < 2.0, "localized after {localized} s");
    assert_eq!(drive.reached, route().len());
    assert_eq!(drive.collisions, 0);
    assert!(drive.max_error < 0.15, "max error {}", drive.max_error);
}

#[test]
fn ground_truth_mode_has_no_error() {
    let drive = drive(LocalizationMode::GroundTruth);
    assert_eq!(drive.reached, route().len());
    assert!(drive.max_error < 1e-9);
}
