use std::path::PathBuf;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_basalt::imu::{BiasRandomWalkNoise, ImuNoiseModel};
use visloc_basalt::vio::{
    triangulate_two_rays, visual_reprojection_factor, FactorConfig, LmProblem, ScalarMode,
    WindowLandmark, WindowObservation, WindowProblem, WindowState,
};
use visloc_basalt::{BasaltCalibration, BasaltNavState};
use visloc_core::geometry::SE3;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("euroc_ds_calib_minimal.json")
}

fn fixture() -> BasaltCalibration {
    BasaltCalibration::from_path(fixture_path()).unwrap()
}

fn state(pose: SE3) -> WindowState {
    WindowState {
        frame_id: 0,
        timestamp_ns: 1,
        nav: BasaltNavState {
            imu_to_world: pose.clone(),
            ..BasaltNavState::default()
        },
        stored_current_nav: BasaltNavState {
            imu_to_world: pose.clone(),
            ..BasaltNavState::default()
        },
        linearized_nav: BasaltNavState {
            imu_to_world: pose,
            ..BasaltNavState::default()
        },
        linearized_delta: nalgebra::DVector::zeros(15),
        is_keyframe: true,
        is_latest: true,
        linearized: true,
    }
}

#[test]
fn random_3d_projection_matches_per_camera_golden_values() {
    let calibration = fixture();
    let points = [
        Point3::new(0.13, -0.07, 1.7),
        Point3::new(-0.31, 0.22, 2.8),
        Point3::new(0.47, 0.11, 3.4),
        Point3::new(-0.18, -0.29, 1.25),
    ];
    let expected = [
        [
            (388.26336286531335, 237.07482349456748),
            (336.8525786765665, 269.85872608893726),
            (405.1029418916721, 257.2161357519971),
            (328.4427432889461, 186.09353931489332),
        ],
        [
            (400.83630187534925, 244.0545421011534),
            (349.9410961685638, 276.502680859523),
            (417.506935967275, 263.9895784161992),
            (341.6159836146497, 193.59605740941714),
        ],
    ];
    for camera_id in 0..2 {
        let camera = calibration.camera(camera_id as u16).unwrap();
        for (index, point) in points.iter().enumerate() {
            let pixel = camera.project(point).unwrap();
            assert!((pixel.x - expected[camera_id][index].0).abs() < 1e-10);
            assert!((pixel.y - expected[camera_id][index].1).abs() < 1e-10);
        }
    }
}

#[test]
fn cam0_and_cam1_visual_residuals_are_zero_with_rig_extrinsics() {
    let calibration = fixture();
    let t_w_i = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.07, -0.11, 0.13)),
        Vector3::new(0.4, -0.2, 0.3),
    );
    let t_w_c0 = t_w_i.compose(calibration.camera_to_imu(0).unwrap());
    let t_w_c1 = t_w_i.compose(calibration.camera_to_imu(1).unwrap());
    let point_world = t_w_c0.transform_point(&Point3::new(0.18, -0.11, 2.7));
    let pixel0 = calibration
        .camera(0)
        .unwrap()
        .project(&t_w_c0.inverse().transform_point(&point_world))
        .unwrap();
    let pixel1 = calibration
        .camera(1)
        .unwrap()
        .project(&t_w_c1.inverse().transform_point(&point_world))
        .unwrap();
    let point_anchor = t_w_c0.inverse().transform_point(&point_world);
    let anchor_distance = point_anchor.coords.norm();

    let problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera: *calibration.camera(0).unwrap(),
        cameras: calibration.cameras.clone(),
        t_imu_cam: calibration.t_imu_cam.clone(),
        poses: Vec::new(),
        states: vec![state(t_w_i.clone())],
        landmarks: vec![WindowLandmark {
            track_id: 7,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction: visloc_basalt::vio::StereographicDirection::from_bearing(
                point_anchor.coords,
            )
            .unwrap(),
            inverse_distance: 1.0 / anchor_distance,
            observations: vec![
                WindowObservation {
                    state_index: 0,
                    camera_id: 0,
                    pixel: pixel0,
                },
                WindowObservation {
                    state_index: 0,
                    camera_id: 1,
                    pixel: pixel1,
                },
            ],
        }],
        imu_links: Vec::new(),
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0e8,
        initial_accel_bias_weight: 1.0e1,
        initial_gyro_bias_weight: 1.0e2,
        prior: None,
        anchor_point: None,
        gravity_world: Vector3::new(0.0, 0.0, -9.81),
        scalar_mode: ScalarMode::ExtendedF64,
        factor_config: FactorConfig::default(),
    };
    let factors = problem.linearize(&problem.initial_state()).unwrap().factors;
    let visual = factors.iter().find(|factor| factor.rows() == 4).unwrap();
    assert!(visual.residual.norm() < 1e-10, "{:?}", visual.residual);

    // The opposite transform direction must not accidentally pass the same
    // contract: applying T_cam_imu as if it were T_imu_cam moves cam1.  Use a
    // wide outlier gate here so the regression checks the residual itself.
    let wrong_pose = t_w_i.compose(&calibration.camera_to_imu(1).unwrap().inverse());
    let wrong = visual_reprojection_factor(
        calibration.camera(1).unwrap(),
        &wrong_pose,
        point_world,
        pixel1,
        FactorConfig {
            outlier_threshold: 1.0e9,
            ..FactorConfig::default()
        },
    )
    .unwrap();
    assert!(wrong.residual.norm() > 1e-3);
}

#[test]
fn stereo_triangulation_uses_t_w_i_times_t_imu_cam() {
    let calibration = fixture();
    let t_w_i = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(-0.04, 0.08, 0.12)),
        Vector3::new(-0.3, 0.25, 0.1),
    );
    let t_w_c0 = t_w_i.compose(calibration.camera_to_imu(0).unwrap());
    let t_w_c1 = t_w_i.compose(calibration.camera_to_imu(1).unwrap());
    let point_world = Point3::new(0.2, -0.1, 3.0);
    let bearing0 = calibration
        .camera(0)
        .unwrap()
        .unproject(
            &calibration
                .camera(0)
                .unwrap()
                .project(&t_w_c0.inverse().transform_point(&point_world))
                .unwrap(),
        )
        .unwrap();
    let bearing1 = calibration
        .camera(1)
        .unwrap()
        .unproject(
            &calibration
                .camera(1)
                .unwrap()
                .project(&t_w_c1.inverse().transform_point(&point_world))
                .unwrap(),
        )
        .unwrap();
    let (triangulated, _) = triangulate_two_rays(&t_w_c0, bearing0, &t_w_c1, bearing1).unwrap();
    assert!((triangulated - point_world.coords).norm() < 1e-8);
}
