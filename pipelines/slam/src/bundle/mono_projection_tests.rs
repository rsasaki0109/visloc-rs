use super::*;

fn camera(model: CameraModel, params: Vec<f64>) -> Camera {
    Camera {
        id: 1,
        model,
        width: 640,
        height: 480,
        params,
    }
}

fn sample_points() -> Vec<Point3<f64>> {
    let mut points = Vec::new();
    for i in 0..7 {
        for j in 0..5 {
            let z = 1.5 + 0.37 * f64::from(i + j);
            points.push(Point3::new(
                -1.3 + 0.41 * f64::from(i),
                -0.9 + 0.43 * f64::from(j),
                z,
            ));
        }
    }
    points
}

/// The pre-lens-model closed forms, copied verbatim, so any change to the
/// distortion-free / radial-only arithmetic is caught bit-for-bit.
fn historical_pinhole(
    (fx, fy, cx, cy): (f64, f64, f64, f64),
    xc: &Point3<f64>,
) -> (Point2<f64>, Matrix2x3<f64>) {
    let predicted = Point2::new(fx * xc.x / xc.z + cx, fy * xc.y / xc.z + cy);
    let z_inv = 1.0 / xc.z;
    let mut j_pi = Matrix2x3::<f64>::zeros();
    j_pi[(0, 0)] = fx * z_inv;
    j_pi[(0, 2)] = -fx * xc.x * z_inv * z_inv;
    j_pi[(1, 1)] = fy * z_inv;
    j_pi[(1, 2)] = -fy * xc.y * z_inv * z_inv;
    (predicted, j_pi)
}

fn historical_radial(
    (fx, fy, cx, cy): (f64, f64, f64, f64),
    (k1, k2): (f64, f64),
    xc: &Point3<f64>,
) -> (Point2<f64>, Matrix2x3<f64>) {
    let x = xc.x / xc.z;
    let y = xc.y / xc.z;
    let r2 = x * x + y * y;
    let d = 1.0 + k1 * r2 + k2 * r2 * r2;
    let predicted = Point2::new(fx * (x * d) + cx, fy * (y * d) + cy);
    let g = k1 + 2.0 * k2 * r2;
    let d11 = d + 2.0 * x * x * g;
    let d12 = 2.0 * x * y * g;
    let d22 = d + 2.0 * y * y * g;
    let z_inv = 1.0 / xc.z;
    let j_pi = Matrix2x3::new(
        fx * d11 * z_inv,
        fx * d12 * z_inv,
        -fx * (d11 * x + d12 * y) * z_inv,
        fy * d12 * z_inv,
        fy * d22 * z_inv,
        -fy * (d12 * x + d22 * y) * z_inv,
    );
    (predicted, j_pi)
}

fn bits(value: &(Point2<f64>, Matrix2x3<f64>)) -> Vec<u64> {
    value
        .0
        .coords
        .iter()
        .chain(value.1.iter())
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn distortion_free_and_radial_cameras_keep_the_historical_closed_forms_bit_for_bit() {
    let intrinsics = (512.5, 498.25, 321.0, 239.5);
    let (fx, fy, cx, cy) = intrinsics;
    let pinholes = [
        Camera::pinhole(1, 640, 480, fx, fy, cx, cy),
        Camera::pinhole_radial(1, 640, 480, fx, fy, cx, cy, 0.0, 0.0),
        camera(
            CameraModel::OpenCv,
            vec![fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0],
        ),
    ];
    let radials = [
        Camera::pinhole_radial(1, 640, 480, fx, fy, cx, cy, -0.17, 0.06),
        camera(
            CameraModel::OpenCv,
            vec![fx, fy, cx, cy, -0.17, 0.06, 0.0, 0.0],
        ),
    ];
    for xc in sample_points() {
        let expected = bits(&historical_pinhole(intrinsics, &xc));
        for camera in &pinholes {
            let projection = MonoProjection::for_camera(camera);
            assert_eq!(projection, MonoProjection::Pinhole, "{camera:?}");
            let got = mono_project_with_jacobian(projection, &intrinsics, &xc).unwrap();
            assert_eq!(bits(&got), expected, "{camera:?} at {xc:?}");
        }
        let expected = bits(&historical_radial(intrinsics, (-0.17, 0.06), &xc));
        for camera in &radials {
            let projection = MonoProjection::for_camera(camera);
            assert_eq!(
                projection,
                MonoProjection::Radial {
                    k1: -0.17,
                    k2: 0.06
                }
            );
            let got = mono_project_with_jacobian(projection, &intrinsics, &xc).unwrap();
            assert_eq!(bits(&got), expected, "{camera:?} at {xc:?}");
            // The cost evaluates `Camera::project`: identical bits.
            assert_eq!(got.0, camera.project(&xc).unwrap());
        }
    }
}

#[test]
fn every_other_lens_uses_project_and_its_analytic_jacobian() {
    let lenses = [
        camera(
            CameraModel::OpenCv,
            vec![500.0, 500.0, 320.0, 240.0, -0.15, 0.05, 0.004, -0.003],
        ),
        camera(
            CameraModel::FullOpenCv,
            vec![
                500.0, 500.0, 320.0, 240.0, -0.15, 0.05, 0.004, -0.003, 0.01, 0.02, -0.01, 0.005,
            ],
        ),
        Camera::opencv_fisheye(
            1,
            640,
            480,
            300.0,
            300.0,
            320.0,
            240.0,
            [0.05, -0.02, 0.0, 0.0],
        ),
        camera(
            CameraModel::SimpleRadialFisheye,
            vec![300.0, 320.0, 240.0, 0.04],
        ),
        camera(
            CameraModel::RadialFisheye,
            vec![300.0, 320.0, 240.0, 0.04, -0.01],
        ),
        camera(CameraModel::Fov, vec![400.0, 400.0, 320.0, 240.0, 0.9]),
        Camera::double_sphere(1, 640, 480, 250.0, 250.0, 320.0, 240.0, -0.2, 0.6),
    ];
    for camera in &lenses {
        let projection = MonoProjection::for_camera(camera);
        assert_eq!(projection, MonoProjection::Lens(camera));
        let intrinsics = camera.intrinsics().unwrap();
        for xc in sample_points() {
            let (predicted, j_pi) =
                mono_project_with_jacobian(projection, &intrinsics, &xc).unwrap();
            assert_eq!(predicted, camera.project(&xc).unwrap());
            let h = 1.0e-6;
            for axis in 0..3 {
                let mut plus = xc;
                let mut minus = xc;
                plus[axis] += h;
                minus[axis] -= h;
                let numeric =
                    (camera.project(&plus).unwrap() - camera.project(&minus).unwrap()) / (2.0 * h);
                for row in 0..2 {
                    let analytic = j_pi[(row, axis)];
                    assert!(
                        (analytic - numeric[row]).abs() < 1.0e-5 * (1.0 + analytic.abs()),
                        "{:?} at {xc:?}: d{row}/d{axis} {analytic} vs {}",
                        camera.model,
                        numeric[row]
                    );
                }
            }
        }
        // Behind the camera the BA skips the observation, like the cost.
        assert!(
            mono_project_with_jacobian(projection, &intrinsics, &Point3::new(0.1, 0.1, -1.0))
                .is_none()
        );
    }
}
