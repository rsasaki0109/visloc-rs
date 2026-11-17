//! KITTI raw OXTS GPS/INS as geodetic ground truth and GNSS input.
//!
//! [`crate::kitti_imu`] parses the OXTS packets; this module turns them into
//! metric quantities in a local East-North-Up (ENU) frame:
//!
//! - [`LocalEnuFrame`]: exact WGS84 geodetic → ECEF → local tangent-plane ENU
//!   conversion (not the KITTI devkit's Mercator approximation).
//! - [`LocalEnuFrame::imu_to_enu`]: the OXTS body (IMU) pose in that ENU
//!   frame from latitude/longitude/altitude and roll/pitch/yaw. KITTI's
//!   convention (`dataformat.txt`): IMU `x` forward, `y` left, `z` up; yaw is
//!   the heading with 0 = east, counter-clockwise positive; the attitude is
//!   `R = Rz(yaw)·Ry(pitch)·Rx(roll)` w.r.t. the local level frame at the
//!   sample, which is rotated into the origin's ENU frame (meridian
//!   convergence) here.
//! - [`kitti_oxts_gnss_measurements`]: the OXTS positions as a
//!   [`GnssMeasurement`] stream, each tagged with its reported
//!   `pos_accuracy`.
//! - [`KittiRawStereoCalibration`]: the raw calibration chain IMU → Velodyne →
//!   camera 0 → rectified camera 0, and the rectified stereo projections
//!   (`P_rect_00` / `P_rect_01`), which can be written as an odometry-style
//!   `calib.txt` (`P0` / `P1`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Isometry3, Matrix3, Point3, Rotation3, Translation3, UnitQuaternion, Vector3};
use thiserror::Error;
use visloc_fusion::{GnssMeasurement, Timestamp};

use crate::kitti_imu::{KittiOxtsRecord, KittiOxtsSample};

/// WGS84 semi-major axis (metres).
pub const WGS84_SEMI_MAJOR_AXIS_M: f64 = 6_378_137.0;
/// WGS84 flattening.
pub const WGS84_FLATTENING: f64 = 1.0 / 298.257_223_563;

/// WGS84 geodetic coordinates (degrees, degrees, metres above the ellipsoid)
/// to Earth-centred Earth-fixed coordinates (metres).
pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Vector3<f64> {
    let e2 = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING);
    let (sin_lat, cos_lat) = lat_deg.to_radians().sin_cos();
    let (sin_lon, cos_lon) = lon_deg.to_radians().sin_cos();
    let n = WGS84_SEMI_MAJOR_AXIS_M / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    Vector3::new(
        (n + alt_m) * cos_lat * cos_lon,
        (n + alt_m) * cos_lat * sin_lon,
        (n * (1.0 - e2) + alt_m) * sin_lat,
    )
}

/// Rotation taking ECEF vectors to the local ENU frame at a geodetic point
/// (rows are the east, north, and up unit vectors in ECEF).
pub fn enu_from_ecef_rotation(lat_deg: f64, lon_deg: f64) -> Rotation3<f64> {
    let (sin_lat, cos_lat) = lat_deg.to_radians().sin_cos();
    let (sin_lon, cos_lon) = lon_deg.to_radians().sin_cos();
    Rotation3::from_matrix_unchecked(Matrix3::new(
        -sin_lon,
        cos_lon,
        0.0,
        -sin_lat * cos_lon,
        -sin_lat * sin_lon,
        cos_lat,
        cos_lat * cos_lon,
        cos_lat * sin_lon,
        sin_lat,
    ))
}

/// OXTS body (IMU) attitude w.r.t. the local level frame at the sample
/// (`x` east, `y` north, `z` up): `Rz(yaw)·Ry(pitch)·Rx(roll)`.
pub fn kitti_oxts_body_to_local_level(
    roll_rad: f64,
    pitch_rad: f64,
    yaw_rad: f64,
) -> Rotation3<f64> {
    Rotation3::from_axis_angle(&Vector3::z_axis(), yaw_rad)
        * Rotation3::from_axis_angle(&Vector3::y_axis(), pitch_rad)
        * Rotation3::from_axis_angle(&Vector3::x_axis(), roll_rad)
}

/// A local tangent-plane East-North-Up frame anchored at a geodetic origin.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalEnuFrame {
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    pub origin_alt_m: f64,
    origin_ecef: Vector3<f64>,
    enu_from_ecef: Rotation3<f64>,
}

impl LocalEnuFrame {
    pub fn new(origin_lat_deg: f64, origin_lon_deg: f64, origin_alt_m: f64) -> Self {
        Self {
            origin_lat_deg,
            origin_lon_deg,
            origin_alt_m,
            origin_ecef: geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m),
            enu_from_ecef: enu_from_ecef_rotation(origin_lat_deg, origin_lon_deg),
        }
    }

    /// ENU frame anchored at an OXTS sample's position.
    pub fn at_oxts_sample(sample: &KittiOxtsSample) -> Self {
        Self::new(sample.lat_deg, sample.lon_deg, sample.alt_m)
    }

    /// Geodetic coordinates to ENU metres in this frame.
    pub fn geodetic_to_enu(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> Vector3<f64> {
        self.enu_from_ecef * (geodetic_to_ecef(lat_deg, lon_deg, alt_m) - self.origin_ecef)
    }

    /// Position of an OXTS sample (the INS reference point) in this frame.
    pub fn oxts_position(&self, sample: &KittiOxtsSample) -> Vector3<f64> {
        self.geodetic_to_enu(sample.lat_deg, sample.lon_deg, sample.alt_m)
    }

    /// OXTS body (IMU) to ENU pose: maps IMU-frame points into this frame.
    pub fn imu_to_enu(&self, sample: &KittiOxtsSample) -> Isometry3<f64> {
        let level =
            kitti_oxts_body_to_local_level(sample.roll_rad, sample.pitch_rad, sample.yaw_rad);
        // Local level frame at the sample -> ECEF -> this frame's ENU.
        let local_from_ecef = enu_from_ecef_rotation(sample.lat_deg, sample.lon_deg);
        let rotation = self.enu_from_ecef * local_from_ecef.inverse() * level;
        Isometry3::from_parts(
            Translation3::from(self.oxts_position(sample)),
            UnitQuaternion::from_rotation_matrix(&rotation),
        )
    }
}

/// Every OXTS record's position as a GNSS fix in `frame`, timestamped with the
/// record's `oxts/timestamps.txt` time and tagged with its reported
/// `pos_accuracy` as the horizontal accuracy (vertical left unset, so
/// consumers apply their own default ratio). Non-finite or non-positive
/// accuracies are dropped.
pub fn kitti_oxts_gnss_measurements(
    records: &[KittiOxtsRecord],
    frame: &LocalEnuFrame,
) -> Vec<GnssMeasurement> {
    records
        .iter()
        .map(|record| {
            let accuracy = record.sample.position_accuracy_m;
            let accuracy = (accuracy.is_finite() && accuracy > 0.0).then_some(accuracy);
            GnssMeasurement::new(
                Timestamp::from_nanoseconds(record.timestamp_nanoseconds),
                Point3::from(frame.oxts_position(&record.sample)),
            )
            .with_accuracy(accuracy, None)
        })
        .collect()
}

#[derive(Debug, Error)]
pub enum KittiRawCalibrationError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("calibration key `{key}` is missing")]
    MissingKey { key: String },
    #[error("calibration key `{key}` has {actual} values, expected {expected}")]
    WrongLength {
        key: String,
        expected: usize,
        actual: usize,
    },
}

/// Parse a KITTI raw calibration file (`calib_cam_to_cam.txt`,
/// `calib_imu_to_velo.txt`, `calib_velo_to_cam.txt`) into `label -> values`.
///
/// Lines are `label: v0 v1 ...`. Lines whose values are not all numbers (for
/// example `calib_time: 25-May-2012 16:47:16`) are skipped.
pub fn parse_kitti_raw_calib_txt(text: &str) -> BTreeMap<String, Vec<f64>> {
    let mut entries = BTreeMap::new();
    for line in text.lines() {
        let Some((label, rest)) = line.split_once(':') else {
            continue;
        };
        let values: Option<Vec<f64>> = rest
            .split_ascii_whitespace()
            .map(|token| token.parse::<f64>().ok().filter(|value| value.is_finite()))
            .collect();
        if let Some(values) = values.filter(|values| !values.is_empty()) {
            entries.insert(label.trim().to_owned(), values);
        }
    }
    entries
}

fn calib_values<'a>(
    entries: &'a BTreeMap<String, Vec<f64>>,
    key: &str,
    expected: usize,
) -> Result<&'a [f64], KittiRawCalibrationError> {
    let values = entries
        .get(key)
        .ok_or_else(|| KittiRawCalibrationError::MissingKey {
            key: key.to_owned(),
        })?;
    if values.len() != expected {
        return Err(KittiRawCalibrationError::WrongLength {
            key: key.to_owned(),
            expected,
            actual: values.len(),
        });
    }
    Ok(values)
}

fn calib_rotation(
    entries: &BTreeMap<String, Vec<f64>>,
    key: &str,
) -> Result<Rotation3<f64>, KittiRawCalibrationError> {
    let r = calib_values(entries, key, 9)?;
    // The files store 6 significant digits; project onto SO(3).
    let matrix = Matrix3::from_row_slice(r);
    Ok(Rotation3::from_matrix_eps(
        &matrix,
        1e-12,
        100,
        Rotation3::identity(),
    ))
}

/// The rigid transform stored as `R` (row-major 3×3) and `T` in a KITTI raw
/// calibration file: it maps points of the source sensor frame into the target
/// frame (`calib_imu_to_velo.txt`: IMU → Velodyne).
pub fn kitti_raw_rigid_transform(
    entries: &BTreeMap<String, Vec<f64>>,
) -> Result<Isometry3<f64>, KittiRawCalibrationError> {
    let rotation = calib_rotation(entries, "R")?;
    let t = calib_values(entries, "T", 3)?;
    Ok(Isometry3::from_parts(
        Translation3::new(t[0], t[1], t[2]),
        UnitQuaternion::from_rotation_matrix(&rotation),
    ))
}

/// The parts of a KITTI raw date's calibration a stereo + GNSS/INS evaluation
/// needs.
#[derive(Debug, Clone, PartialEq)]
pub struct KittiRawStereoCalibration {
    /// Rectified projection of camera 0 (left grey), row-major 3×4.
    pub p_rect_00: [f64; 12],
    /// Rectified projection of camera 1 (right grey), row-major 3×4.
    pub p_rect_01: [f64; 12],
    /// Rectifying rotation of camera 0.
    pub r_rect_00: Rotation3<f64>,
    /// IMU → Velodyne.
    pub imu_to_velo: Isometry3<f64>,
    /// Velodyne → (unrectified) camera 0.
    pub velo_to_cam0: Isometry3<f64>,
}

impl KittiRawStereoCalibration {
    /// Build from the parsed contents of the three calibration files.
    pub fn from_entries(
        cam_to_cam: &BTreeMap<String, Vec<f64>>,
        imu_to_velo: &BTreeMap<String, Vec<f64>>,
        velo_to_cam: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Self, KittiRawCalibrationError> {
        let projection = |key: &str| -> Result<[f64; 12], KittiRawCalibrationError> {
            let values = calib_values(cam_to_cam, key, 12)?;
            let mut out = [0.0; 12];
            out.copy_from_slice(values);
            Ok(out)
        };
        Ok(Self {
            p_rect_00: projection("P_rect_00")?,
            p_rect_01: projection("P_rect_01")?,
            r_rect_00: calib_rotation(cam_to_cam, "R_rect_00")?,
            imu_to_velo: kitti_raw_rigid_transform(imu_to_velo)?,
            velo_to_cam0: kitti_raw_rigid_transform(velo_to_cam)?,
        })
    }

    /// Read `calib_cam_to_cam.txt`, `calib_imu_to_velo.txt`, and
    /// `calib_velo_to_cam.txt` from a KITTI raw date directory.
    pub fn read(date_dir: impl AsRef<Path>) -> Result<Self, KittiRawCalibrationError> {
        let date_dir = date_dir.as_ref();
        let read = |name: &str| -> Result<BTreeMap<String, Vec<f64>>, KittiRawCalibrationError> {
            let path = date_dir.join(name);
            let text = fs::read_to_string(&path)
                .map_err(|source| KittiRawCalibrationError::Io { path, source })?;
            Ok(parse_kitti_raw_calib_txt(&text))
        };
        Self::from_entries(
            &read("calib_cam_to_cam.txt")?,
            &read("calib_imu_to_velo.txt")?,
            &read("calib_velo_to_cam.txt")?,
        )
    }

    /// IMU → rectified camera 0 (the frame the rectified images and a stereo
    /// VO on them live in): `R_rect_00 · T_cam0_velo · T_velo_imu`.
    pub fn imu_to_rectified_cam0(&self) -> Isometry3<f64> {
        let rect = Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_rotation_matrix(&self.r_rect_00),
        );
        rect * self.velo_to_cam0 * self.imu_to_velo
    }

    /// The IMU / INS reference point expressed in the rectified camera-0 frame
    /// (the GNSS lever arm when the OXTS position is the fix).
    pub fn imu_origin_in_rectified_cam0(&self) -> Vector3<f64> {
        self.imu_to_rectified_cam0().translation.vector
    }

    /// The two rectified projections as a KITTI-odometry-style `calib.txt`
    /// (`P0:` = `P_rect_00`, `P1:` = `P_rect_01`).
    pub fn odometry_calib_txt(&self) -> String {
        let row = |label: &str, values: &[f64; 12]| {
            let values: Vec<String> = values.iter().map(|value| format!("{value:.12e}")).collect();
            format!("{label}: {}\n", values.join(" "))
        };
        format!(
            "{}{}",
            row("P0", &self.p_rect_00),
            row("P1", &self.p_rect_01)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(lat: f64, lon: f64, alt: f64, roll: f64, pitch: f64, yaw: f64) -> KittiOxtsSample {
        KittiOxtsSample {
            lat_deg: lat,
            lon_deg: lon,
            alt_m: alt,
            roll_rad: roll,
            pitch_rad: pitch,
            yaw_rad: yaw,
            velocity_north_mps: 0.0,
            velocity_east_mps: 0.0,
            velocity_forward_mps: 0.0,
            velocity_left_mps: 0.0,
            velocity_up_mps: 0.0,
            acceleration_body_mps2: Vector3::zeros(),
            acceleration_nav_mps2: Vector3::zeros(),
            angular_rate_body_rps: Vector3::zeros(),
            angular_rate_nav_rps: Vector3::zeros(),
            position_accuracy_m: 0.02,
            velocity_accuracy_mps: 0.01,
            navigation_status: 4,
            number_of_satellites: 10,
            position_mode: 6,
            velocity_mode: 6,
            orientation_mode: 6,
        }
    }

    #[test]
    fn ecef_matches_known_values() {
        let equator = geodetic_to_ecef(0.0, 0.0, 0.0);
        assert!((equator - Vector3::new(WGS84_SEMI_MAJOR_AXIS_M, 0.0, 0.0)).norm() < 1e-6);
        let pole = geodetic_to_ecef(90.0, 0.0, 0.0);
        // WGS84 semi-minor axis.
        assert!((pole.z - 6_356_752.314_245).abs() < 1e-3);
        assert!(pole.x.abs() < 1e-6);
        let east = geodetic_to_ecef(0.0, 90.0, 100.0);
        assert!((east - Vector3::new(0.0, WGS84_SEMI_MAJOR_AXIS_M + 100.0, 0.0)).norm() < 1e-6);
    }

    #[test]
    fn enu_axes_follow_north_east_and_altitude() {
        // Karlsruhe, where KITTI was recorded.
        let frame = LocalEnuFrame::new(49.0, 8.4, 115.0);
        assert!(frame.geodetic_to_enu(49.0, 8.4, 115.0).norm() < 1e-9);
        let north = frame.geodetic_to_enu(49.001, 8.4, 115.0);
        // One millidegree of latitude is ~111.2 m at 49 deg.
        assert!(north.x.abs() < 1e-6, "{north}");
        assert!((north.y - 111.2).abs() < 0.3, "{north}");
        assert!(north.z.abs() < 0.01, "{north}");
        let east = frame.geodetic_to_enu(49.0, 8.401, 115.0);
        assert!((east.x - 73.1).abs() < 0.3, "{east}");
        assert!(east.y.abs() < 0.01, "{east}");
        let up = frame.geodetic_to_enu(49.0, 8.4, 125.0);
        assert!((up - Vector3::new(0.0, 0.0, 10.0)).norm() < 1e-6);
    }

    #[test]
    fn oxts_attitude_follows_kitti_convention() {
        let frame = LocalEnuFrame::new(49.0, 8.4, 115.0);
        // Yaw 0 = facing east: IMU x (forward) maps to east, y (left) to north.
        let pose = frame.imu_to_enu(&sample(49.0, 8.4, 115.0, 0.0, 0.0, 0.0));
        assert!((pose.rotation * Vector3::x() - Vector3::x()).norm() < 1e-12);
        assert!((pose.rotation * Vector3::y() - Vector3::y()).norm() < 1e-12);
        // Yaw +90 deg = counter-clockwise = facing north.
        let pose = frame.imu_to_enu(&sample(
            49.0,
            8.4,
            115.0,
            0.0,
            0.0,
            std::f64::consts::FRAC_PI_2,
        ));
        assert!((pose.rotation * Vector3::x() - Vector3::y()).norm() < 1e-12);
        // Positive pitch = front down.
        let pitch = 0.1_f64;
        let pose = frame.imu_to_enu(&sample(49.0, 8.4, 115.0, 0.0, pitch, 0.0));
        let forward = pose.rotation * Vector3::x();
        assert!((forward.z + pitch.sin()).abs() < 1e-12, "{forward}");
        // Positive roll = left side up.
        let roll = 0.1_f64;
        let pose = frame.imu_to_enu(&sample(49.0, 8.4, 115.0, roll, 0.0, 0.0));
        let left = pose.rotation * Vector3::y();
        assert!((left.z - roll.sin()).abs() < 1e-12, "{left}");
    }

    #[test]
    fn imu_pose_translation_and_meridian_convergence() {
        let frame = LocalEnuFrame::new(49.0, 8.4, 115.0);
        let s = sample(49.0, 8.41, 116.0, 0.0, 0.0, 0.0);
        let pose = frame.imu_to_enu(&s);
        assert!((pose.translation.vector - frame.oxts_position(&s)).norm() < 1e-12);
        // ~730 m east, the local east axis is rotated by about
        // dlon * sin(lat) = 0.01 deg * 0.755 relative to the origin's.
        let east_axis = pose.rotation * Vector3::x();
        let angle = east_axis.y.atan2(east_axis.x).to_degrees();
        assert!(
            (angle - 0.01 * 49.0_f64.to_radians().sin()).abs() < 2e-4,
            "{angle}"
        );
    }

    #[test]
    fn gnss_measurements_carry_positions_times_and_accuracy() {
        let frame = LocalEnuFrame::new(49.0, 8.4, 115.0);
        let mut bad = sample(49.0005, 8.4, 115.0, 0.0, 0.0, 0.0);
        bad.position_accuracy_m = f64::NAN;
        let records = vec![
            KittiOxtsRecord {
                timestamp_nanoseconds: 1_000,
                sample: sample(49.0, 8.4, 115.0, 0.0, 0.0, 0.0),
                path: PathBuf::from("0.txt"),
            },
            KittiOxtsRecord {
                timestamp_nanoseconds: 2_000,
                sample: bad,
                path: PathBuf::from("1.txt"),
            },
        ];
        let fixes = kitti_oxts_gnss_measurements(&records, &frame);
        assert_eq!(fixes.len(), 2);
        assert_eq!(fixes[0].timestamp, Timestamp::from_nanoseconds(1_000));
        assert!(fixes[0].position_world.coords.norm() < 1e-9);
        assert_eq!(fixes[0].horizontal_accuracy, Some(0.02));
        assert_eq!(fixes[0].vertical_accuracy, None);
        assert!((fixes[1].position_world.y - 55.6).abs() < 0.2);
        assert_eq!(fixes[1].horizontal_accuracy, None);
    }

    const CAM_TO_CAM: &str = "calib_time: 09-Jan-2012 13:57:47\n\
        corner_dist: 9.950000e-02\n\
        S_00: 1.392000e+03 5.120000e+02\n\
        R_rect_00: 1 0 0 0 1 0 0 0 1\n\
        P_rect_00: 7.0e+02 0 6.0e+02 0 0 7.0e+02 1.8e+02 0 0 0 1 0\n\
        P_rect_01: 7.0e+02 0 6.0e+02 -3.8e+02 0 7.0e+02 1.8e+02 0 0 0 1 0\n";
    const IMU_TO_VELO: &str = "calib_time: 25-May-2012 16:47:16\n\
        R: 1 0 0 0 1 0 0 0 1\n\
        T: -0.8 0.3 -0.8\n";
    // Velodyne x forward / y left / z up -> camera x right / y down / z forward.
    const VELO_TO_CAM: &str = "calib_time: 15-Mar-2012 11:51:20\n\
        R: 0 -1 0 0 0 -1 1 0 0\n\
        T: 0 -0.08 -0.27\n\
        delta_f: 0 0\n";

    #[test]
    fn raw_calibration_parses_and_skips_non_numeric_lines() {
        let entries = parse_kitti_raw_calib_txt(CAM_TO_CAM);
        assert!(!entries.contains_key("calib_time"));
        assert_eq!(entries["corner_dist"], vec![0.0995]);
        assert_eq!(entries["P_rect_01"].len(), 12);
        let missing = KittiRawStereoCalibration::from_entries(
            &entries,
            &parse_kitti_raw_calib_txt(IMU_TO_VELO),
            &parse_kitti_raw_calib_txt("R: 1 0 0\nT: 0 0 0\n"),
        );
        assert!(matches!(
            missing,
            Err(KittiRawCalibrationError::WrongLength {
                expected: 9,
                actual: 3,
                ..
            })
        ));
    }

    #[test]
    fn raw_calibration_chains_imu_to_rectified_camera() {
        let calib = KittiRawStereoCalibration::from_entries(
            &parse_kitti_raw_calib_txt(CAM_TO_CAM),
            &parse_kitti_raw_calib_txt(IMU_TO_VELO),
            &parse_kitti_raw_calib_txt(VELO_TO_CAM),
        )
        .expect("calibration");
        let imu_to_cam = calib.imu_to_rectified_cam0();
        // IMU forward (x) is camera forward (z); IMU left (y) is camera -x.
        assert!((imu_to_cam.rotation * Vector3::x() - Vector3::z()).norm() < 1e-12);
        assert!((imu_to_cam.rotation * Vector3::y() + Vector3::x()).norm() < 1e-12);
        // IMU origin: velodyne (-0.8, 0.3, -0.8) -> camera (-0.3, 0.8 - 0.08, -0.8 - 0.27).
        let lever = calib.imu_origin_in_rectified_cam0();
        assert!(
            (lever - Vector3::new(-0.3, 0.72, -1.07)).norm() < 1e-12,
            "{lever}"
        );
        let calib_txt = calib.odometry_calib_txt();
        let reparsed = parse_kitti_raw_calib_txt(&calib_txt);
        assert_eq!(reparsed["P0"], calib.p_rect_00.to_vec());
        assert_eq!(reparsed["P1"], calib.p_rect_01.to_vec());
    }
}
