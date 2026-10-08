use nalgebra::{Matrix3, Point3, Vector3};
use visloc_fusion::{
    gnss_fix_covariance, interpolate_gnss_fix, GnssFixSource, GnssInterpolationConfig,
    GnssMeasurement, MeasurementBuffer, PositionCovariance, TimeDelta, Timestamp,
};

fn ms(milliseconds: i128) -> Timestamp {
    Timestamp::from_nanoseconds(milliseconds * 1_000_000)
}

fn buffer() -> MeasurementBuffer<GnssMeasurement> {
    MeasurementBuffer::from_measurements([
        GnssMeasurement::new(ms(0), Point3::new(0.0, 0.0, 0.0)).with_accuracy(Some(1.0), Some(2.0)),
        GnssMeasurement::new(ms(200), Point3::new(2.0, 4.0, 1.0))
            .with_accuracy(Some(3.0), Some(4.0)),
        // 5 s gap: a dropout.
        GnssMeasurement::new(ms(5_200), Point3::new(50.0, 0.0, 0.0))
            .with_accuracy(Some(1.0), Some(2.0)),
    ])
}

fn config() -> GnssInterpolationConfig {
    GnssInterpolationConfig {
        motion_sigma_per_second: 0.0,
        ..GnssInterpolationConfig::default()
    }
}

#[test]
fn nearby_fix_is_used_directly() {
    let fix = interpolate_gnss_fix(&buffer(), ms(195), &config()).expect("fix");
    assert_eq!(fix.source, GnssFixSource::Nearest { index: 1 });
    assert_eq!(fix.position_enu, Vector3::new(2.0, 4.0, 1.0));
    assert!((fix.horizontal_sigma() - 3.0).abs() < 1e-12);
    assert!((fix.vertical_sigma() - 4.0).abs() < 1e-12);
}

#[test]
fn bracketing_fixes_are_interpolated_with_conservative_covariance() {
    let fix = interpolate_gnss_fix(&buffer(), ms(50), &config()).expect("fix");
    match fix.source {
        GnssFixSource::Interpolated {
            before,
            after,
            alpha,
        } => {
            assert_eq!((before, after), (0, 1));
            assert!((alpha - 0.25).abs() < 1e-12);
        }
        other => panic!("unexpected source {other:?}"),
    }
    assert!((fix.position_enu - Vector3::new(0.5, 1.0, 0.25)).norm() < 1e-12);
    // Convex blend of variances: 0.75·1² + 0.25·3² = 3.0.
    assert!((fix.covariance[(0, 0)] - 3.0).abs() < 1e-12);
    // Never more confident than the better of the two fixes.
    assert!(fix.horizontal_sigma() >= 1.0);
}

#[test]
fn motion_term_inflates_interpolated_covariance() {
    let config = GnssInterpolationConfig {
        motion_sigma_per_second: 1.0,
        ..config()
    };
    let fix = interpolate_gnss_fix(&buffer(), ms(100), &config).expect("fix");
    // 0.5·1 + 0.5·9 = 5, plus (1 m/s · 0.1 s)².
    assert!((fix.covariance[(0, 0)] - 5.01).abs() < 1e-9);
}

#[test]
fn wide_gaps_are_dropouts() {
    assert!(interpolate_gnss_fix(&buffer(), ms(2_000), &config()).is_none());
    // Before the first fix and after the last one (beyond the nearest
    // tolerance) there is no bracket either.
    assert!(interpolate_gnss_fix(&buffer(), ms(-500), &config()).is_none());
    assert!(interpolate_gnss_fix(&buffer(), ms(6_000), &config()).is_none());
    let tolerant = GnssInterpolationConfig {
        max_bracket_gap: TimeDelta::from_nanoseconds(6_000_000_000),
        ..config()
    };
    assert!(interpolate_gnss_fix(&buffer(), ms(2_000), &tolerant).is_some());
}

#[test]
fn covariance_prefers_full_matrix_then_accuracies_then_defaults() {
    let config = config();
    let full = Matrix3::new(4.0, 1.0, 0.0, 1.0, 9.0, 0.0, 0.0, 0.0, 16.0);
    let with_full = GnssMeasurement::new(ms(0), Point3::origin())
        .with_accuracy(Some(100.0), Some(100.0))
        .with_position_covariance(PositionCovariance::new(full));
    assert_eq!(gnss_fix_covariance(&with_full, &config), full);

    let horizontal_only =
        GnssMeasurement::new(ms(0), Point3::origin()).with_accuracy(Some(2.0), None);
    let covariance = gnss_fix_covariance(&horizontal_only, &config);
    assert_eq!(covariance.diagonal(), Vector3::new(4.0, 4.0, 16.0));

    let bare = GnssMeasurement::new(ms(0), Point3::origin());
    let covariance = gnss_fix_covariance(&bare, &config);
    assert_eq!(
        covariance.diagonal(),
        Vector3::new(
            config.default_horizontal_sigma.powi(2),
            config.default_horizontal_sigma.powi(2),
            config.default_vertical_sigma.powi(2)
        )
    );
}
