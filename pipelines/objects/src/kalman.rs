//! Constant-velocity Kalman filter on image boxes, as in SORT, DeepSORT and ByteTrack.
//!
//! The state is `(cx, cy, a, h, vcx, vcy, va, vh)`: box centre, aspect ratio
//! `a = w / h`, height, and their per-frame velocities. The noise is scaled by
//! the box height, so a far (small) object and a near (large) one are tracked
//! with the same relative uncertainty.

use nalgebra::{Matrix4, SMatrix, SVector, Vector4};

use crate::bbox::BBox;

pub type BoxState = SVector<f64, 8>;
pub type BoxCovariance = SMatrix<f64, 8, 8>;

/// Noise weights of the filter; the defaults are ByteTrack's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxKalmanFilter {
    /// Position noise as a fraction of the box height (1/20).
    pub std_weight_position: f64,
    /// Velocity noise as a fraction of the box height (1/160).
    pub std_weight_velocity: f64,
}

impl Default for BoxKalmanFilter {
    fn default() -> Self {
        Self {
            std_weight_position: 1.0 / 20.0,
            std_weight_velocity: 1.0 / 160.0,
        }
    }
}

/// Mean and covariance of one track's box.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxKalmanState {
    pub mean: BoxState,
    pub covariance: BoxCovariance,
}

impl BoxKalmanState {
    /// The box at the state's mean.
    pub fn bbox(&self) -> BBox {
        let height = self.mean[3];
        BBox::from_center_size(self.mean[0], self.mean[1], self.mean[2] * height, height)
    }
}

/// `(cx, cy, w / h, h)` of a box with a positive height.
pub fn box_measurement(bbox: &BBox) -> Vector4<f64> {
    let (cx, cy) = bbox.center();
    let height = bbox.height();
    Vector4::new(cx, cy, bbox.width() / height, height)
}

impl BoxKalmanFilter {
    /// A state at `bbox` with zero velocity.
    pub fn initiate(&self, bbox: &BBox) -> BoxKalmanState {
        let z = box_measurement(bbox);
        let mut mean = BoxState::zeros();
        mean.fixed_rows_mut::<4>(0).copy_from(&z);
        let p = self.std_weight_position * z[3];
        let v = self.std_weight_velocity * z[3];
        let std = [
            2.0 * p,
            2.0 * p,
            1e-2,
            2.0 * p,
            10.0 * v,
            10.0 * v,
            1e-5,
            10.0 * v,
        ];
        BoxKalmanState {
            mean,
            covariance: BoxCovariance::from_diagonal(&BoxState::from_iterator(
                std.iter().map(|s| s * s),
            )),
        }
    }

    /// Advance the state by one frame.
    pub fn predict(&self, state: &mut BoxKalmanState) {
        let height = state.mean[3];
        let p = self.std_weight_position * height;
        let v = self.std_weight_velocity * height;
        let std = [p, p, 1e-2, p, v, v, 1e-5, v];
        let mut motion = BoxCovariance::identity();
        for i in 0..4 {
            motion[(i, i + 4)] = 1.0;
        }
        state.mean = motion * state.mean;
        state.covariance = motion * state.covariance * motion.transpose()
            + BoxCovariance::from_diagonal(&BoxState::from_iterator(std.iter().map(|s| s * s)));
    }

    /// Correct the state with a measured box. A box with a non-positive height
    /// is ignored.
    pub fn update(&self, state: &mut BoxKalmanState, bbox: &BBox) {
        if bbox.height() <= 0.0 {
            return;
        }
        let height = state.mean[3];
        let p = self.std_weight_position * height;
        let measurement_noise = Matrix4::from_diagonal(&Vector4::new(p * p, p * p, 1e-2, p * p));
        // H = [I4 0]: the measurement is the first four state entries.
        let projected_covariance =
            state.covariance.fixed_view::<4, 4>(0, 0).into_owned() + measurement_noise;
        let covariance_ht = state.covariance.fixed_view::<8, 4>(0, 0).into_owned();
        let Some(cholesky) = projected_covariance.cholesky() else {
            *state = self.initiate(bbox);
            return;
        };
        // K = P Hᵀ S⁻¹, from S Kᵀ = H P.
        let gain = cholesky.solve(&covariance_ht.transpose()).transpose();
        let innovation = box_measurement(bbox) - state.mean.fixed_rows::<4>(0);
        state.mean += gain * innovation;
        let covariance = state.covariance - gain * projected_covariance * gain.transpose();
        state.covariance = 0.5 * (covariance + covariance.transpose());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initiate_round_trips_the_box() {
        let filter = BoxKalmanFilter::default();
        let bbox = BBox::from_xywh(10.0, 20.0, 30.0, 60.0);
        let state = filter.initiate(&bbox);
        let back = state.bbox();
        for (a, b) in [
            (back.x_min, bbox.x_min),
            (back.y_min, bbox.y_min),
            (back.x_max, bbox.x_max),
            (back.y_max, bbox.y_max),
        ] {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn learns_a_constant_velocity_and_predicts_with_it() {
        let filter = BoxKalmanFilter::default();
        let at = |k: f64| BBox::from_center_size(100.0 + 4.0 * k, 50.0 - 1.0 * k, 40.0, 80.0);
        let mut state = filter.initiate(&at(0.0));
        for k in 1..30 {
            filter.predict(&mut state);
            filter.update(&mut state, &at(k as f64));
        }
        assert!((state.mean[4] - 4.0).abs() < 0.05, "vx {}", state.mean[4]);
        assert!((state.mean[5] + 1.0).abs() < 0.05, "vy {}", state.mean[5]);
        // Five frames without measurements: the box keeps moving.
        for _ in 0..5 {
            filter.predict(&mut state);
        }
        let (cx, cy) = state.bbox().center();
        assert!((cx - (100.0 + 4.0 * 34.0)).abs() < 1.0, "cx {cx}");
        assert!((cy - (50.0 - 34.0)).abs() < 1.0, "cy {cy}");
        assert_eq!(state.covariance, state.covariance.transpose());
    }

    #[test]
    fn a_zero_height_measurement_is_ignored() {
        let filter = BoxKalmanFilter::default();
        let mut state = filter.initiate(&BBox::from_xywh(0.0, 0.0, 10.0, 20.0));
        let before = state.clone();
        filter.update(&mut state, &BBox::new(0.0, 5.0, 10.0, 5.0));
        assert_eq!(state, before);
    }
}
