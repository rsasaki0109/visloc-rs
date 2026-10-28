//! Correcting gravity-aligned odometry into a map's frame with intermittent
//! map fixes.
//!
//! Map-based localization gives an absolute pose only for some images (those
//! that match the map well enough), while odometry such as VIO gives a pose for
//! every frame but drifts. [`OdomCorrection`] keeps the transform from the
//! odometry frame to the map frame (`map -> odom` in REP-105), updated from
//! each map fix, so every odometry pose can be reported in the map frame.
//!
//! Both frames are assumed gravity-aligned (z up), as a VIO world and a map
//! built by VIO are. The offset is then four degrees of freedom: a yaw about
//! the vertical and a translation. Roll and pitch come from the odometry,
//! which observes gravity directly.
//!
//! The correction is causal: it only uses fixes already added. Per fix it
//!
//! - **initializes** once `init_fixes` consecutive fixes agree on one offset,
//!   so a single wrong fix cannot place the trajectory;
//! - **accepts** a fix whose map pose lies within a gate of where the current
//!   offset puts the odometry pose; the gate widens with the odometry
//!   distance travelled since the last accepted fix, to absorb drift;
//! - **rejects** any other fix, and **resets** the offset when `reset_after`
//!   consecutive rejected fixes agree with each other (the odometry itself
//!   jumped, or the earlier offset was wrong).
//!
//! A *weak* fix ([`OdomCorrection::add_weak_fix`]) is a localization with
//! fewer inliers than a fix needs, which is more often wrong. Weak fixes only
//! help to set the first offset, and then `init_fixes_weak` (more than
//! `init_fixes`) consecutive fixes must agree; once the offset is set they
//! are not used. Under lighting the map was not built in, the first images
//! that localize at all often have too few inliers for a fix.
//!
//! The offset is the mean over the accepted fixes of the last `window_s`
//! seconds (circular mean for the yaw), which averages out localization noise
//! while following slow odometry drift.

use std::collections::VecDeque;

use nalgebra::{UnitQuaternion, Vector3};
use visloc_core::geometry::SE3;

#[derive(Debug, Clone, PartialEq)]
pub struct OdomCorrectionConfig {
    /// Consecutive mutually consistent fixes needed before the first offset.
    pub init_fixes: usize,
    /// ... when any of them is weak ([`OdomCorrection::add_weak_fix`]).
    pub init_fixes_weak: usize,
    /// Consecutive mutually consistent rejected fixes that replace the offset.
    pub reset_after: usize,
    /// Position tolerance, in metres, for fixes to count as agreeing with
    /// each other (initialization, reset).
    pub agreement_m: f64,
    /// Position gate, in metres, for accepting a fix right after another.
    pub gate_m: f64,
    /// Extra position gate per metre of odometry travel since the last
    /// accepted fix.
    pub gate_per_m: f64,
    /// Yaw gate, in degrees, for agreement and acceptance.
    pub gate_yaw_deg: f64,
    /// Extra yaw gate, in degrees, per metre of odometry travel since the
    /// last accepted fix.
    pub gate_yaw_per_m_deg: f64,
    /// Accepted fixes older than this (relative to the newest) leave the
    /// averaging window.
    pub window_s: f64,
}

impl Default for OdomCorrectionConfig {
    fn default() -> Self {
        Self {
            init_fixes: 2,
            init_fixes_weak: 3,
            reset_after: 3,
            agreement_m: 0.10,
            gate_m: 0.15,
            gate_per_m: 0.05,
            gate_yaw_deg: 3.0,
            gate_yaw_per_m_deg: 1.0,
            window_s: 5.0,
        }
    }
}

/// What [`OdomCorrection::add_fix`] did with a fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixOutcome {
    /// No offset yet; the fix waits for agreeing ones.
    Pending,
    /// This fix completed initialization.
    Initialized,
    /// The fix agreed with the offset and updated it.
    Accepted,
    /// The fix disagreed with the offset and was not used.
    Rejected,
    /// This fix completed a run of agreeing rejected fixes, which replaced
    /// the offset.
    Reset,
    /// A weak fix after the offset was set: not used.
    Unused,
}

/// Four-degree-of-freedom transform: `p_map = Rz(yaw) * p_odom + translation`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Offset {
    pub(crate) yaw: f64,
    pub(crate) translation: Vector3<f64>,
}

impl Offset {
    pub(crate) fn rotation(&self) -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), self.yaw)
    }

    pub(crate) fn se3(&self) -> SE3 {
        SE3::new(self.rotation(), self.translation)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Fix {
    timestamp_ns: i64,
    map_position: Vector3<f64>,
    odom_position: Vector3<f64>,
    /// Yaw of `map_from_body * odom_from_body^-1`, the offset this fix alone
    /// implies.
    yaw: f64,
    weak: bool,
}

pub(crate) fn yaw_of(rotation: &UnitQuaternion<f64>) -> f64 {
    let matrix = rotation.to_rotation_matrix();
    matrix[(1, 0)].atan2(matrix[(0, 0)])
}

pub(crate) fn wrap_angle(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// Mean offset over `fixes` (non-empty): circular mean yaw, then the mean
/// translation under that yaw.
fn fit(fixes: &[&Fix]) -> Offset {
    let (sin, cos) = fixes.iter().fold((0.0, 0.0), |(sin, cos), fix| {
        (sin + fix.yaw.sin(), cos + fix.yaw.cos())
    });
    let yaw = sin.atan2(cos);
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
    let translation = fixes
        .iter()
        .map(|fix| fix.map_position - rotation * fix.odom_position)
        .sum::<Vector3<f64>>()
        / fixes.len() as f64;
    Offset { yaw, translation }
}

fn residuals(offset: &Offset, fix: &Fix) -> (f64, f64) {
    let predicted = offset.rotation() * fix.odom_position + offset.translation;
    (
        (predicted - fix.map_position).norm(),
        wrap_angle(fix.yaw - offset.yaw).abs().to_degrees(),
    )
}

#[derive(Debug, Clone)]
pub struct OdomCorrection {
    config: OdomCorrectionConfig,
    offset: Option<Offset>,
    /// Accepted fixes inside the averaging window, oldest first.
    window: VecDeque<Fix>,
    /// Pending (before initialization) or consecutive rejected fixes.
    candidates: Vec<Fix>,
    /// Odometry position of the previous fix, and the odometry path length
    /// (sampled at fixes) since the last accepted one.
    previous_odom_position: Option<Vector3<f64>>,
    travel_since_accept_m: f64,
}

impl OdomCorrection {
    pub fn new(config: OdomCorrectionConfig) -> Self {
        Self {
            config,
            offset: None,
            window: VecDeque::new(),
            candidates: Vec::new(),
            previous_odom_position: None,
            travel_since_accept_m: 0.0,
        }
    }

    pub const fn config(&self) -> &OdomCorrectionConfig {
        &self.config
    }

    /// Whether the offset is set, i.e. odometry poses have a map pose.
    pub const fn is_set(&self) -> bool {
        self.offset.is_some()
    }

    /// The current odometry-to-map transform, once set.
    pub fn map_from_odom(&self) -> Option<SE3> {
        self.offset.as_ref().map(Offset::se3)
    }

    /// Timestamp and odometry position of the newest fix the offset uses.
    /// The error of [`Self::map_from_body`] grows with the odometry travel
    /// (and time) since this fix, so callers can use it to judge a pose.
    pub fn last_fix(&self) -> Option<(i64, Vector3<f64>)> {
        self.offset?;
        self.window
            .back()
            .map(|fix| (fix.timestamp_ns, fix.odom_position))
    }

    /// An odometry pose expressed in the map frame, once set.
    pub fn map_from_body(&self, odom_from_body: &SE3) -> Option<SE3> {
        self.map_from_odom()
            .map(|map_from_odom| map_from_odom.compose(odom_from_body))
    }

    /// Adds a map fix: the body's map pose from localization, and its
    /// odometry pose at the same instant. Fixes must arrive in time order.
    pub fn add_fix(
        &mut self,
        timestamp_ns: i64,
        map_from_body: &SE3,
        odom_from_body: &SE3,
    ) -> FixOutcome {
        self.add(timestamp_ns, map_from_body, odom_from_body, false)
    }

    /// Adds a weak fix: a localization with fewer inliers than a fix needs.
    /// It counts toward setting the first offset, which then takes
    /// `init_fixes_weak` agreeing fixes, and is [`FixOutcome::Unused`] once
    /// the offset is set.
    pub fn add_weak_fix(
        &mut self,
        timestamp_ns: i64,
        map_from_body: &SE3,
        odom_from_body: &SE3,
    ) -> FixOutcome {
        self.add(timestamp_ns, map_from_body, odom_from_body, true)
    }

    fn add(
        &mut self,
        timestamp_ns: i64,
        map_from_body: &SE3,
        odom_from_body: &SE3,
        weak: bool,
    ) -> FixOutcome {
        let fix = Fix {
            timestamp_ns,
            map_position: map_from_body.translation,
            odom_position: odom_from_body.translation,
            yaw: yaw_of(&(map_from_body.rotation * odom_from_body.rotation.inverse())),
            weak,
        };
        if let Some(previous) = self.previous_odom_position {
            self.travel_since_accept_m += (fix.odom_position - previous).norm();
        }
        self.previous_odom_position = Some(fix.odom_position);

        let Some(offset) = self.offset else {
            return if self.push_initial(fix) {
                Self::adopt(&mut self.window, &mut self.candidates);
                self.refit();
                self.travel_since_accept_m = 0.0;
                FixOutcome::Initialized
            } else {
                FixOutcome::Pending
            };
        };

        if weak {
            return FixOutcome::Unused;
        }
        let (position_residual, yaw_residual) = residuals(&offset, &fix);
        let travel = self.travel_since_accept_m;
        let accepted = position_residual <= self.config.gate_m + self.config.gate_per_m * travel
            && yaw_residual <= self.config.gate_yaw_deg + self.config.gate_yaw_per_m_deg * travel;
        if accepted {
            self.candidates.clear();
            self.window.push_back(fix);
            self.refit();
            self.travel_since_accept_m = 0.0;
            return FixOutcome::Accepted;
        }
        if self.push_candidate(fix, self.config.reset_after) {
            Self::adopt(&mut self.window, &mut self.candidates);
            self.refit();
            self.travel_since_accept_m = 0.0;
            FixOutcome::Reset
        } else {
            FixOutcome::Rejected
        }
    }

    /// Appends to the candidate run, keeping only its last `needed` fixes,
    /// and reports whether that many now agree on one offset.
    fn push_candidate(&mut self, fix: Fix, needed: usize) -> bool {
        self.candidates.push(fix);
        let needed = needed.max(1);
        if self.candidates.len() > needed {
            self.candidates.drain(..self.candidates.len() - needed);
        }
        self.candidates.len() == needed && self.agree(&self.candidates)
    }

    /// Appends to the candidate run before the first offset and reports
    /// whether its newest fixes now set it: `init_fixes` of them agreeing, or
    /// `init_fixes_weak` when any of those is weak. On success the run keeps
    /// only those.
    fn push_initial(&mut self, fix: Fix) -> bool {
        let strong = self.config.init_fixes.max(1);
        let weak = self.config.init_fixes_weak.max(strong);
        self.candidates.push(fix);
        if self.candidates.len() > weak {
            self.candidates.drain(..self.candidates.len() - weak);
        }
        for needed in [strong, weak] {
            let Some(start) = self.candidates.len().checked_sub(needed) else {
                continue;
            };
            let newest = &self.candidates[start..];
            let required = if newest.iter().any(|fix| fix.weak) {
                weak
            } else {
                strong
            };
            if needed == required && self.agree(newest) {
                self.candidates.drain(..start);
                return true;
            }
        }
        false
    }

    /// Whether `fixes` agree on one offset.
    fn agree(&self, fixes: &[Fix]) -> bool {
        let fixes = fixes.iter().collect::<Vec<_>>();
        let offset = fit(&fixes);
        fixes.iter().all(|fix| {
            let (position, yaw) = residuals(&offset, fix);
            position <= self.config.agreement_m && yaw <= self.config.gate_yaw_deg
        })
    }

    fn adopt(window: &mut VecDeque<Fix>, candidates: &mut Vec<Fix>) {
        window.clear();
        window.extend(candidates.drain(..));
    }

    /// Drops fixes outside the window and re-estimates the offset.
    fn refit(&mut self) {
        let Some(newest) = self.window.back().map(|fix| fix.timestamp_ns) else {
            return;
        };
        let window_ns = (self.config.window_s * 1e9) as i64;
        while self.window.len() > 1
            && self
                .window
                .front()
                .is_some_and(|fix| newest - fix.timestamp_ns > window_ns)
        {
            self.window.pop_front();
        }
        let fixes = self.window.iter().collect::<Vec<_>>();
        self.offset = Some(fit(&fixes));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECOND: i64 = 1_000_000_000;

    fn truth() -> SE3 {
        SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.6),
            Vector3::new(3.0, -2.0, 0.5),
        )
    }

    /// Odometry pose `index` on a curved, tilting path.
    fn odom(index: usize) -> SE3 {
        let t = index as f64 * 0.2;
        SE3::new(
            UnitQuaternion::from_euler_angles(0.05 * t.sin(), -0.03, 0.4 * t),
            Vector3::new(t.cos() * 2.0, t.sin() * 2.0, 0.1 * t),
        )
    }

    fn fix_at(
        correction: &mut OdomCorrection,
        index: usize,
        map_from_odom: &SE3,
        shift: Vector3<f64>,
    ) -> FixOutcome {
        let odom_from_body = odom(index);
        let mut map_from_body = map_from_odom.compose(&odom_from_body);
        map_from_body.translation += shift;
        correction.add_fix(index as i64 * SECOND / 10, &map_from_body, &odom_from_body)
    }

    fn assert_pose_close(lhs: &SE3, rhs: &SE3, tolerance: f64) {
        assert!(
            (lhs.translation - rhs.translation).norm() < tolerance,
            "{lhs:?} vs {rhs:?}"
        );
        assert!(
            lhs.rotation.angle_to(&rhs.rotation) < tolerance,
            "{lhs:?} vs {rhs:?}"
        );
    }

    #[test]
    fn exact_fixes_recover_the_offset_after_initialization() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig::default());
        assert_eq!(correction.map_from_body(&odom(0)), None);
        assert_eq!(
            fix_at(&mut correction, 0, &truth(), Vector3::zeros()),
            FixOutcome::Pending
        );
        assert_eq!(correction.last_fix(), None);
        assert_eq!(
            fix_at(&mut correction, 1, &truth(), Vector3::zeros()),
            FixOutcome::Initialized
        );
        assert_eq!(
            fix_at(&mut correction, 2, &truth(), Vector3::zeros()),
            FixOutcome::Accepted
        );
        assert_eq!(
            correction.last_fix(),
            Some((2 * SECOND / 10, odom(2).translation))
        );
        for index in [3, 10, 40] {
            let pose = correction.map_from_body(&odom(index)).expect("set");
            assert_pose_close(&pose, &truth().compose(&odom(index)), 1e-9);
        }
    }

    #[test]
    fn a_wrong_first_fix_does_not_place_the_trajectory() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig::default());
        let off = Vector3::new(1.5, 0.0, 0.0);
        assert_eq!(
            fix_at(&mut correction, 0, &truth(), off),
            FixOutcome::Pending
        );
        assert_eq!(
            fix_at(&mut correction, 1, &truth(), Vector3::zeros()),
            FixOutcome::Pending
        );
        assert_eq!(
            fix_at(&mut correction, 2, &truth(), Vector3::zeros()),
            FixOutcome::Initialized
        );
        assert_pose_close(&correction.map_from_odom().unwrap(), &truth(), 1e-9);
    }

    #[test]
    fn an_outlier_is_rejected_and_leaves_the_offset_alone() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig::default());
        for index in 0..4 {
            fix_at(&mut correction, index, &truth(), Vector3::zeros());
        }
        assert_eq!(
            fix_at(&mut correction, 4, &truth(), Vector3::new(0.0, 2.0, 0.0)),
            FixOutcome::Rejected
        );
        assert_pose_close(&correction.map_from_odom().unwrap(), &truth(), 1e-9);
        assert_eq!(
            fix_at(&mut correction, 5, &truth(), Vector3::zeros()),
            FixOutcome::Accepted
        );
    }

    fn weak_fix_at(
        correction: &mut OdomCorrection,
        index: usize,
        shift: Vector3<f64>,
    ) -> FixOutcome {
        let odom_from_body = odom(index);
        let mut map_from_body = truth().compose(&odom_from_body);
        map_from_body.translation += shift;
        correction.add_weak_fix(index as i64 * SECOND / 10, &map_from_body, &odom_from_body)
    }

    #[test]
    fn weak_fixes_need_three_agreeing_and_only_set_the_first_offset() {
        // Two weak fixes do not set the offset; a third agreeing one does.
        let mut correction = OdomCorrection::new(OdomCorrectionConfig::default());
        assert_eq!(
            weak_fix_at(&mut correction, 0, Vector3::zeros()),
            FixOutcome::Pending
        );
        assert_eq!(
            weak_fix_at(&mut correction, 1, Vector3::zeros()),
            FixOutcome::Pending
        );
        assert_eq!(
            weak_fix_at(&mut correction, 2, Vector3::zeros()),
            FixOutcome::Initialized
        );
        assert_pose_close(&correction.map_from_odom().unwrap(), &truth(), 1e-9);

        // A strong fix after a weak one: still three, since one is weak.
        let mut mixed = OdomCorrection::new(OdomCorrectionConfig::default());
        assert_eq!(
            weak_fix_at(&mut mixed, 0, Vector3::zeros()),
            FixOutcome::Pending
        );
        assert_eq!(
            fix_at(&mut mixed, 1, &truth(), Vector3::zeros()),
            FixOutcome::Pending
        );
        assert_eq!(
            fix_at(&mut mixed, 2, &truth(), Vector3::zeros()),
            FixOutcome::Initialized
        );
        // Two strong fixes alone, as before.
        let mut strong = OdomCorrection::new(OdomCorrectionConfig::default());
        assert_eq!(
            fix_at(&mut strong, 0, &truth(), Vector3::zeros()),
            FixOutcome::Pending
        );
        assert_eq!(
            fix_at(&mut strong, 1, &truth(), Vector3::zeros()),
            FixOutcome::Initialized
        );

        // A wrong weak fix breaks the run; three agreeing ones after it set
        // the offset to theirs.
        let mut broken = OdomCorrection::new(OdomCorrectionConfig::default());
        let off = Vector3::new(1.5, 0.0, 0.0);
        assert_eq!(
            weak_fix_at(&mut broken, 0, Vector3::zeros()),
            FixOutcome::Pending
        );
        assert_eq!(weak_fix_at(&mut broken, 1, off), FixOutcome::Pending);
        assert_eq!(
            weak_fix_at(&mut broken, 2, Vector3::zeros()),
            FixOutcome::Pending
        );
        assert_eq!(
            weak_fix_at(&mut broken, 3, Vector3::zeros()),
            FixOutcome::Pending
        );
        assert_eq!(
            weak_fix_at(&mut broken, 4, Vector3::zeros()),
            FixOutcome::Initialized
        );
        assert_pose_close(&broken.map_from_odom().unwrap(), &truth(), 1e-9);

        // Once set, weak fixes are not used, however far off.
        let before = correction.map_from_odom().unwrap();
        assert_eq!(weak_fix_at(&mut correction, 3, off), FixOutcome::Unused);
        assert_eq!(
            weak_fix_at(&mut correction, 4, Vector3::zeros()),
            FixOutcome::Unused
        );
        assert_pose_close(&correction.map_from_odom().unwrap(), &before, 1e-12);
        assert_eq!(
            fix_at(&mut correction, 5, &truth(), Vector3::zeros()),
            FixOutcome::Accepted
        );
    }

    #[test]
    fn agreeing_rejected_fixes_reset_the_offset_after_an_odometry_jump() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig::default());
        for index in 0..4 {
            fix_at(&mut correction, index, &truth(), Vector3::zeros());
        }
        // The odometry jumped: from now on the true offset is a different one.
        let jumped = SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.75),
            Vector3::new(2.0, -1.0, 0.5),
        );
        assert_eq!(
            fix_at(&mut correction, 4, &jumped, Vector3::zeros()),
            FixOutcome::Rejected
        );
        assert_eq!(
            fix_at(&mut correction, 5, &jumped, Vector3::zeros()),
            FixOutcome::Rejected
        );
        assert_eq!(
            fix_at(&mut correction, 6, &jumped, Vector3::zeros()),
            FixOutcome::Reset
        );
        assert_pose_close(&correction.map_from_odom().unwrap(), &jumped, 1e-9);
    }

    #[test]
    fn the_gate_widens_with_travel_since_the_last_accepted_fix() {
        let config = OdomCorrectionConfig::default();
        let mut near = OdomCorrection::new(config.clone());
        let mut far = OdomCorrection::new(config);
        for index in 0..3 {
            fix_at(&mut near, index, &truth(), Vector3::zeros());
            fix_at(&mut far, index, &truth(), Vector3::zeros());
        }
        // 0.3 m off: beyond the 0.15 m base gate right after a fix, inside it
        // after several metres of travel (sampled by an outlier fix on the way).
        let drift = Vector3::new(0.0, 0.0, 0.3);
        assert_eq!(fix_at(&mut near, 3, &truth(), drift), FixOutcome::Rejected);
        assert_eq!(
            fix_at(&mut far, 20, &truth(), Vector3::new(5.0, 0.0, 0.0)),
            FixOutcome::Rejected
        );
        assert_eq!(fix_at(&mut far, 40, &truth(), drift), FixOutcome::Accepted);
    }

    #[test]
    fn noisy_fixes_average_within_the_window() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig::default());
        // Alternating +-2 cm noise along x averages out.
        for index in 0..20 {
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            fix_at(
                &mut correction,
                index,
                &truth(),
                Vector3::new(0.02 * sign, 0.0, 0.0),
            );
        }
        let estimate = correction.map_from_odom().unwrap();
        assert!((estimate.translation - truth().translation).norm() < 0.005);
        // A 5 s window at 10 fixes per second keeps all 20 fixes.
        assert_eq!(correction.window.len(), 20);
    }

    #[test]
    fn old_fixes_leave_the_window() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig {
            window_s: 1.0,
            ..OdomCorrectionConfig::default()
        });
        for index in 0..30 {
            fix_at(&mut correction, index, &truth(), Vector3::zeros());
        }
        // 0.1 s apart: the newest fix and the ten before it are within 1 s.
        assert_eq!(correction.window.len(), 11);
    }
}
