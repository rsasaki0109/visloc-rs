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
//!   so a single wrong fix cannot place the trajectory; with
//!   `init_min_travel_m` or `init_min_turn_deg` set, once that many fixes of
//!   the last `init_window_s` agree *and* the odometry moved that far or
//!   turned that much between them. Fixes from one viewpoint can be wrong
//!   together (a view mostly hidden by people leaves one patch, and every
//!   image of it gives the same wrong pose); a bias that depends on the view
//!   does not survive a change of view, the truth does;
//! - **accepts** a fix whose map pose lies within a gate of where the current
//!   offset puts the odometry pose; the gate widens with the odometry
//!   distance travelled since the last accepted fix, to absorb drift;
//!   that distance is the odometry's path through the poses passed to
//!   [`OdomCorrection::observe_odometry`], or, without them, the straight
//!   lines between the odometry positions of consecutive fixes, which
//!   undercount a path that turns or doubles back;
//! - **rejects** any other fix, and **resets** the offset when `reset_after`
//!   consecutive rejected fixes agree with each other (the odometry itself
//!   jumped, or the earlier offset was wrong).
//!
//! A *weak* fix ([`OdomCorrection::add_weak_fix`]) is a localization with
//! fewer inliers than a fix needs, which is more often wrong. Weak fixes help
//! to set the first offset, and then `init_fixes_weak` (more than
//! `init_fixes`) consecutive fixes must agree. Under lighting the map was not
//! built in, the first images that localize at all often have too few
//! inliers for a fix. Once the offset is set, weak fixes are not used by
//! default; with `weak_gate_m` set, one that lies that close to where the
//! offset puts the odometry pose is accepted, which keeps the offset
//! following the odometry's drift where full fixes are rare. It counts
//! toward a reset only with `weak_reset_after` set: that many consecutive
//! weak or rejected fixes that agree with each other replace the offset,
//! as weak fixes set the first one (an odometry that drifted metres while
//! no full fix came leaves its weak fixes outside any gate).
//!
//! The offset is the mean over the accepted fixes of the last `window_s`
//! seconds (circular mean for the yaw), which averages out localization noise
//! while following slow odometry drift.

use std::collections::VecDeque;

/// How far back [`OdomCorrection::observe_odometry`] keeps the odometry's
/// path. A fix older than that, relative to the newest odometry pose, falls
/// back to the straight-line distance.
const PATH_HISTORY_NS: i64 = 30_000_000_000;

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
    /// Once the offset is set, a weak fix within this distance of where the
    /// offset puts the odometry pose (and within `weak_gate_yaw_deg`) is
    /// accepted like a fix; 0, the default, leaves weak fixes unused then.
    /// Unlike a fix's gate, it does not widen with travel.
    pub weak_gate_m: f64,
    /// Yaw gate, in degrees, for accepting a weak fix (`weak_gate_m`).
    pub weak_gate_yaw_deg: f64,
    /// Once the offset is set, this many consecutive fixes that it rejects,
    /// weak ones among them, replace it when they agree with each other
    /// (rejected full fixes alone still need `reset_after`). 0, the default:
    /// weak fixes never reset the offset.
    pub weak_reset_after: usize,
    /// When either is positive, the fixes that set the first offset must
    /// span this much odometry travel (m) or turn (deg), between any two of
    /// them; they are then drawn from the fixes of the last `init_window_s`,
    /// not only consecutive ones. 0 and 0, the default: consecutive fixes,
    /// no span.
    pub init_min_travel_m: f64,
    pub init_min_turn_deg: f64,
    /// How far back the fixes that may set the first offset reach, with a
    /// span required or a warm-up.
    pub init_window_s: f64,
    /// When positive, the first offset waits until fixes have come in for
    /// this long, then grows from the strongest fix of the last
    /// `init_window_s` (by the score given with
    /// [`OdomCorrection::add_scored_fix`], e.g. its inlier count): the fixes
    /// that agree with it set the offset once there are `init_fixes` of
    /// them; failing that, the next strongest is tried. Fixes from a view
    /// mostly hidden by people can agree on a wrong pose, and they are
    /// weaker than the fixes of a clear view. 0, the default: the first
    /// agreeing consecutive fixes.
    pub init_warmup_s: f64,
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
            weak_gate_m: 0.0,
            weak_gate_yaw_deg: 2.0,
            weak_reset_after: 0,
            init_min_travel_m: 0.0,
            init_min_turn_deg: 0.0,
            init_window_s: 10.0,
            init_warmup_s: 0.0,
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
    /// The body's odometry orientation (for the initial span).
    odom_rotation: UnitQuaternion<f64>,
    /// Yaw of `map_from_body * odom_from_body^-1`, the offset this fix alone
    /// implies.
    yaw: f64,
    weak: bool,
    /// Strength (e.g. inliers) for `init_warmup_s`; 0 when not given.
    score: f64,
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
    /// Observed odometry, oldest first: stamp and path length from the first
    /// observed pose. Covers at most `PATH_HISTORY_NS`.
    path: VecDeque<(i64, f64)>,
    /// Position of the newest observed odometry pose.
    path_end: Option<Vector3<f64>>,
    /// Path length at the last accepted fix, when the path covered it.
    path_at_accept_m: Option<f64>,
    /// Stamp of the first fix before the first offset (`init_warmup_s`).
    init_first_ns: Option<i64>,
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
            path: VecDeque::new(),
            path_end: None,
            path_at_accept_m: None,
            init_first_ns: None,
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

    /// Adds an odometry pose, so the gate measures the odometry's travel
    /// along its path rather than between fixes. Poses must arrive in time
    /// order (older or repeated stamps are ignored), but may run ahead of the
    /// fixes: a fix's travel is read off the path at its own stamp.
    pub fn observe_odometry(&mut self, timestamp_ns: i64, odom_from_body: &SE3) {
        if self
            .path
            .back()
            .is_some_and(|(last_ns, _)| timestamp_ns <= *last_ns)
        {
            return;
        }
        let position = odom_from_body.translation;
        let length = match (self.path.back(), self.path_end) {
            (Some((_, length)), Some(end)) => length + (position - end).norm(),
            _ => 0.0,
        };
        self.path.push_back((timestamp_ns, length));
        self.path_end = Some(position);
        while self
            .path
            .front()
            .is_some_and(|(first_ns, _)| timestamp_ns - first_ns > PATH_HISTORY_NS)
        {
            self.path.pop_front();
        }
    }

    /// Path length of the observed odometry at `timestamp_ns`, interpolated
    /// between the poses around it; `None` outside the observed path.
    fn path_length_at(&self, timestamp_ns: i64) -> Option<f64> {
        let (first_ns, _) = self.path.front()?;
        let (last_ns, _) = self.path.back()?;
        if timestamp_ns < *first_ns || timestamp_ns > *last_ns {
            return None;
        }
        let index = self
            .path
            .partition_point(|(stamp, _)| *stamp < timestamp_ns);
        let (after_ns, after) = self.path[index];
        if after_ns == timestamp_ns || index == 0 {
            return Some(after);
        }
        let (before_ns, before) = self.path[index - 1];
        let alpha = (timestamp_ns - before_ns) as f64 / (after_ns - before_ns) as f64;
        Some(before + alpha * (after - before))
    }

    /// Adds a map fix: the body's map pose from localization, and its
    /// odometry pose at the same instant. Fixes must arrive in time order.
    pub fn add_fix(
        &mut self,
        timestamp_ns: i64,
        map_from_body: &SE3,
        odom_from_body: &SE3,
    ) -> FixOutcome {
        self.add(timestamp_ns, map_from_body, odom_from_body, false, 0.0)
    }

    /// [`Self::add_fix`] with a strength, e.g. the localization's inlier
    /// count, which `init_warmup_s` ranks the first fixes by.
    pub fn add_scored_fix(
        &mut self,
        timestamp_ns: i64,
        map_from_body: &SE3,
        odom_from_body: &SE3,
        score: f64,
    ) -> FixOutcome {
        self.add(timestamp_ns, map_from_body, odom_from_body, false, score)
    }

    /// [`Self::add_weak_fix`] with a strength (see [`Self::add_scored_fix`]).
    pub fn add_scored_weak_fix(
        &mut self,
        timestamp_ns: i64,
        map_from_body: &SE3,
        odom_from_body: &SE3,
        score: f64,
    ) -> FixOutcome {
        self.add(timestamp_ns, map_from_body, odom_from_body, true, score)
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
        self.add(timestamp_ns, map_from_body, odom_from_body, true, 0.0)
    }

    fn add(
        &mut self,
        timestamp_ns: i64,
        map_from_body: &SE3,
        odom_from_body: &SE3,
        weak: bool,
        score: f64,
    ) -> FixOutcome {
        let fix = Fix {
            timestamp_ns,
            map_position: map_from_body.translation,
            odom_position: odom_from_body.translation,
            odom_rotation: odom_from_body.rotation,
            yaw: yaw_of(&(map_from_body.rotation * odom_from_body.rotation.inverse())),
            weak,
            score,
        };
        if let Some(previous) = self.previous_odom_position {
            self.travel_since_accept_m += (fix.odom_position - previous).norm();
        }
        self.previous_odom_position = Some(fix.odom_position);
        let path_m = self.path_length_at(timestamp_ns);

        let Some(offset) = self.offset else {
            return if self.push_initial(fix) {
                Self::adopt(&mut self.window, &mut self.candidates);
                self.refit();
                self.restart_travel(path_m);
                FixOutcome::Initialized
            } else {
                FixOutcome::Pending
            };
        };

        let (position_residual, yaw_residual) = residuals(&offset, &fix);
        if weak {
            if self.config.weak_gate_m > 0.0
                && position_residual <= self.config.weak_gate_m
                && yaw_residual <= self.config.weak_gate_yaw_deg
            {
                self.candidates.clear();
                self.window.push_back(fix);
                self.refit();
                self.restart_travel(path_m);
                return FixOutcome::Accepted;
            }
            if self.config.weak_reset_after == 0 {
                return FixOutcome::Unused;
            }
            return if self.push_rejected(fix) {
                Self::adopt(&mut self.window, &mut self.candidates);
                self.refit();
                self.restart_travel(path_m);
                FixOutcome::Reset
            } else {
                FixOutcome::Unused
            };
        }
        let travel = match (path_m, self.path_at_accept_m) {
            (Some(now), Some(then)) => self.travel_since_accept_m.max(now - then),
            _ => self.travel_since_accept_m,
        };
        let accepted = position_residual <= self.config.gate_m + self.config.gate_per_m * travel
            && yaw_residual <= self.config.gate_yaw_deg + self.config.gate_yaw_per_m_deg * travel;
        if accepted {
            self.candidates.clear();
            self.window.push_back(fix);
            self.refit();
            self.restart_travel(path_m);
            return FixOutcome::Accepted;
        }
        if self.push_rejected(fix) {
            Self::adopt(&mut self.window, &mut self.candidates);
            self.refit();
            self.restart_travel(path_m);
            FixOutcome::Reset
        } else {
            FixOutcome::Rejected
        }
    }

    /// Measures travel from a fix just accepted, at path length `path_m`.
    fn restart_travel(&mut self, path_m: Option<f64>) {
        self.travel_since_accept_m = 0.0;
        self.path_at_accept_m = path_m;
    }

    /// Appends a fix the offset rejects to the candidate run and reports
    /// whether it now replaces the offset: the last `reset_after` fixes, all
    /// full, agree, or (with `weak_reset_after`) the last `weak_reset_after`
    /// do. The run then keeps only the agreeing fixes.
    fn push_rejected(&mut self, fix: Fix) -> bool {
        let weak_needed = self.config.weak_reset_after;
        if weak_needed == 0 {
            return self.push_candidate(fix, self.config.reset_after);
        }
        let strong_needed = self.config.reset_after.max(1);
        self.candidates.push(fix);
        let keep = strong_needed.max(weak_needed);
        if self.candidates.len() > keep {
            self.candidates.drain(..self.candidates.len() - keep);
        }
        for (needed, all_full) in [(strong_needed, true), (weak_needed, false)] {
            let Some(start) = self.candidates.len().checked_sub(needed) else {
                continue;
            };
            let run = &self.candidates[start..];
            if (!all_full || run.iter().all(|fix| !fix.weak)) && self.agree(run) {
                self.candidates.drain(..start);
                return true;
            }
        }
        false
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

    /// Whether the initial span is required.
    fn spans_required(&self) -> bool {
        self.config.init_min_travel_m > 0.0 || self.config.init_min_turn_deg > 0.0
    }

    /// Whether `fixes` span the required odometry travel or turn.
    fn spans(&self, fixes: &[Fix]) -> bool {
        let travel = self.config.init_min_travel_m;
        let turn = self.config.init_min_turn_deg.to_radians();
        fixes.iter().enumerate().any(|(i, a)| {
            fixes[i + 1..].iter().any(|b| {
                (travel > 0.0 && (a.odom_position - b.odom_position).norm() >= travel)
                    || (turn > 0.0 && a.odom_rotation.angle_to(&b.odom_rotation) >= turn)
            })
        })
    }

    /// With a warm-up: once the first fix is `init_warmup_s` old, the fixes
    /// of the last `init_window_s` (at least the warm-up) are tried as seeds,
    /// strongest first (newest among equals). A seed's group is the fixes
    /// that agree with it, the one agreeing least with their joint offset
    /// dropped until all agree; the first group with enough fixes
    /// (`init_fixes`, or `init_fixes_weak` with a weak one) sets the offset.
    fn push_initial_strongest(&mut self, fix: Fix) -> bool {
        let warmup_ns = (self.config.init_warmup_s * 1e9) as i64;
        let window_ns = ((self.config.init_window_s * 1e9) as i64).max(warmup_ns);
        let newest_ns = fix.timestamp_ns;
        let first_ns = *self.init_first_ns.get_or_insert(newest_ns);
        self.candidates.push(fix);
        self.candidates
            .retain(|candidate| newest_ns - candidate.timestamp_ns <= window_ns);
        if newest_ns - first_ns < warmup_ns {
            return false;
        }
        let strong = self.config.init_fixes.max(1);
        let weak = self.config.init_fixes_weak.max(strong);
        let mut seeds: Vec<usize> = (0..self.candidates.len()).collect();
        seeds.sort_by(|&a, &b| {
            self.candidates[b]
                .score
                .total_cmp(&self.candidates[a].score)
                .then(b.cmp(&a))
        });
        for seed in seeds {
            let seed_fix = self.candidates[seed].clone();
            let mut group: Vec<Fix> = self
                .candidates
                .iter()
                .filter(|candidate| self.agree(&[(*candidate).clone(), seed_fix.clone()]))
                .cloned()
                .collect();
            loop {
                let needed = if group.iter().any(|fix| fix.weak) {
                    weak
                } else {
                    strong
                };
                if group.len() < needed {
                    break;
                }
                if self.agree(&group) {
                    self.candidates = group;
                    return true;
                }
                let offset = fit(&group.iter().collect::<Vec<_>>());
                let worst = group
                    .iter()
                    .enumerate()
                    .filter(|(_, fix)| fix.timestamp_ns != seed_fix.timestamp_ns)
                    .max_by(|(_, a), (_, b)| {
                        residuals(&offset, a).0.total_cmp(&residuals(&offset, b).0)
                    })
                    .map(|(index, _)| index);
                match worst {
                    Some(index) => {
                        group.remove(index);
                    }
                    None => break,
                }
            }
        }
        false
    }

    /// With a span required: among the fixes of the last `init_window_s`,
    /// those that agree with the newest; the one that agrees least with
    /// their joint offset is dropped until all agree. They set the offset if
    /// there are enough (`init_fixes`, or `init_fixes_weak` with a weak one)
    /// and they span the required travel or turn; the candidates then keep
    /// only them.
    fn push_initial_spread(&mut self, fix: Fix) -> bool {
        let window_ns = (self.config.init_window_s * 1e9) as i64;
        let newest_ns = fix.timestamp_ns;
        self.candidates.push(fix);
        self.candidates
            .retain(|candidate| newest_ns - candidate.timestamp_ns <= window_ns);
        let newest = self.candidates.last().expect("just pushed").clone();
        let mut group: Vec<Fix> = self
            .candidates
            .iter()
            .filter(|candidate| self.agree(&[(*candidate).clone(), newest.clone()]))
            .cloned()
            .collect();
        let strong = self.config.init_fixes.max(1);
        let weak = self.config.init_fixes_weak.max(strong);
        loop {
            let needed = if group.iter().any(|fix| fix.weak) {
                weak
            } else {
                strong
            };
            if group.len() < needed {
                return false;
            }
            if self.agree(&group) {
                if !self.spans(&group) {
                    return false;
                }
                self.candidates = group;
                return true;
            }
            let offset = fit(&group.iter().collect::<Vec<_>>());
            let worst = group
                .iter()
                .enumerate()
                .filter(|(_, fix)| fix.timestamp_ns != newest_ns)
                .max_by(|(_, a), (_, b)| {
                    residuals(&offset, a).0.total_cmp(&residuals(&offset, b).0)
                })
                .map(|(index, _)| index);
            match worst {
                Some(index) => {
                    group.remove(index);
                }
                None => return false,
            }
        }
    }

    /// Appends to the candidate run before the first offset and reports
    /// whether its newest fixes now set it: `init_fixes` of them agreeing, or
    /// `init_fixes_weak` when any of those is weak. On success the run keeps
    /// only those.
    fn push_initial(&mut self, fix: Fix) -> bool {
        if self.config.init_warmup_s > 0.0 {
            return self.push_initial_strongest(fix);
        }
        if self.spans_required() {
            return self.push_initial_spread(fix);
        }
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

    /// Ten fixes from one viewpoint, all off by the same 20 cm, then the
    /// body moves 10 cm per fix with exact fixes.
    fn biased_start(correction: &mut OdomCorrection) -> Vec<FixOutcome> {
        let map_from_odom = truth();
        let mut outcomes = Vec::new();
        for index in 0..30_i64 {
            let moved = (index - 9).max(0) as f64 * 0.1;
            let odom_from_body = SE3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.2),
                Vector3::new(moved, 0.0, 0.0),
            );
            let mut map_from_body = map_from_odom.compose(&odom_from_body);
            if index < 10 {
                map_from_body.translation += Vector3::new(0.2, 0.0, 0.0);
            }
            outcomes.push(correction.add_fix(index * SECOND / 10, &map_from_body, &odom_from_body));
        }
        outcomes
    }

    #[test]
    fn agreeing_fixes_from_one_viewpoint_set_a_biased_offset_by_default() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig::default());
        let outcomes = biased_start(&mut correction);
        assert_eq!(outcomes[1], FixOutcome::Initialized);
        // The exact fixes then disagree with the biased offset, until three
        // in a row reset it.
        assert!(outcomes[10..].contains(&FixOutcome::Reset));
    }

    #[test]
    fn a_required_span_waits_for_fixes_that_agree_across_motion() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig {
            init_min_travel_m: 0.3,
            ..OdomCorrectionConfig::default()
        });
        let outcomes = biased_start(&mut correction);
        // The biased fixes never span 0.3 m; the exact ones (from 0.1 m on)
        // do at the fourth.
        assert!(
            outcomes[..10].iter().all(|o| *o == FixOutcome::Pending),
            "{outcomes:?}"
        );
        let initialized = outcomes
            .iter()
            .position(|o| *o == FixOutcome::Initialized)
            .expect("initialized");
        assert_eq!(initialized, 13, "{outcomes:?}");
        assert!(!outcomes.contains(&FixOutcome::Reset));
        assert!(!outcomes.contains(&FixOutcome::Rejected));
        let odom_from_body = SE3::new(UnitQuaternion::identity(), Vector3::new(5.0, 1.0, 0.0));
        assert_pose_close(
            &correction.map_from_body(&odom_from_body).expect("set"),
            &truth().compose(&odom_from_body),
            1e-9,
        );
    }

    #[test]
    fn a_required_turn_counts_rotation_too() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig {
            init_min_turn_deg: 10.0,
            ..OdomCorrectionConfig::default()
        });
        let map_from_odom = truth();
        let mut outcomes = Vec::new();
        // Turning in place by 4 degrees per fix, exact fixes.
        for index in 0..5 {
            let odom_from_body = SE3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, (4.0 * index as f64).to_radians()),
                Vector3::new(1.0, 2.0, 0.0),
            );
            outcomes.push(correction.add_fix(
                index as i64 * SECOND / 10,
                &map_from_odom.compose(&odom_from_body),
                &odom_from_body,
            ));
        }
        assert_eq!(
            outcomes,
            vec![
                FixOutcome::Pending,
                FixOutcome::Pending,
                FixOutcome::Pending,
                FixOutcome::Initialized,
                FixOutcome::Accepted
            ]
        );
    }

    /// A still camera: every third fix exact and strong (50), the others
    /// off by the same 20 cm and weaker (25).
    fn mixed_still_start(correction: &mut OdomCorrection) -> Vec<FixOutcome> {
        let map_from_odom = truth();
        let odom_from_body = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.2),
            Vector3::new(0.5, 0.0, 0.0),
        );
        (0..20_i64)
            .map(|index| {
                let exact = index % 3 == 0;
                let mut map_from_body = map_from_odom.compose(&odom_from_body);
                if !exact {
                    map_from_body.translation += Vector3::new(0.2, 0.0, 0.0);
                }
                correction.add_scored_fix(
                    index * SECOND / 10,
                    &map_from_body,
                    &odom_from_body,
                    if exact { 50.0 } else { 25.0 },
                )
            })
            .collect()
    }

    #[test]
    fn a_warm_up_starts_from_the_strongest_fixes() {
        let at = SE3::new(UnitQuaternion::identity(), Vector3::new(2.0, 1.0, 0.0));
        // By default the first two (biased) fixes that agree set it.
        let mut plain = OdomCorrection::new(OdomCorrectionConfig::default());
        let outcomes = mixed_still_start(&mut plain);
        assert_eq!(outcomes[2], FixOutcome::Initialized);
        // With a 1 s warm-up, it waits, then grows from the strongest fixes,
        // which are the exact ones; a still camera is no obstacle.
        let mut warm = OdomCorrection::new(OdomCorrectionConfig {
            init_warmup_s: 1.0,
            ..OdomCorrectionConfig::default()
        });
        let outcomes = mixed_still_start(&mut warm);
        assert!(outcomes[..10].iter().all(|o| *o == FixOutcome::Pending));
        assert_eq!(outcomes[10], FixOutcome::Initialized, "{outcomes:?}");
        assert_pose_close(
            &warm.map_from_body(&at).expect("set"),
            &truth().compose(&at),
            1e-9,
        );
    }

    #[test]
    fn a_strong_lone_fix_does_not_block_the_start() {
        // One strong fix far off, then agreeing weaker exact ones.
        let mut correction = OdomCorrection::new(OdomCorrectionConfig {
            init_warmup_s: 0.5,
            ..OdomCorrectionConfig::default()
        });
        let map_from_odom = truth();
        let odom_from_body = odom(0);
        let mut outcomes = Vec::new();
        for index in 0..8_i64 {
            let mut map_from_body = map_from_odom.compose(&odom_from_body);
            let score = if index == 0 {
                map_from_body.translation += Vector3::new(1.0, 0.0, 0.0);
                90.0
            } else {
                30.0
            };
            outcomes.push(correction.add_scored_fix(
                index * SECOND / 10,
                &map_from_body,
                &odom_from_body,
                score,
            ));
        }
        assert_eq!(outcomes[5], FixOutcome::Initialized, "{outcomes:?}");
        assert_pose_close(
            &correction.map_from_body(&odom_from_body).expect("set"),
            &truth().compose(&odom_from_body),
            1e-9,
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
    fn a_weak_gate_accepts_close_weak_fixes_after_the_first_offset() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig {
            weak_gate_m: 0.05,
            ..OdomCorrectionConfig::default()
        });
        for index in 0..2 {
            fix_at(&mut correction, index, &truth(), Vector3::zeros());
        }
        assert!(correction.is_set());
        // 3 cm off: inside the weak gate, and it moves the offset.
        let close = Vector3::new(0.03, 0.0, 0.0);
        assert_eq!(weak_fix_at(&mut correction, 2, close), FixOutcome::Accepted);
        let moved = correction.map_from_odom().unwrap().translation - truth().translation;
        assert!((moved - close / 3.0).norm() < 1e-9, "{moved:?}");
        // 8 cm off: inside a fix's gate, outside the weak one.
        assert_eq!(
            weak_fix_at(&mut correction, 3, Vector3::new(0.08, 0.0, 0.0)),
            FixOutcome::Unused
        );
    }

    #[test]
    fn agreeing_weak_fixes_reset_a_drifted_offset_only_when_asked() {
        // The odometry drifted 2 m while no full fix came: its weak fixes all
        // lie outside the weak gate and agree on the new offset.
        let drifted = Vector3::new(2.0, 0.0, 0.0);
        for weak_reset_after in [0, 3] {
            let mut correction = OdomCorrection::new(OdomCorrectionConfig {
                weak_gate_m: 0.05,
                weak_reset_after,
                ..OdomCorrectionConfig::default()
            });
            for index in 0..2 {
                fix_at(&mut correction, index, &truth(), Vector3::zeros());
            }
            let outcomes: Vec<_> = (2..5)
                .map(|index| weak_fix_at(&mut correction, index, drifted))
                .collect();
            let moved = correction.map_from_odom().unwrap().translation - truth().translation;
            if weak_reset_after == 0 {
                assert_eq!(outcomes, [FixOutcome::Unused; 3]);
                assert!(moved.norm() < 1e-9, "{moved:?}");
            } else {
                assert_eq!(
                    outcomes,
                    [FixOutcome::Unused, FixOutcome::Unused, FixOutcome::Reset]
                );
                assert!((moved - drifted).norm() < 1e-9, "{moved:?}");
            }
        }
    }

    #[test]
    fn weak_fixes_that_disagree_do_not_reset_the_offset() {
        let mut correction = OdomCorrection::new(OdomCorrectionConfig {
            weak_reset_after: 3,
            ..OdomCorrectionConfig::default()
        });
        for index in 0..2 {
            fix_at(&mut correction, index, &truth(), Vector3::zeros());
        }
        for (index, shift) in [(2, 2.0), (3, -1.0), (4, 2.0), (5, 0.5)] {
            assert_eq!(
                weak_fix_at(&mut correction, index, Vector3::new(shift, 0.0, 0.0)),
                FixOutcome::Unused
            );
        }
        assert_pose_close(&correction.map_from_odom().unwrap(), &truth(), 1e-9);
        // Two rejected full fixes and one weak one that agree: a reset with
        // `weak_reset_after`, though two full ones alone are not enough.
        let shift = Vector3::new(0.0, 3.0, 0.0);
        let jumped = SE3::new(truth().rotation, truth().translation + shift);
        let mut mixed = OdomCorrection::new(OdomCorrectionConfig {
            weak_reset_after: 3,
            ..OdomCorrectionConfig::default()
        });
        for index in 0..2 {
            fix_at(&mut mixed, index, &truth(), Vector3::zeros());
        }
        assert_eq!(
            fix_at(&mut mixed, 2, &jumped, Vector3::zeros()),
            FixOutcome::Rejected
        );
        assert_eq!(
            fix_at(&mut mixed, 3, &jumped, Vector3::zeros()),
            FixOutcome::Rejected
        );
        assert_eq!(weak_fix_at(&mut mixed, 4, shift), FixOutcome::Reset);
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
    fn observed_odometry_measures_travel_along_the_path() {
        let config = OdomCorrectionConfig::default();
        let mut sampled = OdomCorrection::new(config.clone());
        let mut observed = OdomCorrection::new(config);
        // The odometry runs ahead of the fixes, as a localization finishes
        // after later odometry has arrived.
        for index in 0..=40 {
            observed.observe_odometry(index as i64 * SECOND / 10, &odom(index));
        }
        for index in 0..3 {
            fix_at(&mut sampled, index, &truth(), Vector3::zeros());
            fix_at(&mut observed, index, &truth(), Vector3::zeros());
        }
        // Almost a full circle (11.6 m of path) ends 1.1 m from the last
        // fix: 0.3 m off is outside the gate that straight line gives, and
        // inside the one the path gives.
        let drift = Vector3::new(0.0, 0.0, 0.3);
        assert_eq!(
            fix_at(&mut sampled, 31, &truth(), drift),
            FixOutcome::Rejected
        );
        assert_eq!(
            fix_at(&mut observed, 31, &truth(), drift),
            FixOutcome::Accepted
        );
        // Travel restarts at the accepted fix.
        assert_eq!(
            fix_at(&mut observed, 32, &truth(), drift * 2.0),
            FixOutcome::Rejected
        );
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
