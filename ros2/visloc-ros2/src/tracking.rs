//! Detects VIO that has lost tracking, for `visloc_vio_node`.
//!
//! When the cameras see too little (a blank wall up close, a covered lens,
//! the lights off), the VIO estimator integrates the IMU alone and its pose
//! runs away while the robot may be standing still. A robot that keeps
//! driving on that odometry is blind. [`TrackingMonitor`] watches each
//! estimate: too few tracked observations for too long, or an implausible
//! speed, makes it [`TrackingVerdict::Lost`]. The node then resets the
//! estimator and stops publishing odometry, so the navigation stack loses
//! the robot's pose and stops it. Once tracking holds again for
//! `recover_after_s`, the node publishes again, in a new odometry frame,
//! and announces the reset (`odom_reset`) so that map localization
//! relocalizes instead of applying its old `map -> odom`.

/// When tracking counts as lost. Both checks off (the default) disables the
/// monitor: every estimate is published, as before.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackingConfig {
    /// Fewer tracked observations than this in a frame is weak tracking
    /// (0: off).
    pub min_observations: usize,
    /// Weak tracking for this long is lost tracking (seconds).
    pub lost_after_s: f64,
    /// After a reset, tracking must hold this long before odometry is
    /// published again (seconds).
    pub recover_after_s: f64,
    /// A speed above this (m/s) is lost tracking at once (0: off).
    pub max_speed_m_s: f64,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            min_observations: 0,
            lost_after_s: 0.5,
            recover_after_s: 1.0,
            max_speed_m_s: 0.0,
        }
    }
}

impl TrackingConfig {
    /// Whether any check is on.
    pub fn enabled(&self) -> bool {
        self.min_observations > 0 || self.max_speed_m_s > 0.0
    }
}

/// Why tracking counts as lost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LostReason {
    /// Fewer than `min_observations` for `for_s` seconds.
    FewObservations { observations: usize, for_s: f64 },
    /// Faster than `max_speed_m_s`.
    Speed { speed_m_s: f64 },
}

impl std::fmt::Display for LostReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FewObservations {
                observations,
                for_s,
            } => write!(
                f,
                "{observations} tracked observations, weak for {for_s:.2} s"
            ),
            Self::Speed { speed_m_s } => write!(f, "speed {speed_m_s:.2} m/s"),
        }
    }
}

/// What to do with one estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrackingVerdict {
    /// Tracking holds: publish it.
    Publish,
    /// Recovering after a reset: do not publish yet.
    Hold,
    /// Tracking is lost: reset the estimator, publish nothing.
    Lost(LostReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Tracking,
    Recovering,
}

/// The lost-tracking state machine; see the module docs.
#[derive(Clone, Debug)]
pub struct TrackingMonitor {
    config: TrackingConfig,
    state: State,
    weak_since_ns: Option<i64>,
    good_since_ns: Option<i64>,
}

impl TrackingMonitor {
    /// A monitor that waits for good tracking before the first publication
    /// (when enabled).
    pub fn new(config: TrackingConfig) -> Self {
        Self {
            config,
            state: State::Recovering,
            weak_since_ns: None,
            good_since_ns: None,
        }
    }

    pub const fn config(&self) -> &TrackingConfig {
        &self.config
    }

    /// Whether estimates are being published.
    pub fn is_tracking(&self) -> bool {
        !self.config.enabled() || self.state == State::Tracking
    }

    /// The estimator was reset for another reason (an estimator error):
    /// wait for good tracking again.
    pub fn restart(&mut self) {
        self.state = State::Recovering;
        self.weak_since_ns = None;
        self.good_since_ns = None;
    }

    /// Judges the estimate at `stamp_ns` with `observations` tracked
    /// observations and speed `speed_m_s`.
    pub fn update(
        &mut self,
        stamp_ns: i64,
        observations: usize,
        speed_m_s: f64,
    ) -> TrackingVerdict {
        if !self.config.enabled() {
            return TrackingVerdict::Publish;
        }
        if self.config.max_speed_m_s > 0.0
            && (speed_m_s.is_nan() || speed_m_s > self.config.max_speed_m_s)
        {
            self.restart();
            return TrackingVerdict::Lost(LostReason::Speed { speed_m_s });
        }
        if observations < self.config.min_observations {
            self.good_since_ns = None;
            let since = *self.weak_since_ns.get_or_insert(stamp_ns);
            let weak_s = (stamp_ns - since) as f64 * 1e-9;
            if weak_s >= self.config.lost_after_s {
                self.restart();
                return TrackingVerdict::Lost(LostReason::FewObservations {
                    observations,
                    for_s: weak_s,
                });
            }
            // A short dip while tracking is still published.
            return match self.state {
                State::Tracking => TrackingVerdict::Publish,
                State::Recovering => TrackingVerdict::Hold,
            };
        }
        self.weak_since_ns = None;
        if self.state == State::Recovering {
            let since = *self.good_since_ns.get_or_insert(stamp_ns);
            if ((stamp_ns - since) as f64 * 1e-9) < self.config.recover_after_s {
                return TrackingVerdict::Hold;
            }
            self.state = State::Tracking;
            self.good_since_ns = None;
        }
        TrackingVerdict::Publish
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    fn config() -> TrackingConfig {
        TrackingConfig {
            min_observations: 20,
            lost_after_s: 0.5,
            recover_after_s: 1.0,
            max_speed_m_s: 2.0,
        }
    }

    /// Feeds frames every 50 ms from `start` and returns the verdicts.
    fn feed(
        monitor: &mut TrackingMonitor,
        start_ms: i64,
        frames: &[(usize, f64)],
    ) -> Vec<TrackingVerdict> {
        frames
            .iter()
            .enumerate()
            .map(|(i, &(n, v))| monitor.update((start_ms + 50 * i as i64) * MS, n, v))
            .collect()
    }

    #[test]
    fn disabled_publishes_everything() {
        let mut monitor = TrackingMonitor::new(TrackingConfig::default());
        assert!(!monitor.config().enabled());
        assert!(feed(&mut monitor, 0, &[(0, 100.0); 40])
            .iter()
            .all(|v| *v == TrackingVerdict::Publish));
        assert!(monitor.is_tracking());
    }

    #[test]
    fn holds_until_tracking_is_good_for_recover_after() {
        let mut monitor = TrackingMonitor::new(config());
        let verdicts = feed(&mut monitor, 0, &[(100, 0.1); 25]);
        // 0 .. 950 ms held, published from 1000 ms on.
        assert!(verdicts[..20].iter().all(|v| *v == TrackingVerdict::Hold));
        assert!(verdicts[20..]
            .iter()
            .all(|v| *v == TrackingVerdict::Publish));
        assert!(monitor.is_tracking());
    }

    #[test]
    fn short_dips_are_published_long_ones_are_lost() {
        let mut monitor = TrackingMonitor::new(config());
        feed(&mut monitor, 0, &[(100, 0.1); 21]);
        assert!(monitor.is_tracking());
        // 0.45 s weak (9 frames, 0 .. 400 ms in), then good: still tracking.
        let dip = feed(&mut monitor, 2000, &[(5, 0.1); 9]);
        assert!(dip.iter().all(|v| *v == TrackingVerdict::Publish));
        assert_eq!(
            monitor.update(2450 * MS, 100, 0.1),
            TrackingVerdict::Publish
        );
        // Weak from 3000 ms: lost at 3500 ms.
        let verdicts = feed(&mut monitor, 3000, &[(5, 0.1); 11]);
        assert!(verdicts[..10]
            .iter()
            .all(|v| *v == TrackingVerdict::Publish));
        assert!(matches!(
            verdicts[10],
            TrackingVerdict::Lost(LostReason::FewObservations {
                observations: 5,
                ..
            })
        ));
        assert!(!monitor.is_tracking());
        // Blank for a while after the reset: lost again every 0.5 s, never published.
        let blank = feed(&mut monitor, 3550, &[(0, 0.0); 30]);
        assert!(blank.iter().all(|v| *v != TrackingVerdict::Publish));
        assert!(blank.iter().any(|v| matches!(v, TrackingVerdict::Lost(_))));
        // Good again: published after recover_after_s.
        let back = feed(&mut monitor, 5100, &[(100, 0.0); 22]);
        assert!(back[..20].iter().all(|v| *v == TrackingVerdict::Hold));
        assert_eq!(back[21], TrackingVerdict::Publish);
    }

    #[test]
    fn implausible_speed_is_lost_at_once() {
        let mut monitor = TrackingMonitor::new(config());
        feed(&mut monitor, 0, &[(100, 0.1); 21]);
        assert!(matches!(
            monitor.update(1100 * MS, 100, 2.5),
            TrackingVerdict::Lost(LostReason::Speed { .. })
        ));
        assert!(matches!(
            monitor.update(1150 * MS, 100, f64::NAN),
            TrackingVerdict::Lost(LostReason::Speed { .. })
        ));
        monitor.restart();
        assert_eq!(monitor.update(1200 * MS, 100, 0.1), TrackingVerdict::Hold);
    }
}
