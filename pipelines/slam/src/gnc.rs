//! Graduated Non-Convexity (GNC) for outlier-robust pose-graph optimization.
//!
//! A plain robust M-estimator (Huber / Cauchy — see [`crate::RobustKernel`])
//! down-weights large residuals, but its influence function is *non-convex*, so
//! iteratively-reweighted least squares (IRLS) only finds a **local** minimum:
//! seeded poorly, or with many outlier loop closures, it converges to a wrong
//! basin that the outliers still corrupt. GNC (Yang, Antonante, Tzoumas &
//! Carlone, *"Graduated Non-Convexity for Robust Spatial Perception"*, RA-L
//! 2020 — the engine behind Kimera-RPGO and TEASER++) escapes that trap by a
//! homotopy: it optimizes a sequence of surrogate costs governed by a control
//! parameter `μ`, starting from a **convex** surrogate that trusts every edge
//! (ordinary least squares) and gradually annealing `μ` toward the true,
//! sharply non-convex robust shape that rejects outliers. Each surrogate is
//! minimized from the previous solution, so the optimizer is shepherded into
//! the correct basin before the cost becomes non-convex.
//!
//! By the Black-Rangarajan duality the per-surrogate solve is itself a *weighted*
//! least-squares problem with closed-form weights `w_i ∈ [0, 1]`, so GNC drops
//! straight into the existing weighted normal-equation assembly — it just
//! supplies the weights and the `μ` schedule instead of the IRLS kernel. This
//! module is the pure-math core: it knows nothing about [`crate::PoseGraph`],
//! operates only on (whitened) squared residuals, and is exercised in isolation
//! by the unit tests below. The driver that runs the outer `μ` loop over a pose
//! graph lives in [`crate::PoseGraph::optimize_se3_gnc`].
//!
//! Two surrogate families are provided, both from the 2020 paper:
//! - [`GncKernel::GemanMcClure`] — a smooth saturating surrogate; weights decay
//!   continuously in `(0, 1]`. `μ` starts large and is **divided** down to `1`,
//!   at which point the surrogate is the true Geman-McClure cost.
//! - [`GncKernel::TruncatedLeastSquares`] — the TEASER++ default; a hard
//!   inlier/outlier verdict (`w = 1` or `w = 0`) outside a soft transition band.
//!   `μ` starts small and is **multiplied** up until the band collapses to the
//!   threshold `c²`, giving a crisp truncated-quadratic cost.

/// Which GNC surrogate family the [`GncState`] anneals through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GncKernel {
    /// Geman-McClure surrogate: smooth, weights in `(0, 1]`, `μ` annealed
    /// large → 1.
    #[default]
    GemanMcClure,
    /// Truncated-least-squares surrogate (TEASER++ default): a hard 0/1 verdict
    /// outside a soft band, `μ` annealed small → large.
    TruncatedLeastSquares,
}

/// Configuration for a GNC run. The scale `c` is the inlier threshold in the
/// same (whitened / Mahalanobis) units as the squared residuals fed to the
/// optimizer: an edge whose residual norm is well below `c` is treated as an
/// inlier, well above `c` as an outlier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GncConfig {
    /// Surrogate family.
    pub kernel: GncKernel,
    /// Inlier scale `c` (a residual-norm threshold). The squared threshold `c²`
    /// is what the weight formulas use.
    pub c: f64,
    /// Geometric `μ` annealing factor (`> 1`). Geman-McClure divides `μ` by it;
    /// truncated-least-squares multiplies. `1.4` is the paper's default.
    pub anneal_factor: f64,
    /// Hard cap on the number of outer `μ` levels.
    pub max_outer: usize,
    /// Inner weighted-least-squares iterations to run at each fixed `μ` level.
    pub inner_iterations: usize,
}

impl Default for GncConfig {
    fn default() -> Self {
        Self {
            kernel: GncKernel::GemanMcClure,
            c: 1.0,
            anneal_factor: 1.4,
            max_outer: 100,
            inner_iterations: 5,
        }
    }
}

/// The annealing state of a GNC run: the surrogate family, the squared inlier
/// scale `c²`, and the current control parameter `μ`. Construct it from the
/// largest squared residual at the (least-squares) initialization so the first
/// surrogate is convex, then alternate [`GncState::weight`] (to reweight the
/// edges) with [`GncState::anneal`] (to sharpen the surrogate) until
/// [`GncState::is_terminal`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GncState {
    kernel: GncKernel,
    c2: f64,
    factor: f64,
    mu: f64,
}

/// `μ` beyond which the truncated-least-squares transition band (whose width is
/// `≈ 2 c² / μ`) is under 1 % of `c²`, i.e. the surrogate is effectively the
/// hard truncated quadratic and further annealing changes nothing.
const TLS_MU_TERMINAL: f64 = 200.0;

impl GncState {
    /// Initialize the control parameter so the first surrogate is convex over
    /// the data. `max_residual_sq` is the largest squared residual at the
    /// least-squares (e.g. chordal-seeded) starting point.
    ///
    /// Geman-McClure becomes convex for `μ ≥ 2·s_max / c²` (paper, §IV), so we
    /// start there. Truncated-least-squares starts at `μ = c² / (2·s_max − c²)`
    /// (a small positive value when outliers are present), the smallest `μ` for
    /// which the surrogate is non-trivial; with no residual exceeding `c²` the
    /// problem is already all-inlier and we start at the terminal `μ`.
    pub fn new(config: &GncConfig, max_residual_sq: f64) -> Self {
        let c2 = config.c * config.c;
        let s_max = max_residual_sq.max(0.0);
        let mu = match config.kernel {
            GncKernel::GemanMcClure => (2.0 * s_max / c2).max(1.0),
            GncKernel::TruncatedLeastSquares => {
                let denom = 2.0 * s_max - c2;
                if denom <= 0.0 {
                    // Every residual is already within the inlier band.
                    TLS_MU_TERMINAL
                } else {
                    (c2 / denom).clamp(f64::MIN_POSITIVE, TLS_MU_TERMINAL)
                }
            }
        };
        Self {
            kernel: config.kernel,
            c2,
            factor: config.anneal_factor,
            mu,
        }
    }

    /// Current control parameter `μ`.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Black-Rangarajan weight `w ∈ [0, 1]` for an edge whose (whitened) squared
    /// residual is `s`. `w = 1` keeps the edge at full strength, `w → 0` rejects
    /// it as an outlier.
    pub fn weight(&self, s: f64) -> f64 {
        let s = s.max(0.0);
        match self.kernel {
            // Geman-McClure: w = (μ c² / (s + μ c²))². At μ → ∞ this is 1 for
            // every edge (least squares); at μ = 1 it is the true GM weight
            // (c² / (s + c²))².
            GncKernel::GemanMcClure => {
                let mc2 = self.mu * self.c2;
                let r = mc2 / (s + mc2);
                r * r
            }
            // Truncated least squares: inside the lower band fully trusted,
            // above the upper band fully rejected, with a smooth bridge in
            // between (paper, eq. 14). As μ → ∞ both bands meet at c²,
            // recovering the hard truncated quadratic.
            GncKernel::TruncatedLeastSquares => {
                let mu = self.mu;
                let lo = (mu / (mu + 1.0)) * self.c2;
                let hi = ((mu + 1.0) / mu) * self.c2;
                if s <= lo {
                    1.0
                } else if s >= hi {
                    0.0
                } else {
                    (self.c2 * mu * (mu + 1.0) / s).sqrt() - mu
                }
            }
        }
    }

    /// Sharpen the surrogate one geometric step toward the true robust cost,
    /// clamped at the terminal `μ`. Returns whether the state is now terminal.
    pub fn anneal(&mut self) -> bool {
        match self.kernel {
            GncKernel::GemanMcClure => self.mu = (self.mu / self.factor).max(1.0),
            GncKernel::TruncatedLeastSquares => {
                self.mu = (self.mu * self.factor).min(TLS_MU_TERMINAL)
            }
        }
        self.is_terminal()
    }

    /// Whether `μ` has reached the true robust cost (no further annealing will
    /// change the weights meaningfully).
    pub fn is_terminal(&self) -> bool {
        match self.kernel {
            GncKernel::GemanMcClure => self.mu <= 1.0,
            GncKernel::TruncatedLeastSquares => self.mu >= TLS_MU_TERMINAL,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(kernel: GncKernel) -> GncConfig {
        GncConfig {
            kernel,
            c: 1.0,
            anneal_factor: 1.4,
            max_outer: 100,
            inner_iterations: 5,
        }
    }

    #[test]
    fn gm_first_surrogate_is_nearly_least_squares() {
        // With outliers present, μ₀ is large, so even the largest residual keeps
        // a substantial weight — the optimizer trusts every edge at first.
        let s_max = 100.0;
        let state = GncState::new(&cfg(GncKernel::GemanMcClure), s_max);
        // μ₀ = 2·100 / 1 = 200; weight at s_max = (200/(100+200))² = (2/3)².
        assert!((state.weight(s_max) - (2.0f64 / 3.0).powi(2)).abs() < 1e-12);
        // Small residuals are essentially fully trusted.
        assert!(state.weight(0.01) > 0.999);
    }

    #[test]
    fn gm_terminal_recovers_geman_mcclure_weight() {
        let mut state = GncState::new(&cfg(GncKernel::GemanMcClure), 100.0);
        for _ in 0..200 {
            if state.anneal() {
                break;
            }
        }
        assert!(state.is_terminal());
        assert!((state.mu() - 1.0).abs() < 1e-12);
        // At μ = 1 the weight is the true GM IRLS weight (c²/(s+c²))².
        for &s in &[0.0, 0.5, 1.0, 4.0, 25.0] {
            let expected = (1.0_f64 / (s + 1.0)).powi(2);
            assert!((state.weight(s) - expected).abs() < 1e-12, "s = {s}");
        }
    }

    #[test]
    fn gm_weight_is_monotone_decreasing_in_residual() {
        let state = GncState::new(&cfg(GncKernel::GemanMcClure), 50.0);
        let mut prev = f64::INFINITY;
        for i in 0..200 {
            let s = i as f64 * 0.5;
            let w = state.weight(s);
            assert!(w <= prev + 1e-15 && (0.0..=1.0).contains(&w));
            prev = w;
        }
    }

    #[test]
    fn gm_weight_decreases_monotonically_as_mu_anneals() {
        // For a fixed large residual the weight should only drop as μ sharpens.
        let mut state = GncState::new(&cfg(GncKernel::GemanMcClure), 100.0);
        let s = 25.0; // a clear outlier vs c = 1
        let mut prev = state.weight(s);
        for _ in 0..50 {
            let terminal = state.anneal();
            let w = state.weight(s);
            assert!(w <= prev + 1e-15, "weight rose during annealing");
            prev = w;
            if terminal {
                break;
            }
        }
        // The outlier ends up firmly rejected (GM weight at s=25 is 1/26² ≈ 0.0015).
        assert!(prev < 0.01);
    }

    #[test]
    fn tls_terminal_is_hard_threshold_at_c_squared() {
        let mut state = GncState::new(&cfg(GncKernel::TruncatedLeastSquares), 100.0);
        for _ in 0..200 {
            if state.anneal() {
                break;
            }
        }
        assert!(state.is_terminal());
        // Below c² fully trusted, above c² fully rejected (band width < 1%·c²).
        assert!((state.weight(0.9) - 1.0).abs() < 1e-9);
        assert!(state.weight(1.2) < 1e-9);
    }

    #[test]
    fn tls_weight_is_in_unit_interval_and_brackets_the_band() {
        let state = GncState::new(&cfg(GncKernel::TruncatedLeastSquares), 100.0);
        for i in 0..400 {
            let s = i as f64 * 0.25;
            let w = state.weight(s);
            assert!((0.0..=1.0).contains(&w), "w = {w} out of range at s = {s}");
        }
        // The low end of the soft band is fully trusted, the high end rejected.
        let mu = state.mu();
        let lo = mu / (mu + 1.0); // · c² with c = 1
        let hi = (mu + 1.0) / mu;
        assert!((state.weight(lo * 0.99) - 1.0).abs() < 1e-12);
        assert!(state.weight(hi * 1.01) < 1e-12);
    }

    #[test]
    fn tls_all_inlier_problem_starts_terminal() {
        // No residual exceeds c² ⇒ nothing to reject ⇒ start at the hard cost.
        let state = GncState::new(&cfg(GncKernel::TruncatedLeastSquares), 0.3);
        assert!(state.is_terminal());
    }

    #[test]
    fn anneal_is_deterministic() {
        let run = || {
            let mut s = GncState::new(&cfg(GncKernel::GemanMcClure), 73.0);
            let mut trace = Vec::new();
            for _ in 0..40 {
                trace.push(s.mu());
                if s.anneal() {
                    break;
                }
            }
            trace
        };
        assert_eq!(run(), run());
    }
}
