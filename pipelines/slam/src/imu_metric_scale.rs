//! Metric scale of an SfM model from the camera's own IMU.
//!
//! The per-run visual-inertial linear alignment of Cuttini et al., arXiv
//! 2610.35554, with the camera-centre term of docs/insta360_imu_scale.md
//! (where the Python reference, `scripts/insta360_imu_scale.py`, was
//! validated on the 50 OverMaps360 walks).
//!
//! Input: the registered frames of one camera of the model, each with its
//! frame number in the video, and the IMU samples (rad/s, m/s²) on the
//! video's clock.
//!
//! 1. **Frame timing** `t = period · number + offset` and the
//!    camera-from-IMU rotation `R_CB`: a coarse grid on the frame-to-frame
//!    rotation angles (gyro against model), a finer grid within ±0.4% of the
//!    period, then Nelder-Mead on the median residual of the Kabsch fit
//!    `r_C = R_CB r_B`. Drift check: the offset refitted on the first and
//!    last quarters of the frames.
//! 2. **Runs**: frames at most 1.2 s apart, broken where the model's speed
//!    exceeds three times its median (a misregistered frame), with at least
//!    8 intervals.
//! 3. **Per run**, linear least squares for the scale `s`, gravity `g`, the
//!    accelerometer bias `b_a`, one velocity per frame and (by default) a
//!    camera-centre error `e = R_WC (c × ω_C)` proportional to the camera's
//!    angular velocity `ω_C` (rolling shutter), with IMU preintegration
//!    between frames (rotation from the model, propagated by the gyro):
//!    `s ΔC - Δ(E c') - v_k dt - g dt²/2 + J_k b_a = Δp_k` and
//!    `v_{k+1} - v_k - g dt + K_k b_a = Δv_k` (`c' = s c`). Intervals with
//!    residuals above five times the median are dropped and the run
//!    refitted; runs whose |g| is outside [9.3, 10.3] m/s² are discarded.
//!    The scale is the mean over the remaining runs.
//!
//! The scale is in metres per model unit: multiply the model's camera
//! centres and points by it.

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, UnitQuaternion, Vector3};
use rayon::prelude::*;

/// IMU samples on the video's clock: times in seconds (increasing), angular
/// velocity in rad/s and specific force in m/s², in the IMU's frame.
#[derive(Debug, Clone, Default)]
pub struct ImuSamples {
    pub time: Vec<f64>,
    pub gyro: Vec<Vector3<f64>>,
    pub accel: Vec<Vector3<f64>>,
}

/// One registered frame of the camera stream.
#[derive(Debug, Clone, Copy)]
pub struct ScaleFrame {
    /// Frame number in the video; the frame is at `period · number + offset`.
    pub number: i64,
    /// World-from-camera rotation in the model.
    pub rotation: Matrix3<f64>,
    /// Camera centre in model units.
    pub centre: Vector3<f64>,
}

/// How the frames' times are found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameTiming {
    /// Search the period within ±10% of `nominal_period` (default: the IMU
    /// span over the largest frame number) and the offset within ±30 s.
    Search { nominal_period: Option<f64> },
    /// Known: `t = period · number + offset`.
    Known { period: f64, offset: f64 },
}

/// Settings for [`estimate_imu_scale`].
#[derive(Debug, Clone)]
pub struct ImuScaleConfig {
    pub timing: FrameTiming,
    /// Longest gap between consecutive frames of a run, seconds.
    pub max_gap_s: f64,
    /// Fewest intervals in a run.
    pub min_intervals: usize,
    /// Accepted gravity magnitude per run, m/s².
    pub gravity_range: (f64, f64),
    /// Runs break where the model's speed exceeds this many times its median.
    pub speed_jump_factor: f64,
    /// Intervals whose residual exceeds this many times the run's median are
    /// dropped before the refit.
    pub outlier_factor: f64,
    /// Fit the camera-centre error proportional to the angular velocity.
    pub rolling_shutter_term: bool,
}

impl Default for ImuScaleConfig {
    fn default() -> Self {
        Self {
            timing: FrameTiming::Search {
                nominal_period: None,
            },
            max_gap_s: 1.2,
            min_intervals: 8,
            gravity_range: (9.3, 10.3),
            speed_jump_factor: 3.0,
            outlier_factor: 5.0,
            rolling_shutter_term: true,
        }
    }
}

/// The fit of one run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunScale {
    /// Time of the run's first and last frame, seconds.
    pub start: f64,
    pub end: f64,
    pub intervals: usize,
    /// Metres per model unit.
    pub scale: f64,
    /// |g|, m/s².
    pub gravity: f64,
    /// The camera-centre term `c` (metres times seconds, camera frame).
    pub centre_term: Option<Vector3<f64>>,
    /// Whether |g| passed the gravity check.
    pub kept: bool,
}

/// Result of [`estimate_imu_scale`].
#[derive(Debug, Clone)]
pub struct ImuScaleEstimate {
    /// Metres per model unit: the mean over the kept runs.
    pub scale: f64,
    /// Standard deviation of the kept runs' scales.
    pub scale_std: f64,
    pub runs: Vec<RunScale>,
    /// Median |g| of the kept runs.
    pub gravity_median: f64,
    /// Frame timing `t = period · number + offset`.
    pub period: f64,
    pub offset: f64,
    /// Camera-from-IMU rotation.
    pub camera_from_imu: Matrix3<f64>,
    /// Median angle between the model's and the gyro's frame-to-frame
    /// rotations after the fit, degrees.
    pub rotation_residual_deg: f64,
    /// Offset refitted on the first and last quarters of the frames, minus
    /// the offset, milliseconds (`[0, 0]` with known timing).
    pub drift_ms: [f64; 2],
    /// Intervals where the model's speed jumps (runs break there).
    pub speed_jumps: usize,
}

impl ImuScaleEstimate {
    pub fn runs_kept(&self) -> usize {
        self.runs.iter().filter(|r| r.kept).count()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImuScaleError {
    TooFewFrames(usize),
    TooFewImuSamples(usize),
    /// No consecutive frames overlap the IMU samples.
    NoOverlap,
    /// No run of enough intervals.
    NoRun,
    /// No run passed the gravity check.
    NoRunPassedGravity,
}

impl std::fmt::Display for ImuScaleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewFrames(n) => write!(f, "only {n} registered frames"),
            Self::TooFewImuSamples(n) => write!(f, "only {n} IMU samples"),
            Self::NoOverlap => write!(f, "no consecutive frames overlap the IMU samples"),
            Self::NoRun => write!(f, "no run of frames long enough for the IMU alignment"),
            Self::NoRunPassedGravity => write!(f, "no run passed the gravity check"),
        }
    }
}

impl std::error::Error for ImuScaleError {}

/// Estimates the model's metric scale from the IMU; see the module docs.
pub fn estimate_imu_scale(
    imu: &ImuSamples,
    frames: &[ScaleFrame],
    cfg: &ImuScaleConfig,
) -> Result<ImuScaleEstimate, ImuScaleError> {
    if imu.time.len() < 100 {
        return Err(ImuScaleError::TooFewImuSamples(imu.time.len()));
    }
    let mut frames = frames.to_vec();
    frames.sort_by_key(|f| f.number);
    frames.dedup_by_key(|f| f.number);
    if frames.len() < cfg.min_intervals + 1 {
        return Err(ImuScaleError::TooFewFrames(frames.len()));
    }
    let gyro = GyroIntegral::new(&imu.time, &imu.gyro);
    let (period, offset, drift_ms) = match cfg.timing {
        FrameTiming::Known { period, offset } => (period, offset, [0.0; 2]),
        FrameTiming::Search { nominal_period } => {
            estimate_timing(&imu.time, &gyro, &frames, nominal_period)?
        }
    };
    let times: Vec<f64> = frames
        .iter()
        .map(|f| period * f.number as f64 + offset)
        .collect();

    // Camera-from-IMU rotation from consecutive frames (outlier-robust).
    let pairs = rotation_pairs(&frames, &times, &gyro, 3);
    if pairs.rb.is_empty() {
        return Err(ImuScaleError::NoOverlap);
    }
    let (camera_from_imu, residuals) = rotation_fit(&pairs.rb, &pairs.d_rc);
    let rotation_residual_deg = median(&residuals).to_degrees();

    let world_from_imu: Vec<Matrix3<f64>> = frames
        .iter()
        .map(|f| f.rotation * camera_from_imu)
        .collect();
    let (t0, t1) = (imu.time[0], *imu.time.last().unwrap());
    let speed: Vec<f64> = (0..frames.len() - 1)
        .map(|k| (frames[k + 1].centre - frames[k].centre).norm() / (times[k + 1] - times[k]))
        .collect();
    let speed_median = median(&speed);
    let jump: Vec<bool> = speed
        .iter()
        .map(|&v| v > cfg.speed_jump_factor * speed_median)
        .collect();
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut cur = vec![0usize];
    for k in 0..frames.len() - 1 {
        let ok = times[k + 1] - times[k] <= cfg.max_gap_s
            && times[k] >= t0
            && times[k + 1] <= t1
            && !jump[k];
        if ok {
            cur.push(k + 1);
        } else {
            runs.push(std::mem::replace(&mut cur, vec![k + 1]));
        }
    }
    runs.push(cur);
    runs.retain(|r| r.len() > cfg.min_intervals);
    if runs.is_empty() {
        return Err(ImuScaleError::NoRun);
    }

    let centre_error: Option<Vec<Matrix3<f64>>> = cfg.rolling_shutter_term.then(|| {
        let h = 0.01;
        frames
            .iter()
            .zip(&times)
            .map(|(f, &t)| {
                let w_b = gyro.rel(t - h, t + h).scaled_axis() / (2.0 * h);
                let w_c = camera_from_imu * w_b;
                // e = R_WC (c x w) = R_WC [w]x^T c
                f.rotation * skew(&w_c).transpose()
            })
            .collect()
    });
    let fits: Vec<RunScale> = runs
        .par_iter()
        .map(|run| {
            let pre: Vec<Preintegrated> = run[..run.len() - 1]
                .iter()
                .map(|&k| preintegrate(imu, &gyro, &world_from_imu[k], times[k], times[k + 1]))
                .collect();
            let (scale, gravity, centre_term) = solve_run(
                run,
                &pre,
                &frames,
                centre_error.as_deref(),
                cfg.outlier_factor,
            );
            RunScale {
                start: times[run[0]],
                end: times[*run.last().unwrap()],
                intervals: run.len() - 1,
                scale,
                gravity,
                centre_term,
                kept: (cfg.gravity_range.0..=cfg.gravity_range.1).contains(&gravity),
            }
        })
        .collect();
    let kept: Vec<&RunScale> = fits.iter().filter(|r| r.kept).collect();
    if kept.is_empty() {
        return Err(ImuScaleError::NoRunPassedGravity);
    }
    let scale = kept.iter().map(|r| r.scale).sum::<f64>() / kept.len() as f64;
    let scale_std =
        (kept.iter().map(|r| (r.scale - scale).powi(2)).sum::<f64>() / kept.len() as f64).sqrt();
    let gravity_median = median(&kept.iter().map(|r| r.gravity).collect::<Vec<_>>());
    Ok(ImuScaleEstimate {
        scale,
        scale_std,
        runs: fits,
        gravity_median,
        period,
        offset,
        camera_from_imu,
        rotation_residual_deg,
        drift_ms,
        speed_jumps: jump.iter().filter(|&&j| j).count(),
    })
}

/// Body rotation integrated from the gyro (midpoint rule); the rotation from
/// `ti` to `tj` is `at(ti)⁻¹ at(tj)`.
struct GyroIntegral<'a> {
    t: &'a [f64],
    w: &'a [Vector3<f64>],
    p: Vec<UnitQuaternion<f64>>,
}

impl<'a> GyroIntegral<'a> {
    fn new(t: &'a [f64], w: &'a [Vector3<f64>]) -> Self {
        let mut p = Vec::with_capacity(t.len());
        p.push(UnitQuaternion::identity());
        for k in 0..t.len() - 1 {
            let mid = 0.5 * (w[k] + w[k + 1]);
            let step = UnitQuaternion::from_scaled_axis(mid * (t[k + 1] - t[k]));
            p.push(p[k] * step);
        }
        Self { t, w, p }
    }

    /// The last sample at or before `tq` (clamped).
    fn index(&self, tq: f64) -> usize {
        self.t
            .partition_point(|&x| x <= tq)
            .saturating_sub(1)
            .min(self.t.len() - 2)
    }

    fn at(&self, tq: f64) -> UnitQuaternion<f64> {
        let k = self.index(tq);
        self.p[k] * UnitQuaternion::from_scaled_axis(self.w[k] * (tq - self.t[k]))
    }

    fn rel(&self, ti: f64, tj: f64) -> UnitQuaternion<f64> {
        self.at(ti).inverse() * self.at(tj)
    }

    /// The integral at the first sample at or after `tq` (for the coarse search).
    fn nearest(&self, tq: f64) -> &UnitQuaternion<f64> {
        &self.p[self.t.partition_point(|&x| x < tq).min(self.p.len() - 1)]
    }
}

/// Frame-to-frame rotations of consecutive frames at most `max_number_gap`
/// numbers apart: the model's (camera frame) and the gyro's (IMU frame).
struct RotationPairs {
    rb: Vec<Vector3<f64>>,
    d_rc: Vec<Rotation3<f64>>,
}

fn rotation_pairs(
    frames: &[ScaleFrame],
    times: &[f64],
    gyro: &GyroIntegral,
    max_number_gap: i64,
) -> RotationPairs {
    let (t0, t1) = (gyro.t[0], *gyro.t.last().unwrap());
    let mut out = RotationPairs {
        rb: Vec::new(),
        d_rc: Vec::new(),
    };
    for k in 0..frames.len() - 1 {
        if frames[k + 1].number - frames[k].number > max_number_gap
            || times[k] < t0
            || times[k + 1] > t1
        {
            continue;
        }
        out.rb.push(gyro.rel(times[k], times[k + 1]).scaled_axis());
        out.d_rc.push(Rotation3::from_matrix_unchecked(
            frames[k].rotation.transpose() * frames[k + 1].rotation,
        ));
    }
    out
}

/// `R` minimising `Σ |rc - R rb|²`.
fn kabsch(rb: &[Vector3<f64>], rc: &[Vector3<f64>]) -> Matrix3<f64> {
    let h: Matrix3<f64> = rb.iter().zip(rc).map(|(b, c)| b * c.transpose()).sum();
    let svd = h.svd(true, true);
    let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
    let d = (v_t.transpose() * u.transpose()).determinant().signum();
    v_t.transpose() * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * u.transpose()
}

/// Angles between the model's rotations and the gyro's mapped by `r`.
fn rotation_residuals(r: &Matrix3<f64>, rb: &[Vector3<f64>], d_rc: &[Rotation3<f64>]) -> Vec<f64> {
    rb.iter()
        .zip(d_rc)
        .map(|(b, c)| {
            let pred = r * Rotation3::from_scaled_axis(*b).matrix() * r.transpose();
            Rotation3::from_matrix_unchecked(c.matrix().transpose() * pred).angle()
        })
        .collect()
}

/// `R_CB` from the Kabsch fit, refitted twice without residuals above three
/// times the median; returns it and the residuals (radians).
fn rotation_fit(rb: &[Vector3<f64>], d_rc: &[Rotation3<f64>]) -> (Matrix3<f64>, Vec<f64>) {
    let rc: Vec<Vector3<f64>> = d_rc.iter().map(|r| r.scaled_axis()).collect();
    let mut r = kabsch(rb, &rc);
    let mut resid = rotation_residuals(&r, rb, d_rc);
    for _ in 0..2 {
        let cut = 3.0 * median(&resid);
        let (b, c): (Vec<_>, Vec<_>) = rb
            .iter()
            .zip(&rc)
            .zip(&resid)
            .filter(|(_, &e)| e <= cut)
            .map(|((b, c), _)| (*b, *c))
            .unzip();
        r = kabsch(&b, &c);
        resid = rotation_residuals(&r, rb, d_rc);
    }
    (r, resid)
}

/// Period, offset and drift check (ms) from the gyro and the model's rotations.
fn estimate_timing(
    t: &[f64],
    gyro: &GyroIntegral,
    frames: &[ScaleFrame],
    nominal_period: Option<f64>,
) -> Result<(f64, f64, [f64; 2]), ImuScaleError> {
    let mut i0 = Vec::new();
    let mut i1 = Vec::new();
    let mut d_rc = Vec::new();
    for k in 0..frames.len() - 1 {
        if frames[k + 1].number - frames[k].number <= 3 {
            i0.push(frames[k].number as f64);
            i1.push(frames[k + 1].number as f64);
            d_rc.push(Rotation3::from_matrix_unchecked(
                frames[k].rotation.transpose() * frames[k + 1].rotation,
            ));
        }
    }
    if i0.len() < 8 {
        return Err(ImuScaleError::TooFewFrames(i0.len()));
    }
    let angle_c: Vec<f64> = d_rc.iter().map(|r| r.angle()).collect();
    let rc: Vec<Vector3<f64>> = d_rc.iter().map(|r| r.scaled_axis()).collect();
    let (t0, t_end) = (t[0], *t.last().unwrap());
    let (n_min, n_max) = (i0[0], *i1.last().unwrap());
    let in_range = |a: f64, b: f64| a * n_min + b >= t0 && a * n_max + b <= t_end;

    // Median |gyro angle - model angle| at the samples nearest the frame times.
    let magnitude_cost = |a: f64, b: f64| -> f64 {
        let mut d: Vec<f64> = i0
            .iter()
            .zip(&i1)
            .zip(&angle_c)
            .map(|((&n0, &n1), &ac)| {
                let qi = gyro.nearest(a * n0 + b);
                let qj = gyro.nearest(a * n1 + b);
                let angle_b = 2.0 * qi.coords.dot(&qj.coords).abs().min(1.0).acos();
                (angle_b - ac).abs()
            })
            .collect();
        median_in_place(&mut d)
    };
    let grid = |a_values: &[f64], b_values: &[f64]| -> Option<(f64, f64, f64)> {
        a_values
            .par_iter()
            .filter_map(|&a| {
                b_values
                    .iter()
                    .filter(|&&b| in_range(a, b))
                    .map(|&b| (magnitude_cost(a, b), a, b))
                    .min_by(|x, y| x.0.total_cmp(&y.0))
            })
            .min_by(|x, y| x.0.total_cmp(&y.0))
    };
    let a0 = nominal_period.unwrap_or((t_end - t0) / n_max);
    let a_coarse: Vec<f64> = (0..)
        .map(|k| a0 * (0.9 + 0.0012 * k as f64))
        .take_while(|&a| a < a0 * 1.1)
        .collect();
    let b_coarse: Vec<f64> = (0..1200).map(|k| -30.0 + 0.05 * k as f64).collect();
    let (_, a, b) = grid(&a_coarse, &b_coarse).ok_or(ImuScaleError::NoOverlap)?;
    let a_fine: Vec<f64> = (0..81).map(|k| a * (0.996 + 0.0001 * k as f64)).collect();
    let b_fine: Vec<f64> = (0..120).map(|k| b - 0.3 + 0.005 * k as f64).collect();
    let (_, a, b) = grid(&a_fine, &b_fine).ok_or(ImuScaleError::NoOverlap)?;

    // Median residual of the plain Kabsch fit over a subset of the pairs.
    let fit = |a: f64, b: f64, range: std::ops::Range<usize>| -> f64 {
        if !in_range(a, b) {
            return f64::INFINITY;
        }
        let rb: Vec<Vector3<f64>> = range
            .clone()
            .map(|k| gyro.rel(a * i0[k] + b, a * i1[k] + b).scaled_axis())
            .collect();
        let r = kabsch(&rb, &rc[range.clone()]);
        let mut resid = rotation_residuals(&r, &rb, &d_rc[range]);
        median_in_place(&mut resid)
    };
    let all = 0..i0.len();
    let x = nelder_mead(
        |x| fit(x[0], x[1], all.clone()),
        vec![vec![a, b], vec![a * 1.0005, b], vec![a, b + 0.02]],
        1e-7,
        1e-8,
        300,
    );
    let (a, b) = (x[0], x[1]);
    let q = i0.len() / 4;
    let mut drift = [0.0; 2];
    for (d, range) in drift.iter_mut().zip([0..q, 3 * q..i0.len()]) {
        let bq = nelder_mead(
            |x| fit(a, x[0], range.clone()),
            vec![vec![b], vec![b + 0.01]],
            1e-5,
            1e-4,
            200,
        );
        *d = (bq[0] - b) * 1e3;
    }
    Ok((a, b, drift))
}

/// Nelder-Mead minimisation from an initial simplex (`n + 1` points),
/// stopping when the simplex is within `xatol` of its best point in every
/// coordinate and the values within `fatol`, or after `max_iter` iterations
/// (scipy's defaults: reflection 1, expansion 2, contraction and shrink 0.5).
fn nelder_mead(
    f: impl Fn(&[f64]) -> f64,
    simplex: Vec<Vec<f64>>,
    xatol: f64,
    fatol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = simplex[0].len();
    let mut pts: Vec<(f64, Vec<f64>)> = simplex.into_iter().map(|x| (f(&x), x)).collect();
    let lerp = |a: &[f64], b: &[f64], t: f64| -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + t * (y - x)).collect()
    };
    for _ in 0..max_iter {
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let spread_x = pts[1..]
            .iter()
            .flat_map(|(_, x)| x.iter().zip(&pts[0].1).map(|(a, b)| (a - b).abs()))
            .fold(0.0, f64::max);
        let spread_f = pts[1..]
            .iter()
            .map(|(v, _)| (v - pts[0].0).abs())
            .fold(0.0, f64::max);
        if spread_x <= xatol && spread_f <= fatol {
            break;
        }
        let centroid: Vec<f64> = (0..n)
            .map(|i| pts[..n].iter().map(|(_, x)| x[i]).sum::<f64>() / n as f64)
            .collect();
        let worst = pts[n].1.clone();
        let reflected = lerp(&centroid, &worst, -1.0);
        let fr = f(&reflected);
        if fr < pts[0].0 {
            let expanded = lerp(&centroid, &worst, -2.0);
            let fe = f(&expanded);
            pts[n] = if fe < fr {
                (fe, expanded)
            } else {
                (fr, reflected)
            };
        } else if fr < pts[n - 1].0 {
            pts[n] = (fr, reflected);
        } else {
            let (contracted, fc) = if fr < pts[n].0 {
                let c = lerp(&centroid, &worst, -0.5);
                let v = f(&c);
                (c, v)
            } else {
                let c = lerp(&centroid, &worst, 0.5);
                let v = f(&c);
                (c, v)
            };
            if fc < pts[n].0.min(fr) {
                pts[n] = (fc, contracted);
            } else {
                let best = pts[0].1.clone();
                for p in pts.iter_mut().skip(1) {
                    let x = lerp(&best, &p.1, 0.5);
                    *p = (f(&x), x);
                }
            }
        }
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    pts.swap_remove(0).1
}

/// `dp, dv` (world frame) and the bias Jacobians `J, K` over one interval.
struct Preintegrated {
    dp: Vector3<f64>,
    dv: Vector3<f64>,
    j: Matrix3<f64>,
    k: Matrix3<f64>,
    dt: f64,
}

/// Euler steps at the IMU rate over `[t0, t1]`, each with the specific force
/// of the sample at or before its start, rotated into the world by the
/// model's rotation at `t0` propagated by the gyro.
fn preintegrate(
    imu: &ImuSamples,
    gyro: &GyroIntegral,
    world_from_imu: &Matrix3<f64>,
    t0: f64,
    t1: f64,
) -> Preintegrated {
    let t = &imu.time;
    let n0 = t.partition_point(|&x| x <= t0);
    let n1 = t.partition_point(|&x| x <= t1);
    let start = gyro.at(t0).inverse();
    let mut out = Preintegrated {
        dp: Vector3::zeros(),
        dv: Vector3::zeros(),
        j: Matrix3::zeros(),
        k: Matrix3::zeros(),
        dt: t1 - t0,
    };
    let mut ts = t0;
    // Step ends: the samples inside the interval, then t1; each step uses the
    // sample at or before its start.
    let ends = t[n0..n1].iter().copied().chain(std::iter::once(t1));
    for (te, accel) in ends.zip(&imu.accel[n0 - 1..n1]) {
        let dt = te - ts;
        let rw = world_from_imu * (start * gyro.at(ts)).to_rotation_matrix().matrix();
        let aw = rw * accel;
        out.dp += out.dv * dt + 0.5 * aw * dt * dt;
        out.dv += aw * dt;
        out.j += out.k * dt + 0.5 * rw * dt * dt;
        out.k += rw * dt;
        ts = te;
    }
    out
}

/// One run's least squares: returns `(s, |g|, c)`.
fn solve_run(
    run: &[usize],
    pre: &[Preintegrated],
    frames: &[ScaleFrame],
    centre_error: Option<&[Matrix3<f64>]>,
    outlier_factor: f64,
) -> (f64, f64, Option<Vector3<f64>>) {
    let ne = if centre_error.is_some() { 3 } else { 0 };
    let vo = 7 + ne;
    let cols = vo + 3 * run.len();
    let rows = 6 * (run.len() - 1);
    let mut a = DMatrix::<f64>::zeros(rows, cols);
    let mut y = DVector::<f64>::zeros(rows);
    let eye = Matrix3::<f64>::identity();
    for (j, (&i, p)) in run.iter().zip(pre).enumerate() {
        let r = 6 * j;
        let dc = frames[i + 1].centre - frames[i].centre;
        a.view_mut((r, 0), (3, 1)).copy_from(&dc);
        a.view_mut((r, 1), (3, 3))
            .copy_from(&(-0.5 * p.dt * p.dt * eye));
        a.view_mut((r, 4), (3, 3)).copy_from(&p.j);
        if let Some(e) = centre_error {
            a.view_mut((r, 7), (3, 3)).copy_from(&(-(e[i + 1] - e[i])));
        }
        a.view_mut((r, vo + 3 * j), (3, 3))
            .copy_from(&(-p.dt * eye));
        y.rows_mut(r, 3).copy_from(&p.dp);
        a.view_mut((r + 3, 1), (3, 3)).copy_from(&(-p.dt * eye));
        a.view_mut((r + 3, 4), (3, 3)).copy_from(&p.k);
        a.view_mut((r + 3, vo + 3 * j), (3, 3)).copy_from(&(-eye));
        a.view_mut((r + 3, vo + 3 * j + 3), (3, 3)).copy_from(&eye);
        y.rows_mut(r + 3, 3).copy_from(&p.dv);
    }
    let x = least_squares(&a, &y);
    let resid = &a * &x - &y;
    let e: Vec<f64> = (0..run.len() - 1)
        .map(|j| resid.rows(6 * j, 6).norm())
        .collect();
    let cut = outlier_factor * median(&e);
    let keep: Vec<usize> = (0..rows).filter(|&r| e[r / 6] <= cut).collect();
    let x = least_squares(&a.select_rows(&keep), &y.select_rows(&keep));
    let s = x[0];
    let g = Vector3::new(x[1], x[2], x[3]).norm();
    let c = (ne > 0).then(|| Vector3::new(x[7], x[8], x[9]) / s);
    (s, g, c)
}

/// Least squares by the normal equations (Cholesky), falling back to SVD.
fn least_squares(a: &DMatrix<f64>, y: &DVector<f64>) -> DVector<f64> {
    let ata = a.transpose() * a;
    let aty = a.transpose() * y;
    if let Some(ch) = ata.clone().cholesky() {
        return ch.solve(&aty);
    }
    let svd = a.clone().svd(true, true);
    let eps = svd.singular_values.max() * 1e-12;
    svd.solve(y, eps)
        .unwrap_or_else(|_| DVector::zeros(a.ncols()))
}

fn skew(w: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0)
}

fn median(v: &[f64]) -> f64 {
    median_in_place(&mut v.to_vec())
}

/// Median (the mean of the middle two for an even count), as numpy.
fn median_in_place(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let (_, &mut hi, _) = v.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 1 {
        hi
    } else {
        let lo = v[..n / 2].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: Vector3<f64> = Vector3::new(0.0, 9.80665, 0.0);

    fn position(t: f64) -> (Vector3<f64>, Vector3<f64>) {
        use std::f64::consts::PI;
        let (w1, w2, w3, w4) = (
            2.0 * PI * 0.85,
            2.0 * PI * 1.7,
            2.0 * PI / 40.0,
            2.0 * PI / 33.0,
        );
        let p = Vector3::new(
            1.2 * t + 0.04 * (w1 * t).sin(),
            0.03 * (w2 * t).sin(),
            8.0 * (w3 * t).sin() + 3.0 * (w4 * t).sin(),
        );
        let a = Vector3::new(
            -0.04 * w1 * w1 * (w1 * t).sin(),
            -0.03 * w2 * w2 * (w2 * t).sin(),
            -8.0 * w3 * w3 * (w3 * t).sin() - 3.0 * w4 * w4 * (w4 * t).sin(),
        );
        (p, a)
    }

    fn body_rotation(t: f64) -> Rotation3<f64> {
        use std::f64::consts::PI;
        let yaw = 0.4 * (2.0 * PI * t / 25.0).sin() + 0.05 * (2.0 * PI * 0.85 * t + 0.5).sin();
        let pitch = 0.04 * (2.0 * PI * 1.7 * t + 0.3).sin();
        let roll = 0.05 * (2.0 * PI * 0.85 * t + 1.0).sin();
        Rotation3::from_axis_angle(&Vector3::y_axis(), yaw)
            * Rotation3::from_axis_angle(&Vector3::x_axis(), pitch)
            * Rotation3::from_axis_angle(&Vector3::z_axis(), roll)
    }

    fn body_rate(t: f64) -> Vector3<f64> {
        let h = 1e-4;
        (body_rotation(t - h).inverse() * body_rotation(t + h)).scaled_axis() / (2.0 * h)
    }

    /// A walk in a model of `s_true` metres per unit; frames every
    /// `period` from `offset`, with a 1.7 s gap; `centre_term` adds the
    /// rolling-shutter-like error to the centres.
    fn scene(
        s_true: f64,
        period: f64,
        offset: f64,
        centre_term: Option<Vector3<f64>>,
    ) -> (ImuSamples, Vec<ScaleFrame>, Matrix3<f64>) {
        let b_a = Vector3::new(0.05, -0.1, 0.08);
        let camera_from_imu = *Rotation3::from_euler_angles(0.17, 1.4, -0.09).matrix();
        let mut imu = ImuSamples::default();
        for k in 0..75_000 {
            let t = k as f64 * 1e-3;
            let (_, acc) = position(t);
            imu.time.push(t);
            imu.gyro.push(body_rate(t));
            imu.accel.push(body_rotation(t).inverse() * (acc - G) + b_a);
        }
        let mut frames = Vec::new();
        for number in 1..((74.0 - offset) / period) as i64 {
            if (120..125).contains(&number) {
                continue;
            }
            let t = period * number as f64 + offset;
            let rotation = body_rotation(t).matrix() * camera_from_imu.transpose();
            let mut p = position(t).0;
            if let Some(c) = centre_term {
                let w_c = camera_from_imu * body_rate(t);
                p += rotation * c.cross(&w_c);
            }
            frames.push(ScaleFrame {
                number,
                rotation,
                centre: p / s_true,
            });
        }
        (imu, frames, camera_from_imu)
    }

    #[test]
    fn recovers_scale_and_timing() {
        let (imu, frames, camera_from_imu) = scene(1.05, 1.0 / 3.0, 0.4, None);
        let est = estimate_imu_scale(&imu, &frames, &ImuScaleConfig::default()).unwrap();
        assert!(
            (est.period - 1.0 / 3.0).abs() < 1e-5,
            "period {}",
            est.period
        );
        assert!((est.offset - 0.4).abs() < 2e-3, "offset {}", est.offset);
        assert!(est.rotation_residual_deg < 0.01);
        assert!((est.camera_from_imu - camera_from_imu).norm() < 1e-3);
        assert_eq!(est.runs_kept(), 2);
        assert!((est.scale / 1.05 - 1.0).abs() < 3e-3, "scale {}", est.scale);
        assert!((est.gravity_median - 9.80665).abs() < 0.15);
        let c = est.runs[0].centre_term.unwrap();
        assert!(c.norm() < 1e-3, "c {c}");
    }

    #[test]
    fn centre_term_recovers_the_rolling_shutter_error() {
        let c = Vector3::new(0.0, -0.015, 0.0);
        let (imu, frames, _) = scene(1.05, 1.0 / 3.0, 0.4, Some(c));
        let est = estimate_imu_scale(&imu, &frames, &ImuScaleConfig::default()).unwrap();
        assert!((est.scale / 1.05 - 1.0).abs() < 3e-3, "scale {}", est.scale);
        for run in est.runs.iter().filter(|r| r.kept) {
            assert!(
                (run.centre_term.unwrap() - c).norm() < 1.5e-3,
                "c {}",
                run.centre_term.unwrap()
            );
        }
    }

    #[test]
    fn known_timing_skips_the_search() {
        let (imu, frames, _) = scene(0.8, 0.25, -0.1, None);
        let cfg = ImuScaleConfig {
            timing: FrameTiming::Known {
                period: 0.25,
                offset: -0.1,
            },
            rolling_shutter_term: false,
            ..Default::default()
        };
        let est = estimate_imu_scale(&imu, &frames, &cfg).unwrap();
        assert_eq!(est.drift_ms, [0.0; 2]);
        assert!((est.scale / 0.8 - 1.0).abs() < 3e-3, "scale {}", est.scale);
    }

    #[test]
    fn nelder_mead_finds_a_quadratic_minimum() {
        let x = nelder_mead(
            |x| (x[0] - 1.5).powi(2) + 3.0 * (x[1] + 0.25).powi(2),
            vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1]],
            1e-9,
            1e-12,
            1000,
        );
        assert!((x[0] - 1.5).abs() < 1e-6 && (x[1] + 0.25).abs() < 1e-6);
    }

    #[test]
    fn median_matches_numpy() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
