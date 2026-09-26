//! Adaptive density control (the Inria clone / split / prune strategy), on the
//! host.
//!
//! Every `interval` steps the trainer downloads the gaussians, their Adam
//! moments and the accumulated view-space gradient statistics, and this module
//! decides the new population:
//!
//! - **clone** a gaussian whose mean screen-space position gradient (averaged
//!   over the frames it was visible in) exceeds `grad_threshold` and whose
//!   largest scale is at most `percent_dense * extent` (under-reconstruction);
//! - **split** such a gaussian when it is larger (over-reconstruction): two
//!   children sampled from its own distribution, scales divided by 1.6;
//! - **prune** gaussians with opacity below `min_opacity` or, after the first
//!   opacity reset, a world scale above `max_world_scale * extent`.
//!
//! Parameters and moments are laid out as in the device buffers: `stride`
//! floats per gaussian per group (transforms 10, opacity 1, SH 3 * cpc2).

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Densification thresholds (Inria defaults).
#[derive(Debug, Clone)]
pub struct DensifyConfig {
    pub start: usize,
    pub stop: usize,
    pub interval: usize,
    pub opacity_reset_interval: usize,
    /// Threshold on the mean |dL/d mean2d| in NDC units.
    pub grad_threshold: f32,
    pub percent_dense: f32,
    pub min_opacity: f32,
    pub max_world_scale: f32,
    /// Upper bound on the population (densification stops growing past it).
    pub max_gaussians: usize,
}

impl Default for DensifyConfig {
    fn default() -> Self {
        Self {
            start: 500,
            stop: 15_000,
            interval: 100,
            opacity_reset_interval: 3000,
            grad_threshold: 2e-4,
            percent_dense: 0.01,
            min_opacity: 0.005,
            max_world_scale: 0.1,
            max_gaussians: 3_000_000,
        }
    }
}

/// One parameter group: values and the two Adam moments, `stride` per gaussian.
#[derive(Debug, Clone)]
pub struct Group {
    pub stride: usize,
    pub values: Vec<f32>,
    pub m1: Vec<f32>,
    pub m2: Vec<f32>,
}

impl Group {
    fn row(&self, i: usize) -> &[f32] {
        &self.values[i * self.stride..(i + 1) * self.stride]
    }
    fn with_capacity(stride: usize, n: usize) -> Self {
        Self {
            stride,
            values: Vec::with_capacity(n * stride),
            m1: Vec::with_capacity(n * stride),
            m2: Vec::with_capacity(n * stride),
        }
    }
    /// Append gaussian `i` of `src`, keeping its moments.
    fn keep(&mut self, src: &Group, i: usize) {
        let r = i * self.stride..(i + 1) * self.stride;
        self.values.extend_from_slice(&src.values[r.clone()]);
        self.m1.extend_from_slice(&src.m1[r.clone()]);
        self.m2.extend_from_slice(&src.m2[r]);
    }
    /// Append new values with zeroed moments.
    fn push_new(&mut self, values: &[f32]) {
        debug_assert_eq!(values.len(), self.stride);
        self.values.extend_from_slice(values);
        self.m1.extend(std::iter::repeat_n(0.0, self.stride));
        self.m2.extend(std::iter::repeat_n(0.0, self.stride));
    }
}

/// The full optimisable state: transforms (mean | quat | log-scale), opacity
/// logit, SH.
#[derive(Debug, Clone)]
pub struct Population {
    pub transforms: Group,
    pub opacity: Group,
    pub sh: Group,
}

impl Population {
    pub fn len(&self) -> usize {
        self.opacity.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What a densification step did.
#[derive(Debug, Clone, Copy, Default)]
pub struct DensifyReport {
    pub cloned: usize,
    pub split: usize,
    pub pruned: usize,
    pub before: usize,
    pub after: usize,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Deterministic standard normals (xorshift + Box-Muller).
struct Normal(u64);

impl Normal {
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn sample(&mut self) -> f32 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// One densification step. `grad_accum[i] / grad_count[i]` is gaussian `i`'s
/// mean NDC gradient norm; `prune_large` enables the world-scale prune.
pub fn densify(
    pop: &Population,
    grad_accum: &[f32],
    grad_count: &[f32],
    extent: f32,
    cfg: &DensifyConfig,
    prune_large: bool,
    seed: u64,
) -> (Population, DensifyReport) {
    let n = pop.len();
    let mut rng = Normal(seed.max(1));
    let mut out = Population {
        transforms: Group::with_capacity(10, n),
        opacity: Group::with_capacity(1, n),
        sh: Group::with_capacity(pop.sh.stride, n),
    };
    let mut report = DensifyReport {
        before: n,
        ..Default::default()
    };
    let can_grow = n < cfg.max_gaussians;
    let big = cfg.percent_dense * extent;
    for i in 0..n {
        let t = pop.transforms.row(i);
        let scale = Vector3::new(t[7].exp(), t[8].exp(), t[9].exp());
        let max_scale = scale.max();
        let opacity = sigmoid(pop.opacity.values[i]);
        let grad = if grad_count[i] > 0.0 {
            grad_accum[i] / grad_count[i]
        } else {
            0.0
        };
        let dense = can_grow && grad > cfg.grad_threshold;

        if dense && max_scale > big {
            // Split: two children drawn from the parent's gaussian, smaller.
            let q = UnitQuaternion::from_quaternion(Quaternion::new(t[3], t[4], t[5], t[6]));
            for _ in 0..2 {
                let local = Vector3::new(
                    rng.sample() * scale.x,
                    rng.sample() * scale.y,
                    rng.sample() * scale.z,
                );
                let m = Vector3::new(t[0], t[1], t[2]) + q * local;
                let child_ls = (scale / 1.6).map(f32::ln);
                let row = [
                    m.x, m.y, m.z, t[3], t[4], t[5], t[6], child_ls.x, child_ls.y, child_ls.z,
                ];
                out.transforms.push_new(&row);
                out.opacity.push_new(&[pop.opacity.values[i]]);
                out.sh.push_new(pop.sh.row(i));
            }
            report.split += 1;
            continue;
        }

        let prune =
            opacity < cfg.min_opacity || (prune_large && max_scale > cfg.max_world_scale * extent);
        if prune {
            report.pruned += 1;
            continue;
        }
        out.transforms.keep(&pop.transforms, i);
        out.opacity.keep(&pop.opacity, i);
        out.sh.keep(&pop.sh, i);
        if dense {
            // Clone: an identical copy that the optimiser will pull apart.
            out.transforms.push_new(t);
            out.opacity.push_new(&[pop.opacity.values[i]]);
            out.sh.push_new(pop.sh.row(i));
            report.cloned += 1;
        }
    }
    report.after = out.len();
    (out, report)
}

/// Opacity reset: clamp every opacity to at most `max_opacity` (Inria uses
/// 0.01), zeroing that group's moments.
pub fn reset_opacity(pop: &mut Population, max_opacity: f32) {
    let cap = (max_opacity / (1.0 - max_opacity)).ln();
    for v in &mut pop.opacity.values {
        *v = v.min(cap);
    }
    pop.opacity.m1.iter_mut().for_each(|x| *x = 0.0);
    pop.opacity.m2.iter_mut().for_each(|x| *x = 0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pop(scales: &[f32], opac: &[f32]) -> Population {
        let n = scales.len();
        let mut t = Vec::new();
        for (i, s) in scales.iter().enumerate() {
            t.extend_from_slice(&[
                i as f32,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                s.ln(),
                s.ln(),
                s.ln(),
            ]);
        }
        let g = |stride: usize, v: Vec<f32>| Group {
            stride,
            m1: vec![1.0; v.len()],
            m2: vec![2.0; v.len()],
            values: v,
        };
        Population {
            transforms: g(10, t),
            opacity: g(1, opac.iter().map(|o| (o / (1.0 - o)).ln()).collect()),
            sh: g(3, vec![0.5; 3 * n]),
        }
    }

    #[test]
    fn clone_split_prune() {
        // 0: small, high grad -> clone; 1: large, high grad -> split;
        // 2: low grad, transparent -> prune; 3: low grad -> keep.
        let p = pop(&[0.001, 1.0, 0.001, 0.001], &[0.5, 0.5, 0.001, 0.5]);
        let accum = [1.0, 1.0, 0.0, 0.0];
        let count = [1.0, 1.0, 1.0, 1.0];
        let (out, rep) = densify(&p, &accum, &count, 1.0, &DensifyConfig::default(), false, 1);
        assert_eq!((rep.cloned, rep.split, rep.pruned), (1, 1, 1));
        // kept 0 + its clone + 2 split children + kept 3
        assert_eq!(out.len(), 5);
        assert_eq!(rep.after, 5);
        // Kept rows carry their moments; new rows start at zero.
        assert_eq!(out.opacity.m1[0], 1.0);
        assert_eq!(out.opacity.m1[1], 0.0);
        // Split children are 1/1.6 the parent's scale.
        let child_ls = out.transforms.values[2 * 10 + 7];
        assert!((child_ls - (1.0f32 / 1.6).ln()).abs() < 1e-6);
    }

    #[test]
    fn reset_caps_opacity() {
        let mut p = pop(&[0.1, 0.1], &[0.9, 0.001]);
        reset_opacity(&mut p, 0.01);
        assert!((sigmoid(p.opacity.values[0]) - 0.01).abs() < 1e-6);
        assert!((sigmoid(p.opacity.values[1]) - 0.001).abs() < 1e-6);
        assert!(p.opacity.m1.iter().all(|&x| x == 0.0));
    }
}

/// brush's refine strategy (brush 0.3 `refine_if_needed`), an alternative to
/// the Inria clone / split / prune rule above. Every `refine_every` steps:
/// prune gaussians with opacity < `min_opacity`, a log-scale below -15 or a
/// centre more than 10 bound sizes away; replace them by splitting as many
/// gaussians sampled by opacity; and, until `growth_stop`, split a further
/// `growth_select_fraction` of those whose max refine weight (divided by
/// their visible count) exceeds `growth_grad_threshold`, sampled by that
/// weight. A split replaces the parent by two copies offset by
/// +-R (N(0, 0.5) * s), with scales / sqrt 2 and opacity 1 - sqrt(1 - a).
/// Nearly transparent gaussians also get mean noise every step.
#[derive(Debug, Clone)]
pub struct BrushRefineConfig {
    pub refine_every: usize,
    pub growth_grad_threshold: f32,
    pub growth_select_fraction: f32,
    pub growth_stop: usize,
    pub min_opacity: f32,
    pub max_gaussians: usize,
    pub mean_noise_weight: f32,
}

impl Default for BrushRefineConfig {
    fn default() -> Self {
        Self {
            refine_every: 200,
            growth_grad_threshold: 4e-5,
            growth_select_fraction: 0.1,
            growth_stop: 15_000,
            min_opacity: 2.0 / 255.0,
            max_gaussians: 10_000_000,
            mean_noise_weight: 40.0,
        }
    }
}
