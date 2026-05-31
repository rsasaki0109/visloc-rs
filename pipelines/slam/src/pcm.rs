//! Pairwise Consistency Maximization (PCM) — a front-end outlier screen for
//! loop closures, run *before* the pose-graph back-end.
//!
//! Robust back-end solvers (GNC, switchable constraints, robust kernels) reject
//! outliers *during* optimization, but they have two failure modes seen on real
//! data (see the `online_slam_kitti_loop_demo` findings): a non-robust
//! initializer (e.g. chordal rotation averaging) is corrupted by a handful of
//! gross wrong loops before the robust solve even starts, and a wrong loop can
//! be indistinguishable from a genuine one by residual alone when the
//! initialization is poor. PCM (Mangelson et al., *Pairwise Consistent
//! Measurement Set Maximization*, ICRA 2018 — the Kimera-RPGO front-end) attacks
//! the problem combinatorially instead: it keeps only the largest subset of loop
//! closures that are *mutually geometrically consistent*, so the back-end ever
//! sees a clean set.
//!
//! Two loop closures `k: a→b` and `l: c→d` are pairwise consistent if the cycle
//! they induce together with the odometry connecting their endpoints composes to
//! (approximately) the identity:
//!
//! ```text
//!   a --z_k--> b --odo--> d --z_l⁻¹--> c --odo--> a   ≈  identity
//! ```
//!
//! The residual is `‖log(cycle)‖` on the SE(3) tangent; a pair is consistent
//! when it is below [`PcmConfig::threshold`]. (The paper uses a Mahalanobis χ²
//! test with per-edge covariance; this isotropic SE(3)-tangent norm is the
//! simplification used here, matching the isotropic edge weights the rest of the
//! pose-graph code uses.) Building one node per loop closure and one edge per
//! consistent pair, the maximum set of mutually-consistent loops is the
//! **maximum clique** of that consistency graph. Random wrong loops are
//! consistent with almost nothing, so they fall outside the large genuine
//! clique and are dropped.
//!
//! This module is pure geometry + combinatorics with no [`crate::PoseGraph`]
//! dependency, mirroring [`crate::gnc`].

use std::collections::BTreeMap;

use nalgebra::{Matrix6, Vector6};
use visloc_core::geometry::SE3;

/// First-order SE(3) covariance model for the **Mahalanobis** PCM test. When
/// supplied via [`PcmConfig::noise`], the cycle residual is normalized by the
/// covariance compounded along the cycle's legs — the textbook PCM χ² test
/// (Mangelson et al.) — instead of the raw isotropic SE(3)-tangent norm.
///
/// This directly lifts **recall under drift**: a genuine cycle that spans many
/// drifted odometry edges has a large raw residual *but also a large expected
/// covariance*, so its Mahalanobis distance stays small and it passes; a wrong
/// loop's residual is inconsistent with even that inflated covariance and is
/// still rejected. The isotropic-norm test (no noise model) instead drops the
/// long-span genuine cycles, the failure mode seen on KITTI cross-session
/// screening.
///
/// Covariances are 6×6 in the SE(3) tangent layout `[ρ; ω]` (translation first,
/// rotation second — matching [`SE3::log`] / [`SE3::adjoint`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcmNoiseModel {
    /// Covariance contributed by a single odometry edge. An odometry leg
    /// spanning `n` sequential keyframes accumulates `n ×` this (an isotropic
    /// random-walk model: the within-leg adjoint rotation between consecutive
    /// edges is neglected, while the dominant between-leg adjoints of the full
    /// four-leg cycle *are* applied). Assumes contiguous sequential keyframe ids
    /// so the edge count of `odo(x→y)` is `|x − y|`.
    pub odometry_per_edge: Matrix6<f64>,
    /// Covariance of one loop-closure measurement `z` (a direct observation, so
    /// independent of how far apart its endpoints are in the sequence).
    pub loop_measurement: Matrix6<f64>,
}

impl PcmNoiseModel {
    /// Isotropic model: per-edge odometry variance `(odo_rot, odo_trans)` and
    /// per-measurement variance `(loop_rot, loop_trans)` (rotation rad², trans
    /// length²), as diagonal `[ρ; ω]` covariances.
    pub fn isotropic(odo_rot: f64, odo_trans: f64, loop_rot: f64, loop_trans: f64) -> Self {
        let diag = |rot: f64, trans: f64| {
            let mut m = Matrix6::zeros();
            for i in 0..3 {
                m[(i, i)] = trans;
                m[(i + 3, i + 3)] = rot;
            }
            m
        };
        Self {
            odometry_per_edge: diag(odo_rot, odo_trans),
            loop_measurement: diag(loop_rot, loop_trans),
        }
    }
}

/// Configuration for [`maximum_consistent_set`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcmConfig {
    /// Consistency threshold. Without [`Self::noise`] it is the maximum
    /// SE(3)-tangent norm `‖log(cycle)‖` (mixes rotation rad and translation
    /// length — pick it a few × the expected noise of a genuine cycle). With a
    /// noise model it is instead a unitless **Mahalanobis distance** bound
    /// `√(ξᵀ Σ⁻¹ ξ)`, i.e. `√` of a χ²(6) quantile (e.g. `√16.8 ≈ 4.1` for the
    /// 0.99 quantile) — scale-free, so it no longer conflates rad with metres.
    pub threshold: f64,
    /// Also require every kept loop to be individually consistent with the
    /// odometry (the single-loop cycle `z ⊖ odo`). When the odometry between a
    /// loop's endpoints is good (small drift) this is a strong, cheap
    /// pre-filter; with large accumulated drift it would reject genuine loops,
    /// so disable it and rely on pairwise consistency alone.
    pub require_individual: bool,
    /// When set, use the covariance-aware Mahalanobis cycle test instead of the
    /// isotropic SE(3)-tangent norm — see [`PcmNoiseModel`]. `None` preserves the
    /// original isotropic behaviour.
    pub noise: Option<PcmNoiseModel>,
}

impl Default for PcmConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            require_individual: true,
            noise: None,
        }
    }
}

/// A loop-closure measurement: the relative transform `relative` taking frame
/// `from` to frame `to`, in the same `world_to_camera` convention as
/// [`crate::relative_world_to_camera`] / [`crate::PoseGraphEdge::measurement`].
#[derive(Debug, Clone, PartialEq)]
pub struct LoopMeasurement {
    pub from: u64,
    pub to: u64,
    pub relative: SE3,
}

/// Odometry relative transform `rel(x → y)` from the per-keyframe
/// `world_to_camera` poses, i.e. `T_y ∘ T_x⁻¹`. `None` if either id is absent.
fn odometry_relative(odometry: &BTreeMap<u64, SE3>, x: u64, y: u64) -> Option<SE3> {
    let tx = odometry.get(&x)?;
    let ty = odometry.get(&y)?;
    Some(ty.compose(&tx.inverse()))
}

/// Residual of the single-loop cycle `a → b → a` (the loop measurement against
/// the odometry estimate of the same relative pose): `‖log(odo(b→a) ∘ z)‖`.
/// `None` if the odometry lacks either endpoint.
pub fn individual_residual(m: &LoopMeasurement, odometry: &BTreeMap<u64, SE3>) -> Option<f64> {
    let odo_ba = odometry_relative(odometry, m.to, m.from)?;
    Some(odo_ba.compose(&m.relative).log().norm())
}

/// Residual of the two-loop cycle `a → b → d → c → a` formed by `k: a→b`,
/// odometry `b→d`, `l⁻¹: d→c`, odometry `c→a`. `None` if any endpoint is
/// missing from the odometry.
pub fn pairwise_residual(
    k: &LoopMeasurement,
    l: &LoopMeasurement,
    odometry: &BTreeMap<u64, SE3>,
) -> Option<f64> {
    let odo_bd = odometry_relative(odometry, k.to, l.to)?;
    let odo_ca = odometry_relative(odometry, l.from, k.from)?;
    // cycle = odo(c→a) ∘ z_l⁻¹ ∘ odo(b→d) ∘ z_k  (rightmost applied first).
    let cycle = odo_ca.compose(&l.relative.inverse().compose(&odo_bd.compose(&k.relative)));
    Some(cycle.log().norm())
}

/// Edge count of an odometry leg `x → y`, assuming contiguous sequential
/// keyframe ids (so the number of sequential edges traversed is `|x − y|`).
/// Clamped to ≥ 1 so even an adjacent leg carries one edge of covariance.
fn leg_edges(x: u64, y: u64) -> f64 {
    ((x as i64 - y as i64).unsigned_abs() as f64).max(1.0)
}

/// First-order covariance compounding of a cycle expressed as an ordered product
/// of legs `legs[0] ∘ legs[1] ∘ … ∘ legs[L-1]`, each `(mean, cov)` with `cov` in
/// the leg's own right-tangent. Returns `(cycle_mean, cycle_cov)`.
///
/// For `T = A ∘ B` with right-perturbations `T = T̄ exp(ξ)`, `ξ_T = Ad(B̄⁻¹) ξ_A +
/// ξ_B`, so each leg's covariance is rotated by the adjoint of the inverse of
/// everything to its right. Accumulating right-to-left:
/// `Σ += Ad(R⁻¹) Σ_leg Ad(R⁻¹)ᵀ`, `R ← leg ∘ R`.
fn compound_cycle(legs: &[(SE3, Matrix6<f64>)]) -> (SE3, Matrix6<f64>) {
    let mut cov = Matrix6::zeros();
    let mut right = SE3::identity();
    for (mean, leg_cov) in legs.iter().rev() {
        let ad = right.inverse().adjoint();
        cov += ad * leg_cov * ad.transpose();
        right = mean.compose(&right);
    }
    (right, cov)
}

/// Mahalanobis distance `√(ξᵀ Σ⁻¹ ξ)` of a compounded cycle (`ξ = log(mean)`),
/// with a tiny diagonal jitter for numerical safety. `None` if `Σ` is singular.
fn mahalanobis(mean: &SE3, cov: &Matrix6<f64>) -> Option<f64> {
    let xi: Vector6<f64> = mean.log();
    let regularized = cov + Matrix6::identity() * 1e-12;
    let inv = regularized.try_inverse()?;
    let d2 = (xi.transpose() * inv * xi)[(0, 0)];
    Some(d2.max(0.0).sqrt())
}

/// Mahalanobis residual of the single-loop cycle `a → b → a`, with the odometry
/// leg's covariance scaled by the number of edges it spans. `None` if the
/// odometry lacks either endpoint.
pub fn individual_mahalanobis(
    m: &LoopMeasurement,
    odometry: &BTreeMap<u64, SE3>,
    noise: &PcmNoiseModel,
) -> Option<f64> {
    let odo_ba = odometry_relative(odometry, m.to, m.from)?;
    // cycle = odo(b→a) ∘ z   (G1 = odo_ba, G2 = z).
    let legs = [
        (odo_ba, noise.odometry_per_edge * leg_edges(m.to, m.from)),
        (m.relative.clone(), noise.loop_measurement),
    ];
    let (mean, cov) = compound_cycle(&legs);
    mahalanobis(&mean, &cov)
}

/// Mahalanobis residual of the two-loop cycle `a → b → d → c → a`, with each
/// odometry leg's covariance scaled by its edge span and the inverted-measurement
/// leg's covariance carried through its adjoint. `None` if any endpoint is
/// missing from the odometry.
pub fn pairwise_mahalanobis(
    k: &LoopMeasurement,
    l: &LoopMeasurement,
    odometry: &BTreeMap<u64, SE3>,
    noise: &PcmNoiseModel,
) -> Option<f64> {
    let odo_bd = odometry_relative(odometry, k.to, l.to)?;
    let odo_ca = odometry_relative(odometry, l.from, k.from)?;
    // For z_l⁻¹: if z_l = z̄_l exp(ξ) with cov Σ, then z_l⁻¹ = z̄_l⁻¹ exp(−Ad(z̄_l)ξ),
    // so its covariance is Ad(z_l) Σ Ad(z_l)ᵀ.
    let ad_l = l.relative.adjoint();
    let cov_l_inv = ad_l * noise.loop_measurement * ad_l.transpose();
    // Legs in composition order: odo_ca ∘ z_l⁻¹ ∘ odo_bd ∘ z_k.
    let legs = [
        (odo_ca, noise.odometry_per_edge * leg_edges(l.from, k.from)),
        (l.relative.inverse(), cov_l_inv),
        (odo_bd, noise.odometry_per_edge * leg_edges(k.to, l.to)),
        (k.relative.clone(), noise.loop_measurement),
    ];
    let (mean, cov) = compound_cycle(&legs);
    mahalanobis(&mean, &cov)
}

/// Run PCM: return the indices (into `loops`) of the maximum mutually-consistent
/// subset. Loops whose endpoints are missing from `odometry` are dropped. When
/// [`PcmConfig::require_individual`] is set, loops that fail the odometry
/// self-check are dropped before the clique search.
pub fn maximum_consistent_set(
    loops: &[LoopMeasurement],
    odometry: &BTreeMap<u64, SE3>,
    config: &PcmConfig,
) -> Vec<usize> {
    // Candidate loops: present in the odometry and (optionally) individually
    // consistent with it.
    let candidates: Vec<usize> = (0..loops.len())
        .filter(|&i| {
            let m = &loops[i];
            if !odometry.contains_key(&m.from) || !odometry.contains_key(&m.to) {
                return false;
            }
            if config.require_individual {
                let residual = match &config.noise {
                    Some(n) => individual_mahalanobis(m, odometry, n),
                    None => individual_residual(m, odometry),
                };
                match residual {
                    Some(r) => r <= config.threshold,
                    None => false,
                }
            } else {
                true
            }
        })
        .collect();

    if candidates.is_empty() {
        return Vec::new();
    }

    // Consistency graph over the candidates (local indices 0..m map back to
    // `candidates`). An edge means the pair is pairwise consistent.
    let m = candidates.len();
    let mut adj = vec![vec![false; m]; m];
    for a in 0..m {
        for b in (a + 1)..m {
            let (ka, kb) = (&loops[candidates[a]], &loops[candidates[b]]);
            let residual = match &config.noise {
                Some(n) => pairwise_mahalanobis(ka, kb, odometry, n),
                None => pairwise_residual(ka, kb, odometry),
            };
            let consistent = residual.map(|r| r <= config.threshold).unwrap_or(false);
            adj[a][b] = consistent;
            adj[b][a] = consistent;
        }
    }

    // Maximum clique of the consistency graph = largest mutually-consistent set.
    let mut best: Vec<usize> = Vec::new();
    let mut r: Vec<usize> = Vec::new();
    let p: Vec<usize> = (0..m).collect();
    let x: Vec<usize> = Vec::new();
    bron_kerbosch(&mut r, p, x, &adj, &mut best);

    // Map local indices back to indices into `loops`, sorted for determinism.
    let mut kept: Vec<usize> = best.into_iter().map(|li| candidates[li]).collect();
    kept.sort_unstable();
    kept
}

/// Bron–Kerbosch maximum-clique enumeration with pivoting. `best` accumulates
/// the largest clique seen. Loop-closure sets are small (tens of edges), so the
/// exponential worst case is irrelevant in practice.
fn bron_kerbosch(
    r: &mut Vec<usize>,
    p: Vec<usize>,
    x: Vec<usize>,
    adj: &[Vec<bool>],
    best: &mut Vec<usize>,
) {
    if p.is_empty() && x.is_empty() {
        if r.len() > best.len() {
            *best = r.clone();
        }
        return;
    }
    // Bound: this branch cannot beat the incumbent.
    if r.len() + p.len() <= best.len() {
        return;
    }
    // Pivot u from P ∪ X maximizing |P ∩ N(u)| (Tomita pivoting).
    let pivot = p
        .iter()
        .chain(x.iter())
        .copied()
        .max_by_key(|&u| p.iter().filter(|&&v| adj[u][v]).count());

    let candidates: Vec<usize> = match pivot {
        Some(u) => p.iter().copied().filter(|&v| !adj[u][v]).collect(),
        None => p.clone(),
    };

    let mut p = p;
    let mut x = x;
    for v in candidates {
        let p_next: Vec<usize> = p.iter().copied().filter(|&w| adj[v][w]).collect();
        let x_next: Vec<usize> = x.iter().copied().filter(|&w| adj[v][w]).collect();
        r.push(v);
        bron_kerbosch(r, p_next, x_next, adj, best);
        r.pop();
        p.retain(|&w| w != v);
        x.push(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    /// A straight trajectory of `n` keyframes, each 1 m forward along +z, no
    /// rotation. `world_to_camera` = inverse of the world pose; for a pure
    /// translation the camera-from-world translation is `-center`.
    fn straight_odometry(n: u64) -> BTreeMap<u64, SE3> {
        let mut odo = BTreeMap::new();
        for i in 0..n {
            let center = Vector3::new(0.0, 0.0, i as f64);
            odo.insert(i, SE3::new(UnitQuaternion::identity(), -center));
        }
        odo
    }

    /// The genuine (truth-relative) loop measurement between two keyframes.
    fn genuine(odo: &BTreeMap<u64, SE3>, a: u64, b: u64) -> LoopMeasurement {
        LoopMeasurement {
            from: a,
            to: b,
            relative: odometry_relative(odo, a, b).unwrap(),
        }
    }

    #[test]
    fn genuine_loops_are_individually_and_pairwise_consistent() {
        let odo = straight_odometry(10);
        let l1 = genuine(&odo, 0, 5);
        let l2 = genuine(&odo, 1, 6);
        assert!(individual_residual(&l1, &odo).unwrap() < 1e-9);
        assert!(pairwise_residual(&l1, &l2, &odo).unwrap() < 1e-9);
    }

    #[test]
    fn wrong_loop_is_inconsistent_with_genuine_loops() {
        let odo = straight_odometry(10);
        let l1 = genuine(&odo, 0, 8);
        let l2 = genuine(&odo, 1, 7);
        // A wrong loop claiming frames 2 and 9 (7 m apart) are co-located.
        let wrong = LoopMeasurement {
            from: 2,
            to: 9,
            relative: SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0)),
        };
        assert!(individual_residual(&wrong, &odo).unwrap() > 1.0);
        assert!(pairwise_residual(&l1, &wrong, &odo).unwrap() > 1.0);
        assert!(pairwise_residual(&l2, &wrong, &odo).unwrap() > 1.0);
    }

    #[test]
    fn pcm_keeps_the_genuine_clique_and_drops_wrong_loops() {
        let odo = straight_odometry(12);
        let loops = vec![
            genuine(&odo, 0, 6),
            genuine(&odo, 1, 7),
            genuine(&odo, 2, 8),
            // Two random wrong loops claiming co-location of far-apart frames.
            LoopMeasurement {
                from: 0,
                to: 11,
                relative: SE3::new(UnitQuaternion::identity(), Vector3::new(0.2, 0.1, 0.0)),
            },
            LoopMeasurement {
                from: 3,
                to: 10,
                relative: SE3::new(
                    UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.05),
                    Vector3::new(-0.3, 0.0, 0.1),
                ),
            },
        ];
        let kept = maximum_consistent_set(&loops, &odo, &PcmConfig::default());
        assert_eq!(kept, vec![0, 1, 2], "PCM keeps the genuine loops only");
    }

    #[test]
    fn pcm_with_individual_filter_disabled_still_drops_wrong_via_clique() {
        let odo = straight_odometry(12);
        let loops = vec![
            genuine(&odo, 0, 6),
            genuine(&odo, 1, 7),
            genuine(&odo, 2, 8),
            LoopMeasurement {
                from: 0,
                to: 11,
                relative: SE3::new(UnitQuaternion::identity(), Vector3::new(0.2, 0.1, 0.0)),
            },
        ];
        let config = PcmConfig {
            threshold: 1.0,
            require_individual: false,
            noise: None,
        };
        let kept = maximum_consistent_set(&loops, &odo, &config);
        // The three genuine loops form the largest mutually-consistent clique;
        // the lone wrong loop is consistent with none of them.
        assert_eq!(kept, vec![0, 1, 2]);
    }

    #[test]
    fn empty_and_singleton_inputs() {
        let odo = straight_odometry(4);
        assert!(maximum_consistent_set(&[], &odo, &PcmConfig::default()).is_empty());
        let one = vec![genuine(&odo, 0, 3)];
        assert_eq!(
            maximum_consistent_set(&one, &odo, &PcmConfig::default()),
            vec![0]
        );
    }

    #[test]
    fn loops_referencing_missing_keyframes_are_dropped() {
        let odo = straight_odometry(5);
        let loops = vec![
            genuine(&odo, 0, 4),
            LoopMeasurement {
                from: 0,
                to: 99, // not in odometry
                relative: SE3::identity(),
            },
        ];
        let kept = maximum_consistent_set(&loops, &odo, &PcmConfig::default());
        assert_eq!(kept, vec![0]);
    }

    /// A straight trajectory whose odometry estimate accumulates a constant yaw
    /// drift per edge: `odo` is the drifted `world_to_camera`, `truth` the exact
    /// one. Returns `(drifted_odo, truth_odo)`.
    fn drifted_odometry(n: u64, yaw_per_edge: f64) -> (BTreeMap<u64, SE3>, BTreeMap<u64, SE3>) {
        let truth = straight_odometry(n);
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw_per_edge);
        let mut drifted = BTreeMap::new();
        drifted.insert(0, truth[&0].clone());
        for i in 1..n {
            // Truth relative edge (i-1 → i), yaw-perturbed, integrated forward.
            let edge = odometry_relative(&truth, i - 1, i).unwrap();
            let noisy = SE3::new(yaw * edge.rotation, edge.translation);
            let prev = drifted[&(i - 1)].clone();
            drifted.insert(i, noisy.compose(&prev));
        }
        (drifted, truth)
    }

    /// With zero noise the Mahalanobis residual is just `‖log(cycle)‖` scaled by
    /// the (isotropic) inverse covariance — so a genuine cycle is ~0 and a wrong
    /// one is large, like the plain residual but unitless.
    #[test]
    fn mahalanobis_on_exact_odometry_separates_genuine_from_wrong() {
        let odo = straight_odometry(12);
        let noise = PcmNoiseModel::isotropic(1e-4, 1e-3, 1e-6, 1e-6);
        let g1 = genuine(&odo, 0, 6);
        let g2 = genuine(&odo, 1, 7);
        let wrong = LoopMeasurement {
            from: 2,
            to: 11,
            relative: SE3::new(UnitQuaternion::identity(), Vector3::new(0.2, 0.0, 0.0)),
        };
        assert!(pairwise_mahalanobis(&g1, &g2, &odo, &noise).unwrap() < 1e-3);
        assert!(pairwise_mahalanobis(&g1, &wrong, &odo, &noise).unwrap() > 5.0);
    }

    /// The headline: with accumulated odometry drift, two genuine loops whose
    /// endpoints are spread several edges apart induce a pairwise cycle that spans
    /// many drifted edges — its raw `‖log(cycle)‖` exceeds an isotropic threshold,
    /// so the isotropic test drops part of the genuine clique. The Mahalanobis
    /// test inflates the cycle covariance by the edge span, so the same genuine
    /// pairs stay consistent and the full clique is kept, while the wrong loop
    /// (whose residual is inconsistent with even the inflated covariance) is
    /// dropped. This is the cross-session-merge recall fix, in miniature.
    #[test]
    fn mahalanobis_keeps_drift_spanning_genuine_loops_the_isotropic_norm_drops() {
        let (odo, truth) = drifted_odometry(40, 0.008);
        // Genuine loops, truth-relative, with starts 8 edges apart. The two
        // adjacent pairs induce an 8-edge cycle (raw residual ≈ 1.28); the outer
        // pair (0,20)-(16,36) induces a 16-edge cycle (raw ≈ 2.56) — double the
        // drift, so a single isotropic threshold cannot admit all three at once.
        let genuine_pairs = [(0u64, 20u64), (8, 28), (16, 36)];
        let mut loops: Vec<LoopMeasurement> = genuine_pairs
            .iter()
            .map(|&(a, b)| LoopMeasurement {
                from: a,
                to: b,
                relative: odometry_relative(&truth, a, b).unwrap(),
            })
            .collect();
        // A wrong loop: frames 2 and 38 (36 m apart) asserted near-coincident.
        loops.push(LoopMeasurement {
            from: 2,
            to: 38,
            relative: SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0)),
        });

        // Isotropic norm, threshold between the 8-edge (≈1.28) and 16-edge
        // (≈2.56) cycle residuals: the outer pair is cut, so the genuine
        // consistency graph is a path 0—1—2 missing the 0—2 edge → max clique 2.
        let iso = maximum_consistent_set(
            &loops,
            &odo,
            &PcmConfig {
                threshold: 1.5,
                require_individual: false,
                noise: None,
            },
        );
        assert!(
            iso.len() < genuine_pairs.len(),
            "isotropic norm drops a drift-spanning genuine loop, kept {iso:?}"
        );

        // Mahalanobis: each cycle's covariance scales with its edge span, so the
        // 16-edge pair is as consistent as the 8-edge ones — all three genuine
        // loops form the clique and the wrong loop is still rejected.
        let noise = PcmNoiseModel::isotropic(2e-3, 8e-2, 1e-8, 1e-8);
        let maha = maximum_consistent_set(
            &loops,
            &odo,
            &PcmConfig {
                threshold: 4.0,
                require_individual: false,
                noise: Some(noise),
            },
        );
        assert_eq!(
            maha,
            vec![0, 1, 2],
            "Mahalanobis keeps all {} genuine loops, drops the wrong one",
            genuine_pairs.len()
        );
    }

    /// Precision under the Mahalanobis test: span-scaled covariance does NOT let a
    /// subtly-corrupted bridge slip through when a genuine consensus surrounds it.
    /// A corrupted loop whose endpoints sit ONE edge from a genuine loop's induces
    /// a near-zero-span cycle against it, so its covariance there is tiny and the
    /// corruption (a 3 m relative-pose offset) is caught — it cannot join the
    /// genuine clique even though its cycles against the FAR genuine loops have
    /// large, forgiving covariance. This is the demo's corrupted-bridge stress in
    /// miniature: covariance inflation lifts recall without costing precision when
    /// the outlier is bracketed by nearby genuine measurements.
    #[test]
    fn mahalanobis_rejects_a_corrupted_bridge_bracketed_by_genuine_consensus() {
        let (odo, truth) = drifted_odometry(40, 0.008);
        let genuine = |a, b| LoopMeasurement {
            from: a,
            to: b,
            relative: odometry_relative(&truth, a, b).unwrap(),
        };
        // Four genuine loops form the consensus clique.
        let mut loops = vec![
            genuine(0, 20),
            genuine(4, 24),
            genuine(8, 28),
            genuine(16, 36),
        ];
        // A corrupted loop: endpoints (9, 29) sit one edge from genuine (8, 28),
        // but the reported relative pose is offset by 3 m.
        let offset = SE3::new(UnitQuaternion::identity(), Vector3::new(3.0, 0.0, 0.0));
        loops.push(LoopMeasurement {
            from: 9,
            to: 29,
            relative: offset.compose(&odometry_relative(&truth, 9, 29).unwrap()),
        });

        let noise = PcmNoiseModel::isotropic(2e-3, 8e-2, 1e-8, 1e-8);
        let kept = maximum_consistent_set(
            &loops,
            &odo,
            &PcmConfig {
                threshold: 4.0,
                require_individual: false,
                noise: Some(noise),
            },
        );
        assert_eq!(
            kept,
            vec![0, 1, 2, 3],
            "the four genuine loops are kept and the corrupted bridge (index 4) is dropped"
        );
    }
}
