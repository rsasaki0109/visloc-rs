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

use visloc_core::geometry::SE3;

/// Configuration for [`maximum_consistent_set`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcmConfig {
    /// Maximum SE(3)-tangent norm `‖log(cycle)‖` for two measurements to count
    /// as consistent. Mixes rotation (rad) and translation (in the trajectory's
    /// length unit); pick it a few × the expected odometry+measurement noise of
    /// a genuine cycle. A Mahalanobis χ² threshold with per-edge covariance is
    /// the textbook choice; this isotropic norm is the simplification used here.
    pub threshold: f64,
    /// Also require every kept loop to be individually consistent with the
    /// odometry (the single-loop cycle `z ⊖ odo`). When the odometry between a
    /// loop's endpoints is good (small drift) this is a strong, cheap
    /// pre-filter; with large accumulated drift it would reject genuine loops,
    /// so disable it and rely on pairwise consistency alone.
    pub require_individual: bool,
}

impl Default for PcmConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            require_individual: true,
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
                match individual_residual(m, odometry) {
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
            let consistent =
                pairwise_residual(&loops[candidates[a]], &loops[candidates[b]], odometry)
                    .map(|r| r <= config.threshold)
                    .unwrap_or(false);
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
}
