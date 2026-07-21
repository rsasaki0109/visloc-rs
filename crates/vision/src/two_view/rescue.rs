//! Rescue-matching: bridge disconnected components of a verified view graph.
//!
//! M5 in `docs/colmap_port_plan.md` diagnoses ETH3D `courtyard`'s 14/38
//! registration ceiling as, in part, a genuinely disconnected verified-pair
//! graph — images `{0..24}` and `{25..37}` never verify a single ≥30-inlier
//! two-view pair against each other, at any pair budget the M3/M4 milestones
//! tried (`docs/colmap_port_plan.md`'s "M4 results", diagnosis step 2). No
//! registration policy in `pipelines/slam/src/incremental_sfm.rs` can place an
//! image into a reconstruction with zero verified correspondences to it — this
//! is a frontend coverage problem, not a mapper problem, so the fix has to
//! live upstream of the mapper: propose *more* candidate pairs specifically
//! across whatever components the standard pipeline already produced, using a
//! deliberately relaxed matching profile, then hand every candidate to the
//! *same* full COLMAP-style verifier (`colmap_verification::
//! TwoViewGeometryVerifier`) everything else goes through — so a relaxed
//! profile can only ever *propose* a bridge, never *admit* an unverified one
//! (the M1.1 lesson: loosening thresholds is only safe when a real classifier
//! still gates what gets kept).
//!
//! This module provides the two pieces of that pass that are pure graph
//! algorithms, independent of any particular matcher or verifier:
//! - [`connected_components`] — the disconnection detector: partitions image
//!   indices into components given the current verified-pair edge list.
//! - [`generate_bridge_candidates`] — the candidate-pair generator: given a
//!   component partition and a retrieval-style similarity score, proposes
//!   *only* cross-component pairs (same-component pairs already have a path
//!   and are not the problem), ranked by descending similarity and capped to
//!   a budget.
//!
//! The actual re-matching (relaxed ratio / no cross-check) and re-verification
//! (full `TwoViewGeometryVerifier`) are ordinary calls to
//! `crate::matching`/`colmap_verification` types already in this crate — no
//! new matcher or verifier is needed, only a looser *configuration* of the
//! existing ones — so the demo (`examples/unordered_sfm_demo.rs`'s
//! `--rescue-bridging`) composes them directly rather than this module owning
//! a redundant wrapper. [`rescue_bridges_two_disconnected_components`] (test)
//! demonstrates the full composition end-to-end on a synthetic fixture so the
//! claim "a relaxed profile can bridge a component a strict one cannot" is
//! pinned somewhere runnable without a real dataset.

use std::collections::HashMap;

/// Union-find (disjoint-set) with path compression + union by size — the
/// standard near-linear connected-components primitive. Kept private: the
/// only public surface this module needs is [`connected_components`] itself.
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }
}

/// Partitions `0..num_images` into connected components given the current
/// verified-pair edge list `(image_i, image_j)`. This is the M5
/// "disconnection detector": the exact question `docs/colmap_port_plan.md`'s
/// M4 diagnosis answered by hand with a Python union-find over a dumped pair
/// log (`E:/visloc_archive/colmap_m4_20260717/diagnosis/courtyard_pairs.log`)
/// — this function makes that check a reusable, tested primitive instead of a
/// one-off script.
///
/// Returns each component as a sorted `Vec<usize>` of image indices; the
/// components themselves are ordered by ascending smallest member (so the
/// result is deterministic and independent of `edges`' iteration order).
/// Images with no edge at all form their own singleton component — the same
/// convention COLMAP's `CorrespondenceGraph` uses for a never-matched image
/// (it still "exists", just with zero correspondences).
pub fn connected_components(num_images: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut uf = UnionFind::new(num_images);
    for &(i, j) in edges {
        if i < num_images && j < num_images && i != j {
            uf.union(i, j);
        }
    }
    let mut by_root: HashMap<usize, Vec<usize>> = HashMap::new();
    for image in 0..num_images {
        let root = uf.find(image);
        by_root.entry(root).or_default().push(image);
    }
    let mut components: Vec<Vec<usize>> = by_root.into_values().collect();
    for component in &mut components {
        component.sort_unstable();
    }
    components.sort_by_key(|component| component[0]);
    components
}

/// Options for [`generate_bridge_candidates`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BridgeCandidateOptions {
    /// Maximum number of candidate pairs to return, globally across every
    /// pair of components (not per-component-pair) — the "budget-capped" cap
    /// the M5 brief asks for, so a rescue pass on a large collection cannot
    /// accidentally degenerate into an all-pairs cross-component rematch.
    pub max_candidates: usize,
}

impl Default for BridgeCandidateOptions {
    fn default() -> Self {
        Self {
            max_candidates: 200,
        }
    }
}

/// Generates candidate bridge pairs `(i, j)` with `i < j` — **cross-component
/// only** (both endpoints in different entries of `components`; same-
/// component pairs are never proposed, since they already have a path and
/// are not the disconnection this pass targets) — ranked by descending
/// `similarity(i, j)` (a retrieval-style score; the caller typically supplies
/// VLAD cosine similarity or vocab-tree TF-IDF score, whichever pair source
/// the main pipeline used) and capped to `options.max_candidates`.
///
/// `components` is normally [`connected_components`]'s own output, but any
/// partition of `0..num_images` works (each inner `Vec` is one component;
/// membership, not order, is what matters here).
pub fn generate_bridge_candidates(
    components: &[Vec<usize>],
    similarity: impl Fn(usize, usize) -> f32,
    options: &BridgeCandidateOptions,
) -> Vec<(usize, usize)> {
    let mut scored: Vec<((usize, usize), f32)> = Vec::new();
    for (a, comp_a) in components.iter().enumerate() {
        for comp_b in components.iter().skip(a + 1) {
            for &i in comp_a {
                for &j in comp_b {
                    let (lo, hi) = (i.min(j), i.max(j));
                    scored.push(((lo, hi), similarity(i, j)));
                }
            }
        }
    }
    // Descending similarity; ties broken by pair order for determinism
    // (no HashMap-iteration-order dependence — `components` here came from
    // `connected_components`'s already-sorted output, and this sort is
    // stable, so equal-similarity runs keep that order).
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored
        .into_iter()
        .take(options.max_candidates)
        .map(|(pair, _)| pair)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two cleanly disconnected triangles (0-1-2 and 3-4-5) must resolve to
    /// exactly two components, each holding its own three images.
    #[test]
    fn detects_two_disconnected_components() {
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let components = connected_components(6, &edges);
        assert_eq!(components.len(), 2);
        assert_eq!(components[0], vec![0, 1, 2]);
        assert_eq!(components[1], vec![3, 4, 5]);
    }

    /// A single connected chain (0-1-2-3) is one component, matching the
    /// union-find's own transitive-closure semantics
    /// (`correspondence_graph.rs`'s `transitivity_parameter_bounds_the_
    /// closure_depth` fixture uses the identical chain shape).
    #[test]
    fn fully_connected_graph_is_one_component() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let components = connected_components(4, &edges);
        assert_eq!(components, vec![vec![0, 1, 2, 3]]);
    }

    /// An image with zero verified pairs is still reported — as its own
    /// singleton component — never silently dropped. Mirrors
    /// `CorrespondenceGraph`'s convention that a never-matched image still
    /// exists with zero correspondences (see that module's doc).
    #[test]
    fn isolated_image_is_a_singleton_component() {
        let edges = vec![(0, 1)];
        let components = connected_components(3, &edges);
        assert_eq!(components.len(), 2);
        assert_eq!(components[0], vec![0, 1]);
        assert_eq!(components[1], vec![2]);
    }

    /// Empty edge list: every image is isolated, `n` singleton components.
    #[test]
    fn no_edges_gives_n_singleton_components() {
        let components = connected_components(3, &[]);
        assert_eq!(components, vec![vec![0], vec![1], vec![2]]);
    }

    /// Self-loops and out-of-range edges (defensive — no caller in this repo
    /// should produce either) are ignored rather than panicking.
    #[test]
    fn self_loops_and_out_of_range_edges_are_ignored() {
        let edges = vec![(0, 0), (0, 1), (5, 6)];
        let components = connected_components(3, &edges);
        assert_eq!(components, vec![vec![0, 1], vec![2]]);
    }

    /// The core candidate-generation contract: only cross-component pairs
    /// ever appear, regardless of how similarity is scored — this is the
    /// property the rescue pass depends on to avoid re-proposing pairs that
    /// were never the problem (same-component pairs already have a path).
    #[test]
    fn candidates_are_cross_component_only() {
        let components = vec![vec![0usize, 1, 2], vec![3, 4, 5]];
        // Uniform similarity: with no ranking signal, every cross-component
        // pair should still appear exactly once, and no within-component
        // pair ever should.
        let candidates = generate_bridge_candidates(
            &components,
            |_, _| 1.0,
            &BridgeCandidateOptions {
                max_candidates: usize::MAX,
            },
        );
        assert_eq!(candidates.len(), 9, "3x3 cross-component pairs expected");
        for &(i, j) in &candidates {
            let comp_of = |x: usize| components.iter().position(|c| c.contains(&x)).unwrap();
            assert_ne!(
                comp_of(i),
                comp_of(j),
                "pair ({i}, {j}) is within one component, not a bridge candidate"
            );
        }
    }

    /// Ranking: candidates come back in descending similarity order.
    #[test]
    fn candidates_are_ranked_by_descending_similarity() {
        let components = vec![vec![0usize, 1], vec![2usize, 3]];
        // (0,2) and (1,3) are the "true" bridge (high score); (0,3) and (1,2)
        // are noise (low score).
        let similarity = |i: usize, j: usize| -> f32 {
            match (i.min(j), i.max(j)) {
                (0, 2) => 0.9,
                (1, 3) => 0.8,
                (0, 3) => 0.2,
                (1, 2) => 0.1,
                _ => 0.0,
            }
        };
        let candidates = generate_bridge_candidates(
            &components,
            similarity,
            &BridgeCandidateOptions {
                max_candidates: usize::MAX,
            },
        );
        assert_eq!(candidates, vec![(0, 2), (1, 3), (0, 3), (1, 2)]);
    }

    /// Budget cap: only the top `max_candidates` (by similarity) are
    /// returned, never more, even when far more cross-component pairs exist.
    #[test]
    fn candidates_are_budget_capped() {
        let components = vec![(0..10).collect::<Vec<_>>(), (10..20).collect::<Vec<_>>()];
        let candidates = generate_bridge_candidates(
            &components,
            |i, j| -(i as f32 + j as f32), // arbitrary but deterministic
            &BridgeCandidateOptions { max_candidates: 5 },
        );
        assert_eq!(candidates.len(), 5);
    }

    /// Three or more components: candidates span every pair of components,
    /// not just consecutive ones.
    #[test]
    fn candidates_span_every_component_pair_not_just_adjacent_ones() {
        let components = vec![vec![0usize], vec![1usize], vec![2usize]];
        let candidates = generate_bridge_candidates(
            &components,
            |_, _| 1.0,
            &BridgeCandidateOptions {
                max_candidates: usize::MAX,
            },
        );
        assert_eq!(candidates.len(), 3);
        assert!(candidates.contains(&(0, 1)));
        assert!(candidates.contains(&(0, 2)));
        assert!(candidates.contains(&(1, 2)));
    }

    /// **The end-to-end rescue claim**, on a synthetic fixture: two
    /// disconnected three-camera components (0,1,2 and 3,4,5, each internally
    /// bridged already) share a genuine wide-baseline pair (2, 3) whose raw
    /// descriptor matching is contaminated by near-duplicate "distractor"
    /// descriptors — realistic of a wide-baseline/viewpoint-change pair where
    /// detection repeatability is lower, per this milestone's own diagnosis.
    /// Under the *initial* pipeline's strict Lowe ratio (0.8, the demo's
    /// default `--match-ratio`), too few matches survive to clear the
    /// `min_matches` gate, so (2, 3) never verifies and the two components
    /// stay disconnected. Under the *rescue* pass's relaxed ratio (0.95, no
    /// cross-check), enough of the same true correspondences survive to
    /// clear the gate, and the resulting correspondence set is a real,
    /// non-degenerate two-view geometry that
    /// [`colmap_verification::TwoViewGeometryVerifier`] independently
    /// confirms (`CALIBRATED`, comfortably above `min_num_inliers`) — so the
    /// rescue pass would bridge the two components, and the M1.1 lesson
    /// holds: it is the *verifier*, not the looser matcher alone, that makes
    /// the admitted pair trustworthy.
    #[test]
    fn rescue_bridges_two_disconnected_components() {
        use super::super::colmap_verification::{
            ConfigurationType, TwoViewGeometryOptions, TwoViewGeometryVerifier,
        };
        use super::super::TwoViewCorrespondence;
        use crate::matching::{BruteForceMatcher, Matcher};
        use nalgebra::{Point3, UnitQuaternion, Vector3};
        use visloc_core::geometry::Pose;
        use visloc_core::types::Camera;

        // 1. The verified-pair graph *before* rescue: (0,1),(1,2) inside the
        // first component, (3,4),(4,5) inside the second — no (2,3) edge yet.
        let edges_before = vec![(0, 1), (1, 2), (3, 4), (4, 5)];
        let components_before = connected_components(6, &edges_before);
        assert_eq!(
            components_before.len(),
            2,
            "fixture sanity: must start genuinely disconnected"
        );

        // 2. Cross-component candidate generation finds (2, 3) as the
        // top-ranked bridge candidate (a stand-in for VLAD/vocab-tree
        // retrieval scoring it highest because images 2 and 3 are the
        // temporally-adjacent boundary pair, exactly the M5 brief's own
        // "temporally adjacent across the boundary" heuristic).
        let similarity = |i: usize, j: usize| -> f32 {
            match (i.min(j), i.max(j)) {
                (2, 3) => 0.95,
                _ => 0.1,
            }
        };
        let candidates = generate_bridge_candidates(
            &components_before,
            similarity,
            &BridgeCandidateOptions { max_candidates: 4 },
        );
        assert_eq!(candidates[0], (2, 3), "the true bridge must rank first");

        // 3. Build a genuine wide-baseline two-view geometry for (2, 3): a
        // translated + slightly-yawed camera pair viewing 20 scattered 3D
        // points (same general-scene-point shape M1's own fixtures use).
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..4 {
                points.push(Point3::new(
                    -1.5 + 0.7 * i as f64,
                    -1.0 + 0.6 * j as f64,
                    3.0 + 0.7 * ((i + j) % 5) as f64,
                ));
            }
        }
        assert_eq!(points.len(), 20);
        let pose_i = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.08);
        let pose_j = Pose::from_world_to_camera(yaw, Vector3::new(-0.5, 0.0, 0.05));

        let previous_xy: Vec<_> = points
            .iter()
            .map(|p| camera.project(&pose_i.transform_world_point(p)).unwrap())
            .collect();
        let current_xy: Vec<_> = points
            .iter()
            .map(|p| camera.project(&pose_j.transform_world_point(p)).unwrap())
            .collect();

        // 4. Synthetic descriptors: 24-dim, one canonical direction per
        // point so unrelated points are maximally separated. The true match
        // for point k sits at distance 1.0 from its query descriptor. For
        // the first 8 points, add a "distractor" descriptor (never anyone's
        // true match) at distance 1.15 from the same query — close enough
        // that distance/second_distance ~ 0.87, which the strict ratio
        // (0.8) rejects (0.87 >= 0.8) but the relaxed ratio (0.95) keeps
        // (0.87 < 0.95). This is exactly the M5 brief's lever (b):
        // "mutual-NN with Lowe ratio instead of ... strict cross-check;
        // ... a looser descriptor distance cap."
        const DIM: usize = 24;
        let basis = |k: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; DIM];
            v[k % DIM] = 10.0;
            v
        };
        let query: Vec<Vec<f32>> = (0..20).map(basis).collect();
        let mut train: Vec<Vec<f32>> = (0..20)
            .map(|k| {
                let mut v = basis(k);
                v[(k + 1) % DIM] += 1.0; // true match: distance 1.0 from query k
                v
            })
            .collect();
        // Distractors: appended after the 20 true train descriptors, each
        // near (but not identical to) query k for k in 0..8, and far from
        // every other query so it can never win anyone else's best match.
        let n_true = train.len();
        for k in 0..8 {
            let mut v = basis(k);
            v[(k + 1) % DIM] += 1.15; // 1.15 away from query k
            train.push(v);
        }
        assert_eq!(train.len(), n_true + 8);

        // 5. Strict pass: the initial pipeline's own default profile
        // (ratio 0.8, matching `--match-ratio`'s default in
        // `examples/unordered_sfm_demo.rs`).
        let strict = BruteForceMatcher { ratio: Some(0.8) };
        let strict_matches = strict.match_descriptors(&query, &train);
        let min_matches = 15usize;
        assert!(
            strict_matches.len() < min_matches,
            "fixture sanity: strict ratio must fail to reach the gate, got {}",
            strict_matches.len()
        );

        // 6. Rescue pass: the relaxed profile (ratio 0.95, i.e.
        // `--rescue-match-ratio` in the demo).
        let relaxed = BruteForceMatcher { ratio: Some(0.95) };
        let relaxed_matches = relaxed.match_descriptors(&query, &train);
        assert!(
            relaxed_matches.len() >= min_matches,
            "relaxed ratio must clear the gate, got {}",
            relaxed_matches.len()
        );
        // Every recovered match must still be geometrically correct (train
        // index == query index, i.e. the true match, not a distractor) —
        // the relaxed ratio widens *which* correct matches survive, it does
        // not let a wrong one through, since the distractor is never any
        // query's *nearest* neighbour (distance 1.15 > 1.0).
        for m in &relaxed_matches {
            assert_eq!(
                m.train_index, m.query_index,
                "relaxed ratio admitted a wrong correspondence"
            );
        }

        // 7. Full COLMAP-style verification on the rescued matches — the
        // gate that makes admitting a looser-matched pair safe (the M1.1
        // lesson): build correspondences from the recovered matches' pixel
        // coordinates and classify.
        let correspondences: Vec<TwoViewCorrespondence> = relaxed_matches
            .iter()
            .map(|m| {
                TwoViewCorrespondence::new(previous_xy[m.query_index], current_xy[m.train_index])
            })
            .collect();
        let verifier =
            TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(&camera, 4.0));
        let report = verifier.classify(&correspondences, &camera);
        assert!(
            matches!(
                report.config,
                ConfigurationType::Calibrated | ConfigurationType::Uncalibrated
            ),
            "rescued pair must verify as a real, non-degenerate two-view geometry, got {:?}",
            report.config
        );
        assert!(
            report.inliers.len() >= min_matches,
            "verified inlier count too low to trust as a bridge: {}",
            report.inliers.len()
        );

        // 8. With (2, 3) admitted, the view graph is now a single component.
        let mut edges_after = edges_before.clone();
        edges_after.push((2, 3));
        let components_after = connected_components(6, &edges_after);
        assert_eq!(
            components_after.len(),
            1,
            "admitting the rescued bridge must reconnect the whole graph"
        );
    }
}
