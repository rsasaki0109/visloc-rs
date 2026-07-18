//! DPV-SLAM's mid-term "proximity" loop-closure backend — Milestone M6 of
//! `docs/dpvo_droid_port_plan.md`.
//!
//! # Ground truth
//!
//! Lipson, Teed, Deng, *"Deep Patch Visual SLAM"*, ECCV 2024
//! ([arXiv:2408.01654](https://arxiv.org/abs/2408.01654)) describes the
//! method; the *code* — the only thing actually re-derived line-by-line
//! here — lives in
//! [princeton-vl/DPVO](https://github.com/princeton-vl/DPVO) (MIT license,
//! see `docs/dpvo_droid_port_plan.md`'s §2 license verdict), cloned locally to
//! `E:/tools/DPVO`:
//!
//! * [`dpvo/patchgraph.py::PatchGraph.edges_loop`](https://github.com/princeton-vl/DPVO/blob/main/dpvo/patchgraph.py)
//!   (lines 55-77 at the commit cloned here) — candidate generation, the
//!   pose-proximity/flow-magnitude gate, and the call into `reduce_edges`.
//! * [`dpvo/loop_closure/optim_utils.py::reduce_edges`](https://github.com/princeton-vl/DPVO/blob/main/dpvo/loop_closure/optim_utils.py)
//!   (lines 19-58) — the greedy, flow-magnitude-ranked, temporal-gap-gated,
//!   NMS-suppressed edge selector.
//! * [`dpvo/dpvo.py::DPVO.__call__`](https://github.com/princeton-vl/DPVO/blob/main/dpvo/dpvo.py)
//!   (lines 449-455) — the `GLOBAL_OPT_FREQ`-throttled call site that decides
//!   *when* to run `edges_loop` and feeds its output into `append_factors`
//!   (the same function that adds every ordinary temporal edge).
//! * [`dpvo/dpvo.py::DPVO.keyframe`](https://github.com/princeton-vl/DPVO/blob/main/dpvo/dpvo.py)
//!   (lines 305-309) — the `LOOP_CLOSURE`-only exemption keeping a loop edge
//!   alive past the ordinary removal-window drop; ported as
//!   `crate::dpvo_patch_graph::DpvoPatchGraph::keyframe_with_loop_protection`.
//! * [`dpvo/config.py`](https://github.com/princeton-vl/DPVO/blob/main/dpvo/config.py)
//!   (lines 28-31) — `LOOP_CLOSURE`, `BACKEND_THRESH=64.0`, `MAX_EDGE_AGE=1000`,
//!   `GLOBAL_OPT_FREQ=15` — the four numeric defaults this module ports as
//!   [`DpvoLoopClosureConfig`]'s own defaults (see that struct's doc for the
//!   one deliberate CPU-bounded exception, `max_edges_per_batch`).
//!
//! # Scope: the mid-term ("proximity") backend only
//!
//! DPV-SLAM ships **two** loop-closure backends (`docs/dpvo_droid_port_plan.md`
//! §1.3): this milestone ports only the first —
//!
//! 1. **Mid-term / proximity** (`LOOP_CLOSURE`): detected by camera-pose
//!    proximity + flow-magnitude, not appearance; reactivates old patches
//!    against the current frame using the **same** patch-graph edge/BA
//!    machinery as ordinary temporal edges. This is the paper's own stated
//!    main efficiency contribution over DROID-SLAM's frame-graph-only
//!    consistency mechanism, and the whole of this module's scope.
//! 2. **Long-term / classical** (`CLASSIC_LOOP_CLOSURE`): dBoW2 image
//!    retrieval + PnP + RANSAC/Umeyama + a classical CPU pose-graph
//!    optimizer, run in parallel to the GPU odometry process. **Explicitly
//!    out of scope** — `docs/dpvo_droid_port_plan.md`'s own port-surface
//!    table (§4) already identifies `visloc-rs`'s existing
//!    `online_slam.rs` appearance-loop-candidate pipeline +
//!    `map_atlas.rs`'s cross-submap PnP/RANSAC/scale-consensus/Sim3 welding
//!    as **already more rigorous** than DPV-SLAM's own classical backend
//!    (metric-scale gates, covisibility-disagreement bounds, MAD-based scale
//!    consensus vs. DPV-SLAM's plain 3-point RANSAC); nothing in this
//!    milestone touches those modules, matching the task's own instruction to
//!    note this rather than re-port a backend this codebase already exceeds.
//!
//! # Why this lives in the DPVO-native edge system, not
//! `crate::sparse_factor_graph`
//!
//! `docs/dpvo_droid_port_plan.md`'s original §4 port-surface table (written
//! before M1-M3 ran) speculated that DPV-SLAM's proximity backend would map
//! onto `sparse_factor_graph.rs`'s existing `SparseFactorKind::Proximity` +
//! `enforce_active_budget` machinery. M3's own results section overturned the
//! matching half of that same table for the BA layer (`fastba.BA` vs.
//! `bundle.rs`'s `BundleAdjustment` turned out "structurally incompatible for
//! a clean merge" — different perturbation convention, different damping
//! scheme, different variable parameterization) and built a dedicated
//! sibling module (`dpvo_patch_ba.rs`) instead. The same reasoning applies
//! here, for a more basic reason than convention mismatch: DPV-SLAM's own
//! proximity backend is not a *bridge* into some other system's edge
//! representation — it is new entries in `patchgraph.py`'s own `ii/jj/kk`
//! arrays, consumed by the **same** `append_factors`/`update()`/`fastba.BA`
//! call chain every ordinary temporal edge already goes through. This crate's
//! M4 port already reproduced that exact chain as
//! `DpvoPatchGraph::append_edges` + `DpvoOdometry::update_step` +
//! `dpvo_patch_ba::dpvo_ba` — a native, already-working analog of upstream's
//! own mechanism. Routing loop edges through `sparse_factor_graph.rs`/
//! `bundle.rs` instead would mean maintaining a *second*, redundant edge
//! representation and BA entry point for exactly the one edge kind upstream
//! itself keeps unified with ordinary temporal edges — the opposite of "the
//! same patch-BA machinery handles loop edges," the task's own stated key
//! trick. `sparse_factor_graph.rs`/`bundle.rs` remain untouched by this
//! milestone (read-only per the task's own scope note); this crate's
//! existing appearance-loop pipeline in `online_slam.rs`/`map_atlas.rs`
//! keeps doing long-term loop closure for that (separate) pipeline, as it
//! already did before this milestone.
//!
//! # What "global BA" becomes on this CPU port
//!
//! Upstream's `__run_global_BA` (`dpvo.py:312-325`) re-runs `fastba.BA` over
//! `t0 = self.pg.ii.min()` (every frame any *retained* edge — active or
//! inactive — still references) whenever a stale edge exists
//! (`(self.pg.ii < self.n - REMOVAL_WINDOW - 1).any()`), which is exactly the
//! condition a live proximity edge creates. This module does not port a
//! *separate* "global BA" call: `crate::dpvo_vo::DpvoOdometry::update_step`'s
//! own windowing derivation (`docs/dpvo_droid_port_plan.md`'s M4 results) is
//! generalized (Milestone M6) to compute `frame_lo` as the minimum over both
//! its original bound *and* every currently-active edge's endpoints — so
//! whenever a loop edge references a frame older than the ordinary
//! `removal_window + patch_lifetime` bound, the very same windowed
//! `dpvo_ba`/`dpvo_vi_ba` call this module already makes every frame widens
//! to cover it automatically, with **no new BA entry point**. This is a
//! faithful behavioral match to upstream's own trigger condition (a stale
//! edge existing), just implemented as "widen the existing window" rather
//! than "call a second, differently-scoped BA function" — the two are
//! mathematically identical because `dpvo_ba`'s own `fixedp` bookkeeping
//! already keeps the *free* pose count pinned at `optimization_window`
//! regardless of how far back `frame_lo` reaches (older poses the window
//! grows to cover are added as *fixed* anchors, not new free variables — see
//! `dpvo_patch_ba.rs`'s M3 "t0/fixedp" convention-mapping note). Bounded-CPU
//! deviation, honestly stated: upstream's own `t0 = ii.min()` can reach all
//! the way back to `MAX_EDGE_AGE`-old patch memory; this port does not cap
//! `frame_lo`'s growth beyond what naturally results from
//! [`DpvoLoopClosureConfig::max_edge_age`]/`max_edges_per_batch` bounding how
//! old and how numerous new loop edges can be — see that struct's own doc for
//! the exact CPU-feasibility reasoning (correlation-assembly cost, not BA
//! cost, is what M4-perf's own results identified as this port's real
//! bottleneck, so the edge *count* budget, not the window depth, is the
//! knob actually worth tightening below upstream's own numbers).
//!
//! Poses outside the optimized window (i.e. folded away entirely by
//! `DpvoPatchGraph::keyframe`'s low-motion folding, not merely "old but
//! still live") are unaffected by this module and continue to use
//! `DpvoPatchGraph::reconstruct_pose`'s existing delta-chain propagation —
//! nothing here changes that mechanism, matching upstream's own `get_pose`
//! (`dpvo.py:166-171`), which this crate already ported in M4.

use std::collections::HashSet;

use crate::dpvo_patch_ba::{flow_mag, reprojected_center_depth};
use crate::dpvo_patch_graph::DpvoPatchGraph;

/// `optim_utils.py::reduce_edges`'s own hardcoded `(j - i) < 30` literal
/// (never a `config.py` field upstream) — exposed here as a
/// [`DpvoLoopClosureConfig`] field purely for testability (a real trajectory
/// needs dozens of frames before any candidate could pass this gate; a unit
/// test wants a much smaller number). The shipped default (`30`) matches
/// upstream exactly.
pub const UPSTREAM_MIN_LOOP_GAP: usize = 30;

/// Configuration for the mid-term proximity loop-closure backend. Every
/// field's *name* and *default* traces to `dpvo/config.py`'s own
/// `LOOP_CLOSURE`-adjacent block (lines 28-31) except where noted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoLoopClosureConfig {
    /// `config.py::BACKEND_THRESH` (default `64.0`) — a candidate `(i, j)`
    /// frame pair's mean [`flow_mag`] (over patches owned by `i`, valid ones
    /// only) must fall strictly below this to be considered at all.
    pub backend_thresh: f64,
    /// `config.py::MAX_EDGE_AGE` (default `1000`) — how far back (in frames,
    /// from the removal-window boundary `l = n - removal_window`) old
    /// patches remain eligible as loop-edge sources. Kept at upstream's own
    /// number: the candidate *search* this bounds is cheap pure arithmetic
    /// (`flow_mag` evaluations, no ONNX/correlation), not the CPU cost driver
    /// M4-perf identified (see the module doc's "What 'global BA' becomes"
    /// section) — [`Self::max_edges_per_batch`] is the field actually
    /// tightened for CPU feasibility, not this one.
    ///
    /// # Milestone M10 finding: structurally unreachable on this port, confirmed not merely suspected
    ///
    /// [`find_loop_edges`]'s own `i` search floor is `ii_lo = max(l -
    /// max_edge_age, 0)`, where `l = n - removal_window` and `n =
    /// DpvoPatchGraph::n_frames()` — the CURRENT LIVE frame count, not an
    /// arrival-index position. On every real MH_01 run across M6-M10, `n`
    /// stays bounded to roughly the live patch buffer's own size (`~40-55`
    /// frames, `docs/dpvo_droid_port_plan.md`'s M10 results), so `l ≤ ~40`
    /// always — meaning `ii_lo = max(l - 1000, 0) = 0` on every real call
    /// this port has ever made: the `1000` never binds, because `l` itself
    /// never gets anywhere close to it. The practical search floor is
    /// simply "the oldest frame still in the live buffer," never a
    /// genuinely `1000`-frames-old one. This is the root cause the M10
    /// results section traces in full: [`find_loop_edges`] can only ever
    /// propose a loop whose old endpoint is still LIVE, capping every
    /// accepted pair's temporal gap at roughly the live buffer's own size —
    /// a mid-term consistency check, not a long-range revisit detector,
    /// regardless of how large this field is set. Confirmed on real data,
    /// not merely reasoned about — see the M10 results section for the
    /// measured `~30-49`-frame gaps this produced.
    pub max_edge_age: usize,
    /// `config.py::GLOBAL_OPT_FREQ` (default `15`) — attempt a new
    /// loop-candidate search only every this many committed frames since the
    /// last attempt (`dpvo.py:449-455`'s own throttle), regardless of whether
    /// that attempt found anything.
    pub global_opt_freq: usize,
    /// `optim_utils.py::reduce_edges`'s hardcoded `(j - i) < 30` temporal-gap
    /// gate, exposed for testability — see [`UPSTREAM_MIN_LOOP_GAP`]'s own
    /// doc. Shipped default matches upstream exactly.
    pub min_loop_gap: usize,
    /// `optim_utils.py::reduce_edges`'s own `max_num_edges` parameter, called
    /// with `1000` at its one real call site (`patchgraph.py:76`) — **a
    /// deliberate, documented CPU-bounded deviation, not a faithful port of
    /// the number itself**. Each accepted `(i, j)` *frame* pair expands to
    /// `patches_per_frame` new patch-graph edges (`edges_loop`'s own
    /// `repeat(..., M=M)` expansion), each of which costs exactly as much
    /// per-frame correlation-assembly work as an ordinary temporal edge for
    /// as long as it stays active (`docs/dpvo_droid_port_plan.md`'s M4-perf
    /// results: correlation assembly, not BA, is this CPU port's dominant
    /// per-frame cost). Accepting upstream's own `1000`-frame-pair cap at
    /// `patches_per_frame≈48-96` could add tens of thousands of edges to a
    /// single `update_step` call — squarely the M4-perf-diagnosed
    /// bottleneck, at a scale that milestone never budgeted for. Default `8`
    /// here: a handful of accepted revisits per `global_opt_freq`-frame
    /// attempt, bounded to stay well inside the correlation-assembly budget
    /// M4-perf already measured as feasible for ordinary edges at this
    /// port's `fast.yaml` graph sizing.
    pub max_edges_per_batch: usize,
    /// `optim_utils.py::reduce_edges`'s own `nms` parameter (called with `1`
    /// at its one real call site) — once an `(i, j)` pair is accepted, every
    /// `(i', j)` with `|i' - i| <= nms_radius` is suppressed from later
    /// acceptance in the same batch (avoids near-duplicate revisits of
    /// almost the same source frame against the same target). Default `1`,
    /// matching upstream exactly.
    pub nms_radius: usize,
    /// `edges_loop`'s own inline `num_val > (self.M * 0.75)` validity-fraction
    /// gate (`patchgraph.py:74`) — a candidate `(i, j)` pair needs at least
    /// this fraction of frame `i`'s patches to reproject in front of camera
    /// `j` (`reprojected_center_depth(...) > 0.2`) before its mean
    /// [`flow_mag`] is trusted at all. Default `0.75`, matching upstream
    /// exactly.
    pub min_valid_fraction: f64,
}

impl Default for DpvoLoopClosureConfig {
    fn default() -> Self {
        Self {
            backend_thresh: 64.0,
            max_edge_age: 1000,
            global_opt_freq: 15,
            min_loop_gap: UPSTREAM_MIN_LOOP_GAP,
            max_edges_per_batch: 8,
            nms_radius: 1,
            min_valid_fraction: 0.75,
        }
    }
}

/// One `(i, j)` frame-pair candidate surviving `edges_loop`'s own
/// `BACKEND_THRESH`/validity-fraction gate, before [`select_loop_edges`]'s
/// greedy NMS selection — `patchgraph.py:74,76`'s `mask`-filtered
/// `(flow_mag, ii, jj)` triples, one entry per accepted `(i, j)` pair (`ii`
/// there is already downsampled to one value per pair, `ii[::self.M]`; this
/// struct is that same one-per-pair granularity, not one-per-patch).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopEdgeCandidate {
    /// Source (old) frame — patches here get the new edge.
    pub i: usize,
    /// Target (recent) frame — the edge's correlation destination.
    pub j: usize,
    /// Mean [`flow_mag`] over frame `i`'s valid patches against frame `j`.
    pub flow_mag: f64,
}

/// Port of `optim_utils.py::reduce_edges` (lines 19-58): greedily accept
/// candidates in ascending `flow_mag` order (best/smallest first), rejecting
/// any whose temporal gap is too small or whose `i` (within `nms_radius`) has
/// already been accepted for the same `j`, until `max_edges` accepted pairs
/// have been kept. Deterministic given a stable sort (`argsort` is stable in
/// the reference's own `numpy`/`numba` implementation; Rust's
/// `sort_by(partial_cmp)` on a `Vec` is a stable sort too, so ties break in
/// input order on both sides).
///
/// Upstream's own `if len(es) > max_num_edges: break` (with `es` seeded by a
/// one-element sentinel, `optim_utils.py:22`) is equivalent to "stop once
/// `max_edges` real candidates have been accepted" — ported directly as that
/// simpler, sentinel-free condition (`accepted.len() >= max_edges`), not the
/// off-by-one-looking literal translation, since the sentinel is discarded
/// by `_format` before upstream's own caller ever sees it
/// (`optim_utils.py:19-21`).
///
/// Upstream's `mag >= 1000` defensive check (`optim_utils.py:46`, guarding
/// against the `inf` sentinel `edges_loop` uses for its own invalid entries,
/// `patchgraph.py:74`) is vestigial at this function's own call site — by
/// construction, [`find_loop_edges`] never constructs a [`LoopEdgeCandidate`]
/// for a pair that failed the `backend_thresh`/validity gate in the first
/// place, so no candidate reaching this function ever carries an `inf`-like
/// sentinel value. Kept anyway (a finite-value assertion via `is_finite`)
/// as cheap, harmless defense-in-depth matching upstream's own belt-and-braces
/// structure, not because a real call site is expected to need it.
pub fn select_loop_edges(
    candidates: &[LoopEdgeCandidate],
    min_loop_gap: usize,
    max_edges: usize,
    nms_radius: usize,
) -> Vec<(usize, usize)> {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| {
        candidates[a]
            .flow_mag
            .partial_cmp(&candidates[b].flow_mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut accepted: Vec<(usize, usize)> = Vec::new();
    let mut suppressed: HashSet<(usize, usize)> = HashSet::new();
    for idx in order {
        if accepted.len() >= max_edges {
            break;
        }
        let c = candidates[idx];
        if !c.flow_mag.is_finite() {
            continue; // Vestigial upstream defense-in-depth — see this fn's doc.
        }
        if c.j <= c.i || c.j - c.i < min_loop_gap {
            continue;
        }
        if suppressed.contains(&(c.i, c.j)) {
            continue;
        }
        accepted.push((c.i, c.j));
        let nms = nms_radius as i64;
        for di in -nms..=nms {
            let i1 = c.i as i64 + di;
            if i1 >= 0 {
                suppressed.insert((i1 as usize, c.j));
            }
        }
    }
    accepted
}

/// Port of `patchgraph.py::PatchGraph.edges_loop` (lines 55-77): scan every
/// candidate `(i, j)` frame pair (`i` an "old" frame within
/// `[max(l - max_edge_age, 0), l)` where `l = n - removal_window`; `j` a
/// "recent-but-not-newest" frame within `[n - global_opt_freq, n -
/// keyframe_index)`), score each surviving [`DpvoLoopClosureConfig::min_valid_fraction`]/
/// `backend_thresh` gate by its mean [`flow_mag`], then run
/// [`select_loop_edges`] over the survivors.
///
/// Returns `(candidates_evaluated, accepted_frame_pairs)` — the raw
/// gate-surviving candidate count (for diagnostics, `docs/dpvo_droid_port_plan.md`'s
/// M6 acceptance criteria's own "candidates" number) and the frame pairs
/// [`select_loop_edges`] actually kept. Expand the latter to patch-level
/// `(patch_id, target_frame)` pairs with [`expand_frame_pairs_to_patch_edges`]
/// before calling `DpvoPatchGraph::append_edges` — mirroring
/// `edges_loop`'s own final `repeat(edges, 'E ij -> ij E M', M=M)` expansion
/// (`patchgraph.py:75-76`), which this function deliberately stops short of
/// so a caller can log/diagnose at the cheaper frame-pair granularity first.
pub fn find_loop_edges(
    graph: &DpvoPatchGraph,
    config: &DpvoLoopClosureConfig,
) -> (usize, Vec<(usize, usize)>) {
    let n = graph.n_frames();
    let removal_window = graph.config().removal_window;
    let keyframe_index = graph.config().keyframe_index;
    let patches_per_frame = graph.config().patches_per_frame;

    // l = n - REMOVAL_WINDOW (patchgraph.py:58); l<=0 => no old frames exist
    // yet to reactivate (patchgraph.py:59-60).
    let l = n.saturating_sub(removal_window);
    if l == 0 {
        return (0, Vec::new());
    }

    // jj in [n - GLOBAL_OPT_FREQ, n - KEYFRAME_INDEX) (patchgraph.py:63-64).
    let jj_lo = n.saturating_sub(config.global_opt_freq);
    let jj_hi = n.saturating_sub(keyframe_index);
    if jj_hi <= jj_lo {
        return (0, Vec::new());
    }

    // kk (as owner frames, before the M-patch expansion) in
    // [max(l - MAX_EDGE_AGE, 0), l) (patchgraph.py:65-66).
    let ii_lo = l.saturating_sub(config.max_edge_age);
    let ii_hi = l;
    if ii_hi <= ii_lo {
        return (0, Vec::new());
    }

    let min_valid = config.min_valid_fraction * patches_per_frame as f64;
    let mut candidates = Vec::new();
    for i in ii_lo..ii_hi {
        let frame_i = &graph.frames()[i];
        for j in jj_lo..jj_hi {
            if j == i {
                continue;
            }
            let frame_j = &graph.frames()[j];
            let mut sum = 0.0_f64;
            let mut valid_count = 0usize;
            for local in 0..patches_per_frame {
                let patch = graph.patches()[i * patches_per_frame + local];
                // patchgraph.py:71's own validity mask (via projective_ops.py::flow_mag's
                // second `val` return, itself `transform(..., valid=True)`'s
                // `X1[...,2] > 0.2`, projective_ops.py:113): reprojected
                // center depth > 0.2 in frame j.
                let z = reprojected_center_depth(&frame_i.pose, &frame_j.pose, &frame_i.intrinsics, &patch);
                if z > 0.2 {
                    sum +=
                        flow_mag(&frame_i.pose, &frame_j.pose, &frame_i.intrinsics, &frame_j.intrinsics, &patch, 0.5);
                    valid_count += 1;
                }
            }
            if (valid_count as f64) <= min_valid {
                continue; // patchgraph.py:74's `num_val > (self.M * 0.75)` gate.
            }
            let mean_flow = sum / valid_count as f64;
            if mean_flow < config.backend_thresh {
                candidates.push(LoopEdgeCandidate { i, j, flow_mag: mean_flow });
            }
        }
    }

    let candidates_evaluated = candidates.len();
    let accepted =
        select_loop_edges(&candidates, config.min_loop_gap, config.max_edges_per_batch, config.nms_radius);
    (candidates_evaluated, accepted)
}

/// `edges_loop`'s own final expansion (`patchgraph.py:75-76`:
/// `repeat(edges, 'E ij -> ij E M', M=M); kk = ii.mul(M) + arange(M)`) — every
/// patch owned by an accepted source frame `i` gets a new edge targeting the
/// accepted `j`, matching `DpvoPatchGraph::append_edges`'s own
/// `(patch_id, target_frame)` pair convention (already used by
/// `edges_forw`/`edges_back`).
pub fn expand_frame_pairs_to_patch_edges(
    accepted_frame_pairs: &[(usize, usize)],
    patches_per_frame: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(accepted_frame_pairs.len() * patches_per_frame);
    for &(i, j) in accepted_frame_pairs {
        for local in 0..patches_per_frame {
            out.push((i * patches_per_frame + local, j));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector2, Vector3};
    use visloc_core::geometry::SE3;

    use crate::dpvo_patch_ba::{DpvoIntrinsics, DpvoPatch};
    use crate::dpvo_patch_graph::DpvoVoConfig;

    fn intr() -> DpvoIntrinsics {
        DpvoIntrinsics { fx: 120.0, fy: 120.0, cx: 32.0, cy: 24.0 }
    }

    fn patches_for_frame(m: usize) -> Vec<DpvoPatch> {
        // Spread anchors around the principal point so `flow_mag`'s
        // translation-only term is well-conditioned (not all patches sitting
        // exactly on the optical axis).
        (0..m)
            .map(|i| {
                let dx = (i as f64 - m as f64 / 2.0) * 2.0;
                DpvoPatch { x: 32.0 + dx, y: 24.0 + dx * 0.5, inverse_depth: 0.5 }
            })
            .collect()
    }

    /// A small square-loop test trajectory: the camera walks a square path
    /// and returns near its own starting pose at the end — the canonical
    /// "the robot revisits a place it has already seen" scenario
    /// `edges_loop`'s own pose-proximity gate is built to detect.
    fn square_loop_config() -> DpvoVoConfig {
        DpvoVoConfig {
            buffer_size: 4096,
            patches_per_frame: 6,
            // Small enough that `l = n - removal_window` (edges_loop's own
            // upper bound on eligible source frames) stays comfortably
            // positive for this test's short (~50-60 frame) trajectory —
            // this test cares about proximity detection over old-but-still-
            // graph-resident frames, not the removal/folding policy itself,
            // and never calls `keyframe()` at all (so no frame is ever
            // physically folded away regardless of this value).
            removal_window: 10,
            optimization_window: 8,
            patch_lifetime: 6,
            keyframe_index: 2,
            // A high threshold keeps every frame of this synthetic
            // trajectory live (no keyframe folding) so the test's own frame
            // indices are predictable.
            keyframe_thresh: 1.0e9,
            motion_damping: 0.5,
        }
    }

    /// Build a square trajectory of `steps_per_side`-frame segments per side,
    /// walking `side_length` per step along each edge of the square, in the
    /// XZ ground plane (`Y` fixed) so patches placed a few meters ahead
    /// (`+Z` in the camera's own frame) stay in front of the camera the
    /// whole way around, then `tail_frames` more steps continuing in the same
    /// final direction. Frame 0 sits at the square's starting corner; the
    /// frame at the end of the square segment (index `4*steps_per_side - 1`)
    /// sits one step short of frame 0's own pose (having walked all the way
    /// around) — the `tail_frames` extension exists purely so THAT near-
    /// revisit frame is not itself among `edges_loop`'s own excluded
    /// "newest `keyframe_index` frames", i.e. so the revisit falls inside the
    /// candidate `jj` window `[n - global_opt_freq, n - keyframe_index)`
    /// rather than past its upper end.
    fn build_square_loop(steps_per_side: usize, side_length: f64, tail_frames: usize) -> DpvoPatchGraph {
        let config = square_loop_config();
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        let step = side_length / steps_per_side as f64;

        // Four straight segments (+X, +Z, -X, -Z), identity orientation
        // throughout (a translating, non-rotating square) — enough to create
        // real pose-proximity revisits without needing to model rotation.
        let directions = [Vector3::new(step, 0.0, 0.0), Vector3::new(0.0, 0.0, step), Vector3::new(-step, 0.0, 0.0), Vector3::new(0.0, 0.0, -step)];
        let mut position = Vector3::new(0.0, 0.0, 0.0);
        let mut t = 0.0_f64;
        for dir in directions {
            for _ in 0..steps_per_side {
                let pose = SE3::new(UnitQuaternion::identity(), position);
                graph.begin_frame(t);
                graph.commit_frame(pose, intr(), patches_for_frame(m)).unwrap();
                let forw = graph.edges_forw();
                let back = graph.edges_back();
                graph.append_edges(&forw, 4);
                graph.append_edges(&back, 4);
                position += dir;
                t += 0.05;
            }
        }
        // Tail: keep walking away from the revisit point along the square's
        // final direction, so the near-revisit frame ages into the eligible
        // `jj` window instead of staying among the newest-excluded frames.
        let tail_dir = *directions.last().unwrap();
        for _ in 0..tail_frames {
            let pose = SE3::new(UnitQuaternion::identity(), position);
            graph.begin_frame(t);
            graph.commit_frame(pose, intr(), patches_for_frame(m)).unwrap();
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
            position += tail_dir;
            t += 0.05;
        }
        graph
    }

    #[test]
    fn find_loop_edges_detects_the_revisit_on_a_square_loop_and_not_adjacent_frames() {
        // 12 steps/side * 4 sides = 48 frames (indices 0..47), the walk
        // returning to (0,0,0) one final step short at frame 47, plus 10
        // tail frames (indices 48..57, n=58 total) walking further away so
        // frame 47 (a few centimeters from frame 0's own pose — well inside
        // BACKEND_THRESH's reach for patches a few meters out) sits inside
        // `edges_loop`'s own eligible `jj` window `[n-15, n-2) = [43, 56)`
        // rather than among the newest-excluded frames.
        let graph = build_square_loop(12, 2.0, 10);
        let n = graph.n_frames();
        assert_eq!(n, 58);

        let mut config = DpvoLoopClosureConfig { min_loop_gap: 20, ..Default::default() };
        // global_opt_freq/keyframe_index bound the *candidate* jj range; use
        // upstream's own defaults (15/2) so jj sits in [n-15, n-2) =
        // [43, 56) — covering the near-revisit frame (47), thanks to the
        // 10-frame tail `build_square_loop` walked past it.
        config.max_edge_age = n; // search the whole trajectory's history.
        let (candidates_evaluated, accepted) = find_loop_edges(&graph, &config);

        assert!(candidates_evaluated > 0, "expected at least one candidate near the square's revisited corner");
        assert!(!accepted.is_empty(), "expected the near-identical start/end poses to be accepted as a loop");
        for &(i, j) in &accepted {
            assert!(j > i, "accepted pair should have a forward-in-time target");
            assert!(j - i >= config.min_loop_gap, "accepted pair violated the temporal-gap gate: i={i} j={j}");
            // The revisit is near the trajectory's START (low i, since the
            // square returns to its own origin) — not two merely-adjacent
            // frames a few steps apart within the same straight segment.
            assert!(i < 5, "expected the loop's source frame to be near the trajectory start, got i={i}");
        }
    }

    #[test]
    fn find_loop_edges_finds_nothing_on_a_straight_line_that_never_revisits() {
        // A straight-line walk never returns near any earlier pose, so no
        // candidate should ever clear BACKEND_THRESH.
        let config = square_loop_config();
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        let mut position = Vector3::new(0.0, 0.0, 0.0);
        for i in 0..48 {
            let pose = SE3::new(UnitQuaternion::identity(), position);
            graph.begin_frame(i as f64 * 0.05);
            graph.commit_frame(pose, intr(), patches_for_frame(m)).unwrap();
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
            position += Vector3::new(0.5, 0.0, 0.0);
        }
        let lc_config = DpvoLoopClosureConfig { min_loop_gap: 20, max_edge_age: 48, ..Default::default() };
        let (_, accepted) = find_loop_edges(&graph, &lc_config);
        assert!(accepted.is_empty(), "a straight-line trajectory should never produce a loop candidate");
    }

    #[test]
    fn select_loop_edges_enforces_the_temporal_gap_gate() {
        let candidates = vec![
            LoopEdgeCandidate { i: 10, j: 15, flow_mag: 1.0 }, // gap=5, too small
            LoopEdgeCandidate { i: 10, j: 60, flow_mag: 2.0 }, // gap=50, ok
        ];
        let accepted = select_loop_edges(&candidates, 30, 10, 1);
        assert_eq!(accepted, vec![(10, 60)]);
    }

    #[test]
    fn select_loop_edges_respects_the_edge_budget() {
        let candidates: Vec<LoopEdgeCandidate> =
            (0..20).map(|k| LoopEdgeCandidate { i: k * 3, j: k * 3 + 100, flow_mag: k as f64 }).collect();
        let accepted = select_loop_edges(&candidates, 30, 4, 0);
        assert_eq!(accepted.len(), 4, "edge budget must cap the number of accepted candidates");
        // Greedy ascending-flow_mag order => the 4 best (lowest flow_mag)
        // candidates are the first 4 by construction.
        assert_eq!(accepted, vec![(0, 100), (3, 103), (6, 106), (9, 109)]);
    }

    #[test]
    fn select_loop_edges_nms_suppresses_near_duplicate_sources_for_the_same_target() {
        let candidates = vec![
            LoopEdgeCandidate { i: 50, j: 100, flow_mag: 1.0 }, // best
            LoopEdgeCandidate { i: 49, j: 100, flow_mag: 2.0 }, // within nms_radius=1 of (50,100)
            LoopEdgeCandidate { i: 51, j: 100, flow_mag: 3.0 }, // within nms_radius=1 of (50,100)
            LoopEdgeCandidate { i: 40, j: 100, flow_mag: 4.0 }, // far enough away => kept
        ];
        let accepted = select_loop_edges(&candidates, 30, 10, 1);
        assert_eq!(accepted, vec![(50, 100), (40, 100)]);
    }

    #[test]
    fn select_loop_edges_rejects_non_finite_flow_mag() {
        let candidates = vec![
            LoopEdgeCandidate { i: 0, j: 100, flow_mag: f64::INFINITY },
            LoopEdgeCandidate { i: 1, j: 101, flow_mag: 5.0 },
        ];
        let accepted = select_loop_edges(&candidates, 30, 10, 1);
        assert_eq!(accepted, vec![(1, 101)]);
    }

    #[test]
    fn expand_frame_pairs_to_patch_edges_emits_one_edge_per_patch_per_pair() {
        let accepted = vec![(2, 40), (5, 41)];
        let edges = expand_frame_pairs_to_patch_edges(&accepted, 4);
        assert_eq!(edges.len(), 8);
        assert!(edges.contains(&(2 * 4, 40)));
        assert!(edges.contains(&(2 * 4 + 3, 40)));
        assert!(edges.contains(&(5 * 4, 41)));
        assert!(edges.contains(&(5 * 4 + 3, 41)));
    }

    /// Milestone M6's required synthetic accuracy test: a small window whose
    /// last frame has DRIFTED away from its true (loop-closing) pose;
    /// closing the loop — adding a patch-BA edge between the drifted frame
    /// and the true revisited landmark set, then re-solving — must reduce
    /// the endpoint error versus not closing it at all. Exercises
    /// `crate::dpvo_patch_ba::dpvo_ba` directly (the same BA entry point
    /// `crate::dpvo_vo::DpvoOdometry::update_step` calls), not a live
    /// `DpvoOdometry`/ONNX session — the loop-*correction* claim is a BA-level
    /// fact, independent of how the edge's `target`/`weight` were produced.
    #[test]
    fn closing_a_synthetic_drifted_loop_reduces_endpoint_error() {
        use crate::dpvo_patch_ba::{dpvo_ba, transform_point, DpvoBaConfig, DpvoBaProblem, DpvoEdge};

        // Truth: 4 frames, translating along +X, one patch anchored in frame
        // 0 observed by every frame (an ordinary temporal chain) PLUS a
        // second patch that would, at the true poses, also be visible from
        // frame 0 in frame 3 (the "loop" observation) — the frame-3 pose is
        // then perturbed (drifted) away from truth, and the loop edge's
        // target is computed at the TRUE (undrifted) frame-3 pose, exactly
        // as a correct GRU/correlation prediction would (it predicts where
        // the patch truly is in image space, regardless of the current pose
        // estimate's own error).
        let intrinsics = DpvoIntrinsics { fx: 200.0, fy: 200.0, cx: 64.0, cy: 48.0 };
        let true_poses: Vec<SE3> = (0..4)
            .map(|i| SE3::new(UnitQuaternion::identity(), Vector3::new(i as f64 * 0.2, 0.0, 0.0)))
            .collect();
        let patch = DpvoPatch { x: 64.0, y: 48.0, inverse_depth: 0.2 }; // depth=5, anchored in frame 0.

        // Perturb frame 3 (the "current" frame) away from truth: drift.
        let drift = Vector3::new(0.15, 0.0, 0.0); // 0.15m translation drift.
        let mut drifted_poses = true_poses.clone();
        drifted_poses[3] =
            SE3::new(true_poses[3].rotation, true_poses[3].translation + drift);

        // Ordinary temporal edges: frame 0's patch observed in frames 1, 2
        // (chain edges only — frame 3 is NOT temporally chained here, only
        // reachable via the loop edge, isolating the loop edge's own effect).
        let temporal_edges = vec![
            DpvoEdge { i: 0, j: 0, k: 0 },
            DpvoEdge { i: 0, j: 1, k: 0 },
            DpvoEdge { i: 0, j: 2, k: 0 },
        ];
        let temporal_targets: Vec<_> = temporal_edges
            .iter()
            .map(|e| transform_point(&true_poses[e.i], &true_poses[e.j], &intrinsics, &intrinsics, &patch, false))
            .collect();

        let config = DpvoBaConfig { iterations: 2, fixedp: 1, lmbda: 1e-4, ep: 100.0, bounds: [-1e6, -1e6, 1e6, 1e6] };

        // (A) No loop edge: frame 3 has NO edge touching it at all, so BA
        // cannot correct its drift — it stays exactly as drifted.
        let problem_no_loop = DpvoBaProblem {
            poses: drifted_poses.clone(),
            patches: vec![patch],
            intrinsics: vec![intrinsics; 4],
            edges: temporal_edges.clone(),
            targets: temporal_targets.clone(),
            weights: vec![Vector2::new(1.0, 1.0); temporal_edges.len()],
            depth_damping: None,
        };
        let solved_no_loop = dpvo_ba(&problem_no_loop, &config).expect("ba solve");
        let error_no_loop = (solved_no_loop.poses[3].translation - true_poses[3].translation).norm();

        // (B) Add the loop edge (frame 0's patch, observed in frame 3, target
        // computed at the TRUE frame-3 pose — the "correctly predicted
        // revisit" a real GRU update would supply).
        let mut edges_with_loop = temporal_edges.clone();
        edges_with_loop.push(DpvoEdge { i: 0, j: 3, k: 0 });
        let mut targets_with_loop = temporal_targets.clone();
        targets_with_loop.push(transform_point(&true_poses[0], &true_poses[3], &intrinsics, &intrinsics, &patch, false));
        let problem_with_loop = DpvoBaProblem {
            poses: drifted_poses,
            patches: vec![patch],
            intrinsics: vec![intrinsics; 4],
            edges: edges_with_loop,
            targets: targets_with_loop,
            weights: vec![Vector2::new(1.0, 1.0); temporal_edges.len() + 1],
            depth_damping: None,
        };
        let solved_with_loop = dpvo_ba(&problem_with_loop, &config).expect("ba solve");
        let error_with_loop = (solved_with_loop.poses[3].translation - true_poses[3].translation).norm();

        assert!(
            error_with_loop < error_no_loop,
            "closing the loop should reduce frame 3's endpoint error: with_loop={error_with_loop:.6} \
             no_loop={error_no_loop:.6}"
        );
        // The undrifted baseline (A) should reproduce the drift almost
        // exactly (no information at all constrains frame 3 there).
        assert!(
            (error_no_loop - drift.norm()).abs() < 1e-3,
            "with no edge touching frame 3 at all, BA should leave its drift essentially untouched: \
             error_no_loop={error_no_loop:.6} drift={:.6}",
            drift.norm()
        );
    }
}
