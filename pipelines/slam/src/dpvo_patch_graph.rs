//! DPVO's patch-graph sliding-window bookkeeping — Milestone M4 of
//! `docs/dpvo_droid_port_plan.md`.
//!
//! Ground truth: `E:/tools/DPVO/dpvo/dpvo.py`'s `DPVO` class (state
//! attributes, `keyframe()`, `__edges_forw`/`__edges_back`, `motionmag`, the
//! `DAMPED_LINEAR` motion model in `__call__`) and `dpvo/patchgraph.py`'s
//! `PatchGraph` (the array-of-structs state `DPVO` composes). `dpvo/
//! config.py` supplies every default constant in [`DpvoVoConfig::default`].
//!
//! # Scope: graph/policy only, no ONNX
//!
//! This module is deliberately **inference-independent** (no `ort`/
//! `ndarray`, no `onnx-inference` feature gate) — it owns exactly the parts
//! of `dpvo.py`'s state machine that are pure integer/SE3 bookkeeping:
//! frame/patch buffers, the forward/backward edge policy, the keyframe
//! (motion-magnitude) removal policy, and the damped-linear motion model.
//! Everything that needs a learned forward pass (patchify's `fnet`/`inet`,
//! the GRU update cell, `motion_probe`'s network call) lives in the sibling,
//! `onnx-inference`-gated `crate::dpvo_vo` module, which drives this graph's
//! public methods rather than duplicating its bookkeeping. This mirrors
//! `dpvo_patch_ba.rs`'s own placement rationale (see that module's doc): the
//! task's M4 brief explicitly asks for "the graph/BA logic can be
//! unconditional" while "inference-dependent parts must stay behind
//! `onnx-inference`".
//!
//! # A note on the task brief vs. upstream: hidden state is per-*edge*, not
//! per-patch
//!
//! The M4 task brief describes "a hidden-state buffer per patch". Reading
//! `dpvo.py`/`net.py` line-by-line shows this is not quite upstream's own
//! design: the GRU update cell's hidden state (`self.pg.net`, `(1, E, DIM)`)
//! is allocated **per active edge** (`append_factors`: `net = torch.zeros(1,
//! len(ii), self.DIM, ...)`), not per patch — a patch with three active
//! edges (e.g. one back-edge to itself plus two forward edges to later
//! frames) carries three independent hidden states, one per edge, that never
//! share memory. This module's [`DpvoGraphEdge::net`] therefore lives on the
//! edge record, matching upstream exactly rather than the brief's
//! description; flagged here since it is an easy point to port wrong.
//!
//! # Patch/frame addressing: no `BUFFER_SIZE`-shaped preallocation
//!
//! `dpvo.py` preallocates every per-frame/per-patch tensor to
//! `cfg.BUFFER_SIZE` (4096) rows up front (a CUDA/fixed-shape-tensor
//! necessity) and tracks a live prefix via `self.n`/`self.m`, shifting rows
//! down on keyframe removal. This port instead uses plain growable `Vec`s
//! and physically truncates/shifts them — behaviorally identical (same
//! shift-down-by-one-frame/`M`-patches compaction on removal, see
//! [`DpvoPatchGraph::keyframe`]) but without a fixed-capacity ceiling.
//! [`DpvoVoConfig::buffer_size`] is kept anyway, ported faithfully as a hard
//! cap that raises an error if exceeded (`dpvo.py`'s own `__call__` raises
//! `Exception` when `self.n+1 >= self.N`, see [`DpvoGraphError::BufferFull`]),
//! matching upstream's own user-facing behavior rather than silently growing
//! forever.
//!
//! One consequence worth stating explicitly: because patches are stored in
//! **contiguous per-frame blocks of exactly `patches_per_frame`**, and both
//! frames and patch-blocks are compacted in lockstep on removal, the
//! invariant `owner_frame(patch_id) == patch_id / patches_per_frame` holds
//! at all times — this is provably true of upstream too (see this module's
//! source comment above [`DpvoPatchGraph::owner_frame`] for the derivation),
//! so this port needs no separate `index_`/`ix` table the way `patchgraph.py`
//! carries one (that table is upstream's own tautology: `self.ix[kk] ==
//! kk // M` always, by construction — ported here as arithmetic instead of a
//! redundant array).
//!
//! # What M4 deliberately does not port (documented, not silent)
//!
//! * **DPV-SLAM mid-term ("proximity") loop closure** — implemented as of
//!   Milestone M6 (`docs/dpvo_droid_port_plan.md`'s "M6 results"), not by
//!   this module directly: `crate::dpvo_loop_closure::find_loop_edges` is
//!   the `edges_loop` port (candidate generation, `MAX_EDGE_AGE`,
//!   `BACKEND_THRESH`), `crate::dpvo_vo::DpvoOdometry::try_loop_closure`
//!   applies `GLOBAL_OPT_FREQ`'s own throttle and feeds accepted edges
//!   through this module's own [`Self::append_edges`] — the same entry
//!   point ordinary temporal edges use, per DPV-SLAM's own "same patch-BA
//!   machinery" design — and this module gained
//!   [`Self::keyframe_with_loop_protection`], the one piece of `keyframe`'s
//!   own removal-window logic (`dpvo.py:305-309`) M4 had deliberately left
//!   out (see that method's own doc). **The long-term/classical backend**
//!   (`CLASSIC_LOOP_CLOSURE`, dBoW2 retrieval) remains out of scope — see
//!   `crate::dpvo_loop_closure`'s module doc for why this codebase's
//!   existing `online_slam.rs`/`map_atlas.rs` appearance-loop pipeline
//!   already exceeds it and needs no replacement.
//! * **The "run global BA over active+inactive edges" fallback**
//!   (`dpvo.py::__run_global_BA`, triggered when `(self.pg.ii < self.n -
//!   REMOVAL_WINDOW - 1).any()`) is not ported as a *separate* BA entry
//!   point — M4's own reasoning below (why the trigger condition cannot
//!   fire without loop closure) is unchanged and still explains M4's own
//!   configuration; Milestone M6 instead reaches the same *effect* (a wider
//!   BA window whenever a stale/proximity edge exists) by generalizing
//!   `crate::dpvo_vo::DpvoOdometry::update_step`'s own windowing derivation
//!   — see that module's doc, "Windowing the BA problem" — rather than
//!   adding a second BA call site. The reasoning below for *why* M4 alone
//!   could never trigger this condition is preserved verbatim as the
//!   still-correct explanation for the non-loop-closure configuration:
//!   with `LOOP_CLOSURE`-equivalent functionality off (`config.loop_closure
//!   = None`) and `keyframe()` called exactly once per processed frame
//!   (this module's own invariant), the newest edge added by
//!   [`edges_forw`]/[`edges_back`] is never older than `PATCH_LIFETIME`
//!   (12, default) and the per-frame cleanup always prunes anything older
//!   than `REMOVAL_WINDOW` (20, default) *before* the next `update()` call
//!   can observe it — with `PATCH_LIFETIME ≤ REMOVAL_WINDOW` (true of every
//!   shipped default), the trigger condition `ii < n - REMOVAL_WINDOW - 1`
//!   can never fire in that configuration.
//! * **Inactive-edge retention** (`remove_factors(..., store=True)`'s
//!   `ii_inac`/`jj_inac`/`kk_inac`/`weight_inac`/`target_inac` arrays).
//!   **Implemented as of Milestone M8** (`docs/dpvo_droid_port_plan.md`'s "M8
//!   results") — see [`InactiveEdge`], [`Self::enable_inactive_edge_retention`],
//!   and the "Inactive-edge retention" section below for the full design and
//!   why it is keyed by [`DpvoGraphFrame::arrival_index`] rather than raw
//!   live-frame indices (a deliberate, MORE careful choice than upstream's
//!   own `ii_inac`/`jj_inac`/`kk_inac`, which are silently NOT re-shifted by
//!   a later `keyframe`-fold — `dpvo.py:283-285` only rewrites the active
//!   `ii`/`jj`/`kk` arrays, never the `_inac` ones — an assumption that only
//!   happens to hold because fold targets are always near-recent frames
//!   while inactive edges are always old ones; this port does not rely on
//!   that same assumption holding, see below).
//!
//! # Inactive-edge retention (Milestone M8, `docs/dpvo_droid_port_plan.md`)
//!
//! Off by default (`inactive_edge_cap = 0`, set by [`Self::new`]) — every
//! M4-M7 call site is unaffected byte-for-byte unless a caller opts in via
//! [`Self::enable_inactive_edge_retention`] (`crate::dpvo_vo::DpvoOdometry::new`
//! does so when `DpvoOdometryConfig::global_ba` is `Some`, mirroring how
//! `loop_closure`/`imu`/`scale_coupling` are each gated at the odometry
//! layer rather than baked into [`DpvoVoConfig`] itself — this struct stays
//! a faithful 1:1 port of `config.py`'s own field set).
//!
//! When enabled, [`Self::keyframe_inner`]'s ordinary removal-window edge drop
//! (the `to_remove`/`remove_factors(..., store=True)` port, NOT the
//! fold-frame edge drop, which upstream itself calls with `store=False` —
//! see `dpvo.py:280-281` vs. `dpvo.py:305-310`) archives each dropped edge
//! that has ever been touched by an `update()` call (`target_weight.is_some()`
//! — an edge appended but never yet solved carries no learned measurement to
//! freeze) into a bounded [`InactiveEdge`] ring buffer instead of discarding
//! it outright. The correspondence target/weight are frozen at eviction time
//! (never touched by a later GRU update, matching upstream's own `target_inac`/
//! `weight_inac` semantics — an inactive edge's measurement is whatever the
//! network last predicted for it while it was still active).
//!
//! **Why keyed by `arrival_index`, not live frame index**: this port
//! physically compacts `frames`/`patches`/`edges` on every fold
//! ([`Self::fold_frame`]), so a live index valid at archival time can go
//! stale if a LATER fold happens to shift it. Upstream's own `ii_inac`/
//! `jj_inac`/`kk_inac` are never re-shifted at all (confirmed by reading
//! `dpvo.py:283-285` directly — only `self.pg.ii`/`jj`/`kk`, the ACTIVE
//! arrays, get the `> k` shift), so upstream implicitly relies on fold
//! targets (`k = n - KEYFRAME_INDEX`, always near-recent) never exceeding an
//! inactive edge's own (always old, aged-past-`REMOVAL_WINDOW`) index — true
//! in practice given `KEYFRAME_INDEX ≪ REMOVAL_WINDOW` on every shipped
//! config, but not a maintained invariant. This port does the analogous
//! optimistic thing for ACTIVE edges (their `i`/`j`/`k` are always current,
//! by construction of the shift itself) but is deliberately MORE careful for
//! the retained INACTIVE store: each entry's `arrival_i`/`arrival_j` is
//! re-resolved against the CURRENT live frame set at the point a global-BA
//! pass consumes it (`crate::dpvo_vo::DpvoOdometry::run_global_ba`), and an
//! entry whose endpoint has since been folded away entirely (no longer
//! live) is simply skipped for that pass rather than risking a silently
//! wrong pose reference.
//!
//! **Patch state is NOT snapshotted by [`InactiveEdge`] itself** — only the
//! `(target, weight)` measurement is frozen there. An inactive edge's patch
//! inverse depth is read from the LIVE patch buffer at global-BA time (via
//! `arrival_i`/`local_patch_offset` → current live patch index) whenever its
//! owner frame is STILL LIVE — a patch's owner frame being "inactive" for
//! edge purposes does not mean the patch itself has left the live buffer,
//! only a full [`Self::fold_frame`] removes a patch. This is simpler than
//! snapshotting and strictly more correct in that common case: the global
//! pass optimizes the patch's actual current state, consistent with every
//! other BA call in this port.
//!
//! **Milestone M10 addition**: once a frame HAS been folded away,
//! [`Self::fold_frame`] separately snapshots that frame's intrinsics + patch
//! geometry into [`Self::retained_folded_frames`] (a [`RetainedFoldedFrame`]
//! per folded frame, unconditional, mirroring [`Self::retained_poses`]'s own
//! placement) — see that struct's own doc for why a widened global-BA pass
//! (`crate::dpvo_vo`'s "Milestone M10: widened global BA" section) needs
//! this to build a REAL factor for an edge whose owner has folded away,
//! rather than merely a frozen-target pull against a value it can no longer
//! read live.
//!
//! # Ported line ranges
//!
//! | Piece | Upstream source | Lines |
//! | --- | --- | --- |
//! | `PATCHES_PER_FRAME`, `BUFFER_SIZE`, `REMOVAL_WINDOW`, `OPTIMIZATION_WINDOW`, `PATCH_LIFETIME`, `KEYFRAME_INDEX`, `KEYFRAME_THRESH`, `MOTION_MODEL`, `MOTION_DAMPING` defaults | `dpvo/config.py` | 6-23 |
//! | `__edges_forw`/`__edges_back` | `dpvo/dpvo.py` | 362-375 |
//! | `append_factors`/`remove_factors` | same | 215-238 |
//! | `motionmag`/`keyframe` | same | 257-310 |
//! | `DAMPED_LINEAR` motion model (`__call__`'s `if self.n > 1` block) | same | 410-424 |
//! | `flatmeshgrid` | `dpvo/utils.py` | 85-87 |

use std::collections::{BTreeMap, HashMap};

use nalgebra::{Vector2, Vector6};
use visloc_core::geometry::SE3;

use crate::dpvo_patch_ba::{flow_mag, DpvoEdge, DpvoIntrinsics, DpvoPatch};

/// Port of `dpvo/config.py`'s VO-loop knobs (lines 6-23) — the loop-closure
/// fields (`LOOP_CLOSURE`, `BACKEND_THRESH`, `MAX_EDGE_AGE`,
/// `GLOBAL_OPT_FREQ`, `CLASSIC_LOOP_CLOSURE`, `LOOP_CLOSE_WINDOW_SIZE`,
/// `LOOP_RETR_THRESH`) are omitted, not merely left at their `False`
/// defaults — see the module doc's "What M4 deliberately does not port".
/// `CENTROID_SEL_STRAT` and `MIXED_PRECISION` are also omitted: centroid
/// selection is a patchify-time (ONNX-adjacent) concern living in
/// `crate::dpvo_vo`, and mixed precision has no meaning for this crate's
/// `f64` solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoVoConfig {
    /// Hard cap on live frame count (`config.py:6`). Exceeding it is a
    /// caller/config error (`dpvo.py:383-384`'s own `Exception`), not silent
    /// truncation — see [`DpvoGraphError::BufferFull`].
    pub buffer_size: usize,
    /// Patches sampled per incoming frame (`config.py:12`).
    pub patches_per_frame: usize,
    /// Active-edge age limit, in frames, before unconditional removal
    /// (`config.py:13`).
    pub removal_window: usize,
    /// BA's sliding pose window (`config.py:14`) — the number of trailing
    /// frames whose poses stay free (unfixed) in a `dpvo_ba` call.
    pub optimization_window: usize,
    /// How many trailing frames a patch keeps gaining new forward/backward
    /// edges (`config.py:15`).
    pub patch_lifetime: usize,
    /// Frames-back offset of the keyframe removal candidate (`config.py:18`).
    pub keyframe_index: usize,
    /// Motion-magnitude threshold below which the candidate frame is folded
    /// away rather than kept (`config.py:19`).
    pub keyframe_thresh: f64,
    /// `MOTION_DAMPING` (`config.py:23`) — damping factor on the
    /// extrapolated log-pose used to predict a new frame's initial pose.
    pub motion_damping: f64,
}

impl Default for DpvoVoConfig {
    /// Every field matches `dpvo/config.py`'s own shipped default exactly
    /// (`MOTION_MODEL='DAMPED_LINEAR'` is the only model this module
    /// implements, so it is not a separate field — see
    /// [`DpvoPatchGraph::predict_motion`]).
    fn default() -> Self {
        Self {
            buffer_size: 4096,
            patches_per_frame: 80,
            removal_window: 20,
            optimization_window: 12,
            patch_lifetime: 12,
            keyframe_index: 4,
            keyframe_thresh: 12.5,
            motion_damping: 0.5,
        }
    }
}

/// Errors from [`DpvoPatchGraph`]'s mutating methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpvoGraphError {
    /// `dpvo.py:383-384`'s own guard: `self.n + 1 >= self.N` raises. Ported
    /// as a `Result` rather than a panic since this is a plausible
    /// configuration/runtime condition for a very long sequence, not a
    /// programming bug.
    BufferFull { buffer_size: usize },
}

impl std::fmt::Display for DpvoGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferFull { buffer_size } => write!(
                f,
                "dpvo patch graph: buffer size {buffer_size} exceeded (dpvo.py's own BUFFER_SIZE guard)"
            ),
        }
    }
}

impl std::error::Error for DpvoGraphError {}

/// One live frame's pose/intrinsics/bookkeeping. Patches owned by this frame
/// occupy a contiguous block in [`DpvoPatchGraph`]'s flat patch buffer — see
/// [`DpvoPatchGraph::owner_frame`].
#[derive(Debug, Clone, PartialEq)]
pub struct DpvoGraphFrame {
    /// The monotonically increasing arrival index (`dpvo.py`'s
    /// `self.counter` value at the time this frame was accepted) — matches
    /// upstream's own choice to key `pg.tstamps_`/`pg.delta` by this integer
    /// counter, *not* the sensor's floating-point timestamp (that lives in
    /// [`DpvoPatchGraph::tlist`], looked up separately when needed).
    pub arrival_index: usize,
    pub pose: SE3,
    pub intrinsics: DpvoIntrinsics,
}

/// One active patch-graph edge: patch `k` (owned by frame `k /
/// patches_per_frame`) observed in frame `j`, carrying the GRU's persistent
/// per-edge hidden state. See the module doc's "hidden state is per-edge"
/// section for why `net` lives here rather than on the patch.
#[derive(Debug, Clone, PartialEq)]
pub struct DpvoGraphEdge {
    /// Owner (source) frame of patch `k` — always `k / patches_per_frame`
    /// (see [`DpvoPatchGraph::owner_frame`]), stored redundantly here purely
    /// so [`DpvoEdge`] (the plain triple `dpvo_patch_ba` consumes) can be
    /// built by value without a graph reference.
    pub i: usize,
    pub j: usize,
    pub k: usize,
    /// The GRU update cell's persistent hidden state for this edge, `(1,
    /// DIM=384)` flattened — `append_factors` (`dpvo.py:220`) initializes
    /// this to zero for a newly created edge; `crate::dpvo_vo` overwrites it
    /// with the update cell's `net_out` every `update()` call.
    pub net: Vec<f32>,
    /// The GRU update cell's last learned target pixel/weight for this edge
    /// (`dpvo.py:342-343`'s `self.pg.target`/`self.pg.weight`) — `None`
    /// until the first `update()` call touches this edge (a newly appended
    /// edge has no target/weight yet, matching upstream, where `pg.target`/
    /// `pg.weight` are populated only by `update()`, never by
    /// `append_factors`).
    pub target_weight: Option<(nalgebra::Vector2<f64>, nalgebra::Vector2<f64>)>,
}

/// One retained "inactive" edge (Milestone M8) — the frozen correspondence
/// measurement of an edge dropped by the ordinary removal-window rule (NOT
/// a fold-frame drop, which is never retained, matching upstream's own
/// `store=False` there — see the module doc). Keyed by
/// [`DpvoGraphFrame::arrival_index`], not live frame index — see the module
/// doc's "Inactive-edge retention" section for why.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InactiveEdge {
    /// Source (patch-owning) frame's stable arrival index.
    pub arrival_i: usize,
    /// Target frame's stable arrival index.
    pub arrival_j: usize,
    /// Patch `k`'s offset within its owner frame's contiguous block
    /// (`0..patches_per_frame`) — combined with `arrival_i` (the owner,
    /// always equal to the edge's own source frame — see
    /// [`DpvoPatchGraph::owner_frame`]'s invariant) to resolve the CURRENT
    /// live patch index at global-BA time.
    pub local_patch_offset: usize,
    /// Frozen learned target pixel, as of the last `update()` call that
    /// touched this edge before it aged out.
    pub target: Vector2<f64>,
    /// Frozen learned 2-channel confidence, same provenance as `target`.
    pub weight: Vector2<f64>,
}

/// Milestone M10 (`docs/dpvo_droid_port_plan.md`): a folded-away frame's
/// FULL geometry, snapshotted the instant [`DpvoPatchGraph::fold_frame`]
/// removes it — not just its pose (M9's [`DpvoPatchGraph::retained_poses`]),
/// but also the intrinsics and per-patch `(x, y, inverse_depth)` triples that
/// [`crate::dpvo_patch_ba::dpvo_ba`] needs to build a REAL reprojection
/// factor. M8's own inactive-edge retention kept a dropped edge's frozen
/// `(target, weight)` measurement alive past a fold, but explicitly did NOT
/// snapshot patch state (module doc, "Patch state is NOT snapshotted") —
/// reasonable when the owner frame stays live, but useless once
/// [`DpvoPatchGraph::fold_frame`] has physically drained that frame's
/// patches from [`DpvoPatchGraph::patches`]. This struct is what closes that
/// gap: see [`DpvoPatchGraph::retained_folded_frames`]'s own doc for the
/// full "why", and `crate::dpvo_vo`'s "Milestone M10: widened global BA"
/// section for how a global-BA pass folds this back into a solvable
/// [`crate::dpvo_patch_ba::DpvoBaProblem`].
#[derive(Debug, Clone, PartialEq)]
pub struct RetainedFoldedFrame {
    pub intrinsics: DpvoIntrinsics,
    /// Exactly `patches_per_frame` entries, indexed by
    /// [`InactiveEdge::local_patch_offset`] — the SAME contiguous-block
    /// convention live patches use (see [`DpvoPatchGraph::owner_frame`]'s
    /// invariant), just snapshotted at fold time rather than read live.
    pub patches: Vec<DpvoPatch>,
}

/// The DPVO patch graph: live frames, their patches, and the active
/// forward/backward edges between them. See the module doc for scope and
/// the deliberate omissions (loop closure, global-BA fallback), and the
/// "Inactive-edge retention" section for Milestone M8's own addition.
#[derive(Debug, Clone, PartialEq)]
pub struct DpvoPatchGraph {
    config: DpvoVoConfig,
    frames: Vec<DpvoGraphFrame>,
    /// Flat, per-frame-contiguous-block patch buffer — length always
    /// `frames.len() * config.patches_per_frame`.
    patches: Vec<DpvoPatch>,
    edges: Vec<DpvoGraphEdge>,
    /// `dpvo.py`'s `self.counter` — total frames ever *seen* (accepted or
    /// rejected by `motion_probe`), used only to key [`Self::delta`] and to
    /// index [`Self::tlist`].
    counter: usize,
    /// `dpvo.py`'s `self.tlist` — every incoming timestamp, including
    /// rejected frames (used by the `DAMPED_LINEAR` motion model's
    /// varying-frame-rate correction factor, `dpvo.py:416-417`).
    tlist: Vec<f64>,
    /// `dpvo.py`'s `self.pg.delta` — for a frame folded away by
    /// [`Self::keyframe`] or rejected by the (ONNX-dependent, hence not
    /// implemented here — see `crate::dpvo_vo`) `motion_probe` gate, the
    /// `(parent_arrival_index, relative_pose)` needed to reconstruct its
    /// pose at trajectory-finalization time (`dpvo.py::get_pose`, ported as
    /// [`Self::reconstruct_pose`]). Keyed by `arrival_index` (see
    /// [`DpvoGraphFrame::arrival_index`]'s doc for why this is the counter,
    /// not the sensor timestamp).
    delta: HashMap<usize, (usize, SE3)>,
    is_initialized: bool,
    /// Bounded store of retained inactive edges — see the module doc's
    /// "Inactive-edge retention" section. Empty and never grown unless
    /// [`Self::enable_inactive_edge_retention`] has been called with a
    /// nonzero cap.
    ///
    /// # Milestone M10: decimating retention, not FIFO
    ///
    /// M8 shipped this as a `VecDeque` with plain oldest-evicted-first
    /// eviction — simple, but M8's own real-MH_01 finding
    /// (`docs/dpvo_droid_port_plan.md`'s "M8 results", point 3) was that this
    /// throws away exactly the OLD edges a later loop closure needs: a
    /// recency-biased cap only ever remembers the most recent `cap` edges,
    /// a moving window whose reach is independent of (and, on a long run,
    /// much narrower than) the full sequence. This port instead keeps a
    /// plain `Vec` plus [`Self::inactive_edge_retention_stride`] and
    /// decimates via "keep every Nth archived edge, doubling N (and halving
    /// the buffer) whenever the cap would be exceeded" — the same shape of
    /// algorithm streaming histogram/log-retention systems use to keep a
    /// bounded sample that stays roughly UNIFORMLY spread over the entire
    /// history seen so far, not just its tail. See
    /// [`Self::archive_inactive_edge`]'s doc for the exact algorithm and
    /// `dpvo_patch_graph.rs`'s own `retention_preserves_temporal_coverage_across_the_full_history`
    /// test for the coverage property this buys (old edges from near the
    /// very start of a long run remain retained, at a coarser sampling
    /// rate, alongside recent ones — not merely a shifted recency window).
    inactive_edges: Vec<InactiveEdge>,
    /// `0` disables retention entirely (the default, every M4-M7 call site's
    /// behavior) — see [`Self::enable_inactive_edge_retention`].
    inactive_edge_cap: usize,
    /// Cumulative count of inactive edges evicted (either never sampled, by
    /// the decimation stride, or thinned out once the cap was hit) — a
    /// Milestone M8 diagnostic, still named `inactive_edges_evicted_total`
    /// (`crate::dpvo_vo::DpvoGlobalBaDiagnostics`) after the M10 retention
    /// rewrite since its meaning ("archived edges no longer retained") is
    /// unchanged.
    inactive_edges_evicted: usize,
    /// Milestone M10: total archive attempts ever seen (kept or not) — the
    /// decimation counter [`Self::archive_inactive_edge`]'s sampling decision
    /// is keyed on.
    inactive_edges_archived_seen: u64,
    /// Milestone M10: current "keep 1 in every `stride`" sampling rate —
    /// starts at `1` (keep everything) and only ever doubles, never shrinks
    /// (a monotonically coarsening sample, matching the streaming-decimation
    /// algorithms this borrows from).
    inactive_edge_retention_stride: u64,
    /// Milestone M10 (`docs/dpvo_droid_port_plan.md`): every folded-away
    /// frame's intrinsics + per-patch `(x, y, inverse_depth)` snapshot at the
    /// instant [`Self::fold_frame`] removed it — see [`RetainedFoldedFrame`]'s
    /// own doc for why this exists alongside [`Self::retained_poses`].
    /// Unconditional and uncapped, mirroring [`Self::retained_poses`]'s own
    /// "cheap even over a long sequence" reasoning (`patches_per_frame`
    /// `DpvoPatch`es, each 3 `f64`s, per folded frame).
    retained_folded_frames: BTreeMap<usize, RetainedFoldedFrame>,
    /// Milestone M9 (`docs/dpvo_droid_port_plan.md`): every folded-away
    /// frame's FINAL world pose (the value it held the instant
    /// [`Self::fold_frame`] removed it from [`Self::frames`]), keyed by
    /// [`DpvoGraphFrame::arrival_index`] — an append-only, unbounded store
    /// (unlike [`Self::inactive_edges`], never capped: this holds one `SE3`
    /// per folded frame, cheap even over a very long sequence — see the
    /// module doc's "Retained pose history" section for the memory-growth
    /// note and why the M8 finding makes this necessary: `fold_frame` is the
    /// ONLY way a frame's pose stops being reachable at all once it leaves
    /// `frames` — M8's own inactive-edge retention kept EDGES alive past a
    /// fold but never the POSE a resolved edge's endpoint needs, and that
    /// gap is exactly what this store closes). `crate::dpvo_vo`'s Sim3
    /// pose-graph backend (Milestone M9) is the one reader; entries are also
    /// the one thing that store OVERWRITES in place (not merely appends),
    /// when a later Sim3 solve corrects a folded frame's pose (see
    /// `crate::dpvo_sim3_backend`'s module doc).
    retained_poses: BTreeMap<usize, SE3>,
}

impl DpvoPatchGraph {
    pub fn new(config: DpvoVoConfig) -> Self {
        Self {
            config,
            frames: Vec::new(),
            patches: Vec::new(),
            edges: Vec::new(),
            counter: 0,
            tlist: Vec::new(),
            delta: HashMap::new(),
            is_initialized: false,
            inactive_edges: Vec::new(),
            inactive_edge_cap: 0,
            inactive_edges_evicted: 0,
            inactive_edges_archived_seen: 0,
            inactive_edge_retention_stride: 1,
            retained_folded_frames: BTreeMap::new(),
            retained_poses: BTreeMap::new(),
        }
    }

    /// Opt into inactive-edge retention with a bounded cap — see
    /// [`Self::inactive_edges`]'s own doc for the Milestone M10 decimating
    /// policy that keeps this bounded store spread over the WHOLE history
    /// rather than just its recent tail. Idempotent; calling again simply
    /// replaces the cap (a smaller cap takes effect gradually, via eviction
    /// on the next few archival events, not by immediately truncating
    /// existing entries — kept simple since the one real caller,
    /// `crate::dpvo_vo::DpvoOdometry::new`, only ever calls this once, at
    /// construction).
    pub fn enable_inactive_edge_retention(&mut self, cap: usize) {
        self.inactive_edge_cap = cap;
    }

    /// Every currently retained inactive edge — see [`InactiveEdge`]'s doc.
    pub fn inactive_edges(&self) -> &[InactiveEdge] {
        &self.inactive_edges
    }

    /// Every folded-away frame's retained intrinsics + patch geometry — see
    /// [`RetainedFoldedFrame`]'s own doc.
    pub fn retained_folded_frames(&self) -> &BTreeMap<usize, RetainedFoldedFrame> {
        &self.retained_folded_frames
    }

    /// Mutable access so a Milestone M10 widened global-BA pass can write a
    /// solved inverse depth back into an already-folded frame's patches — the
    /// folded-frame counterpart of [`Self::patches_mut`] (the live one) and
    /// [`Self::retained_poses_mut`] (the folded-pose one).
    pub fn retained_folded_frames_mut(&mut self) -> &mut BTreeMap<usize, RetainedFoldedFrame> {
        &mut self.retained_folded_frames
    }

    /// `(currently retained, cumulative evicted)` — Milestone M8 diagnostic,
    /// unchanged in meaning by M10's retention-policy rewrite.
    pub fn inactive_edge_stats(&self) -> (usize, usize) {
        (self.inactive_edges.len(), self.inactive_edges_evicted)
    }

    /// Milestone M9: every folded-away frame's retained pose, keyed by
    /// `arrival_index` — see [`Self::retained_poses`]'s own field doc.
    pub fn retained_poses(&self) -> &BTreeMap<usize, SE3> {
        &self.retained_poses
    }

    /// Mutable access so `crate::dpvo_sim3_backend`'s pose-graph solve can
    /// write corrected poses back into already-folded frames (the live
    /// counterpart is [`Self::frames_mut`]).
    pub fn retained_poses_mut(&mut self) -> &mut BTreeMap<usize, SE3> {
        &mut self.retained_poses
    }

    pub fn config(&self) -> &DpvoVoConfig {
        &self.config
    }

    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    pub fn set_initialized(&mut self, value: bool) {
        self.is_initialized = value;
    }

    pub fn counter(&self) -> usize {
        self.counter
    }

    pub fn frames(&self) -> &[DpvoGraphFrame] {
        &self.frames
    }

    pub fn patches(&self) -> &[DpvoPatch] {
        &self.patches
    }

    /// Mutable patch access for `crate::dpvo_vo`'s BA write-back step (the
    /// solved inverse depths — and, for a windowed re-linearization, the
    /// anchor pixel never changes, only `inverse_depth` does, but this
    /// exposes the whole record for simplicity).
    pub fn patches_mut(&mut self) -> &mut [DpvoPatch] {
        &mut self.patches
    }

    /// Mutable frame access for `crate::dpvo_vo`'s BA write-back step (the
    /// solved poses).
    pub fn frames_mut(&mut self) -> &mut [DpvoGraphFrame] {
        &mut self.frames
    }

    pub fn edges(&self) -> &[DpvoGraphEdge] {
        &self.edges
    }

    pub fn edges_mut(&mut self) -> &mut [DpvoGraphEdge] {
        &mut self.edges
    }

    /// Patch `k`'s owner (source) frame. Always `k / patches_per_frame` —
    /// see the module doc's "Patch/frame addressing" section for the
    /// derivation of why this holds at all times, including after
    /// [`Self::keyframe`] removals (upstream's own `self.ix[kk]` is the same
    /// tautology, `patchgraph.py:34`/`107-108`).
    pub fn owner_frame(&self, patch_id: usize) -> usize {
        patch_id / self.config.patches_per_frame
    }

    /// Every patch id owned by `frame`.
    pub fn patch_ids_of_frame(&self, frame: usize) -> std::ops::Range<usize> {
        let m = self.config.patches_per_frame;
        frame * m..(frame + 1) * m
    }

    /// Record an incoming timestamp and predict the new frame's initial pose
    /// via the `DAMPED_LINEAR` motion model (`dpvo.py`'s `__call__`, lines
    /// 410-424) — the *only* motion model this module implements
    /// (`config.py`'s alternative, a bare "hold last pose" branch, is folded
    /// into the `n <= 1` case below rather than kept as a separate
    /// `MOTION_MODEL` switch, since `dpvo.py`'s own `else` branch **is**
    /// exactly "hold last pose", identical to what this module does when
    /// there are fewer than 2 frames to extrapolate from).
    ///
    /// Must be called exactly once per incoming frame, *before* deciding
    /// whether to accept or reject it (this only touches [`Self::tlist`],
    /// which upstream appends unconditionally at the top of `__call__`,
    /// before `motion_probe`'s accept/reject gate) — accepting later calls
    /// [`Self::commit_frame`] with the returned pose as the new frame's
    /// initial estimate.
    ///
    /// # Damped-linear derivation (`dpvo.py:410-424`)
    ///
    /// With at least two existing frames `n-1`, `n-2`:
    /// `xi = MOTION_DAMPING · fac · log(P[n-1] · P[n-2]⁻¹)`,
    /// `pose[n] = exp(xi) · P[n-1]`, where `fac = (c-b)/(b-a)` corrects for
    /// non-uniform frame timing (`a, b, c` are the last three entries of
    /// `[1,1,1] + tlist`, i.e. left-padded with `1.0` if fewer than three
    /// timestamps exist yet — `dpvo.py:416`). With fewer than two existing
    /// frames, the pose is simply held at the last frame's value (or
    /// identity if there is no previous frame at all).
    pub fn begin_frame(&mut self, timestamp: f64) -> SE3 {
        self.tlist.push(timestamp);
        self.counter += 1;
        self.predict_motion()
    }

    fn predict_motion(&self) -> SE3 {
        let n = self.frames.len();
        if n == 0 {
            return SE3::identity();
        }
        if n == 1 {
            return self.frames[0].pose.clone();
        }
        let p1 = &self.frames[n - 1].pose;
        let p2 = &self.frames[n - 2].pose;

        // `*_, a, b, c = [1]*3 + self.tlist` (dpvo.py:416): left-pad with
        // three 1.0s, then take the *last three* entries of the padded
        // list.
        let mut padded: Vec<f64> = vec![1.0, 1.0, 1.0];
        padded.extend_from_slice(&self.tlist);
        let len = padded.len();
        let (a, b, c) = (padded[len - 3], padded[len - 2], padded[len - 1]);
        let fac = (c - b) / (b - a);

        let relative = p1.compose(&p2.inverse());
        let xi: Vector6<f64> = self.config.motion_damping * fac * relative.log();
        SE3::exp(&xi).compose(p1)
    }

    /// Commit a new frame (patches already extracted/depth-initialized by
    /// the caller — see `crate::dpvo_vo`'s per-frame step) with the pose
    /// [`Self::begin_frame`] predicted. Returns the new frame's index
    /// (always `n_frames() - 1` after this call) or [`DpvoGraphError::BufferFull`]
    /// if `dpvo.py`'s own `BUFFER_SIZE` guard (`dpvo.py:383-384`) would trip.
    ///
    /// `patches.len()` must equal `config.patches_per_frame` — enforced by a
    /// debug assertion (a caller-side invariant, not a data-dependent error
    /// condition, matching this crate's convention of reserving `Result` for
    /// genuinely-fallible-at-runtime conditions).
    pub fn commit_frame(
        &mut self,
        pose: SE3,
        intrinsics: DpvoIntrinsics,
        patches: Vec<DpvoPatch>,
    ) -> Result<usize, DpvoGraphError> {
        debug_assert_eq!(patches.len(), self.config.patches_per_frame);
        if self.frames.len() + 1 >= self.config.buffer_size {
            return Err(DpvoGraphError::BufferFull {
                buffer_size: self.config.buffer_size,
            });
        }
        self.frames.push(DpvoGraphFrame {
            arrival_index: self.counter - 1,
            pose,
            intrinsics,
        });
        self.patches.extend(patches);
        Ok(self.frames.len() - 1)
    }

    /// Reject the just-`begin_frame`d timestamp (the ONNX-dependent
    /// `motion_probe` gate lives in `crate::dpvo_vo`; this method only
    /// records the bookkeeping consequence, matching `dpvo.py:443-444`):
    /// `pg.delta[counter-1] = (counter-2, Identity)` — the rejected frame's
    /// pose is defined, at trajectory-finalization time, as *exactly* its
    /// immediate predecessor's pose (an identity relative transform), not a
    /// motion-model extrapolation.
    pub fn reject_pending_frame(&mut self) {
        if self.counter >= 2 {
            self.delta
                .insert(self.counter - 1, (self.counter - 2, SE3::identity()));
        }
    }

    /// `dpvo.py::__edges_forw` (lines 362-368): patches owned by frames in
    /// `[max(n - patch_lifetime, 0), n - 1)` get a new edge targeting the
    /// just-committed newest frame `n - 1`. Call *after* [`Self::commit_frame`]
    /// (this reads `n_frames()` post-commit, matching `dpvo.py`'s own
    /// call site, which runs `__edges_forw`/`__edges_back` after `self.n +=
    /// 1`).
    ///
    /// Returns `(patch_id, target_frame)` pairs, matching
    /// `flatmeshgrid`'s `(patch_ids, frames)` output order
    /// (`dpvo.py:366-368`) — `append_factors`'s own parameter name for this
    /// is `(ii, jj)` even though `ii` here means *patch ids*, not frame
    /// indices (`dpvo.py:215`'s own confusing-but-verified naming, see
    /// [`Self::append_edges`]'s doc).
    pub fn edges_forw(&self) -> Vec<(usize, usize)> {
        let n = self.frames.len();
        if n == 0 {
            return Vec::new();
        }
        let r = self.config.patch_lifetime;
        let m = self.config.patches_per_frame;
        let t0 = m * n.saturating_sub(r);
        let t1 = m * n.saturating_sub(1);
        let target = n - 1;
        (t0..t1).map(|patch_id| (patch_id, target)).collect()
    }

    /// `dpvo.py::__edges_back` (lines 370-375): the just-committed newest
    /// frame's own patches get new edges targeting every frame in
    /// `[max(n - patch_lifetime, 0), n)` — including itself (a `i == j`
    /// self-edge, ported faithfully, not filtered out; see the module doc's
    /// citation for why upstream includes it).
    pub fn edges_back(&self) -> Vec<(usize, usize)> {
        let n = self.frames.len();
        if n == 0 {
            return Vec::new();
        }
        let r = self.config.patch_lifetime;
        let m = self.config.patches_per_frame;
        let t0 = m * n.saturating_sub(1);
        let t1 = m * n;
        let frame_lo = n.saturating_sub(r);
        let mut out = Vec::with_capacity((t1 - t0) * (n - frame_lo));
        for patch_id in t0..t1 {
            for target in frame_lo..n {
                out.push((patch_id, target));
            }
        }
        out
    }

    /// Port of `dpvo.py::append_factors` (lines 215-221): append new active
    /// edges from `(patch_id, target_frame)` pairs, initializing each edge's
    /// GRU hidden state to zero (`torch.zeros(1, len(ii), DIM)`,
    /// `dpvo.py:220`) and no target/weight yet.
    pub fn append_edges(&mut self, pairs: &[(usize, usize)], hidden_dim: usize) {
        self.edges.reserve(pairs.len());
        for &(patch_id, target) in pairs {
            self.edges.push(DpvoGraphEdge {
                i: self.owner_frame(patch_id),
                j: target,
                k: patch_id,
                net: vec![0.0; hidden_dim],
                target_weight: None,
            });
        }
    }

    /// `dpvo.py::motionmag` (lines 257-264): the mean [`flow_mag`] (`beta =
    /// 0.5`) over every currently-active edge with source frame `i` and
    /// target frame `j`. Returns `0.0` if no such edge exists (upstream's
    /// `flow.mean()` over an empty tensor is `nan`; this port returns `0.0`
    /// instead — a deliberate, documented deviation, chosen because
    /// `keyframe()`'s only use of this value is a `<` comparison against
    /// `KEYFRAME_THRESH`, where `nan` would silently and permanently disable
    /// keyframe folding for that frame pair rather than upstream's evident
    /// intent, and because this situation cannot arise under this module's
    /// own edge policy in the first place — `keyframe()` only calls this
    /// with `i, j` inside the always-freshly-connected `patch_lifetime`
    /// window).
    pub fn motionmag(&self, i: usize, j: usize) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for edge in &self.edges {
            if edge.i == i && edge.j == j {
                let frame_i = &self.frames[i];
                let frame_j = &self.frames[j];
                sum += flow_mag(
                    &frame_i.pose,
                    &frame_j.pose,
                    &frame_i.intrinsics,
                    &frame_j.intrinsics,
                    &self.patches[edge.k],
                    0.5,
                );
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// Port of `dpvo.py::keyframe` (lines 266-310). Returns the physically
    /// removed frame's pre-removal index, if the motion-magnitude gate
    /// folded one away — the caller (`crate::dpvo_vo`) must apply the exact
    /// same removal to its parallel ONNX feature store (per-frame feature
    /// pyramids, per-patch anchor/context features), since this module owns
    /// only the pose/patch/edge bookkeeping, not those tensors. See the
    /// module doc's "Patch/frame addressing" section for why compaction here
    /// (shift every later frame down by one, every later patch block down by
    /// `patches_per_frame`) preserves `owner_frame(k) == k / M` unchanged.
    ///
    /// After the fold check (whether or not it fired), every active edge
    /// whose source frame has aged past `removal_window` is unconditionally
    /// dropped (`dpvo.py:305-310`, minus the `LOOP_CLOSURE`-only exemption —
    /// see [`Self::keyframe_with_loop_protection`] for that variant, added in
    /// Milestone M6).
    pub fn keyframe(&mut self) -> Option<usize> {
        self.keyframe_inner(None)
    }

    /// Milestone M6 (`docs/dpvo_droid_port_plan.md`) variant of
    /// [`Self::keyframe`] that additionally exempts loop-closure edges from
    /// the unconditional removal-window drop — the one piece of `keyframe`
    /// (`dpvo.py:305-310`) M4's own module doc flagged as "the
    /// `LOOP_CLOSURE`-only exemption" and deliberately left unported:
    ///
    /// ```python
    /// to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW
    /// if self.cfg.LOOP_CLOSURE:
    ///     lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
    ///     to_remove = to_remove & ~lc_edges
    /// self.remove_factors(to_remove, store=True)
    /// ```
    ///
    /// (`dpvo.py:305-309`). A loop edge (defined here purely by its temporal
    /// gap, `jj - ii > 30`, matching `crate::dpvo_loop_closure`'s own
    /// `min_loop_gap` default and `optim_utils.py::reduce_edges`'s hardcoded
    /// `(j - i) < 30` literal) survives the removal-window drop for as long
    /// as its *target* frame `jj` is still within the last `optimization_window`
    /// frames — this is upstream's own bound on how long a reactivated old
    /// patch's edge stays alive: once `jj` itself ages past that window, the
    /// exemption stops applying and the edge is dropped on a later call,
    /// exactly like an ordinary temporal edge. Callers should use this
    /// instead of [`Self::keyframe`] whenever loop closure is enabled (see
    /// `crate::dpvo_vo::DpvoOdometry`'s own branch on
    /// `config.loop_closure.is_some()`); using [`Self::keyframe`] with loop
    /// edges present would silently prune them the very first time their
    /// owner frame `ii` ages past `removal_window` — often within the same
    /// call that added them.
    pub fn keyframe_with_loop_protection(&mut self, optimization_window: usize) -> Option<usize> {
        self.keyframe_inner(Some(optimization_window))
    }

    /// Shared implementation for [`Self::keyframe`]/
    /// [`Self::keyframe_with_loop_protection`] — `loop_protection` is `None`
    /// for the former (no exemption, M4's own behavior, confirmed unchanged
    /// by the fact that [`Self::keyframe`] is a one-line wrapper around
    /// this) and `Some(optimization_window)` for the latter.
    fn keyframe_inner(&mut self, loop_protection: Option<usize>) -> Option<usize> {
        let n = self.frames.len();
        let ki = self.config.keyframe_index;
        let removed = if n > ki + 1 && ki >= 1 {
            let i = n - ki - 1;
            let j = n - ki + 1;
            let m = self.motionmag(i, j) + self.motionmag(j, i);
            if m / 2.0 < self.config.keyframe_thresh {
                Some(self.fold_frame(n - ki))
            } else {
                None
            }
        } else {
            None
        };

        // to_remove = ix[kk] < n - REMOVAL_WINDOW (dpvo.py:305), evaluated
        // against the *post-fold* frame count.
        let n_after = self.frames.len();
        let removal_window = self.config.removal_window;
        let threshold = n_after.saturating_sub(removal_window);
        let patches_per_frame = self.config.patches_per_frame;
        const MIN_LOOP_GAP: usize = 30; // `optim_utils.py::reduce_edges`'s own hardcoded `(j - i) < 30` literal.
                                        // Milestone M8: this used to be a plain `self.edges.retain(...)`;
                                        // rewritten to a single-pass partition so a dropped edge can be
                                        // archived into `self.inactive_edges` (`remove_factors(...,
                                        // store=True)`'s port — see the module doc's "Inactive-edge
                                        // retention" section) instead of merely discarded. `retain`'s
                                        // closure has no side-channel for the removed items, so this takes
                                        // ownership of the old `Vec` via `std::mem::take` and rebuilds it —
                                        // still a single O(edges) pass, not the O(n^2) a `Vec::remove`-based
                                        // rewrite would be.
        let old_edges = std::mem::take(&mut self.edges);
        self.edges = Vec::with_capacity(old_edges.len());
        for edge in old_edges {
            let owner_frame = edge.k / patches_per_frame;
            let keep = if owner_frame >= threshold {
                true // Not stale — kept unconditionally, same as `keyframe()`.
            } else {
                match loop_protection {
                    Some(optimization_window) => {
                        let is_loop_edge = edge.j.saturating_sub(edge.i) > MIN_LOOP_GAP;
                        let target_still_in_window = edge.j + optimization_window > n_after;
                        is_loop_edge && target_still_in_window
                    }
                    None => false,
                }
            };
            if keep {
                self.edges.push(edge);
            } else {
                self.archive_inactive_edge(&edge, patches_per_frame);
            }
        }

        removed
    }

    /// Archive one edge dropped by the ordinary removal-window rule (never
    /// called from [`Self::fold_frame`]'s own edge drop — that one matches
    /// upstream's `store=False`, see the module doc) into
    /// [`Self::inactive_edges`], subject to [`Self::inactive_edge_cap`].
    /// No-op if retention is disabled (`inactive_edge_cap == 0`, the
    /// default) or the edge was never touched by an `update()` call
    /// (`target_weight` still `None` — no measurement exists to freeze).
    ///
    /// # Milestone M10: decimating sample, not FIFO
    ///
    /// Every archive attempt increments [`Self::inactive_edges_archived_seen`]
    /// first; the edge is actually kept only if `(seen - 1) % stride == 0`
    /// (`stride` starts at `1`, i.e. "keep everything", so this is a no-op
    /// change until the cap is first hit). If keeping it would push the
    /// store's length past the cap, the store is thinned in place — every
    /// OTHER retained entry (by position, not recency) is dropped and
    /// `stride` doubles — rather than popping the single oldest entry. One
    /// thinning pass always suffices: pushing one entry at a time can only
    /// ever put the buffer one over the cap, and halving a buffer of length
    /// `cap + 1` leaves `⌈(cap + 1) / 2⌉ ≤ cap` for any `cap ≥ 1`. Because
    /// thinning removes evenly-spaced entries rather than only the front,
    /// the surviving set stays spread across the ENTIRE archival history
    /// seen so far (at a progressively coarser rate) instead of collapsing
    /// to a pure recency window — see [`Self::inactive_edges`]'s own doc for
    /// why M8's plain FIFO cap was the direct cause of a real-run finding
    /// (`docs/dpvo_droid_port_plan.md`'s M8 results) that a loop's own old
    /// edges were long gone from this store by the time a loop fired.
    fn archive_inactive_edge(&mut self, edge: &DpvoGraphEdge, patches_per_frame: usize) {
        if self.inactive_edge_cap == 0 {
            return;
        }
        let Some((target, weight)) = edge.target_weight else {
            return;
        };
        // `edge.i` is always patch `k`'s owner frame (see
        // `DpvoPatchGraph::owner_frame`'s invariant, restated in
        // `append_edges`'s own construction), so no separate lookup is
        // needed for the patch's owning arrival index.
        let arrival_i = self.frames[edge.i].arrival_index;
        let arrival_j = self.frames[edge.j].arrival_index;
        let local_patch_offset = edge.k % patches_per_frame;

        let seen_before = self.inactive_edges_archived_seen;
        self.inactive_edges_archived_seen += 1;
        if seen_before % self.inactive_edge_retention_stride != 0 {
            // Decimated out by the current sampling rate — never retained at
            // all (still counted as "lost to the retention policy").
            self.inactive_edges_evicted += 1;
            return;
        }

        self.inactive_edges.push(InactiveEdge {
            arrival_i,
            arrival_j,
            local_patch_offset,
            target,
            weight,
        });
        if self.inactive_edges.len() > self.inactive_edge_cap {
            let before = self.inactive_edges.len();
            let mut kept_index = 0usize;
            for i in (0..before).step_by(2) {
                self.inactive_edges[kept_index] = self.inactive_edges[i];
                kept_index += 1;
            }
            self.inactive_edges.truncate(kept_index);
            self.inactive_edges_evicted += before - kept_index;
            self.inactive_edge_retention_stride *= 2;
        }
    }

    /// Fold frame `k` away: record its relative-pose delta from `k - 1`
    /// (`dpvo.py:274-278`), drop every edge touching it (`store=False`,
    /// `dpvo.py:280-281`), then physically compact frames/patches `[k+1,
    /// n)` down into `[k, n-1)` and shift every remaining edge's frame
    /// indices `> k` down by one / patch ids `> k`'s owner down by `M`
    /// (`dpvo.py:283-299`).
    fn fold_frame(&mut self, k: usize) -> usize {
        let m = self.config.patches_per_frame;
        let n = self.frames.len();

        let t0 = self.frames[k - 1].arrival_index;
        let t1 = self.frames[k].arrival_index;
        // dP = poses[k] * poses[k-1].inv() (dpvo.py:277).
        let delta_pose = self.frames[k]
            .pose
            .compose(&self.frames[k - 1].pose.inverse());
        self.delta.insert(t1, (t0, delta_pose));
        // Milestone M9: archive frame k's own FINAL pose (unconditional,
        // unlike inactive-edge retention's opt-in cap — see
        // `Self::retained_poses`'s own doc) before it stops being reachable
        // via `self.frames` at all.
        self.retained_poses.insert(t1, self.frames[k].pose.clone());
        // Milestone M10: also snapshot frame k's intrinsics + patch geometry
        // (unconditional, same reasoning as `retained_poses` above) — see
        // [`RetainedFoldedFrame`]'s own doc for why this is needed alongside
        // the pose: a widened global-BA pass whose free range reaches a
        // FOLDED frame needs this frame's patches' `(x, y, inverse_depth)` to
        // build a real reprojection factor, not just its pose.
        self.retained_folded_frames.insert(
            t1,
            RetainedFoldedFrame {
                intrinsics: self.frames[k].intrinsics,
                patches: self.patches[k * m..(k + 1) * m].to_vec(),
            },
        );

        // to_remove = (ii == k) | (jj == k); store=False (dpvo.py:280-281).
        self.edges.retain(|edge| edge.i != k && edge.j != k);

        // kk[ii>k] -= M; ii[ii>k] -= 1; jj[jj>k] -= 1 (dpvo.py:283-285).
        for edge in &mut self.edges {
            if edge.i > k {
                edge.k -= m;
                edge.i -= 1;
            }
            if edge.j > k {
                edge.j -= 1;
            }
        }

        // Physically compact frame/patch storage (dpvo.py:287-297).
        self.frames.remove(k);
        self.patches.drain(k * m..(k + 1) * m);
        debug_assert_eq!(self.frames.len() * m, self.patches.len());
        let _ = n;
        k
    }

    /// Reconstruct frame `arrival_index`'s pose at trajectory-finalization
    /// time (`dpvo.py::get_pose`, called from `terminate()`). If the frame
    /// is still live, this is simply its current (BA-refined) pose; if it
    /// was folded away or rejected, this recurses through [`Self::delta`]
    /// (`t = t0 -> dP * get_pose(t0)`, `dpvo.py:166-171`) until it reaches a
    /// live frame. `live_pose_of` is the caller's lookup from a live frame's
    /// `arrival_index` to its current pose (this module does not keep a
    /// reverse `arrival_index -> live frame` index of its own, since
    /// `crate::dpvo_vo` already has to maintain one for its parallel feature
    /// store).
    pub fn reconstruct_pose(
        &self,
        arrival_index: usize,
        live_pose_of: &dyn Fn(usize) -> Option<SE3>,
    ) -> Option<SE3> {
        if let Some(pose) = live_pose_of(arrival_index) {
            return Some(pose);
        }
        let (parent, delta_pose) = self.delta.get(&arrival_index)?;
        let parent_pose = self.reconstruct_pose(*parent, live_pose_of)?;
        Some(delta_pose.compose(&parent_pose))
    }

    pub fn tlist(&self) -> &[f64] {
        &self.tlist
    }
}

/// Build the `dpvo_patch_ba::DpvoEdge` triples for the current active-edge
/// set, in the same order as [`DpvoPatchGraph::edges`] — a convenience the
/// `crate::dpvo_vo` orchestration layer uses when assembling a
/// [`crate::dpvo_patch_ba::DpvoBaProblem`].
pub fn active_edge_triples(graph: &DpvoPatchGraph) -> Vec<DpvoEdge> {
    graph
        .edges()
        .iter()
        .map(|e| DpvoEdge {
            i: e.i,
            j: e.j,
            k: e.k,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    fn intr() -> DpvoIntrinsics {
        DpvoIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 32.0,
            cy: 24.0,
        }
    }

    fn patches_for_frame(m: usize, seed: f64) -> Vec<DpvoPatch> {
        (0..m)
            .map(|i| DpvoPatch {
                x: 10.0 + i as f64 + seed,
                y: 10.0 + seed,
                inverse_depth: 0.5,
            })
            .collect()
    }

    fn small_config() -> DpvoVoConfig {
        DpvoVoConfig {
            buffer_size: 4096,
            patches_per_frame: 4,
            removal_window: 6,
            optimization_window: 4,
            patch_lifetime: 3,
            keyframe_index: 2,
            keyframe_thresh: 12.5,
            motion_damping: 0.5,
        }
    }

    /// Push `count` frames with a small constant-translation motion,
    /// appending forward/backward edges after every commit (mirroring
    /// `dpvo.py`'s own call order), returning the graph.
    fn push_frames(count: usize) -> DpvoPatchGraph {
        let config = small_config();
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        for i in 0..count {
            let pose = graph.begin_frame(i as f64 * 0.05);
            let idx = graph
                .commit_frame(pose, intr(), patches_for_frame(m, i as f64))
                .expect("buffer has room");
            assert_eq!(idx, i);
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
        }
        graph
    }

    #[test]
    fn edges_forw_targets_only_the_newest_frame() {
        let graph = push_frames(3);
        // patch_lifetime=3, n=3 => forw covers patch ids [0, 2*4)=[0,8),
        // all targeting frame 2.
        let forw = graph.edges_forw();
        assert!(forw.iter().all(|&(_, j)| j == 2));
        assert_eq!(forw.len(), 8);
    }

    #[test]
    fn edges_back_covers_the_patch_lifetime_window_including_self() {
        let graph = push_frames(3);
        // patch_lifetime=3, n=3 => back covers patch ids of frame 2 (ids
        // [8,12)) targeting frames [max(3-3,0), 3) = [0,3), i.e. 3 targets
        // per patch, 4 patches => 12 edges, and includes target==owner (2).
        let back = graph.edges_back();
        assert_eq!(back.len(), 12);
        assert!(back.iter().any(|&(k, j)| graph.owner_frame(k) == j));
    }

    #[test]
    fn edge_owner_frame_survives_a_keyframe_fold() {
        let mut graph = push_frames(6);
        // Sanity: before folding, every edge's stored `i` matches
        // owner_frame(k).
        for edge in graph.edges() {
            assert_eq!(edge.i, graph.owner_frame(edge.k));
        }
        graph.set_initialized(true);
        // Force a fold regardless of motion magnitude, exercising the
        // compaction path directly.
        let removed = graph.fold_frame(2);
        assert_eq!(removed, 2);
        assert_eq!(graph.n_frames(), 5);
        for edge in graph.edges() {
            assert_eq!(
                edge.i,
                graph.owner_frame(edge.k),
                "owner_frame(k) invariant broken after fold: edge={edge:?}"
            );
            assert!(
                edge.i < 5 && edge.j < 5,
                "dangling frame reference after fold: edge={edge:?}"
            );
        }
    }

    #[test]
    fn keyframe_removal_window_drops_stale_active_edges() {
        let mut graph = push_frames(8);
        graph.set_initialized(true);
        graph.keyframe();
        let threshold = graph
            .n_frames()
            .saturating_sub(graph.config().removal_window);
        for edge in graph.edges() {
            assert!(
                graph.owner_frame(edge.k) >= threshold,
                "a stale edge survived keyframe()'s cleanup"
            );
        }
    }

    /// Milestone M6: a manually-injected loop edge whose owner frame is
    /// already stale (well below the removal-window threshold) must survive
    /// [`DpvoPatchGraph::keyframe_with_loop_protection`] as long as its
    /// target frame is still within `optimization_window` of `n`, but must be
    /// dropped by the *unprotected* [`DpvoPatchGraph::keyframe`] — confirming
    /// the exemption is real, not a no-op, and that `keyframe()`'s own
    /// behavior (M4's) is unchanged.
    #[test]
    fn keyframe_with_loop_protection_keeps_a_fresh_loop_edge_the_plain_keyframe_would_drop() {
        // `MIN_LOOP_GAP` is a hardcoded 30 inside `keyframe_inner` (matching
        // upstream's own hardcoded literal, not a config field), so a graph
        // exercising the *real* exemption path needs `edge.j - edge.i > 30`.
        // Build one directly at the patch-graph level rather than via
        // `push_frames`' small test config (`patch_lifetime=3`), which would
        // require an impractically long synthetic trajectory: append the
        // loop edge by hand onto an already-small graph and drive `n` up
        // with cheap frame commits (no forward/backward edges needed for
        // this test, only the injected loop edge and the removal-window
        // arithmetic).
        let config = small_config();
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        for i in 0..40 {
            let pose = graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(pose, intr(), patches_for_frame(m, i as f64))
                .unwrap();
        }
        graph.set_initialized(true);
        // Inject a single loop edge: patch 0 (owner frame 0) -> frame 39.
        // Gap = 39 - 0 = 39 > 30 (the hardcoded MIN_LOOP_GAP).
        graph.append_edges(&[(0, 39)], 4);
        assert_eq!(graph.edges().len(), 1);

        let mut protected = graph.clone();
        // Plain keyframe(): removal_window=6, n=40 => threshold=34; owner
        // frame 0 is far below threshold and there is no exemption => the
        // loop edge must be dropped.
        graph.keyframe();
        assert!(
            graph.edges().is_empty(),
            "plain keyframe() should drop a stale, unprotected loop edge"
        );

        // Protected keyframe: optimization_window=4 (small_config's own
        // value) => target frame must satisfy `39 + 4 > 40`, i.e. `43 > 40`
        // — true, so the edge survives.
        protected.keyframe_with_loop_protection(config.optimization_window);
        assert_eq!(
            protected.edges().len(),
            1,
            "keyframe_with_loop_protection should keep a loop edge whose target is still within optimization_window"
        );
    }

    /// Milestone M6: once `n` grows far enough that the loop edge's *target*
    /// frame also falls outside `optimization_window`, the exemption stops
    /// applying and a later `keyframe_with_loop_protection` call drops it —
    /// confirming the protection is temporary (bounded window growth, not a
    /// permanent retention mechanism), matching upstream's own
    /// `jj > (n - OPTIMIZATION_WINDOW)` condition.
    #[test]
    fn keyframe_with_loop_protection_eventually_drops_a_loop_edge_once_its_target_ages_out() {
        let config = small_config();
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        for i in 0..40 {
            let pose = graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(pose, intr(), patches_for_frame(m, i as f64))
                .unwrap();
        }
        graph.set_initialized(true);
        graph.append_edges(&[(0, 39)], 4);
        // Advance several more frames without touching the loop edge, so its
        // target (39) eventually falls outside `n - optimization_window`.
        for i in 40..50 {
            let pose = graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(pose, intr(), patches_for_frame(m, i as f64))
                .unwrap();
        }
        // n=50, optimization_window=4 => target must satisfy 39+4>50 i.e.
        // 43>50 — false, so the exemption no longer applies.
        graph.keyframe_with_loop_protection(config.optimization_window);
        assert!(
            graph.edges().is_empty(),
            "a loop edge must eventually be dropped once its target frame ages past optimization_window"
        );
    }

    #[test]
    fn keyframe_no_op_before_enough_frames_exist() {
        let mut graph = push_frames(2);
        // keyframe_index=2 => needs n > 3 to even attempt a fold.
        assert_eq!(graph.keyframe(), None);
        assert_eq!(graph.n_frames(), 2);
    }

    #[test]
    fn zero_motion_below_threshold_folds_the_keyframe_candidate() {
        // Every frame shares the exact same pose/patch state (zero
        // parallax) => motionmag() is 0 for every pair => well below
        // KEYFRAME_THRESH => keyframe() must fold frame `n - keyframe_index`.
        let config = small_config();
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        for i in 0..5 {
            graph.begin_frame(i as f64 * 0.05);
            let idx = graph
                .commit_frame(SE3::identity(), intr(), patches_for_frame(m, 0.0))
                .expect("buffer has room");
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
            let _ = idx;
        }
        graph.set_initialized(true);
        let n_before = graph.n_frames();
        let removed = graph.keyframe();
        assert_eq!(removed, Some(n_before - config.keyframe_index));
        assert_eq!(graph.n_frames(), n_before - 1);
    }

    /// Milestone M9: `retained_poses` must capture the EXACT pose (keyed by
    /// the correct `arrival_index`) of whichever frame `keyframe()` folds
    /// away — distinct per-frame poses here (unlike the zero-motion fixture
    /// above, which uses identity for every frame) so a wrong key or a
    /// stale/duplicated value would be caught, and repeated across TWO folds
    /// so a later fold cannot accidentally clobber an earlier entry.
    #[test]
    fn fold_frame_retains_the_folded_frames_final_pose_across_multiple_folds() {
        let config = small_config();
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        // Tiny, DISTINCT per-frame translations (near-zero motion so
        // `keyframe()` keeps folding, matching `zero_motion_...`'s own
        // near-identity-pose recipe, but distinguishable per frame).
        for i in 0..12 {
            let pose = SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(i as f64 * 0.001, 0.0, 0.0),
            );
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(pose, intr(), patches_for_frame(m, 0.0))
                .expect("buffer has room");
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
        }
        graph.set_initialized(true);
        assert!(graph.retained_poses().is_empty(), "nothing folded yet");

        // First fold.
        let folded_local_1 = graph.n_frames() - config.keyframe_index;
        let arrival_1 = graph.frames()[folded_local_1].arrival_index;
        let pose_1 = graph.frames()[folded_local_1].pose.clone();
        let removed_1 = graph.keyframe().expect("near-zero motion should fold");
        assert_eq!(removed_1, folded_local_1);
        assert_eq!(graph.retained_poses().len(), 1);
        assert_eq!(
            graph.retained_poses()[&arrival_1].translation,
            pose_1.translation
        );

        // Second fold: must ADD a new entry, not overwrite/lose the first.
        let folded_local_2 = graph.n_frames() - config.keyframe_index;
        let arrival_2 = graph.frames()[folded_local_2].arrival_index;
        let pose_2 = graph.frames()[folded_local_2].pose.clone();
        assert_ne!(
            arrival_1, arrival_2,
            "the two folds must retire distinct frames"
        );
        let removed_2 = graph
            .keyframe()
            .expect("near-zero motion should fold again");
        assert_eq!(removed_2, folded_local_2);
        assert_eq!(graph.retained_poses().len(), 2);
        assert_eq!(
            graph.retained_poses()[&arrival_1].translation,
            pose_1.translation,
            "first entry must survive"
        );
        assert_eq!(
            graph.retained_poses()[&arrival_2].translation,
            pose_2.translation
        );
    }

    #[test]
    fn large_motion_above_threshold_keeps_every_frame() {
        // Frames translate by 5m along x each step against a tight-FOV
        // patch depth of 0.5 (2m) => huge reprojection flow => motionmag()
        // is far above KEYFRAME_THRESH => keyframe() must not fold anything.
        let config = small_config();
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        for i in 0..5 {
            let pose = SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(5.0 * i as f64, 0.0, 0.0),
            );
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(pose, intr(), patches_for_frame(m, 0.0))
                .expect("buffer has room");
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
        }
        graph.set_initialized(true);
        let n_before = graph.n_frames();
        let removed = graph.keyframe();
        assert_eq!(removed, None);
        assert_eq!(graph.n_frames(), n_before);
    }

    #[test]
    fn motion_model_holds_last_pose_with_fewer_than_two_frames() {
        let mut graph = DpvoPatchGraph::new(small_config());
        assert_eq!(graph.begin_frame(0.0), SE3::identity());
        graph
            .commit_frame(
                SE3::new(UnitQuaternion::identity(), Vector3::new(1.0, 2.0, 3.0)),
                intr(),
                patches_for_frame(4, 0.0),
            )
            .unwrap();
        let predicted = graph.begin_frame(0.05);
        assert_eq!(predicted.translation, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn motion_model_extrapolates_constant_velocity_translation() {
        // Two frames moving +1.0 in x per 0.05s step at uniform cadence
        // (fac == 1.0) with damping == 1.0 should extrapolate to +1.0 more.
        let mut config = small_config();
        config.motion_damping = 1.0;
        let mut graph = DpvoPatchGraph::new(config);
        graph.begin_frame(0.0);
        graph
            .commit_frame(SE3::identity(), intr(), patches_for_frame(4, 0.0))
            .unwrap();
        graph.begin_frame(0.05);
        graph
            .commit_frame(
                SE3::new(UnitQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0)),
                intr(),
                patches_for_frame(4, 0.0),
            )
            .unwrap();
        let predicted = graph.begin_frame(0.10);
        assert!(
            (predicted.translation - Vector3::new(2.0, 0.0, 0.0)).norm() < 1e-9,
            "expected constant-velocity extrapolation to (2,0,0), got {:?}",
            predicted.translation
        );
    }

    #[test]
    fn motion_model_uneven_cadence_scales_extrapolation_by_fac() {
        // First interval 0.05s (a-b), second interval 0.10s (b-c) => fac =
        // (c-b)/(b-a) = 0.10/0.05 = 2.0, so with damping=1.0 the
        // extrapolated step should be twice the observed (P1*P2^-1) step.
        let mut config = small_config();
        config.motion_damping = 1.0;
        let mut graph = DpvoPatchGraph::new(config);
        graph.begin_frame(0.0);
        graph
            .commit_frame(SE3::identity(), intr(), patches_for_frame(4, 0.0))
            .unwrap();
        graph.begin_frame(0.05);
        graph
            .commit_frame(
                SE3::new(UnitQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0)),
                intr(),
                patches_for_frame(4, 0.0),
            )
            .unwrap();
        let predicted = graph.begin_frame(0.15);
        assert!(
            (predicted.translation - Vector3::new(3.0, 0.0, 0.0)).norm() < 1e-9,
            "expected fac-scaled extrapolation to (3,0,0), got {:?}",
            predicted.translation
        );
    }

    #[test]
    fn reject_pending_frame_records_identity_delta_from_predecessor() {
        let mut graph = DpvoPatchGraph::new(small_config());
        graph.begin_frame(0.0);
        graph
            .commit_frame(SE3::identity(), intr(), patches_for_frame(4, 0.0))
            .unwrap();
        graph.begin_frame(0.05); // counter becomes 2, rejected
        graph.reject_pending_frame();
        // delta[1] = (0, Identity); the recursive lookup needs a live-pose
        // stub for arrival_index 0 (frame 0, still live) to terminate on —
        // in a real setting `crate::dpvo_vo` supplies this from its own
        // frame index, so the test stubs it directly here.
        let live_pose_of = |arrival_index: usize| (arrival_index == 0).then(SE3::identity);
        let reconstructed = graph.reconstruct_pose(1, &live_pose_of);
        assert_eq!(reconstructed, Some(SE3::identity()));
    }

    /// Milestone M14: this is the structural property the low-parallax
    /// ("hover") freeze in `crate::dpvo_vo::DpvoOdometry::low_parallax_gate`
    /// relies on — repeatedly calling `begin_frame` + `reject_pending_frame`
    /// (with NO `commit_frame`/`keyframe`/`keyframe_with_loop_protection`
    /// call interleaved, exactly how a suppressed low-parallax candidate is
    /// handled) leaves `frames()`/`patches()`/`edges()` byte-for-byte
    /// untouched no matter how many "hover" frames are suppressed in a row:
    /// `n_frames()` never advances (nothing is ever admitted via
    /// `commit_frame`), so the removal-window aging check inside
    /// `keyframe_inner` — which only ever runs from a
    /// `keyframe()`/`keyframe_with_loop_protection()` call, never from
    /// `reject_pending_frame` — never gets a chance to purge a pre-hover
    /// patch's active edges either. This is exactly "freezing patch
    /// aging/lifetime and window advancement" (see `crate::dpvo_vo`'s
    /// module doc, "Low-parallax hover freeze"): it falls out of simply not
    /// calling `commit_frame`, with no new suppression flag needed anywhere
    /// in this module's own bookkeeping.
    #[test]
    fn suppressing_frames_via_reject_pending_frame_freezes_frames_patches_and_edges() {
        let mut graph = DpvoPatchGraph::new(small_config());
        let m = graph.config().patches_per_frame;
        // A handful of "pre-hover" frames with real edges, standing in for
        // well-constrained pre-hover patches.
        for i in 0..4 {
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(SE3::identity(), intr(), patches_for_frame(m, i as f64))
                .unwrap();
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
        }
        graph.set_initialized(true);

        let frames_before = graph.frames().to_vec();
        let patches_before = graph.patches().to_vec();
        let edges_before = graph.edges().to_vec();
        let n_before = graph.n_frames();

        // Simulate 50 consecutive suppressed "hover" frames — the
        // low-parallax gate's own call pattern (`begin_frame` then
        // `reject_pending_frame`, never `commit_frame`), spanning far more
        // than this fixture's own `removal_window`/`patch_lifetime` so any
        // aging that DID leak through would show up as a real difference.
        for i in 0..50 {
            graph.begin_frame(0.2 + i as f64 * 0.05);
            graph.reject_pending_frame();
        }

        assert_eq!(
            graph.n_frames(),
            n_before,
            "n_frames() must not advance while every candidate is rejected"
        );
        assert_eq!(
            graph.frames(),
            frames_before.as_slice(),
            "live frame poses/intrinsics must be untouched"
        );
        assert_eq!(
            graph.patches(),
            patches_before.as_slice(),
            "patches must be untouched (none admitted)"
        );
        assert_eq!(
            graph.edges(),
            edges_before.as_slice(),
            "active edges must be untouched (no aging occurred)"
        );
    }

    /// Milestone M8: an edge dropped by the ordinary removal-window rule
    /// (NOT a fold) that HAS been touched by an `update()` call
    /// (`target_weight.is_some()`) is archived, frozen, into
    /// [`DpvoPatchGraph::inactive_edges`] when retention is enabled — while
    /// a sibling edge dropped in the SAME call that was never touched
    /// (`target_weight` still `None`) is correctly skipped, not archived
    /// with a meaningless placeholder measurement. Uses the same
    /// proven-not-to-fold big-motion trajectory shape as
    /// `large_motion_above_threshold_keeps_every_frame` (unmodified
    /// `patch_lifetime`/`keyframe_index`/`keyframe_thresh`, only
    /// `removal_window` shrunk so a stale-edge drop happens within a small
    /// test trajectory) so the removal-window drop, not a keyframe fold, is
    /// what triggers archiving.
    #[test]
    fn keyframe_archives_touched_stale_edges_and_skips_untouched_ones() {
        let mut config = small_config();
        config.removal_window = 4;
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        graph.enable_inactive_edge_retention(100);
        for i in 0..8 {
            let pose = SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(5.0 * i as f64, 0.0, 0.0),
            );
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(pose, intr(), patches_for_frame(m, 0.0))
                .unwrap();
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
        }
        graph.set_initialized(true);

        // Simulate `update()` having touched only frame 0's active edges
        // with a distinctive target/weight before the removal-window drop
        // below runs; frame 1/2/3's edges (also about to age out, since
        // threshold = 8 - 4 = 4) are deliberately left untouched.
        for edge in graph.edges_mut() {
            if edge.i == 0 {
                edge.target_weight = Some((Vector2::new(11.0, 22.0), Vector2::new(0.9, 0.8)));
            }
        }
        assert_eq!(graph.inactive_edge_stats(), (0, 0));

        let removed = graph.keyframe();
        assert_eq!(
            removed, None,
            "big motion must not have triggered a fold — only the removal-window drop"
        );
        assert!(
            graph.edges().iter().all(|e| e.i != 0),
            "frame 0's edges should have aged out of the active set"
        );

        let (retained, evicted) = graph.inactive_edge_stats();
        assert!(retained > 0, "expected at least one archived inactive edge");
        assert_eq!(evicted, 0, "cap (100) was never exceeded");
        for ie in graph.inactive_edges() {
            assert_eq!(
                ie.arrival_i, 0,
                "only frame 0's (touched) edges should have been archived"
            );
            assert_eq!(ie.target, Vector2::new(11.0, 22.0));
            assert_eq!(ie.weight, Vector2::new(0.9, 0.8));
        }
    }

    /// Milestone M10: the inactive-edge store is bounded — once more edges
    /// are archived than the configured cap, some are dropped (via the
    /// decimating policy [`DpvoPatchGraph::inactive_edges`]'s own doc
    /// describes, not M8's original plain FIFO) and
    /// [`DpvoPatchGraph::inactive_edge_stats`]'s eviction counter grows. This
    /// test only pins the CAP invariant (never more than `cap` retained,
    /// eviction happens when more are archived than that) — see
    /// `retention_preserves_temporal_coverage_across_the_full_history` below
    /// for the coverage property the M10 rewrite specifically buys over the
    /// original FIFO policy.
    #[test]
    fn inactive_edge_cap_bounds_the_retained_count() {
        let mut config = small_config();
        config.removal_window = 4;
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        graph.enable_inactive_edge_retention(1); // Cap of exactly 1.
        for i in 0..8 {
            let pose = SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(5.0 * i as f64, 0.0, 0.0),
            );
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(pose, intr(), patches_for_frame(m, 0.0))
                .unwrap();
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
        }
        graph.set_initialized(true);
        for edge in graph.edges_mut() {
            edge.target_weight = Some((Vector2::new(1.0, 1.0), Vector2::new(1.0, 1.0)));
        }

        graph.keyframe();

        let (retained, evicted) = graph.inactive_edge_stats();
        assert_eq!(
            retained, 1,
            "cap of 1 must bound the retained count regardless of how many were archived"
        );
        assert!(
            evicted > 0,
            "more than one stale touched edge existed, so eviction must have happened"
        );
    }

    /// Milestone M10 (`docs/dpvo_droid_port_plan.md`): the required
    /// retention-policy test — archives far more distinct-`arrival_i` edges
    /// than the cap, across a long span, and checks the retained set stays
    /// spread across (most of) that ENTIRE span rather than collapsing to
    /// the most recent `cap` arrivals, which is exactly what M8's original
    /// plain-FIFO policy would do (a moving window whose reach never
    /// exceeds `cap` entries back from "now" — the M8 results section's own
    /// diagnosed cause of a loop's old edges being long gone by the time a
    /// loop fired).
    #[test]
    fn retention_preserves_temporal_coverage_across_the_full_history() {
        let mut config = small_config();
        config.patches_per_frame = 1;
        config.removal_window = 2;
        config.patch_lifetime = 2;
        // `0.0` unconditionally disables the (unrelated) low-motion FOLD
        // mechanism regardless of `motionmag`'s value (`m / 2.0 < 0.0` is
        // never true — the same "never fold" recipe
        // `global_ba_tests::small_graph_config`/`m10_graph_config` use in
        // `dpvo_vo.rs`). NOT simply "use a big per-step translation": with
        // this test's own tight `patch_lifetime = 2`, the fold-candidate
        // check's specific `(i, j)` edge pair can legitimately not exist at
        // all, making `motionmag` return `0.0` regardless of how large the
        // actual motion is — an early version of this test relied on "big
        // motion never folds" (`large_motion_above_threshold_keeps_every_frame`'s
        // own recipe) and was WRONG here: frame `2` folded away at frame
        // index `3` anyway, starving the removal-window archiving path this
        // test is supposed to exercise (everything after that point left via
        // `fold_frame`'s own `store=False` drop instead of being archived).
        config.keyframe_thresh = 0.0;
        let mut graph = DpvoPatchGraph::new(config);
        graph.enable_inactive_edge_retention(20); // Cap far smaller than the total archived below.
        const N_FRAMES: usize = 150;
        for i in 0..N_FRAMES {
            let pose = SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(5.0 * i as f64, 0.0, 0.0),
            );
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(pose, intr(), patches_for_frame(1, 0.0))
                .unwrap();
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
            graph.set_initialized(true);
            // Touch every currently-active edge with a distinctive target
            // encoding its OWN arrival index, so a later inspection of
            // `inactive_edges()` can tell which arrival range survived.
            for edge in graph.edges_mut() {
                let arrival_i = i; // not yet folded, so live index == arrival index order here.
                edge.target_weight =
                    Some((Vector2::new(arrival_i as f64, 0.0), Vector2::new(1.0, 1.0)));
            }
            graph.keyframe();
        }

        let (retained, evicted) = graph.inactive_edge_stats();
        // The decimating policy's "keep every other" thinning does not
        // always land EXACTLY on the cap (a halving pass over `cap + 1`
        // entries yields `⌈(cap + 1) / 2⌉`, and the store then refills
        // toward the cap again at the CURRENT stride until it thins again)
        // — the invariant this policy actually guarantees is the UPPER
        // bound, never exceeding the cap, not landing on it exactly.
        assert!(
            retained <= 20,
            "cap must still bound the retained count, got {retained}"
        );
        assert!(retained > 0, "the cap must not have degenerated to zero");
        assert!(
            evicted > 0,
            "this trajectory must have archived far more than the cap"
        );

        let min_arrival_i = graph
            .inactive_edges()
            .iter()
            .map(|ie| ie.target.x as usize)
            .min()
            .unwrap();
        let max_arrival_i = graph
            .inactive_edges()
            .iter()
            .map(|ie| ie.target.x as usize)
            .max()
            .unwrap();
        // A plain FIFO cap of 20 could never retain anything with
        // arrival_i below roughly `N_FRAMES - 20`; the decimating policy
        // must retain something from at least the first THIRD of the run.
        assert!(
            min_arrival_i < N_FRAMES / 3,
            "expected coverage to reach well into the early part of the run, oldest retained arrival_i={min_arrival_i}"
        );
        assert!(
            max_arrival_i > 2 * N_FRAMES / 3,
            "expected coverage to also reach recent history, newest retained arrival_i={max_arrival_i}"
        );
    }

    /// Milestone M8: an archived inactive edge's frozen `target`/`weight`
    /// never changes afterward, even as the graph keeps advancing (no code
    /// path re-touches a retained [`InactiveEdge`] — only
    /// `crate::dpvo_vo::DpvoOdometry::run_global_ba` ever reads it).
    #[test]
    fn archived_inactive_edges_stay_frozen_as_the_graph_advances() {
        let mut config = small_config();
        config.removal_window = 4;
        let m = config.patches_per_frame;
        let mut graph = DpvoPatchGraph::new(config);
        graph.enable_inactive_edge_retention(1000);
        for i in 0..8 {
            let pose = SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(5.0 * i as f64, 0.0, 0.0),
            );
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(pose, intr(), patches_for_frame(m, 0.0))
                .unwrap();
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
        }
        graph.set_initialized(true);
        for edge in graph.edges_mut() {
            edge.target_weight = Some((Vector2::new(3.0, 4.0), Vector2::new(0.5, 0.5)));
        }
        graph.keyframe();
        let snapshot: Vec<InactiveEdge> = graph.inactive_edges().to_vec();
        assert!(!snapshot.is_empty());

        // Advance the graph further (more frames, more folds/removals).
        for i in 8..14 {
            let pose = SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(5.0 * i as f64, 0.0, 0.0),
            );
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(pose, intr(), patches_for_frame(m, 0.0))
                .unwrap();
            let forw = graph.edges_forw();
            let back = graph.edges_back();
            graph.append_edges(&forw, 4);
            graph.append_edges(&back, 4);
            graph.keyframe();
        }

        for ie in &snapshot {
            assert!(
                graph.inactive_edges().contains(ie),
                "an already-archived inactive edge's frozen measurement must be unchanged by later graph activity: {ie:?}"
            );
        }
    }

    #[test]
    fn buffer_full_is_reported_as_an_error_not_a_panic() {
        let mut config = small_config();
        config.buffer_size = 2;
        let mut graph = DpvoPatchGraph::new(config);
        graph.begin_frame(0.0);
        graph
            .commit_frame(SE3::identity(), intr(), patches_for_frame(4, 0.0))
            .unwrap();
        graph.begin_frame(0.05);
        let err = graph.commit_frame(SE3::identity(), intr(), patches_for_frame(4, 0.0));
        assert_eq!(err, Err(DpvoGraphError::BufferFull { buffer_size: 2 }));
    }
}
