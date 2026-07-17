//! The full DPVO visual-odometry per-frame loop — Milestone M4 of
//! `docs/dpvo_droid_port_plan.md`, the ONNX-inference-dependent half (see
//! `crate::dpvo_patch_graph`'s module doc for the unconditional half this
//! module drives).
//!
//! Ground truth: `E:/tools/DPVO/dpvo/dpvo.py`'s `DPVO.__call__`/`update`/
//! `motion_probe`, `dpvo/net.py`'s `Patchifier.forward` (centroid sampling +
//! `altcorr.patchify` calls), and `dpvo/dpvo.py`'s `corr`/`reproject`
//! methods (the 2-pyramid-level correlation assembly M2 left as its own
//! "biggest blocker" — see `docs/dpvo_droid_port_plan.md`'s M2 results,
//! "corr_cpu's single-shared-target-frame scope"). Every ONNX/native-math
//! primitive this module drives (`fnet`/`inet` sessions, `patchify_cpu`,
//! `corr_cpu`, the GRU update cell, `SoftAgg`) was ported and parity-tested
//! in M1/M2 (`crates/vision/src/dpvo/`); the BA solver in M3
//! (`crate::dpvo_patch_ba`); this module is the first thing that calls all
//! of them together, end to end, per incoming frame.
//!
//! # A genuine finding, not an assumption: EuRoC eval config differs from
//! `config.py`'s bare defaults
//!
//! `E:/tools/DPVO/evaluate_euroc.py`/`config/default.yaml` — the config that
//! actually produced DPVO's published EuRoC number — overrides
//! `PATCHES_PER_FRAME: 96` (not `config.py`'s bare `80`), `REMOVAL_WINDOW:
//! 22`, `OPTIMIZATION_WINDOW: 10`, `PATCH_LIFETIME: 13`, `KEYFRAME_THRESH:
//! 15.0`. [`crate::dpvo_patch_graph::DpvoVoConfig::default`] still matches
//! `config.py`'s bare defaults (the more "canonical" reference point per the
//! plan doc's own architecture table); this crate's EuRoC example
//! (`examples/euroc_dpvo_vo_demo.rs`) instead constructs the `default.yaml`
//! values explicitly and documents why. Also confirmed from
//! `dpvo/stream.py::image_stream` (the actual EuRoC image loader): **no
//! resolution downscaling happens** (the `if 0:` half-res branch is dead
//! code) — the task brief's own "likely half-res" guess was a hypothesis to
//! check, not a fact, and checking it against the primary source shows it
//! is false; DPVO's published EuRoC number runs at full 752×480 resolution,
//! only *temporally* subsampling frames (`--stride 2`, i.e. every other
//! frame, `~10`Hz instead of `~20`Hz).
//!
//! # Windowing the BA problem: a derivation the plan doc's M3 blockers list
//! left open
//!
//! `dpvo_patch_ba::dpvo_ba` derives its frame count from `poses.len()` (M3's
//! own documented simplification). Passing this module's *entire* live
//! trajectory every `update()` call would make each BA call `O(total
//! frames)` — quadratic over a long sequence. This module instead windows
//! every BA call to `[frame_lo, n)` where `frame_lo = n.saturating_sub(
//! REMOVAL_WINDOW + PATCH_LIFETIME)`. Derivation: `keyframe()`'s own cleanup
//! guarantees every surviving active edge's *owner* frame `i ≥ n -
//! REMOVAL_WINDOW` at the start of any `update()` call (see
//! `crate::dpvo_patch_graph`'s module doc). But an edge's *target* frame `j`
//! is fixed at creation time to something as old as `i - PATCH_LIFETIME` (an
//! `edges_back` edge can target any frame in the trailing `PATCH_LIFETIME`
//! window, not just its own owner frame) and never moves afterward — so in
//! the worst case, right before such an edge is finally pruned (`i` just
//! above the removal threshold), `j` can be as old as
//! `n - REMOVAL_WINDOW - PATCH_LIFETIME + 1`. `frame_lo` above is a safe (if
//! slightly loose) lower bound covering that worst case, checked with a
//! `debug_assert` at
//! the point [`DpvoOdometry::update_step`] builds a windowed
//! [`crate::dpvo_patch_ba::DpvoBaProblem`]. The *free* pose count inside the
//! window (`n2` in `dpvo_patch_ba`'s own terms) still equals exactly
//! `OPTIMIZATION_WINDOW` in steady state, since `fixedp` is computed the
//! same way upstream does (`max(n - OPTIMIZATION_WINDOW, 1)`) and then
//! re-based onto the window's own local indexing — the window is just wide
//! enough to hold every pose an active edge might reference, not wide
//! enough to change which poses are actually free.
//!
//! **Milestone M6 generalizes this from an assertion to a derivation.** A
//! proximity (loop-closure) edge's source frame can be far older than
//! `REMOVAL_WINDOW + PATCH_LIFETIME` bounds (that derivation assumed only
//! ordinary `edges_forw`/`edges_back` edges exist). `update_step` now widens
//! `frame_lo` to `min(the formula above, the oldest frame any currently
//! active edge references)` before building the window — a strict
//! generalization that reduces to the exact M4 formula whenever no edge is
//! older than it (i.e. whenever loop closure is disabled, or simply hasn't
//! found anything yet), so this changes nothing for a non-loop-closure run.
//! Growing `frame_lo` only ever adds *fixed* poses to the window (the free
//! pose count stays pinned at `OPTIMIZATION_WINDOW` regardless — see
//! `crate::dpvo_loop_closure`'s module doc, "What 'global BA' becomes on
//! this CPU port", for why), so this is safe by the same `fixedp`/`t0`
//! reasoning `dpvo_patch_ba.rs`'s own M3 convention-mapping notes already
//! established, not a new risk this milestone introduces.
//!
//! # Loop closure (Milestone M6, `docs/dpvo_droid_port_plan.md`)
//!
//! [`DpvoOdometryConfig::loop_closure`] is `None` by default — every prior
//! milestone's call site keeps compiling and behaving byte-for-byte as
//! before. When `Some`, [`DpvoOdometry::try_loop_closure`] (see its own doc)
//! runs `crate::dpvo_loop_closure::find_loop_edges` at the point upstream's
//! `__call__` does (`dpvo.py:449-455`, right before `update()`/`keyframe()`),
//! appends any accepted proximity edges via the same
//! `DpvoPatchGraph::append_edges` ordinary temporal edges already use, and
//! dispatches `keyframe()`'s cleanup through
//! `DpvoPatchGraph::keyframe_with_loop_protection` instead of the plain
//! `keyframe()` so a freshly-added loop edge is not immediately pruned by the
//! removal-window rule. See `crate::dpvo_loop_closure`'s own module doc for
//! the full port (candidate generation, edge-budget/NMS selection, and why
//! this deliberately reuses DPVO's own patch-graph edge system rather than
//! `crate::sparse_factor_graph`) and this module's own
//! [`DpvoOdometry::update_step`] doc for how the windowed BA problem widens
//! to cover a loop edge's (potentially much older) source frame with **no
//! new BA entry point** — the CPU-bounded stand-in for upstream's own
//! `__run_global_BA`.
//!
//! # What this module does not implement (see `crate::dpvo_patch_graph`'s
//! module doc for the graph-level list)
//!
//! The classical/long-term ("`CLASSIC_LOOP_CLOSURE`") backend is out of
//! scope — `crate::dpvo_loop_closure`'s module doc explains why this
//! codebase's existing `online_slam.rs`/`map_atlas.rs` appearance-loop
//! pipeline already exceeds it and needs no replacement.
//!
//! # IMU coupling (Milestone M5, then M5b — `docs/dpvo_droid_port_plan.md`)
//!
//! [`DpvoOdometryConfig::imu`] is `None` by default — every M4 call site
//! keeps compiling and running byte-for-byte as before. When `Some`, three
//! pieces layer on top of the M4 loop above without changing it:
//!
//! 1. [`DpvoOdometry::push_imu`] buffers raw samples; every
//!    [`DpvoOdometry::process_frame`] call folds whatever arrived since the
//!    previous frame into an [`crate::imu_preintegration::ImuPreintegrator`]
//!    and banks the resulting delta, keyed by the two frames' stable
//!    `arrival_index` (`integrate_imu_for_new_frame`).
//! 2. Once enough evidence has accumulated, [`DpvoOdometry::try_imu_bootstrap`]
//!    runs `vi_motion_initializer.rs`'s own `estimate_gyro_bias` (rotation-only,
//!    genuinely scale-invariant — reused as-is, but now gated harder, see
//!    below) followed by **`crate::dpvo_vi_ba::estimate_mono_vi_alignment`**
//!    — Milestone M5b's monocular-aware replacement for reusing
//!    `estimate_gravity_and_velocities` against still-non-metric poses, see
//!    that function's own module-doc section for the full formulation —
//!    against pose SNAPSHOTS decoupled from the live BA window (see that
//!    method's own doc for why: the live window churns via
//!    `DpvoPatchGraph::keyframe`'s motion-magnitude folding faster than a
//!    handful-of-keyframes window could otherwise fill).
//! 3. Once bootstrapped, `update_step` couples banked deltas into the
//!    **same** windowed Gauss-Newton solve via `crate::dpvo_vi_ba::dpvo_vi_ba`
//!    instead of the plain visual-only `crate::dpvo_patch_ba::dpvo_ba` — see
//!    that module's own doc for the math (left-perturbation IMU Jacobian
//!    derivation, sign convention, scale handling) — and monitors the
//!    coupled solve's own IMU-factor NIS for a **rollback** (Milestone M5b,
//!    see [`DpvoOdometry::rollback_imu_bootstrap`]'s doc) back to
//!    visual-only if it blows past a configured bound for too many
//!    consecutive frames.
//!
//! ## Milestone M5's honest negative, and what M5b changes
//!
//! M5's own real-EuRoC-run finding (`docs/dpvo_droid_port_plan.md`'s "M5
//! results"): `estimate_gyro_bias`/`estimate_gravity_and_velocities` were
//! designed for, and everywhere else in this codebase are run against,
//! already-metric visual poses — a precondition DPVO's own non-metric
//! reconstruction does not satisfy at bootstrap time. M5's design also
//! accepted whatever the bootstrap chain returned unconditionally (gated
//! only on gravity-norm deviation) and then froze it forever (the
//! staged-bias philosophy), so a single bad-quality bootstrap poisoned the
//! rest of the run — measured as a collapsed similarity scale (`0.006`) and
//! a blown-up rigid ATE (`24.47 m`) against a `1.0` target.
//!
//! Milestone M5b (`docs/dpvo_droid_port_plan.md`'s "M5b results") replaces
//! the gravity/velocity half of the bootstrap with an explicit-scale
//! monocular alignment (`estimate_mono_vi_alignment`, described above), adds
//! real acceptance gates to the gyro-bias half instead of accepting it
//! unconditionally
//! ([`DpvoImuConfig::max_gyro_bias_magnitude_rad_s`]/
//! [`DpvoImuConfig::gyro_bias_max_rms_after`]/
//! [`DpvoImuConfig::gyro_bias_max_rms_fraction`] — see
//! [`DpvoOdometry::try_imu_bootstrap`]'s own doc), applies the recovered
//! scale to the live window before enabling coupling (translations and
//! patch inverse depths — see `crate::dpvo_vi_ba`'s module doc, "Applying
//! the recovered scale"), and — because even a gated bootstrap can still be
//! wrong — adds a **rollback**: if the post-bootstrap coupled solve's own
//! IMU-factor NIS stays pathological for
//! [`DpvoImuConfig::rollback_consecutive_frames`] frames in a row, the
//! odometry un-bootstraps back to visual-only and allows a later
//! re-attempt, rather than staying poisoned for the rest of the run by
//! construction.
//!
//! **Honest outcome** (see `docs/dpvo_droid_port_plan.md`'s "M5b results"
//! for the full numbers): at this module's SHIPPED conservative default
//! (`max_gyro_bias_magnitude_rad_s = 0.05`), the bootstrap never fires on
//! MH_01's first 400 frames — a safe, byte-identical-to-visual-only
//! outcome, confirmed by running the full sequence. A real-data experiment
//! that loosened that one gate (reasoning the rollback net made it safe
//! to) DID let the bootstrap fire, and the rollback monitor correctly
//! caught and undid 3 of the resulting 4 bootstrap events — but the 4th's
//! recovered scale (`18.66`) passed every other observability gate this
//! module has (gravity-norm, scale-range, conditioning) while still being
//! numerically wrong, corrupting the rest of that run (rigid ATE `55.49
//! m`). One-shot bootstrap-then-trust is therefore not yet safe on real
//! DPVO windows even with a working rollback net; the conservative default
//! is what ships until a stronger acceptance check exists (see the plan
//! doc's own "forward path" note).
#![cfg(feature = "onnx-inference")]

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::time::Instant;

use nalgebra::{Vector2, Vector3};
use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Frame, Keyframe, VisualMap};
use visloc_vision::dpvo::correlation::{corr_cpu_prebuilt_target, ChannelLastImage};
use visloc_vision::dpvo::npz::{NpzArchive, NpzError};
use visloc_vision::dpvo::onnx_session::{DpvoOnnxError, DpvoOnnxSession};
use visloc_vision::dpvo::patchify::patchify_cpu;
use visloc_vision::dpvo::softagg::{SoftAgg, SoftAggError};
use visloc_vision::dpvo::{CORR_DIM, CORR_RADIUS, DIM, FNET_DIM, PATCH, RES};
use visloc_vision::features::superpoint_onnx::OnnxBackend;

use crate::dpvo_patch_ba::{
    dpvo_ba, reproject_patch_grid, DpvoBaConfig, DpvoBaError, DpvoBaProblem, DpvoEdge,
    DpvoIntrinsics, DpvoPatch,
};
use crate::dpvo_loop_closure::{
    expand_frame_pairs_to_patch_edges, find_loop_edges, DpvoLoopClosureConfig, UPSTREAM_MIN_LOOP_GAP,
};
use crate::dpvo_patch_graph::{DpvoGraphError, DpvoPatchGraph, DpvoVoConfig};
use crate::dpvo_vi_ba::{
    dpvo_vi_ba, estimate_mono_vi_alignment, imu_factor_nis, DpvoImuFactor, DpvoMonoViAlignmentGates,
    DpvoMonoViAlignmentRejection, DpvoViWindow,
};
use crate::dpvo_scale_coupling::{
    apply_gentle_scale_correction, blend_solutions, scale_measurement_from_alignment, AnnealingWeight,
    RecursiveGyroBiasEstimator, RecursiveScaleEstimator, ScaleCouplingConfig,
};
use crate::imu_preintegration::{
    ImuNoiseModel, ImuPreintegratedDelta, ImuPreintegrationFactor, ImuPreintegrator,
};
use crate::vi_motion_initializer::{estimate_gyro_bias, GyroBiasAlignment};

/// Cap on [`DpvoOdometry`]'s `imu_bootstrap_history` — see that field's doc.
const IMU_BOOTSTRAP_HISTORY_CAP: usize = 64;

/// Errors from [`DpvoOdometry`].
#[derive(Debug)]
pub enum DpvoOdometryError {
    Onnx(DpvoOnnxError),
    SoftAgg(SoftAggError),
    Npz(NpzError),
    Graph(DpvoGraphError),
    Ba(DpvoBaError),
    /// `image.dim()` did not match [`DpvoOdometryConfig::width`]/`height`.
    ImageShapeMismatch { expected: (usize, usize), actual: (usize, usize) },
}

impl std::fmt::Display for DpvoOdometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Onnx(e) => write!(f, "dpvo odometry: onnx error: {e}"),
            Self::SoftAgg(e) => write!(f, "dpvo odometry: softagg weight error: {e}"),
            Self::Npz(e) => write!(f, "dpvo odometry: npz error: {e}"),
            Self::Graph(e) => write!(f, "dpvo odometry: graph error: {e}"),
            Self::Ba(e) => write!(f, "dpvo odometry: bundle adjustment error: {e}"),
            Self::ImageShapeMismatch { expected, actual } => {
                write!(f, "dpvo odometry: expected image shape {expected:?}, got {actual:?}")
            }
        }
    }
}

impl std::error::Error for DpvoOdometryError {}

impl From<DpvoOnnxError> for DpvoOdometryError {
    fn from(value: DpvoOnnxError) -> Self {
        Self::Onnx(value)
    }
}
impl From<SoftAggError> for DpvoOdometryError {
    fn from(value: SoftAggError) -> Self {
        Self::SoftAgg(value)
    }
}
impl From<NpzError> for DpvoOdometryError {
    fn from(value: NpzError) -> Self {
        Self::Npz(value)
    }
}
impl From<DpvoGraphError> for DpvoOdometryError {
    fn from(value: DpvoGraphError) -> Self {
        Self::Graph(value)
    }
}
impl From<DpvoBaError> for DpvoOdometryError {
    fn from(value: DpvoBaError) -> Self {
        Self::Ba(value)
    }
}

/// Construction-time configuration for [`DpvoOdometry`].
///
/// `Clone`, not `Copy` — [`DpvoImuConfig::body_to_camera`] is an [`SE3`],
/// which is not `Copy` (`crates/core/src/geometry/se3.rs`); this is the one
/// change Milestone M5 (`docs/dpvo_droid_port_plan.md`) made to this struct's
/// derive list. Nothing in this module relied on `Copy` (every use is a
/// field read through `&self`, never an implicit bitwise copy), so this is
/// non-breaking.
#[derive(Debug, Clone)]
pub struct DpvoOdometryConfig {
    pub vo: DpvoVoConfig,
    /// Input image width/height in pixels (every frame passed to
    /// [`DpvoOdometry::process_frame`] must match exactly — this module
    /// does no resizing of its own; the caller undistorts/downscales
    /// upstream, matching `dpvo/stream.py::image_stream`'s own caller-side
    /// preprocessing).
    pub width: usize,
    pub height: usize,
    /// Pinhole intrinsics at the *full* `(width, height)` resolution above
    /// (this module divides by [`RES`] internally before storing them in
    /// the patch graph — `dpvo.py:401`'s own `intrinsics / self.RES`).
    pub intrinsics: DpvoIntrinsics,
    /// `ba.py`'s `lmbda` (default `1e-4`, every real call site).
    pub ba_lmbda: f64,
    /// `ba.py`'s `ep` (default `100.0`).
    pub ba_ep: f64,
    /// `motion_probe`'s hardcoded gate (`dpvo.py:442`: `< 2.0`), exposed for
    /// tests rather than hardcoded a second time.
    pub motion_probe_min_flow: f64,
    /// Seed for centroid sampling (`Patchifier.forward`'s
    /// `torch.randint`/`torch.rand` calls) — deterministic runs for a fixed
    /// seed, matching how this codebase already threads RNG seeds through
    /// other ONNX-adjacent demos.
    pub seed: u64,
    /// IMU coupling (Milestone M5, `docs/dpvo_droid_port_plan.md`). `None`
    /// (the default constructed by every M4 call site) preserves the
    /// exact visual-only behavior of M4/M4-perf; `Some` enables
    /// [`DpvoOdometry::push_imu`]/the bootstrap chain/`crate::dpvo_vi_ba`
    /// coupling described on [`DpvoOdometry`]'s own doc comment.
    pub imu: Option<DpvoImuConfig>,
    /// DPV-SLAM mid-term proximity loop closure (Milestone M6,
    /// `docs/dpvo_droid_port_plan.md`). `None` (every prior milestone's
    /// default) preserves M4/M4-perf/M5/M5b's exact visual-only-graph
    /// behavior — see [`DpvoOdometry`]'s own doc, "Loop closure", and
    /// `crate::dpvo_loop_closure`'s module doc for the full port.
    pub loop_closure: Option<DpvoLoopClosureConfig>,
}

/// IMU coupling configuration — Milestone M5. See [`DpvoOdometry`]'s module
/// doc for the bootstrap chain this feeds and `crate::dpvo_vi_ba`'s module
/// doc for the math it ultimately drives.
#[derive(Debug, Clone, PartialEq)]
pub struct DpvoImuConfig {
    /// EuRoC-style `T_BS` extrinsic, taken literally (maps a CAMERA-frame
    /// coordinate to its BODY-frame coordinate) — see `crate::dpvo_vi_ba`'s
    /// module doc, "jacobian convention conversion" section, for the exact
    /// convention this must satisfy.
    pub body_to_camera: SE3,
    /// Continuous-time IMU noise densities, used for both this module's own
    /// [`ImuPreintegrator`] calls and the resulting factors' whitening
    /// (`ImuPreintegrationFactor::covariance_sqrt_information`).
    pub noise: ImuNoiseModel,
    /// Expected local gravity magnitude (m/s², EuRoC/Earth-surface default
    /// `9.81`) fed to `crate::dpvo_vi_ba::estimate_mono_vi_alignment`
    /// (Milestone M5b; `estimate_gravity_and_velocities` used this same
    /// field pre-M5b).
    pub gravity_magnitude: f64,
    /// Bootstrap acceptance gate on
    /// `crate::dpvo_vi_ba::DpvoMonoViAlignment::raw_gravity_norm`'s relative
    /// deviation from `gravity_magnitude` — mirrors
    /// `MotionBasedViInitializerConfig::max_gravity_norm_deviation_ratio`'s
    /// own default (`0.3`) exactly, for the same reason: an
    /// insufficiently-excited window's *unconstrained* gravity-norm
    /// solve is the direct observability signal, before any
    /// magnitude-constrained refinement papers over it.
    pub gravity_norm_deviation_ratio: f64,
    /// Minimum number of banked IMU deltas before a bootstrap attempt is
    /// even tried (both `estimate_gyro_bias`/`estimate_mono_vi_alignment`
    /// already refuse below 2 internally; this is an additional, coarser
    /// gate so a bootstrap attempt is not retried every single frame from
    /// frame 2 onward during the initial burst). Default `10` (the
    /// gyro-bias estimator's own `MAX_ALIGNMENT_WINDOW` cap) — chosen
    /// empirically (see `docs/dpvo_droid_port_plan.md`'s "M5 results"): a
    /// smaller value (e.g. `3`) lets the bootstrap fire almost immediately
    /// after graph initialization, against a visual reconstruction that has
    /// had essentially no time to stabilize.
    pub min_bootstrap_factors: usize,
    /// Milestone M5b: gyro-bias bootstrap gate — reject a recovered
    /// `GyroBiasAlignment` whose magnitude exceeds this (rad/s).
    ///
    /// # Default `0.05` — kept conservative after a real-data A/B, not a
    /// guess (see `docs/dpvo_droid_port_plan.md`'s "M5b results" for the
    /// full numbers)
    ///
    /// EuRoC's own real gyro bias sits around `1e-3`–`1e-2` rad/s. A first
    /// MH_01 run at `0.05` found the bootstrap never fired in 400 frames
    /// (`estimate_gyro_bias` recovers a STABLE — not noisy —
    /// systematically-inflated bias around `0.09`–`0.51` rad/s throughout,
    /// because DPVO's own monocular rotation reconstruction carries a
    /// small systematic error this rotation-only fit partially absorbs
    /// into the bias term). Reasoning that the downstream rollback monitor
    /// ([`DpvoOdometry::rollback_imu_bootstrap`]) was now a real safety
    /// net, this bound was raised to `0.3` and the SAME run repeated: the
    /// bootstrap fired 4 times, the rollback monitor correctly caught and
    /// undid 3 of them, but the 4th's recovered scale (`18.66`, inside the
    /// nominally-plausible `[0.05, 20]` range) stuck for the rest of the
    /// run and drove the rigid ATE to `55.49 m` (similarity scale
    /// collapsed to `0.0014`) — i.e. a scale that passes every one of this
    /// module's own observability gates can still be numerically wrong,
    /// and the gates as designed do not catch that. This is an HONEST
    /// NEGATIVE result on the `0.3` experiment, not a bug: it shows
    /// one-shot bootstrap-then-trust is not yet safe on real DPVO windows,
    /// even with a working rollback net (3-for-4 is a real result, not a
    /// disqualifying one, but not "safe by default" either). The bound is
    /// therefore reverted to `0.05` for the SHIPPED default: this is the
    /// setting empirically confirmed (400/400 frames, both A/B runs) to
    /// never admit a bad bootstrap, falling back to `dpvo_ba`'s unmodified
    /// visual-only path with `tracked_fraction=1.0` and ATE identical to
    /// the M4-perf baseline — the safe, byte-reproducible default until a
    /// stronger acceptance check exists (see the plan doc's own "forward
    /// path" for what such a check would need to look like: a
    /// scale-consistency cross-check between the alignment and the BA's
    /// own solve, or continuous in-window scale refinement instead of a
    /// single admit-or-reject bootstrap event). A caller who has verified
    /// their own dataset's bootstrap behavior may still override this
    /// field explicitly.
    pub max_gyro_bias_magnitude_rad_s: f64,
    /// Milestone M5b: gyro-bias bootstrap gate — reject unless
    /// `GyroBiasAlignment::rotation_residual_rms_after` (radians) drops
    /// below this ABSOLUTE bound. Default `0.03` rad (~1.7°): no single M5
    /// real-run number for this quantity exists to calibrate against (M5
    /// never computed `rotation_residual_rms_after` at all — see
    /// `try_imu_bootstrap`'s own doc), so this is a conservative,
    /// physically-reasoned bound: EuRoC's own gyro noise density
    /// (`~1.7e-4 rad/s/√Hz`) integrated over a sub-second alignment window
    /// implies a pure-noise residual roughly two orders of magnitude below
    /// this, so `0.03` rad comfortably separates "noise" from "genuinely
    /// wrong rotation alignment" without being so tight normal EuRoC data
    /// can never pass it.
    pub gyro_bias_max_rms_after: f64,
    /// Milestone M5b: gyro-bias bootstrap gate — reject unless
    /// `rotation_residual_rms_after ≤ rotation_residual_rms_before ·` this
    /// fraction. Default `0.5`: the alignment must have actually moved the
    /// residual by at least half, not merely landed under the absolute
    /// bound above by starting close to it already (a genuinely
    /// rotation-noisy window can have a small `rms_before` too — this
    /// fraction gate catches "barely moved the needle" bootstraps the
    /// absolute bound alone would miss).
    pub gyro_bias_max_rms_fraction: f64,
    /// Milestone M5b: lower bound on `estimate_mono_vi_alignment`'s
    /// recovered scale `s` — task-specified default `0.05`.
    pub min_mono_scale: f64,
    /// Milestone M5b: upper bound on `estimate_mono_vi_alignment`'s
    /// recovered scale `s` — task-specified default `20.0`.
    pub max_mono_scale: f64,
    /// Milestone M5b: excitation/conditioning gate on
    /// `estimate_mono_vi_alignment`'s unconstrained-solve condition number
    /// — see that function's own module-doc section ("Observability
    /// gates") for the derivation. Default `1e8`, calibrated against that
    /// crate's own two synthetic measurements: a genuinely-3D-excited
    /// window's condition number `≈361` (comfortably below) vs. a
    /// constant-velocity window's `∞` (`min_sv` exactly `0.0`, rejected
    /// regardless of how loose this bound is) — a wide margin, not a
    /// knife-edge tuning.
    pub max_mono_alignment_condition_number: f64,
    /// Milestone M5b rollback monitor: mean whitened IMU-factor NIS bound
    /// (`crate::dpvo_vi_ba::imu_factor_nis`) — a `dpvo_vi_ba` solve whose
    /// in-window IMU factors average above this after re-linearizing is
    /// treated as one "bad" frame toward [`Self::rollback_consecutive_frames`].
    /// Default `500.0`: generously above a correctly-calibrated 9-dof
    /// chi-square's own ~`27.9` (99.9th percentile) to tolerate ordinary
    /// linearization/model-mismatch noise, while still catching a solve
    /// that is genuinely fighting a badly-scaled bootstrap every iteration
    /// — not empirically tuned against a specific M5b real-run NIS
    /// distribution (that distribution is exactly what this milestone's
    /// own acceptance run characterizes for the first time).
    pub rollback_mean_nis_bound: f64,
    /// Milestone M5b rollback monitor: number of CONSECUTIVE bad frames
    /// (mean NIS above [`Self::rollback_mean_nis_bound`]) before rolling
    /// back to visual-only. Default `5`: tolerates an isolated noisy
    /// frame's transient spike without treating it as a diagnosis, while
    /// still reacting within roughly half a second of real EuRoC-rate
    /// (`~10` Hz post-stride) frames once a bootstrap is genuinely bad.
    pub rollback_consecutive_frames: usize,
    /// Milestone M7 (`docs/dpvo_droid_port_plan.md`): continuous,
    /// uncertainty-weighted scale coupling — REPLACES the one-shot
    /// `Self::try_imu_bootstrap`-then-trust mechanism above when `Some`
    /// (every field above this one is then ignored except
    /// `body_to_camera`/`noise`/`gravity_magnitude`, still needed by the
    /// continuous re-estimation itself — see
    /// [`DpvoOdometry::scale_coupling_step`]'s own doc). `None` (default)
    /// preserves M5/M5b's exact one-shot behavior byte-for-byte.
    pub scale_coupling: Option<DpvoScaleCouplingConfig>,
}

impl Default for DpvoImuConfig {
    /// `body_to_camera = identity` is almost certainly wrong for a real rig
    /// — every real caller (see `examples/euroc_dpvo_vo_demo.rs`) must
    /// override it from the dataset's own `T_BS`. `noise` mirrors EuRoC's
    /// own MPU-9250-class sensor.yaml order-of-magnitude values (this
    /// codebase's own `examples/euroc_imu_dead_reckon_demo.rs` and
    /// `crates/io/src/euroc.rs` use the same real numbers when available —
    /// this default is a documented placeholder, not a claim about any
    /// specific sensor).
    fn default() -> Self {
        Self {
            body_to_camera: SE3::identity(),
            noise: ImuNoiseModel {
                gyroscope_noise_density: 1.6968e-4,
                accelerometer_noise_density: 2.0e-3,
            },
            gravity_magnitude: 9.81,
            gravity_norm_deviation_ratio: 0.3,
            min_bootstrap_factors: 10,
            max_gyro_bias_magnitude_rad_s: 0.05,
            gyro_bias_max_rms_after: 0.03,
            gyro_bias_max_rms_fraction: 0.5,
            min_mono_scale: 0.05,
            max_mono_scale: 20.0,
            max_mono_alignment_condition_number: 1.0e8,
            rollback_mean_nis_bound: 500.0,
            rollback_consecutive_frames: 5,
            scale_coupling: None,
        }
    }
}

/// Milestone M7: configuration for the continuous scale-coupling mechanism —
/// see `crate::dpvo_scale_coupling`'s module doc for the full design this
/// gates into. `min_window_factors` is this module's own gate (mirrors
/// [`DpvoImuConfig::min_bootstrap_factors`]'s role for M5b); every other
/// numeric knob lives on the reusable [`ScaleCouplingConfig`] itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoScaleCouplingConfig {
    /// Reused verbatim from `crate::dpvo_scale_coupling` — see that module's
    /// doc for every field's default and rationale.
    pub scale: ScaleCouplingConfig,
    /// Minimum number of usable in-window IMU factors before even
    /// ATTEMPTING a re-estimation this frame (a coarser gate than
    /// `estimate_gyro_bias`/`estimate_mono_vi_alignment`'s own internal
    /// `< 2` checks, avoiding a wasted SVD solve on a still-tiny window —
    /// same role as [`DpvoImuConfig::min_bootstrap_factors`]). Default `4`:
    /// deliberately smaller than M5b's `10`, because a rejected/degenerate
    /// measurement here costs nothing (no state is committed on a single
    /// bad attempt — see the module doc) whereas M5b's one-shot bootstrap
    /// needed a bigger, more conservative window before its single
    /// irreversible attempt.
    pub min_window_factors: usize,
}

impl Default for DpvoScaleCouplingConfig {
    fn default() -> Self {
        Self { scale: ScaleCouplingConfig::default(), min_window_factors: 4 }
    }
}

/// Milestone M7 snapshot of [`DpvoOdometry`]'s continuous scale-coupling
/// state — see [`DpvoOdometry::scale_coupling_diagnostics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoScaleCouplingDiagnostics {
    pub enabled: bool,
    /// Current annealing weight in `[0, 1]` — `0.0` means "behaving exactly
    /// like visual-only DPVO this frame", `1.0` means "fully trusting the
    /// IMU-coupled + gentle-scale-corrected solve".
    pub weight: f64,
    /// Whether the recursive scale estimator currently satisfies BOTH
    /// convergence gates (see `crate::dpvo_scale_coupling`'s module doc).
    pub converged: bool,
    /// `exp(posterior_mean)` — the current best estimate of the residual
    /// monocular-to-metric scale correction, once at least one measurement
    /// has been fused. `None` before the first measurement.
    pub recovered_scale: Option<f64>,
    /// Posterior standard deviation, in LOG-scale units (so `0.05` means
    /// "roughly 5% linear-scale uncertainty").
    pub posterior_log_std: Option<f64>,
    /// Current continuously-re-estimated gyro bias (never hard-fixed — see
    /// the module doc's "never-trusted-all-at-once" framing).
    pub bias_gyro: Vector3<f64>,
    pub measurements_taken: usize,
    pub measurements_rejected: usize,
    pub soft_rollback_count: usize,
    /// Diagnostic instrumentation added while investigating why real MH_01
    /// runs plateaued at a handful of accepted measurements (see this
    /// struct's own `last_rejection` doc and the plan doc's "M7 results"
    /// section, "Diagnosis" subsection) — per-reason breakdown of every
    /// `estimate_mono_vi_alignment` call this method's window was rejected
    /// by, mirroring `DpvoImuBootstrapRejectionCounts`'s own "isolate which
    /// gate" precedent from M5b.
    pub rejection_counts: DpvoScaleCouplingRejectionCounts,
    /// The MOST RECENT rejection's own full detail (carries the actual
    /// numeric value(s) that tripped it, e.g. the out-of-range scale itself,
    /// or the condition number vs. its bound) — `None` if no attempt has
    /// ever been rejected.
    pub last_rejection: Option<DpvoMonoViAlignmentRejection>,
}

/// Milestone M7 diagnostic addition: per-reason breakdown of
/// `estimate_mono_vi_alignment` rejections inside
/// [`DpvoOdometry::scale_coupling_step`] — see
/// [`DpvoScaleCouplingDiagnostics::rejection_counts`]'s own doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DpvoScaleCouplingRejectionCounts {
    pub not_enough_factors: usize,
    pub underdetermined: usize,
    pub ill_conditioned: usize,
    pub degenerate_solve: usize,
    pub gravity_norm: usize,
    pub scale_range: usize,
}

/// Snapshot of [`DpvoOdometry`]'s IMU bootstrap state, for a caller (e.g.
/// `examples/euroc_dpvo_vo_demo.rs`) to echo in a run summary. See
/// [`DpvoOdometry::imu_diagnostics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoImuDiagnostics {
    /// Whether the bootstrap chain (gyro-bias estimate, gated, then
    /// `crate::dpvo_vi_ba::estimate_mono_vi_alignment`, gated — Milestone
    /// M5b) is currently active. While `false`, [`DpvoOdometry::update_step`]
    /// runs the plain visual-only `crate::dpvo_patch_ba::dpvo_ba` solve,
    /// identical to M4 — IMU coupling only engages once this is `true`.
    /// Unlike M5, this CAN revert to `false` again: see
    /// [`DpvoOdometry::rollback_imu_bootstrap`]'s doc for the M5b rollback
    /// monitor that flips it back.
    pub bootstrapped: bool,
    /// Recovered world-frame gravity vector, while bootstrapped (cleared on
    /// rollback).
    pub gravity_world: Option<Vector3<f64>>,
    pub bias_gyro: Vector3<f64>,
    pub bias_accel: Vector3<f64>,
    /// Milestone M5b: the monocular scale recovered by the most recent
    /// SUCCESSFUL bootstrap (`crate::dpvo_vi_ba::DpvoMonoViAlignment::scale`).
    /// Not cleared by a later rollback — a caller inspecting a finished
    /// run's diagnostics still wants to know what scale, if any, was ever
    /// recovered, even if the run subsequently rolled back and (possibly)
    /// never re-bootstrapped.
    pub recovered_scale: Option<f64>,
    /// Milestone M5b: total number of times [`DpvoOdometry::try_imu_bootstrap`]
    /// got far enough to actually run the gyro-bias/mono-alignment gates
    /// (i.e. [`DpvoImuConfig::min_bootstrap_factors`] was already met) —
    /// includes both attempts that passed and attempts that were rejected.
    pub bootstrap_attempts: usize,
    /// Milestone M5b: number of those attempts rejected by ANY gate (gyro
    /// magnitude/rms, mono-alignment DOF/conditioning/gravity-norm/scale).
    pub bootstrap_rejections: usize,
    /// Milestone M5b: number of times the post-bootstrap rollback monitor
    /// actually tripped (see [`DpvoOdometry::rollback_imu_bootstrap`]).
    pub rollback_count: usize,
    /// Milestone M5b: per-reason breakdown of every rejected attempt — the
    /// task's own "isolate which gate" acceptance requirement, answerable
    /// from a live run's own diagnostics rather than guesswork. See
    /// [`DpvoImuBootstrapRejectionCounts`].
    pub rejection_counts: DpvoImuBootstrapRejectionCounts,
    /// Milestone M5b: the MOST RECENT rejection's own reason plus the
    /// specific value(s) that tripped it (e.g. the actual `rms_after` vs.
    /// its bound, or the actual condition number vs. its bound) — lets a
    /// caller report "how close" a real run's own gates are sitting to
    /// their thresholds, not just a bare pass/fail count. `None` if no
    /// attempt has ever been rejected (either none have been made yet, or
    /// every attempt so far has succeeded).
    pub last_rejection: Option<DpvoImuRejectionDetail>,
}

/// Milestone M6 snapshot of [`DpvoOdometry`]'s loop-closure state - see
/// [`DpvoOdometry::loop_closure_diagnostics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoLoopClosureDiagnostics {
    /// Whether `config.loop_closure` is `Some` at all - `false` means every
    /// other field here is a static zero (the feature was never engaged),
    /// not a claim that closure was attempted and found nothing.
    pub enabled: bool,
    /// Number of times [`DpvoOdometry::try_loop_closure`] actually ran
    /// `crate::dpvo_loop_closure::find_loop_edges` (i.e. was "due" per
    /// `GLOBAL_OPT_FREQ`'s own throttle - see that method's doc for why this
    /// can fire on consecutive frames before the first successful batch,
    /// matching upstream's own `last_global_ba` bookkeeping exactly).
    pub batches_attempted: usize,
    /// Cumulative candidate `(i, j)` frame pairs that cleared
    /// `find_loop_edges`'s own `backend_thresh`/validity-fraction gate,
    /// across every attempted batch (before the edge-budget/NMS selection -
    /// the task's own "candidates" diagnostic).
    pub candidates_evaluated_total: usize,
    /// Cumulative accepted `(i, j)` frame pairs (post edge-budget/NMS
    /// selection) across every batch - the task's own "accepted loops"
    /// diagnostic.
    pub accepted_loops_total: usize,
    /// Cumulative patch-level `(patch_id, target_frame)` edges actually
    /// appended to the live patch graph (`accepted_loops_total *
    /// patches_per_frame`, tracked directly rather than recomputed) - the
    /// task's own "edges added" diagnostic.
    pub patch_edges_added_total: usize,
    /// Accepted frame-pair count from the MOST RECENT batch that found
    /// anything (`0` if none ever did) - lets a caller's log line report
    /// "just found N loops" without re-deriving it from the cumulative
    /// totals.
    pub last_batch_accepted_loops: usize,
    /// Number of `update_step` calls whose BA solve incorporated at least
    /// one freshly-added loop batch (i.e. the number of samples underlying
    /// [`Self::correction_magnitude_max_m`]/`_mean_m`).
    pub correction_events: usize,
    /// Largest observed pose-translation correction (meters) at a loop
    /// batch's own source frame(s), comparing that frame's pose immediately
    /// before vs. immediately after the same `update_step` call that
    /// incorporated the new loop edge(s) - the task's own "correction
    /// magnitude" diagnostic. `0.0` if [`Self::correction_events`] is `0`.
    pub correction_magnitude_max_m: f64,
    /// Mean of the same per-event correction magnitude. `0.0` if
    /// [`Self::correction_events`] is `0`.
    pub correction_magnitude_mean_m: f64,
}

/// Milestone M5b: cumulative counters, one per DISTINCT rejection reason
/// across both bootstrap gates (gyro-bias, then mono-alignment) — see
/// [`DpvoOdometry::try_imu_bootstrap`]'s doc for exactly which check each
/// one corresponds to. Every rejected attempt increments EXACTLY one of
/// these (the gates are checked in a fixed order and the first failure
/// short-circuits the rest), so `gyro_estimator_none + gyro_magnitude +
/// gyro_rms_absolute + gyro_rms_fraction + mono_not_enough_factors +
/// mono_underdetermined + mono_ill_conditioned + mono_degenerate_solve +
/// mono_gravity_norm + mono_scale_range == bootstrap_rejections` always.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DpvoImuBootstrapRejectionCounts {
    /// `vi_motion_initializer::estimate_gyro_bias` itself returned `None`
    /// (degenerate window: too few usable rotation factors).
    pub gyro_estimator_none: usize,
    /// [`GyroGateRejection::MagnitudeTooLarge`].
    pub gyro_magnitude: usize,
    /// [`GyroGateRejection::RmsAboveAbsoluteBound`].
    pub gyro_rms_absolute: usize,
    /// [`GyroGateRejection::RmsNotEnoughImprovement`].
    pub gyro_rms_fraction: usize,
    /// `crate::dpvo_vi_ba::DpvoMonoViAlignmentRejection::NotEnoughFactors`.
    pub mono_not_enough_factors: usize,
    /// `crate::dpvo_vi_ba::DpvoMonoViAlignmentRejection::Underdetermined`.
    pub mono_underdetermined: usize,
    /// `crate::dpvo_vi_ba::DpvoMonoViAlignmentRejection::IllConditioned`
    /// (the excitation/conditioning gate).
    pub mono_ill_conditioned: usize,
    /// `crate::dpvo_vi_ba::DpvoMonoViAlignmentRejection::DegenerateSolve`.
    pub mono_degenerate_solve: usize,
    /// `crate::dpvo_vi_ba::DpvoMonoViAlignmentRejection::GravityNormDeviation`.
    pub mono_gravity_norm: usize,
    /// `crate::dpvo_vi_ba::DpvoMonoViAlignmentRejection::ScaleOutOfRange`.
    pub mono_scale_range: usize,
}

/// Milestone M5b: the most recent bootstrap-attempt rejection's reason plus
/// the specific value(s) that tripped it — see
/// [`DpvoImuDiagnostics::last_rejection`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DpvoImuRejectionDetail {
    /// `vi_motion_initializer::estimate_gyro_bias` returned `None`.
    GyroEstimatorNone,
    /// The gyro-bias gate rejected — see [`gyro_bootstrap_gate_check`]'s
    /// doc for `reason`'s meaning; `bias_norm`/`rms_before`/`rms_after` are
    /// the actual recovered values (vs. `DpvoImuConfig`'s configured
    /// bounds) at the time of rejection.
    GyroGate { reason: GyroGateRejection, bias_norm: f64, rms_before: f64, rms_after: f64 },
    /// `crate::dpvo_vi_ba::estimate_mono_vi_alignment` rejected — the
    /// wrapped `DpvoMonoViAlignmentRejection` already carries its own
    /// specific offending value(s).
    MonoGate(DpvoMonoViAlignmentRejection),
}

/// Cumulative per-run timing/tracking counters, snapshotted after every
/// [`DpvoOdometry::process_frame`] call — the "ms/frame (encoder/update/BA
/// split)" breakdown the M4 acceptance criteria ask for (divide by
/// `frames_processed`/`frames_tracked` for an average).
#[derive(Debug, Clone, Copy, Default)]
pub struct DpvoOdometryStats {
    pub frames_processed: usize,
    pub frames_tracked: usize,
    pub encode_ms_total: f64,
    /// Time spent reprojecting every active edge's patch grid
    /// (`reproject_patch_grid`) and assembling the 2-pyramid-level
    /// correlation tensor (`corr_pyramid`/`corr_cpu`) — measured
    /// separately from `update_ms_total` because it turned out to
    /// dominate this naive CPU port's per-frame cost (see the plan doc's
    /// M4 results, "timing breakdown").
    pub correlation_ms_total: f64,
    /// Time spent inside the ONNX GRU update-cell call
    /// (`DpvoOnnxSession::update_iteration`) only.
    pub update_ms_total: f64,
    pub ba_ms_total: f64,
}

/// One live frame's cached `fnet` feature pyramid (`DPVO.pyramid`,
/// `dpvo.py:72-76`): `level0` is the full-stride-4 feature map
/// (`avg_pool2d(fmap, 1, 1)` — a no-op, kept only for naming symmetry with
/// upstream), `level1` is a further ×4 average-pooled map
/// (`avg_pool2d(fmap, 4, 4)`, `dpvo.py:438`).
///
/// # M4-perf: stored channel-last, not channel-first (`docs/dpvo_droid_port_plan.md`)
///
/// Both levels are stored as [`ChannelLastImage`] — already transposed into
/// the layout [`corr_cpu_prebuilt_target`] needs — rather than the raw
/// `(channels, height, width)` `ort`/`avg_pool_4x4` output. A frame's pyramid
/// is built exactly once (`process_frame`, when the frame arrives) but read
/// by [`corr_pyramid`] once per active edge group per `update_step`/
/// `motion_probe` call for as long as the frame stays inside the active
/// window (up to `REMOVAL_WINDOW`/`PATCH_LIFETIME` calls) — profiling
/// (plan doc, "M4-perf results") showed re-transposing the same ~120x188x128
/// feature map from scratch on every one of those reads was a real,
/// avoidable cost, not just a theoretical one. Nothing outside this module
/// reads `level0`/`level1` in channel-first form (checked directly — see the
/// plan doc's writeup), so there is no other consumer to keep a duplicate
/// channel-first copy for.
#[derive(Debug, Clone)]
struct FramePyramid {
    level0: ChannelLastImage,
    level1: ChannelLastImage,
}

/// Milestone M7: `(poses, patches, velocities)` in a window's own local
/// indexing — [`DpvoOdometry::scale_coupling_step`]'s return type, named
/// here only to satisfy `clippy::type_complexity` (no semantic meaning
/// beyond "one of these three per window-local index").
type ScaleCouplingSolution = (Vec<SE3>, Vec<DpvoPatch>, Option<Vec<Vector3<f64>>>);

/// The full DPVO visual-odometry loop: ONNX sessions + the [`DpvoPatchGraph`]
/// they drive. See the module doc for scope and the windowing derivation.
pub struct DpvoOdometry {
    config: DpvoOdometryConfig,
    session: DpvoOnnxSession,
    agg_kk: SoftAgg,
    agg_ij: SoftAgg,
    graph: DpvoPatchGraph,
    /// Parallel to `graph.frames()`.
    frame_pyramids: Vec<FramePyramid>,
    /// Parallel to `graph.patches()` (flat, per-frame contiguous blocks).
    patch_gmap: Vec<Array3<f32>>,
    patch_imap: Vec<Array1<f32>>,
    rng: StdRng,
    stats: DpvoOdometryStats,

    // ---- Milestone M5 (IMU coupling) state — see the module doc's own
    // "IMU coupling" section and `crate::dpvo_vi_ba`'s module doc for the
    // math. All of this is inert (never read, harmlessly accumulates
    // nothing of consequence) when `config.imu` is `None`. ----
    /// Raw `(timestamp, gyro, accel)` samples from [`Self::push_imu`], not
    /// yet folded into a preintegrated delta. Drained (never re-read) by
    /// [`Self::integrate_imu_for_new_frame`] every time a new frame commits.
    pending_imu: VecDeque<(f64, Vector3<f64>, Vector3<f64>)>,
    /// The timestamp boundary the next integration window starts from —
    /// either the previous committed frame's timestamp, or `None` before
    /// any frame has committed. See [`Self::integrate_imu_for_new_frame`]'s
    /// doc for why the sub-sample fragment right at a frame boundary is
    /// deliberately left un-integrated (negligible at IMU rates ≫ camera
    /// rate).
    last_imu_boundary_timestamp: Option<f64>,
    /// Preintegrated deltas between CONSECUTIVE COMMITTED frames, keyed by
    /// `(arrival_index_from, arrival_index_to)` — stable across
    /// `DpvoPatchGraph::keyframe`'s frame-compaction (which renumbers live
    /// frame *indices*, but never touches `arrival_index`, see that
    /// module's own doc). A missing key for an otherwise-consecutive pair
    /// means no IMU samples were available for that gap (no factor is
    /// banked; the IMU chain simply has a gap there — a documented,
    /// graceful degradation, not an error).
    imu_deltas_by_arrival: HashMap<(usize, usize), ImuPreintegrationFactor>,
    /// Bootstrap-only evidence, decoupled from `DpvoPatchGraph::keyframe`'s
    /// live-window churn: `(arrival_from, arrival_to, pose_from_snapshot,
    /// pose_to_snapshot, delta)`, one entry per banked
    /// [`Self::imu_deltas_by_arrival`] insertion, with the two frames'
    /// **poses as they were at bank time** (right after that frame's own
    /// `update_step`, so already visually-BA-refined at least once — see
    /// [`Self::try_imu_bootstrap`]'s doc for why using the live graph's
    /// *current* frame set instead was a genuine bug, not a design choice:
    /// EuRoC MH_01's slow opening segment folds frames away via
    /// `DpvoPatchGraph::keyframe`'s motion-magnitude gate faster than 10
    /// (`estimate_gyro_bias`/`estimate_gravity_and_velocities`'s own
    /// `MAX_ALIGNMENT_WINDOW`) usable factors could ever accumulate against
    /// a live-frames-only view, so bootstrap simply never fired). Capped at
    /// [`IMU_BOOTSTRAP_HISTORY_CAP`] entries (pruned oldest-first) — far
    /// more than the estimators' own 10-keyframe window ever uses, just
    /// enough headroom that a slow-motion opening stretch doesn't force a
    /// bootstrap attempt before real excitation shows up.
    imu_bootstrap_history: Vec<(usize, usize, SE3, SE3, ImuPreintegratedDelta)>,
    /// Per-live-frame world-frame velocity estimate, parallel to
    /// `graph.frames()` (grown on commit, removed on fold — mirrors
    /// `frame_pyramids`'s own lifecycle exactly).
    velocities: Vec<Vector3<f64>>,
    /// Shared gyro bias, fixed once [`Self::try_imu_bootstrap`] succeeds
    /// (staged-bias philosophy — see `crate::dpvo_vi_ba`'s module doc).
    imu_bias_gyro: Vector3<f64>,
    imu_bias_accel: Vector3<f64>,
    /// `Some` while bootstrapped; cleared on [`Self::rollback_imu_bootstrap`]
    /// (Milestone M5b — see [`DpvoImuDiagnostics::bootstrapped`]'s doc for
    /// why this can now revert, unlike M5).
    imu_gravity_world: Option<Vector3<f64>>,
    imu_bootstrapped: bool,

    // ---- Milestone M5b additions: bootstrap gating diagnostics + the
    // rollback monitor. See the module doc's "Milestone M5's honest
    // negative, and what M5b changes" section. ----
    /// Total bootstrap attempts that got far enough to run the gates (see
    /// [`DpvoImuDiagnostics::bootstrap_attempts`]).
    imu_bootstrap_attempts: usize,
    /// Attempts rejected by any gate (see
    /// [`DpvoImuDiagnostics::bootstrap_rejections`]).
    imu_bootstrap_rejections: usize,
    /// Times [`Self::rollback_imu_bootstrap`] actually fired (see
    /// [`DpvoImuDiagnostics::rollback_count`]).
    imu_rollback_count: usize,
    /// Running count of consecutive `update_step` calls (while bootstrapped)
    /// whose mean IMU-factor NIS exceeded
    /// [`DpvoImuConfig::rollback_mean_nis_bound`] — reset to `0` on any
    /// frame back under the bound, or by [`Self::rollback_imu_bootstrap`]
    /// itself (a fresh bootstrap starts this fresh too).
    imu_consecutive_bad_frames: usize,
    /// The most recently recovered mono scale — see
    /// [`DpvoImuDiagnostics::recovered_scale`]'s doc for why this is NOT
    /// cleared by a rollback.
    recovered_mono_scale: Option<f64>,
    /// Per-reason rejection tally — see [`DpvoImuDiagnostics::rejection_counts`].
    imu_rejection_counts: DpvoImuBootstrapRejectionCounts,
    /// The most recent rejection's own detail — see
    /// [`DpvoImuDiagnostics::last_rejection`].
    imu_last_rejection: Option<DpvoImuRejectionDetail>,

    // ---- Milestone M6 (loop closure) state — see the module doc's own
    // "Loop closure" section and `crate::dpvo_loop_closure`'s module doc for
    // the math. All of this is inert when `config.loop_closure` is `None`.
    // ----
    /// Live-frame index `n` at the last batch that found and appended at
    /// least one accepted loop edge, or `None` before any such batch (the
    /// "always eligible on the very next frame" state — matches upstream's
    /// own `self.last_global_ba = -1000` sentinel, see
    /// [`Self::try_loop_closure`]'s own doc for why `None` behaves the same
    /// way).
    last_loop_batch_frame: Option<usize>,
    loop_batches_attempted: usize,
    loop_candidates_evaluated_total: usize,
    loop_accepted_total: usize,
    loop_patch_edges_added_total: usize,
    loop_last_batch_accepted: usize,
    loop_correction_events: usize,
    loop_correction_sum_m: f64,
    loop_correction_max_m: f64,

    // ---- Milestone M7 (continuous scale coupling) state — see
    // `crate::dpvo_scale_coupling`'s module doc and
    // [`Self::scale_coupling_step`]'s own doc. Inert (never read/updated
    // meaningfully) whenever `config.imu.scale_coupling` is `None`. ----
    scale_estimator: RecursiveScaleEstimator,
    gyro_bias_estimator: RecursiveGyroBiasEstimator,
    scale_coupling_weight: AnnealingWeight,
    /// Most recently ACCEPTED alignment's own recovered gravity — see
    /// `crate::dpvo_scale_coupling`'s module doc for why gravity itself is
    /// not put through its own recursive filter this milestone.
    scale_coupling_gravity: Option<Vector3<f64>>,
    scale_coupling_consecutive_bad: usize,
    scale_coupling_measurements: usize,
    scale_coupling_measurement_rejections: usize,
    scale_coupling_rollback_count: usize,
    /// Diagnostic instrumentation (see [`DpvoScaleCouplingDiagnostics::rejection_counts`]).
    scale_coupling_rejection_counts: DpvoScaleCouplingRejectionCounts,
    scale_coupling_last_rejection: Option<DpvoMonoViAlignmentRejection>,
}

impl DpvoOdometry {
    /// Load the four M1-exported ONNX graphs plus the `SoftAgg` weight
    /// artifact (`softagg_weights_fixture.npz` — see
    /// `crates/vision/src/dpvo/mod.rs`'s module doc, "Why does `SoftAgg`
    /// need to load weights from an npz at all?"; this is the *same*
    /// checkpoint-derived artifact M2 produced, reused here as-is rather
    /// than re-exported under a new name, since the weights it carries are
    /// already real, just fixture-shaped file-naming).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: DpvoOdometryConfig,
        fnet_path: impl AsRef<Path>,
        inet_path: impl AsRef<Path>,
        update_pre_agg_path: impl AsRef<Path>,
        update_post_agg_path: impl AsRef<Path>,
        softagg_weights_npz_path: impl AsRef<Path>,
        backend: OnnxBackend,
    ) -> Result<Self, DpvoOdometryError> {
        let session = DpvoOnnxSession::load_from_paths_with_backend(
            fnet_path,
            inet_path,
            update_pre_agg_path,
            update_post_agg_path,
            backend,
        )?;
        let archive = NpzArchive::open(softagg_weights_npz_path)?;
        let agg_kk = SoftAgg::load_from_npz(&archive, "agg_kk_")?;
        let agg_ij = SoftAgg::load_from_npz(&archive, "agg_ij_")?;
        let graph = DpvoPatchGraph::new(config.vo);
        let seed = config.seed;
        // Milestone M7: derive the scale-coupling sub-config once, up front
        // (before `config` moves into `Self` below) — `Default` whenever
        // `config.imu`/`config.imu.scale_coupling` is `None`, harmless since
        // nothing reads these estimators unless that config is `Some` (see
        // `Self::scale_coupling_step`'s own guard).
        let sc_cfg = config
            .imu
            .as_ref()
            .and_then(|imu| imu.scale_coupling)
            .map(|sc| sc.scale)
            .unwrap_or_default();
        Ok(Self {
            config,
            session,
            agg_kk,
            agg_ij,
            graph,
            frame_pyramids: Vec::new(),
            patch_gmap: Vec::new(),
            patch_imap: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            stats: DpvoOdometryStats::default(),
            pending_imu: VecDeque::new(),
            last_imu_boundary_timestamp: None,
            imu_deltas_by_arrival: HashMap::new(),
            imu_bootstrap_history: Vec::new(),
            velocities: Vec::new(),
            imu_bias_gyro: Vector3::zeros(),
            imu_bias_accel: Vector3::zeros(),
            imu_gravity_world: None,
            imu_bootstrapped: false,
            imu_bootstrap_attempts: 0,
            imu_bootstrap_rejections: 0,
            imu_rollback_count: 0,
            imu_consecutive_bad_frames: 0,
            recovered_mono_scale: None,
            imu_rejection_counts: DpvoImuBootstrapRejectionCounts::default(),
            imu_last_rejection: None,
            last_loop_batch_frame: None,
            loop_batches_attempted: 0,
            loop_candidates_evaluated_total: 0,
            loop_accepted_total: 0,
            loop_patch_edges_added_total: 0,
            loop_last_batch_accepted: 0,
            loop_correction_events: 0,
            loop_correction_sum_m: 0.0,
            loop_correction_max_m: 0.0,
            scale_estimator: RecursiveScaleEstimator::new(sc_cfg),
            gyro_bias_estimator: RecursiveGyroBiasEstimator::new(sc_cfg),
            scale_coupling_weight: AnnealingWeight::new(sc_cfg.anneal_frames, sc_cfg.decay_frames),
            scale_coupling_gravity: None,
            scale_coupling_consecutive_bad: 0,
            scale_coupling_measurements: 0,
            scale_coupling_measurement_rejections: 0,
            scale_coupling_rollback_count: 0,
            scale_coupling_rejection_counts: DpvoScaleCouplingRejectionCounts::default(),
            scale_coupling_last_rejection: None,
        })
    }

    pub fn stats(&self) -> DpvoOdometryStats {
        self.stats
    }

    pub fn graph(&self) -> &DpvoPatchGraph {
        &self.graph
    }

    /// Snapshot of the IMU bootstrap chain's current state (Milestone M5,
    /// extended M5b). See [`DpvoImuDiagnostics`].
    pub fn imu_diagnostics(&self) -> DpvoImuDiagnostics {
        DpvoImuDiagnostics {
            bootstrapped: self.imu_bootstrapped,
            gravity_world: self.imu_gravity_world,
            bias_gyro: self.imu_bias_gyro,
            bias_accel: self.imu_bias_accel,
            recovered_scale: self.recovered_mono_scale,
            bootstrap_attempts: self.imu_bootstrap_attempts,
            bootstrap_rejections: self.imu_bootstrap_rejections,
            rollback_count: self.imu_rollback_count,
            rejection_counts: self.imu_rejection_counts,
            last_rejection: self.imu_last_rejection,
        }
    }

    /// Snapshot of the loop-closure chain's current state (Milestone M6).
    /// See [`DpvoLoopClosureDiagnostics`].
    pub fn loop_closure_diagnostics(&self) -> DpvoLoopClosureDiagnostics {
        let correction_mean = if self.loop_correction_events > 0 {
            self.loop_correction_sum_m / self.loop_correction_events as f64
        } else {
            0.0
        };
        DpvoLoopClosureDiagnostics {
            enabled: self.config.loop_closure.is_some(),
            batches_attempted: self.loop_batches_attempted,
            candidates_evaluated_total: self.loop_candidates_evaluated_total,
            accepted_loops_total: self.loop_accepted_total,
            patch_edges_added_total: self.loop_patch_edges_added_total,
            last_batch_accepted_loops: self.loop_last_batch_accepted,
            correction_events: self.loop_correction_events,
            correction_magnitude_max_m: self.loop_correction_max_m,
            correction_magnitude_mean_m: correction_mean,
        }
    }

    /// Snapshot of the Milestone M7 continuous scale-coupling state — see
    /// [`DpvoScaleCouplingDiagnostics`].
    pub fn scale_coupling_diagnostics(&self) -> DpvoScaleCouplingDiagnostics {
        let posterior = self.scale_estimator.posterior();
        DpvoScaleCouplingDiagnostics {
            enabled: self.config.imu.as_ref().is_some_and(|c| c.scale_coupling.is_some()),
            weight: self.scale_coupling_weight.value,
            converged: self.scale_estimator.is_converged(),
            recovered_scale: posterior.map(|p| p.mean.exp()),
            posterior_log_std: posterior.map(|p| p.variance.sqrt()),
            bias_gyro: self.gyro_bias_estimator.mean(),
            measurements_taken: self.scale_coupling_measurements,
            measurements_rejected: self.scale_coupling_measurement_rejections,
            soft_rollback_count: self.scale_coupling_rollback_count,
            rejection_counts: self.scale_coupling_rejection_counts,
            last_rejection: self.scale_coupling_last_rejection,
        }
    }

    /// Buffer one raw body-frame IMU sample (Milestone M5). No-op (samples
    /// are simply discarded on the next [`Self::process_frame`]'s drain if
    /// `config.imu` is `None` — accepted, not rejected with an error, since
    /// a caller streaming both cam0 and imu0 in real time from a dataset
    /// like EuRoC has no natural place to gate this on config without
    /// threading it back out again). Samples must arrive in non-decreasing
    /// `timestamp` order (seconds) — the same precondition
    /// `crate::imu_preintegration::ImuPreintegrator::integrate_sample`
    /// already has for `dt > 0`.
    pub fn push_imu(&mut self, timestamp: f64, gyro: Vector3<f64>, accel: Vector3<f64>) {
        self.pending_imu.push_back((timestamp, gyro, accel));
    }

    /// Process one incoming grayscale frame (`(height, width)`, `RES`- and
    /// distortion-corrected upstream by the caller — see
    /// `examples/euroc_dpvo_vo_demo.rs`). Returns the just-processed frame's
    /// current best pose estimate, or `None` if `motion_probe` rejected it
    /// (`dpvo.py:441-444`) — a rejected frame's pose is only recoverable
    /// later via [`crate::dpvo_patch_graph::DpvoPatchGraph::reconstruct_pose`].
    pub fn process_frame(
        &mut self,
        image: ArrayView2<'_, u8>,
        timestamp: f64,
    ) -> Result<Option<SE3>, DpvoOdometryError> {
        let (h, w) = image.dim();
        if (w, h) != (self.config.width, self.config.height) {
            return Err(DpvoOdometryError::ImageShapeMismatch {
                expected: (self.config.width, self.config.height),
                actual: (w, h),
            });
        }
        self.stats.frames_processed += 1;

        let encode_start = Instant::now();
        let input = grayscale_to_input_tensor(image);
        let fmap4 = self.session.run_fnet(input.view())?;
        let imap4 = self.session.run_inet(input.view())?;
        self.stats.encode_ms_total += encode_start.elapsed().as_secs_f64() * 1000.0;

        let fmap = fmap4.index_axis(Axis(0), 0).to_owned();
        let imap_full = imap4.index_axis(Axis(0), 0).to_owned();
        let (_, hs, ws) = fmap.dim();

        // Centroid sampling (`Patchifier.forward`, `RANDOM` strategy,
        // `net.py:131-133`): integers in `[1, w-1)`/`[1, h-1)` in `fmap`'s
        // own (stride-RES) coordinate space.
        let m = self.graph.config().patches_per_frame;
        let coords: Vec<(f32, f32)> = (0..m)
            .map(|_| {
                let x = self.rng.gen_range(1..ws - 1) as f32;
                let y = self.rng.gen_range(1..hs - 1) as f32;
                (x, y)
            })
            .collect();

        let gmap = patchify_cpu(fmap.view(), &coords, 1); // (M, 128, 3, 3)
        let imap_patch4 = patchify_cpu(imap_full.view(), &coords, 0); // (M, 384, 1, 1)

        // Depth init (`dpvo.py:427-430`): always start from `rand()`, then
        // overwrite with the median of the last 3 frames' depths once
        // initialized.
        let mut depths: Vec<f64> = (0..m).map(|_| self.rng.gen_range(0.0..1.0)).collect();
        if self.graph.is_initialized() {
            let median = self.median_recent_depth();
            depths.iter_mut().for_each(|d| *d = median);
        }
        let patches_vec: Vec<DpvoPatch> = (0..m)
            .map(|i| DpvoPatch { x: coords[i].0 as f64, y: coords[i].1 as f64, inverse_depth: depths[i] })
            .collect();

        let predicted_pose = self.graph.begin_frame(timestamp);
        let intr = DpvoIntrinsics {
            fx: self.config.intrinsics.fx / RES as f64,
            fy: self.config.intrinsics.fy / RES as f64,
            cx: self.config.intrinsics.cx / RES as f64,
            cy: self.config.intrinsics.cy / RES as f64,
        };
        // M4-perf (`docs/dpvo_droid_port_plan.md`): transpose both pyramid
        // levels to channel-last exactly once here, at the point the
        // pyramid is built — see `FramePyramid`'s doc for why this replaces
        // storing (and repeatedly re-transposing) the raw channel-first
        // arrays.
        let level1_chw = avg_pool_4x4(fmap.view());
        let candidate_pyramid = FramePyramid {
            level0: ChannelLastImage::from_chw(fmap.view()),
            level1: ChannelLastImage::from_chw(level1_chw.view()),
        };

        if self.graph.n_frames() > 0 && !self.graph.is_initialized() {
            let flow = self.motion_probe(&predicted_pose, &intr, &candidate_pyramid)?;
            if flow < self.config.motion_probe_min_flow {
                self.graph.reject_pending_frame();
                return Ok(None);
            }
        }

        self.graph.commit_frame(predicted_pose, intr, patches_vec)?;
        self.frame_pyramids.push(candidate_pyramid);
        for i in 0..m {
            self.patch_gmap.push(gmap.index_axis(Axis(0), i).to_owned());
            self.patch_imap.push(squeeze_patch_vector(&imap_patch4, i));
        }
        // Milestone M5: one velocity slot per live frame, parallel to
        // `frame_pyramids` — see this struct's own field doc. Seeded at
        // zero; `try_imu_bootstrap`/`update_step` overwrite it once IMU
        // coupling is active.
        self.velocities.push(Vector3::zeros());
        self.integrate_imu_for_new_frame(timestamp);

        let forw = self.graph.edges_forw();
        let back = self.graph.edges_back();
        self.graph.append_edges(&forw, DIM);
        self.graph.append_edges(&back, DIM);

        if self.graph.n_frames() == 8 && !self.graph.is_initialized() {
            self.graph.set_initialized(true);
            for _ in 0..12 {
                self.update_step()?;
            }
        } else if self.graph.is_initialized() {
            // Milestone M7: the continuous scale-coupling mechanism (see
            // `Self::scale_coupling_step`, called from `update_step` below)
            // REPLACES M5b's one-shot bootstrap entirely when enabled — it
            // re-estimates gyro bias/scale itself, every window, so the
            // one-shot `try_imu_bootstrap` must not also run (the two
            // mechanisms would otherwise both try to own
            // `self.imu_bias_gyro`/`self.imu_bootstrapped`).
            let use_scale_coupling = self.config.imu.as_ref().is_some_and(|c| c.scale_coupling.is_some());
            if !use_scale_coupling {
                self.try_imu_bootstrap();
            }
            self.try_loop_closure();
            self.update_step()?;
            if let Some(k) = self.keyframe_dispatch() {
                self.frame_pyramids.remove(k);
                let m = self.graph.config().patches_per_frame;
                self.patch_gmap.drain(k * m..(k + 1) * m);
                self.patch_imap.drain(k * m..(k + 1) * m);
                self.velocities.remove(k);
                self.prune_stale_imu_deltas();
            }
        }

        self.stats.frames_tracked += 1;
        Ok(self.graph.frames().last().map(|f| f.pose.clone()))
    }

    /// `torch.median` over the last 3 committed frames' patch inverse
    /// depths (`dpvo.py:428-430`). `torch.median` (unlike `torch.quantile`)
    /// returns the *lower* of the two middle values for an even count —
    /// `sorted[(n-1)/2]` under integer division reproduces that exactly.
    fn median_recent_depth(&self) -> f64 {
        let n = self.graph.n_frames();
        let m = self.graph.config().patches_per_frame;
        let lo = n.saturating_sub(3) * m;
        let hi = n * m;
        let mut values: Vec<f64> = self.graph.patches()[lo..hi].iter().map(|p| p.inverse_depth).collect();
        if values.is_empty() {
            return 1.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values[(values.len() - 1) / 2]
    }

    /// `dpvo.py::motion_probe` (lines 240-255): reproject the *previous*
    /// committed frame's patches into the *candidate* frame's predicted
    /// pose (not yet committed), run one zero-history GRU update, and
    /// return the median predicted-correction magnitude — the gate that
    /// decides whether the candidate carries enough motion to accept during
    /// the pre-initialization phase.
    fn motion_probe(
        &mut self,
        predicted_pose: &SE3,
        candidate_intr: &DpvoIntrinsics,
        candidate_pyramid: &FramePyramid,
    ) -> Result<f64, DpvoOdometryError> {
        let n = self.graph.n_frames();
        let prev_frame = n - 1;
        let m = self.graph.config().patches_per_frame;
        let patch_lo = prev_frame * m;
        let prev_pose = self.graph.frames()[prev_frame].pose.clone();
        let prev_intr = self.graph.frames()[prev_frame].intrinsics;

        let corr_start = Instant::now();
        let mut anchor_gmap = Array4::<f32>::zeros((m, FNET_DIM, PATCH, PATCH));
        let mut coords_grid_px = Array4::<f32>::zeros((m, PATCH, PATCH, 2));
        for local in 0..m {
            let patch = self.graph.patches()[patch_lo + local];
            let grid = reproject_patch_grid(&prev_pose, predicted_pose, &prev_intr, candidate_intr, &patch);
            anchor_gmap.index_axis_mut(Axis(0), local).assign(&self.patch_gmap[patch_lo + local]);
            for py in 0..PATCH {
                for px in 0..PATCH {
                    coords_grid_px[(local, py, px, 0)] = grid[py][px].x as f32;
                    coords_grid_px[(local, py, px, 1)] = grid[py][px].y as f32;
                }
            }
        }
        let corr_flat = corr_pyramid(
            anchor_gmap.view(),
            coords_grid_px.view(),
            &candidate_pyramid.level0,
            &candidate_pyramid.level1,
        );
        self.stats.correlation_ms_total += corr_start.elapsed().as_secs_f64() * 1000.0;

        let net_zero = Array3::<f32>::zeros((1, m, DIM));
        let mut inp_arr = Array3::<f32>::zeros((1, m, DIM));
        for local in 0..m {
            inp_arr.index_axis_mut(Axis(0), 0).index_axis_mut(Axis(0), local).assign(&self.patch_imap[patch_lo + local]);
        }
        let kk: Vec<i64> = (patch_lo..patch_lo + m).map(|k| k as i64).collect();
        let ii = vec![prev_frame as i64; m];
        let jj = vec![n as i64; m];
        let corr3 = corr_flat.insert_axis(Axis(0));
        let update_start = Instant::now();
        let (_net_out, delta, _weight) = self
            .session
            .update_iteration(net_zero.view(), inp_arr.view(), corr3.view(), &kk, &ii, &jj, &self.agg_kk, &self.agg_ij)?;
        self.stats.update_ms_total += update_start.elapsed().as_secs_f64() * 1000.0;

        let mut norms: Vec<f64> = (0..m)
            .map(|i| {
                let dx = delta[(0, i, 0)] as f64;
                let dy = delta[(0, i, 1)] as f64;
                (dx * dx + dy * dy).sqrt()
            })
            .collect();
        norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(torch_quantile_50(&norms))
    }

    /// Milestone M5: fold every buffered [`Self::push_imu`] sample with
    /// `timestamp <= frame_timestamp` into a fresh
    /// [`ImuPreintegrator`], and — if the graph already has a previous
    /// committed frame and at least one sample was actually integrated —
    /// bank the result into [`Self::imu_deltas_by_arrival`], keyed by the
    /// two frames' stable `arrival_index` pair. No-op if `config.imu` is
    /// `None`.
    ///
    /// # The sub-sample boundary fragment is deliberately dropped
    ///
    /// This integrates every consecutive *sample* pair up to and including
    /// the last sample at or before `frame_timestamp`, then advances
    /// [`Self::last_imu_boundary_timestamp`] to `frame_timestamp` itself —
    /// meaning the tiny interval between that last sample and the actual
    /// frame timestamp (at most one IMU sample period, e.g. ~5 ms at
    /// EuRoC's 200 Hz IMU rate against a ~50-100 ms camera frame gap) is
    /// never integrated. This is a deliberate, bounded simplification
    /// (not a bug): its own worst-case error is a small fraction of one
    /// sample period's contribution to a multi-sample window, well inside
    /// the noise this factor's own covariance already accounts for.
    fn integrate_imu_for_new_frame(&mut self, frame_timestamp: f64) {
        let Some(imu_cfg) = self.config.imu.clone() else { return };
        let mut integrator =
            ImuPreintegrator::new_with_bias_and_noise(self.imu_bias_gyro, self.imu_bias_accel, imu_cfg.noise)
                .unwrap_or_else(|| ImuPreintegrator::new_with_bias(self.imu_bias_gyro, self.imu_bias_accel));

        let mut last_ts = self.last_imu_boundary_timestamp;
        let mut integrated_any = false;
        while let Some(&(ts, _, _)) = self.pending_imu.front() {
            if ts > frame_timestamp {
                break;
            }
            let (ts, gyro, accel) = self.pending_imu.pop_front().expect("front() just matched Some");
            if let Some(prev) = last_ts {
                let dt = ts - prev;
                if dt > 0.0 {
                    integrator.integrate_sample(gyro, accel, dt);
                    integrated_any = true;
                }
            }
            last_ts = Some(ts);
        }
        self.last_imu_boundary_timestamp = Some(frame_timestamp);

        if !integrated_any {
            return;
        }
        let n = self.graph.n_frames();
        if n < 2 {
            return;
        }
        let from_arrival = self.graph.frames()[n - 2].arrival_index;
        let to_arrival = self.graph.frames()[n - 1].arrival_index;
        let delta = integrator.delta();
        self.imu_deltas_by_arrival.insert(
            (from_arrival, to_arrival),
            ImuPreintegrationFactor {
                keyframe_id_from: from_arrival as u64,
                keyframe_id_to: to_arrival as u64,
                delta: delta.clone(),
                // Placeholder — overwritten from `self.imu_gravity_world`
                // by every reader (`try_imu_bootstrap`'s own factor list
                // doesn't read this field at all; `update_step`'s
                // `DpvoViWindow` construction fills in the real value).
                gravity_world: Vector3::zeros(),
                weight_rotation: 1.0,
                weight_velocity: 1.0,
                weight_position: 1.0,
            },
        );

        // Bootstrap-only snapshot — see `imu_bootstrap_history`'s field doc
        // for why this must NOT be re-derived from the live graph later.
        if !self.imu_bootstrapped {
            let pose_from = self.graph.frames()[n - 2].pose.clone();
            let pose_to = self.graph.frames()[n - 1].pose.clone();
            self.imu_bootstrap_history.push((from_arrival, to_arrival, pose_from, pose_to, delta));
            if self.imu_bootstrap_history.len() > IMU_BOOTSTRAP_HISTORY_CAP {
                let overflow = self.imu_bootstrap_history.len() - IMU_BOOTSTRAP_HISTORY_CAP;
                self.imu_bootstrap_history.drain(0..overflow);
            }
        }
    }

    /// Milestone M5: drop banked IMU deltas that can no longer be reached
    /// by any live frame (memory hygiene only — a stale entry is otherwise
    /// harmless, just never looked up again once its frames have aged out
    /// of the graph). Cheap: `arrival_index` is monotonically increasing,
    /// so a single comparison against the oldest live frame suffices.
    fn prune_stale_imu_deltas(&mut self) {
        let Some(oldest_live) = self.graph.frames().first().map(|f| f.arrival_index) else { return };
        self.imu_deltas_by_arrival.retain(|&(_, to), _| to >= oldest_live);
    }

    /// Milestone M5b's bootstrap chain: gyro-bias estimate (rotation-only,
    /// gated — see below), then `crate::dpvo_vi_ba::estimate_mono_vi_alignment`
    /// (gated on its own three-stage observability check — see that
    /// function's module-doc section), run against
    /// [`Self::imu_bootstrap_history`]'s pose SNAPSHOTS treated as fixed.
    /// No-op once [`Self::imu_bootstrapped`] is already `true`, or if
    /// `config.imu` is `None`. Unlike M5, a REJECTED attempt here does not
    /// consume or corrupt any state — every gyro-bias/gravity/velocity/scale
    /// candidate is discarded wholesale on any gate failure, and
    /// [`Self::imu_bootstrap_history`] keeps growing (bounded by
    /// [`IMU_BOOTSTRAP_HISTORY_CAP`]) for a later attempt with more evidence.
    ///
    /// # Why history snapshots, not the live graph (a real bug M5 fixed)
    ///
    /// An earlier version of this method built its `VisualMap`/factor list
    /// directly from `self.graph.frames()` — the graph's CURRENT live
    /// window. On a real EuRoC run this bootstrap never fired at all past
    /// the initial burst: `DpvoPatchGraph::keyframe`'s motion-magnitude
    /// gate folds away low-motion frames (MH_01's opening seconds are close
    /// to stationary) faster than a handful of usable factors could ever
    /// accumulate against a live-frames-only view — every fold silently
    /// invalidated one or two already-banked deltas whose endpoint had just
    /// left the live set, even though the delta itself was still perfectly
    /// good evidence. [`Self::imu_bootstrap_history`] decouples bootstrap
    /// evidence accumulation from the BA window's own churn entirely.
    ///
    /// # Milestone M5b's gyro-bias gate (M5's own missing piece)
    ///
    /// `estimate_gyro_bias` is reused UNCHANGED (rotation-only alignment is
    /// genuinely scale-invariant — see `crate::dpvo_vi_ba`'s module doc,
    /// "Sequencing" section), but its result is no longer accepted
    /// unconditionally: [`gyro_bootstrap_gate_check`] additionally requires
    /// the recovered bias to be plausibly small
    /// ([`DpvoImuConfig::max_gyro_bias_magnitude_rad_s`]) AND for the
    /// rotation-residual RMS to have both dropped under an absolute bound
    /// AND shrunk by a minimum fraction from its pre-alignment value
    /// ([`DpvoImuConfig::gyro_bias_max_rms_after`]/`gyro_bias_max_rms_fraction`).
    /// Failing either check means the window's rotation evidence is still
    /// too noisy to trust — this method does NOT fix a bias in that case
    /// (M5's own "poisoned forever" mechanism), it simply returns and tries
    /// again on a later frame's bigger `imu_bootstrap_history`.
    fn try_imu_bootstrap(&mut self) {
        if self.imu_bootstrapped {
            return;
        }
        let Some(imu_cfg) = self.config.imu.clone() else { return };
        if self.imu_bootstrap_history.len() < imu_cfg.min_bootstrap_factors {
            return;
        }
        self.imu_bootstrap_attempts += 1;

        // Local (0..num_unique) index <-> arrival-id mapping, plus
        // first-seen pose snapshots — the SAME frozen-per-arrival-id
        // construction M5 already used for the (still-needed)
        // `VisualMap`/`estimate_gyro_bias` call, reused here to ALSO build
        // a plain window-local `Vec<SE3>`/`Vec<DpvoImuFactor>` for
        // `estimate_mono_vi_alignment` (which — unlike the metric
        // estimators — reads DPVO's own poses directly, no `VisualMap`
        // needed; see that function's own doc).
        let mut arrival_id_set: HashSet<usize> = HashSet::new();
        for &(from, to, ..) in &self.imu_bootstrap_history {
            arrival_id_set.insert(from);
            arrival_id_set.insert(to);
        }
        let mut arrival_ids_sorted: Vec<usize> = arrival_id_set.into_iter().collect();
        arrival_ids_sorted.sort_unstable();
        let local_index: HashMap<usize, usize> =
            arrival_ids_sorted.iter().enumerate().map(|(idx, &id)| (id, idx)).collect();
        let arrival_ids: Vec<u64> = arrival_ids_sorted.iter().map(|&id| id as u64).collect();

        let mut map = VisualMap::new();
        let mut local_poses: Vec<SE3> = vec![SE3::identity(); arrival_ids_sorted.len()];
        for &(from, to, ref pose_from, ref pose_to, _) in &self.imu_bootstrap_history {
            for (id, pose) in [(from, pose_from), (to, pose_to)] {
                if map.keyframes.contains_key(&(id as u64)) {
                    continue;
                }
                let body = imu_cfg.body_to_camera.compose(pose);
                let mut frame = Frame::new(id as u64, 0);
                frame.pose = Some(Pose { world_to_camera: body });
                map.keyframes.insert(id as u64, Keyframe { frame, observations: Vec::new() });
                local_poses[local_index[&id]] = pose.clone();
            }
        }

        let factors: Vec<ImuPreintegrationFactor> = self
            .imu_bootstrap_history
            .iter()
            .map(|&(from, to, _, _, ref delta)| ImuPreintegrationFactor {
                keyframe_id_from: from as u64,
                keyframe_id_to: to as u64,
                delta: delta.clone(),
                gravity_world: Vector3::zeros(), // unused by either estimator below.
                weight_rotation: 1.0,
                weight_velocity: 1.0,
                weight_position: 1.0,
            })
            .collect();
        let local_factors: Vec<DpvoImuFactor> = self
            .imu_bootstrap_history
            .iter()
            .map(|&(from, to, _, _, ref delta)| DpvoImuFactor {
                i: local_index[&from],
                j: local_index[&to],
                factor: ImuPreintegrationFactor {
                    keyframe_id_from: from as u64,
                    keyframe_id_to: to as u64,
                    delta: delta.clone(),
                    gravity_world: Vector3::zeros(),
                    weight_rotation: 1.0,
                    weight_velocity: 1.0,
                    weight_position: 1.0,
                },
            })
            .collect();

        // ---- Stage 1: gyro bias (rotation-only, scale-invariant) ----
        let Some(gyro_bias) = estimate_gyro_bias(&map, &arrival_ids, &factors, self.imu_bias_gyro) else {
            self.imu_bootstrap_rejections += 1;
            self.imu_rejection_counts.gyro_estimator_none += 1;
            self.imu_last_rejection = Some(DpvoImuRejectionDetail::GyroEstimatorNone);
            return;
        };
        match gyro_bootstrap_gate_check(&gyro_bias, &imu_cfg) {
            Ok(()) => {}
            Err(reason) => {
                // Do NOT fix a bias yet — see this method's own doc,
                // "Milestone M5b's gyro-bias gate".
                self.imu_bootstrap_rejections += 1;
                match reason {
                    GyroGateRejection::MagnitudeTooLarge => self.imu_rejection_counts.gyro_magnitude += 1,
                    GyroGateRejection::RmsAboveAbsoluteBound => self.imu_rejection_counts.gyro_rms_absolute += 1,
                    GyroGateRejection::RmsNotEnoughImprovement => {
                        self.imu_rejection_counts.gyro_rms_fraction += 1
                    }
                }
                self.imu_last_rejection = Some(DpvoImuRejectionDetail::GyroGate {
                    reason,
                    bias_norm: gyro_bias.bias_gyro.norm(),
                    rms_before: gyro_bias.rotation_residual_rms_before,
                    rms_after: gyro_bias.rotation_residual_rms_after,
                });
                return;
            }
        }

        // ---- Stage 2: monocular-aware scale/gravity/velocity alignment ----
        let gates = DpvoMonoViAlignmentGates {
            expected_gravity_magnitude: imu_cfg.gravity_magnitude,
            gravity_norm_deviation_ratio: imu_cfg.gravity_norm_deviation_ratio,
            min_scale: imu_cfg.min_mono_scale,
            max_scale: imu_cfg.max_mono_scale,
            max_condition_number: imu_cfg.max_mono_alignment_condition_number,
        };
        let alignment = match estimate_mono_vi_alignment(
            &local_poses,
            &local_factors,
            &imu_cfg.body_to_camera,
            gyro_bias.bias_gyro,
            self.imu_bias_accel,
            &gates,
        ) {
            Ok(alignment) => alignment,
            Err(reason) => {
                self.imu_bootstrap_rejections += 1;
                match reason {
                    DpvoMonoViAlignmentRejection::NotEnoughFactors => {
                        self.imu_rejection_counts.mono_not_enough_factors += 1
                    }
                    DpvoMonoViAlignmentRejection::Underdetermined { .. } => {
                        self.imu_rejection_counts.mono_underdetermined += 1
                    }
                    DpvoMonoViAlignmentRejection::IllConditioned { .. } => {
                        self.imu_rejection_counts.mono_ill_conditioned += 1
                    }
                    DpvoMonoViAlignmentRejection::DegenerateSolve => {
                        self.imu_rejection_counts.mono_degenerate_solve += 1
                    }
                    DpvoMonoViAlignmentRejection::GravityNormDeviation { .. } => {
                        self.imu_rejection_counts.mono_gravity_norm += 1
                    }
                    DpvoMonoViAlignmentRejection::ScaleOutOfRange { .. } => {
                        self.imu_rejection_counts.mono_scale_range += 1
                    }
                }
                self.imu_last_rejection = Some(DpvoImuRejectionDetail::MonoGate(reason));
                return;
            }
        };

        // ---- Both gates passed: commit the gyro bias, apply the recovered
        // scale to the LIVE window, seed gravity/velocities, enable coupling ----
        self.imu_bias_gyro = gyro_bias.bias_gyro;
        self.imu_gravity_world = Some(alignment.gravity_world);
        self.recovered_mono_scale = Some(alignment.scale);

        // See `crate::dpvo_vi_ba`'s module doc, "Applying the recovered
        // scale", for the translation/inverse-depth transformation derivation.
        let s = alignment.scale;
        for frame in self.graph.frames_mut() {
            frame.pose.translation *= s;
        }
        for patch in self.graph.patches_mut() {
            patch.inverse_depth /= s;
        }

        // Seed velocities for every CURRENTLY LIVE frame the alignment
        // covers (frames the alignment used that have since aged out of
        // the live graph simply have no velocity slot left to seed).
        for (local, f) in self.graph.frames().iter().enumerate() {
            if let Some(&window_local) = local_index.get(&f.arrival_index) {
                if let Some(&v) = alignment.velocities.get(window_local) {
                    self.velocities[local] = v;
                }
            }
        }
        self.imu_bootstrapped = true;
        self.imu_consecutive_bad_frames = 0;
        // No longer needed once bootstrapped — release the memory rather
        // than let it sit (a later rollback re-grows it from scratch).
        self.imu_bootstrap_history.clear();
        self.imu_bootstrap_history.shrink_to_fit();
    }

    /// Milestone M5b rollback: un-bootstrap back to visual-only. Does NOT
    /// attempt to undo the scale already baked into every live pose/patch
    /// translation/inverse-depth by [`Self::try_imu_bootstrap`] — harmless,
    /// since visual-only reprojection residuals are scale-invariant (exactly
    /// the gauge freedom `dpvo_ba` already tolerates on every M4/M4-perf
    /// run; the run simply resumes accumulating its own ordinary monocular
    /// scale drift from wherever it happened to be, rather than staying
    /// frozen against a since-discredited bootstrap). Clears every piece of
    /// bootstrap-only/coupling state so a later re-bootstrap starts from a
    /// clean slate rather than replaying stale, possibly already-poisoned
    /// evidence — see the module doc's "Milestone M5's honest negative, and
    /// what M5b changes" section.
    fn rollback_imu_bootstrap(&mut self) {
        self.imu_bootstrapped = false;
        self.imu_gravity_world = None;
        self.imu_bias_gyro = Vector3::zeros();
        self.imu_bias_accel = Vector3::zeros();
        self.imu_deltas_by_arrival.clear();
        self.imu_bootstrap_history.clear();
        for v in &mut self.velocities {
            *v = Vector3::zeros();
        }
        self.imu_consecutive_bad_frames = 0;
        self.imu_rollback_count += 1;
    }

    /// Milestone M6: `dpvo.py:449-455`'s own loop-closure call site —
    ///
    /// ```python
    /// if self.cfg.LOOP_CLOSURE:
    ///     if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
    ///         lii, ljj = self.pg.edges_loop()
    ///         if lii.numel() > 0:
    ///             self.last_global_ba = self.n
    ///             self.append_factors(lii, ljj)
    /// ```
    ///
    /// ported exactly, including a subtlety easy to miss on a first read:
    /// `last_global_ba` (this method's [`Self::last_loop_batch_frame`]) is
    /// only updated on a batch that actually found something — a batch that
    /// runs `find_loop_edges` and comes up empty does NOT push the next
    /// eligible attempt out by `GLOBAL_OPT_FREQ` frames, so (matching
    /// upstream) this can fire on every single subsequent committed frame
    /// until the first successful batch, not just every `GLOBAL_OPT_FREQ`
    /// frames from the start. This is cheap even at that cadence — the
    /// candidate search is pure `flow_mag`/`reprojected_center_depth`
    /// arithmetic over a bounded `(i, j)` grid, no ONNX/correlation call
    /// (see `crate::dpvo_loop_closure`'s module doc).
    ///
    /// No-op if `config.loop_closure` is `None`. Any accepted edges are
    /// appended directly onto the live patch graph (`DpvoPatchGraph::append_edges`)
    /// — the *effect* of a new loop batch (whether it actually moves any
    /// pose) only shows up later, inside [`Self::update_step`]'s own BA
    /// solve, once the edge's target frame ages into the free
    /// `optimization_window` (a loop edge's source frame is typically
    /// `fixedp`-excluded — an anchor, never itself solved for — see
    /// [`Self::update_step`]'s own "Milestone M6" correction-magnitude
    /// tracking for where the observable effect is actually measured).
    fn try_loop_closure(&mut self) {
        let Some(lc_cfg) = self.config.loop_closure else { return };
        let n = self.graph.n_frames();
        let due = match self.last_loop_batch_frame {
            None => true,
            Some(last) => n.saturating_sub(last) >= lc_cfg.global_opt_freq,
        };
        if !due {
            return;
        }

        let (candidates_evaluated, accepted) = find_loop_edges(&self.graph, &lc_cfg);
        self.loop_batches_attempted += 1;
        self.loop_candidates_evaluated_total += candidates_evaluated;
        if accepted.is_empty() {
            return;
        }

        self.last_loop_batch_frame = Some(n);
        self.loop_accepted_total += accepted.len();
        self.loop_last_batch_accepted = accepted.len();
        let patch_edges = expand_frame_pairs_to_patch_edges(&accepted, self.graph.config().patches_per_frame);
        self.loop_patch_edges_added_total += patch_edges.len();
        self.graph.append_edges(&patch_edges, DIM);
    }

    /// Milestone M6: record one `update_step` call's own correction-magnitude
    /// sample — see that method's own "Milestone M6" section for how
    /// `magnitude_m` (the max pose-translation delta across the whole BA
    /// window, before vs. after that one solve) is computed and why it is
    /// only sampled on calls where at least one active edge is a loop edge.
    fn record_loop_correction(&mut self, magnitude_m: f64) {
        self.loop_correction_events += 1;
        self.loop_correction_sum_m += magnitude_m;
        self.loop_correction_max_m = self.loop_correction_max_m.max(magnitude_m);
    }

    /// Milestone M6: dispatch to `DpvoPatchGraph::keyframe_with_loop_protection`
    /// (loop closure enabled — see that method's doc for the exemption it
    /// applies) or the plain `DpvoPatchGraph::keyframe` (loop closure
    /// disabled, byte-for-byte M4/M4-perf/M5/M5b behavior).
    fn keyframe_dispatch(&mut self) -> Option<usize> {
        if self.config.loop_closure.is_some() {
            let optimization_window = self.graph.config().optimization_window;
            self.graph.keyframe_with_loop_protection(optimization_window)
        } else {
            self.graph.keyframe()
        }
    }

    /// One `update()` call (`dpvo.py:328-360`): reproject every active
    /// edge's patch grid, assemble the 2-pyramid-level correlation tensor
    /// (grouped by target frame — see the module doc's windowing/`corr_cpu`
    /// notes), run the GRU update cell, then a windowed [`dpvo_ba`] call
    /// (or, once Milestone M5's IMU bootstrap has succeeded, the
    /// IMU-coupled [`dpvo_vi_ba`] instead — see this module's doc, "IMU
    /// coupling").
    ///
    /// # Milestone M6: correction-magnitude sampling
    ///
    /// Whenever loop closure is enabled and at least one currently active
    /// edge is itself a loop edge, this call snapshots every pose in the
    /// window before the solve and diffs against the solve's own output,
    /// recording the largest translation delta via
    /// [`Self::record_loop_correction`] — see the snapshot block's own doc
    /// (right before `DpvoBaProblem` is built) for why the whole window is
    /// sampled rather than just the loop edge's two endpoints.
    fn update_step(&mut self) -> Result<(), DpvoOdometryError> {
        let n = self.graph.n_frames();
        let removal_window = self.graph.config().removal_window;
        let patch_lifetime = self.graph.config().patch_lifetime;
        let optimization_window = self.graph.config().optimization_window;
        let patches_per_frame = self.graph.config().patches_per_frame;
        let mut frame_lo = n.saturating_sub(removal_window + patch_lifetime);

        let edges = self.graph.edges().to_vec();
        if edges.is_empty() {
            return Ok(());
        }
        let e_count = edges.len();

        // Milestone M6: widen `frame_lo` to cover every currently active
        // edge's endpoints exactly, generalizing the M4 derivation above
        // (a `debug_assert`-only check back then) to tolerate a
        // proximity/loop-closure edge whose source frame is older than the
        // ordinary `removal_window + patch_lifetime` bound — see the module
        // doc's "Windowing the BA problem" section for why widening only
        // ever adds *fixed* poses to the window, never changes the free
        // pose count. A strict no-op whenever no edge is older than the
        // formula above already covers (i.e. every M4/M4-perf/M5/M5b run,
        // and every M6 run before its first successful loop batch).
        if let Some(min_edge_frame) = edges.iter().map(|e| e.i.min(e.j)).min() {
            frame_lo = frame_lo.min(min_edge_frame);
        }

        let mut by_target: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, edge) in edges.iter().enumerate() {
            by_target.entry(edge.j).or_default().push(idx);
        }

        let mut coords_center = vec![Vector2::new(0.0_f64, 0.0_f64); e_count];
        let mut corr_flat = Array2::<f32>::zeros((e_count, CORR_DIM));

        let corr_start = Instant::now();
        for (j, idxs) in &by_target {
            let target_pyramid = self.frame_pyramids[*j].clone();
            let group_len = idxs.len();
            let mut anchor_gmap = Array4::<f32>::zeros((group_len, FNET_DIM, PATCH, PATCH));
            let mut coords_grid_px = Array4::<f32>::zeros((group_len, PATCH, PATCH, 2));
            for (local, &idx) in idxs.iter().enumerate() {
                let edge = &edges[idx];
                let pose_i = self.graph.frames()[edge.i].pose.clone();
                let pose_j = self.graph.frames()[edge.j].pose.clone();
                let intr_i = self.graph.frames()[edge.i].intrinsics;
                let intr_j = self.graph.frames()[edge.j].intrinsics;
                let patch = self.graph.patches()[edge.k];
                let grid = reproject_patch_grid(&pose_i, &pose_j, &intr_i, &intr_j, &patch);
                coords_center[idx] = grid[1][1];
                anchor_gmap.index_axis_mut(Axis(0), local).assign(&self.patch_gmap[edge.k]);
                for py in 0..PATCH {
                    for px in 0..PATCH {
                        coords_grid_px[(local, py, px, 0)] = grid[py][px].x as f32;
                        coords_grid_px[(local, py, px, 1)] = grid[py][px].y as f32;
                    }
                }
            }
            let group_corr = corr_pyramid(
                anchor_gmap.view(),
                coords_grid_px.view(),
                &target_pyramid.level0,
                &target_pyramid.level1,
            );
            for (local, &idx) in idxs.iter().enumerate() {
                corr_flat.row_mut(idx).assign(&group_corr.row(local));
            }
        }
        self.stats.correlation_ms_total += corr_start.elapsed().as_secs_f64() * 1000.0;

        let mut net_arr = Array3::<f32>::zeros((1, e_count, DIM));
        let mut inp_arr = Array3::<f32>::zeros((1, e_count, DIM));
        let mut kk = Vec::with_capacity(e_count);
        let mut ii = Vec::with_capacity(e_count);
        let mut jj = Vec::with_capacity(e_count);
        for (idx, edge) in edges.iter().enumerate() {
            net_arr
                .index_axis_mut(Axis(0), 0)
                .index_axis_mut(Axis(0), idx)
                .assign(&Array1::from_vec(edge.net.clone()));
            inp_arr.index_axis_mut(Axis(0), 0).index_axis_mut(Axis(0), idx).assign(&self.patch_imap[edge.k]);
            kk.push(edge.k as i64);
            ii.push(edge.i as i64);
            jj.push(edge.j as i64);
        }
        let corr3 = corr_flat.insert_axis(Axis(0));

        let update_start = Instant::now();
        let (net_out, delta, weight) = self
            .session
            .update_iteration(net_arr.view(), inp_arr.view(), corr3.view(), &kk, &ii, &jj, &self.agg_kk, &self.agg_ij)?;
        self.stats.update_ms_total += update_start.elapsed().as_secs_f64() * 1000.0;

        let mut targets = Vec::with_capacity(e_count);
        let mut weights = Vec::with_capacity(e_count);
        for idx in 0..e_count {
            let net_row: Vec<f32> = net_out.index_axis(Axis(0), 0).index_axis(Axis(0), idx).to_owned().into_raw_vec_and_offset().0;
            self.graph.edges_mut()[idx].net = net_row;
            let dx = delta[(0, idx, 0)] as f64;
            let dy = delta[(0, idx, 1)] as f64;
            let target = Vector2::new(coords_center[idx].x + dx, coords_center[idx].y + dy);
            let w = Vector2::new(weight[(0, idx, 0)] as f64, weight[(0, idx, 1)] as f64);
            self.graph.edges_mut()[idx].target_weight = Some((target, w));
            targets.push(target);
            weights.push(w);
        }

        // See the module doc's windowing derivation for why [frame_lo, n)
        // is guaranteed to cover every edge referenced below.
        let global_fixedp =
            if self.graph.is_initialized() { n.saturating_sub(optimization_window).max(1) } else { 1 };
        let local_fixedp = global_fixedp.saturating_sub(frame_lo);
        let patches_lo = frame_lo * patches_per_frame;

        debug_assert!(
            edges.iter().all(|e| e.i >= frame_lo && e.j >= frame_lo && e.k >= patches_lo),
            "update_step's window [frame_lo,n) did not cover every active edge — Milestone M6's \
             own min-over-edges widening above should make this unconditionally true; a failure \
             here means that widening itself has a bug, not just a loose bound \
             (removal_window={removal_window}, patch_lifetime={patch_lifetime})"
        );

        let window_poses: Vec<SE3> = self.graph.frames()[frame_lo..n].iter().map(|f| f.pose.clone()).collect();
        let window_intr: Vec<DpvoIntrinsics> = self.graph.frames()[frame_lo..n].iter().map(|f| f.intrinsics).collect();
        let window_patches: Vec<DpvoPatch> = self.graph.patches()[patches_lo..].to_vec();
        let ba_edges: Vec<DpvoEdge> = edges
            .iter()
            .map(|e| DpvoEdge { i: e.i - frame_lo, j: e.j - frame_lo, k: e.k - patches_lo })
            .collect();

        // `net.py:220`'s own BA call site bounds: image extent (in `fnet`
        // stride-4 space) padded by 64px — see the module doc's bounds note.
        let ws = self.config.width as f64 / RES as f64;
        let hs = self.config.height as f64 / RES as f64;
        let bounds = [-64.0, -64.0, ws + 64.0, hs + 64.0];

        // Milestone M6: snapshot the pre-solve window poses for the
        // loop-correction-magnitude diagnostic, but ONLY when loop closure
        // is enabled AND at least one currently active edge is itself a
        // loop edge (temporal gap `j - i` exceeding `UPSTREAM_MIN_LOOP_GAP`
        // — the same criterion `DpvoPatchGraph::keyframe_with_loop_protection`
        // uses to recognize one). Sampling the WHOLE window rather than just
        // the loop edge's own two endpoints is deliberate: its *source*
        // frame `i` is very often `fixedp`-excluded (an anchor outside
        // `optimization_window`, never itself solved for), so the
        // observable correction — if any — shows up on whichever pose in
        // the FREE `[global_fixedp, n)` range the edge's target `j` (or a
        // frame chained to it through ordinary temporal edges) happens to
        // pull on, not necessarily at the edge's own two endpoints.
        let loop_correction_pre_solve: Option<Vec<SE3>> = self.config.loop_closure.as_ref().and_then(|_| {
            edges
                .iter()
                .any(|e| e.j.saturating_sub(e.i) > UPSTREAM_MIN_LOOP_GAP)
                .then(|| window_poses.clone())
        });

        let problem = DpvoBaProblem {
            poses: window_poses,
            patches: window_patches,
            intrinsics: window_intr,
            edges: ba_edges,
            targets,
            weights,
        };
        let ba_config = DpvoBaConfig {
            iterations: 2,
            fixedp: local_fixedp,
            lmbda: self.config.ba_lmbda,
            ep: self.config.ba_ep,
            bounds,
        };

        let ba_start = Instant::now();
        // Milestone M7 takes priority over M5/M5b's one-shot bootstrap when
        // enabled (see `Self::scale_coupling_step`'s own doc) — checked
        // FIRST so the M5/M5b branch below is completely unreachable, not
        // merely unused, whenever `config.imu.scale_coupling` is `Some`.
        let use_scale_coupling = self.config.imu.as_ref().is_some_and(|c| c.scale_coupling.is_some());
        // Milestone M5: once the IMU bootstrap chain has succeeded, couple
        // consecutive-window IMU factors into the SAME Gauss-Newton solve
        // (`crate::dpvo_vi_ba::dpvo_vi_ba`) instead of the plain visual-only
        // `dpvo_ba` — see this module's own doc, "IMU coupling", and
        // `crate::dpvo_vi_ba`'s module doc for the math. Falls back to the
        // unmodified M4 path whenever `config.imu` is `None` or the
        // bootstrap has not (yet) succeeded — visual-only behavior is
        // therefore byte-for-byte unchanged from M4 in both of those cases.
        let (new_poses, new_patches, new_velocities) = if use_scale_coupling {
            self.scale_coupling_step(&problem, frame_lo, n, local_fixedp, &ba_config)?
        } else if self.imu_bootstrapped {
            let imu_cfg = self
                .config
                .imu
                .clone()
                .expect("imu_bootstrapped can only be true when config.imu is Some — set together in try_imu_bootstrap");
            let window_arrivals: Vec<usize> =
                self.graph.frames()[frame_lo..n].iter().map(|f| f.arrival_index).collect();
            let mut imu_factors = Vec::new();
            for local in 0..window_arrivals.len().saturating_sub(1) {
                let key = (window_arrivals[local], window_arrivals[local + 1]);
                if let Some(banked) = self.imu_deltas_by_arrival.get(&key) {
                    let mut factor = banked.clone();
                    factor.gravity_world = self
                        .imu_gravity_world
                        .expect("imu_bootstrapped implies imu_gravity_world is Some");
                    imu_factors.push(DpvoImuFactor { i: local, j: local + 1, factor });
                }
            }
            let imu_window = DpvoViWindow {
                velocities: self.velocities[frame_lo..n].to_vec(),
                factors: imu_factors,
                body_to_camera: imu_cfg.body_to_camera,
                bias_gyro: self.imu_bias_gyro,
                bias_accel: self.imu_bias_accel,
            };
            let solved = dpvo_vi_ba(&problem, &imu_window, &ba_config)?;

            // Milestone M5b rollback monitor (module doc, "Milestone M5's
            // honest negative, and what M5b changes"): mean whitened
            // IMU-factor NIS at the just-solved state. A persistently
            // pathological value across `rollback_consecutive_frames`
            // frames means this bootstrap's own scale/gravity/bias is
            // fighting its own IMU evidence — exactly the "poisoned
            // forever" failure M5 had no way to recover from.
            let mut nis_sum = 0.0_f64;
            let mut nis_count = 0usize;
            for f in &imu_window.factors {
                nis_sum += imu_factor_nis(
                    &solved.poses[f.i],
                    &solved.poses[f.j],
                    &solved.velocities[f.i],
                    &solved.velocities[f.j],
                    &imu_window.body_to_camera,
                    &f.factor,
                    &self.imu_bias_gyro,
                    &self.imu_bias_accel,
                );
                nis_count += 1;
            }
            let mean_nis = if nis_count > 0 { nis_sum / nis_count as f64 } else { 0.0 };
            let (next_bad, should_rollback) = rollback_monitor_step(
                mean_nis,
                imu_cfg.rollback_mean_nis_bound,
                self.imu_consecutive_bad_frames,
                imu_cfg.rollback_consecutive_frames,
            );
            self.imu_consecutive_bad_frames = next_bad;

            let out = (solved.poses, solved.patches, Some(solved.velocities));
            if should_rollback {
                self.rollback_imu_bootstrap();
            }
            out
        } else {
            let solved = dpvo_ba(&problem, &ba_config)?;
            (solved.poses, solved.patches, None)
        };
        self.stats.ba_ms_total += ba_start.elapsed().as_secs_f64() * 1000.0;

        // Milestone M6: complete the loop-correction-magnitude sample this
        // call started above (see that block's own doc for what/why).
        if let Some(pre_solve) = loop_correction_pre_solve {
            let magnitude = pre_solve
                .iter()
                .zip(new_poses.iter())
                .map(|(before, after)| (after.translation - before.translation).norm())
                .fold(0.0_f64, f64::max);
            self.record_loop_correction(magnitude);
        }

        for (local, pose) in new_poses.into_iter().enumerate() {
            self.graph.frames_mut()[frame_lo + local].pose = pose;
        }
        for (local, patch) in new_patches.into_iter().enumerate() {
            self.graph.patches_mut()[patches_lo + local] = patch;
        }
        // Milestone M5b: if this very frame's rollback monitor just fired
        // (`self.imu_bootstrapped` flipped back to `false` above),
        // `rollback_imu_bootstrap` already zeroed every velocity slot — do
        // NOT immediately overwrite that with the (possibly still-poisoned)
        // solve's own velocities. Milestone M7's own path never sets
        // `self.imu_bootstrapped` at all (see `Self::scale_coupling_step`'s
        // doc), so it is included here explicitly rather than relying on
        // that flag.
        if let Some(velocities) = new_velocities.filter(|_| self.imu_bootstrapped || use_scale_coupling) {
            for (local, v) in velocities.into_iter().enumerate() {
                self.velocities[frame_lo + local] = v;
            }
        }
        Ok(())
    }

    /// Milestone M7 (`docs/dpvo_droid_port_plan.md`): the continuous,
    /// uncertainty-weighted scale-coupling per-frame step — see
    /// `crate::dpvo_scale_coupling`'s module doc for the full design this
    /// implements. Called from [`Self::update_step`] in place of the
    /// M5/M5b branch whenever `config.imu.scale_coupling` is `Some`.
    /// Returns `(poses, patches, velocities)` in the SAME window-local
    /// indexing `problem`/`dpvo_ba`/`dpvo_vi_ba` already use.
    ///
    /// # Why this never touches `self.imu_bootstrapped`/`self.imu_bias_gyro`
    /// as a COMMITTED, staged value
    ///
    /// Unlike M5b's `try_imu_bootstrap`, this method re-derives its own
    /// gyro-bias/scale evidence EVERY call from the current window — nothing
    /// here is a one-shot "compute once, fix forever" decision, so there is
    /// no analogous boolean to flip. `self.imu_bias_gyro` is still updated
    /// (purely for [`DpvoOdometry::imu_diagnostics`]'s own echo — a caller
    /// inspecting that struct still wants to see SOME current bias value),
    /// but nothing downstream treats it as authoritative the way M5b's
    /// `dpvo_vi_ba` call site does.
    fn scale_coupling_step(
        &mut self,
        problem: &DpvoBaProblem,
        frame_lo: usize,
        n: usize,
        local_fixedp: usize,
        ba_config: &DpvoBaConfig,
    ) -> Result<ScaleCouplingSolution, DpvoOdometryError> {
        let imu_cfg = self
            .config
            .imu
            .clone()
            .expect("scale_coupling_step is only called when config.imu.scale_coupling is Some");
        let sc_cfg = imu_cfg
            .scale_coupling
            .expect("scale_coupling_step is only called when config.imu.scale_coupling is Some");

        // The visual-only solve is ALWAYS computed — it is both the cheap
        // fallback (weight == 0, or an under-evidenced window) and one of
        // `blend_solutions`'s two endpoints even once coupling is active
        // (module doc, "Why output-space blending").
        let visual_solved = dpvo_ba(problem, ba_config)?;

        // Build this window's usable IMU factors (window-local indexing,
        // matching `problem.poses`) — the SAME construction M5/M5b's own
        // branch above uses, reused here rather than shared as a helper
        // because the two branches' surrounding bookkeeping (arrival-id
        // mapping vs. plain local indices) differs enough that extracting a
        // shared function would need its own new abstraction for a single
        // ~10-line loop.
        let window_arrivals: Vec<usize> =
            self.graph.frames()[frame_lo..n].iter().map(|f| f.arrival_index).collect();
        let mut window_factors: Vec<DpvoImuFactor> = Vec::new();
        for local in 0..window_arrivals.len().saturating_sub(1) {
            let key = (window_arrivals[local], window_arrivals[local + 1]);
            if let Some(banked) = self.imu_deltas_by_arrival.get(&key) {
                window_factors.push(DpvoImuFactor { i: local, j: local + 1, factor: banked.clone() });
            }
        }

        if window_factors.len() < sc_cfg.min_window_factors {
            // Not enough evidence to even attempt a re-estimation this
            // frame — visual-only, and the annealing weight does not move
            // either direction (this is "no data yet", not "data
            // disagrees" — see `crate::dpvo_scale_coupling`'s own
            // "Convergence and annealing" section for why those two cases
            // are treated differently).
            return Ok((visual_solved.poses, visual_solved.patches, None));
        }

        // ---- Stage 1: continuous gyro-bias re-estimation (never
        // hard-fixed — module doc's "gyro bias as a soft prior"
        // requirement). Reuses `estimate_gyro_bias` UNCHANGED (rotation-only
        // alignment is scale-invariant — same reasoning
        // `crate::dpvo_vi_ba`'s own "Sequencing" section gives for M5b), but
        // called EVERY frame against the CURRENT live window instead of
        // once against a decoupled bootstrap history — the recursive
        // estimator's own robustness (not a decoupled-history mechanism) is
        // what protects this from M5's original "stationary opening
        // segment" bug, since a degenerate/rejected window here just
        // produces no measurement, not a wrong permanently-fixed bias. ----
        let mut map = VisualMap::new();
        let mut local_poses: Vec<SE3> = vec![SE3::identity(); window_arrivals.len()];
        for (idx, &arrival) in window_arrivals.iter().enumerate() {
            let pose = self.graph.frames()[frame_lo + idx].pose.clone();
            let body = imu_cfg.body_to_camera.compose(&pose);
            let mut frame = Frame::new(arrival as u64, 0);
            frame.pose = Some(Pose { world_to_camera: body });
            map.keyframes.insert(arrival as u64, Keyframe { frame, observations: Vec::new() });
            local_poses[idx] = pose;
        }
        let arrival_ids: Vec<u64> = window_arrivals.iter().map(|&a| a as u64).collect();
        let factors_for_gyro: Vec<ImuPreintegrationFactor> =
            window_factors.iter().map(|f| f.factor.clone()).collect();

        let seed_bias = self.gyro_bias_estimator.mean();
        if let Some(alignment) = estimate_gyro_bias(&map, &arrival_ids, &factors_for_gyro, seed_bias) {
            // Honest variance proxy (same "derive it from the LSQ's own
            // fit quality" philosophy as the scale estimator — see
            // `crate::dpvo_scale_coupling`'s module doc): the ROTATION
            // alignment's own converged residual RMS is the direct
            // empirical noise-level estimate for THIS measurement.
            let variance = alignment.rotation_residual_rms_after.max(1.0e-9).powi(2);
            self.gyro_bias_estimator.update(alignment.bias_gyro, variance);
        }
        let bias_gyro = self.gyro_bias_estimator.mean();
        self.imu_bias_gyro = bias_gyro; // diagnostics echo only — see this method's own doc.

        // ---- Stage 2: continuous mono-scale/gravity/velocity
        // re-estimation. ----
        let gates = DpvoMonoViAlignmentGates {
            expected_gravity_magnitude: imu_cfg.gravity_magnitude,
            gravity_norm_deviation_ratio: imu_cfg.gravity_norm_deviation_ratio,
            min_scale: imu_cfg.min_mono_scale,
            max_scale: imu_cfg.max_mono_scale,
            max_condition_number: imu_cfg.max_mono_alignment_condition_number,
        };
        let mut window_velocities = vec![Vector3::zeros(); window_arrivals.len()];
        for (idx, local) in window_velocities.iter_mut().enumerate() {
            *local = self.velocities[frame_lo + idx];
        }

        // Diagnostic finding (see the plan doc's "M7 results", "Diagnosis"
        // subsection): the live BA window `[frame_lo, n)` can contain
        // frames whose `arrival_index` is NOT consecutive between adjacent
        // window slots — `DpvoPatchGraph::keyframe`'s own motion-magnitude
        // folding (the SAME mechanism M5's "why history snapshots, not the
        // live graph" bug report already diagnosed for the one-shot
        // bootstrap) can retain two temporally-adjacent LIVE frames whose
        // banked `imu_deltas_by_arrival` delta only covers a since-folded
        // intermediate frame, not the surviving pair directly. `window_factors`
        // above already only includes a factor where a direct delta exists,
        // so such a gap simply produces one fewer factor — but
        // `estimate_mono_vi_alignment`'s own degrees-of-freedom requirement
        // (`unknowns = 3·n_poses + 4`) grows with EVERY pose regardless,
        // so a window with several such gaps can become UNDERDETERMINED for
        // a reason that has nothing to do with real motion excitation. A
        // live 300-frame diagnostic run (`E:/visloc_archive/dpvo_m7_20260717/diag_300`)
        // measured this directly: `usable_factors` plateaued at exactly `8`
        // while `n_poses` grew past `19` over ~120 consecutive frames, with
        // `Underdetermined { usable_factors: 8, n_poses: 19 }` as the
        // dominant (and, by the run's end, ONLY growing) rejection reason.
        // Fix: restrict the mono-alignment call to the maximal TRAILING run
        // of arrival-consecutive frames (ending at the newest frame in the
        // window) — the sub-window where every pose actually participates
        // in a factor, so `usable_factors == mono_poses.len() - 1` exactly,
        // the best-conditioned DOF ratio reachable from this window's own
        // data. This does not touch `window_factors`/`local_poses` used by
        // Stage 1 (gyro bias) or the later `dpvo_vi_ba` coupled solve — only
        // this call's own inputs are trimmed.
        let mono_start = trailing_consecutive_run_start(&window_arrivals);
        let mono_poses = &local_poses[mono_start..];
        let mono_factors: Vec<DpvoImuFactor> = window_factors
            .iter()
            .filter(|f| f.i >= mono_start && f.j >= mono_start)
            .map(|f| DpvoImuFactor { i: f.i - mono_start, j: f.j - mono_start, factor: f.factor.clone() })
            .collect();

        match estimate_mono_vi_alignment(
            mono_poses, &mono_factors, &imu_cfg.body_to_camera, bias_gyro, self.imu_bias_accel, &gates,
        ) {
            Ok(alignment) => {
                self.scale_coupling_measurements += 1;
                let measurement = scale_measurement_from_alignment(&alignment, &sc_cfg.scale);
                self.scale_estimator.update(measurement);
                self.scale_coupling_gravity = Some(alignment.gravity_world);
                for (idx, &v) in alignment.velocities.iter().enumerate() {
                    window_velocities[mono_start + idx] = v;
                }
            }
            Err(rejection) => {
                self.scale_coupling_measurement_rejections += 1;
                match rejection {
                    DpvoMonoViAlignmentRejection::NotEnoughFactors => {
                        self.scale_coupling_rejection_counts.not_enough_factors += 1
                    }
                    DpvoMonoViAlignmentRejection::Underdetermined { .. } => {
                        self.scale_coupling_rejection_counts.underdetermined += 1
                    }
                    DpvoMonoViAlignmentRejection::IllConditioned { .. } => {
                        self.scale_coupling_rejection_counts.ill_conditioned += 1
                    }
                    DpvoMonoViAlignmentRejection::DegenerateSolve => {
                        self.scale_coupling_rejection_counts.degenerate_solve += 1
                    }
                    DpvoMonoViAlignmentRejection::GravityNormDeviation { .. } => {
                        self.scale_coupling_rejection_counts.gravity_norm += 1
                    }
                    DpvoMonoViAlignmentRejection::ScaleOutOfRange { .. } => {
                        self.scale_coupling_rejection_counts.scale_range += 1
                    }
                }
                self.scale_coupling_last_rejection = Some(rejection);
            }
        }

        let should_increase = self.scale_estimator.is_converged();
        self.scale_coupling_weight.step(should_increase);
        let weight = self.scale_coupling_weight.value;

        if weight <= 0.0 {
            // Module doc: "at weight == 0.0 this is byte-identical to the
            // visual-only path" — no `dpvo_vi_ba` call at all, not merely a
            // zero-effect one.
            return Ok((visual_solved.poses, visual_solved.patches, None));
        }

        let gravity_world = self
            .scale_coupling_gravity
            .unwrap_or_else(|| Vector3::new(0.0, 0.0, -imu_cfg.gravity_magnitude));
        let imu_factors: Vec<DpvoImuFactor> = window_factors
            .iter()
            .map(|f| {
                let mut factor = f.factor.clone();
                factor.gravity_world = gravity_world;
                DpvoImuFactor { i: f.i, j: f.j, factor }
            })
            .collect();
        let imu_window = DpvoViWindow {
            velocities: window_velocities,
            factors: imu_factors,
            body_to_camera: imu_cfg.body_to_camera.clone(),
            bias_gyro,
            bias_accel: self.imu_bias_accel,
        };
        let coupled = dpvo_vi_ba(problem, &imu_window, ba_config)?;
        let mut imu_poses = coupled.poses;
        let mut imu_patches = coupled.patches;

        // Gentle scale-prior correction (module doc, "Gentle scale-prior
        // application") — a no-op unless the posterior already has at
        // least one measurement.
        if let Some(posterior) = self.scale_estimator.posterior() {
            apply_gentle_scale_correction(
                &mut imu_poses,
                &mut imu_patches,
                local_fixedp,
                &imu_window.factors,
                &coupled.velocities,
                &imu_window.body_to_camera,
                &bias_gyro,
                &self.imu_bias_accel,
                posterior,
                weight,
                &sc_cfg.scale,
            );
        }

        // Continuous cross-check -> SOFT rollback (module doc, "Continuous
        // cross-check and soft rollback"): decay the weight an extra step
        // and widen both posteriors' variance, rather than M5b's hard
        // un-bootstrap — no pose/depth/velocity state needs undoing, since
        // `blend_solutions` below never let the live map get MORE than
        // `weight`-far from the pure-visual solution.
        let mut nis_sum = 0.0_f64;
        let mut nis_count = 0usize;
        for f in &imu_window.factors {
            nis_sum += imu_factor_nis(
                &imu_poses[f.i],
                &imu_poses[f.j],
                &coupled.velocities[f.i],
                &coupled.velocities[f.j],
                &imu_window.body_to_camera,
                &f.factor,
                &bias_gyro,
                &self.imu_bias_accel,
            );
            nis_count += 1;
        }
        let mean_nis = if nis_count > 0 { nis_sum / nis_count as f64 } else { 0.0 };
        let (next_bad, should_soft_rollback) = rollback_monitor_step(
            mean_nis,
            sc_cfg.scale.rollback_mean_nis_bound,
            self.scale_coupling_consecutive_bad,
            sc_cfg.scale.rollback_consecutive_frames,
        );
        self.scale_coupling_consecutive_bad = next_bad;
        if should_soft_rollback {
            self.scale_estimator.soft_reset();
            self.gyro_bias_estimator.soft_reset();
            self.scale_coupling_weight.force_decay();
            self.scale_coupling_rollback_count += 1;
        }

        let (blended_poses, blended_patches) =
            blend_solutions(&visual_solved.poses, &visual_solved.patches, &imu_poses, &imu_patches, weight);
        Ok((blended_poses, blended_patches, Some(coupled.velocities)))
    }
}

/// Why [`gyro_bootstrap_gate_check`] rejected a [`GyroBiasAlignment`] —
/// checked in this order (a magnitude failure is reported even if the rms
/// checks would also have failed, so a caller tallying rejection reasons
/// gets one bucket per attempt, not a double count). `pub`, not private —
/// embedded in the `pub` [`DpvoImuRejectionDetail::GyroGate`] diagnostic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GyroGateRejection {
    /// `bias_gyro.norm() > max_gyro_bias_magnitude_rad_s`.
    MagnitudeTooLarge,
    /// `rotation_residual_rms_after` is non-finite, or exceeds
    /// `gyro_bias_max_rms_after`.
    RmsAboveAbsoluteBound,
    /// `rotation_residual_rms_after` didn't drop to at least
    /// `gyro_bias_max_rms_fraction` of `rotation_residual_rms_before`.
    RmsNotEnoughImprovement,
}

/// Milestone M5b: the gyro-bias bootstrap acceptance gate (see
/// [`DpvoOdometry::try_imu_bootstrap`]'s doc, "Milestone M5b's gyro-bias
/// gate"), factored out as a pure function of a
/// [`GyroBiasAlignment`]/[`DpvoImuConfig`] pair so it can be unit-tested
/// directly against synthetic alignment results — no ONNX session or live
/// `DpvoOdometry` required (this crate's own DPVO unit tests already draw
/// this line elsewhere, e.g. `corr_pyramid`'s standalone-function tests
/// below vs. the module's one `--ignored` real-session benchmark). Returns
/// the specific [`GyroGateRejection`] on failure, not just a bare `bool` —
/// the task's own "isolate which gate" acceptance requirement needs to be
/// answerable for THIS gate too, not only `estimate_mono_vi_alignment`'s.
fn gyro_bootstrap_gate_check(alignment: &GyroBiasAlignment, cfg: &DpvoImuConfig) -> Result<(), GyroGateRejection> {
    if alignment.bias_gyro.norm() > cfg.max_gyro_bias_magnitude_rad_s {
        return Err(GyroGateRejection::MagnitudeTooLarge);
    }
    if !alignment.rotation_residual_rms_after.is_finite()
        || alignment.rotation_residual_rms_after > cfg.gyro_bias_max_rms_after
    {
        return Err(GyroGateRejection::RmsAboveAbsoluteBound);
    }
    if alignment.rotation_residual_rms_after
        > alignment.rotation_residual_rms_before * cfg.gyro_bias_max_rms_fraction
    {
        return Err(GyroGateRejection::RmsNotEnoughImprovement);
    }
    Ok(())
}

/// Milestone M5b: the rollback monitor's pure counter/threshold decision
/// (see [`DpvoOdometry::rollback_imu_bootstrap`]'s doc) — given this
/// frame's mean IMU-factor NIS and the running consecutive-bad-frame
/// count, returns the counter's updated value and whether this frame trips
/// the rollback. Factored out of [`DpvoOdometry::update_step`] for the same
/// ONNX-free testability reason as [`gyro_bootstrap_gate_check`] above —
/// "inject inconsistent factors, confirm rollback fires" is exercised here
/// directly on the NIS sequence a genuinely poisoned bootstrap would
/// produce, without needing a live session to generate one.
///
/// `pub(crate)`, not private: Milestone M7's `crate::dpvo_scale_coupling`
/// reuses this exact counter/threshold logic for its own SOFT rollback
/// (decay the annealing weight + reset the scale posterior's variance,
/// rather than this module's hard un-bootstrap) — same reasoning as every
/// other `pub(crate)` widening in this file's M5b/M6 history.
pub(crate) fn rollback_monitor_step(
    mean_nis: f64,
    bound: f64,
    consecutive_bad: usize,
    threshold: usize,
) -> (usize, bool) {
    let next = if mean_nis.is_finite() && mean_nis <= bound { 0 } else { consecutive_bad + 1 };
    (next, next >= threshold)
}

/// Milestone M7 diagnostic fix (see [`DpvoOdometry::scale_coupling_step`]'s
/// own doc, "Stage 2" comment, for the real-run finding this addresses):
/// given a WINDOW-ORDERED (oldest-to-newest) slice of `arrival_index`
/// values, return the start index of the maximal TRAILING run in which
/// every consecutive pair differs by exactly `1` — i.e. the largest
/// suffix with no `DpvoPatchGraph::keyframe`-folding gap. Returns
/// `arrivals.len()` for an empty slice (an empty trailing run — the only
/// sane answer, and one every call site already treats as "nothing to
/// use" via a zero-length resulting slice) and `0` for a slice with no gap
/// at all (the whole window is already one consecutive run — the common
/// case before enough folding has happened for a gap to appear, confirmed
/// by every M4/M4-perf/M5/M5b/M6 run's own windowing never needing this
/// concept at all).
fn trailing_consecutive_run_start(arrivals: &[usize]) -> usize {
    if arrivals.is_empty() {
        return 0;
    }
    let mut start = arrivals.len() - 1;
    while start > 0 && arrivals[start] - arrivals[start - 1] == 1 {
        start -= 1;
    }
    start
}

#[cfg(test)]
mod scale_coupling_windowing_tests {
    use super::trailing_consecutive_run_start;

    #[test]
    fn whole_window_is_one_run_when_there_is_no_gap() {
        assert_eq!(trailing_consecutive_run_start(&[10, 11, 12, 13, 14]), 0);
    }

    #[test]
    fn trims_to_the_maximal_trailing_consecutive_run() {
        // Arrivals 10 and 13 survive with 11/12 folded away (a gap of 3),
        // then 13..17 are all consecutive — the trailing run starts at the
        // first index whose PRECEDING pair is non-consecutive.
        let arrivals = [5, 7, 10, 13, 14, 15, 16, 17];
        // index of value 13 is 3; preceding pair (10,13) has a gap of 3.
        assert_eq!(trailing_consecutive_run_start(&arrivals), 3);
    }

    #[test]
    fn single_pose_window_returns_zero() {
        assert_eq!(trailing_consecutive_run_start(&[42]), 0);
    }

    #[test]
    fn empty_window_returns_zero() {
        assert_eq!(trailing_consecutive_run_start(&[]), 0);
    }

    #[test]
    fn a_gap_immediately_before_the_last_frame_leaves_only_that_frame() {
        let arrivals = [1, 2, 3, 9];
        assert_eq!(trailing_consecutive_run_start(&arrivals), 3);
    }
}

/// Given per-item anchor features (`num_items, 128, 3, 3`) and per-item
/// reprojected `3×3` grids (`num_items, 3, 3, 2`, in level-0/native pixel
/// coordinates), run the 2-pyramid-level correlation lookup and interleave
/// into DPVO's own `(num_items, 882)` layout — `DPVO.corr` (`dpvo.py:200-
/// 207`): `torch.stack([corr1, corr2], -1).view(1, len(ii), -1)`, i.e. the
/// pyramid level is the fastest-varying axis, nested inside tap, inside
/// patch-pixel-column, inside patch-pixel-row.
fn corr_pyramid(
    anchor_gmap: ArrayView4<'_, f32>,
    coords_grid_px: ArrayView4<'_, f32>,
    target_level0: &ChannelLastImage,
    target_level1: &ChannelLastImage,
) -> Array2<f32> {
    let num_items = anchor_gmap.dim().0;
    let mut coords_l1 = Array4::<f32>::zeros((num_items, PATCH, PATCH, 2));
    for i in 0..num_items {
        for py in 0..PATCH {
            for px in 0..PATCH {
                coords_l1[(i, py, px, 0)] = coords_grid_px[(i, py, px, 0)] / 4.0;
                coords_l1[(i, py, px, 1)] = coords_grid_px[(i, py, px, 1)] / 4.0;
            }
        }
    }
    // M4-perf (`docs/dpvo_droid_port_plan.md`): `target_level0`/`target_level1`
    // arrive pre-transposed (see `FramePyramid`'s doc) — `corr_cpu_prebuilt_target`
    // skips the per-call target-side transpose `corr_cpu` would otherwise redo.
    let corr1 = corr_cpu_prebuilt_target(anchor_gmap, target_level0, coords_grid_px, CORR_RADIUS);
    let corr2 = corr_cpu_prebuilt_target(anchor_gmap, target_level1, coords_l1.view(), CORR_RADIUS);
    let taps = 2 * CORR_RADIUS + 1;
    let mut out = Array2::<f32>::zeros((num_items, CORR_DIM));
    for i in 0..num_items {
        for py in 0..PATCH {
            for px in 0..PATCH {
                for t in 0..taps * taps {
                    let base = ((py * PATCH + px) * taps * taps + t) * 2;
                    out[(i, base)] = corr1[(i, py, px, t)];
                    out[(i, base + 1)] = corr2[(i, py, px, t)];
                }
            }
        }
    }
    out
}

fn squeeze_patch_vector(imap_patch4: &Array4<f32>, patch_index: usize) -> Array1<f32> {
    let v = imap_patch4.index_axis(Axis(0), patch_index);
    let v = v.index_axis(Axis(1), 0);
    let v = v.index_axis(Axis(1), 0);
    v.to_owned()
}

fn grayscale_to_input_tensor(image: ArrayView2<'_, u8>) -> Array4<f32> {
    let (h, w) = image.dim();
    let mut out = Array4::<f32>::zeros((1, 3, h, w));
    for y in 0..h {
        for x in 0..w {
            let v = image[(y, x)] as f32;
            out[(0, 0, y, x)] = v;
            out[(0, 1, y, x)] = v;
            out[(0, 2, y, x)] = v;
        }
    }
    out
}

fn avg_pool_4x4(x: ArrayView3<'_, f32>) -> Array3<f32> {
    let (c, h, w) = x.dim();
    let (ho, wo) = (h / 4, w / 4);
    let mut out = Array3::<f32>::zeros((c, ho, wo));
    for ch in 0..c {
        for oy in 0..ho {
            for ox in 0..wo {
                let mut sum = 0.0_f32;
                for dy in 0..4 {
                    for dx in 0..4 {
                        sum += x[(ch, oy * 4 + dy, ox * 4 + dx)];
                    }
                }
                out[(ch, oy, ox)] = sum / 16.0;
            }
        }
    }
    out
}

/// `torch.quantile(x, 0.5)`'s linear-interpolation convention (distinct from
/// `torch.median`'s "lower of the two middles" — see
/// [`DpvoOdometry::median_recent_depth`]'s doc for why these differ).
fn torch_quantile_50(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let pos = 0.5 * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torch_quantile_50_matches_linear_interpolation_for_even_count() {
        // [1,2,3,4] -> position 1.5 -> average of index1(2) and index2(3) = 2.5.
        assert!((torch_quantile_50(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn torch_quantile_50_matches_exact_middle_for_odd_count() {
        assert!((torch_quantile_50(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn torch_quantile_50_empty_is_zero() {
        assert_eq!(torch_quantile_50(&[]), 0.0);
    }

    /// Milestone M5b: [`gyro_bootstrap_gate_check`] must reject a
    /// noisy/implausible rotation alignment — this is the task's own
    /// required "gyro gate rejects on noisy synthetic rotations" test,
    /// exercised directly on the pure gate function (no ONNX/live
    /// `DpvoOdometry` needed — see that function's own doc).
    ///
    /// The magnitude bound's SHIPPED default is `0.05`, kept conservative
    /// after a real-data A/B on MH_01 — see
    /// [`DpvoImuConfig::max_gyro_bias_magnitude_rad_s`]'s own doc for the
    /// full story: a `0.3` experiment let a bootstrap through whose
    /// recovered scale (`18.66`) passed every OTHER gate yet still
    /// corrupted the run (rigid ATE `55.49 m`), so `0.05` is what ships.
    /// Both M5's own collapsed-run bias and this milestone's own MH_01
    /// run's worst observed magnitude are exercised below and must both
    /// still be rejected at this conservative default.
    #[test]
    fn gyro_bootstrap_gate_rejects_noisy_rotation_alignment_and_accepts_a_clean_one() {
        let cfg = DpvoImuConfig::default();

        let m5_collapsed_run_bias = GyroBiasAlignment {
            bias_gyro: Vector3::new(-0.081, -0.182, 0.077),
            iterations: 5,
            rotation_residual_rms_before: 0.20,
            rotation_residual_rms_after: 0.19, // barely moved: fails both magnitude and fraction gates.
        };
        assert_eq!(
            gyro_bootstrap_gate_check(&m5_collapsed_run_bias, &cfg),
            Err(GyroGateRejection::MagnitudeTooLarge),
            "M5's own collapsed-run bias must be rejected (on magnitude, checked first)"
        );

        let m5b_worst_observed_magnitude = GyroBiasAlignment {
            // This milestone's own MH_01 `0.05`-bound run's worst observed
            // magnitude (`docs/dpvo_droid_port_plan.md`'s "M5b results").
            bias_gyro: Vector3::new(0.51, 0.0, 0.0),
            iterations: 5,
            rotation_residual_rms_before: 0.02,
            rotation_residual_rms_after: 0.01, // rms gates alone would pass this — magnitude must still catch it.
        };
        assert_eq!(
            gyro_bootstrap_gate_check(&m5b_worst_observed_magnitude, &cfg),
            Err(GyroGateRejection::MagnitudeTooLarge),
            "a magnitude far beyond any plausible MEMS gyro bias must still be rejected even with excellent rms"
        );

        let noisy_but_small_magnitude = GyroBiasAlignment {
            bias_gyro: Vector3::new(0.01, -0.01, 0.005),
            iterations: 5,
            rotation_residual_rms_before: 0.20,
            rotation_residual_rms_after: 0.19, // magnitude passes, but rms is nowhere near converged.
        };
        assert_eq!(
            gyro_bootstrap_gate_check(&noisy_but_small_magnitude, &cfg),
            Err(GyroGateRejection::RmsAboveAbsoluteBound),
            "a small-magnitude bias whose rms alignment is still way above the absolute bound must be rejected"
        );

        let converged_but_not_enough = GyroBiasAlignment {
            bias_gyro: Vector3::new(0.01, -0.01, 0.005),
            iterations: 5,
            // Both under the absolute bound (0.03) but rms barely moved
            // from its starting point — the fraction gate's own reason to
            // exist, distinct from the absolute-bound gate above.
            rotation_residual_rms_before: 0.029,
            rotation_residual_rms_after: 0.028,
        };
        assert_eq!(
            gyro_bootstrap_gate_check(&converged_but_not_enough, &cfg),
            Err(GyroGateRejection::RmsNotEnoughImprovement),
            "an rms that clears the absolute bound but barely moved from its start must still be rejected"
        );

        let good = GyroBiasAlignment {
            bias_gyro: Vector3::new(0.002, -0.001, 0.0015),
            iterations: 3,
            rotation_residual_rms_before: 0.10,
            rotation_residual_rms_after: 0.01,
        };
        assert_eq!(
            gyro_bootstrap_gate_check(&good, &cfg),
            Ok(()),
            "a plausible, well-converged bias must be accepted"
        );
    }

    /// Milestone M5b: [`rollback_monitor_step`]'s pure counter logic — the
    /// task's own required "rollback triggers on injected inconsistent
    /// factors and restores visual-only behavior" check, at the level that
    /// actually makes the decision. `imu_factor_nis_is_large_for_an_obviously_inconsistent_factor`
    /// (in `crate::dpvo_vi_ba`) is the companion check that the NIS this
    /// function consumes is itself a meaningful signal, not an arbitrary
    /// number.
    #[test]
    fn rollback_monitor_step_triggers_after_k_consecutive_bad_frames_and_resets_on_good() {
        let bound = 500.0;
        let threshold = 5;

        let mut consecutive = 0usize;
        for expected_count in 1..=4 {
            let (next, tripped) = rollback_monitor_step(10_000.0, bound, consecutive, threshold);
            assert_eq!(next, expected_count);
            assert!(!tripped, "must not roll back before {threshold} consecutive bad frames");
            consecutive = next;
        }
        let (next, tripped) = rollback_monitor_step(10_000.0, bound, consecutive, threshold);
        assert_eq!(next, 5);
        assert!(tripped, "must roll back on the {threshold}th consecutive bad frame");

        // A single good frame resets the counter to zero, not just decrements.
        let (reset, tripped_after_good) = rollback_monitor_step(1.0, bound, next, threshold);
        assert_eq!(reset, 0);
        assert!(!tripped_after_good);

        // Non-finite NIS (e.g. a solve that diverged to NaN) counts as bad.
        let (next_nan, _) = rollback_monitor_step(f64::NAN, bound, 0, threshold);
        assert_eq!(next_nan, 1);
    }

    #[test]
    fn corr_pyramid_shape_matches_corr_dim() {
        let anchor = Array4::<f32>::zeros((2, FNET_DIM, PATCH, PATCH));
        let coords = Array4::<f32>::zeros((2, PATCH, PATCH, 2));
        let level0 = ChannelLastImage::from_chw(Array3::<f32>::zeros((FNET_DIM, 16, 16)).view());
        let level1 = ChannelLastImage::from_chw(Array3::<f32>::zeros((FNET_DIM, 4, 4)).view());
        let out = corr_pyramid(anchor.view(), coords.view(), &level0, &level1);
        assert_eq!(out.shape(), &[2, CORR_DIM]);
    }

    /// M4-perf micro-benchmark (`docs/dpvo_droid_port_plan.md`'s "M4-perf
    /// results"): times the **full** per-group correlation-assembly path —
    /// [`reproject_patch_grid`] (this crate's own M3/M4 addition) followed
    /// by [`corr_pyramid`] (this file's 2-pyramid-level `corr_cpu`
    /// assembly) — at a single target-frame group's worth of DPVO's real
    /// working set (a "few thousand edges" one `by_target` group can hold
    /// at `fast.yaml`/`default.yaml` scale; see `update_step`'s own
    /// `by_target` grouping). This is the same call sequence
    /// `update_step`/`motion_probe` run per group, just with a synthetic
    /// pose/patch graph instead of a live EuRoC session, so this test needs
    /// no ONNX runtime, fixtures, or `ORT_DYLIB_PATH` — only
    /// `--release --features onnx-inference` (this whole module's gate) and
    /// `--ignored` (it is a timing report, not a correctness check; shape/
    /// numeric correctness of both pieces is already covered by their own
    /// unit/fixture tests elsewhere).
    ///
    /// ```text
    /// cargo test -p visloc-slam --release --features onnx-inference \
    ///   --lib dpvo_vo::tests::correlation_assembly_perf_at_realistic_working_set \
    ///   -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "timing report, not a correctness check; run --release, see doc comment"]
    fn correlation_assembly_perf_at_realistic_working_set() {
        use nalgebra::{UnitQuaternion, Vector3};

        let num_edges = 3000;
        // EuRoC-shaped stride-4 (`RES`) feature map: 752x480 input -> 188x120.
        let (level0_h, level0_w) = (120usize, 188usize);
        let intr = DpvoIntrinsics { fx: 190.0, fy: 190.0, cx: 94.0, cy: 60.0 };
        let pose_i = SE3::identity();
        // A modest, non-degenerate baseline (small rotation + translation),
        // shared by every synthetic edge below — realistic magnitude for a
        // temporal-neighbour edge, not chosen to stress any particular
        // reprojection edge case.
        let pose_j = SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.03),
            Vector3::new(0.05, 0.0, 0.02),
        );

        // Deterministic, dependency-free pseudo-randomness (xorshift), same
        // rationale as `crates/vision/src/dpvo/correlation.rs`'s own perf
        // test: this is a timing report, not a statistical study, so a tiny
        // in-file PRNG avoids pulling in `rand`'s distribution API just for
        // this.
        let mut state = 0x1234_5678_9abc_def1_u64;
        let mut next_f64 = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut anchor_gmap = Array4::<f32>::zeros((num_edges, FNET_DIM, PATCH, PATCH));
        for v in anchor_gmap.iter_mut() {
            *v = next_f64() as f32;
        }
        let level0 = Array3::<f32>::from_shape_fn((FNET_DIM, level0_h, level0_w), |_| next_f64() as f32);
        let level1 = avg_pool_4x4(level0.view());
        // Built once, outside the timed section below — mirrors the real
        // per-frame steady state after the M4-perf caching change
        // (`FramePyramid` stores these pre-transposed; see its doc comment),
        // not the one-time construction cost.
        let level0_hwc = ChannelLastImage::from_chw(level0.view());
        let level1_hwc = ChannelLastImage::from_chw(level1.view());

        let patches: Vec<DpvoPatch> = (0..num_edges)
            .map(|_| DpvoPatch {
                x: next_f64() * level0_w as f64,
                y: next_f64() * level0_h as f64,
                inverse_depth: 0.1 + next_f64(),
            })
            .collect();

        let reproject_start = std::time::Instant::now();
        let mut coords_grid_px = Array4::<f32>::zeros((num_edges, PATCH, PATCH, 2));
        for (edge, patch) in patches.iter().enumerate() {
            let grid = reproject_patch_grid(&pose_i, &pose_j, &intr, &intr, patch);
            for py in 0..PATCH {
                for px in 0..PATCH {
                    coords_grid_px[(edge, py, px, 0)] = grid[py][px].x as f32;
                    coords_grid_px[(edge, py, px, 1)] = grid[py][px].y as f32;
                }
            }
        }
        let reproject_ms = reproject_start.elapsed().as_secs_f64() * 1000.0;
        println!("  [perf] reproject_patch_grid x{num_edges} edges: {reproject_ms:.3} ms/call");

        let corr_start = std::time::Instant::now();
        let corr_flat =
            corr_pyramid(anchor_gmap.view(), coords_grid_px.view(), &level0_hwc, &level1_hwc);
        let corr_ms = corr_start.elapsed().as_secs_f64() * 1000.0;
        println!("  [perf] corr_pyramid (2 levels) x{num_edges} edges: {corr_ms:.3} ms/call");
        assert_eq!(corr_flat.shape(), &[num_edges, CORR_DIM]);

        println!(
            "  [perf] total correlation-assembly (reproject + corr_pyramid): {:.3} ms/call",
            reproject_ms + corr_ms
        );
    }
}
