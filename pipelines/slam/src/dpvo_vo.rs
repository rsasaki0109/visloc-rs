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
//!
//! # Low-parallax hover freeze (Milestone M14, `docs/dpvo_droid_port_plan.md`)
//!
//! M13's own diagnosis (`docs/dpvo_droid_port_plan.md`'s "M13 results"):
//! MH_01 800f contains a genuine ~24s near-total-stillness hover
//! (processed frames ~200-440); `keyframe`/`KEYFRAME_THRESH` folds ~91% of
//! its frames away (`store=False`, no trace left), but the surviving ~9%
//! commit real patches with unconstrained, `rand()`/median-fallback depth
//! (`crate::dpvo_patch_graph`'s own "Patch/frame addressing" section).
//! Separately, ordinary removal-window aging (`threshold = n - REMOVAL_WINDOW`
//! in [`DpvoPatchGraph`]'s own live-frame-count terms) purges every
//! PRE-hover, well-constrained patch's active edges well before the hover
//! ends, since `n` still advances by one for every surviving (non-folded)
//! frame even though real motion is near zero — by the time real motion
//! resumes, nothing old enough to pin scale is still live, and the BA must
//! reconcile fresh, well-constrained evidence against only the
//! ill-conditioned hover patches, baking in the ~20x scale explosion M6-M13
//! all measured downstream of this event.
//!
//! [`DpvoOdometryConfig::low_parallax`] is `None` by default — every prior
//! milestone's call site keeps compiling and running byte-for-byte as
//! before (the detector is never even evaluated at all whenever this is
//! `None`, since the gate sits entirely inside an `if let Some(cfg) = ...`
//! block).
//!
//! ## A genuine finding: the obvious geometric proxy does not separate
//! hover from motion — calibrated against real MH_01 data, not assumed
//!
//! The first candidate tried was purely geometric and ONNX-free:
//! `flow_mag(prev_pose, candidate_pose, ...)` (the same primitive
//! [`DpvoPatchGraph::motionmag`] already uses for `KEYFRAME_THRESH`
//! decimation) between the last COMMITTED frame's own patches (at their
//! CURRENT depth) and the not-yet-committed candidate's motion-model-
//! predicted pose. A calibration run (`--hover-freeze` with an
//! unreachable `enter_flow`, so the detector computes and logs but never
//! fires) logged this value for every frame of an 800f MH_01 run and
//! showed it sitting in a narrow `~0.9-1.3` band **for the ENTIRE run**,
//! including deep inside the GT-confirmed hover (frames 200-440, where
//! `docs/dpvo_droid_port_plan.md`'s own M13 windowed-profile data puts GT
//! angular rate at `0.0007-0.002 rad/frame` — far too small to explain a
//! ~1px reprojection flow via rotation) — no separation from the
//! surrounding real-motion frames at all. Root cause: `flow_mag` is
//! `beta=0.5 * (full 6-DoF flow) + 0.5 * (translation-only flow)`
//! evaluated against the previous frame's patches at THEIR inverse depth —
//! but a patch born during (or just before) a low-parallax span has
//! exactly the ill-conditioned depth M13 diagnosed as the problem in the
//! first place, so even the tiny pose-to-pose noise an imperfect monocular
//! BA solve produces between two genuinely-static frames reprojects,
//! through that bad depth, into an O(1) pixel "flow" reading — a noise
//! floor that swamps the real signal. Raising `beta` toward the
//! translation-only term alone does not fix this either, since the
//! CONTAMINATION is in the depth the reprojection divides by, not in which
//! half of `flow_mag`'s blend is used.
//!
//! [`DpvoOdometry::process_frame`] therefore reuses
//! [`DpvoOdometry::motion_probe`] itself — the SAME learned, GRU-based
//! correction-magnitude signal the bootstrap-only gate already uses (and
//! which this codebase has trusted since M4/M5 via
//! [`DpvoOdometryConfig::motion_probe_min_flow`]'s own `< 2.0` bootstrap
//! gate) — as the causal "parallax" proxy, instead of the geometric
//! attempt above. Being a learned correlation-network output rather than a
//! raw depth-dependent reprojection, it does not inherit the same-depth
//! contamination, and a calibration run confirmed a real (if modest —
//! median `~12` inside the hover vs. `~17` immediately before it, not a
//! collapse to near-zero) separation. The cost is real and worth stating
//! plainly regardless of the two findings below: this doubles the
//! correlation+GRU-update work for every frame once the graph is
//! initialized (one call from `Self::process_frame`'s own gate, one from
//! `Self::update_step`'s ordinary solve) whenever `config.low_parallax` is
//! `Some` — accepted because the cheaper geometric signal was tried first
//! and demonstrably does not work.
//!
//! ## Two more real-run findings, on top of the first: raw thresholding is
//! too noise-sensitive, and the signal is not stationary across a run
//!
//! Two more live A/B attempts, each an honest finding in its own right (the
//! full numbers are in `docs/dpvo_droid_port_plan.md`'s "M14 results"):
//!
//! 1. A raw "N consecutive frames below threshold" streak requirement (this
//!    module's first design) is too fragile to `motion_probe`'s own
//!    per-frame noise: even deep inside the confirmed hover, individual
//!    readings cross a tight threshold every few frames, so a strict streak
//!    breaks almost as often as it should fire. Worse, replaying one run's
//!    own logged trace to pick a longer, apparently-safe streak length
//!    (looked clean OFFLINE) still produced a WRONG live result on a fresh
//!    binary — this codebase's own documented "binary rebuilds shift
//!    RANSAC/HashMap ordering" gotcha is sharp enough to flip which side of
//!    a tight per-frame threshold a noisy reading lands on.
//! 2. Even after fixing (1) with [`LowParallaxRegimeState`]'s current
//!    windowed-MEDIAN design (smooths the per-frame noise out), a live run
//!    surfaced a THIRD, more fundamental problem: `motion_probe`'s own
//!    baseline is not stationary across an 800-frame run. It sits
//!    `~17-18` for frames 0-200, correctly drops to `~12` for the confirmed
//!    hover 200-450, only PARTIALLY recovers to `~14` for 450-500, then
//!    drops right back to `~12-13.5` for the REST of the run (500-800) —
//!    a span with real GT speed `0.27-0.88 m/s` throughout, not a second
//!    hover. Plausible mechanism: once M13's own diagnosed scale corruption
//!    has baked itself into the pose chain (which happens in exactly this
//!    450-780 window), the constant-velocity motion model's prediction
//!    becomes self-consistently "easy" to satisfy within the now-corrupted
//!    coordinate frame, so `motion_probe`'s own learned correction reads
//!    low for reasons unrelated to true camera stillness. No fixed absolute
//!    threshold can tell "genuinely still" apart from "already corrupted,
//!    now internally consistent" — this is a property of the signal in
//!    this post-corruption regime, not a tuning miss.
//!
//! Finding 2 is answered by an explicitly-scoped limitation, not a better
//! detector (out of scope here — see the results doc's "what a real fix
//! would need"): [`LowParallaxRegimeState`] permanently DISARMS itself the
//! first time it exits the regime. This protects the ONE hover M13
//! diagnosed on MH_01 (the only one confirmed in this 800f range) and will
//! not detect a genuine second hover later in a longer sequence — a real,
//! stated constraint, not a hidden one.
//!
//! **The freeze itself reuses existing plumbing, not a new mechanism**:
//! while the regime is active, every candidate frame is rejected via the
//! SAME [`DpvoPatchGraph::reject_pending_frame`] path the bootstrap-only
//! `motion_probe` gate already uses (this module's own patches_vec/depth
//! sampling above still runs — so RNG call counts stay identical to a
//! `motion_probe` rejection — but [`DpvoPatchGraph::commit_frame`] never
//! runs). Concretely, this means: no new patch is admitted to the graph (no
//! unconstrained depth is created at all, rather than merely damping one
//! after the fact), `n_frames()` does not advance, and — because
//! `keyframe_dispatch`/`update_step`/every other per-frame mechanism below
//! only runs in the `else if self.graph.is_initialized()` branch AFTER a
//! successful commit — the removal-window aging check inside
//! `crate::dpvo_patch_graph::DpvoPatchGraph::keyframe_inner` simply never
//! runs during the frozen span either. Every pre-hover patch's active edges
//! stay exactly as fresh as they were the instant the regime was entered,
//! for as long as the regime lasts — "freezing patch aging/lifetime and
//! window advancement" (the design brief's own phrasing) falls out of "stop
//! calling `commit_frame`" for free, rather than needing a new suppression
//! flag threaded through the patch graph's own bookkeeping.
//!
//! Diagnostics: [`DpvoOdometry::low_parallax_diagnostics`] (enter/exit
//! counts, frames suppressed, current streak) and
//! [`DpvoOdometry::low_parallax_flow_log`] (every evaluated frame's own
//! flow value + regime state — the acceptance evidence for "did this fire
//! at the right place, for the right duration").
//!
//! ## Milestone M15: "depth-trust damping" — same detector, a different
//! response
//!
//! M14's own "what a real fix would need" section proposed two untried
//! responses to the SAME [`LowParallaxRegimeState`] detector: Option C
//! ("exit re-anchor", a one-shot rescale at hover exit) and Option B
//! ("depth-trust damping" — let hover-span frames commit NORMALLY,
//! preserving the baseline's own gradual `KEYFRAME_THRESH`-decimated
//! reconnection through the hover that M14's freeze destroyed, but heavily
//! damp the depth channel of whichever patches DO get committed while the
//! regime is active, so they contribute rotation/pose constraints without
//! dragging scale). [`DpvoLowParallaxConfig::response`] selects between
//! [`LowParallaxResponse::Freeze`] (M14's mechanism, default, unchanged) and
//! [`LowParallaxResponse::DepthDamp`] (M15's Option B) — the SAME
//! [`LowParallaxRegimeState`] enter/exit/one-shot-disarm state machine drives
//! both; only what happens once a candidate is known to be "hover-active"
//! differs.
//!
//! **Mechanism**: [`DpvoOdometry::low_parallax_gate`] no longer calls
//! [`crate::dpvo_patch_graph::DpvoPatchGraph::reject_pending_frame`] under
//! `DepthDamp` — the candidate falls through to the ordinary
//! `commit_frame`/keyframe-decimation path exactly like every prior
//! milestone's default behavior. Immediately after a successful commit,
//! [`DpvoOdometry::process_frame`] flags that frame's `patches_per_frame`
//! patches into [`LowParallaxDampState`] whenever
//! [`LowParallaxRegimeState::in_regime`] was true for that commit. Flagging
//! is FRAME-level, not per-patch: every patch born in the same frame shares
//! one anchor pose/timestamp, so "how much real parallax has accumulated
//! since birth" is the same signal (this run's own ego-motion since that
//! frame) for all of that frame's patches — a documented simplification, not
//! an oversight, that also keeps the bookkeeping a plain
//! `HashSet<arrival_index>` (stable across
//! `crate::dpvo_patch_graph::DpvoPatchGraph` compaction on keyframe removal,
//! unlike a live index) instead of a second per-patch parallel `Vec` needing
//! its own manual removal-compaction hook the way `frame_pyramids`/
//! `patch_gmap`/`patch_imap` already need.
//!
//! At every [`crate::dpvo_patch_ba::dpvo_ba`] call site in this module
//! (the per-frame windowed solve in [`DpvoOdometry::update_step`], and both
//! Milestone M8/M10 global-BA passes in [`DpvoOdometry::run_legacy_global_ba`]/
//! [`DpvoOdometry::run_widened_global_ba`]), a flagged frame's patches get a
//! [`crate::dpvo_patch_ba::DpvoBaProblem::depth_damping`] multiplier of
//! [`DpvoLowParallaxConfig::depth_damp_factor`] instead of the implicit
//! `1.0` — see that field's own doc for exactly what this does to the
//! depth-channel Tikhonov term (`q = 1/(C + lmbda·multiplier)`, `ba.py:158`):
//! a low-parallax patch's own Jacobian information `C` is near-zero (no
//! baseline to observe depth from), so `q` is normally dominated by the bare
//! `lmbda` floor rather than real visual evidence — inflating that floor is
//! a direct, surgical counter to the unconstrained-depth mechanism M13's own
//! diagnosis pinned as the root cause, without withholding the frame's real
//! ROTATION/pose evidence (which the freeze response withheld too, and
//! which M14's own post-mortem blames for the freeze's abrupt, worse
//! resume transition). `dpvo_vi_ba.rs`'s own separately-duplicated visual
//! assembly (see that module's doc, "Visual assembly is a deliberate, tested
//! duplication") is NOT threaded through this mechanism — out of scope for
//! M15's visual-only acceptance runs (`config.imu` stays `None` throughout),
//! a documented limitation, not a silent gap.
//!
//! **Un-flag rule (age-based, not a repeated geometric flow probe)**: a
//! flagged frame is un-flagged once [`DpvoLowParallaxConfig::unflag_after_commits`]
//! further frames have committed since its own birth AND the regime is no
//! longer active (never un-flag while still inside a — possibly still
//! ongoing — hover; nothing has had a chance to accumulate real motion yet).
//! Deliberately NOT the geometric `flow_mag` probe this module's own
//! "Two more real-run findings" section above already rejected as a
//! DETECTOR: that finding's root cause (a patch reprojected through its own
//! still-unconstrained depth produces an `O(1)`-pixel noise floor
//! independent of true motion) applies just as much to a patch's SELF-flow
//! immediately after birth as it does during the hover itself, so reusing it
//! as the UN-flag signal would risk the identical contamination. Age (a
//! frame count) carries no such risk and is directly measurable without any
//! extra ONNX inference. See [`LowParallaxDampState::advance_unflagging`]'s
//! own doc for why this bookkeeping is self-cleaning (never accumulates
//! unbounded stale entries) even without a keyframe-removal hook.
//!
//! Diagnostics: [`DpvoLowParallaxDiagnostics`] gained
//! `response`/`currently_damped_frames`/`frames_flagged_total`/
//! `patches_flagged_total`/`damped_solve_count`/`unflagged_total` — see that
//! struct's own field docs.
#![cfg(feature = "onnx-inference")]

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Instant;

use nalgebra::{Vector2, Vector3};
use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Frame, Keyframe, VisualMap};
use visloc_vision::dpvo::correlation::{corr_cpu_prebuilt_target, ChannelLastImage};
use visloc_vision::dpvo::native_cuda_correlation::{
    NativeCudaCorrelation, NativeCudaCorrelationError,
};
use visloc_vision::dpvo::npz::{NpzArchive, NpzError};
use visloc_vision::dpvo::onnx_session::{DpvoOnnxError, DpvoOnnxSession};
use visloc_vision::dpvo::patchify::patchify_cpu;
use visloc_vision::dpvo::softagg::{SoftAgg, SoftAggError};
use visloc_vision::dpvo::{CORR_DIM, CORR_RADIUS, DIM, FNET_DIM, PATCH, RES};
use visloc_vision::features::deep::DeepFeatureSet;
use visloc_vision::features::superpoint_onnx::{
    OnnxBackend, SuperPointOnnxConfig, SuperPointOnnxError, SuperPointOnnxExtractor,
};
use visloc_vision::features::{DeepFeatureExtractor, GrayscaleImage};

use crate::dpvo_long_loop::{
    sp_anchored_patch_centers, DpvoLongLoopConfig, DpvoLongLoopDiagnostics, DpvoLongLoopIndex,
    QueryCandidateLogEntry,
};
use crate::dpvo_loop_closure::{
    expand_frame_pairs_to_patch_edges, find_loop_edges, DpvoLoopClosureConfig,
    UPSTREAM_MIN_LOOP_GAP,
};
use crate::dpvo_patch_ba::{
    dpvo_ba, reproject_patch_grid, DpvoBaConfig, DpvoBaError, DpvoBaProblem, DpvoEdge,
    DpvoIntrinsics, DpvoPatch,
};
use crate::dpvo_patch_graph::{DpvoGraphError, DpvoPatchGraph, DpvoVoConfig};
use crate::dpvo_scale_coupling::{
    apply_gentle_scale_correction, blend_solutions, scale_measurement_from_alignment,
    AnnealingWeight, RecursiveGyroBiasEstimator, RecursiveScaleEstimator, ScaleCouplingConfig,
};
use crate::dpvo_sim3_backend::{
    run_sim3_backend, DpvoSim3BackendConfig, Sim3BackendRejection, Sim3LoopMeasurement,
};
use crate::dpvo_vi_ba::{
    dpvo_vi_ba, estimate_mono_vi_alignment, imu_factor_nis, DpvoImuFactor,
    DpvoMonoViAlignmentGates, DpvoMonoViAlignmentRejection, DpvoViWindow,
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
    NativeCudaCorrelation(NativeCudaCorrelationError),
    /// `image.dim()` did not match [`DpvoOdometryConfig::width`]/`height`.
    ImageShapeMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
    },
    /// Milestone M11: `config.long_loop` was `Some` but no SuperPoint ONNX
    /// model path was supplied to [`DpvoOdometry::new`] — the long-range
    /// loop mechanism needs a real per-frame appearance descriptor, so this
    /// is a construction-time error, not a silent no-op.
    LongLoopModelRequired,
    /// Milestone M11: the SuperPoint ONNX session failed to load.
    LongLoop(SuperPointOnnxError),
}

impl std::fmt::Display for DpvoOdometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Onnx(e) => write!(f, "dpvo odometry: onnx error: {e}"),
            Self::SoftAgg(e) => write!(f, "dpvo odometry: softagg weight error: {e}"),
            Self::Npz(e) => write!(f, "dpvo odometry: npz error: {e}"),
            Self::Graph(e) => write!(f, "dpvo odometry: graph error: {e}"),
            Self::Ba(e) => write!(f, "dpvo odometry: bundle adjustment error: {e}"),
            Self::NativeCudaCorrelation(e) => {
                write!(f, "dpvo odometry: native CUDA correlation error: {e}")
            }
            Self::ImageShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "dpvo odometry: expected image shape {expected:?}, got {actual:?}"
                )
            }
            Self::LongLoopModelRequired => {
                write!(f, "dpvo odometry: config.long_loop is Some but no superpoint_model_path was supplied")
            }
            Self::LongLoop(e) => write!(f, "dpvo odometry: long-loop superpoint error: {e}"),
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
impl From<NativeCudaCorrelationError> for DpvoOdometryError {
    fn from(value: NativeCudaCorrelationError) -> Self {
        Self::NativeCudaCorrelation(value)
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
    /// Optional versioned runtime DLL for the single-call indexed CUDA
    /// correlation backend. `None` preserves the CPU correlation path.
    pub native_cuda_correlation_dll: Option<PathBuf>,
    /// Experimental grouped ONNX correlation graph. Default callers keep
    /// this false because the first target-by-target CUDA implementation is
    /// slower than the optimized CPU path due to repeated session and host
    /// transfer overhead.
    pub fused_correlation: bool,
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
    /// Milestone M8 (`docs/dpvo_droid_port_plan.md`): periodic full-graph
    /// bundle adjustment over every retained active + inactive edge — the
    /// CPU-bounded stand-in for upstream's `__run_global_BA`
    /// (`dpvo.py:312-325`). `None` (default) preserves every prior
    /// milestone's exact behavior byte-for-byte: [`DpvoOdometry::new`] does
    /// not enable inactive-edge retention on the patch graph at all, and
    /// [`DpvoOdometry::process_frame`] never calls
    /// [`DpvoOdometry::run_global_ba`]. See [`DpvoOdometry`]'s own doc,
    /// "Global BA (Milestone M8)".
    pub global_ba: Option<DpvoGlobalBaConfig>,
    /// Milestone M9 (`docs/dpvo_droid_port_plan.md`): a `Sim(3)` pose-graph
    /// scale-drift correction over the FULL retained + live pose history —
    /// see `crate::dpvo_sim3_backend`'s module doc for the full design and
    /// why this is a new addition (not a straight DPVO port) built on M8's
    /// own diagnosed limitation. `None` (default) preserves every prior
    /// milestone's exact behavior byte-for-byte: [`DpvoOdometry::process_frame`]
    /// never calls [`crate::dpvo_sim3_backend::run_sim3_backend`].
    pub sim3_backend: Option<DpvoSim3BackendConfig>,
    /// Milestone M11 (`docs/dpvo_droid_port_plan.md`): a long-range,
    /// appearance-based loop-candidate source feeding M9's Sim(3) backend
    /// and M10's widened global BA — see `crate::dpvo_long_loop`'s module
    /// doc for the full design and why this exists (M10's own real-run
    /// finding: the M6 proximity mechanism can never propose a loop edge
    /// wider than the live patch buffer, an order of magnitude short of the
    /// trajectory-spanning revisit MH_01's own scale drift needs). `None`
    /// (default) preserves every prior milestone's exact behavior
    /// byte-for-byte — no SuperPoint inference runs at all, and
    /// [`DpvoOdometry::new`] does not require a `superpoint_model_path`.
    pub long_loop: Option<DpvoLongLoopConfig>,
    /// Milestone M14 (`docs/dpvo_droid_port_plan.md`): detect a sustained
    /// near-zero-parallax ("hover") regime and freeze new-patch admission +
    /// patch/edge aging for its duration — see [`DpvoOdometry`]'s own doc,
    /// "Low-parallax hover freeze", for the full mechanism and why this
    /// answers M13's own diagnosed root cause. `None` (default) preserves
    /// every prior milestone's exact behavior byte-for-byte: the detector is
    /// never evaluated at all (not merely disabled after computing
    /// something), so there is no extra cost and no RNG-call-count change
    /// on any existing call site.
    pub low_parallax: Option<DpvoLowParallaxConfig>,
    /// A3 ranking-lab offline dump (`docs/visual_slam_sequential_sfm_plan.md`,
    /// "A3 — Sound long-range loop closure", ranking slice A): when `true`
    /// AND `long_loop` is `Some`, [`DpvoOdometry::process_frame`] additionally
    /// clones each ingested frame's raw SuperPoint keypoints+descriptors
    /// (the EXACT values `crate::dpvo_long_loop::DpvoLongLoopIndex::ingest_frame`
    /// receives — patch-grid coordinates, i.e. already divided by `RES`) into
    /// [`DpvoOdometry::long_loop_last_ingested`], for `examples/euroc_dpvo_vo_demo.rs`'s
    /// `--ll-dump-frame-descriptors` to write to disk. Default `false`: no
    /// extra clone, no extra memory, and `long_loop_last_ingested` stays
    /// `None` for the whole run — byte-for-byte the same behavior as every
    /// milestone before this flag existed, even with `long_loop` itself
    /// `Some`.
    pub long_loop_dump_enabled: bool,
}

/// Milestone M15 (`docs/dpvo_droid_port_plan.md`): which action
/// [`DpvoOdometry`] takes once [`LowParallaxRegimeState`] reports the regime
/// active for a candidate frame — see the module doc's "Milestone M15:
/// depth-trust damping" section for the full design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LowParallaxResponse {
    /// Milestone M14's mechanism, unchanged: reject the candidate via
    /// [`crate::dpvo_patch_graph::DpvoPatchGraph::reject_pending_frame`] —
    /// no patch is admitted, `n_frames()` does not advance, keyframe aging
    /// is frozen for the regime's duration. Kept as the default so every
    /// pre-M15 caller (and `DpvoLowParallaxConfig::default`'s own doc
    /// evidence) is unaffected.
    #[default]
    Freeze,
    /// Milestone M15's Option B: commit candidates normally (through the
    /// ordinary keyframe-decimation path, unchanged), but flag whichever
    /// frames DO commit while the regime is active so their patches' depth
    /// channel gets [`DpvoLowParallaxConfig::depth_damp_factor`]-heavier
    /// Tikhonov damping in every subsequent [`crate::dpvo_patch_ba::dpvo_ba`]
    /// call until [`DpvoLowParallaxConfig::unflag_after_commits`]'s own
    /// age-based un-flag rule fires.
    DepthDamp,
}

/// Milestone M14 (`docs/dpvo_droid_port_plan.md`): configuration for the
/// low-parallax ("hover") freeze described on
/// [`DpvoOdometryConfig::low_parallax`]'s own doc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoLowParallaxConfig {
    /// Sliding-window size (in evaluated post-initialization frames) the
    /// smoothed parallax statistic — a rolling MEDIAN of
    /// [`DpvoOdometry::motion_probe`]'s own per-frame output — is computed
    /// over, fed to [`LowParallaxRegimeState::update`]. See that struct's
    /// own doc, and [`DpvoOdometry`]'s module-doc section "Low-parallax
    /// hover freeze", for why a raw per-frame streak-of-consecutive-low-
    /// readings design was tried FIRST and replaced with this windowed
    /// design: the raw signal oscillates enough, even deep inside the true
    /// hover, that individual readings cross a tight threshold every few
    /// frames — a strict "every one of the last K frames must be below
    /// threshold" streak breaks on that noise almost as often as it should
    /// fire on the real thing. A rolling median absorbs that per-frame
    /// noise while still tracking the real, sustained level shift between
    /// a hover and ordinary motion.
    pub window: usize,
    /// Enter the regime once the rolling window is full and its median
    /// drops below this value. Units match
    /// [`DpvoOdometryConfig::motion_probe_min_flow`]'s own — the SAME
    /// learned GRU correction-magnitude signal, just evaluated post-
    /// initialization instead of only at bootstrap.
    pub enter_flow: f64,
    /// Exit the regime once the rolling window's median reaches this
    /// (`>= exit_flow`). Must be `>= enter_flow` for the hysteresis band to
    /// be non-empty (enforced by [`LowParallaxRegimeState::update`]'s own
    /// doc, not an invariant this struct itself checks).
    pub exit_flow: f64,
    /// Milestone M15: which response [`DpvoOdometry`] takes once the regime
    /// is active for a candidate — see [`LowParallaxResponse`]'s own doc.
    /// Defaults to [`LowParallaxResponse::Freeze`] (M14's mechanism,
    /// unchanged) so every pre-M15 construction of this struct (including
    /// every `Default::default()` call site already shipped) is byte-for-byte
    /// unaffected; the fields below are only consulted when this is
    /// [`LowParallaxResponse::DepthDamp`].
    pub response: LowParallaxResponse,
    /// Milestone M15 (`DepthDamp` only): multiplier on
    /// [`crate::dpvo_patch_ba::DpvoBaConfig::lmbda`] applied to a flagged
    /// frame's patches in every [`crate::dpvo_patch_ba::dpvo_ba`] call — see
    /// [`crate::dpvo_patch_ba::DpvoBaProblem::depth_damping`]'s own doc for
    /// the mechanism. `docs/dpvo_droid_port_plan.md`'s "M15 results" reports
    /// the calibration sweep this default (`1000.0`, two orders of
    /// magnitude above the bare `lmbda=1e-4` floor's own scale) was chosen
    /// from. Irrelevant (never read) when `response == Freeze`.
    pub depth_damp_factor: f64,
    /// Milestone M15 (`DepthDamp` only): a flagged frame is un-flagged once
    /// this many further frames have committed since its OWN birth
    /// (measured as an `arrival_index` gap — see
    /// [`LowParallaxDampState::advance_unflagging`]'s own doc) AND the
    /// regime is no longer active. Irrelevant when `response == Freeze`.
    pub unflag_after_commits: usize,
    /// Milestone M16 (`DepthDamp` only): number of committed frames over
    /// which an eligible cohort's multiplier decays geometrically from
    /// `depth_damp_factor` to `1.0`. Zero preserves M15's binary un-flag.
    pub gradual_release_duration_commits: usize,
    /// Milestone M16 (`DepthDamp` only): maximum frame cohorts allowed to
    /// begin gradual release in one commit. Each cohort contains exactly
    /// `patches_per_frame` patches, so this directly bounds release mass.
    pub gradual_release_start_cap_frames: usize,
}

impl Default for DpvoLowParallaxConfig {
    /// Calibrated against THREE real MH_01 800f runs, not one — each
    /// earlier attempt was tried live and found unsafe, and each finding is
    /// preserved here rather than quietly overwritten (see
    /// `docs/dpvo_droid_port_plan.md`'s "M14 results" for the full,
    /// honest blow-by-blow):
    ///
    /// 1. A raw per-frame streak design (`enter_streak=5`) re-triggered a
    ///    dozen+ brief 1-10-frame cycles throughout processed frames
    ///    500-800 — a real, FAST-motion span (GT speed `0.27-0.88 m/s`,
    ///    confirmed by direct query), not a second hover — corrupting an
    ///    800f run (`tracked_fraction` `1.00 -> 0.65`,
    ///    `ate_similarity_scale` `20.6 -> 26.2`, worse not better).
    /// 2. A longer streak (`enter_streak=10`), re-calibrated against that
    ///    SAME run's own logged trace, looked clean offline (one cycle,
    ///    `~220 -> 447`) — but a FRESH run using it entered only once,
    ///    briefly, at completely the wrong place (`frame 623-625`, deep in
    ///    the fast-motion span). Root cause: `motion_probe`'s raw per-frame
    ///    value oscillates enough, even deep inside the confirmed hover,
    ///    that individual readings cross a threshold like `12.0` every few
    ///    frames (e.g. `220=11.55, 221=12.01, ..., 229=13.15`) — a strict
    ///    all-K-consecutive-frames-below-threshold streak is fragile to
    ///    exactly this kind of noise, and (per this codebase's own
    ///    documented gotcha) a REBUILT binary's own floating-point/HashMap-
    ///    ordering differences are enough to shift which side of a tight
    ///    threshold a given frame's noisy reading lands on.
    /// 3. Windowed-median smoothing (this design) fixes the noise problem,
    ///    but surfaced a THIRD, more fundamental finding: `motion_probe`'s
    ///    own baseline is not stationary across an 800f run.
    ///    Bucket-by-50-frames medians: `~17-18` for frames `0-200`, drops
    ///    to `~12` for the confirmed hover `200-450` (correct), bounces
    ///    only PARTIALLY back to `~14` for `450-500`, then drops right back
    ///    to `~12-13.5` for the REST of the run (`500-800`) — a span with
    ///    real GT speed `0.27-0.88 m/s` throughout. Plausible mechanism:
    ///    once M13's own diagnosed scale corruption has baked itself into
    ///    the pose chain (which happens exactly in this `450-780` window),
    ///    the constant-velocity motion model's prediction becomes
    ///    self-consistently "easy" to satisfy within the now-corrupted (but
    ///    internally smooth) coordinate frame, so `motion_probe`'s own
    ///    learned correction reads low for reasons UNRELATED to true
    ///    camera stillness. A global absolute threshold cannot distinguish
    ///    "genuinely still" from "already corrupted, now self-consistent" —
    ///    this is a property of the signal on this dataset in this
    ///    post-corruption regime, not a tuning miss.
    ///
    /// Because (3) cannot be fixed by threshold/window tuning alone
    /// without a materially different (e.g. adaptive/relative-baseline)
    /// detector — out of scope for this milestone, see the results
    /// section's "what a real fix would need" — [`LowParallaxRegimeState`]
    /// additionally DISARMS itself permanently after its first exit (see
    /// that struct's own doc). This is a deliberate, explicitly-scoped
    /// limitation: it protects the ONE hover M13 diagnosed on MH_01 (the
    /// only one confirmed to exist in this 800f range) and will not detect
    /// a genuine SECOND hover later in a longer sequence. `window=20`,
    /// `enter_flow=13.0`, `exit_flow=15.0` replayed against the run-3 log
    /// enters at frame `216` and exits at frame `461` — matching M13's own
    /// independently-derived hover span (`~200-440`) closely, with the
    /// one-shot guard suppressing the would-be spurious later re-entry this
    /// same log shows starting at frame `518` and never letting go again
    /// before frame `800`.
    fn default() -> Self {
        Self {
            window: 20,
            enter_flow: 13.0,
            exit_flow: 15.0,
            response: LowParallaxResponse::Freeze,
            depth_damp_factor: 1000.0,
            unflag_after_commits: 16,
            gradual_release_duration_commits: 0,
            gradual_release_start_cap_frames: 4,
        }
    }
}

/// Milestone M14: causal regime state for the low-parallax freeze — see
/// [`DpvoOdometryConfig::low_parallax`]'s doc for the mechanism this
/// drives. Deliberately free-standing (no [`DpvoOdometry`] dependency) so
/// its enter/exit logic is unit-testable against synthetic flow sequences
/// without an ONNX session.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LowParallaxRegimeState {
    in_regime: bool,
    /// One-shot guard (see [`Self::update`]'s own doc): once `true`, every
    /// subsequent `update` call is a permanent no-op. Set the instant the
    /// regime exits for the first time — never cleared.
    disarmed: bool,
    /// Ring buffer of the last (up to) `cfg.window` raw flow readings, in
    /// arrival order (front = oldest). `VecDeque` rather than a fixed-size
    /// array since `cfg.window` is a runtime config value, not a constant.
    window: VecDeque<f64>,
}

/// One [`LowParallaxRegimeState::update`] call's outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LowParallaxTransition {
    /// Whether THIS frame should be suppressed (frozen) — the caller's own
    /// "reject this candidate" decision.
    pub suppress: bool,
    /// Whether this call caused the regime to become active (it was not
    /// active on entry, the windowed median just dropped below
    /// `enter_flow`).
    pub just_entered: bool,
    /// Whether this call caused the regime to become inactive (it was
    /// active on entry, the windowed median reached `exit_flow`).
    pub just_exited: bool,
}

impl LowParallaxRegimeState {
    /// Feed this frame's causal parallax-proxy value (in practice,
    /// [`DpvoOdometry::motion_probe`] — see [`DpvoOdometryConfig::low_parallax`]'s
    /// own doc); returns whether this frame should be suppressed and
    /// whether a transition happened, updating internal window/regime
    /// state in place.
    ///
    /// Always pushes `flow` into the rolling window first (even once
    /// [`Self::disarmed`], so the internal state stays consistent — though
    /// a disarmed instance never reads it again). Once permanently
    /// disarmed, every call returns the all-`false` no-op transition
    /// immediately, regardless of `flow` — see [`DpvoLowParallaxConfig::default`]'s
    /// own doc, finding 3, for the real-run evidence this guards against
    /// (a non-stationary `motion_probe` baseline that reads "hover-like"
    /// again, for unrelated reasons, later in the same run). Before the
    /// window is full for the first time, this always returns the no-op
    /// transition too (not enough history for a reliable statistic; the
    /// regime cannot yet be active at that point by construction, since
    /// entering requires a full window).
    ///
    /// Hysteresis, once armed and the window is full: while NOT in the
    /// regime, the window's median dropping below `cfg.enter_flow` enters
    /// the regime THIS frame (so the frame that fills the window below
    /// threshold is itself the first suppressed frame). While IN the
    /// regime, the window's median reaching `cfg.exit_flow` exits
    /// immediately AND disarms — the exiting frame itself is NOT suppressed
    /// (it is the frame whose real motion just proved the hover is over,
    /// so it should commit normally).
    pub fn update(&mut self, cfg: &DpvoLowParallaxConfig, flow: f64) -> LowParallaxTransition {
        let no_op = LowParallaxTransition {
            suppress: false,
            just_entered: false,
            just_exited: false,
        };
        if self.disarmed {
            return no_op;
        }
        let capacity = cfg.window.max(1);
        self.window.push_back(flow);
        while self.window.len() > capacity {
            self.window.pop_front();
        }
        if self.window.len() < capacity {
            return no_op;
        }
        let median = windowed_median(&self.window);
        if self.in_regime {
            if median >= cfg.exit_flow {
                self.in_regime = false;
                self.disarmed = true;
                return LowParallaxTransition {
                    suppress: false,
                    just_entered: false,
                    just_exited: true,
                };
            }
            return LowParallaxTransition {
                suppress: true,
                just_entered: false,
                just_exited: false,
            };
        }
        if median < cfg.enter_flow {
            self.in_regime = true;
            return LowParallaxTransition {
                suppress: true,
                just_entered: true,
                just_exited: false,
            };
        }
        no_op
    }

    /// Whether the regime is currently active.
    pub fn in_regime(&self) -> bool {
        self.in_regime
    }

    /// Whether the one-shot guard has fired (see [`Self::update`]'s own
    /// doc) — every subsequent `update` call is a permanent no-op.
    pub fn disarmed(&self) -> bool {
        self.disarmed
    }
}

/// The rolling-window "median" [`LowParallaxRegimeState::update`] smooths
/// over — [`torch_quantile_50`] applied to the window's own contents,
/// matching the same interpolated-median convention
/// [`DpvoOdometry::motion_probe`] itself already uses (via [`torch_quantile_50`]
/// over one frame's per-patch correction norms) — smoothing a sequence of
/// already-quantile-50'd values with the SAME convention, rather than
/// switching to `median_recent_depth`'s different `torch.median` (lower-of-
/// two-middle, non-interpolated) convention, which exists only to match a
/// specific upstream opcode this windowed statistic has no upstream
/// reference for.
fn windowed_median(window: &VecDeque<f64>) -> f64 {
    let mut values: Vec<f64> = window.iter().copied().collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    torch_quantile_50(&values)
}

/// Milestone M15 (`docs/dpvo_droid_port_plan.md`): free-standing (no
/// [`DpvoOdometry`]/ONNX dependency — exactly like [`LowParallaxRegimeState`]
/// itself, so the flag/un-flag lifecycle is unit-testable without a live
/// ONNX session) bookkeeping for the "depth-trust damping" response
/// ([`LowParallaxResponse::DepthDamp`]) — see the module doc's own
/// "Milestone M15: depth-trust damping" section for the full mechanism this
/// drives.
///
/// Frame-level, not per-patch: every patch committed in the same frame
/// shares one anchor pose/timestamp, so "real parallax accumulated since
/// birth" is the same signal (this run's own ego-motion since that frame)
/// for every one of that frame's patches — a documented simplification that
/// keeps this a plain `HashSet<usize>` keyed by
/// [`crate::dpvo_patch_graph::DpvoGraphFrame::arrival_index`] (stable across
/// patch-graph compaction, unlike a live frame index) rather than a second
/// per-patch parallel `Vec` needing its own manual keyframe-removal
/// compaction hook.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LowParallaxDampState {
    damped_frames: HashSet<usize>,
    /// Release start, keyed by the stable frame arrival index. This is a
    /// subset of `damped_frames`; absence means still fully damped.
    release_started_at: HashMap<usize, usize>,
    last_advance_now: usize,
    release_duration_commits: usize,
    frames_flagged_total: usize,
    patches_flagged_total: usize,
    unflagged_total: usize,
    damped_solve_count: usize,
    release_started_total: usize,
    max_release_started_per_advance: usize,
}

impl LowParallaxDampState {
    /// Flag `arrival`'s `patches_per_frame` patches as depth-damped —
    /// called once, immediately after a frame commits while the regime is
    /// active. A no-op (does not double-count) if `arrival` is already
    /// flagged.
    pub fn flag(&mut self, arrival: usize, patches_per_frame: usize) {
        if self.damped_frames.insert(arrival) {
            self.frames_flagged_total += 1;
            self.patches_flagged_total += patches_per_frame;
        }
    }

    /// Age-based un-flag rule — see the module doc's own "Un-flag rule"
    /// section for why NOT a repeated geometric flow probe. Every currently
    /// flagged frame whose `arrival_index` gap to `now` (the most recently
    /// committed frame's own arrival index) is at least
    /// `unflag_after_commits` is un-flagged, but ONLY when `still_in_regime`
    /// is `false` — nothing has had a chance to accumulate real parallax
    /// while the regime remains active, so a hover longer than
    /// `unflag_after_commits` itself must not self-unflag mid-hover.
    ///
    /// Self-cleaning without a keyframe-removal hook: a flagged frame that
    /// gets pruned from the live graph long before it would otherwise
    /// un-flag still ages out here purely from `now` monotonically growing
    /// (arrival indices are never reused or decreased) — `damped_frames`
    /// therefore never accumulates unbounded stale entries for frames that
    /// are long gone from the live graph, even though this state has no
    /// visibility into `DpvoPatchGraph`'s own removal bookkeeping at all.
    pub fn advance_unflagging(
        &mut self,
        now: usize,
        unflag_after_commits: usize,
        still_in_regime: bool,
    ) {
        self.last_advance_now = now;
        if still_in_regime {
            return;
        }
        let graduated: Vec<usize> = self
            .damped_frames
            .iter()
            .copied()
            .filter(|&arrival| now.saturating_sub(arrival) >= unflag_after_commits)
            .collect();
        for arrival in graduated {
            self.damped_frames.remove(&arrival);
            self.release_started_at.remove(&arrival);
            self.unflagged_total += 1;
        }
    }

    /// Milestone M16 gradual alternative to [`Self::advance_unflagging`].
    /// Existing releases advance concurrently, while at most
    /// `start_cap_frames` newly eligible cohorts enter release per commit.
    /// Oldest arrivals start first, making the schedule deterministic.
    pub fn advance_gradual_release(
        &mut self,
        now: usize,
        unflag_after_commits: usize,
        still_in_regime: bool,
        duration_commits: usize,
        start_cap_frames: usize,
    ) {
        self.last_advance_now = now;
        self.release_duration_commits = duration_commits.max(1);
        if still_in_regime {
            return;
        }

        let completed: Vec<usize> = self
            .release_started_at
            .iter()
            .filter_map(|(&arrival, &start)| {
                (now.saturating_sub(start) >= self.release_duration_commits).then_some(arrival)
            })
            .collect();
        for arrival in completed {
            self.release_started_at.remove(&arrival);
            if self.damped_frames.remove(&arrival) {
                self.unflagged_total += 1;
            }
        }

        let mut eligible: Vec<usize> = self
            .damped_frames
            .iter()
            .copied()
            .filter(|arrival| !self.release_started_at.contains_key(arrival))
            .filter(|&arrival| now.saturating_sub(arrival) >= unflag_after_commits)
            .collect();
        eligible.sort_unstable();
        let started = eligible.len().min(start_cap_frames.max(1));
        for arrival in eligible.into_iter().take(started) {
            self.release_started_at.insert(arrival, now);
        }
        self.release_started_total += started;
        self.max_release_started_per_advance = self.max_release_started_per_advance.max(started);
    }

    /// Build a per-patch [`crate::dpvo_patch_ba::DpvoBaProblem::depth_damping`]
    /// vector for a problem whose `patches` are contiguous per-frame blocks
    /// of `patches_per_frame`, one block per `frame_arrivals` entry, in the
    /// SAME order as the problem's own `poses`/`patches` — every `dpvo_ba`
    /// call site in this module satisfies this (the per-frame windowed
    /// solve, and both the legacy and widened-with-folded-prefix global-BA
    /// passes). Returns `None` (every patch's multiplier is the implicit
    /// `1.0` — see [`crate::dpvo_patch_ba::DpvoBaProblem::depth_damping`]'s
    /// own doc) whenever no currently LIVE frame in `frame_arrivals` is
    /// flagged, so a run before the first flag (or with `DepthDamp` never
    /// enabled at all, since [`Self::flag`] is then never called) pays zero
    /// extra allocation and is byte-identical to `depth_damping: None`.
    /// Increments [`Self::damped_solve_count`] exactly when it returns
    /// `Some` (i.e. this call site's own solve was genuinely affected).
    pub fn multipliers(
        &mut self,
        frame_arrivals: &[usize],
        patches_per_frame: usize,
        damp_factor: f64,
    ) -> Option<Vec<f64>> {
        if self.damped_frames.is_empty() {
            return None;
        }
        let mut any = false;
        let mut out = vec![1.0_f64; frame_arrivals.len() * patches_per_frame];
        for (local, &arrival) in frame_arrivals.iter().enumerate() {
            if self.damped_frames.contains(&arrival) {
                any = true;
                let lo = local * patches_per_frame;
                let multiplier =
                    self.release_started_at
                        .get(&arrival)
                        .map_or(damp_factor, |&start| {
                            let elapsed = self.last_advance_now.saturating_sub(start);
                            let remaining =
                                1.0 - elapsed as f64 / self.release_duration_commits.max(1) as f64;
                            damp_factor.powf(remaining.clamp(0.0, 1.0))
                        });
                out[lo..lo + patches_per_frame].fill(multiplier);
            }
        }
        if !any {
            return None;
        }
        self.damped_solve_count += 1;
        Some(out)
    }

    /// Number of frames flagged RIGHT NOW (already un-flagged frames are not
    /// counted).
    pub fn currently_damped_frames(&self) -> usize {
        self.damped_frames.len()
    }
    /// Cumulative frames ever flagged (never decremented by un-flagging).
    pub fn frames_flagged_total(&self) -> usize {
        self.frames_flagged_total
    }
    /// Cumulative patches ever flagged (`frames_flagged_total *
    /// patches_per_frame`, summed incrementally at flag time in case
    /// `patches_per_frame` were ever to change mid-run — it does not in
    /// practice, but this avoids baking in that assumption twice).
    pub fn patches_flagged_total(&self) -> usize {
        self.patches_flagged_total
    }
    /// Cumulative frames un-flagged.
    pub fn unflagged_total(&self) -> usize {
        self.unflagged_total
    }
    /// Cumulative [`Self::multipliers`] calls that returned `Some` (i.e. a
    /// real `dpvo_ba` solve was actually damped, not just evaluated).
    pub fn damped_solve_count(&self) -> usize {
        self.damped_solve_count
    }
    /// Frame cohorts currently partway through M16's gradual release.
    pub fn currently_releasing_frames(&self) -> usize {
        self.release_started_at.len()
    }
    /// Cumulative cohorts that entered gradual release.
    pub fn release_started_total(&self) -> usize {
        self.release_started_total
    }
    /// Largest number of cohorts started by one advance call.
    pub fn max_release_started_per_advance(&self) -> usize {
        self.max_release_started_per_advance
    }
    /// `[fully_damped, 0-25%, 25-50%, 50-75%, 75-100% released]` cohorts.
    pub fn release_histogram_frames(&self) -> [usize; 5] {
        let mut histogram = [0usize; 5];
        for arrival in &self.damped_frames {
            let Some(start) = self.release_started_at.get(arrival) else {
                histogram[0] += 1;
                continue;
            };
            let elapsed = self.last_advance_now.saturating_sub(*start);
            let progress = elapsed as f64 / self.release_duration_commits.max(1) as f64;
            let bucket = 1 + (progress.clamp(0.0, 0.999_999) * 4.0) as usize;
            histogram[bucket] += 1;
        }
        histogram
    }
}

/// Milestone M15: [`DpvoOdometry::low_parallax_gate`]'s outcome — separated
/// from a plain `bool` (M14's own return type) because
/// [`LowParallaxResponse::DepthDamp`] needs to communicate two INDEPENDENT
/// things to [`DpvoOdometry::process_frame`]: whether to reject the
/// candidate (only ever `true` under [`LowParallaxResponse::Freeze`]) and
/// whether to flag it once committed (only ever `true` under `DepthDamp`) —
/// the two responses are mutually exclusive by construction (see
/// [`DpvoOdometry::low_parallax_gate`]'s own `match`), but keeping both
/// fields rather than a tri-state enum keeps the caller's own `if`/`else`
/// shape unchanged from M14.
struct LowParallaxGateOutcome {
    /// The candidate must be rejected via
    /// [`crate::dpvo_patch_graph::DpvoPatchGraph::reject_pending_frame`] —
    /// the caller must return `Ok(None)` without calling `commit_frame`.
    reject: bool,
    /// The candidate, once (normally) committed, must be flagged into
    /// [`LowParallaxDampState`].
    flag_on_commit: bool,
}

/// Milestone M8 (`docs/dpvo_droid_port_plan.md`): configuration for the
/// periodic full-graph "global" bundle adjustment described on
/// [`DpvoOdometryConfig::global_ba`]'s own doc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoGlobalBaConfig {
    /// Re-check throttle, in committed (live) frames since the last global-BA
    /// call. Upstream's own `GLOBAL_OPT_FREQ` (`config.py`) drives BOTH the
    /// loop-candidate search throttle (`DpvoLoopClosureConfig::global_opt_freq`)
    /// AND `__run_global_BA`'s own "already ran this frame" check
    /// (`dpvo.py:449-455`'s `self.ran_global_ba[self.n]`) from the SAME
    /// config field — kept here as an INDEPENDENT knob instead, since this
    /// port's own loop-closure throttle and this global-BA throttle are two
    /// separate call sites with two separate "due" clocks (see
    /// [`DpvoOdometry::try_global_ba`]'s own doc for why unifying them would
    /// not actually simplify anything here). Default `15`, matching
    /// upstream's own number.
    pub frequency: usize,
    /// Gauss-Newton iteration count for this pass's `dpvo_ba` call — bounded
    /// independently from the ordinary per-frame windowed BA's own
    /// (hardcoded) `2`, since a global pass's free-pose count can be far
    /// larger and this is the knob most worth lowering if a real run shows
    /// the global pass dominating the per-frame budget (see
    /// `docs/dpvo_droid_port_plan.md`'s M8 results for the measured cost).
    /// Default `2`, matching upstream's own `iterations=2`
    /// (`dpvo.py:325`).
    pub iterations: usize,
    /// `ba.py`'s own `ep` (pose-diagonal damping) for this pass. Default
    /// `100.0`, matching [`DpvoOdometryConfig::ba_ep`]'s own default.
    pub ep: f64,
    /// `ba.py`'s own `lmbda` (depth-channel Tikhonov) for this pass.
    /// Default `1e-4`, matching [`DpvoOdometryConfig::ba_lmbda`]'s own
    /// default.
    pub lmbda: f64,
    /// Bound on [`crate::dpvo_patch_graph::DpvoPatchGraph`]'s retained
    /// inactive-edge store
    /// (`crate::dpvo_patch_graph::DpvoPatchGraph::enable_inactive_edge_retention`'s
    /// own cap argument, applied once by [`DpvoOdometry::new`]). See
    /// `docs/dpvo_droid_port_plan.md`'s M8 results for the CPU-cost
    /// reasoning behind the chosen default, and the M10 results for why the
    /// STORE's own eviction policy changed even though this cap number did
    /// not (`crate::dpvo_patch_graph::DpvoPatchGraph::inactive_edges`'s own
    /// doc).
    pub inactive_edge_cap: usize,
    /// Milestone M10 (`docs/dpvo_droid_port_plan.md`): widen this pass's own
    /// free-pose gauge (`t0`) using every accepted proximity-loop edge's OLD
    /// endpoint, decoupled from `optimization_window`'s unrelated per-frame
    /// sizing purpose — see [`gather_widened_global_ba_problem`]'s own doc
    /// for the full mechanism, and `docs/dpvo_droid_port_plan.md`'s M8
    /// results, "Why the global pass barely moved anything", for the exact
    /// finding this answers (`last_free_pose_count` pinned at
    /// `removal_window` on every M8 call because `t0` was computed from
    /// ACTIVE edges only, and the loop-protection exemption keeping a loop
    /// edge active rarely survived long enough to be seen). `false`
    /// (default) preserves M8's exact `t0 = min(active edges' owner frame)`
    /// behavior byte-for-byte — this field, not `inactive_edge_cap` above,
    /// is the actual "M10 on/off" switch.
    pub widen_t0_with_loop_edges: bool,
    /// Milestone M10: hard cap on the free-pose count (`n - t0`) a widened
    /// pass will ever solve over. Only meaningful once
    /// [`Self::widen_t0_with_loop_edges`] is `true` — a non-widened pass's
    /// own free-pose count is already bounded by `removal_window`/the
    /// loop-protection exemption and never needs this. See
    /// [`gather_widened_global_ba_problem`]'s own doc, "Cost bounds", for
    /// why the dense pose Hessian this solves (`dpvo_patch_ba.rs`'s own
    /// `DMatrix::zeros(6 * n2, 6 * n2)`) makes an unbounded widened window
    /// a real risk once a loop reaches back hundreds of frames, and why
    /// this is a knob rather than a silent internal truncation — a capped
    /// call is always reported via
    /// [`DpvoGlobalBaDiagnostics::last_free_pose_count_capped`], never
    /// silently narrower than what the loop evidence would otherwise
    /// justify. Default `Some(256)`: chosen as a dense-solve cost budget
    /// (a `1536×1536` pose Hessian), not a data-derived number — see the
    /// M10 results section for the measured per-call cost at this default.
    pub max_free_poses: Option<usize>,
}

impl Default for DpvoGlobalBaConfig {
    fn default() -> Self {
        Self {
            frequency: 15,
            iterations: 2,
            ep: 100.0,
            lmbda: 1e-4,
            inactive_edge_cap: 4096,
            widen_t0_with_loop_edges: false,
            max_free_poses: Some(256),
        }
    }
}

/// Milestone M8 snapshot of [`DpvoOdometry`]'s global-BA state — see
/// [`DpvoOdometry::global_ba_diagnostics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoGlobalBaDiagnostics {
    /// Whether `config.global_ba` is `Some` at all.
    pub enabled: bool,
    /// Total number of times [`DpvoOdometry::run_global_ba`] actually ran a
    /// `dpvo_ba` solve (not merely "was due" — a due call with zero
    /// resolvable edges returns early without counting, see that method's
    /// own doc).
    pub calls: usize,
    /// Currently retained inactive-edge count
    /// (`DpvoPatchGraph::inactive_edge_stats`'s first element).
    pub inactive_edges_retained: usize,
    /// Cumulative evicted-due-to-cap count
    /// (`DpvoPatchGraph::inactive_edge_stats`'s second element).
    pub inactive_edges_evicted_total: usize,
    /// Free-pose count (`n - t0`) of the MOST RECENT global-BA call — the
    /// direct driver of that call's own dense-solve cost (see
    /// `docs/dpvo_droid_port_plan.md`'s M8 results for the measured
    /// relationship).
    pub last_free_pose_count: usize,
    /// Total edge count (active + resolved inactive) fed into the most
    /// recent call's `dpvo_ba` problem.
    pub last_edge_count: usize,
    /// How many retained inactive edges resolved to a still-live frame pair
    /// on the most recent call (see [`DpvoOdometry::run_global_ba`]'s doc
    /// for why some may not).
    pub last_resolved_inactive_edges: usize,
    /// How many did not (their endpoint frame has since been folded away
    /// entirely) — informational only, not an error.
    pub last_unresolved_inactive_edges: usize,
    /// Largest pose-translation delta (meters), among the free
    /// `[t0, n)` range, comparing that pose immediately before vs.
    /// immediately after the most recent global-BA solve.
    pub last_pose_delta_max_m: f64,
    /// Mean of the same per-pose delta.
    pub last_pose_delta_mean_m: f64,
    /// Wall-clock cost of the most recent call, milliseconds.
    pub last_elapsed_ms: f64,
    /// Cumulative wall-clock cost across every call, milliseconds.
    pub total_elapsed_ms: f64,
    /// Milestone M10: largest [`Self::last_free_pose_count`] ever observed
    /// across every call this run — the acceptance diagnostic for "did
    /// `free_pose_count` ever exceed the ordinary `removal_window` bound and
    /// reach toward a loop endpoint," answerable from a live run without
    /// grepping the console log the way M8's own diagnosis needed to.
    pub max_free_pose_count: usize,
    /// Milestone M10: whether the MOST RECENT call's free-pose window was
    /// widened by a known loop edge's endpoint beyond where an ordinary
    /// active-edges-only `t0` would have landed (`false` on every call while
    /// `DpvoGlobalBaConfig::widen_t0_with_loop_edges` is `false`, by
    /// construction).
    pub last_t0_widened_by_loop_edge: bool,
    /// Milestone M10: how many FOLDED (no longer live) frames were
    /// materialized as free pose variables on the most recent call, using
    /// `crate::dpvo_patch_graph::DpvoPatchGraph::retained_poses`/
    /// `retained_folded_frames` — `0` whenever the loop's own old endpoint
    /// was still live (the common case on this port's real MH_01 runs, see
    /// the M10 results) or widening is disabled.
    pub last_folded_poses_included: usize,
    /// Milestone M10: whether `DpvoGlobalBaConfig::max_free_poses` actually
    /// clamped the most recent call's free-pose window short of what the
    /// known loop evidence would otherwise justify — see that field's own
    /// doc; never silent, always visible here.
    pub last_free_pose_count_capped: bool,
}

/// Milestone M9 snapshot of [`DpvoOdometry`]'s Sim(3) pose-graph backend
/// state — see [`DpvoOdometry::sim3_backend_diagnostics`] and
/// `crate::dpvo_sim3_backend`'s module doc for the mechanism this reports on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoSim3BackendDiagnostics {
    /// Whether `config.sim3_backend` is `Some` at all.
    pub enabled: bool,
    /// Total number of times [`DpvoOdometry::try_sim3_backend`] actually ran
    /// a solve (not merely "was due" — a due call with zero resolvable loop
    /// edges returns early without counting, mirroring
    /// [`DpvoGlobalBaDiagnostics::calls`]'s own contract).
    pub calls: usize,
    /// Cumulative distinct loop measurements ever recorded (captured at
    /// proximity-loop-acceptance time — see
    /// `crate::dpvo_sim3_backend::Sim3LoopMeasurement`), whether or not they
    /// have been consumed by a solve yet.
    pub loop_edges_total: usize,
    /// `Sim3PoseGraph` node count (after subsampling) of the MOST RECENT
    /// call.
    pub last_node_count: usize,
    /// Total edges (sequential + loop) fed into the most recent call's
    /// solve.
    pub last_edge_count: usize,
    /// Of those, how many were loop edges.
    pub last_loop_edges_used: usize,
    /// How many retained + live poses received a (possibly interpolated)
    /// correction on the most recent call — the task's own "scale
    /// corrections applied" diagnostic.
    pub last_scale_corrections_applied: usize,
    /// Largest observed pose-translation correction (meters) across every
    /// corrected pose on the most recent call.
    pub last_pose_delta_max_m: f64,
    /// Mean of the same per-pose correction magnitude.
    pub last_pose_delta_mean_m: f64,
    /// Smallest/largest solved-or-interpolated `Sim(3)` scale across every
    /// corrected pose on the most recent call (`1.0`/`1.0` if nothing was
    /// corrected).
    pub last_scale_min: f64,
    pub last_scale_max: f64,
    /// Largest absolute log-scale correction that was ever COMMITTED during
    /// this run. Rejected proposals never contribute, so this is the direct
    /// trajectory scale-cliff diagnostic rather than a proposal diagnostic.
    pub max_committed_abs_log_scale: f64,
    /// Number of proposals rejected transactionally by the scale-jump gate.
    pub scale_jump_rejections_total: usize,
    /// Whether the most recent solved proposal passed all transactional
    /// write-back gates.
    pub last_committed: bool,
    pub last_rejection: Option<Sim3BackendRejection>,
    /// Wall-clock cost of the most recent call, milliseconds.
    pub last_elapsed_ms: f64,
    /// Cumulative wall-clock cost across every call, milliseconds.
    pub total_elapsed_ms: f64,
}

fn update_sim3_scale_cliff_diagnostics(
    max_committed_abs_log_scale: &mut f64,
    scale_jump_rejections_total: &mut usize,
    committed: bool,
    rejection: Option<Sim3BackendRejection>,
    scale_min: f64,
    scale_max: f64,
) {
    if committed {
        let max_abs_log_scale = scale_min.ln().abs().max(scale_max.ln().abs());
        *max_committed_abs_log_scale = (*max_committed_abs_log_scale).max(max_abs_log_scale);
    }
    if rejection == Some(Sim3BackendRejection::ScaleJump) {
        *scale_jump_rejections_total += 1;
    }
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
    /// window's condition number `≁E61` (comfortably below) vs. a
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
        Self {
            scale: ScaleCouplingConfig::default(),
            min_window_factors: 4,
        }
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

/// Milestone M14 snapshot of [`DpvoOdometry`]'s low-parallax hover-freeze
/// state — see [`DpvoOdometry::low_parallax_diagnostics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoLowParallaxDiagnostics {
    /// Whether `config.low_parallax` is `Some` at all — `false` means every
    /// other field here is a static default (the mechanism was never
    /// engaged), not a claim that a hover was searched for and never found.
    pub enabled: bool,
    /// Whether the regime is active RIGHT NOW (after the most recent
    /// `process_frame` call).
    pub regime_active: bool,
    /// Cumulative times the regime was entered.
    pub times_entered: usize,
    /// Cumulative times the regime was exited.
    pub times_exited: usize,
    /// Cumulative candidate frames suppressed (rejected via
    /// `DpvoPatchGraph::reject_pending_frame`) because the regime was
    /// active — the direct "how much of the hover did this actually
    /// freeze" measurement.
    pub frames_suppressed_total: usize,
    /// Whether the one-shot guard has fired (see
    /// [`LowParallaxRegimeState::update`]'s own doc) — once `true`, the
    /// detector is permanently inert for the rest of the run, by design
    /// (see [`DpvoLowParallaxConfig::default`]'s own doc, finding 3, for
    /// the real-run evidence this guards against).
    pub disarmed: bool,
    /// The most recently evaluated frame's own [`DpvoOdometry::motion_probe`]
    /// value.
    pub last_flow: f64,
    /// `stats.frames_processed` at the moment the regime was most recently
    /// entered (`None` if never).
    pub last_enter_frame: Option<usize>,
    /// `stats.frames_processed` at the moment the regime was most recently
    /// exited (`None` if never).
    pub last_exit_frame: Option<usize>,
    /// Milestone M15: which response is configured — a static default
    /// ([`LowParallaxResponse::Freeze`]) whenever `enabled` is `false`.
    pub response: LowParallaxResponse,
    /// Milestone M15 (`DepthDamp` only, `0` otherwise): frames flagged RIGHT
    /// NOW (already un-flagged frames are not counted) — see
    /// [`LowParallaxDampState::currently_damped_frames`].
    pub currently_damped_frames: usize,
    /// Milestone M15: cumulative frames ever flagged — see
    /// [`LowParallaxDampState::frames_flagged_total`].
    pub frames_flagged_total: usize,
    /// Milestone M15: cumulative patches ever flagged — see
    /// [`LowParallaxDampState::patches_flagged_total`].
    pub patches_flagged_total: usize,
    /// Milestone M15: cumulative frames un-flagged — see
    /// [`LowParallaxDampState::unflagged_total`].
    pub unflagged_total: usize,
    /// Milestone M15: cumulative `dpvo_ba` solves that were genuinely
    /// damped (i.e. at least one currently-live frame in that solve's own
    /// window was flagged) — see [`LowParallaxDampState::damped_solve_count`].
    pub damped_solve_count: usize,
    /// Milestone M16: cohorts currently between full damping and 1.0.
    pub currently_releasing_frames: usize,
    /// Milestone M16: cumulative cohorts admitted to gradual release.
    pub release_started_total: usize,
    /// Milestone M16: observed one-commit release-start maximum.
    pub max_release_started_per_advance: usize,
    /// Milestone M16 release progress histogram; see
    /// [`LowParallaxDampState::release_histogram_frames`].
    pub release_histogram_frames: [usize; 5],
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
    GyroGate {
        reason: GyroGateRejection,
        bias_norm: f64,
        rms_before: f64,
        rms_after: f64,
    },
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
    /// Upstream `DPVO::terminate()` runs 12 final update/BA iterations
    /// before exporting the trajectory. This counts those iterations.
    pub final_refinement_iterations: usize,
    pub encode_ms_total: f64,
    /// Time spent reprojecting every active edge's patch grid
    /// (`reproject_patch_grid`) and assembling the 2-pyramid-level
    /// correlation tensor (`corr_pyramid`/`corr_cpu`) — measured
    /// separately from `update_ms_total` because it turned out to
    /// dominate this naive CPU port's per-frame cost (see the plan doc's
    /// M4 results, "timing breakdown").
    pub correlation_ms_total: f64,
    /// CUDA event time reported by the native correlation DLL, including
    /// its H2D/D2H copies and kernel, but excluding Rust-side assembly.
    pub native_correlation_device_ms_total: f64,
    /// Time spent inside the ONNX GRU update-cell call
    /// (`DpvoOnnxSession::update_iteration`) only.
    pub update_ms_total: f64,
    pub ba_ms_total: f64,
}

/// Encoder output for one input frame, prepared independently of the
/// stateful patch graph. Keeping this boundary explicit lets streaming
/// callers use a bounded one-frame look-ahead queue: frame N+1's independent
/// CNN work can overlap frame N's correlation/update/BA without dropping or
/// reordering any camera sample.
pub struct DpvoEncodedFrame {
    fmap: Array3<f32>,
    imap: Array3<f32>,
    encode_ms: f64,
}

/// Cheaply cloned, thread-safe handle to DPVO's two image encoders.
#[derive(Clone)]
pub struct DpvoFrameEncoder {
    session: DpvoOnnxSession,
    width: usize,
    height: usize,
    concurrent_pair: bool,
}

impl DpvoFrameEncoder {
    pub fn encode(&self, image: ArrayView2<'_, u8>) -> Result<DpvoEncodedFrame, DpvoOdometryError> {
        let (h, w) = image.dim();
        if (w, h) != (self.width, self.height) {
            return Err(DpvoOdometryError::ImageShapeMismatch {
                expected: (self.width, self.height),
                actual: (w, h),
            });
        }
        let started = Instant::now();
        let input = grayscale_to_input_tensor(image);
        let (fmap4, imap4) = if self.concurrent_pair {
            self.session.run_encoders(input.view())?
        } else {
            self.session.run_encoders_serial(input.view())?
        };
        Ok(DpvoEncodedFrame {
            // Encoder outputs own their allocations already; move away the
            // singleton batch axis instead of copying either feature map.
            fmap: fmap4.index_axis_move(Axis(0), 0),
            imap: imap4.index_axis_move(Axis(0), 0),
            encode_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

/// One live frame's cached `fnet` feature pyramid (`DPVO.pyramid`,
/// `dpvo.py:72-76`): `level0` is the full-stride-4 feature map
/// (`avg_pool2d(fmap, 1, 1)` — a no-op, kept only for naming symmetry with
/// upstream), `level1` is a further ×4 average-pooled map
/// (`avg_pool2d(fmap, 4, 4)`, `dpvo.py:438`).
///
/// # M4-perf: cached CPU and optional fused-CUDA layouts
///
/// On the CPU path both levels are stored as [`ChannelLastImage`] — already
/// transposed into the layout [`corr_cpu_prebuilt_target`] needs — rather than the raw
/// `(channels, height, width)` `ort`/`avg_pool_4x4` output. A frame's pyramid
/// is built exactly once (`process_frame`, when the frame arrives) but read
/// by [`corr_pyramid`] once per active edge group per `update_step`/
/// `motion_probe` call for as long as the frame stays inside the active
/// window (up to `REMOVAL_WINDOW`/`PATCH_LIFETIME` calls) — profiling
/// (plan doc, "M4-perf results") showed re-transposing the same ~120x188x128
/// feature map from scratch on every one of those reads was a real,
/// avoidable cost, not just a theoretical one. Nothing outside this module
/// reads `level0`/`level1` in channel-first form on the legacy path. A model
/// bundle containing `dpvo_corr_pyramid.onnx`, or a native CUDA runtime,
/// stores only the two channel-first tensors. Keeping both layouts used to
/// copy and transpose another ~13 MB for every EuRoC frame even though the
/// CUDA path never read the HWC copies.
#[derive(Debug, Clone)]
struct FramePyramid {
    level0: Option<ChannelLastImage>,
    level1: Option<ChannelLastImage>,
    level0_chw: Option<Array3<f32>>,
    level1_chw: Option<Array3<f32>>,
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
    native_correlation: Option<NativeCudaCorrelation>,
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
    trajectory_finalized: bool,

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
    /// Latest camera arrival whose visual/IMU window was admitted as NEW
    /// recursive evidence. `update_step` legitimately runs multiple solver
    /// iterations for one arrival (including 12 final-refinement passes),
    /// but those iterations are not independent sensor measurements.
    scale_coupling_last_evidence_arrival: Option<usize>,
    scale_coupling_consecutive_bad: usize,
    scale_coupling_measurements: usize,
    scale_coupling_measurement_rejections: usize,
    scale_coupling_rollback_count: usize,
    /// Diagnostic instrumentation (see [`DpvoScaleCouplingDiagnostics::rejection_counts`]).
    scale_coupling_rejection_counts: DpvoScaleCouplingRejectionCounts,
    scale_coupling_last_rejection: Option<DpvoMonoViAlignmentRejection>,

    // ---- Milestone M8 (global BA over retained active+inactive edges)
    // state — see the module doc's "Global BA (Milestone M8)" section and
    // [`Self::run_global_ba`]/[`Self::try_global_ba`]'s own docs. Inert
    // (never read/updated meaningfully) whenever `config.global_ba` is
    // `None`. ----
    /// Live-frame index `n` at the last `try_global_ba` call that actually
    /// ran a solve, or `None` before the first one — mirrors
    /// `last_loop_batch_frame`'s own "always eligible on the very next
    /// frame" semantics.
    last_global_ba_frame: Option<usize>,
    /// Whether a loop edge has EVER been accepted — see
    /// [`Self::try_global_ba`]'s doc for why the whole mechanism stays a
    /// no-op until this is `true` (a global pass is redundant, strictly
    /// more expensive work whenever `t0` cannot differ from the ordinary
    /// per-frame window bound).
    global_ba_ever_had_loop_edge: bool,
    global_ba_calls: usize,
    global_ba_ms_total: f64,
    global_ba_last_ms: f64,
    global_ba_last_free_pose_count: usize,
    global_ba_last_edge_count: usize,
    global_ba_last_resolved_inactive: usize,
    global_ba_last_unresolved_inactive: usize,
    global_ba_last_pose_delta_max_m: f64,
    global_ba_last_pose_delta_mean_m: f64,
    /// Milestone M10: every accepted proximity-loop pair EVER seen, as
    /// `(arrival_i, arrival_j)` — persists independently of whether the
    /// patch-graph edge itself is still active, resolvable-inactive, or has
    /// aged out of both (arrival indices never change, unlike live indices —
    /// see `crate::dpvo_patch_graph::DpvoGraphFrame::arrival_index`'s own
    /// doc). This is the actual "currently-known loop edges" list
    /// [`gather_widened_global_ba_problem`]'s own doc, part A, computes `t0`
    /// from — a deliberately separate, unbounded-but-tiny store (at most
    /// `loop_accepted_total` entries) from `crate::dpvo_sim3_backend`'s own
    /// `Sim3LoopMeasurement` list, since that one is gated on
    /// `config.sim3_backend` while this one is gated on
    /// `config.global_ba.map(|c| c.widen_t0_with_loop_edges)`.
    loop_edge_arrival_pairs: Vec<(usize, usize)>,
    global_ba_max_free_pose_count: usize,
    global_ba_last_widened: bool,
    global_ba_last_folded_included: usize,
    global_ba_last_capped: bool,

    // ---- Milestone M9 (Sim(3) pose-graph scale-drift backend) state — see
    // `crate::dpvo_sim3_backend`'s module doc and
    // [`Self::run_sim3_backend`]/[`Self::try_sim3_backend`]'s own docs.
    // Inert whenever `config.sim3_backend` is `None`. ----
    /// Every proximity loop pair ever accepted, frozen as a `Sim(3)`
    /// measurement — see
    /// `crate::dpvo_sim3_backend::Sim3LoopMeasurement`'s own doc and
    /// [`Self::capture_pending_sim3_loop_measurements`]'s doc for exactly
    /// WHEN each entry's `relative_pose` is captured (deliberately not the
    /// instant `try_loop_closure` accepts the pair). Collected only when
    /// `config.sim3_backend` is `Some` (cheap: one small struct per accepted
    /// batch, never more than `loop_accepted_total` entries).
    sim3_loop_measurements: Vec<Sim3LoopMeasurement>,
    /// Live-index `(i, j)` pairs accepted by [`Self::try_loop_closure`] this
    /// same frame but not yet frozen into a [`Sim3LoopMeasurement`] — see
    /// [`Self::capture_pending_sim3_loop_measurements`]'s own doc. Always
    /// drained (to empty) by the very next call to that method, which
    /// `process_frame` makes unconditionally right after `update_step`
    /// every frame — never carries over across a `process_frame` call.
    pending_sim3_loop_pairs: Vec<(usize, usize)>,
    /// Mirrors `global_ba_ever_had_loop_edge`'s own no-op-until-evidence
    /// gate — see [`Self::try_sim3_backend`]'s doc.
    sim3_backend_ever_had_loop_edge: bool,
    last_sim3_backend_frame: Option<usize>,
    sim3_backend_calls: usize,
    sim3_backend_ms_total: f64,
    sim3_backend_last_ms: f64,
    sim3_backend_last_node_count: usize,
    sim3_backend_last_edge_count: usize,
    sim3_backend_last_loop_edges_used: usize,
    sim3_backend_last_corrected_pose_count: usize,
    sim3_backend_last_pose_delta_max_m: f64,
    sim3_backend_last_pose_delta_mean_m: f64,
    sim3_backend_last_scale_min: f64,
    sim3_backend_last_scale_max: f64,
    sim3_backend_max_committed_abs_log_scale: f64,
    sim3_backend_scale_jump_rejections_total: usize,
    sim3_backend_last_committed: bool,
    sim3_backend_last_rejection: Option<Sim3BackendRejection>,

    // ---- Milestone M11 (long-range appearance loop candidate source) state
    // — see `crate::dpvo_long_loop`'s module doc. `None` whenever
    // `config.long_loop` is `None` (every prior milestone's default): no
    // SuperPoint inference runs, `process_frame`'s own long-loop block is a
    // no-op. ----
    long_loop: Option<DpvoLongLoopRuntime>,

    // ---- Milestone M14 (low-parallax hover freeze) state — see the module
    // doc's own "Low-parallax hover freeze" section and
    // [`LowParallaxRegimeState`]'s own doc. Inert (never evaluated at all)
    // whenever `config.low_parallax` is `None`. ----
    low_parallax_regime: LowParallaxRegimeState,
    low_parallax_times_entered: usize,
    low_parallax_times_exited: usize,
    low_parallax_frames_suppressed_total: usize,
    low_parallax_last_flow: f64,
    low_parallax_last_enter_frame: Option<usize>,
    low_parallax_last_exit_frame: Option<usize>,
    /// `(frames_processed, flow, regime_active_after_this_frame)` for every
    /// frame the detector was evaluated on — the acceptance-run evidence for
    /// "did this fire at the right place, for the right duration" (see
    /// [`DpvoOdometry::low_parallax_flow_log`]). Bounded implicitly by run
    /// length (one entry per processed frame, at most), never by an
    /// explicit cap — MH_01-scale runs (hundreds to low thousands of
    /// frames) keep this negligible.
    low_parallax_flow_log: Vec<(usize, f64, bool)>,
    /// Milestone M15 (`LowParallaxResponse::DepthDamp` only): the flag/
    /// un-flag bookkeeping described on [`LowParallaxDampState`]'s own doc.
    /// Stays at its `Default` (empty, all counters `0`) for the entire run
    /// whenever `response != DepthDamp` — [`Self::process_frame`] only ever
    /// calls [`LowParallaxDampState::flag`] from the `DepthDamp` branch of
    /// [`Self::low_parallax_gate`].
    low_parallax_damp: LowParallaxDampState,

    /// A3 ranking-lab offline dump: `(arrival_index, keypoints, descriptors)`
    /// for the MOST RECENTLY ingested long-loop frame, captured only when
    /// `config.long_loop_dump_enabled` is `true` — see that field's own doc.
    /// Overwritten on every ingest, never accumulated (the demo reads this
    /// once per `process_frame` call, right after it returns, and writes it
    /// to disk itself — see [`DpvoOdometry::long_loop_last_ingested`]).
    long_loop_last_ingested: Option<(usize, Vec<nalgebra::Point2<f64>>, Vec<Vec<f32>>)>,
}

/// Milestone M11: the ONNX-runtime-dependent half of the long-range loop
/// mechanism (the SuperPoint session) paired with the pure-logic index/
/// verifier (`crate::dpvo_long_loop::DpvoLongLoopIndex`) — kept as one
/// `Option` field on [`DpvoOdometry`] so both are constructed/dropped
/// together.
struct DpvoLongLoopRuntime {
    extractor: SuperPointOnnxExtractor,
    index: DpvoLongLoopIndex,
}

impl DpvoOdometry {
    /// Load the four M1-exported ONNX graphs plus the `SoftAgg` weight
    /// artifact (`softagg_weights_fixture.npz` — see
    /// `crates/vision/src/dpvo/mod.rs`'s module doc, "Why does `SoftAgg`
    /// need to load weights from an npz at all?"; this is the *same*
    /// checkpoint-derived artifact M2 produced, reused here as-is rather
    /// than re-exported under a new name, since the weights it carries are
    /// already real, just fixture-shaped file-naming). When the same model
    /// directory also contains `dpvo_update_full.onnx`, the session uses
    /// that fused graph and retains the split artifacts as compatibility
    /// fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: DpvoOdometryConfig,
        fnet_path: impl AsRef<Path>,
        inet_path: impl AsRef<Path>,
        update_pre_agg_path: impl AsRef<Path>,
        update_post_agg_path: impl AsRef<Path>,
        softagg_weights_npz_path: impl AsRef<Path>,
        backend: OnnxBackend,
        // Milestone M11: required (returns `DpvoOdometryError::LongLoopModelRequired`
        // otherwise) whenever `config.long_loop` is `Some` — reuses `backend`
        // above for the SuperPoint session too (one shared execution-provider
        // choice, not a second knob). `None` whenever `config.long_loop` is
        // `None`, matching every other optional-mechanism argument's own
        // "byte-identical when absent" contract.
        superpoint_model_path: Option<impl AsRef<Path>>,
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
        let long_loop = match (config.long_loop.clone(), superpoint_model_path) {
            (Some(ll_cfg), Some(model_path)) => {
                let sp_config = SuperPointOnnxConfig {
                    // Milestone M11: bounds this module's own per-frame
                    // memory footprint (see `crate::dpvo_long_loop`'s module
                    // doc, "Failure modes") — deliberately smaller than
                    // `SuperPointOnnxConfig::default()`'s own `1500` (tuned
                    // for full place-recognition retrieval, not this bounded
                    // streaming index).
                    max_keypoints: 250,
                    ..SuperPointOnnxConfig::default()
                };
                let extractor = SuperPointOnnxExtractor::load_from_path_with_backend(
                    model_path, sp_config, backend,
                )
                .map_err(DpvoOdometryError::LongLoop)?;
                Some(DpvoLongLoopRuntime {
                    extractor,
                    index: DpvoLongLoopIndex::new(ll_cfg),
                })
            }
            (Some(_), None) => return Err(DpvoOdometryError::LongLoopModelRequired),
            (None, _) => None,
        };
        let mut graph = DpvoPatchGraph::new(config.vo);
        // Milestone M8: opt the patch graph into inactive-edge retention
        // only when a caller actually asked for the global-BA mechanism —
        // every M4-M7 call site (`config.global_ba: None`) leaves the graph
        // exactly as before (cap `0`, retention fully disabled).
        if let Some(gba) = config.global_ba {
            graph.enable_inactive_edge_retention(gba.inactive_edge_cap);
        }
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
        let native_correlation = config
            .native_cuda_correlation_dll
            .as_ref()
            .map(NativeCudaCorrelation::load)
            .transpose()?;
        Ok(Self {
            config,
            session,
            native_correlation,
            agg_kk,
            agg_ij,
            graph,
            frame_pyramids: Vec::new(),
            patch_gmap: Vec::new(),
            patch_imap: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            stats: DpvoOdometryStats::default(),
            trajectory_finalized: false,
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
            scale_coupling_last_evidence_arrival: None,
            scale_coupling_consecutive_bad: 0,
            scale_coupling_measurements: 0,
            scale_coupling_measurement_rejections: 0,
            scale_coupling_rollback_count: 0,
            scale_coupling_rejection_counts: DpvoScaleCouplingRejectionCounts::default(),
            scale_coupling_last_rejection: None,
            last_global_ba_frame: None,
            global_ba_ever_had_loop_edge: false,
            global_ba_calls: 0,
            global_ba_ms_total: 0.0,
            global_ba_last_ms: 0.0,
            global_ba_last_free_pose_count: 0,
            global_ba_last_edge_count: 0,
            global_ba_last_resolved_inactive: 0,
            global_ba_last_unresolved_inactive: 0,
            global_ba_last_pose_delta_max_m: 0.0,
            global_ba_last_pose_delta_mean_m: 0.0,
            loop_edge_arrival_pairs: Vec::new(),
            global_ba_max_free_pose_count: 0,
            global_ba_last_widened: false,
            global_ba_last_folded_included: 0,
            global_ba_last_capped: false,
            sim3_loop_measurements: Vec::new(),
            pending_sim3_loop_pairs: Vec::new(),
            sim3_backend_ever_had_loop_edge: false,
            last_sim3_backend_frame: None,
            sim3_backend_calls: 0,
            sim3_backend_ms_total: 0.0,
            sim3_backend_last_ms: 0.0,
            sim3_backend_last_node_count: 0,
            sim3_backend_last_edge_count: 0,
            sim3_backend_last_loop_edges_used: 0,
            sim3_backend_last_corrected_pose_count: 0,
            sim3_backend_last_pose_delta_max_m: 0.0,
            sim3_backend_last_pose_delta_mean_m: 0.0,
            sim3_backend_last_scale_min: 1.0,
            sim3_backend_last_scale_max: 1.0,
            sim3_backend_max_committed_abs_log_scale: 0.0,
            sim3_backend_scale_jump_rejections_total: 0,
            sim3_backend_last_committed: false,
            sim3_backend_last_rejection: None,
            long_loop,
            low_parallax_regime: LowParallaxRegimeState::default(),
            low_parallax_times_entered: 0,
            low_parallax_times_exited: 0,
            low_parallax_frames_suppressed_total: 0,
            low_parallax_last_flow: 0.0,
            low_parallax_last_enter_frame: None,
            low_parallax_last_exit_frame: None,
            low_parallax_flow_log: Vec::new(),
            low_parallax_damp: LowParallaxDampState::default(),
            long_loop_last_ingested: None,
        })
    }

    pub fn stats(&self) -> DpvoOdometryStats {
        self.stats
    }

    /// Return a thread-safe encoder handle for bounded streaming prefetch.
    /// It shares the already-loaded ORT sessions; no model or weights are
    /// duplicated.
    pub fn frame_encoder(&self) -> DpvoFrameEncoder {
        DpvoFrameEncoder {
            session: self.session.clone(),
            width: self.config.width,
            height: self.config.height,
            concurrent_pair: true,
        }
    }

    /// Encoder handle tuned for a frame-ahead producer: fnet and inet run
    /// serially while their combined work overlaps the previous frame's
    /// tracking backend.
    pub fn serial_frame_encoder(&self) -> DpvoFrameEncoder {
        DpvoFrameEncoder {
            session: self.session.clone(),
            width: self.config.width,
            height: self.config.height,
            concurrent_pair: false,
        }
    }

    pub fn full_update_graph_enabled(&self) -> bool {
        self.session.full_update_enabled()
    }

    pub fn correlation_graph_enabled(&self) -> bool {
        self.session.correlation_graph_enabled()
    }

    pub fn native_cuda_correlation_enabled(&self) -> bool {
        self.native_correlation.is_some()
    }

    pub fn native_cuda_correlation_abi(&self) -> Option<u32> {
        self.native_correlation
            .as_ref()
            .map(NativeCudaCorrelation::abi_version)
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
            enabled: self
                .config
                .imu
                .as_ref()
                .is_some_and(|c| c.scale_coupling.is_some()),
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

    /// Snapshot of the Milestone M8 global-BA state — see
    /// [`DpvoGlobalBaDiagnostics`].
    pub fn global_ba_diagnostics(&self) -> DpvoGlobalBaDiagnostics {
        let (retained, evicted) = self.graph.inactive_edge_stats();
        DpvoGlobalBaDiagnostics {
            enabled: self.config.global_ba.is_some(),
            calls: self.global_ba_calls,
            inactive_edges_retained: retained,
            inactive_edges_evicted_total: evicted,
            last_free_pose_count: self.global_ba_last_free_pose_count,
            last_edge_count: self.global_ba_last_edge_count,
            last_resolved_inactive_edges: self.global_ba_last_resolved_inactive,
            last_unresolved_inactive_edges: self.global_ba_last_unresolved_inactive,
            last_pose_delta_max_m: self.global_ba_last_pose_delta_max_m,
            last_pose_delta_mean_m: self.global_ba_last_pose_delta_mean_m,
            last_elapsed_ms: self.global_ba_last_ms,
            total_elapsed_ms: self.global_ba_ms_total,
            max_free_pose_count: self.global_ba_max_free_pose_count,
            last_t0_widened_by_loop_edge: self.global_ba_last_widened,
            last_folded_poses_included: self.global_ba_last_folded_included,
            last_free_pose_count_capped: self.global_ba_last_capped,
        }
    }

    /// Snapshot of the Milestone M9 Sim(3) pose-graph backend state — see
    /// [`DpvoSim3BackendDiagnostics`].
    pub fn sim3_backend_diagnostics(&self) -> DpvoSim3BackendDiagnostics {
        DpvoSim3BackendDiagnostics {
            enabled: self.config.sim3_backend.is_some(),
            calls: self.sim3_backend_calls,
            loop_edges_total: self.sim3_loop_measurements.len(),
            last_node_count: self.sim3_backend_last_node_count,
            last_edge_count: self.sim3_backend_last_edge_count,
            last_loop_edges_used: self.sim3_backend_last_loop_edges_used,
            last_scale_corrections_applied: self.sim3_backend_last_corrected_pose_count,
            last_pose_delta_max_m: self.sim3_backend_last_pose_delta_max_m,
            last_pose_delta_mean_m: self.sim3_backend_last_pose_delta_mean_m,
            last_scale_min: self.sim3_backend_last_scale_min,
            last_scale_max: self.sim3_backend_last_scale_max,
            max_committed_abs_log_scale: self.sim3_backend_max_committed_abs_log_scale,
            scale_jump_rejections_total: self.sim3_backend_scale_jump_rejections_total,
            last_committed: self.sim3_backend_last_committed,
            last_rejection: self.sim3_backend_last_rejection,
            last_elapsed_ms: self.sim3_backend_last_ms,
            total_elapsed_ms: self.sim3_backend_ms_total,
        }
    }

    /// Snapshot of the Milestone M11 long-range loop mechanism's own state —
    /// `enabled: false` (a default-zeroed struct) whenever `config.long_loop`
    /// is `None`.
    pub fn long_loop_diagnostics(&self) -> DpvoLongLoopDiagnostics {
        self.long_loop
            .as_ref()
            .map(|runtime| runtime.index.diagnostics())
            .unwrap_or_default()
    }

    /// Milestone M12 (open item 2 carried forward from M11): every top-`K`
    /// retrieval candidate ever surfaced by any long-range query this run,
    /// accepted or not — see `crate::dpvo_long_loop::QueryCandidateLogEntry`'s
    /// own doc. Empty whenever `config.long_loop` is `None`.
    pub fn long_loop_query_log(&self) -> &[QueryCandidateLogEntry] {
        self.long_loop
            .as_ref()
            .map(|runtime| runtime.index.query_log())
            .unwrap_or(&[])
    }

    /// A3 stage-1 (`docs/visual_slam_sequential_sfm_plan.md`, "densify query
    /// cadence" slice): arrival indices where a long-range query was issued
    /// but returned zero candidates after similarity/gap filtering — see
    /// `crate::dpvo_long_loop::DpvoLongLoopIndex::empty_query_arrivals`'s own
    /// doc. Empty whenever `config.long_loop` is `None`.
    pub fn long_loop_empty_query_arrivals(&self) -> &[usize] {
        self.long_loop
            .as_ref()
            .map(|runtime| runtime.index.empty_query_arrivals())
            .unwrap_or(&[])
    }

    /// A3 ranking-lab offline dump: the most recently ingested long-loop
    /// frame's `(arrival_index, keypoints, descriptors)` — `None` unless
    /// `config.long_loop_dump_enabled` is `true` AND at least one frame has
    /// been ingested since the last call site read this (the demo reads it
    /// once per `process_frame` call; see `DpvoOdometryConfig::long_loop_dump_enabled`'s
    /// own doc for exactly what "ingested" means and why this is never
    /// perturbing the odometry solve itself).
    pub fn long_loop_last_ingested(
        &self,
    ) -> Option<(usize, &[nalgebra::Point2<f64>], &[Vec<f32>])> {
        self.long_loop_last_ingested
            .as_ref()
            .map(|(arrival, keypoints, descriptors)| {
                (*arrival, keypoints.as_slice(), descriptors.as_slice())
            })
    }

    /// Snapshot of the Milestone M14 low-parallax hover-freeze state —
    /// `enabled: false` (every other field a static default) whenever
    /// `config.low_parallax` is `None`.
    pub fn low_parallax_diagnostics(&self) -> DpvoLowParallaxDiagnostics {
        DpvoLowParallaxDiagnostics {
            enabled: self.config.low_parallax.is_some(),
            regime_active: self.low_parallax_regime.in_regime(),
            times_entered: self.low_parallax_times_entered,
            times_exited: self.low_parallax_times_exited,
            frames_suppressed_total: self.low_parallax_frames_suppressed_total,
            disarmed: self.low_parallax_regime.disarmed(),
            last_flow: self.low_parallax_last_flow,
            last_enter_frame: self.low_parallax_last_enter_frame,
            last_exit_frame: self.low_parallax_last_exit_frame,
            response: self
                .config
                .low_parallax
                .map(|cfg| cfg.response)
                .unwrap_or_default(),
            currently_damped_frames: self.low_parallax_damp.currently_damped_frames(),
            frames_flagged_total: self.low_parallax_damp.frames_flagged_total(),
            patches_flagged_total: self.low_parallax_damp.patches_flagged_total(),
            unflagged_total: self.low_parallax_damp.unflagged_total(),
            damped_solve_count: self.low_parallax_damp.damped_solve_count(),
            currently_releasing_frames: self.low_parallax_damp.currently_releasing_frames(),
            release_started_total: self.low_parallax_damp.release_started_total(),
            max_release_started_per_advance: self
                .low_parallax_damp
                .max_release_started_per_advance(),
            release_histogram_frames: self.low_parallax_damp.release_histogram_frames(),
        }
    }

    /// Milestone M14: every frame the low-parallax detector was evaluated
    /// on, as `(frames_processed, flow, regime_active_after_this_frame)` —
    /// the acceptance-run profile-flatness evidence. Empty whenever
    /// `config.low_parallax` is `None`.
    pub fn low_parallax_flow_log(&self) -> &[(usize, f64, bool)] {
        &self.low_parallax_flow_log
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

    /// Match upstream `DPVO::terminate()` by refining the final live graph
    /// for 12 additional update/BA iterations before trajectory export.
    /// Repeated calls without another processed frame are idempotent.
    pub fn finalize_trajectory(&mut self) -> Result<(), DpvoOdometryError> {
        if self.trajectory_finalized || !self.graph.is_initialized() {
            self.trajectory_finalized = true;
            return Ok(());
        }
        for _ in 0..12 {
            self.update_step()?;
            self.stats.final_refinement_iterations += 1;
        }
        self.trajectory_finalized = true;
        Ok(())
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
        let encoded = self.frame_encoder().encode(image)?;
        self.process_encoded_frame(image, timestamp, encoded)
    }

    /// Process a frame whose independent CNN encoders have already run.
    /// This is semantically identical to [`Self::process_frame`]; only the
    /// scheduling boundary differs.
    pub fn process_encoded_frame(
        &mut self,
        image: ArrayView2<'_, u8>,
        timestamp: f64,
        encoded: DpvoEncodedFrame,
    ) -> Result<Option<SE3>, DpvoOdometryError> {
        let (h, w) = image.dim();
        if (w, h) != (self.config.width, self.config.height) {
            return Err(DpvoOdometryError::ImageShapeMismatch {
                expected: (self.config.width, self.config.height),
                actual: (w, h),
            });
        }
        self.trajectory_finalized = false;
        self.stats.frames_processed += 1;
        self.stats.encode_ms_total += encoded.encode_ms;

        let fmap = encoded.fmap;
        let imap_full = encoded.imap;
        let (_, hs, ws) = fmap.dim();

        // Milestone M12 (`docs/dpvo_droid_port_plan.md`, `crate::dpvo_long_loop`'s
        // module doc): extract this frame's SuperPoint features BEFORE
        // choosing patch centers below (M11 extracted them only AFTER
        // `commit_frame`, purely for retrieval indexing) so that, when
        // `sp_anchored_patches` is enabled, the patch sampler can anchor
        // centers at the SAME keypoints this frame will also index for
        // long-range retrieval — computed ONCE, reused for BOTH purposes
        // (never runs SuperPoint twice per frame). `None` whenever
        // `config.long_loop` is `None` or extraction fails (a soft,
        // non-fatal skip, exactly like M11's own commit-time extraction —
        // the odometry solve never depends on this). Extraction is a pure
        // function of `image` (no RNG, no graph state), so moving it earlier
        // does not change its own output nor perturb `self.rng`'s call
        // sequence relative to the coords/depths sampling below — M11's own
        // `long_loop` runs (`sp_anchored_patches` unset, i.e. `false`)
        // reproduce byte-for-byte.
        let index_due = self.long_loop.as_ref().is_some_and(|runtime| {
            self.graph.counter() % runtime.index.config().index_frequency.max(1) == 0
        });
        let sp_features: Option<DeepFeatureSet> = index_due
            .then(|| {
                self.long_loop.as_mut().and_then(|runtime| {
                    GrayscaleImage::from_luma_u8(w, h, image.iter().copied().collect())
                        .ok()
                        .and_then(|gray| runtime.extractor.extract_deep(&gray).ok())
                })
            })
            .flatten();

        // Centroid sampling (`Patchifier.forward`, `RANDOM` strategy,
        // `net.py:131-133`): integers in `[1, w-1)`/`[1, h-1)` in `fmap`'s
        // own (stride-RES) coordinate space. Milestone M12: when
        // `sp_anchored_patches` is enabled, centers are anchored at this
        // frame's OWN SuperPoint keypoints instead (see
        // `crate::dpvo_long_loop::sp_anchored_patch_centers`'s own doc for
        // the coordinate mapping and the "off = legacy" contract) — any
        // shortfall falls back to the SAME random sampler below, byte-for-
        // byte.
        let m = self.graph.config().patches_per_frame;
        let sp_anchored = self
            .long_loop
            .as_ref()
            .is_some_and(|r| r.index.config().sp_anchored_patches);
        let sp_keypoints: Vec<(f64, f64, f32)> = if sp_anchored {
            sp_features
                .as_ref()
                .map(|f| {
                    f.keypoints
                        .iter()
                        .zip(f.scores.iter())
                        .map(|(k, &s)| (k.x, k.y, s))
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let sp_min_separation = self
            .long_loop
            .as_ref()
            .map(|r| r.index.config().sp_patch_min_separation)
            .unwrap_or(2.0);
        let coords: Vec<(f32, f32)> = sp_anchored_patch_centers(
            m,
            ws,
            hs,
            &sp_keypoints,
            RES as f64,
            sp_min_separation,
            &mut self.rng,
        );

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
            .map(|i| DpvoPatch {
                x: coords[i].0 as f64,
                y: coords[i].1 as f64,
                inverse_depth: depths[i],
            })
            .collect();

        let predicted_pose = self.graph.begin_frame(timestamp);
        let intr = DpvoIntrinsics {
            fx: self.config.intrinsics.fx / RES as f64,
            fy: self.config.intrinsics.fy / RES as f64,
            cx: self.config.intrinsics.cx / RES as f64,
            cy: self.config.intrinsics.cy / RES as f64,
        };
        // M4-perf (`docs/dpvo_droid_port_plan.md`): cache only the layout
        // consumed by the selected correlation backend (see
        // `FramePyramid`), rather than building both CPU-HWC and CUDA-CHW.
        let level1_chw = avg_pool_4x4(fmap.view());
        let retain_chw = self.native_correlation.is_some()
            || (self.config.fused_correlation && self.session.correlation_graph_enabled());
        let candidate_pyramid = FramePyramid {
            level0: (!retain_chw).then(|| ChannelLastImage::from_chw(fmap.view())),
            level1: (!retain_chw).then(|| ChannelLastImage::from_chw(level1_chw.view())),
            // Transfer ownership into the native/ONNX backing store instead
            // of cloning another ~13 MB/frame.
            level0_chw: retain_chw.then_some(fmap),
            level1_chw: retain_chw.then_some(level1_chw),
        };

        // Milestone M15: `true` only under `LowParallaxResponse::DepthDamp`
        // when the regime is active for this candidate — read AFTER a
        // successful commit below to flag the just-committed frame (see the
        // module doc's own "Milestone M15" section). Declared here (rather
        // than inline) so it survives past the `if`/`else if` below.
        let mut low_parallax_flag_on_commit = false;
        if self.graph.n_frames() > 0 && !self.graph.is_initialized() {
            let flow = self.motion_probe(&predicted_pose, &intr, &candidate_pyramid)?;
            if flow < self.config.motion_probe_min_flow {
                self.graph.reject_pending_frame();
                return Ok(None);
            }
        } else if self.graph.is_initialized() {
            let outcome = self.low_parallax_gate(&predicted_pose, &intr, &candidate_pyramid)?;
            if outcome.reject {
                // Milestone M14: the low-parallax ("hover") freeze — see the
                // module doc's own "Low-parallax hover freeze" section.
                // Rejects exactly like the bootstrap-only `motion_probe` gate
                // above: `patches_vec`/`depths` were already sampled (so RNG
                // call counts are unaffected by whether this fires), but
                // `commit_frame` never runs, so no patch is admitted and
                // `n_frames()` does not advance.
                return Ok(None);
            }
            low_parallax_flag_on_commit = outcome.flag_on_commit;
        }

        self.graph.commit_frame(predicted_pose, intr, patches_vec)?;
        // Milestone M15 ("depth-trust damping"): flag the frame we just
        // committed AFTER `commit_frame` succeeds (so the flag is keyed by
        // this frame's own, now-assigned `arrival_index`) — see the module
        // doc's own "Milestone M15" section for why flagging is frame-level.
        if low_parallax_flag_on_commit {
            if let Some(arrival) = self.graph.frames().last().map(|f| f.arrival_index) {
                self.low_parallax_damp
                    .flag(arrival, self.graph.config().patches_per_frame);
            }
        }
        self.frame_pyramids.push(candidate_pyramid);
        for i in 0..m {
            self.patch_gmap.push(gmap.index_axis(Axis(0), i).to_owned());
            self.patch_imap.push(squeeze_patch_vector(&imap_patch4, i));
        }
        // Milestone M11 (`crate::dpvo_long_loop`'s module doc): index this
        // frame's SuperPoint appearance descriptor RIGHT NOW — `image` is a
        // borrowed, transient view that will not be available again once
        // this call returns, so this is the only chance to index a
        // retrieval descriptor for this committed frame. Runs unconditionally
        // on every commit (not gated on `is_initialized`, unlike loop-
        // closure/BA below) so early frames are still indexed as future
        // candidates. Milestone M12: `sp_features` was already extracted
        // ABOVE (before patch-center sampling) and is reused here verbatim —
        // SuperPoint never runs twice for the same frame. A missing/failed
        // extraction (`sp_features: None`) is a soft, non-fatal skip (this
        // frame simply never becomes a retrieval candidate) — the odometry
        // solve itself does not depend on this mechanism at all.
        if let Some(runtime) = self.long_loop.as_mut() {
            let arrival_index = self.graph.frames().last().map(|f| f.arrival_index);
            if let (Some(arrival_index), Some(features)) = (arrival_index, sp_features) {
                let keypoints: Vec<nalgebra::Point2<f64>> = features
                    .keypoints
                    .iter()
                    .map(|k| nalgebra::Point2::new(k.x / RES as f64, k.y / RES as f64))
                    .collect();
                // A3 ranking-lab offline dump: clone BEFORE `ingest_frame`
                // moves both vectors — only when opted in, so every other
                // run pays zero extra clone/memory cost (see
                // `DpvoOdometryConfig::long_loop_dump_enabled`'s own doc).
                // Mirrors `DpvoLongLoopIndex::ingest_frame`'s own
                // `descriptors.is_empty()` early return: an empty-descriptor
                // frame is never actually indexed, so it should not be
                // reported as "ingested" by this dump either.
                if self.config.long_loop_dump_enabled && !features.descriptors.is_empty() {
                    self.long_loop_last_ingested = Some((
                        arrival_index,
                        keypoints.clone(),
                        features.descriptors.clone(),
                    ));
                }
                runtime
                    .index
                    .ingest_frame(arrival_index, keypoints, features.descriptors);
            }
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
            // Milestone M15: advance the depth-damping un-flag age check
            // BEFORE this frame's own BA solves below, so a frame that just
            // crossed `unflag_after_commits` this very frame is already
            // un-flagged for its own solve. A no-op whenever `response !=
            // DepthDamp` (see `Self::advance_low_parallax_unflagging`'s own
            // doc).
            self.advance_low_parallax_unflagging();
            // Milestone M7: the continuous scale-coupling mechanism (see
            // `Self::scale_coupling_step`, called from `update_step` below)
            // REPLACES M5b's one-shot bootstrap entirely when enabled — it
            // re-estimates gyro bias/scale itself, every window, so the
            // one-shot `try_imu_bootstrap` must not also run (the two
            // mechanisms would otherwise both try to own
            // `self.imu_bias_gyro`/`self.imu_bootstrapped`).
            let use_scale_coupling = self
                .config
                .imu
                .as_ref()
                .is_some_and(|c| c.scale_coupling.is_some());
            if !use_scale_coupling {
                self.try_imu_bootstrap();
            }
            let loop_accepted_this_frame = self.try_loop_closure();
            self.update_step()?;
            // Milestone M9: freeze any pending loop pairs' `Sim3LoopMeasurement`
            // NOW — after this frame's own `update_step` (so the new loop
            // edges have gone through at least one windowed BA pass that
            // could genuinely move either endpoint via real visual evidence,
            // not the pre-BA values `try_loop_closure` saw) but BEFORE
            // `keyframe_dispatch` below (which can still shift/fold LIVE
            // indices — this call must run while the accepted pairs' `(i,
            // j)` live indices are still valid). See
            // [`Self::capture_pending_sim3_loop_measurements`]'s own doc.
            self.capture_pending_sim3_loop_measurements();
            // Milestone M11: throttled long-range appearance loop search —
            // AFTER this frame's own `update_step` (so the current frame's
            // own pose has already had at least one visual BA refinement,
            // matching M9's own capture-timing lesson) but BEFORE
            // `keyframe_dispatch` (which can still shift/fold LIVE indices;
            // the current frame's own live pose/patches must still be
            // resolvable via `self.graph` at the point this runs). See
            // `crate::dpvo_long_loop`'s module doc for the full mechanism.
            self.try_long_loop_closure();
            if let Some(k) = self.keyframe_dispatch() {
                self.frame_pyramids.remove(k);
                let m = self.graph.config().patches_per_frame;
                self.patch_gmap.drain(k * m..(k + 1) * m);
                self.patch_imap.drain(k * m..(k + 1) * m);
                self.velocities.remove(k);
                self.prune_stale_imu_deltas();
            }
            // Milestone M8: after the ordinary windowed solve and keyframe
            // cleanup (so any edge archived into the inactive store THIS
            // frame is already available) — see `Self::try_global_ba`'s own
            // doc for the throttle/gating logic.
            self.try_global_ba(loop_accepted_this_frame)?;
            // Milestone M9: after the global-BA pass — both mechanisms are
            // independent (different node sets, different solvers), but
            // running the (much cheaper) Sim3 backend last means it corrects
            // whatever the global-BA pass just wrote, rather than the other
            // way around.
            self.try_sim3_backend(loop_accepted_this_frame)?;
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
        let mut values: Vec<f64> = self.graph.patches()[lo..hi]
            .iter()
            .map(|p| p.inverse_depth)
            .collect();
        if values.is_empty() {
            return 1.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values[(values.len() - 1) / 2]
    }

    fn correlate_group(
        &mut self,
        anchor_gmap: ArrayView4<'_, f32>,
        coords_grid_px: ArrayView4<'_, f32>,
        target: &FramePyramid,
    ) -> Result<Array2<f32>, DpvoOdometryError> {
        if self.native_correlation.is_some() {
            let level0 = target.level0_chw.as_ref().ok_or_else(|| {
                DpvoOdometryError::NativeCudaCorrelation(NativeCudaCorrelationError::Shape(
                    "missing cached level0 CHW map".into(),
                ))
            })?;
            let level1 = target.level1_chw.as_ref().ok_or_else(|| {
                DpvoOdometryError::NativeCudaCorrelation(NativeCudaCorrelationError::Shape(
                    "missing cached level1 CHW map".into(),
                ))
            })?;
            let targets = vec![0_i32; anchor_gmap.dim().0];
            let (correlation, device_ms) = self
                .native_correlation
                .as_mut()
                .expect("checked above")
                .run(anchor_gmap, &[level0], &[level1], coords_grid_px, &targets)?;
            self.stats.native_correlation_device_ms_total += device_ms as f64;
            return Ok(correlation);
        }
        if let (Some(level0), Some(level1)) = (&target.level0_chw, &target.level1_chw) {
            return self
                .session
                .run_correlation_pyramid(
                    anchor_gmap,
                    level0.view().insert_axis(Axis(0)),
                    level1.view().insert_axis(Axis(0)),
                    coords_grid_px,
                )
                .map_err(DpvoOdometryError::Onnx);
        }
        let level0 = target.level0.as_ref().ok_or_else(|| {
            DpvoOdometryError::Onnx(DpvoOnnxError::InputShapeMismatch {
                message: "missing cached level0 HWC map on CPU correlation path".into(),
            })
        })?;
        let level1 = target.level1.as_ref().ok_or_else(|| {
            DpvoOdometryError::Onnx(DpvoOnnxError::InputShapeMismatch {
                message: "missing cached level1 HWC map on CPU correlation path".into(),
            })
        })?;
        Ok(corr_pyramid(
            anchor_gmap,
            coords_grid_px,
            level0,
            level1,
        ))
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
            let grid = reproject_patch_grid(
                &prev_pose,
                predicted_pose,
                &prev_intr,
                candidate_intr,
                &patch,
            );
            anchor_gmap
                .index_axis_mut(Axis(0), local)
                .assign(&self.patch_gmap[patch_lo + local]);
            for py in 0..PATCH {
                for px in 0..PATCH {
                    coords_grid_px[(local, py, px, 0)] = grid[py][px].x as f32;
                    coords_grid_px[(local, py, px, 1)] = grid[py][px].y as f32;
                }
            }
        }
        let corr_flat =
            self.correlate_group(anchor_gmap.view(), coords_grid_px.view(), candidate_pyramid)?;
        self.stats.correlation_ms_total += corr_start.elapsed().as_secs_f64() * 1000.0;

        let net_zero = Array3::<f32>::zeros((1, m, DIM));
        let mut inp_arr = Array3::<f32>::zeros((1, m, DIM));
        for local in 0..m {
            inp_arr
                .index_axis_mut(Axis(0), 0)
                .index_axis_mut(Axis(0), local)
                .assign(&self.patch_imap[patch_lo + local]);
        }
        let kk: Vec<i64> = (patch_lo..patch_lo + m).map(|k| k as i64).collect();
        let ii = vec![prev_frame as i64; m];
        let jj = vec![n as i64; m];
        let corr3 = corr_flat.insert_axis(Axis(0));
        let update_start = Instant::now();
        let (_net_out, delta, _weight) = self.session.update_iteration(
            net_zero.view(),
            inp_arr.view(),
            corr3.view(),
            &kk,
            &ii,
            &jj,
            &self.agg_kk,
            &self.agg_ij,
        )?;
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

    /// Milestone M14/M15: evaluate the low-parallax ("hover") detector for
    /// the current candidate frame — called only once the graph is
    /// initialized (see [`Self::process_frame`]'s own call site; before
    /// initialization the bootstrap-only `motion_probe` gate is the relevant
    /// one instead). A no-op whenever `config.low_parallax` is `None`, or
    /// before any frame has ever committed (`n_frames() == 0` — cannot
    /// happen in practice once initialized, guarded defensively anyway).
    ///
    /// Reuses [`Self::motion_probe`] as the causal "parallax" proxy — see
    /// the module doc's own "Low-parallax hover freeze" section for why a
    /// cheaper, ONNX-free geometric proxy (`flow_mag` between the previous
    /// frame's patches and the predicted candidate pose) was tried first and
    /// rejected: a real-MH_01 calibration run showed it sitting in a narrow
    /// band for the ENTIRE run, including deep inside the GT-confirmed
    /// hover, with no separation from ordinary motion at all (contaminated
    /// by exactly the ill-conditioned patch depth M13 diagnosed as the
    /// problem in the first place). `motion_probe`'s learned GRU-based
    /// correction magnitude does not share that contamination and DOES
    /// separate cleanly (see `docs/dpvo_droid_port_plan.md`'s "M14 results"
    /// for the calibration evidence) — at the cost of one extra
    /// correlation+GRU-update pass per frame whenever this is engaged.
    ///
    /// The [`LowParallaxRegimeState`] enter/exit/one-shot-disarm state
    /// machine itself, and every diagnostic counter it drives
    /// (`times_entered`/`times_exited`/`flow_log`/`last_flow`), is IDENTICAL
    /// regardless of [`DpvoLowParallaxConfig::response`] — only the final
    /// `match` below (M15's own addition) differs by response.
    fn low_parallax_gate(
        &mut self,
        predicted_pose: &SE3,
        candidate_intr: &DpvoIntrinsics,
        candidate_pyramid: &FramePyramid,
    ) -> Result<LowParallaxGateOutcome, DpvoOdometryError> {
        let no_op = LowParallaxGateOutcome {
            reject: false,
            flag_on_commit: false,
        };
        let Some(cfg) = self.config.low_parallax else {
            return Ok(no_op);
        };
        if self.graph.n_frames() == 0 {
            return Ok(no_op);
        }
        let flow = self.motion_probe(predicted_pose, candidate_intr, candidate_pyramid)?;
        let transition = self.low_parallax_regime.update(&cfg, flow);
        self.low_parallax_last_flow = flow;
        let frames_processed = self.stats.frames_processed;
        if transition.just_entered {
            self.low_parallax_times_entered += 1;
            self.low_parallax_last_enter_frame = Some(frames_processed);
        }
        if transition.just_exited {
            self.low_parallax_times_exited += 1;
            self.low_parallax_last_exit_frame = Some(frames_processed);
        }
        self.low_parallax_flow_log.push((
            frames_processed,
            flow,
            self.low_parallax_regime.in_regime(),
        ));
        match cfg.response {
            LowParallaxResponse::Freeze => {
                if transition.suppress {
                    self.graph.reject_pending_frame();
                    self.low_parallax_frames_suppressed_total += 1;
                    Ok(LowParallaxGateOutcome {
                        reject: true,
                        flag_on_commit: false,
                    })
                } else {
                    Ok(no_op)
                }
            }
            LowParallaxResponse::DepthDamp => {
                // Milestone M15: never reject — the candidate commits
                // normally through the ordinary keyframe-decimation path
                // (see the module doc's own "Milestone M15" section for why
                // this is the whole point of Option B). `in_regime()` is
                // read AFTER `update` above, so it already reflects this
                // frame's own transition (true for the entering frame,
                // false for the exiting frame — the SAME semantics
                // `transition.suppress` encodes for `Freeze`, just without
                // rejecting).
                Ok(LowParallaxGateOutcome {
                    reject: false,
                    flag_on_commit: self.low_parallax_regime.in_regime(),
                })
            }
        }
    }

    /// Milestone M15: advance [`LowParallaxDampState`]'s age-based un-flag
    /// check for the most-recently-committed frame's own `arrival_index` —
    /// see that struct's [`LowParallaxDampState::advance_unflagging`] for
    /// the rule. A no-op whenever `config.low_parallax` is `None` or
    /// `response != DepthDamp` (checked here, not inside
    /// `LowParallaxDampState` itself, since that type has no knowledge of
    /// `DpvoLowParallaxConfig` at all — see its own doc for why it stays
    /// free-standing).
    fn advance_low_parallax_unflagging(&mut self) {
        let Some(cfg) = self.config.low_parallax else {
            return;
        };
        if cfg.response != LowParallaxResponse::DepthDamp {
            return;
        }
        let Some(now) = self.graph.frames().last().map(|f| f.arrival_index) else {
            return;
        };
        let still_in_regime = self.low_parallax_regime.in_regime();
        if cfg.gradual_release_duration_commits == 0 {
            self.low_parallax_damp.advance_unflagging(
                now,
                cfg.unflag_after_commits,
                still_in_regime,
            );
        } else {
            self.low_parallax_damp.advance_gradual_release(
                now,
                cfg.unflag_after_commits,
                still_in_regime,
                cfg.gradual_release_duration_commits,
                cfg.gradual_release_start_cap_frames,
            );
        }
    }

    /// Milestone M15: build a [`crate::dpvo_patch_ba::DpvoBaProblem::depth_damping`]
    /// vector for a `dpvo_ba` problem whose `patches` are contiguous
    /// per-frame blocks, one block per `frame_arrivals` entry (in the SAME
    /// order as the problem's own `poses`/`patches`) — see
    /// [`LowParallaxDampState::multipliers`]'s own doc for the layout
    /// contract every call site in this module satisfies. `None` whenever
    /// `config.low_parallax` is `None` (the `?` on `self.config.low_parallax?`
    /// below) — this also correctly covers `response == Freeze`, since
    /// [`LowParallaxDampState::flag`] is only ever called from the
    /// `DepthDamp` branch of [`Self::low_parallax_gate`], so
    /// `low_parallax_damp`'s own `damped_frames` set stays permanently empty
    /// under `Freeze` and [`LowParallaxDampState::multipliers`] returns
    /// `None` on its own first line regardless.
    fn depth_damping_for(
        &mut self,
        frame_arrivals: &[usize],
        patches_per_frame: usize,
    ) -> Option<Vec<f64>> {
        let damp_factor = self.config.low_parallax?.depth_damp_factor;
        self.low_parallax_damp
            .multipliers(frame_arrivals, patches_per_frame, damp_factor)
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
        let Some(imu_cfg) = self.config.imu.clone() else {
            return;
        };
        let mut integrator = ImuPreintegrator::new_with_bias_and_noise(
            self.imu_bias_gyro,
            self.imu_bias_accel,
            imu_cfg.noise,
        )
        .unwrap_or_else(|| {
            ImuPreintegrator::new_with_bias(self.imu_bias_gyro, self.imu_bias_accel)
        });

        let mut last_ts = self.last_imu_boundary_timestamp;
        let mut integrated_any = false;
        while let Some(&(ts, _, _)) = self.pending_imu.front() {
            if ts > frame_timestamp {
                break;
            }
            let (ts, gyro, accel) = self
                .pending_imu
                .pop_front()
                .expect("front() just matched Some");
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
            self.imu_bootstrap_history
                .push((from_arrival, to_arrival, pose_from, pose_to, delta));
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
        let Some(oldest_live) = self.graph.frames().first().map(|f| f.arrival_index) else {
            return;
        };
        self.imu_deltas_by_arrival
            .retain(|&(_, to), _| to >= oldest_live);
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
        let Some(imu_cfg) = self.config.imu.clone() else {
            return;
        };
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
        let local_index: HashMap<usize, usize> = arrival_ids_sorted
            .iter()
            .enumerate()
            .map(|(idx, &id)| (id, idx))
            .collect();
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
                frame.pose = Some(Pose {
                    world_to_camera: body,
                });
                map.keyframes.insert(
                    id as u64,
                    Keyframe {
                        frame,
                        observations: Vec::new(),
                    },
                );
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
        let Some(gyro_bias) = estimate_gyro_bias(&map, &arrival_ids, &factors, self.imu_bias_gyro)
        else {
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
                    GyroGateRejection::MagnitudeTooLarge => {
                        self.imu_rejection_counts.gyro_magnitude += 1
                    }
                    GyroGateRejection::RmsAboveAbsoluteBound => {
                        self.imu_rejection_counts.gyro_rms_absolute += 1
                    }
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
    ///
    /// Returns whether THIS call accepted a new batch — Milestone M8's
    /// [`Self::try_global_ba`] uses this as its "on loop acceptance" forcing
    /// trigger (`docs/dpvo_droid_port_plan.md`'s M8 task brief).
    fn try_loop_closure(&mut self) -> bool {
        let Some(lc_cfg) = self.config.loop_closure else {
            return false;
        };
        let n = self.graph.n_frames();
        let due = match self.last_loop_batch_frame {
            None => true,
            Some(last) => n.saturating_sub(last) >= lc_cfg.global_opt_freq,
        };
        if !due {
            return false;
        }

        let (candidates_evaluated, accepted) = find_loop_edges(&self.graph, &lc_cfg);
        self.loop_batches_attempted += 1;
        self.loop_candidates_evaluated_total += candidates_evaluated;
        if accepted.is_empty() {
            return false;
        }

        self.last_loop_batch_frame = Some(n);
        self.loop_accepted_total += accepted.len();
        self.loop_last_batch_accepted = accepted.len();
        // Milestone M9: record the accepted PAIRS now (live indices, valid
        // until `keyframe_dispatch` next runs), but do NOT freeze their
        // `Sim3LoopMeasurement` yet — see
        // [`Self::capture_pending_sim3_loop_measurements`]'s own doc for why
        // capturing the relative pose HERE, before this frame's own
        // `update_step` has even seen the new edges once, would freeze a
        // measurement carrying NO independent visual evidence at all (it
        // would just reproduce whatever the sequential VO chain already
        // implied for this exact pair, since both readings come from the
        // SAME pre-BA pose values — a real degeneracy this milestone's own
        // synthetic test caught empirically, see that method's doc for the
        // full account).
        if self.config.sim3_backend.is_some() {
            self.pending_sim3_loop_pairs.extend_from_slice(&accepted);
            self.sim3_backend_ever_had_loop_edge = true;
        }
        // Milestone M10: record every accepted pair's stable ARRIVAL indices
        // (not live indices, which `keyframe_dispatch` below can still shift
        // or fold away entirely) — this is the persistent "currently-known
        // loop edges" list [`gather_widened_global_ba_problem`]'s `t0`
        // computation reads. Unlike `pending_sim3_loop_pairs`, there is no
        // "capture later" subtlety here: an arrival index is a stable
        // identity from the instant a frame commits, so recording it NOW
        // (rather than after this frame's own `update_step`) loses nothing —
        // only the sim3 backend's own scale-RATIO measurement needed to wait
        // for a pose to have moved at least once (see
        // [`Self::capture_pending_sim3_loop_measurements`]'s doc).
        if self
            .config
            .global_ba
            .is_some_and(|gba| gba.widen_t0_with_loop_edges)
        {
            for &(i, j) in &accepted {
                self.loop_edge_arrival_pairs.push((
                    self.graph.frames()[i].arrival_index,
                    self.graph.frames()[j].arrival_index,
                ));
            }
        }
        let patch_edges =
            expand_frame_pairs_to_patch_edges(&accepted, self.graph.config().patches_per_frame);
        self.loop_patch_edges_added_total += patch_edges.len();
        self.graph.append_edges(&patch_edges, DIM);
        // Milestone M8: from this point on, `try_global_ba` is allowed to
        // actually run (see that method's own doc for why it stays a no-op
        // until a loop edge has ever existed).
        self.global_ba_ever_had_loop_edge = true;
        true
    }

    /// Milestone M9: freeze every pending accepted loop pair's
    /// `Sim3LoopMeasurement`, reading `(i, j)`'s CURRENT pose — i.e. AFTER
    /// this frame's own `update_step` has already run at least one windowed
    /// `dpvo_ba`/`dpvo_vi_ba` solve with the new patch-BA loop edges active.
    ///
    /// # Why not capture this the instant `try_loop_closure` accepts the pair
    ///
    /// A first version of this milestone froze the measurement immediately
    /// on acceptance, before `append_edges`/`update_step` had any chance to
    /// touch the new edges — this turned out to be a real bug, not merely a
    /// missed optimization, caught by this milestone's own synthetic unit
    /// test (`dpvo_sim3_backend.rs`'s
    /// `se3_only_chain_cannot_recover_multiplicative_drift_but_sim3_backend_does`):
    /// at the exact moment of acceptance, `frame_i`/`frame_j`'s poses are
    /// whatever the ordinary sequential VO chain already produced, so
    /// `frame_j.pose.compose(&frame_i.pose.inverse())` at THAT instant is
    /// mathematically IDENTICAL to what composing every intervening
    /// sequential edge already implies for that same pair (composition
    /// telescopes exactly) — a measurement with ZERO new information versus
    /// the chain it is supposed to disagree with, which starved the Sim3
    /// solve of any real signal to redistribute as scale (confirmed by
    /// deriving `Sim3PoseGraph::optimize`'s own per-edge Jacobian by hand:
    /// a node's scale tangent dimension couples into another edge's
    /// residual only through the OTHER endpoint's own translation
    /// magnitude, never directly — a self-consistent, zero-residual seed
    /// everywhere gives that weak channel nothing to work with). Waiting
    /// until AFTER `update_step` has run at least once lets the loop edges'
    /// own GRU-refined correlation target (genuine, appearance-based visual
    /// evidence, not merely propagated dead-reckoning) actually pull
    /// `frame_i`/`frame_j` toward mutual consistency FIRST, so the frozen
    /// measurement reflects that pull rather than reproducing the
    /// pre-existing chain state.
    fn capture_pending_sim3_loop_measurements(&mut self) {
        if self.pending_sim3_loop_pairs.is_empty() {
            return;
        }
        for (i, j) in self.pending_sim3_loop_pairs.drain(..) {
            let frame_i = &self.graph.frames()[i];
            let frame_j = &self.graph.frames()[j];
            let relative_pose = frame_j.pose.compose(&frame_i.pose.inverse());
            self.sim3_loop_measurements.push(Sim3LoopMeasurement {
                arrival_i: frame_i.arrival_index,
                arrival_j: frame_j.arrival_index,
                relative_pose,
                // Milestone M6/M9 proximity loop edges have no independent
                // 3D-3D scale measurement — `run_sim3_backend` falls back to
                // `estimate_loop_scale_ratio` exactly as before. See
                // `crate::dpvo_long_loop`'s module doc for the M11 mechanism
                // that DOES supply one.
                measured_scale: None,
            });
        }
    }

    /// Milestone M11 (`docs/dpvo_droid_port_plan.md`): throttled long-range
    /// appearance loop search — no-op when `config.long_loop` is `None`. On
    /// acceptance, feeds the SAME "ever had a loop edge" gates
    /// `crate::dpvo_loop_closure`'s own proximity mechanism already
    /// unlocks — no new gating logic: `(arrival_i, arrival_j)` onto
    /// [`Self::loop_edge_arrival_pairs`] (gated on
    /// `config.global_ba.widen_t0_with_loop_edges`, matching
    /// [`Self::try_loop_closure`]'s own gate) and the `Sim3LoopMeasurement`
    /// onto [`Self::sim3_loop_measurements`]. See `crate::dpvo_long_loop`'s
    /// module doc for the full mechanism this dispatches to.
    fn try_long_loop_closure(&mut self) {
        let Some(runtime) = self.long_loop.as_mut() else {
            return;
        };
        let Some(current) = self.graph.frames().last() else {
            return;
        };
        let current_arrival = current.arrival_index;
        if !runtime.index.due(current_arrival) {
            return;
        }
        let current_pose = current.pose.clone();
        let current_intrinsics = current.intrinsics;
        let patches_per_frame = self.graph.config().patches_per_frame;
        let n = self.graph.n_frames();
        let current_patches: Vec<DpvoPatch> =
            self.graph.patches()[(n - 1) * patches_per_frame..n * patches_per_frame].to_vec();

        let graph_ref = &self.graph;
        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if let Some(live) = graph_ref
                .frames()
                .iter()
                .position(|f| f.arrival_index == arrival)
            {
                let pose = graph_ref.frames()[live].pose.clone();
                let intrinsics = graph_ref.frames()[live].intrinsics;
                let patches = graph_ref.patches()
                    [live * patches_per_frame..(live + 1) * patches_per_frame]
                    .to_vec();
                return Some((pose, intrinsics, patches));
            }
            let pose = graph_ref.retained_poses().get(&arrival)?.clone();
            let folded = graph_ref.retained_folded_frames().get(&arrival)?;
            Some((pose, folded.intrinsics, folded.patches.clone()))
        };

        // A3 stage 2, first slice (`crate::dpvo_long_loop`'s module doc):
        // the stage-2 coverage gate needs the frame's own patch-grid extent
        // — the SAME `ws`/`hs` bound `run_global_ba`'s own visibility gate
        // already derives from `self.config.width`/`height` — passed as a
        // plain `f64` pair since `dpvo_long_loop` stays `onnx-inference`-
        // feature-agnostic (mirrors `RES`'s own "pass as a parameter, not an
        // import" precedent, `sp_anchored_patch_centers`'s own doc).
        let grid_width = self.config.width as f64 / RES as f64;
        let grid_height = self.config.height as f64 / RES as f64;
        let Some(accepted) = runtime.index.find_and_verify_long_range_loop(
            current_arrival,
            &current_pose,
            &current_intrinsics,
            &current_patches,
            grid_width,
            grid_height,
            resolve_old,
        ) else {
            return;
        };

        self.sim3_loop_measurements.push(accepted.measurement);
        self.sim3_backend_ever_had_loop_edge = true;
        if self
            .config
            .global_ba
            .is_some_and(|gba| gba.widen_t0_with_loop_edges)
        {
            self.loop_edge_arrival_pairs
                .push((accepted.arrival_i, accepted.arrival_j));
        }
        self.global_ba_ever_had_loop_edge = true;
    }

    /// Milestone M8 (`docs/dpvo_droid_port_plan.md`): the CPU-bounded
    /// stand-in for upstream's `__run_global_BA` (`dpvo.py:312-325`) — a
    /// full-graph [`dpvo_ba`] solve over EVERY live frame pose (`[0, n)`)
    /// using BOTH the currently active edge set AND every resolvable
    /// retained inactive edge, with `fixedp = t0 = min(active edges' owner
    /// frame)` (upstream's own `self.pg.ii.min()`, `dpvo.py:323`) as the
    /// fixed gauge — i.e. the oldest frame any ACTIVE edge still references
    /// stays fixed; everything from there through `n` is free, matching
    /// upstream's own choice of which poses `fastba.BA`'s `t0` argument
    /// excludes.
    ///
    /// # Why this is usually cheap, and when it spikes
    ///
    /// The dense pose Hessian this solves scales with the FREE pose count
    /// (`n - t0`), not the total frame count — `dpvo_patch_ba`'s own
    /// `fixedp`/`t0` convention keeps poses below the gauge entirely out of
    /// the Hessian (M3's own convention-mapping note). Without loop closure,
    /// `t0` never differs from the ordinary per-frame window's own
    /// `frame_lo` bound (no edge is ever older than that — M4/M6's own
    /// windowing derivation), so this method never actually runs (see
    /// [`Self::try_global_ba`]'s gate). Once a loop batch IS accepted, `t0`
    /// can briefly drop far below that bound (as far back as the loop's own
    /// source frame) for as long as that edge stays ACTIVE — bounded by
    /// `keyframe_with_loop_protection`'s own exemption window
    /// (`optimization_window` frames past acceptance, `dpvo_patch_graph.rs`'s
    /// own doc) — after which the edge survives only as an INACTIVE entry
    /// (no longer counted toward `t0`, since `t0` is derived from ACTIVE
    /// edges only, matching upstream exactly) and `t0` reverts to the
    /// ordinary bound. So the free-pose count, and hence this call's own
    /// cost, is expected to be small on most calls and spike only briefly
    /// around a loop acceptance — see `docs/dpvo_droid_port_plan.md`'s M8
    /// results for the measured cost profile, not just this reasoning.
    ///
    /// # Inactive-edge resolution
    ///
    /// See `crate::dpvo_patch_graph`'s own "Inactive-edge retention" module
    /// doc section for why each [`crate::dpvo_patch_graph::InactiveEdge`] is
    /// re-resolved against the CURRENT live frame set (via
    /// `arrival_index` ↁElive index) rather than trusted at face value, and
    /// why an entry that no longer resolves (its endpoint frame has since
    /// been folded away entirely) is simply skipped for this pass — a rare,
    /// non-error condition tracked only for diagnostics
    /// ([`DpvoGlobalBaDiagnostics::last_unresolved_inactive_edges`]).
    fn run_global_ba(&mut self, cfg: &DpvoGlobalBaConfig) -> Result<(), DpvoOdometryError> {
        let n = self.graph.n_frames();
        if n == 0 {
            return Ok(());
        }
        // Same visibility-gate bounds `update_step` uses (`net.py`'s own BA
        // call site bounds, image extent padded by 64px).
        let ws = self.config.width as f64 / RES as f64;
        let hs = self.config.height as f64 / RES as f64;
        let bounds = [-64.0, -64.0, ws + 64.0, hs + 64.0];

        if cfg.widen_t0_with_loop_edges && !self.loop_edge_arrival_pairs.is_empty() {
            self.run_widened_global_ba(cfg, bounds)
        } else {
            self.run_legacy_global_ba(cfg, bounds)
        }
    }

    /// M8's exact original solve — `t0 = min(active edges' owner frame)`,
    /// no folded-frame materialization. Used whenever
    /// [`DpvoGlobalBaConfig::widen_t0_with_loop_edges`] is `false` (the
    /// default) or no loop edge is known yet — byte-identical to the M8
    /// implementation, unchanged by the M10 split.
    fn run_legacy_global_ba(
        &mut self,
        cfg: &DpvoGlobalBaConfig,
        bounds: [f64; 4],
    ) -> Result<(), DpvoOdometryError> {
        let n = self.graph.n_frames();
        let Some((t0, edges, targets, weights, resolved_inactive, unresolved_inactive)) =
            gather_global_ba_edges(&self.graph)
        else {
            return Ok(()); // No active edges at all => nothing to solve.
        };
        if edges.is_empty() {
            return Ok(());
        }

        let start = Instant::now();
        let poses: Vec<SE3> = self.graph.frames().iter().map(|f| f.pose.clone()).collect();
        let intrinsics: Vec<DpvoIntrinsics> =
            self.graph.frames().iter().map(|f| f.intrinsics).collect();
        let patches: Vec<DpvoPatch> = self.graph.patches().to_vec();
        // Milestone M15: this pass covers every live frame in `graph.frames()`
        // order, exactly matching `poses`/`patches`' own layout above — see
        // `Self::depth_damping_for`'s own doc.
        let arrivals: Vec<usize> = self
            .graph
            .frames()
            .iter()
            .map(|f| f.arrival_index)
            .collect();
        let patches_per_frame = self.graph.config().patches_per_frame;
        let depth_damping = self.depth_damping_for(&arrivals, patches_per_frame);

        let free_pose_count = n.saturating_sub(t0);
        let edge_count = edges.len();
        let problem = DpvoBaProblem {
            poses,
            patches,
            intrinsics,
            edges,
            targets,
            weights,
            depth_damping,
        };
        let ba_cfg = DpvoBaConfig {
            iterations: cfg.iterations,
            fixedp: t0,
            lmbda: cfg.lmbda,
            ep: cfg.ep,
            bounds,
        };
        let solved = dpvo_ba(&problem, &ba_cfg)?;

        let mut max_delta = 0.0_f64;
        let mut sum_delta = 0.0_f64;
        for local in t0..n {
            let delta = (solved.poses[local].translation
                - self.graph.frames()[local].pose.translation)
                .norm();
            max_delta = max_delta.max(delta);
            sum_delta += delta;
        }
        for (local, pose) in solved.poses.into_iter().enumerate() {
            self.graph.frames_mut()[local].pose = pose;
        }
        for (local, patch) in solved.patches.into_iter().enumerate() {
            self.graph.patches_mut()[local] = patch;
        }

        self.global_ba_calls += 1;
        self.global_ba_last_free_pose_count = free_pose_count;
        self.global_ba_max_free_pose_count =
            self.global_ba_max_free_pose_count.max(free_pose_count);
        self.global_ba_last_edge_count = edge_count;
        self.global_ba_last_resolved_inactive = resolved_inactive;
        self.global_ba_last_unresolved_inactive = unresolved_inactive;
        self.global_ba_last_pose_delta_max_m = max_delta;
        self.global_ba_last_pose_delta_mean_m = if free_pose_count > 0 {
            sum_delta / free_pose_count as f64
        } else {
            0.0
        };
        self.global_ba_last_widened = false;
        self.global_ba_last_folded_included = 0;
        self.global_ba_last_capped = false;
        self.global_ba_last_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.global_ba_ms_total += self.global_ba_last_ms;
        Ok(())
    }

    /// Milestone M10 (`docs/dpvo_droid_port_plan.md`): the widened solve —
    /// see [`gather_widened_global_ba_problem`]'s own doc for the full
    /// mechanism (parts A/B/C/E of the M10 design). Gathers a combined
    /// live+folded problem, solves it, and writes corrected poses/patches
    /// back to BOTH the live graph and the folded-frame retention stores.
    fn run_widened_global_ba(
        &mut self,
        cfg: &DpvoGlobalBaConfig,
        bounds: [f64; 4],
    ) -> Result<(), DpvoOdometryError> {
        let Some(gathered) = gather_widened_global_ba_problem(
            &self.graph,
            &self.loop_edge_arrival_pairs,
            cfg.max_free_poses,
        ) else {
            return Ok(());
        };
        if gathered.edges.is_empty() {
            return Ok(());
        }

        let start = Instant::now();
        let free_pose_count = gathered.poses.len() - gathered.fixedp;
        let edge_count = gathered.edges.len();
        let folded_count = gathered.folded_arrivals.len();
        // Milestone M15: `gathered.poses`/`patches` is the folded prefix
        // (`gathered.folded_arrivals`, oldest first) followed by EVERY live
        // frame in `graph.frames()` order (see
        // `gather_widened_global_ba_problem`'s own doc, Parts B/C — the
        // live suffix is never a subset) — so the same concatenation order
        // reproduces the arrival index for every pose/patch block.
        let patches_per_frame = self.graph.config().patches_per_frame;
        let arrivals: Vec<usize> = gathered
            .folded_arrivals
            .iter()
            .copied()
            .chain(self.graph.frames().iter().map(|f| f.arrival_index))
            .collect();
        let depth_damping = self.depth_damping_for(&arrivals, patches_per_frame);
        let problem = DpvoBaProblem {
            poses: gathered.poses,
            patches: gathered.patches,
            intrinsics: gathered.intrinsics,
            edges: gathered.edges,
            targets: gathered.targets,
            weights: gathered.weights,
            depth_damping,
        };
        let ba_cfg = DpvoBaConfig {
            iterations: cfg.iterations,
            fixedp: gathered.fixedp,
            lmbda: cfg.lmbda,
            ep: cfg.ep,
            bounds,
        };
        let solved = dpvo_ba(&problem, &ba_cfg)?;

        let mut max_delta = 0.0_f64;
        let mut sum_delta = 0.0_f64;
        for combined in gathered.fixedp..problem.poses.len() {
            let before = &problem.poses[combined];
            let after = &solved.poses[combined];
            let delta = (after.translation - before.translation).norm();
            max_delta = max_delta.max(delta);
            sum_delta += delta;
        }

        let patches_per_frame = self.graph.config().patches_per_frame;
        // Write back the folded prefix into the retention stores.
        for (idx, &arrival) in gathered.folded_arrivals.iter().enumerate() {
            self.graph
                .set_retained_pose_override(arrival, solved.poses[idx].clone());
            if let Some(ff) = self.graph.retained_folded_frames_mut().get_mut(&arrival) {
                ff.patches =
                    solved.patches[idx * patches_per_frame..(idx + 1) * patches_per_frame].to_vec();
            }
        }
        // Write back the live suffix exactly like the legacy path.
        for (local, pose) in solved.poses[folded_count..].iter().enumerate() {
            self.graph.frames_mut()[local].pose = pose.clone();
        }
        for (local, patch) in solved.patches[folded_count * patches_per_frame..]
            .iter()
            .enumerate()
        {
            self.graph.patches_mut()[local] = *patch;
        }

        self.global_ba_calls += 1;
        self.global_ba_last_free_pose_count = free_pose_count;
        self.global_ba_max_free_pose_count =
            self.global_ba_max_free_pose_count.max(free_pose_count);
        self.global_ba_last_edge_count = edge_count;
        self.global_ba_last_resolved_inactive = gathered.resolved_inactive;
        self.global_ba_last_unresolved_inactive = gathered.unresolved_inactive;
        self.global_ba_last_pose_delta_max_m = max_delta;
        self.global_ba_last_pose_delta_mean_m = if free_pose_count > 0 {
            sum_delta / free_pose_count as f64
        } else {
            0.0
        };
        self.global_ba_last_widened = gathered.t0_widened_by_loop_edge;
        self.global_ba_last_folded_included = folded_count;
        self.global_ba_last_capped = gathered.capped;
        self.global_ba_last_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.global_ba_ms_total += self.global_ba_last_ms;
        Ok(())
    }

    /// Milestone M8: `cfg.frequency`-throttled dispatch to
    /// [`Self::run_global_ba`] — due either because `loop_just_accepted`
    /// (a loop batch was JUST accepted this same frame — the task's own
    /// "on loop acceptance" trigger) or because `cfg.frequency` frames have
    /// passed since the last call ("every `GLOBAL_OPT_FREQ` frames while
    /// loops exist"). No-op entirely if `config.global_ba` is `None` or no
    /// loop edge has EVER existed (see [`Self::run_global_ba`]'s own doc for
    /// why a global pass is redundant, strictly more expensive work in that
    /// case — `t0` cannot differ from the ordinary per-frame window bound
    /// without at least one loop edge ever having been active).
    fn try_global_ba(&mut self, loop_just_accepted: bool) -> Result<(), DpvoOdometryError> {
        let Some(gba_cfg) = self.config.global_ba else {
            return Ok(());
        };
        if !self.global_ba_ever_had_loop_edge {
            return Ok(());
        }
        let n = self.graph.n_frames();
        if !global_ba_due(
            loop_just_accepted,
            self.last_global_ba_frame,
            n,
            gba_cfg.frequency,
        ) {
            return Ok(());
        }
        self.last_global_ba_frame = Some(n);
        self.run_global_ba(&gba_cfg)
    }

    /// Milestone M9: `cfg.frequency`-throttled dispatch to
    /// `crate::dpvo_sim3_backend::run_sim3_backend` — mirrors
    /// [`Self::try_global_ba`]'s own gating exactly (no-op until a loop edge
    /// has EVER been accepted, since a Sim3 correction needs at least one
    /// loop measurement's chain-vs-hop disagreement to have anything to
    /// distribute — see `crate::dpvo_sim3_backend`'s module doc; due either
    /// because `loop_just_accepted` or because `cfg.frequency` frames have
    /// passed since the last call).
    fn try_sim3_backend(&mut self, loop_just_accepted: bool) -> Result<(), DpvoOdometryError> {
        let Some(s3b_cfg) = self.config.sim3_backend.clone() else {
            return Ok(());
        };
        if !self.sim3_backend_ever_had_loop_edge {
            return Ok(());
        }
        let n = self.graph.n_frames();
        if !global_ba_due(
            loop_just_accepted,
            self.last_sim3_backend_frame,
            n,
            s3b_cfg.frequency,
        ) {
            return Ok(());
        }
        self.last_sim3_backend_frame = Some(n);
        let start = Instant::now();
        let Some(result) =
            run_sim3_backend(&mut self.graph, &self.sim3_loop_measurements, &s3b_cfg)
        else {
            return Ok(());
        };
        self.sim3_backend_calls += 1;
        self.sim3_backend_last_node_count = result.node_count;
        self.sim3_backend_last_edge_count = result.edge_count;
        self.sim3_backend_last_loop_edges_used = result.loop_edge_count;
        self.sim3_backend_last_corrected_pose_count = if result.committed {
            result.corrected_pose_count
        } else {
            0
        };
        self.sim3_backend_last_pose_delta_max_m = result.pose_delta_max_m;
        self.sim3_backend_last_pose_delta_mean_m = result.pose_delta_mean_m;
        self.sim3_backend_last_scale_min = result.scale_min;
        self.sim3_backend_last_scale_max = result.scale_max;
        self.sim3_backend_last_committed = result.committed;
        self.sim3_backend_last_rejection = result.rejection;
        update_sim3_scale_cliff_diagnostics(
            &mut self.sim3_backend_max_committed_abs_log_scale,
            &mut self.sim3_backend_scale_jump_rejections_total,
            result.committed,
            result.rejection,
            result.scale_min,
            result.scale_max,
        );
        self.sim3_backend_last_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.sim3_backend_ms_total += self.sim3_backend_last_ms;
        Ok(())
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
            self.graph
                .keyframe_with_loop_protection(optimization_window)
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

        let mut coords_center = vec![Vector2::new(0.0_f64, 0.0_f64); e_count];
        let corr_start = Instant::now();
        let corr_flat = if self.native_correlation.is_some() {
            let mut anchor_gmap = Array4::<f32>::zeros((e_count, FNET_DIM, PATCH, PATCH));
            let mut coords_grid_px = Array4::<f32>::zeros((e_count, PATCH, PATCH, 2));
            // Only frames referenced as correlation targets belong in this
            // invocation. The graph may retain hundreds of keyframes on a
            // full sequence while PATCH_LIFETIME bounds the active target
            // set to a small suffix. Passing every retained pyramid made
            // the native stable-slot wrapper scan/copy an ever-growing set
            // and caused full-sequence latency to grow roughly linearly.
            let edge_targets: Vec<usize> = edges.iter().map(|edge| edge.j).collect();
            let (target_frames, targets) = compact_target_indices(&edge_targets);
            for (idx, edge) in edges.iter().enumerate() {
                let pose_i = self.graph.frames()[edge.i].pose.clone();
                let pose_j = self.graph.frames()[edge.j].pose.clone();
                let intr_i = self.graph.frames()[edge.i].intrinsics;
                let intr_j = self.graph.frames()[edge.j].intrinsics;
                let patch = self.graph.patches()[edge.k];
                let grid = reproject_patch_grid(&pose_i, &pose_j, &intr_i, &intr_j, &patch);
                coords_center[idx] = grid[1][1];
                anchor_gmap
                    .index_axis_mut(Axis(0), idx)
                    .assign(&self.patch_gmap[edge.k]);
                for py in 0..PATCH {
                    for px in 0..PATCH {
                        coords_grid_px[(idx, py, px, 0)] = grid[py][px].x as f32;
                        coords_grid_px[(idx, py, px, 1)] = grid[py][px].y as f32;
                    }
                }
            }
            let level0_frames: Result<Vec<&Array3<f32>>, _> = target_frames
                .iter()
                .map(|&frame| {
                    self.frame_pyramids[frame].level0_chw.as_ref().ok_or_else(|| {
                        DpvoOdometryError::NativeCudaCorrelation(
                            NativeCudaCorrelationError::Shape(
                                "missing cached level0 CHW map".into(),
                            ),
                        )
                    })
                })
                .collect();
            let level1_frames: Result<Vec<&Array3<f32>>, _> = target_frames
                .iter()
                .map(|&frame| {
                    self.frame_pyramids[frame].level1_chw.as_ref().ok_or_else(|| {
                        DpvoOdometryError::NativeCudaCorrelation(
                            NativeCudaCorrelationError::Shape(
                                "missing cached level1 CHW map".into(),
                            ),
                        )
                    })
                })
                .collect();
            let frame_ids: Vec<u64> = target_frames
                .iter()
                .map(|&frame| self.graph.frames()[frame].arrival_index as u64)
                .collect();
            let (correlation, device_ms) = self
                .native_correlation
                .as_mut()
                .expect("checked above")
                .run_stable(
                    anchor_gmap.view(),
                    &level0_frames?,
                    &level1_frames?,
                    coords_grid_px.view(),
                    &targets,
                    &frame_ids,
                )?;
            self.stats.native_correlation_device_ms_total += device_ms as f64;
            correlation
        } else {
            let mut by_target: HashMap<usize, Vec<usize>> = HashMap::new();
            for (idx, edge) in edges.iter().enumerate() {
                by_target.entry(edge.j).or_default().push(idx);
            }
            let mut corr_flat = Array2::<f32>::zeros((e_count, CORR_DIM));
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
                    anchor_gmap
                        .index_axis_mut(Axis(0), local)
                        .assign(&self.patch_gmap[edge.k]);
                    for py in 0..PATCH {
                        for px in 0..PATCH {
                            coords_grid_px[(local, py, px, 0)] = grid[py][px].x as f32;
                            coords_grid_px[(local, py, px, 1)] = grid[py][px].y as f32;
                        }
                    }
                }
                let group_corr = self.correlate_group(
                    anchor_gmap.view(),
                    coords_grid_px.view(),
                    &target_pyramid,
                )?;
                for (local, &idx) in idxs.iter().enumerate() {
                    corr_flat.row_mut(idx).assign(&group_corr.row(local));
                }
            }
            corr_flat
        };
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
            inp_arr
                .index_axis_mut(Axis(0), 0)
                .index_axis_mut(Axis(0), idx)
                .assign(&self.patch_imap[edge.k]);
            kk.push(edge.k as i64);
            ii.push(edge.i as i64);
            jj.push(edge.j as i64);
        }
        let corr3 = corr_flat.insert_axis(Axis(0));

        let update_start = Instant::now();
        let (net_out, delta, weight) = self.session.update_iteration(
            net_arr.view(),
            inp_arr.view(),
            corr3.view(),
            &kk,
            &ii,
            &jj,
            &self.agg_kk,
            &self.agg_ij,
        )?;
        self.stats.update_ms_total += update_start.elapsed().as_secs_f64() * 1000.0;

        let mut targets = Vec::with_capacity(e_count);
        let mut weights = Vec::with_capacity(e_count);
        for idx in 0..e_count {
            let net_row: Vec<f32> = net_out
                .index_axis(Axis(0), 0)
                .index_axis(Axis(0), idx)
                .to_owned()
                .into_raw_vec_and_offset()
                .0;
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
        let global_fixedp = if self.graph.is_initialized() {
            n.saturating_sub(optimization_window).max(1)
        } else {
            1
        };
        let local_fixedp = global_fixedp.saturating_sub(frame_lo);
        let patches_lo = frame_lo * patches_per_frame;

        debug_assert!(
            edges
                .iter()
                .all(|e| e.i >= frame_lo && e.j >= frame_lo && e.k >= patches_lo),
            "update_step's window [frame_lo,n) did not cover every active edge — Milestone M6's \
             own min-over-edges widening above should make this unconditionally true; a failure \
             here means that widening itself has a bug, not just a loose bound \
             (removal_window={removal_window}, patch_lifetime={patch_lifetime})"
        );

        let window_poses: Vec<SE3> = self.graph.frames()[frame_lo..n]
            .iter()
            .map(|f| f.pose.clone())
            .collect();
        let window_intr: Vec<DpvoIntrinsics> = self.graph.frames()[frame_lo..n]
            .iter()
            .map(|f| f.intrinsics)
            .collect();
        let window_patches: Vec<DpvoPatch> = self.graph.patches()[patches_lo..].to_vec();
        let ba_edges: Vec<DpvoEdge> = edges
            .iter()
            .map(|e| DpvoEdge {
                i: e.i - frame_lo,
                j: e.j - frame_lo,
                k: e.k - patches_lo,
            })
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
        let loop_correction_pre_solve: Option<Vec<SE3>> =
            self.config.loop_closure.as_ref().and_then(|_| {
                edges
                    .iter()
                    .any(|e| e.j.saturating_sub(e.i) > UPSTREAM_MIN_LOOP_GAP)
                    .then(|| window_poses.clone())
            });

        // Milestone M15: `window_patches` above is exactly `graph.patches()`'s
        // `[patches_lo, ...)` suffix, one contiguous block of
        // `patches_per_frame` per frame in `[frame_lo, n)` order — the SAME
        // order `window_poses`/`window_intr` use, matching
        // `Self::depth_damping_for`'s own layout contract.
        let window_arrivals: Vec<usize> = self.graph.frames()[frame_lo..n]
            .iter()
            .map(|f| f.arrival_index)
            .collect();
        let depth_damping = self.depth_damping_for(&window_arrivals, patches_per_frame);
        let problem = DpvoBaProblem {
            poses: window_poses,
            patches: window_patches,
            intrinsics: window_intr,
            edges: ba_edges,
            targets,
            weights,
            depth_damping,
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
        let use_scale_coupling = self
            .config
            .imu
            .as_ref()
            .is_some_and(|c| c.scale_coupling.is_some());
        let advance_scale_evidence = if use_scale_coupling {
            window_arrivals.last().is_some_and(|&arrival| {
                advance_once_per_arrival(&mut self.scale_coupling_last_evidence_arrival, arrival)
            })
        } else {
            false
        };
        // Milestone M5: once the IMU bootstrap chain has succeeded, couple
        // consecutive-window IMU factors into the SAME Gauss-Newton solve
        // (`crate::dpvo_vi_ba::dpvo_vi_ba`) instead of the plain visual-only
        // `dpvo_ba` — see this module's own doc, "IMU coupling", and
        // `crate::dpvo_vi_ba`'s module doc for the math. Falls back to the
        // unmodified M4 path whenever `config.imu` is `None` or the
        // bootstrap has not (yet) succeeded — visual-only behavior is
        // therefore byte-for-byte unchanged from M4 in both of those cases.
        let (new_poses, new_patches, new_velocities) = if use_scale_coupling {
            self.scale_coupling_step(
                &problem,
                frame_lo,
                n,
                local_fixedp,
                &ba_config,
                advance_scale_evidence,
            )?
        } else if self.imu_bootstrapped {
            let imu_cfg = self
                .config
                .imu
                .clone()
                .expect("imu_bootstrapped can only be true when config.imu is Some — set together in try_imu_bootstrap");
            // Milestone M15: reuses the SAME `window_arrivals` computed above
            // for `depth_damping_for` — identical `[frame_lo, n)` arrival-index
            // list, no need to recompute it.
            let mut imu_factors = Vec::new();
            for local in 0..window_arrivals.len().saturating_sub(1) {
                let key = (window_arrivals[local], window_arrivals[local + 1]);
                if let Some(banked) = self.imu_deltas_by_arrival.get(&key) {
                    let mut factor = banked.clone();
                    factor.gravity_world = self
                        .imu_gravity_world
                        .expect("imu_bootstrapped implies imu_gravity_world is Some");
                    imu_factors.push(DpvoImuFactor {
                        i: local,
                        j: local + 1,
                        factor,
                    });
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
            let mean_nis = if nis_count > 0 {
                nis_sum / nis_count as f64
            } else {
                0.0
            };
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
        if let Some(velocities) =
            new_velocities.filter(|_| self.imu_bootstrapped || use_scale_coupling)
        {
            for (local, v) in velocities.into_iter().enumerate() {
                self.velocities[frame_lo + local] = v;
            }
        }
        Ok(())
    }

    /// Milestone M7 (`docs/dpvo_droid_port_plan.md`): the continuous,
    /// uncertainty-weighted scale-coupling solver step — see
    /// `crate::dpvo_scale_coupling`'s module doc for the full design this
    /// implements. Called from [`Self::update_step`] in place of the
    /// M5/M5b branch whenever `config.imu.scale_coupling` is `Some`.
    /// Returns `(poses, patches, velocities)` in the SAME window-local
    /// indexing `problem`/`dpvo_ba`/`dpvo_vi_ba` already use.
    ///
    /// # Why this never touches `self.imu_bootstrapped`/`self.imu_bias_gyro`
    /// as a COMMITTED, staged value
    ///
    /// Unlike M5b's `try_imu_bootstrap`, this method continually re-derives
    /// gyro-bias/scale evidence from the current window, but admits recursive
    /// evidence only once per camera arrival. Repeated solver/final-refinement
    /// calls on an unchanged window may reuse the fixed posterior in the
    /// coupled solve, but are not independent sensor observations. Nothing
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
        advance_evidence: bool,
    ) -> Result<ScaleCouplingSolution, DpvoOdometryError> {
        let imu_cfg =
            self.config.imu.clone().expect(
                "scale_coupling_step is only called when config.imu.scale_coupling is Some",
            );
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
        let window_arrivals: Vec<usize> = self.graph.frames()[frame_lo..n]
            .iter()
            .map(|f| f.arrival_index)
            .collect();
        let mut window_factors: Vec<DpvoImuFactor> = Vec::new();
        for local in 0..window_arrivals.len().saturating_sub(1) {
            let key = (window_arrivals[local], window_arrivals[local + 1]);
            if let Some(banked) = self.imu_deltas_by_arrival.get(&key) {
                window_factors.push(DpvoImuFactor {
                    i: local,
                    j: local + 1,
                    factor: banked.clone(),
                });
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
        // called once per camera arrival against the CURRENT live window instead of
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
            frame.pose = Some(Pose {
                world_to_camera: body,
            });
            map.keyframes.insert(
                arrival as u64,
                Keyframe {
                    frame,
                    observations: Vec::new(),
                },
            );
            local_poses[idx] = pose;
        }
        let arrival_ids: Vec<u64> = window_arrivals.iter().map(|&a| a as u64).collect();
        let factors_for_gyro: Vec<ImuPreintegrationFactor> =
            window_factors.iter().map(|f| f.factor.clone()).collect();

        let seed_bias = self.gyro_bias_estimator.mean();
        if advance_evidence {
            if let Some(alignment) =
                estimate_gyro_bias(&map, &arrival_ids, &factors_for_gyro, seed_bias)
            {
                // Honest variance proxy (same "derive it from the LSQ's own
                // fit quality" philosophy as the scale estimator — see
                // `crate::dpvo_scale_coupling`'s module doc): the ROTATION
                // alignment's own converged residual RMS is the direct
                // empirical noise-level estimate for THIS measurement.
                let variance = alignment.rotation_residual_rms_after.max(1.0e-9).powi(2);
                self.gyro_bias_estimator
                    .update(alignment.bias_gyro, variance);
            }
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
            .map(|f| DpvoImuFactor {
                i: f.i - mono_start,
                j: f.j - mono_start,
                factor: f.factor.clone(),
            })
            .collect();

        let alignment_result = advance_evidence.then(|| {
            estimate_mono_vi_alignment(
                mono_poses,
                &mono_factors,
                &imu_cfg.body_to_camera,
                bias_gyro,
                self.imu_bias_accel,
                &gates,
            )
        });
        match alignment_result {
            Some(Ok(alignment)) => {
                self.scale_coupling_measurements += 1;
                let measurement = scale_measurement_from_alignment(&alignment, &sc_cfg.scale);
                self.scale_estimator.update(measurement);
                self.scale_coupling_gravity = Some(alignment.gravity_world);
                for (idx, &v) in alignment.velocities.iter().enumerate() {
                    window_velocities[mono_start + idx] = v;
                }
            }
            Some(Err(rejection)) => {
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
            None => {}
        }

        let should_increase = self.scale_estimator.is_converged();
        if advance_evidence {
            self.scale_coupling_weight.step(should_increase);
        }
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
                DpvoImuFactor {
                    i: f.i,
                    j: f.j,
                    factor,
                }
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
        let mean_nis = if nis_count > 0 {
            nis_sum / nis_count as f64
        } else {
            0.0
        };
        if advance_evidence {
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
        }

        let (blended_poses, blended_patches) = blend_solutions(
            &visual_solved.poses,
            &visual_solved.patches,
            &imu_poses,
            &imu_patches,
            weight,
        );
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
fn gyro_bootstrap_gate_check(
    alignment: &GyroBiasAlignment,
    cfg: &DpvoImuConfig,
) -> Result<(), GyroGateRejection> {
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
    let next = if mean_nis.is_finite() && mean_nis <= bound {
        0
    } else {
        consecutive_bad + 1
    };
    (next, next >= threshold)
}

/// Admit recursive sensor evidence at most once for each camera arrival.
/// Solver iterations on an unchanged graph window are useful numerical
/// refinement, but are correlated repeats of the same images and IMU
/// factors—not additional measurements and not additional elapsed frames
/// for an annealing or consecutive-failure counter.
fn advance_once_per_arrival(last: &mut Option<usize>, current: usize) -> bool {
    if *last == Some(current) {
        return false;
    }
    *last = Some(current);
    true
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

/// Return type of [`gather_global_ba_edges`]: `(t0, edges, targets, weights,
/// resolved_inactive_count, unresolved_inactive_count)` — see that
/// function's own doc for the full semantics of each element. A named alias
/// purely to keep the function signature clippy-clean (`clippy::type_complexity`);
/// no behavior attaches to the name itself.
pub(crate) type GlobalBaGatheredEdges = (
    usize,
    Vec<DpvoEdge>,
    Vec<Vector2<f64>>,
    Vec<Vector2<f64>>,
    usize,
    usize,
);

/// Milestone M8 (`docs/dpvo_droid_port_plan.md`): the pure "gather" step of
/// [`DpvoOdometry::run_global_ba`] — see that method's own doc for the full
/// semantics (`t0` = the oldest frame any ACTIVE edge still references,
/// upstream's own `self.pg.ii.min()`; active edges with a learned
/// measurement, plus every retained [`crate::dpvo_patch_graph::InactiveEdge`]
/// that still resolves against the CURRENT live frame set). A free function
/// taking `&DpvoPatchGraph` directly, not `&DpvoOdometry` — this crate's own
/// established pattern for graph-only logic that should be unit-testable
/// without a live ONNX-backed odometry instance (compare
/// `crate::dpvo_loop_closure::find_loop_edges`'s own `&DpvoPatchGraph`
/// signature). Returns `None` if the graph has no active edges at all
/// (nothing to solve); otherwise [`GlobalBaGatheredEdges`].
pub(crate) fn gather_global_ba_edges(graph: &DpvoPatchGraph) -> Option<GlobalBaGatheredEdges> {
    let patches_per_frame = graph.config().patches_per_frame;
    let t0 = graph.edges().iter().map(|e| e.i).min()?;

    let mut edges: Vec<DpvoEdge> = Vec::new();
    let mut targets: Vec<Vector2<f64>> = Vec::new();
    let mut weights: Vec<Vector2<f64>> = Vec::new();
    for edge in graph.edges() {
        if let Some((target, weight)) = edge.target_weight {
            edges.push(DpvoEdge {
                i: edge.i,
                j: edge.j,
                k: edge.k,
            });
            targets.push(target);
            weights.push(weight);
        }
    }

    // arrival_index -> current live frame index, needed ONLY to re-resolve
    // retained inactive edges — see `crate::dpvo_patch_graph`'s own
    // "Inactive-edge retention" module doc section for why this port
    // re-resolves rather than trusting a possibly-stale stored index.
    let arrival_to_live: HashMap<usize, usize> = graph
        .frames()
        .iter()
        .enumerate()
        .map(|(live, f)| (f.arrival_index, live))
        .collect();

    let mut resolved_inactive = 0usize;
    let mut unresolved_inactive = 0usize;
    for ie in graph.inactive_edges() {
        match (
            arrival_to_live.get(&ie.arrival_i),
            arrival_to_live.get(&ie.arrival_j),
        ) {
            (Some(&live_i), Some(&live_j)) => {
                let k = live_i * patches_per_frame + ie.local_patch_offset;
                edges.push(DpvoEdge {
                    i: live_i,
                    j: live_j,
                    k,
                });
                targets.push(ie.target);
                weights.push(ie.weight);
                resolved_inactive += 1;
            }
            _ => unresolved_inactive += 1,
        }
    }

    Some((
        t0,
        edges,
        targets,
        weights,
        resolved_inactive,
        unresolved_inactive,
    ))
}

/// Return type of [`gather_widened_global_ba_problem`] — a complete,
/// ready-to-solve [`DpvoBaProblem`] laid out as `[folded frames (oldest
/// first)] ++ [live frames (as-is)]`, plus the bookkeeping needed to write a
/// solved result back to the right place in each store.
pub(crate) struct WidenedGlobalBaGather {
    pub poses: Vec<SE3>,
    pub patches: Vec<DpvoPatch>,
    pub intrinsics: Vec<DpvoIntrinsics>,
    pub edges: Vec<DpvoEdge>,
    pub targets: Vec<Vector2<f64>>,
    pub weights: Vec<Vector2<f64>>,
    /// Always `0` when [`Self::folded_arrivals`] is non-empty (the earliest
    /// included folded frame IS `t0` and must be free, matching upstream's
    /// own `fixedp = t0 = ii.min()` when nothing older exists in the array —
    /// see [`gather_widened_global_ba_problem`]'s own doc). Otherwise the
    /// combined-space index of the live frame that plays the same role.
    pub fixedp: usize,
    /// Arrival indices of every FOLDED frame included, in ascending order —
    /// `poses[0..folded_arrivals.len()]`/`patches`'s leading
    /// `folded_arrivals.len() * patches_per_frame` entries correspond to
    /// these, in the same order.
    pub folded_arrivals: Vec<usize>,
    pub resolved_inactive: usize,
    pub unresolved_inactive: usize,
    /// Whether this call's `t0` ended up older than a plain active-edges-only
    /// `t0` would have (see [`DpvoGlobalBaDiagnostics::last_t0_widened_by_loop_edge`]).
    pub t0_widened_by_loop_edge: bool,
    /// Whether `max_free_poses` actually trimmed the folded prefix short of
    /// what the known loop evidence would otherwise justify.
    pub capped: bool,
}

/// Milestone M10 (`docs/dpvo_droid_port_plan.md`): the widened "gather" step
/// — attacks the exact root cause M8's own results section measured
/// (`last_free_pose_count` pinned at `removal_window` on every real call
/// because `t0` was computed from ACTIVE edges only, and the
/// `keyframe_with_loop_protection` exemption keeping a loop edge active
/// rarely survives long enough to be seen by a `try_global_ba` call).
///
/// # Part A: `t0` from loop edges, decoupled from `optimization_window`
///
/// Computes `t0_arrival = min(loop_pairs.iter().map(|&(i, _)| i))` — the
/// EARLIEST endpoint among every accepted proximity-loop edge EVER seen
/// (`DpvoOdometry::loop_edge_arrival_pairs`, persistent and keyed by stable
/// arrival index, never gated on whether the underlying patch-graph edge is
/// still active). This is a strictly wider notion of `t0` than
/// [`gather_global_ba_edges`]'s own `min(active edges' owner frame)`: it
/// does not matter whether the loop edge itself survived
/// `keyframe_with_loop_protection`'s exemption window — once a loop pair is
/// EVER accepted, its old endpoint permanently becomes a `t0` candidate.
///
/// # Parts B/C: folded poses and patch geometry as free variables
///
/// Two cases, depending on whether `t0_arrival`'s frame is still live:
///
/// * **Still live** (the common case on this port's real MH_01 runs, per
///   the M10 results — `keyframe_thresh` rarely folds a frame away on a
///   fast-motion sequence, so `unresolved_inactive_edges` stays `0` and the
///   loop's own old endpoint frame is simply still sitting in
///   [`crate::dpvo_patch_graph::DpvoPatchGraph::frames`], just with few or
///   no ACTIVE edges left touching it): resolve `t0_arrival` to its live
///   index directly, no folded-frame materialization needed at all — `t0`
///   in live-index space is `min(active t0, that live index)`, and any
///   [`crate::dpvo_patch_graph::InactiveEdge`] anchored there (which the
///   ordinary resolution loop already includes) supplies the actual
///   constraint tying it to the rest of the graph.
/// * **Folded away**: every folded frame with an arrival index `>=
///   t0_arrival` that has BOTH a
///   [`crate::dpvo_patch_graph::DpvoPatchGraph::retained_poses`] entry (M9)
///   AND a [`crate::dpvo_patch_graph::DpvoPatchGraph::retained_folded_frames`]
///   entry (M10) is prepended to the live frame array, oldest first, as a
///   FREE pose variable (`fixedp = 0` for the combined array — see
///   [`WidenedGlobalBaGather::fixedp`]'s own doc for why this matches
///   upstream fidelity rather than being an ad-hoc choice). No upper bound
///   is assumed on this range — deliberately NOT "up to the earliest still-
///   live arrival," since [`crate::dpvo_patch_graph::DpvoPatchGraph::keyframe`]'s
///   own fold-candidate arithmetic can never select array position `0` or
///   `1` (the smallest valid frame count for the fold gate to fire at all
///   makes `2` the earliest possible candidate), so the first two committed
///   frames can stay live indefinitely while much later frames fold and
///   unfold around them — "folded" and "live" are NOT simply an
///   old/recent split. [`DpvoGlobalBaConfig::max_free_poses`] (Part E) is
///   what bounds how far this search reaches, not an assumption about where
///   live frames resume. Any arrival in range MISSING a snapshot (e.g. a
///   frame folded before this port added retention, or one that was
///   rejected by `motion_probe` and never committed at all, hence never had
///   patches) is simply skipped — a real gap, not silently patched over,
///   tracked only informally (the resulting chain may have holes; any edge
///   referencing a skipped arrival is `unresolved_inactive` like any other
///   unresolvable entry).
///
/// [`crate::dpvo_patch_graph::InactiveEdge`] resolution is generalized to
/// match: an entry resolves if EITHER endpoint is live OR is one of the
/// included folded frames (previously only live resolved) — this is what
/// makes an edge connecting two folded frames, or a folded frame to a live
/// one, usable as a REAL reprojection factor rather than merely a
/// frozen-target pull, closing the exact gap Part C's task description
/// flagged as "the hard part."
///
/// # Part E: cost bounds
///
/// The combined array's free-pose count (`folded_arrivals.len() + n`, since
/// `fixedp` is `0` whenever any folded frame is included) drives
/// `dpvo_patch_ba::dpvo_ba`'s dense `6·n2 × 6·n2` pose Hessian — see
/// `DpvoGlobalBaConfig::max_free_poses`'s own doc for the cost this bounds.
/// When set, the FRONT of `folded_arrivals` (the OLDEST included frames) is
/// trimmed until the combined free-pose count fits — meaning a very tight
/// cap can trim away the very endpoint `t0_arrival` was chosen for; this is
/// a real, honestly-reported trade-off (see [`WidenedGlobalBaGather::capped`]),
/// not a silent one.
pub(crate) fn gather_widened_global_ba_problem(
    graph: &DpvoPatchGraph,
    loop_pairs: &[(usize, usize)],
    max_free_poses: Option<usize>,
) -> Option<WidenedGlobalBaGather> {
    let patches_per_frame = graph.config().patches_per_frame;
    let n = graph.n_frames();
    if n == 0 || graph.edges().is_empty() {
        return None;
    }
    let active_t0_live = graph.edges().iter().map(|e| e.i).min().unwrap_or(n);
    let t0_arrival = loop_pairs.iter().map(|&(i, _)| i).min()?;

    let arrival_to_live: HashMap<usize, usize> = graph
        .frames()
        .iter()
        .enumerate()
        .map(|(live, f)| (f.arrival_index, live))
        .collect();

    // Branch purely on whether `t0_arrival`'s OWN frame is still live — NOT
    // on whether it is older than the array's current minimum live arrival.
    // An earlier version of this function used the latter, reasoning that
    // "everything folded away is older than everything still live" — false
    // in general for this port: [`crate::dpvo_patch_graph::DpvoPatchGraph::keyframe`]'s
    // own fold-candidate arithmetic (`candidate = n - keyframe_index`) can
    // NEVER select array position `0` or `1` (the smallest valid `n` for the
    // fold gate to fire at all is `keyframe_index + 2`, making
    // `candidate = n - keyframe_index = 2` the earliest possible target),
    // so the very first two committed frames can remain live indefinitely
    // while much later frames fold and unfold around them — confirmed by
    // this module's own `widened_global_ba_closes_a_synthetic_drifted_loop_whose_old_endpoint_is_folded_away`
    // test, which deliberately keeps arrival frames `0`/`1` live while
    // folding arrival frame `2`. Gating on "older than the live minimum"
    // would have wrongly concluded `t0_arrival` was still live in that case.
    let t0_is_live = arrival_to_live.contains_key(&t0_arrival);

    // Every folded arrival at or after `t0_arrival` that has BOTH a
    // retained pose AND retained patch geometry, ascending order — no upper
    // bound (Part E's `max_free_poses` cap below is what bounds this, not an
    // assumption about where "live" resumes).
    let mut folded_arrivals: Vec<usize> = if t0_is_live {
        Vec::new()
    } else {
        graph
            .retained_poses()
            .range(t0_arrival..)
            .map(|(&a, _)| a)
            .filter(|a| graph.retained_folded_frames().contains_key(a))
            .collect()
    };
    folded_arrivals.sort_unstable();

    let mut capped = false;
    if let Some(cap) = max_free_poses {
        let uncapped_total = folded_arrivals.len() + n;
        if uncapped_total > cap {
            let excess = uncapped_total - cap;
            let drop = excess.min(folded_arrivals.len());
            folded_arrivals.drain(0..drop);
            capped = true;
        }
    }
    let folded_count = folded_arrivals.len();

    let folded_index: HashMap<usize, usize> = folded_arrivals
        .iter()
        .enumerate()
        .map(|(idx, &a)| (a, idx))
        .collect();
    // A single resolver covering both stores: folded (this call's own
    // materialized prefix) takes priority in the unlikely event an arrival
    // somehow appears in both (never true by construction, since folded ==
    // no longer live, but kept as an explicit `else` for clarity).
    let combined_index_of = |arrival: usize| -> Option<usize> {
        if let Some(&idx) = folded_index.get(&arrival) {
            Some(idx)
        } else {
            arrival_to_live
                .get(&arrival)
                .map(|&live| folded_count + live)
        }
    };

    let mut poses = Vec::with_capacity(folded_count + n);
    let mut intrinsics = Vec::with_capacity(folded_count + n);
    let mut patches = Vec::with_capacity((folded_count + n) * patches_per_frame);
    for &arrival in &folded_arrivals {
        poses.push(graph.retained_poses()[&arrival].clone());
        let ff = &graph.retained_folded_frames()[&arrival];
        intrinsics.push(ff.intrinsics);
        patches.extend_from_slice(&ff.patches);
    }
    for f in graph.frames() {
        poses.push(f.pose.clone());
        intrinsics.push(f.intrinsics);
    }
    patches.extend_from_slice(graph.patches());

    let mut edges: Vec<DpvoEdge> = Vec::new();
    let mut targets: Vec<Vector2<f64>> = Vec::new();
    let mut weights: Vec<Vector2<f64>> = Vec::new();
    for edge in graph.edges() {
        if let Some((target, weight)) = edge.target_weight {
            edges.push(DpvoEdge {
                i: folded_count + edge.i,
                j: folded_count + edge.j,
                k: folded_count * patches_per_frame + edge.k,
            });
            targets.push(target);
            weights.push(weight);
        }
    }
    let mut resolved_inactive = 0usize;
    let mut unresolved_inactive = 0usize;
    for ie in graph.inactive_edges() {
        match (
            combined_index_of(ie.arrival_i),
            combined_index_of(ie.arrival_j),
        ) {
            (Some(ci), Some(cj)) => {
                let k = ci * patches_per_frame + ie.local_patch_offset;
                edges.push(DpvoEdge { i: ci, j: cj, k });
                targets.push(ie.target);
                weights.push(ie.weight);
                resolved_inactive += 1;
            }
            _ => unresolved_inactive += 1,
        }
    }

    let mut fixedp = if folded_count > 0 {
        0
    } else {
        // No folding needed: t0_arrival's own frame is still live. Widen in
        // live-index space directly, same convention `gather_global_ba_edges`
        // uses, just taking the min against the loop-derived candidate too.
        let loop_t0_live = arrival_to_live
            .get(&t0_arrival)
            .copied()
            .unwrap_or(active_t0_live);
        active_t0_live.min(loop_t0_live)
    };
    // Trimming folded material alone cannot enforce the bound when the live
    // graph itself exceeds `max_free_poses` (observed as 331 free poses with
    // a configured cap of 256). Keep the full pose/edge arrays for residual
    // coverage, but fix the oldest additional poses so the dense solve's
    // actual free block is bounded exactly as the public config promises.
    if let Some(cap) = max_free_poses {
        let required_fixed = (folded_count + n).saturating_sub(cap);
        if required_fixed > fixedp {
            fixedp = required_fixed;
            capped = true;
        }
    }
    let free_pose_count = (folded_count + n).saturating_sub(fixedp);
    let plain_active_free_pose_count = n.saturating_sub(active_t0_live);
    let t0_widened_by_loop_edge = free_pose_count > plain_active_free_pose_count;

    Some(WidenedGlobalBaGather {
        poses,
        patches,
        intrinsics,
        edges,
        targets,
        weights,
        fixedp,
        folded_arrivals,
        resolved_inactive,
        unresolved_inactive,
        t0_widened_by_loop_edge,
        capped,
    })
}

/// Milestone M8: pure "due" computation for
/// [`DpvoOdometry::try_global_ba`]'s own throttle — see that method's doc.
/// Extracted as a free function so the throttle logic is unit-testable
/// without a live ONNX-backed instance.
fn global_ba_due(
    loop_just_accepted: bool,
    last_call_frame: Option<usize>,
    current_frame: usize,
    frequency: usize,
) -> bool {
    loop_just_accepted
        || match last_call_frame {
            None => true,
            Some(last) => current_frame.saturating_sub(last) >= frequency,
        }
}

#[cfg(test)]
mod scale_coupling_windowing_tests {
    use super::{advance_once_per_arrival, trailing_consecutive_run_start};

    #[test]
    fn recursive_evidence_advances_once_per_camera_arrival() {
        let mut last = None;
        assert!(advance_once_per_arrival(&mut last, 42));
        for _ in 0..12 {
            assert!(!advance_once_per_arrival(&mut last, 42));
        }
        assert!(advance_once_per_arrival(&mut last, 43));
        assert!(!advance_once_per_arrival(&mut last, 43));
    }

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

#[cfg(test)]
mod global_ba_tests {
    use nalgebra::UnitQuaternion;

    use super::*;
    use crate::dpvo_patch_ba::transform_point;
    use crate::dpvo_patch_graph::DpvoVoConfig;

    #[test]
    fn global_ba_due_throttle_behaves_as_specified() {
        assert!(
            global_ba_due(false, None, 0, 15),
            "first-ever call is always due"
        );
        assert!(
            !global_ba_due(false, Some(10), 20, 15),
            "only 10 frames since last call: not yet due"
        );
        assert!(
            global_ba_due(false, Some(5), 20, 15),
            "15 frames since last call: due"
        );
        assert!(
            global_ba_due(true, Some(19), 20, 15),
            "a fresh loop acceptance forces it regardless of frequency"
        );
    }

    /// Patches sampled per frame for the synthetic loop-closure test below —
    /// deliberately NOT `1` (see that test's own doc for why a single shared
    /// patch cannot demonstrate the mechanism this test is required to
    /// prove).
    const LOOP_TEST_PATCHES_PER_FRAME: usize = 48; // matches the demo's own `--patches-per-frame 48` FAST bench value.

    fn small_graph_config() -> DpvoVoConfig {
        DpvoVoConfig {
            buffer_size: 4096,
            patches_per_frame: LOOP_TEST_PATCHES_PER_FRAME,
            removal_window: 2,
            optimization_window: 4,
            patch_lifetime: 1,
            keyframe_index: 2,
            // Strictly-less-than-zero is never true for a flow magnitude
            // (a non-negative norm), so this is a deliberate "never fold"
            // knob for this test, not a realistic operating value — the
            // test cares about the removal-window drop archiving edges
            // into the inactive store, not the (unrelated)
            // low-motion-fold mechanism.
            keyframe_thresh: 0.0,
            motion_damping: 0.5,
        }
    }

    /// Milestone M8's required synthetic accuracy test: build a graph whose
    /// ONLY path connecting a set of old patches (frame 0) to a drifted,
    /// recent frame (the last one, `DRIFTED_FRAME`) is
    /// via RETAINED INACTIVE edges (archived by the ordinary removal-window
    /// drop, not live active ones) -- gathering and solving via
    /// [`gather_global_ba_edges`] + [`dpvo_ba`] (exactly the production
    /// `DpvoOdometry::run_global_ba` path) must reduce the drifted frame's
    /// endpoint error by more than 10x versus a baseline graph whose inactive
    /// store never had those loop observations to begin with (only the
    /// ordinary temporal chain). Mirrors `crate::dpvo_loop_closure`'s own M6
    /// "closing a synthetic drifted loop" test in spirit (drift a frame, add
    /// a correctly-predicted revisit observation, solve, compare against a
    /// no-revisit baseline), but exercises the M8 INACTIVE-edge path
    /// specifically rather than a freshly-added active edge.
    ///
    /// # Why this fixture needs many DISTINCT patches over many DISTINCT
    /// pinning frames, not one (or a few) shared points
    ///
    /// An earlier version of this test used a *single* shared patch (same
    /// anchor pixel, same inverse depth) for both the temporal chain (frame
    /// 0's patch observed in frames 1, 2) and the loop revisit (that same
    /// patch observed again in a drifted frame 4) -- the smallest possible
    /// fixture, mirroring `dpvo_loop_closure.rs`'s own
    /// `closing_a_synthetic_drifted_loop_reduces_endpoint_error` test almost
    /// exactly. That fixture reliably reproduced *some* correction (enough
    /// for that other test's weaker `error_with_loop < error_no_loop` bar)
    /// but never came close to this test's required >10x bar, and two
    /// escalations that might look like the obvious fix turned out NOT to be
    /// enough on their own:
    ///
    /// 1. **More edges on the SAME point.** Replaying the identical 3D point
    ///    through `N` parallel patches (same anchor pixel, same depth, each
    ///    with its own independent inverse-depth variable) barely helped:
    ///    from 1 to 1000 duplicated copies of the same point,
    ///    `error_with_loop` moved only from 0.149 to 0.137 against a 0.150
    ///    baseline -- nowhere near >10x, and unaffected by `iterations` (tried
    ///    up to 20; the fixture converges to the same degenerate minimum in a
    ///    single step). This is a genuine mathematical fact, not a damping
    ///    artifact: bundle-adjusting one 3D point observed from a fixed
    ///    anchor frame plus one other (drifted) frame is a classical
    ///    monocular depth/translation ambiguity -- moving the point's depth
    ///    and moving the drifted camera's pose are first-order-equivalent
    ///    ways to explain that one point's reprojection error, and
    ///    duplicating the point `N` times just scales every term of the
    ///    (still 2-way-degenerate) normal equations by `N` (confirmed by
    ///    deriving the Schur complement directly: every duplicate's own
    ///    elimination contributes an identical multiple of the single-point
    ///    terms) without adding any NEW geometric constraint.
    /// 2. **Distinct points, but only 2 pinning frames.** Giving each patch a
    ///    genuinely different anchor pixel AND inverse depth (`patch_for_local`
    ///    below) helped only a little (`error_with_loop` 0.150 -> 0.118 at
    ///    `m=48`), and — surprisingly — scaling up the two pinning frames'
    ///    OWN baseline (tried translation steps from 0.2m to 5.0m) changed
    ///    almost nothing either (0.118 -> 0.115). The reason: with only 2
    ///    pinning views, each patch's own inverse depth is still only weakly
    ///    triangulated (the loop edge's own Jacobian and the pinning edges'
    ///    Jacobians scale together as the whole configuration is scaled up,
    ///    so their RATIO — which is what actually determines how much of the
    ///    loop residual gets diverted into depth instead of pose — stays the
    ///    same regardless of baseline size).
    ///
    /// What actually works is a THIRD, genuinely different axis: MORE
    /// DISTINCT PINNING FRAMES (not a bigger baseline on the same 2, and not
    /// more patches on the same 2). Each additional pinning frame gives every
    /// patch's inverse depth an independent, non-redundant new constraint
    /// (sweeping `N_FRAMES` from 6 to 80 at `m=48` took `error_with_loop` from
    /// 0.073 down to effectively fully corrected), which is exactly how
    /// upstream's own patches actually behave: `PATCH_LIFETIME` (12, default)
    /// gives every real patch many pinning edges across its whole life
    /// *before* `REMOVAL_WINDOW` retires it into the inactive store, not just
    /// one or two. This fixture's `1..DRIFTED_FRAME` pinning-frame loop below
    /// is the analogous (if much longer, to buy comfortable margin past the
    /// required 10x-reduction bar) construction, combined with
    /// `expand_frame_pairs_to_patch_edges`'s
    /// own "one edge per patch, never a single edge" shape
    /// (`crate::dpvo_loop_closure::expand_frame_pairs_to_patch_edges`) for
    /// both the pinning AND the loop edges. This is not "tuning the test
    /// until it passes": the earlier single/few-pinning-frame fixtures were
    /// testing configurations with a real, inherent depth/pose ambiguity that
    /// no amount of BA iteration, duplicate-patch count, or baseline scaling
    /// could ever resolve, which is not what a real loop batch (with its
    /// patches' own long pinning history) actually looks like.
    #[test]
    fn global_ba_closes_a_synthetic_drifted_loop_via_retained_inactive_edges() {
        let m = LOOP_TEST_PATCHES_PER_FRAME;
        let intr = DpvoIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 64.0,
            cy: 48.0,
        };
        // Distinct anchor pixel + inverse depth per patch (see this test's
        // own doc for why a single repeated point is a genuine, unfixable
        // ambiguity rather than an under-powered fixture): spread across an
        // 8-wide grid of image columns/rows, inverse depth cycling over
        // `0.1..=0.22` (depth ~4.5m..10m), all safely inside
        // `crate::dpvo_patch_ba`'s `DISP_MIN..DISP_MAX` clamp range.
        let patch_for_local = |local: usize| -> DpvoPatch {
            let col = (local % 8) as f64;
            let row = (local / 8) as f64;
            DpvoPatch {
                x: 20.0 + col * 10.0,
                y: 15.0 + row * 8.0,
                inverse_depth: 0.1 + 0.02 * (local % 7) as f64,
            }
        };
        let frame0_patches: Vec<DpvoPatch> = (0..m).map(patch_for_local).collect();
        const N_FRAMES: usize = 55;
        const DRIFTED_FRAME: usize = N_FRAMES - 1;
        let true_poses: Vec<SE3> = (0..N_FRAMES)
            .map(|i| {
                SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(i as f64 * 0.2, 0.0, 0.0),
                )
            })
            .collect();
        let drift = Vector3::new(0.15, 0.0, 0.0);

        let run = |with_loop: bool| -> SE3 {
            let mut graph = DpvoPatchGraph::new(small_graph_config());
            graph.enable_inactive_edge_retention(m * (N_FRAMES + 2));
            for pose in &true_poses {
                graph.begin_frame(0.05);
                graph
                    .commit_frame(pose.clone(), intr, frame0_patches.clone())
                    .unwrap();
            }
            // Drift the LAST frame's LIVE pose away from truth (simulating
            // accumulated monocular scale/pose drift).
            graph.frames_mut()[DRIFTED_FRAME].pose = SE3::new(
                true_poses[DRIFTED_FRAME].rotation,
                true_poses[DRIFTED_FRAME].translation + drift,
            );

            // Ordinary temporal chain: every one of frame 0's `m` DISTINCT
            // patches observed in EVERY frame from 1 up to (not including)
            // the drifted frame -- multiple genuinely distinct pinning
            // viewpoints, not just one or two, mirroring how a real patch
            // accumulates edges across its whole `patch_lifetime` before a
            // loop batch ever touches it (`expand_frame_pairs_to_patch_edges`'s
            // own "one edge per patch" shape for each pinning frame too, see
            // this test's own doc for why).
            for target in 1..DRIFTED_FRAME {
                let chain: Vec<(usize, usize)> = (0..m).map(|local| (local, target)).collect();
                graph.append_edges(&chain, 4);
            }
            if with_loop {
                // The "loop" observation: every one of frame 0's `m` patches
                // observed again in the drifted frame -- targets computed at
                // that frame's TRUE pose, i.e. exactly the
                // correctly-predicted-revisit a real GRU update would supply
                // regardless of the CURRENT (drifted) pose estimate.
                let loop_edges: Vec<(usize, usize)> =
                    (0..m).map(|local| (local, DRIFTED_FRAME)).collect();
                graph.append_edges(&loop_edges, 4);
            }
            for edge in graph.edges_mut() {
                // `edge.k` is `frame0_patches`' own index directly: every
                // chain/loop edge above is owned by frame 0, whose patch ids
                // occupy `[0, m)` unshifted (frame 0 is the very first
                // frame committed).
                let owner_patch = &frame0_patches[edge.k];
                let target = transform_point(
                    &true_poses[edge.i],
                    &true_poses[edge.j],
                    &intr,
                    &intr,
                    owner_patch,
                    false,
                );
                edge.target_weight = Some((target, Vector2::new(1.0, 1.0)));
            }

            // Archive everything above into the inactive store (threshold =
            // N_FRAMES - removal_window(2); every injected edge's owner
            // frame is 0, which is far below that threshold for any
            // realistic N_FRAMES).
            graph.keyframe();
            assert!(
                graph.edges().is_empty(),
                "every injected edge should have aged into the inactive store"
            );

            // A mathematically inert "keepalive" self-edge (`i == j`,
            // target == the patch's own anchor pixel) so
            // `gather_global_ba_edges` has at least one ACTIVE edge to
            // derive `t0` from. `crate::dpvo_patch_ba::flow_mag`'s own
            // module doc already derives why a self-edge's `Gij` collapses
            // to the identity regardless of the shared pose's value; since
            // `Ji = -Gij.adjT(Jj)` and `Ad(I) = I`, this means `Ji = -Jj`
            // for a self-edge, so `i_local == j_local` accumulates
            // `Bii+Bjj+Bij+Bji = 0` exactly (a full algebraic cancellation)
            // -- this edge cannot move ANY pose, it exists purely to give
            // `t0` a value (frame 1, the self-edge's own owner) without
            // touching the drifted frame at all.
            let keepalive_patch = m; // frame 1's first patch (owner_frame(m) == 1).
            graph.append_edges(&[(keepalive_patch, 1)], 4);
            let patch1 = graph.patches()[keepalive_patch];
            for edge in graph.edges_mut() {
                edge.target_weight =
                    Some((Vector2::new(patch1.x, patch1.y), Vector2::new(1.0, 1.0)));
            }

            let (t0, edges, targets, weights, resolved_inactive, unresolved_inactive) =
                gather_global_ba_edges(&graph).expect("at least the keepalive edge is active");
            assert_eq!(
                t0, 1,
                "the keepalive self-edge (owner frame 1) should set the gauge"
            );
            assert_eq!(
                unresolved_inactive, 0,
                "every frame referenced by an inactive edge is still live"
            );
            let pinning_frames = DRIFTED_FRAME - 1;
            assert_eq!(
                resolved_inactive,
                if with_loop {
                    (pinning_frames + 1) * m
                } else {
                    pinning_frames * m
                }
            );

            let poses: Vec<SE3> = graph.frames().iter().map(|f| f.pose.clone()).collect();
            let patches: Vec<DpvoPatch> = graph.patches().to_vec();
            let intrinsics = vec![intr; poses.len()];
            let problem = DpvoBaProblem {
                poses,
                patches,
                intrinsics,
                edges,
                targets,
                weights,
                depth_damping: None,
            };
            let config = DpvoBaConfig {
                iterations: 2,
                fixedp: t0,
                lmbda: 1e-4,
                ep: 100.0,
                bounds: [-1e6, -1e6, 1e6, 1e6],
            };
            let solved = dpvo_ba(&problem, &config).expect("global BA solve");
            solved.poses[DRIFTED_FRAME].clone()
        };

        let solved_without_loop = run(false);
        let error_without_loop =
            (solved_without_loop.translation - true_poses[DRIFTED_FRAME].translation).norm();
        let solved_with_loop = run(true);
        let error_with_loop =
            (solved_with_loop.translation - true_poses[DRIFTED_FRAME].translation).norm();

        assert!(
            error_with_loop * 10.0 < error_without_loop,
            "global BA over the retained inactive loop edge should close the drift by >10x: \
             with_loop={error_with_loop:.6} without_loop={error_without_loop:.6}"
        );
        assert!(
            (error_without_loop - drift.norm()).abs() < 1e-3,
            "with no inactive loop edge touching frame 4 at all, global BA should leave its drift essentially \
             untouched: error_without_loop={error_without_loop:.6} drift={:.6}",
            drift.norm()
        );
    }

    fn m10_graph_config(patches_per_frame: usize, keyframe_index: usize) -> DpvoVoConfig {
        DpvoVoConfig {
            buffer_size: 4096,
            patches_per_frame,
            removal_window: 2,
            optimization_window: 4,
            patch_lifetime: 1,
            keyframe_index,
            // Positive (matches upstream's own default): the fold this test
            // relies on is forced by `motionmag() == 0.0` (no matching
            // `(i, j)` edge exists at all, see this test's own doc), which
            // is `< ANY positive threshold` regardless of its exact value.
            keyframe_thresh: 12.5,
            motion_damping: 0.5,
        }
    }

    /// Milestone M10's required synthetic accuracy test (task spec, part G):
    /// a synthetic drifted loop whose old endpoint is OUTSIDE the live
    /// window (physically folded away via [`DpvoPatchGraph::keyframe`], not
    /// merely aged out of the active-edge set the way the M8 fixture above
    /// exercises) — [`gather_widened_global_ba_problem`] must reduce the
    /// drifted frame's endpoint error by more than 10x by materializing the
    /// folded frame as a free pose with real patch geometry, while the
    /// M8-style [`gather_global_ba_edges`] path provably CANNOT touch it at
    /// all (its own `unresolved_inactive` diagnostic proves the folded
    /// endpoint is unreachable, and the drifted pose receives zero
    /// correction because no edge in the legacy problem references it).
    ///
    /// # Engineering a real fold at a chosen frame, deterministically
    ///
    /// [`DpvoPatchGraph::keyframe`]'s own fold-candidate arithmetic
    /// (`candidate = n - keyframe_index`, gated on `n > keyframe_index + 1`)
    /// forces `candidate >= 2` on every possible call (the smallest valid
    /// `n` is `keyframe_index + 2`, giving `candidate = 2` exactly) — so
    /// array positions `0`/`1` can NEVER be folded directly, but position
    /// `2` always CAN be, on the very first call, by choosing
    /// `keyframe_index = n - 2` for whatever `n` the graph has at the time.
    /// `OLD_FRAME` (arrival index `2`, the third frame committed) is this
    /// test's designated fold target. [`DpvoPatchGraph::motionmag`] returns
    /// `0.0` (guaranteed below any positive `keyframe_thresh`) whenever NO
    /// active edge exists with the exact `(i, j)` pair the fold check
    /// derives (`i = n - keyframe_index - 1 = 1`, `j = n - keyframe_index +
    /// 1 = 3` for this test's own arithmetic) — this test simply never
    /// creates such an edge, so the fold is unconditional, not motion-
    /// dependent.
    ///
    /// Two separate `keyframe()` calls are required, in this order:
    /// 1. **Call A** (while `OLD_FRAME` is still live): archives its
    ///    touched pinning/loop edges into the inactive store via the
    ///    ordinary removal-window drop. The fold gate cannot fire yet
    ///    (`n <= keyframe_index + 1` at this point).
    /// 2. **Call B** (after enough filler frames push `n` up to exactly
    ///    `keyframe_index + 2`): folds `OLD_FRAME` away. Its edges are
    ///    already safely in the inactive store by now — `fold_frame`'s own
    ///    `store=False` edge-drop only touches [`DpvoPatchGraph::edges`],
    ///    never the separate inactive-edge store, so nothing is lost.
    ///
    /// This ordering matters: if the SAME call both archived `OLD_FRAME`'s
    /// edges AND folded it, `fold_frame`'s unconditional `store=False` drop
    /// (which runs BEFORE the removal-window archiving phase even sees the
    /// edges) would discard them with no trace at all.
    #[test]
    fn widened_global_ba_closes_a_synthetic_drifted_loop_whose_old_endpoint_is_folded_away() {
        let m = LOOP_TEST_PATCHES_PER_FRAME;
        let intr = DpvoIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 64.0,
            cy: 48.0,
        };
        let patch_for_local = |local: usize| -> DpvoPatch {
            let col = (local % 8) as f64;
            let row = (local / 8) as f64;
            DpvoPatch {
                x: 20.0 + col * 10.0,
                y: 15.0 + row * 8.0,
                inverse_depth: 0.1 + 0.02 * (local % 7) as f64,
            }
        };
        const OLD_FRAME: usize = 2; // arrival index of the frame that gets folded away.
                                    // 50 pinning frames left only a ~9.7x reduction (short of the
                                    // required >10x) — matching the M8 fixture's own documented finding
                                    // that pinning-frame COUNT, not baseline size or patch count, is the
                                    // lever that closes a monocular depth/pose ambiguity; 90 comfortably
                                    // clears the bar with margin, the same way M8's own fixture settled
                                    // on more pinning frames than its first attempt.
        const PINNING_COUNT: usize = 90; // arrival indices 3..=92.
        const DRIFTED_FRAME: usize = OLD_FRAME + PINNING_COUNT; // 92: last pinning frame, stays live.
        const N_BEFORE_ARCHIVE: usize = 3 + PINNING_COUNT; // 93: frame count at call A.
        const N_FILLER: usize = 10;
        const N_AT_FOLD: usize = N_BEFORE_ARCHIVE + N_FILLER; // 103: frame count at call B.
        let keyframe_index = N_AT_FOLD - 2; // candidate = N_AT_FOLD - keyframe_index = 2 = OLD_FRAME.

        let true_pose = |i: usize| {
            SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(i as f64 * 0.2, 0.0, 0.0),
            )
        };
        let drift = Vector3::new(0.15, 0.0, 0.0);

        let mut graph = DpvoPatchGraph::new(m10_graph_config(m, keyframe_index));
        graph.enable_inactive_edge_retention(m * (PINNING_COUNT + 5));

        // Dummy frames 0, 1 — never referenced by any edge; exist purely so
        // OLD_FRAME sits at array index 2 (see this test's own doc for why
        // indices 0/1 can never themselves be folded).
        for i in 0..OLD_FRAME {
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(true_pose(i), intr, (0..m).map(patch_for_local).collect())
                .unwrap();
        }
        // OLD_FRAME itself, hosting `m` distinct patches.
        graph.begin_frame(OLD_FRAME as f64 * 0.05);
        let old_frame_patches: Vec<DpvoPatch> = (0..m).map(patch_for_local).collect();
        graph
            .commit_frame(true_pose(OLD_FRAME), intr, old_frame_patches.clone())
            .unwrap();

        // Many DISTINCT pinning frames observing OLD_FRAME's patches — the
        // same "one edge per patch, many distinct pinning viewpoints"
        // recipe the M8 fixture's own doc derives is required to avoid a
        // depth/pose ambiguity.
        for target in (OLD_FRAME + 1)..=DRIFTED_FRAME {
            graph.begin_frame(target as f64 * 0.05);
            graph
                .commit_frame(
                    true_pose(target),
                    intr,
                    (0..m).map(patch_for_local).collect(),
                )
                .unwrap();
            let pairs: Vec<(usize, usize)> = (0..m)
                .map(|local| (OLD_FRAME * m + local, target))
                .collect();
            graph.append_edges(&pairs, 4);
        }
        for edge in graph.edges_mut() {
            let local = edge.k - OLD_FRAME * m;
            let owner_patch = old_frame_patches[local];
            let target = transform_point(
                &true_pose(OLD_FRAME),
                &true_pose(edge.j),
                &intr,
                &intr,
                &owner_patch,
                false,
            );
            edge.target_weight = Some((target, Vector2::new(1.0, 1.0)));
        }
        assert_eq!(graph.n_frames(), N_BEFORE_ARCHIVE);

        // Call A: archive OLD_FRAME's edges into the inactive store while it
        // is still live — the fold gate cannot fire yet.
        graph.keyframe();
        assert!(
            graph.edges().iter().all(|e| e.i != OLD_FRAME),
            "OLD_FRAME's edges should have aged into the inactive store"
        );
        assert_eq!(graph.inactive_edge_stats(), (m * PINNING_COUNT, 0));
        assert!(
            graph.retained_poses().is_empty(),
            "OLD_FRAME must still be LIVE at this point, not yet folded"
        );

        // Filler frames, no edges — pad `n` up to exactly `N_AT_FOLD`
        // without ever creating an (i=1, j=3) edge (so `motionmag` returns
        // the guaranteed `0.0` "no such edge" case at call B).
        for i in N_BEFORE_ARCHIVE..N_AT_FOLD {
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(true_pose(i), intr, (0..m).map(patch_for_local).collect())
                .unwrap();
        }

        // Call B: fold OLD_FRAME away.
        graph.keyframe();
        assert_eq!(graph.retained_poses().len(), 1);
        assert_eq!(graph.retained_folded_frames().len(), 1);
        assert!(
            graph.frames().iter().all(|f| f.arrival_index != OLD_FRAME),
            "OLD_FRAME must no longer be live"
        );

        // Discover the drifted frame's now-shifted live index and apply the
        // simulated accumulated drift (same recipe as the M8 fixture above).
        let drifted_live = graph
            .frames()
            .iter()
            .position(|f| f.arrival_index == DRIFTED_FRAME)
            .expect("still live");
        let true_drifted_pose = true_pose(DRIFTED_FRAME);
        graph.frames_mut()[drifted_live].pose = SE3::new(
            true_drifted_pose.rotation,
            true_drifted_pose.translation + drift,
        );

        // A minimal keepalive self-edge (mathematically inert — see the M8
        // fixture's own doc for the algebraic cancellation) so the LEGACY
        // (M8-style) path has at least one active edge to compute its own
        // `t0` from, rather than trivially returning `None`.
        graph.append_edges(&[(0, 0)], 4);
        let keepalive_patch = graph.patches()[0];
        for edge in graph.edges_mut() {
            edge.target_weight = Some((
                Vector2::new(keepalive_patch.x, keepalive_patch.y),
                Vector2::new(1.0, 1.0),
            ));
        }

        let loop_pairs = vec![(OLD_FRAME, DRIFTED_FRAME)];

        // --- M8-style (window-pinned) legacy path: PROVABLY cannot touch the drift. ---
        let legacy = gather_global_ba_edges(&graph);
        let error_legacy =
            if let Some((t0, edges, targets, weights, _resolved, unresolved)) = legacy {
                assert!(
                unresolved > 0,
                "the loop edge's old (folded) endpoint must be unresolved under the legacy scheme"
            );
                let poses: Vec<SE3> = graph.frames().iter().map(|f| f.pose.clone()).collect();
                let patches: Vec<DpvoPatch> = graph.patches().to_vec();
                let intrinsics = vec![intr; poses.len()];
                let problem = DpvoBaProblem {
                    poses,
                    patches,
                    intrinsics,
                    edges,
                    targets,
                    weights,
                    depth_damping: None,
                };
                let cfg = DpvoBaConfig {
                    iterations: 2,
                    fixedp: t0,
                    lmbda: 1e-4,
                    ep: 100.0,
                    bounds: [-1e6, -1e6, 1e6, 1e6],
                };
                let solved = dpvo_ba(&problem, &cfg).expect("legacy solve");
                (solved.poses[drifted_live].translation - true_drifted_pose.translation).norm()
            } else {
                drift.norm()
            };
        assert!(
            (error_legacy - drift.norm()).abs() < 1e-6,
            "the legacy path has no edge referencing the drifted pose at all, so it must stay exactly \
             at its drifted value: error_legacy={error_legacy:.6} drift={:.6}",
            drift.norm()
        );

        // --- M10 widened path: reaches back through the fold. ---
        let gathered =
            gather_widened_global_ba_problem(&graph, &loop_pairs, None).expect("widened gather");
        assert_eq!(
            gathered.folded_arrivals,
            vec![OLD_FRAME],
            "the widened gather must materialize OLD_FRAME as a free pose"
        );
        assert_eq!(gathered.fixedp, 0);
        assert_eq!(
            gathered.unresolved_inactive, 0,
            "every pinning+loop edge should now resolve via the folded-frame path"
        );
        assert_eq!(gathered.resolved_inactive, m * PINNING_COUNT);
        assert!(gathered.t0_widened_by_loop_edge);
        assert!(!gathered.capped);

        let ba_cfg = DpvoBaConfig {
            iterations: 2,
            fixedp: gathered.fixedp,
            lmbda: 1e-4,
            ep: 100.0,
            bounds: [-1e6, -1e6, 1e6, 1e6],
        };
        let problem = DpvoBaProblem {
            poses: gathered.poses,
            patches: gathered.patches,
            intrinsics: gathered.intrinsics,
            edges: gathered.edges,
            targets: gathered.targets,
            weights: gathered.weights,
            depth_damping: None,
        };
        let solved = dpvo_ba(&problem, &ba_cfg).expect("widened solve");
        let drifted_combined = 1 + drifted_live; // 1 folded frame prepended, then the live suffix.
        let error_widened =
            (solved.poses[drifted_combined].translation - true_drifted_pose.translation).norm();

        assert!(
            error_widened * 10.0 < error_legacy,
            "widened global BA over the folded-away loop endpoint should close the drift by >10x: \
             widened={error_widened:.6} legacy={error_legacy:.6}"
        );
    }

    /// Milestone M10 (task spec, part G): a tight `max_free_poses` cap can
    /// trim away the very folded endpoint `t0_arrival` was chosen for — this
    /// must be reported via [`WidenedGlobalBaGather::capped`], never silent.
    /// Reuses the same folded-frame fixture shape as the test above, just
    /// with `max_free_poses` set low enough to force the trim.
    #[test]
    fn widened_global_ba_reports_when_max_free_poses_trims_the_folded_endpoint() {
        let m = 4usize; // small enough that this test's own solve isn't the point.
        let intr = DpvoIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 64.0,
            cy: 48.0,
        };
        const OLD_FRAME: usize = 2;
        const PINNING_COUNT: usize = 6;
        const N_BEFORE_ARCHIVE: usize = 3 + PINNING_COUNT; // 9
        const N_FILLER: usize = 3;
        const N_AT_FOLD: usize = N_BEFORE_ARCHIVE + N_FILLER; // 12
        let keyframe_index = N_AT_FOLD - 2;
        let true_pose = |i: usize| {
            SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(i as f64 * 0.2, 0.0, 0.0),
            )
        };
        let patch = |local: usize| DpvoPatch {
            x: 20.0 + local as f64,
            y: 15.0,
            inverse_depth: 0.15,
        };

        let mut graph = DpvoPatchGraph::new(m10_graph_config(m, keyframe_index));
        graph.enable_inactive_edge_retention(1000);
        for i in 0..OLD_FRAME {
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(true_pose(i), intr, (0..m).map(patch).collect())
                .unwrap();
        }
        graph.begin_frame(OLD_FRAME as f64 * 0.05);
        graph
            .commit_frame(true_pose(OLD_FRAME), intr, (0..m).map(patch).collect())
            .unwrap();
        for target in (OLD_FRAME + 1)..=(OLD_FRAME + PINNING_COUNT) {
            graph.begin_frame(target as f64 * 0.05);
            graph
                .commit_frame(true_pose(target), intr, (0..m).map(patch).collect())
                .unwrap();
            let pairs: Vec<(usize, usize)> = (0..m)
                .map(|local| (OLD_FRAME * m + local, target))
                .collect();
            graph.append_edges(&pairs, 4);
        }
        for edge in graph.edges_mut() {
            edge.target_weight = Some((Vector2::new(1.0, 1.0), Vector2::new(1.0, 1.0)));
        }
        assert_eq!(graph.n_frames(), N_BEFORE_ARCHIVE);
        graph.keyframe(); // archive
        for i in N_BEFORE_ARCHIVE..N_AT_FOLD {
            graph.begin_frame(i as f64 * 0.05);
            graph
                .commit_frame(true_pose(i), intr, (0..m).map(patch).collect())
                .unwrap();
        }
        graph.keyframe(); // fold OLD_FRAME
        assert_eq!(graph.retained_poses().len(), 1);
        graph.append_edges(&[(0, 0)], 4);
        for edge in graph.edges_mut() {
            edge.target_weight = Some((Vector2::new(1.0, 1.0), Vector2::new(1.0, 1.0)));
        }

        let loop_pairs = vec![(OLD_FRAME, OLD_FRAME + PINNING_COUNT)];

        // Uncapped: reaches the folded frame.
        let uncapped =
            gather_widened_global_ba_problem(&graph, &loop_pairs, None).expect("uncapped gather");
        assert_eq!(uncapped.folded_arrivals, vec![OLD_FRAME]);
        assert!(!uncapped.capped);

        // Capped to exactly the live frame count: no room left for the
        // folded prefix at all.
        let live_n = graph.n_frames();
        let capped = gather_widened_global_ba_problem(&graph, &loop_pairs, Some(live_n))
            .expect("capped gather");
        assert!(
            capped.folded_arrivals.is_empty(),
            "a cap with no headroom must drop the folded prefix entirely"
        );
        assert!(capped.capped, "the trim must be reported, never silent");

        // The live graph itself can exceed the cap after every folded pose
        // has already been removed. The oldest live poses must then become
        // fixed variables; the configured bound is on the solve, not only
        // on the optional folded prefix.
        let tight = gather_widened_global_ba_problem(&graph, &loop_pairs, Some(1))
            .expect("tight live-graph cap");
        assert_eq!(tight.poses.len() - tight.fixedp, 1);
        assert!(tight.capped);
    }

    /// Milestone M10 (task spec, part G): "no-op when disabled" — with
    /// `DpvoGlobalBaConfig::widen_t0_with_loop_edges` at its default
    /// (`false`), [`DpvoOdometry::run_global_ba`] must dispatch to the
    /// byte-identical M8 legacy path regardless of what
    /// `loop_edge_arrival_pairs` contains — checked here at the "gather"
    /// level (the pure functions), since exercising the full
    /// `DpvoOdometry` requires a live ONNX session.
    #[test]
    fn default_global_ba_config_does_not_widen() {
        let cfg = DpvoGlobalBaConfig::default();
        assert!(
            !cfg.widen_t0_with_loop_edges,
            "M8 legacy behavior must be the default"
        );
        assert_eq!(cfg.max_free_poses, Some(256));
    }
}

/// Given per-item anchor features (`num_items, 128, 3, 3`) and per-item
/// reprojected `3×3` grids (`num_items, 3, 3, 2`, in level-0/native pixel
/// coordinates), run the 2-pyramid-level correlation lookup and interleave
/// into DPVO's own `(num_items, 882)` layout — `DPVO.corr` (`dpvo.py:200-
/// 207`): upstream `altcorr` first returns `(dx, dy, patch_y, patch_x)`, then
/// `torch.stack([corr1, corr2], -1).view(1, len(ii), -1)` makes pyramid level
/// fastest, inside patch x, patch y, dy, and dx.
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
        for t in 0..taps * taps {
            for py in 0..PATCH {
                for px in 0..PATCH {
                    let base = ((t * PATCH + py) * PATCH + px) * 2;
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

/// Compact arbitrary frame indices into a deterministic dense target list
/// plus one remapped index per edge. Native correlation only needs these
/// referenced pyramids, not every frame retained by the trajectory graph.
fn compact_target_indices(edge_targets: &[usize]) -> (Vec<usize>, Vec<i32>) {
    let mut frames = edge_targets.to_vec();
    frames.sort_unstable();
    frames.dedup();
    let remapped = edge_targets
        .iter()
        .map(|target| {
            frames
                .binary_search(target)
                .expect("edge target was collected") as i32
        })
        .collect();
    (frames, remapped)
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

    #[test]
    fn native_target_compaction_is_sorted_dense_and_order_preserving() {
        let (frames, remapped) = compact_target_indices(&[310, 7, 310, 42, 7]);
        assert_eq!(frames, vec![7, 42, 310]);
        assert_eq!(remapped, vec![2, 0, 2, 1, 0]);
    }

    #[test]
    fn scale_cliff_diagnostics_ignore_rejected_proposals_and_keep_run_maximum() {
        let mut max_committed = 0.0;
        let mut scale_jump_rejections = 0;
        update_sim3_scale_cliff_diagnostics(
            &mut max_committed,
            &mut scale_jump_rejections,
            false,
            Some(Sim3BackendRejection::ScaleJump),
            0.01,
            100.0,
        );
        assert_eq!(
            max_committed, 0.0,
            "a rolled-back proposal is not a cliff event"
        );
        assert_eq!(scale_jump_rejections, 1);

        update_sim3_scale_cliff_diagnostics(
            &mut max_committed,
            &mut scale_jump_rejections,
            true,
            None,
            0.8,
            1.1,
        );
        assert!((max_committed - 0.8_f64.ln().abs()).abs() < 1.0e-12);
        update_sim3_scale_cliff_diagnostics(
            &mut max_committed,
            &mut scale_jump_rejections,
            true,
            None,
            0.9,
            1.05,
        );
        assert!((max_committed - 0.8_f64.ln().abs()).abs() < 1.0e-12);
        assert_eq!(scale_jump_rejections, 1);
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
            assert!(
                !tripped,
                "must not roll back before {threshold} consecutive bad frames"
            );
            consecutive = next;
        }
        let (next, tripped) = rollback_monitor_step(10_000.0, bound, consecutive, threshold);
        assert_eq!(next, 5);
        assert!(
            tripped,
            "must roll back on the {threshold}th consecutive bad frame"
        );

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
        let intr = DpvoIntrinsics {
            fx: 190.0,
            fy: 190.0,
            cx: 94.0,
            cy: 60.0,
        };
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
        let level0 =
            Array3::<f32>::from_shape_fn((FNET_DIM, level0_h, level0_w), |_| next_f64() as f32);
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
        let corr_flat = corr_pyramid(
            anchor_gmap.view(),
            coords_grid_px.view(),
            &level0_hwc,
            &level1_hwc,
        );
        let corr_ms = corr_start.elapsed().as_secs_f64() * 1000.0;
        println!("  [perf] corr_pyramid (2 levels) x{num_edges} edges: {corr_ms:.3} ms/call");
        assert_eq!(corr_flat.shape(), &[num_edges, CORR_DIM]);

        println!(
            "  [perf] total correlation-assembly (reproject + corr_pyramid): {:.3} ms/call",
            reproject_ms + corr_ms
        );
    }
}

/// Milestone M14: [`LowParallaxRegimeState`] is a free-standing,
/// signal-agnostic hysteresis state machine — tested directly here, without
/// constructing a live `DpvoOdometry` (mirroring this file's own convention,
/// e.g. `global_ba_due`/`gyro_bootstrap_gate_check`'s own test modules). Its
/// actual production signal ([`DpvoOdometry::motion_probe`], an ONNX-backed
/// GRU correction magnitude — see the module doc's own "Low-parallax hover
/// freeze" section for why) is not free-standing and is instead exercised
/// via real MH_01 calibration runs (`docs/dpvo_droid_port_plan.md`'s "M14
/// results"), not a unit test.
#[cfg(test)]
mod low_parallax_tests {
    use super::*;

    fn cfg(window: usize, enter_flow: f64, exit_flow: f64) -> DpvoLowParallaxConfig {
        // `response`/`depth_damp_factor`/`unflag_after_commits` are
        // irrelevant to every test in this module — they exercise
        // `LowParallaxRegimeState` directly, which has no knowledge of
        // `response` at all (see [`LowParallaxResponse`]'s own doc: only
        // `DpvoOdometry::low_parallax_gate`'s final `match` reads it).
        DpvoLowParallaxConfig {
            window,
            enter_flow,
            exit_flow,
            response: LowParallaxResponse::Freeze,
            depth_damp_factor: 1000.0,
            unflag_after_commits: 16,
            gradual_release_duration_commits: 0,
            gradual_release_start_cap_frames: 4,
        }
    }

    #[test]
    fn stays_out_of_regime_while_the_window_is_still_filling() {
        let c = cfg(5, 1.0, 4.0);
        let mut state = LowParallaxRegimeState::default();
        // Only 4 of the 5-frame window filled with low readings: not yet
        // enough history for a reliable statistic, so this must be a no-op
        // even though every reading so far is well below `enter_flow`.
        for _ in 0..4 {
            let t = state.update(&c, 0.1);
            assert!(!t.suppress && !t.just_entered);
        }
        assert!(!state.in_regime());
    }

    #[test]
    fn enters_the_instant_the_full_window_median_drops_below_enter_flow() {
        let c = cfg(5, 1.0, 4.0);
        let mut state = LowParallaxRegimeState::default();
        for _ in 0..4 {
            state.update(&c, 0.1);
        }
        // The 5th low reading fills the window; its median (0.1) is well
        // below `enter_flow` (1.0) => enters THIS frame, and this frame is
        // itself suppressed (not the next one).
        let t5 = state.update(&c, 0.1);
        assert!(
            t5.suppress,
            "the frame that fills the window below threshold must itself be suppressed"
        );
        assert!(t5.just_entered);
        assert!(!t5.just_exited);
        assert!(state.in_regime());
        // Regime stays active on the next low (or even moderate,
        // sub-exit-threshold) reading, with no further "just_entered".
        let t6 = state.update(&c, 2.0);
        assert!(t6.suppress);
        assert!(!t6.just_entered);
        assert!(!t6.just_exited);
        assert!(state.in_regime());
    }

    #[test]
    fn a_single_high_reading_is_absorbed_by_the_window_median_not_a_hard_reset() {
        // The whole point of windowed-median smoothing over a raw
        // consecutive-streak requirement: ONE noisy high reading among
        // otherwise-low ones must not prevent entry, as long as the
        // window's MEDIAN still clears the threshold — see
        // `DpvoLowParallaxConfig::default`'s own doc, finding 2, for the
        // real-run evidence a strict all-consecutive-frames streak design
        // was too fragile to this exact kind of noise.
        let c = cfg(5, 3.0, 6.0);
        let mut state = LowParallaxRegimeState::default();
        // [0.1, 0.1, 5.0, 0.1, 0.1] -> sorted [0.1,0.1,0.1,0.1,5.0] -> median 0.1.
        let readings = [0.1, 0.1, 5.0, 0.1, 0.1];
        let mut last = None;
        for r in readings {
            last = Some(state.update(&c, r));
        }
        let t = last.unwrap();
        assert!(
            t.suppress && t.just_entered,
            "one noisy high reading must not block entry when the window median is still low"
        );
    }

    #[test]
    fn exits_once_the_window_median_reaches_exit_flow_and_disarms() {
        let c = cfg(3, 1.0, 4.0);
        let mut state = LowParallaxRegimeState::default();
        for _ in 0..3 {
            state.update(&c, 0.1);
        }
        assert!(state.in_regime());
        // A window still mostly low (median under exit_flow) must NOT exit.
        let mid = state.update(&c, 3.9); // window now [0.1,0.1,3.9] -> median 0.1.
        assert!(
            mid.suppress && !mid.just_exited,
            "window median still below exit_flow must not exit"
        );
        assert!(state.in_regime());
        // Push enough high readings that the window's OWN median clears
        // exit_flow.
        state.update(&c, 5.0); // [0.1,3.9,5.0] -> median 3.9, still < 4.0.
        let exited = state.update(&c, 5.0); // [3.9,5.0,5.0] -> median 5.0 >= 4.0.
        assert!(
            !exited.suppress,
            "the frame whose window median proves the hover is over must not be suppressed"
        );
        assert!(exited.just_exited);
        assert!(!state.in_regime());
        assert!(
            state.disarmed(),
            "exiting must permanently disarm the one-shot guard"
        );
    }

    #[test]
    fn disarmed_state_never_re_enters_even_with_sustained_low_readings_afterward() {
        // This is the mechanism's own deliberate limitation (see
        // `DpvoLowParallaxConfig::default`'s own doc, finding 3): a
        // real-run 800f trace showed `motion_probe`'s baseline dropping
        // "hover-like" again later in the SAME run for reasons unrelated to
        // true stillness, and re-triggering there corrupted the
        // trajectory. Once disarmed, sustained low readings must never
        // re-arm the detector.
        let c = cfg(3, 1.0, 4.0);
        let mut state = LowParallaxRegimeState::default();
        for _ in 0..3 {
            state.update(&c, 0.1);
        }
        assert!(state.in_regime());
        for _ in 0..3 {
            state.update(&c, 10.0); // forces the window median well above exit_flow.
        }
        assert!(state.disarmed());
        assert!(!state.in_regime());
        // Many more sustained low readings, well past the window size.
        for _ in 0..50 {
            let t = state.update(&c, 0.05);
            assert!(
                !t.suppress && !t.just_entered && !t.just_exited,
                "a disarmed detector must never re-enter"
            );
        }
        assert!(!state.in_regime());
    }

    #[test]
    fn disabled_mechanism_is_a_pure_no_op_by_construction() {
        // There is no "disabled" state for `LowParallaxRegimeState` itself
        // (it is only ever driven when `config.low_parallax` is `Some` — see
        // `DpvoOdometry::low_parallax_gate`'s own early `let Some(cfg) = ...
        // else { return false }`), so the no-op contract is structural, not
        // a runtime flag this state machine itself carries. What IS testable
        // here: a config whose window can never fill within any realistic
        // run length behaves as an unconditional no-op.
        let c = cfg(usize::MAX, 1.0, 4.0);
        let mut state = LowParallaxRegimeState::default();
        for _ in 0..1000 {
            let t = state.update(&c, 0.0);
            assert!(!t.suppress && !t.just_entered && !t.just_exited);
        }
        assert!(!state.in_regime());
    }
}

/// Milestone M15: [`LowParallaxDampState`] is free-standing (no
/// `DpvoOdometry`/ONNX dependency — see that struct's own doc, mirroring
/// [`LowParallaxRegimeState`]'s own testability precedent above), so its
/// flag/un-flag lifecycle and per-patch multiplier construction are fully
/// unit-testable without a live ONNX session.
#[cfg(test)]
mod low_parallax_damp_tests {
    use super::*;

    #[test]
    fn multipliers_is_none_when_nothing_has_ever_been_flagged() {
        let mut state = LowParallaxDampState::default();
        assert_eq!(state.multipliers(&[1, 2, 3], 4, 1000.0), None);
        assert_eq!(
            state.damped_solve_count(),
            0,
            "an unaffected solve must not be counted as damped"
        );
    }

    #[test]
    fn flag_then_multipliers_damps_only_the_flagged_frames_block() {
        let mut state = LowParallaxDampState::default();
        state.flag(5, 3); // arrival_index 5, patches_per_frame 3.
                          // Window [4, 5, 6): only arrival 5 (local block 1) is flagged.
        let out = state
            .multipliers(&[4, 5, 6], 3, 1000.0)
            .expect("frame 5 is live and flagged, must damp");
        assert_eq!(out.len(), 9);
        assert_eq!(
            &out[0..3],
            &[1.0, 1.0, 1.0],
            "frame 4's block must be untouched"
        );
        assert_eq!(
            &out[3..6],
            &[1000.0, 1000.0, 1000.0],
            "frame 5's block must be damped"
        );
        assert_eq!(
            &out[6..9],
            &[1.0, 1.0, 1.0],
            "frame 6's block must be untouched"
        );
        assert_eq!(state.damped_solve_count(), 1);
        // A second call with the SAME (still-flagged) window must count as
        // another genuinely-damped solve.
        state.multipliers(&[4, 5, 6], 3, 1000.0);
        assert_eq!(state.damped_solve_count(), 2);
    }

    #[test]
    fn multipliers_is_none_when_the_flagged_frame_is_outside_this_windows_arrivals() {
        // A frame can be flagged (still tracked) but not appear in a
        // PARTICULAR `dpvo_ba` call's own window (e.g. it fell outside the
        // per-frame windowed solve's `[frame_lo, n)` bound, even though it
        // is still live in the full graph) — that solve must be a true
        // no-op, not accidentally damp some unrelated local index.
        let mut state = LowParallaxDampState::default();
        state.flag(100, 4);
        assert_eq!(state.multipliers(&[1, 2, 3], 4, 1000.0), None);
        assert_eq!(state.damped_solve_count(), 0);
    }

    #[test]
    fn flag_is_idempotent_and_does_not_double_count() {
        let mut state = LowParallaxDampState::default();
        state.flag(7, 5);
        state.flag(7, 5); // same arrival again — must not double-count.
        assert_eq!(state.frames_flagged_total(), 1);
        assert_eq!(state.patches_flagged_total(), 5);
        assert_eq!(state.currently_damped_frames(), 1);
    }

    #[test]
    fn advance_unflagging_never_fires_while_still_in_regime() {
        // Even an arbitrarily large age gap must not un-flag while the
        // regime is still active — see `advance_unflagging`'s own doc for
        // why (nothing has had a chance to accumulate real parallax yet).
        let mut state = LowParallaxDampState::default();
        state.flag(0, 4);
        state.advance_unflagging(1_000_000, 5, true);
        assert_eq!(state.currently_damped_frames(), 1);
        assert_eq!(state.unflagged_total(), 0);
    }

    #[test]
    fn advance_unflagging_removes_once_the_age_threshold_is_reached() {
        let mut state = LowParallaxDampState::default();
        state.flag(10, 4); // born at arrival_index 10.
                           // Age 4 (< unflag_after_commits=5): must stay flagged.
        state.advance_unflagging(14, 5, false);
        assert_eq!(state.currently_damped_frames(), 1);
        assert_eq!(state.unflagged_total(), 0);
        // Age 5 (>= 5): un-flags now.
        state.advance_unflagging(15, 5, false);
        assert_eq!(state.currently_damped_frames(), 0);
        assert_eq!(state.unflagged_total(), 1);
        // A subsequent solve over a window that still includes arrival 10
        // must no longer be damped.
        assert_eq!(state.multipliers(&[9, 10, 11], 4, 1000.0), None);
    }

    #[test]
    fn advance_unflagging_is_self_cleaning_for_frames_far_older_than_now() {
        // A frame flagged long ago (and, in a real run, very likely already
        // pruned from the live graph) still ages out purely from `now`
        // growing — see `advance_unflagging`'s own doc, "self-cleaning".
        let mut state = LowParallaxDampState::default();
        state.flag(0, 4);
        state.flag(50, 4);
        state.advance_unflagging(60, 5, false);
        assert_eq!(
            state.currently_damped_frames(),
            0,
            "both entries are well past the age threshold"
        );
        assert_eq!(state.unflagged_total(), 2);
    }

    #[test]
    fn gradual_release_is_monotonic_and_reaches_one_without_a_cliff() {
        let mut state = LowParallaxDampState::default();
        state.flag(0, 1);
        let mut values = Vec::new();
        for now in 10..14 {
            state.advance_gradual_release(now, 5, false, 4, 1);
            values.push(state.multipliers(&[0], 1, 1000.0).unwrap()[0]);
        }
        assert_eq!(values[0], 1000.0, "release starts at full damping");
        assert!(values.windows(2).all(|pair| pair[1] < pair[0]));
        state.advance_gradual_release(14, 5, false, 4, 1);
        assert_eq!(state.multipliers(&[0], 1, 1000.0), None);
        assert_eq!(state.unflagged_total(), 1);
    }

    #[test]
    fn gradual_release_start_count_is_bounded_and_oldest_first() {
        let mut state = LowParallaxDampState::default();
        for arrival in 0..10 {
            state.flag(arrival, 3);
        }
        state.advance_gradual_release(20, 5, false, 8, 2);
        assert_eq!(state.currently_releasing_frames(), 2);
        assert!(state.release_started_at.contains_key(&0));
        assert!(state.release_started_at.contains_key(&1));
        assert_eq!(state.max_release_started_per_advance(), 2);
        assert_eq!(state.release_histogram_frames(), [8, 2, 0, 0, 0]);

        state.advance_gradual_release(21, 5, false, 8, 2);
        assert_eq!(state.currently_releasing_frames(), 4);
        assert_eq!(state.release_started_total(), 4);
        assert_eq!(state.max_release_started_per_advance(), 2);
        assert_eq!(state.release_histogram_frames().iter().sum::<usize>(), 10);
    }

    #[test]
    fn gradual_release_never_starts_during_hover() {
        let mut state = LowParallaxDampState::default();
        state.flag(0, 4);
        state.advance_gradual_release(1_000, 5, true, 8, 4);
        assert_eq!(state.currently_releasing_frames(), 0);
        assert_eq!(state.release_started_total(), 0);
        assert_eq!(state.multipliers(&[0], 4, 1000.0).unwrap(), vec![1000.0; 4]);
    }
}
