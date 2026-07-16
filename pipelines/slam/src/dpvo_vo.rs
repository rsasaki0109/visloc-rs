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
//! # What this module does not implement (see `crate::dpvo_patch_graph`'s
//! module doc for the graph-level list)
//!
//! Loop closure (DPV-SLAM, Milestone M6) and the global-BA fallback are out
//! of scope, matching `crate::dpvo_patch_graph`'s own documented omissions.
#![cfg(feature = "onnx-inference")]

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use nalgebra::Vector2;
use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use visloc_core::geometry::SE3;
use visloc_vision::dpvo::correlation::corr_cpu;
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
use crate::dpvo_patch_graph::{DpvoGraphError, DpvoPatchGraph, DpvoVoConfig};

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
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone)]
struct FramePyramid {
    level0: Array3<f32>,
    level1: Array3<f32>,
}

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
        Ok(Self {
            config,
            session,
            agg_kk,
            agg_ij,
            graph,
            frame_pyramids: Vec::new(),
            patch_gmap: Vec::new(),
            patch_imap: Vec::new(),
            rng: StdRng::seed_from_u64(config.seed),
            stats: DpvoOdometryStats::default(),
        })
    }

    pub fn stats(&self) -> DpvoOdometryStats {
        self.stats
    }

    pub fn graph(&self) -> &DpvoPatchGraph {
        &self.graph
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
        let level1 = avg_pool_4x4(fmap.view());
        let candidate_pyramid = FramePyramid { level0: fmap, level1 };

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
            self.update_step()?;
            if let Some(k) = self.graph.keyframe() {
                self.frame_pyramids.remove(k);
                let m = self.graph.config().patches_per_frame;
                self.patch_gmap.drain(k * m..(k + 1) * m);
                self.patch_imap.drain(k * m..(k + 1) * m);
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
            candidate_pyramid.level0.view(),
            candidate_pyramid.level1.view(),
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

    /// One `update()` call (`dpvo.py:328-360`): reproject every active
    /// edge's patch grid, assemble the 2-pyramid-level correlation tensor
    /// (grouped by target frame — see the module doc's windowing/`corr_cpu`
    /// notes), run the GRU update cell, then a windowed [`dpvo_ba`] call.
    fn update_step(&mut self) -> Result<(), DpvoOdometryError> {
        let n = self.graph.n_frames();
        let removal_window = self.graph.config().removal_window;
        let patch_lifetime = self.graph.config().patch_lifetime;
        let optimization_window = self.graph.config().optimization_window;
        let patches_per_frame = self.graph.config().patches_per_frame;
        let frame_lo = n.saturating_sub(removal_window + patch_lifetime);

        let edges = self.graph.edges().to_vec();
        if edges.is_empty() {
            return Ok(());
        }
        let e_count = edges.len();

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
                target_pyramid.level0.view(),
                target_pyramid.level1.view(),
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
            "update_step's window [frame_lo,n) did not cover every active edge — the windowing \
             derivation in this module's doc comment is unsound for this config \
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
        let solved = dpvo_ba(&problem, &ba_config)?;
        self.stats.ba_ms_total += ba_start.elapsed().as_secs_f64() * 1000.0;

        for (local, pose) in solved.poses.into_iter().enumerate() {
            self.graph.frames_mut()[frame_lo + local].pose = pose;
        }
        for (local, patch) in solved.patches.into_iter().enumerate() {
            self.graph.patches_mut()[patches_lo + local] = patch;
        }
        Ok(())
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
    target_level0: ArrayView3<'_, f32>,
    target_level1: ArrayView3<'_, f32>,
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
    let corr1 = corr_cpu(anchor_gmap, target_level0, coords_grid_px, CORR_RADIUS);
    let corr2 = corr_cpu(anchor_gmap, target_level1, coords_l1.view(), CORR_RADIUS);
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

    #[test]
    fn corr_pyramid_shape_matches_corr_dim() {
        let anchor = Array4::<f32>::zeros((2, FNET_DIM, PATCH, PATCH));
        let coords = Array4::<f32>::zeros((2, PATCH, PATCH, 2));
        let level0 = Array3::<f32>::zeros((FNET_DIM, 16, 16));
        let level1 = Array3::<f32>::zeros((FNET_DIM, 4, 4));
        let out = corr_pyramid(anchor.view(), coords.view(), level0.view(), level1.view());
        assert_eq!(out.shape(), &[2, CORR_DIM]);
    }
}
