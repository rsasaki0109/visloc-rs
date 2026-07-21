//! Native (Rust, no ONNX) correlation lookup, ported from
//! `scripts/export_dpvo_onnx.py`'s `corr_cpu` — itself the export script's
//! *own* reimplementation of upstream's CUDA-only `altcorr.corr` (no pure
//! Python/PyTorch reference exists upstream for this op either; see this
//! file's honesty caveat, same as [`super::patchify`]'s).
//!
//! **Honesty caveat (carried from the plan doc's M1 results):** not
//! verified against the real CUDA kernel — no CUDA toolchain available.
//! Passing [`corr_cpu`]'s fixture-based parity test demonstrates agreement
//! with `export_dpvo_onnx.py`'s own reference reimplementation of the
//! (otherwise uninspectable) kernel's documented behaviour — a normalized
//! dot-product cost volume between a patch's per-pixel anchor feature
//! vector and a bilinearly-sampled neighbourhood of a target feature map —
//! not bit-parity with upstream.
//!
//! # Scope note: single target frame per call
//!
//! In full DPVO, different edges within one update-cell call can reference
//! *different* destination frames (`jj` varies per edge), each needing its
//! own slice of a per-frame feature-map pyramid. The parity fixture this
//! function is checked against (`correlation_fixture.npz`) only exercises
//! the single-shared-target-frame case (all edges in that fixture correlate
//! against the same frame), so [`corr_cpu`] takes one `(channels, height,
//! width)` target map per call rather than a `(num_edges, channels, height,
//! width)` per-edge stack. Grouping edges by `jj` and looping (or batching)
//! this function once per distinct target frame — and assembling the
//! 2-pyramid-level, `LEVELS·(2·radius+1)²·PATCH²` = 882-wide `corr` tensor
//! the update cell's ONNX graphs expect — is left to the M3/M4 integration
//! layer that actually owns the patch graph's per-edge `jj` bookkeeping;
//! see the plan doc's M2 blockers list. (M4's `dpvo_vo.rs::corr_pyramid`
//! is exactly that grouping/assembly layer.)
//!
//! # M4-perf rewrite (`docs/dpvo_droid_port_plan.md`, "M4-perf results")
//!
//! M4's own EuRoC integration run measured this function (plus the caller's
//! reprojection and pyramid-assembly wrapper) at **~3.1 s/frame** at DPVO's
//! real `fast.yaml`-scale working set (few thousand active edges) — 8-10×
//! the combined cost of the ONNX encoder, GRU update, and BA stages, and the
//! single largest blocker to a usable CPU-only per-frame budget. Profiling
//! (see the ignored perf test at the bottom of this file) pinned the cost on
//! two things the original nested-loop implementation did *not* do well,
//! neither of which is an algorithmic change to what is being computed:
//!
//! 1. **Non-contiguous channel access.** `patch_feats` is `(num_edges,
//!    channels, patch, patch)` and `target_fmap` is `(channels, height,
//!    width)` — both channel-*first*, so a per-pixel 128-channel feature
//!    vector is strided by `patch²` (anchor side) or `height·width` (target
//!    side) elements between channels, not contiguous. Every one of this
//!    function's bilinear-sample-then-dot-product inner steps walked that
//!    stride 128 times, guaranteeing a cache miss (or at best an
//!    unvectorizable gather) per channel per corner per tap. This layout is
//!    fixed at the public API boundary (every fixture, and every writer —
//!    [`super::patchify::patchify_cpu`], the `fnet`/`inet` ONNX outputs —
//!    already commits to it, matching upstream's own `(N,C,H,W)` tensor
//!    convention), so rather than changing the writers (which would ripple
//!    into every fixture and ONNX-output consumer), this function now
//!    transposes both inputs **once per call** into channel-*last* scratch
//!    buffers ([`transpose_chw_to_hwc`], [`transpose_epp_c_first_to_last`]) —
//!    `O(channels·height·width)` and `O(num_edges·channels·patch²)`
//!    respectively, i.e. paid once instead of once per `(edge, patch-pixel,
//!    tap, corner)` combination — after which every per-corner dot product
//!    below is a straight contiguous `f32` slice reduction ([`dot_slice`]),
//!    letting the compiler autovectorize it (verified in `--release`, see
//!    the perf test).
//! 2. **Redundant per-tap work.** The original implementation recomputed
//!    `x.floor()`/the fractional part and the four bilinear corner weights
//!    inside the 49-tap loop, even though the tap offsets `dx, dy ∈
//!    {-radius, ..., radius}` are *integers* — so `floor(center + dx) =
//!    floor(center) + dx` and the fractional part (hence every bilinear
//!    weight) is identical for all 49 taps of a given `(edge, patch-pixel)`.
//!    Those are now computed once per `(edge, patch-pixel)` and reused
//!    across taps ([`BilinearWeights`]). This function also fuses the old
//!    two-pass "accumulate a weighted neighbourhood vector, then dot it
//!    against the anchor" into one pass per corner — algebraically
//!    equivalent by linearity of the dot product (`anchor · Σ wₖ·corner_k =
//!    Σ wₖ·(anchor · corner_k)`), so this is a floating-point *reordering*
//!    (permitted — the task's own numerics constraint is "fp reordering
//!    only"), not a different computation.
//! 3. **Single-threaded.** `rayon` (already a dependency of `pipelines/slam`
//!    and this workspace's root crate — not a new addition to the
//!    dependency tree; see this crate's `Cargo.toml` for the exact
//!    citation) now parallelizes the outer loop over edges, since each
//!    edge's output slice is independent and disjoint.
//!
//! None of this changes what is computed: [`corr_cpu`]'s public signature,
//! its documented border/normalization semantics, and its existing
//! fixture-based parity test (`crates/vision/tests/dpvo_onnx_parity.rs`)
//! are all unchanged, and now exercise this rewritten implementation
//! directly. A slow, deliberately-naive reference implementation
//! ([`corr_cpu_reference`], `cfg(test)`-only) is kept byte-for-byte
//! equivalent to the pre-M4-perf code so a realistic-shape equivalence test
//! ([`fast_matches_naive_reference_at_realistic_shape`]) can catch any
//! future accidental numerics drift, independent of the fixture (which is a
//! small, fixed shape).

use ndarray::{Array4, ArrayView3, ArrayView4};
// `rayon` is unconditionally available here: this whole `dpvo` module tree
// is only ever compiled under the `onnx-inference` feature
// (`crates/vision/src/dpvo/mod.rs`'s `#![cfg(feature = "onnx-inference")]`),
// and that feature always pulls in `dep:rayon` (see this crate's
// `Cargo.toml`) — no separate `rayon` feature/cfg is needed.
use rayon::prelude::*;

/// Correlation lookup: for every patch pixel of every edge's anchor patch,
/// sample a `(2·radius+1)²`-tap neighbourhood of `target_fmap` around a
/// caller-supplied centre coordinate and take the (channel-normalized) dot
/// product against that patch pixel's own anchor feature vector.
///
/// * `patch_feats`: `(num_edges, channels, patch, patch)` — per-edge anchor
///   patch features (typically [`super::patchify::patchify_cpu`]'s output
///   for the *source* frame of each edge).
/// * `target_fmap`: `(channels, height, width)` — the (single, shared —
///   see module doc) destination frame's feature map.
/// * `coords_center`: `(num_edges, patch, patch, 2)` — the `(x, y)` sample
///   centre, in `target_fmap`'s pixel grid, for each patch pixel of each
///   edge (in general this is the *current pose estimate*'s reprojection
///   of that patch pixel into the target frame — computing that
///   reprojection is a caller/BA-loop concern, out of scope here; this
///   function only consumes already-computed centres).
/// * `radius`: correlation lookup radius (DPVO: `3`, giving `7×7 = 49`
///   taps per pyramid level — see [`super::CORR_RADIUS`]).
///
/// Returns `(num_edges, patch, patch, (2·radius+1)²)`.
///
/// # Bilinear sampling border behaviour
///
/// Unlike [`super::patchify::patchify_cpu`] (which **clamps** out-of-range
/// samples to the border), this matches `grid_sample(...,
/// padding_mode="zeros", align_corners=True)`'s semantics: any of the four
/// bilinear corner samples that falls outside `[0, width) × [0, height)`
/// contributes exactly `0.0` instead of a clamped border value. This
/// difference is intentional and upstream-documented (patchify and
/// correlation are different ops with different padding conventions in the
/// Python reference this was ported from), not an inconsistency to
/// "fix" — flagging it here since it is easy to assume the two should
/// match.
///
/// # Performance
///
/// See this module's doc comment ("M4-perf rewrite") for the hotspot
/// analysis and the three changes that took this from ~3.1 s/frame to
/// (per the plan doc's "M4-perf results" section) the new measured cost at
/// DPVO's real working set. Numerically, this differs from the pre-M4-perf
/// implementation only by floating-point summation-order reordering (see
/// [`corr_cpu_reference`] and its equivalence test).
pub fn corr_cpu(
    patch_feats: ArrayView4<'_, f32>,
    target_fmap: ArrayView3<'_, f32>,
    coords_center: ArrayView4<'_, f32>,
    radius: usize,
) -> Array4<f32> {
    // See [`corr_cpu_prebuilt_target`]'s doc for why this split exists (the
    // M4-perf caching win: a caller that reuses the same target frame across
    // many calls — as `pipelines/slam/src/dpvo_vo.rs`'s per-frame loop does —
    // can transpose it once and skip this step on every subsequent call).
    let fmap_hwc = ChannelLastImage::from_chw(target_fmap);
    corr_cpu_prebuilt_target(patch_feats, &fmap_hwc, coords_center, radius)
}

/// Same as [`corr_cpu`], but takes an already-channel-last-transposed target
/// feature map instead of transposing one internally.
///
/// # Why this exists (M4-perf caching, `docs/dpvo_droid_port_plan.md`)
///
/// [`corr_cpu`]'s one-time-per-call [`ChannelLastImage::from_chw`] transpose
/// (module doc, point 1) is `O(channels·height·width)` — cheap next to the
/// `O(num_edges·taps²)` tap loop for any single call, but DPVO's real
/// integration (`pipelines/slam/src/dpvo_vo.rs`'s `DpvoOdometry`) calls
/// [`corr_cpu`] once per `(target frame, pyramid level)` pair *per active
/// edge group*, and the same target frame is typically referenced across
/// many consecutive per-frame `update()` calls for as long as it stays
/// inside the active window (`REMOVAL_WINDOW`/`PATCH_LIFETIME`) — so the
/// same feature map was being re-transposed from scratch on every one of
/// those calls. This function lets a caller that already caches a frame's
/// feature-map pyramid (`dpvo_vo.rs`'s `FramePyramid`) build the
/// [`ChannelLastImage`] **once**, at the point the pyramid itself is built,
/// and reuse it for every subsequent correlation lookup against that frame —
/// eliminating the redundant re-transpose entirely rather than just making
/// each one faster. Numerically identical to calling [`corr_cpu`] with the
/// same (pre-transpose) `target_fmap`.
pub fn corr_cpu_prebuilt_target(
    patch_feats: ArrayView4<'_, f32>,
    fmap_hwc: &ChannelLastImage,
    coords_center: ArrayView4<'_, f32>,
    radius: usize,
) -> Array4<f32> {
    let (num_edges, channels, patch, patch_check) = patch_feats.dim();
    debug_assert_eq!(
        patch, patch_check,
        "patch_feats must be square in its last two dims"
    );
    debug_assert_eq!(
        channels, fmap_hwc.channels,
        "patch_feats/target_fmap channel count mismatch"
    );
    let taps = 2 * radius + 1;
    let scale = (channels as f32).sqrt();

    // One-time channel-last transpose of the anchor patch features (see
    // module doc, point 1) — the target side's equivalent transpose is the
    // caller-provided `fmap_hwc` (see this function's own doc for why it is
    // not rebuilt here).
    let feats_hwc = ChannelLastPatches::from_ecpp(patch_feats);

    let mut out = Array4::<f32>::zeros((num_edges, patch, patch, taps * taps));
    let chunk_len = patch * patch * taps * taps;
    let out_slice = out
        .as_slice_mut()
        .expect("corr_cpu's freshly-allocated output array is always contiguous C-order");

    let per_edge = |edge: usize, out_chunk: &mut [f32]| {
        for py in 0..patch {
            for px in 0..patch {
                let center_x = coords_center[(edge, py, px, 0)];
                let center_y = coords_center[(edge, py, px, 1)];
                // Bilinear weights depend only on the *fractional* part of
                // the centre coordinate, which is identical across every
                // tap of this patch pixel (see module doc, point 2) — so
                // this is computed once here, not once per tap.
                let weights = BilinearWeights::new(center_x, center_y);
                let anchor = feats_hwc.row(edge, py, px);
                let out_row = &mut out_chunk
                    [(py * patch + px) * taps * taps..(py * patch + px + 1) * taps * taps];
                for ty in 0..taps {
                    let dy = ty as isize - radius as isize;
                    let iy0 = weights.iy0 + dy;
                    for tx in 0..taps {
                        let dx = tx as isize - radius as isize;
                        let ix0 = weights.ix0 + dx;
                        out_row[ty * taps + tx] =
                            weights.sample_dot(fmap_hwc, ix0, iy0, anchor) / scale;
                    }
                }
            }
        }
    };

    out_slice
        .par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(edge, out_chunk)| per_edge(edge, out_chunk));
    out
}

/// A single patch pixel's precomputed bilinear-interpolation state: the
/// integer floor of its (sub-pixel) target coordinate, plus the four
/// corner weights derived from the fractional remainder. Shared across all
/// `(2·radius+1)²` taps of that patch pixel (see the module doc's "M4-perf
/// rewrite" point 2 for why this is valid — the fractional part does not
/// change as an *integer* tap offset is added to the centre coordinate).
struct BilinearWeights {
    ix0: isize,
    iy0: isize,
    w00: f32,
    w01: f32,
    w10: f32,
    w11: f32,
}

impl BilinearWeights {
    fn new(center_x: f32, center_y: f32) -> Self {
        let floor_x = center_x.floor();
        let floor_y = center_y.floor();
        let frac_x = center_x - floor_x;
        let frac_y = center_y - floor_y;
        BilinearWeights {
            ix0: floor_x as isize,
            iy0: floor_y as isize,
            w00: (1.0 - frac_x) * (1.0 - frac_y),
            w01: frac_x * (1.0 - frac_y),
            w10: (1.0 - frac_x) * frac_y,
            w11: frac_x * frac_y,
        }
    }

    /// Sample `fmap` at the 2×2 neighbourhood anchored at `(ix0, iy0)`
    /// (this tap's integer corner, `self.ix0 + dx`/`self.iy0 + dy` — see the
    /// call site), weight-blend it, and dot the blend against `anchor` in
    /// one fused pass: `anchor · (w00·c00 + w01·c01 + w10·c10 + w11·c11) =
    /// w00·(anchor·c00) + w01·(anchor·c01) + w10·(anchor·c10) +
    /// w11·(anchor·c11)` by linearity — algebraically the same value the
    /// pre-M4-perf code computed via an explicit intermediate
    /// weighted-sum vector, just without materializing it (see module doc).
    /// Zero-padded: any corner outside `[0, width) × [0, height)`
    /// contributes `0.0`, matching `grid_sample(..., padding_mode="zeros")`.
    #[inline]
    fn sample_dot(&self, fmap: &ChannelLastImage, ix0: isize, iy0: isize, anchor: &[f32]) -> f32 {
        let mut dot = 0.0_f32;
        if self.w00 != 0.0 {
            if let Some(c) = fmap.row(ix0, iy0) {
                dot += self.w00 * dot_slice(anchor, c);
            }
        }
        if self.w01 != 0.0 {
            if let Some(c) = fmap.row(ix0 + 1, iy0) {
                dot += self.w01 * dot_slice(anchor, c);
            }
        }
        if self.w10 != 0.0 {
            if let Some(c) = fmap.row(ix0, iy0 + 1) {
                dot += self.w10 * dot_slice(anchor, c);
            }
        }
        if self.w11 != 0.0 {
            if let Some(c) = fmap.row(ix0 + 1, iy0 + 1) {
                dot += self.w11 * dot_slice(anchor, c);
            }
        }
        dot
    }
}

/// Contiguous `f32` dot product over two equal-length channel slices —
/// written as a straight slice loop (not `ndarray` indexing) specifically so
/// the compiler can autovectorize it (module doc, point 1); verified via the
/// ignored perf test at the bottom of this file, `--release`.
#[inline]
fn dot_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// A `(channels, height, width)` feature map transposed once into
/// channel-last `(height, width, channels)` storage, so a per-pixel channel
/// vector is a contiguous slice. Zero-padded out-of-bounds reads return
/// `None` rather than panicking (matches [`corr_cpu`]'s documented
/// `grid_sample(..., padding_mode="zeros")` semantics).
///
/// `pub` (not just an internal detail of [`corr_cpu`]) specifically so a
/// caller that reuses the same target frame across many [`corr_cpu`] calls
/// can build this once and pass it to [`corr_cpu_prebuilt_target`] instead —
/// see that function's doc for the caching rationale
/// (`pipelines/slam/src/dpvo_vo.rs`'s `FramePyramid` is exactly this caller).
#[derive(Clone, Debug)]
pub struct ChannelLastImage {
    data: Vec<f32>,
    height: usize,
    width: usize,
    channels: usize,
}

impl ChannelLastImage {
    pub fn from_chw(fmap: ArrayView3<'_, f32>) -> Self {
        let (channels, height, width) = fmap.dim();
        let mut data = vec![0.0_f32; height * width * channels];
        for y in 0..height {
            for x in 0..width {
                let base = (y * width + x) * channels;
                for c in 0..channels {
                    data[base + c] = fmap[(c, y, x)];
                }
            }
        }
        ChannelLastImage {
            data,
            height,
            width,
            channels,
        }
    }

    /// Contiguous channel slice at `(x, y)`, or `None` if either coordinate
    /// falls outside `[0, width) × [0, height)`.
    #[inline]
    fn row(&self, x: isize, y: isize) -> Option<&[f32]> {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return None;
        }
        let base = (y as usize * self.width + x as usize) * self.channels;
        Some(&self.data[base..base + self.channels])
    }
}

/// The per-edge anchor patch features (`num_edges, channels, patch, patch`)
/// transposed once into channel-last `(num_edges, patch, patch, channels)`
/// storage, matching [`ChannelLastImage`]'s rationale.
struct ChannelLastPatches {
    data: Vec<f32>,
    patch: usize,
    channels: usize,
}

impl ChannelLastPatches {
    fn from_ecpp(patch_feats: ArrayView4<'_, f32>) -> Self {
        let (num_edges, channels, patch, _patch_check) = patch_feats.dim();
        let mut data = vec![0.0_f32; num_edges * patch * patch * channels];
        for edge in 0..num_edges {
            for py in 0..patch {
                for px in 0..patch {
                    let base = ((edge * patch + py) * patch + px) * channels;
                    for c in 0..channels {
                        data[base + c] = patch_feats[(edge, c, py, px)];
                    }
                }
            }
        }
        ChannelLastPatches {
            data,
            patch,
            channels,
        }
    }

    #[inline]
    fn row(&self, edge: usize, py: usize, px: usize) -> &[f32] {
        let base = ((edge * self.patch + py) * self.patch + px) * self.channels;
        &self.data[base..base + self.channels]
    }
}

/// Bilinear sample of `fmap` at continuous pixel coordinate `(x, y)`,
/// zero-padding any of the four corner reads that fall outside bounds
/// (`align_corners=True`, `padding_mode="zeros"` semantics — see
/// [`corr_cpu`]'s doc for why this differs from patchify's clamp-to-edge).
/// Writes the per-channel sampled vector into `out` (length `channels`).
///
/// Kept only for [`corr_cpu_reference`] (the `cfg(test)`-only slow
/// reference — see this module's "M4-perf rewrite" doc); [`corr_cpu`]
/// itself now uses [`BilinearWeights::sample_dot`] instead.
#[cfg(test)]
fn bilinear_sample_zero_pad(
    fmap: ArrayView3<'_, f32>,
    x: f32,
    y: f32,
    mut out: ndarray::ArrayViewMut1<'_, f32>,
) {
    let (channels, height, width) = fmap.dim();
    out.fill(0.0);

    let x0 = x.floor();
    let y0 = y.floor();
    let frac_x = x - x0;
    let frac_y = y - y0;
    let corners = [
        (x0, y0, (1.0 - frac_x) * (1.0 - frac_y)),
        (x0 + 1.0, y0, frac_x * (1.0 - frac_y)),
        (x0, y0 + 1.0, (1.0 - frac_x) * frac_y),
        (x0 + 1.0, y0 + 1.0, frac_x * frac_y),
    ];
    for (cx, cy, weight) in corners {
        if weight == 0.0 {
            continue;
        }
        if cx < 0.0 || cy < 0.0 {
            continue;
        }
        let (ix, iy) = (cx as usize, cy as usize);
        if ix >= width || iy >= height {
            continue;
        }
        for c in 0..channels {
            out[c] += weight * fmap[(c, iy, ix)];
        }
    }
}

/// Deliberately-naive reference implementation, kept byte-for-byte
/// equivalent to `corr_cpu` as it stood before the M4-perf rewrite (module
/// doc): channel-first indexing throughout, bilinear weights recomputed
/// per tap, a materialized weighted-neighbourhood vector before the dot
/// product. Used only by [`fast_matches_naive_reference_at_realistic_shape`]
/// below to guard against future accidental numerics drift in the fast
/// path; not itself exercised by the fixture parity test (which targets
/// `corr_cpu` directly, and therefore now exercises the fast path).
#[cfg(test)]
fn corr_cpu_reference(
    patch_feats: ArrayView4<'_, f32>,
    target_fmap: ArrayView3<'_, f32>,
    coords_center: ArrayView4<'_, f32>,
    radius: usize,
) -> Array4<f32> {
    use ndarray::Array1;

    let (num_edges, channels, patch, patch_check) = patch_feats.dim();
    debug_assert_eq!(
        patch, patch_check,
        "patch_feats must be square in its last two dims"
    );
    let taps = 2 * radius + 1;
    let scale = (channels as f32).sqrt();

    let mut out = Array4::<f32>::zeros((num_edges, patch, patch, taps * taps));
    let mut sampled = Array1::<f32>::zeros(channels);
    for edge in 0..num_edges {
        for py in 0..patch {
            for px in 0..patch {
                let center_x = coords_center[(edge, py, px, 0)];
                let center_y = coords_center[(edge, py, px, 1)];
                for ty in 0..taps {
                    let dy = ty as f32 - radius as f32;
                    for tx in 0..taps {
                        let dx = tx as f32 - radius as f32;
                        bilinear_sample_zero_pad(
                            target_fmap,
                            center_x + dx,
                            center_y + dy,
                            sampled.view_mut(),
                        );
                        let mut dot = 0.0_f32;
                        for c in 0..channels {
                            dot += patch_feats[(edge, c, py, px)] * sampled[c];
                        }
                        out[(edge, py, px, ty * taps + tx)] = dot / scale;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

    /// A 1-channel target map with `value(row, col) = col`, sampled at an
    /// exact integer coordinate with `radius=0` (a single tap, no
    /// neighbourhood): the correlation of a unit anchor feature against
    /// that single sampled value, divided by `sqrt(channels)=1`, should
    /// equal the sampled value itself exactly.
    #[test]
    fn single_tap_exact_integer_sample_matches_raw_value() {
        let mut fmap = Array3::<f32>::zeros((1, 4, 4));
        for row in 0..4 {
            for col in 0..4 {
                fmap[(0, row, col)] = col as f32;
            }
        }
        // One edge, one patch pixel, anchor feature = [1.0] (so the dot
        // product is just the sampled value), centred exactly on column 2.
        let patch_feats = Array4::<f32>::from_elem((1, 1, 1, 1), 1.0_f32);
        let coords_center = array![[[[2.0_f32, 1.0_f32]]]]; // (edge=1, py=1, px=1, xy=2)
        let out = corr_cpu(patch_feats.view(), fmap.view(), coords_center.view(), 0);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert!(
            (out[(0, 0, 0, 0)] - 2.0).abs() < 1e-6,
            "got {}",
            out[(0, 0, 0, 0)]
        );
    }

    /// Sampling exactly on the last valid column (no fractional part) must
    /// not zero-pad — only *out-of-range* corners should ever be dropped,
    /// not in-range corners whose neighbour happens to be out of range but
    /// carries zero weight. This guards the `if weight == 0.0 { continue }`
    /// short-circuit against accidentally masking a nonzero-weight corner.
    #[test]
    fn sampling_at_last_valid_pixel_is_not_spuriously_zeroed() {
        let mut fmap = Array3::<f32>::zeros((1, 1, 3));
        fmap[(0, 0, 0)] = 10.0;
        fmap[(0, 0, 1)] = 20.0;
        fmap[(0, 0, 2)] = 30.0;
        let patch_feats = Array4::<f32>::from_elem((1, 1, 1, 1), 1.0_f32);
        let coords_center = array![[[[2.0_f32, 0.0_f32]]]]; // exactly the last column
        let out = corr_cpu(patch_feats.view(), fmap.view(), coords_center.view(), 0);
        assert!(
            (out[(0, 0, 0, 0)] - 30.0).abs() < 1e-6,
            "got {}",
            out[(0, 0, 0, 0)]
        );
    }

    /// Sampling one pixel past the border should zero-pad (not clamp): with
    /// a unit anchor and a target column at `x = width` (one past the last
    /// valid column `width-1`), the in-range corner (`x0 = width-1`) gets
    /// weight `1 - frac_x` and the out-of-range corner (`x0+1 = width`)
    /// contributes zero instead of repeating the border value.
    #[test]
    fn sampling_past_border_zero_pads_the_out_of_range_corner() {
        let mut fmap = Array3::<f32>::zeros((1, 1, 2));
        fmap[(0, 0, 0)] = 5.0;
        fmap[(0, 0, 1)] = 7.0;
        let patch_feats = Array4::<f32>::from_elem((1, 1, 1, 1), 1.0_f32);
        // x = 1.5 -> corners at x0=1 (in range, weight 0.5) and x0+1=2 (out
        // of range, weight 0.5, zero-padded). Expected = 0.5*7.0 + 0.5*0 = 3.5.
        let coords_center = array![[[[1.5_f32, 0.0_f32]]]];
        let out = corr_cpu(patch_feats.view(), fmap.view(), coords_center.view(), 0);
        assert!(
            (out[(0, 0, 0, 0)] - 3.5).abs() < 1e-6,
            "got {}",
            out[(0, 0, 0, 0)]
        );
    }

    /// Normalization by `sqrt(channels)`: two channels, anchor = target =
    /// `[1, 1]`, dot product `2.0`, divided by `sqrt(2)`.
    #[test]
    fn normalizes_by_sqrt_of_channel_count() {
        let mut fmap = Array3::<f32>::zeros((2, 2, 2));
        fmap[(0, 0, 0)] = 1.0;
        fmap[(1, 0, 0)] = 1.0;
        let patch_feats = Array4::<f32>::from_elem((1, 2, 1, 1), 1.0_f32);
        let coords_center = array![[[[0.0_f32, 0.0_f32]]]];
        let out = corr_cpu(patch_feats.view(), fmap.view(), coords_center.view(), 0);
        let expected = 2.0_f32 / (2.0_f32).sqrt();
        assert!(
            (out[(0, 0, 0, 0)] - expected).abs() < 1e-6,
            "got {}",
            out[(0, 0, 0, 0)]
        );
    }

    /// M4-perf equivalence test (task requirement: "keep a slow-reference
    /// implementation ... and add an equivalence test at realistic shapes
    /// (rand inputs, max-abs ≤ 1e-5)"). `256` edges / `128` channels / `3×3`
    /// patch / `radius=3` (`49` taps/level) is the same per-primitive shape
    /// as DPVO's real working set (module doc), just a smaller edge count
    /// than the full few-thousand-edge budget so this stays a fast,
    /// always-on (non-ignored) test rather than needing `--ignored`; the
    /// separate perf test below re-runs at the full few-thousand-edge scale
    /// for timing (not correctness) purposes.
    #[test]
    fn fast_matches_naive_reference_at_realistic_shape() {
        let mut rng = DeterministicRng::new(0x5EED_u64);
        let num_edges = 256;
        let channels = 128;
        let patch = 3;
        let radius = 3;
        let (height, width) = (120, 188); // 480/4, 752/4 — EuRoC stride-4 shape

        let patch_feats =
            Array4::<f32>::from_shape_fn((num_edges, channels, patch, patch), |_| rng.next_f32());
        let target_fmap =
            Array3::<f32>::from_shape_fn((channels, height, width), |_| rng.next_f32());
        // Centres scattered around the valid image extent, including some
        // deliberately out-of-bounds ones (module doc's zero-padding path)
        // by letting the random offset exceed `[0, width) x [0, height)`.
        let coords_center =
            Array4::<f32>::from_shape_fn((num_edges, patch, patch, 2), |(_, _, _, axis)| {
                if axis == 0 {
                    rng.next_f32() * (width as f32 + 8.0) - 4.0
                } else {
                    rng.next_f32() * (height as f32 + 8.0) - 4.0
                }
            });

        let fast = corr_cpu(
            patch_feats.view(),
            target_fmap.view(),
            coords_center.view(),
            radius,
        );
        let reference = corr_cpu_reference(
            patch_feats.view(),
            target_fmap.view(),
            coords_center.view(),
            radius,
        );

        assert_eq!(fast.shape(), reference.shape());
        let max_abs_diff = fast
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff <= 1e-5,
            "fast/reference corr_cpu mismatch: max abs diff {max_abs_diff:.3e}"
        );
    }

    /// Tiny dependency-free PRNG (xorshift64*) so the equivalence/perf tests
    /// below don't need a new `rand` call surface beyond what's already a
    /// workspace dependency elsewhere — deterministic seeding keeps the test
    /// reproducible.
    struct DeterministicRng(u64);
    impl DeterministicRng {
        fn new(seed: u64) -> Self {
            DeterministicRng(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// M4-perf micro-benchmark, at DPVO's real per-frame working set (task's
    /// own framing: "few thousand edges x 3x3 patch grid x ... 7x7 lookup
    /// window"). `--ignored` because this is a timing report, not a
    /// pass/fail correctness check; run with:
    ///
    /// ```text
    /// cargo test -p visloc-vision --release --features onnx-inference \
    ///   --lib dpvo::correlation::tests::corr_cpu_perf_at_realistic_working_set \
    ///   -- --ignored --nocapture
    /// ```
    ///
    /// **Debug-build numbers are 10-100x slower and not representative**
    /// (M2's own perf-test caveat, repeated here) — only trust `--release`
    /// numbers.
    #[test]
    #[ignore = "timing report, not a correctness check; run --release, see doc comment"]
    fn corr_cpu_perf_at_realistic_working_set() {
        let mut rng = DeterministicRng::new(0xC0FFEE_u64);
        let num_edges = 3000;
        let channels = 128;
        let patch = 3;
        let radius = 3;
        let (height, width) = (120, 188);

        let patch_feats =
            Array4::<f32>::from_shape_fn((num_edges, channels, patch, patch), |_| rng.next_f32());
        let target_fmap =
            Array3::<f32>::from_shape_fn((channels, height, width), |_| rng.next_f32());
        let coords_center =
            Array4::<f32>::from_shape_fn((num_edges, patch, patch, 2), |(_, _, _, axis)| {
                if axis == 0 {
                    rng.next_f32() * width as f32
                } else {
                    rng.next_f32() * height as f32
                }
            });

        let repeats = 10;
        let start = std::time::Instant::now();
        for _ in 0..repeats {
            std::hint::black_box(corr_cpu(
                patch_feats.view(),
                target_fmap.view(),
                coords_center.view(),
                radius,
            ));
        }
        let fast_ms = start.elapsed().as_secs_f64() * 1000.0 / repeats as f64;
        println!("  [perf] corr_cpu (fast, {num_edges} edges): {fast_ms:.3} ms/call");

        // The naive reference is O(minutes) at this scale (it is what the
        // M4 EuRoC run measured as the ~3.1 s/frame bottleneck in the first
        // place) — one repeat only, purely to report the before/after ratio.
        let start = std::time::Instant::now();
        std::hint::black_box(corr_cpu_reference(
            patch_feats.view(),
            target_fmap.view(),
            coords_center.view(),
            radius,
        ));
        let naive_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!("  [perf] corr_cpu_reference (naive, {num_edges} edges): {naive_ms:.3} ms/call");
        println!("  [perf] speedup: {:.2}x", naive_ms / fast_ms);
    }
}
