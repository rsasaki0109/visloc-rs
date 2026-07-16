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
//! see the plan doc's M2 blockers list.

use ndarray::{Array1, Array4, ArrayView3, ArrayView4};

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
pub fn corr_cpu(
    patch_feats: ArrayView4<'_, f32>,
    target_fmap: ArrayView3<'_, f32>,
    coords_center: ArrayView4<'_, f32>,
    radius: usize,
) -> Array4<f32> {
    let (num_edges, channels, patch, patch_check) = patch_feats.dim();
    debug_assert_eq!(patch, patch_check, "patch_feats must be square in its last two dims");
    let taps = 2 * radius + 1;
    let scale = (channels as f32).sqrt();

    let mut out = Array4::<f32>::zeros((num_edges, patch, patch, taps * taps));
    let mut sampled = Array1::<f32>::zeros(channels);
    for edge in 0..num_edges {
        for py in 0..patch {
            for px in 0..patch {
                let center_x = coords_center[(edge, py, px, 0)];
                let center_y = coords_center[(edge, py, px, 1)];
                // `ndarray::s!`-style slicing would need `unsafe_code`
                // internally (this crate is `#![forbid(unsafe_code)]`), so
                // the anchor/sample dot product below indexes `patch_feats`
                // directly per-channel instead of taking a `(channels,)`
                // sub-view.
                //
                // `ty` indexes the row (dy) axis, `tx` the column (dx)
                // axis; the flat output index `ty*taps+tx` matches the
                // reference's `(taps_dy, taps_dx)` → flatten order (see
                // this function's port-fidelity note in the module for the
                // dxdy-axis-order derivation from `torch.meshgrid`).
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

/// Bilinear sample of `fmap` at continuous pixel coordinate `(x, y)`,
/// zero-padding any of the four corner reads that fall outside bounds
/// (`align_corners=True`, `padding_mode="zeros"` semantics — see
/// [`corr_cpu`]'s doc for why this differs from patchify's clamp-to-edge).
/// Writes the per-channel sampled vector into `out` (length `channels`).
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
        assert!((out[(0, 0, 0, 0)] - 2.0).abs() < 1e-6, "got {}", out[(0, 0, 0, 0)]);
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
        assert!((out[(0, 0, 0, 0)] - 30.0).abs() < 1e-6, "got {}", out[(0, 0, 0, 0)]);
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
        assert!((out[(0, 0, 0, 0)] - 3.5).abs() < 1e-6, "got {}", out[(0, 0, 0, 0)]);
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
        assert!((out[(0, 0, 0, 0)] - expected).abs() < 1e-6, "got {}", out[(0, 0, 0, 0)]);
    }
}
