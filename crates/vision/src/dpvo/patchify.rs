//! Native (Rust, no ONNX) patch extraction, ported from
//! `scripts/export_dpvo_onnx.py`'s `patchify_cpu` — itself the export
//! script's *own* reimplementation of upstream's CUDA-only
//! `altcorr.patchify` (the integer-window gather; the surrounding
//! bilinear-blend arithmetic in `altcorr/correlation.py`'s Python wrapper
//! *is* upstream's own code and is what got reproduced here).
//!
//! **Honesty caveat (carried from the plan doc's M1 results, repeated here
//! because it is the single most important thing to know before trusting
//! this file):** this is **not** verified against the real CUDA kernel — no
//! CUDA toolchain is available anywhere in this project's development
//! environment. Passing [`patchify_cpu`]'s fixture-based parity test
//! (`crates/vision/tests/dpvo_onnx_parity.rs`) demonstrates bit-for-bit
//! agreement with `export_dpvo_onnx.py`'s own from-scratch reference, not
//! with upstream. Border/edge-clamp handling in particular may differ from
//! the real kernel. Treat this as a regression fixture, not ground truth.

use ndarray::{Array4, ArrayView3};

/// Bilinearly extract a `(2·radius+1) × (2·radius+1)` patch of every
/// channel around each of `coords`' (possibly sub-pixel) centroids from a
/// single feature map.
///
/// * `fmap`: `(channels, height, width)` — a single frame's `fnet`/`inet`
///   output (the batch dimension, always `1` in this crate's usage, is the
///   caller's problem to squeeze out first — see
///   [`crate::dpvo::onnx_session::DpvoOnnxSession::run_fnet`]'s return
///   type).
/// * `coords`: `(x, y)` pixel-space centroid per patch, length `M`.
/// * `radius`: patch half-width; DPVO's own patch size is 3×3, i.e.
///   `radius = 1` (see [`super::PATCH`]).
///
/// Returns `(M, channels, 2·radius+1, 2·radius+1)`.
///
/// # Algorithm (verbatim translation of `patchify_cpu`)
///
/// For each centroid `(cx, cy)`:
/// 1. Gather a `(2·radius+2) × (2·radius+2)` integer-aligned window
///    starting at `(⌊cx⌋, ⌊cy⌋) - radius`, **clamping** out-of-range
///    indices to the feature map's border (this is the "may differ from
///    upstream's CUDA kernel" edge case flagged above — clamp-to-edge was
///    this reimplementation's choice, not something read out of the
///    kernel).
/// 2. Bilinearly blend that oversized window down to the requested
///    `(2·radius+1) × (2·radius+1)` size using the fractional part of the
///    centroid (`dx = cx - ⌊cx⌋`, `dy = cy - ⌊cy⌋`) as the four
///    corner-interpolation weights — i.e. every output pixel is a weighted
///    blend of its own integer-aligned sample and its right/down/diagonal
///    neighbours in the oversized window, exactly compensating for the
///    fact that `(⌊cx⌋, ⌊cy⌋)` is not the true (sub-pixel) centroid.
pub fn patchify_cpu(fmap: ArrayView3<'_, f32>, coords: &[(f32, f32)], radius: usize) -> Array4<f32> {
    let (channels, height, width) = fmap.dim();
    let num_patches = coords.len();
    let window = 2 * radius + 2;
    let out_size = 2 * radius + 1;

    // Step 1: integer-aligned oversized window, clamped to the feature
    // map's border.
    let mut raw = Array4::<f32>::zeros((num_patches, channels, window, window));
    for (patch_index, &(cx, cy)) in coords.iter().enumerate() {
        let floor_x = cx.floor();
        let floor_y = cy.floor();
        for wy in 0..window {
            let offset_y = wy as isize - radius as isize;
            let sample_y = clamp_isize(floor_y as isize + offset_y, height);
            for wx in 0..window {
                let offset_x = wx as isize - radius as isize;
                let sample_x = clamp_isize(floor_x as isize + offset_x, width);
                for c in 0..channels {
                    raw[(patch_index, c, wy, wx)] = fmap[(c, sample_y, sample_x)];
                }
            }
        }
    }

    // Step 2: bilinear blend down to `out_size × out_size` using the
    // sub-pixel offset within the centroid's integer cell.
    let mut out = Array4::<f32>::zeros((num_patches, channels, out_size, out_size));
    for (patch_index, &(cx, cy)) in coords.iter().enumerate() {
        let dx = cx - cx.floor();
        let dy = cy - cy.floor();
        let w00 = (1.0 - dy) * (1.0 - dx);
        let w01 = (1.0 - dy) * dx;
        let w10 = dy * (1.0 - dx);
        let w11 = dy * dx;
        for c in 0..channels {
            for oy in 0..out_size {
                for ox in 0..out_size {
                    let value = w00 * raw[(patch_index, c, oy, ox)]
                        + w01 * raw[(patch_index, c, oy, ox + 1)]
                        + w10 * raw[(patch_index, c, oy + 1, ox)]
                        + w11 * raw[(patch_index, c, oy + 1, ox + 1)];
                    out[(patch_index, c, oy, ox)] = value;
                }
            }
        }
    }
    out
}

/// Clamp a signed window offset into `[0, len - 1]`, matching
/// `.clamp(0, W - 1)` in the Python reference (`len` is `height` or
/// `width`; always ≥ 1 for any real feature map).
fn clamp_isize(value: isize, len: usize) -> usize {
    value.clamp(0, len as isize - 1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// A patch centred exactly on an integer pixel (no sub-pixel offset)
    /// should be a plain crop — the bilinear blend weights collapse to
    /// `(1, 0, 0, 0)` (`dx = dy = 0`), so this exercises the "no
    /// interpolation needed" path independent of the clamp logic.
    #[test]
    fn integer_centroid_is_a_plain_crop() {
        // 1 channel, 5x5 feature map with value = row*5+col so every pixel
        // is uniquely identifiable.
        let mut fmap = Array3::<f32>::zeros((1, 5, 5));
        for row in 0..5 {
            for col in 0..5 {
                fmap[(0, row, col)] = (row * 5 + col) as f32;
            }
        }
        let coords = vec![(2.0_f32, 2.0_f32)]; // centre pixel, away from any border
        let patches = patchify_cpu(fmap.view(), &coords, 1);
        assert_eq!(patches.shape(), &[1, 1, 3, 3]);
        // Expect the 3x3 crop centred at (row=2, col=2):
        let expected = [[6.0, 7.0, 8.0], [11.0, 12.0, 13.0], [16.0, 17.0, 18.0]];
        for oy in 0..3 {
            for ox in 0..3 {
                assert!(
                    (patches[(0, 0, oy, ox)] - expected[oy][ox]).abs() < 1e-6,
                    "mismatch at ({oy},{ox}): got {}, want {}",
                    patches[(0, 0, oy, ox)],
                    expected[oy][ox]
                );
            }
        }
    }

    /// A centroid offset by exactly 0.5px in both axes should produce the
    /// average of the 2x2 neighbourhood at every output pixel (equal
    /// blend weights of 0.25 each) — a hand-checkable bilinear case.
    #[test]
    fn half_pixel_centroid_averages_two_by_two_neighbourhood() {
        // Constant-gradient feature map along columns only, so the
        // expected averaged value is easy to compute by hand:
        // value(row, col) = col (as f32).
        let mut fmap = Array3::<f32>::zeros((1, 5, 5));
        for row in 0..5 {
            for col in 0..5 {
                fmap[(0, row, col)] = col as f32;
            }
        }
        let coords = vec![(2.5_f32, 2.5_f32)];
        let patches = patchify_cpu(fmap.view(), &coords, 1);
        // Every output row is identical (no row-wise variation in fmap).
        // With `radius=1`, the oversized raw window covers source columns
        // `[1, 2, 3, 4]` (`floor(2.5) - 1 ..= floor(2.5) + 2`); output
        // column `ox` averages raw columns `ox` and `ox+1` (weights 0.25
        // each summed over the two blend rows, i.e. an equal 0.5/0.5 mix
        // since `dx = dy = 0.5`), i.e. source columns `1+ox` and `2+ox`:
        // `expected = ((1+ox) + (2+ox)) / 2 = ox + 1.5`.
        for oy in 0..3 {
            for ox in 0..3 {
                let expected = ox as f32 + 1.5;
                assert!(
                    (patches[(0, 0, oy, ox)] - expected).abs() < 1e-5,
                    "mismatch at ({oy},{ox}): got {}, want {expected}",
                    patches[(0, 0, oy, ox)]
                );
            }
        }
    }

    /// A centroid near the border must clamp rather than panic or wrap;
    /// this checks the clamp actually repeats the border pixel rather
    /// than, say, silently reading garbage.
    #[test]
    fn near_border_centroid_clamps_instead_of_panicking() {
        let mut fmap = Array3::<f32>::zeros((1, 3, 3));
        for row in 0..3 {
            for col in 0..3 {
                fmap[(0, row, col)] = (row * 3 + col) as f32;
            }
        }
        let coords = vec![(0.0_f32, 0.0_f32)]; // top-left corner
        let patches = patchify_cpu(fmap.view(), &coords, 1);
        // Top-left output pixel samples (-1,-1) clamped to (0,0) on all
        // four bilinear corners (dx=dy=0 here, so it is a plain crop of
        // the clamped window) -> value at (0,0) = 0.0.
        assert!((patches[(0, 0, 0, 0)] - 0.0).abs() < 1e-6);
    }
}
