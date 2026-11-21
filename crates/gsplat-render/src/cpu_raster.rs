//! CPU (rayon) version of `rasterize_backward.wgsl`, for devices that are
//! themselves the CPU (Mesa lavapipe / llvmpipe).
//!
//! On lavapipe a workgroup runs on one CPU thread with 8-wide vectors, so the
//! shader's per-subgroup reductions and shared-memory CAS float adds (ten per
//! subgroup per blended splat) dominate a training step. Here each tile is a
//! plain loop on one rayon thread: it walks the tile's list back to front
//! exactly as the shader does, sums each splat's gradient over the tile's
//! pixels with ordinary float adds, and writes that sum to the splat's slot
//! in the tile-sorted intersection list. The slots are then added into the
//! per-gaussian records in list order, so the result does not depend on the
//! thread count or scheduling (the shader's global atomics do).
//!
//! Each splat only visits the pixels of its tile that can reach
//! `alpha >= 1/255` (the bounding box of the conic's level set
//! `sigma = ln(255 opacity)`, one pixel wider); the others give exactly zero
//! and leave the per-pixel state untouched, as in the shader.

use rayon::prelude::*;

use super::backward::SCREEN_GRAD_FLOATS;

const TILE: u32 = 16;
const NG: usize = SCREEN_GRAD_FLOATS;
/// Intersections whose gradients are held at once (bounds the scratch memory
/// to `MAX_BATCH * 40` bytes; tiles are processed in row batches below it).
const MAX_BATCH: usize = 1 << 21;

/// One frame's projected splats and tile lists, as the forward pass left them
/// on the device (the residuals and `d_image` only for the backward pass;
/// empty for [`raster_forward`]).
pub(crate) struct Frame<'a> {
    pub img_w: u32,
    pub img_h: u32,
    pub bg: [f32; 3],
    /// 9 floats per compact id: u, v, conic A B C, opacity, r g b.
    pub projected: &'a [f32],
    /// Compact id of every tile-sorted intersection.
    pub sorted: &'a [u32],
    /// Per tile: [start, end) into `sorted` (start = u32::MAX when empty).
    pub offsets: &'a [u32],
    pub final_t: &'a [f32],
    pub last_idx: &'a [u32],
    /// dL/dC per pixel (RGB).
    pub d_image: &'a [f32],
}

/// Screen-space gradient records ([`SCREEN_GRAD_FLOATS`] per compact id,
/// `nv` of them) of `L = sum_px dot(d_image[px], C[px])`.
pub(crate) fn raster_backward(f: &Frame<'_>, nv: usize) -> Vec<f32> {
    let tile_bw = f.img_w.div_ceil(TILE);
    let tile_bh = f.img_h.div_ceil(TILE);
    let mut out = vec![0.0f32; nv * NG];
    let range = |t: u32| -> Option<(usize, usize)> {
        let (s, e) = (f.offsets[2 * t as usize], f.offsets[2 * t as usize + 1]);
        (s != u32::MAX && e > s).then_some((s as usize, e as usize))
    };
    let mut scratch: Vec<[f32; NG]> = Vec::new();
    let mut row = 0;
    while row < tile_bh {
        // A batch of whole tile rows whose intersections fit MAX_BATCH (at
        // least one row).
        let first = (row * tile_bw..tile_bh * tile_bw).find_map(range);
        let Some((lo, _)) = first else { break };
        let mut row_end = row;
        let mut hi = lo;
        while row_end < tile_bh {
            let row_hi = (row_end * tile_bw..(row_end + 1) * tile_bw)
                .filter_map(range)
                .map(|r| r.1)
                .max()
                .unwrap_or(hi);
            if row_end > row && row_hi - lo > MAX_BATCH {
                break;
            }
            hi = hi.max(row_hi);
            row_end += 1;
        }
        scratch.clear();
        scratch.resize(hi - lo, [0.0; NG]);
        // One mutable slot range per non-empty tile.
        let mut jobs: Vec<(u32, usize, &mut [[f32; NG]])> = Vec::new();
        let mut rest: &mut [[f32; NG]] = &mut scratch;
        let mut at = lo;
        for t in row * tile_bw..row_end * tile_bw {
            if let Some((s, e)) = range(t) {
                let (_, tail) = std::mem::take(&mut rest).split_at_mut(s - at);
                let (mine, tail) = tail.split_at_mut(e - s);
                jobs.push((t, s, mine));
                rest = tail;
                at = e;
            }
        }
        jobs.par_iter_mut()
            .for_each(|(t, s, slots)| tile_backward(f, tile_bw, *t, *s, slots));
        // Fixed-order reduction into the per-gaussian records.
        for (k, g) in scratch.iter().enumerate() {
            let c = f.sorted[lo + k] as usize;
            if c < nv {
                let dst = &mut out[c * NG..(c + 1) * NG];
                for (d, v) in dst.iter_mut().zip(g) {
                    *d += v;
                }
            }
        }
        row = row_end;
    }
    out
}

/// `exp(x)` (Cephes `expf`: range reduction by ln 2 and a degree-6
/// polynomial, ~1 ulp, `x` clamped to [-87, 88]), written without calls or
/// branches so the 16-lane loops below vectorize.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let p = 1.987_569_1e-4;
    let p = p * r + 1.398_199_9e-3;
    let p = p * r + 8.333_452e-3;
    let p = p * r + 4.166_579_6e-2;
    let p = p * r + 1.666_666_5e-1;
    let p = p * r + 0.5;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((n as i32).wrapping_add(127) as u32) << 23)
}

/// The backward walk of one tile; `slots[j - start]` receives the tile's sum
/// for list entry `j`.
///
/// Per splat, every tile row in its footprint is one fixed 16-lane loop
/// (lane = pixel column) with masks instead of branches. Pixels outside the
/// image have `last = 0`, so no splat passes for them.
fn tile_backward(f: &Frame<'_>, tile_bw: u32, tile: u32, start: usize, slots: &mut [[f32; NG]]) {
    const W: usize = TILE as usize;
    let end = start + slots.len();
    let x0 = (tile % tile_bw) * TILE;
    let y0 = (tile / tile_bw) * TILE;
    let w = ((x0 + TILE).min(f.img_w) - x0) as usize;
    let h = ((y0 + TILE).min(f.img_h) - y0) as usize;
    // Per pixel, [row][lane].
    let mut t_cur = [[0.0f32; W]; W];
    let mut last = [[0u32; W]; W];
    let mut gr = [[[0.0f32; W]; W]; 3];
    let mut be = [[[0.0f32; W]; W]; 3];
    let mut tile_last = 0u32;
    for ly in 0..h {
        for lx in 0..w {
            let pix = (y0 as usize + ly) * f.img_w as usize + x0 as usize + lx;
            t_cur[ly][lx] = f.final_t[pix];
            last[ly][lx] = f.last_idx[pix];
            for (c, g) in gr.iter_mut().enumerate() {
                g[ly][lx] = f.d_image[pix * 3 + c];
            }
            tile_last = tile_last.max(last[ly][lx]);
        }
        for (b, bg) in be.iter_mut().zip(f.bg) {
            b[ly] = [bg; W];
        }
    }
    let mut xs = [0.0f32; W];
    for (lx, x) in xs.iter_mut().enumerate() {
        *x = (x0 as usize + lx) as f32 + 0.5;
    }
    let stop = (tile_last as usize).min(end);
    let (img_w, img_h) = (f.img_w as f32, f.img_h as f32);
    let mut j = stop;
    while j > start {
        j -= 1;
        let c = f.sorted[j] as usize;
        let p = &f.projected[c * 9..c * 9 + 9];
        let (mu_x, mu_y, c00, c01, c11, opac) = (p[0], p[1], p[2], p[3], p[4], p[5]);
        let col = [p[6].max(0.0), p[7].max(0.0), p[8].max(0.0)];
        if opac.is_nan() || opac < 1.0 / 255.0 {
            // alpha = min(0.99, opac * exp(-sigma)) < 1/255 for sigma >= 0.
            continue;
        }
        // Rows that can pass the alpha test (the conic's level set
        // sigma = ln(255 opacity), one pixel wider).
        let (mut ly0, mut ly1) = (0i64, h as i64 - 1);
        let det = c00 * c11 - c01 * c01;
        if c00 > 0.0 && c11 > 0.0 && det > 0.0 {
            let s2 = 2.0 * (255.0 * opac).ln().max(0.0);
            let hy = (s2 * c00 / det).sqrt() + 1.0;
            ly0 = ly0.max((mu_y - hy - 0.5 - y0 as f32).floor() as i64);
            ly1 = ly1.min((mu_y + hy - 0.5 - y0 as f32).ceil() as i64);
        }
        if ly0 > ly1 {
            continue;
        }
        let ju = j as u32;
        // Per-lane partial sums, reduced once per splat.
        let mut acc = [[0.0f32; W]; NG];
        let mut any = false;
        for ly in ly0 as usize..=ly1 as usize {
            let dy = (y0 as usize + ly) as f32 + 0.5 - mu_y;
            let mut row_any = 0u32;
            for k in 0..W {
                let dx = xs[k] - mu_x;
                let sigma = 0.5 * (c00 * dx * dx + c11 * dy * dy) + c01 * dx * dy;
                let e = fast_exp(-sigma);
                let raw = opac * e;
                let a = raw.min(0.99);
                // (`sigma >= 0` is also false for NaN, as in the shader.)
                let hit = ju < last[ly][k] && sigma >= 0.0 && a >= 1.0 / 255.0;
                row_any |= hit as u32;
                let a = if hit { a } else { 0.0 };
                let t_before = t_cur[ly][k] / (1.0 - a);
                let at = a * t_before;
                let g = [gr[0][ly][k], gr[1][ly][k], gr[2][ly][k]];
                let b = [be[0][ly][k], be[1][ly][k], be[2][ly][k]];
                acc[6][k] += at * g[0];
                acc[7][k] += at * g[1];
                acc[8][k] += at * g[2];
                let d_alpha = t_before
                    * ((col[0] - b[0]) * g[0] + (col[1] - b[1]) * g[1] + (col[2] - b[2]) * g[2]);
                for ch in 0..3 {
                    be[ch][ly][k] = a * col[ch] + (1.0 - a) * b[ch];
                }
                t_cur[ly][k] = t_before;
                // A clamped alpha passes no gradient to opacity, conic, mean.
                let live = hit && raw <= 0.99;
                let d_sigma = if live { -d_alpha * a } else { 0.0 };
                let g_u = d_sigma * -(c00 * dx + c01 * dy);
                let g_v = d_sigma * -(c01 * dx + c11 * dy);
                acc[0][k] += g_u;
                acc[1][k] += g_v;
                acc[2][k] += d_sigma * 0.5 * dx * dx;
                acc[3][k] += d_sigma * dx * dy;
                acc[4][k] += d_sigma * 0.5 * dy * dy;
                acc[5][k] += if live { d_alpha * e } else { 0.0 };
                acc[9][k] += g_u.abs() * img_w + g_v.abs() * img_h;
            }
            any |= row_any != 0;
        }
        if any {
            let slot = &mut slots[j - start];
            for (s, lanes) in slot.iter_mut().zip(&acc) {
                *s = lanes.iter().sum();
            }
        }
    }
}

/// What `rasterize.wgsl` writes for a frame: the image (RGB, row-major), and
/// per pixel the final transmittance and one past the list index of the
/// last blended splat (0 = none).
pub(crate) struct Raster {
    pub image: Vec<f32>,
    pub final_t: Vec<f32>,
    pub last_idx: Vec<u32>,
}

/// Host version of `rasterize.wgsl`: front-to-back alpha compositing of each
/// tile's sorted list, a pixel stopping once its transmittance is at most
/// 1e-4. Tile rows run in parallel; within a splat, every tile row in its
/// footprint is one 16-lane loop, as in [`tile_backward`].
pub(crate) fn raster_forward(f: &Frame<'_>) -> Raster {
    const W: usize = TILE as usize;
    let (iw, ih) = (f.img_w as usize, f.img_h as usize);
    let tile_bw = f.img_w.div_ceil(TILE);
    let mut out = Raster {
        image: vec![0.0; iw * ih * 3],
        final_t: vec![0.0; iw * ih],
        last_idx: vec![0; iw * ih],
    };
    out.image
        .par_chunks_mut(W * iw * 3)
        .zip(out.final_t.par_chunks_mut(W * iw))
        .zip(out.last_idx.par_chunks_mut(W * iw))
        .enumerate()
        .for_each(|(ty, ((img, ft), li))| {
            let y0 = ty * W;
            let h = (ih - y0).min(W);
            for tx in 0..tile_bw as usize {
                let x0 = tx * W;
                let w = (iw - x0).min(W);
                let tile = ty * tile_bw as usize + tx;
                let mut acc = [[[0.0f32; W]; W]; 3];
                let mut trans = [[1.0f32; W]; W];
                let mut last = [[0u32; W]; W];
                // Pixels outside the image start finished.
                for row in trans.iter_mut().take(h) {
                    for t in row.iter_mut().skip(w) {
                        *t = 0.0;
                    }
                }
                for row in trans.iter_mut().skip(h) {
                    *row = [0.0; W];
                }
                let (s, e) = (f.offsets[2 * tile], f.offsets[2 * tile + 1]);
                if s != u32::MAX && e > s {
                    let mut xs = [0.0f32; W];
                    for (lx, x) in xs.iter_mut().enumerate() {
                        *x = (x0 + lx) as f32 + 0.5;
                    }
                    // Rows not yet saturated, as a bit mask.
                    let mut live_rows: u32 = (1u32 << h) - 1;
                    for j in s as usize..e as usize {
                        if live_rows == 0 {
                            break;
                        }
                        let c = f.sorted[j] as usize;
                        let p = &f.projected[c * 9..c * 9 + 9];
                        let (mu_x, mu_y, c00, c01, c11, opac) =
                            (p[0], p[1], p[2], p[3], p[4], p[5]);
                        if opac.is_nan() || opac < 1.0 / 255.0 {
                            continue;
                        }
                        let col = [p[6].max(0.0), p[7].max(0.0), p[8].max(0.0)];
                        let (mut ly0, mut ly1) = (0i64, h as i64 - 1);
                        let det = c00 * c11 - c01 * c01;
                        if c00 > 0.0 && c11 > 0.0 && det > 0.0 {
                            let s2 = 2.0 * (255.0 * opac).ln().max(0.0);
                            let hy = (s2 * c00 / det).sqrt() + 1.0;
                            ly0 = ly0.max((mu_y - hy - 0.5 - y0 as f32).floor() as i64);
                            ly1 = ly1.min((mu_y + hy - 0.5 - y0 as f32).ceil() as i64);
                        }
                        if ly0 > ly1 {
                            continue;
                        }
                        let next = j as u32 + 1;
                        for ly in ly0 as usize..=ly1 as usize {
                            if live_rows & (1 << ly) == 0 {
                                continue;
                            }
                            let dy = (y0 + ly) as f32 + 0.5 - mu_y;
                            let mut alive = 0u32;
                            for k in 0..W {
                                let dx = xs[k] - mu_x;
                                let sigma = 0.5 * (c00 * dx * dx + c11 * dy * dy) + c01 * dx * dy;
                                let a = (opac * fast_exp(-sigma)).min(0.99);
                                let t = trans[ly][k];
                                let hit = t > 1e-4 && sigma >= 0.0 && a >= 1.0 / 255.0;
                                let a = if hit { a } else { 0.0 };
                                let ta = t * a;
                                for ch in 0..3 {
                                    acc[ch][ly][k] += ta * col[ch];
                                }
                                let t = t * (1.0 - a);
                                trans[ly][k] = t;
                                last[ly][k] = if hit { next } else { last[ly][k] };
                                alive |= (t > 1e-4) as u32;
                            }
                            if alive == 0 {
                                live_rows &= !(1 << ly);
                            }
                        }
                    }
                }
                for ly in 0..h {
                    for lx in 0..w {
                        let o = ly * iw + x0 + lx;
                        let t = trans[ly][lx];
                        for ch in 0..3 {
                            img[o * 3 + ch] = acc[ch][ly][lx] + t * f.bg[ch];
                        }
                        ft[o] = t;
                        li[o] = last[ly][lx];
                    }
                }
            }
        });
    out
}

/// Host copies of a frame rasterized by [`raster_forward`], kept for the
/// backward pass so it reads back only dL/dC.
pub(crate) struct HostFrame {
    /// The rendered image (RGB, row-major), as uploaded.
    pub image: Vec<f32>,
    pub projected: Vec<f32>,
    pub sorted: Vec<u32>,
    pub offsets: Vec<u32>,
    pub final_t: Vec<f32>,
    pub last_idx: Vec<u32>,
}

/// Sort `(tile, id)` intersections by tile, stably (a counting sort), and
/// return the sorted tiles, the sorted ids and per tile `[start, end)` into
/// them (`(u32::MAX, 0)` for an empty tile), as the device's radix sort and
/// `tile_offsets.wgsl` produce them. Tiles past `num_tiles` sort last and
/// get no range.
pub(crate) fn sort_by_tile(
    tiles: &[u32],
    ids: &[u32],
    num_tiles: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let nt = num_tiles as usize;
    let mut next = vec![0usize; nt + 1];
    for &t in tiles {
        next[(t as usize).min(nt)] += 1;
    }
    let mut offsets = vec![0u32; nt * 2];
    let mut at = 0usize;
    for (t, slot) in next.iter_mut().enumerate() {
        let count = *slot;
        if t < nt {
            let (s, e) = if count > 0 {
                (at as u32, (at + count) as u32)
            } else {
                (u32::MAX, 0)
            };
            offsets[2 * t] = s;
            offsets[2 * t + 1] = e;
        }
        *slot = at;
        at += count;
    }
    let mut sorted_tiles = vec![0u32; tiles.len()];
    let mut sorted = vec![0u32; tiles.len()];
    for (&t, &id) in tiles.iter().zip(ids) {
        let b = &mut next[(t as usize).min(nt)];
        sorted_tiles[*b] = t;
        sorted[*b] = id;
        *b += 1;
    }
    (sorted_tiles, sorted, offsets)
}
