//! Tile binning, the forward rasterizer and its backward pass on the host.
//!
//! The forward walk is `visloc_gsplat_render::cpu_raster::raster_forward`'s
//! (16x16 tiles, rows of two fixed 8-lane halves that vectorize, a
//! polynomial `exp`, only the half rows that can reach `alpha >= 1/255`).
//!
//! The backward walk is that module's `tile_backward`, extended to also sum,
//! per splat, the Gauss-Newton matrix of the image with respect to its six
//! screen quantities `s = (u, v, A, B, C, opacity)`:
//!
//! - `dC/ds = dC/dalpha (dalpha/ds)^T`, with `dC/dalpha = T (c - B)` (`B`
//!   the colour composited behind the splat, background included), so per
//!   pixel `J^T J = |dC/dalpha|^2 a a^T` with `a = dalpha/ds`;
//! - for the splat's colour, `dC/dc = alpha T I`, so `J^T J = (alpha T)^2 I`.
//!
//! The other splats and the transmittance in front are held fixed, as in
//! 3DGS² (Lan et al. 2025).

use rayon::prelude::*;

use crate::project::Proj;

pub const TILE: u32 = 16;
/// Floats per splat the backward pass returns: screen gradient (6), colour
/// gradient (3), upper triangle of the screen Gauss-Newton matrix (21),
/// colour Gauss-Newton weight (1), and one unused (Inria's densification
/// statistic is the norm of the screen-mean gradient itself).
pub const NB: usize = 32;
pub const B_GRAD: usize = 0;
pub const B_COL: usize = 6;
pub const B_H: usize = 9;
pub const B_HC: usize = 30;
pub const B_STAT: usize = 31;

/// Index of `(i, j)`, `i <= j`, in a row-major upper triangle of a 6x6.
pub const fn tri6(i: usize, j: usize) -> usize {
    i * 6 - i * (i.wrapping_sub(1)) / 2 + (j - i)
}

/// One view's projected splats and tile lists.
pub struct Frame {
    pub img_w: u32,
    pub img_h: u32,
    pub bg: [f32; 3],
    /// 9 floats per compact id (the index into the depth-sorted `Proj`s):
    /// `u v A B C opacity r g b`.
    pub projected: Vec<f32>,
    /// Compact id of every tile-sorted intersection.
    pub sorted: Vec<u32>,
    /// Per tile: `[start, end)` into `sorted` (`start = u32::MAX` when empty).
    pub offsets: Vec<u32>,
}

impl Frame {
    /// Bin the depth-sorted splats into tiles (stable: each tile's list is
    /// front to back).
    pub fn new(proj: &[Proj], img_w: u32, img_h: u32, bg: [f32; 3]) -> Self {
        let tbw = img_w.div_ceil(TILE) as i64;
        let tbh = img_h.div_ceil(TILE) as i64;
        let nt = (tbw * tbh) as usize;
        let rect = |p: &Proj| -> Option<(i64, i64, i64, i64)> {
            let (u, v) = (p.s[0], p.s[1]);
            let x0 = (((u - p.ext[0]) / TILE as f32).floor() as i64).max(0);
            let x1 = (((u + p.ext[0]) / TILE as f32).floor() as i64).min(tbw - 1);
            let y0 = (((v - p.ext[1]) / TILE as f32).floor() as i64).max(0);
            let y1 = (((v + p.ext[1]) / TILE as f32).floor() as i64).min(tbh - 1);
            (x0 <= x1 && y0 <= y1).then_some((x0, x1, y0, y1))
        };
        let counts: Vec<usize> = proj
            .par_iter()
            .map(|p| {
                rect(p).map_or(0, |(x0, x1, y0, y1)| {
                    ((x1 - x0 + 1) * (y1 - y0 + 1)) as usize
                })
            })
            .collect();
        // Each splat's (tile) entries at its own offset, in parallel, then a
        // stable counting sort by tile (lists stay front to back).
        let mut starts = Vec::with_capacity(proj.len() + 1);
        let mut total = 0usize;
        for &c in &counts {
            starts.push(total);
            total += c;
        }
        let mut pairs_tile = vec![0u32; total];
        {
            let mut rest: &mut [u32] = &mut pairs_tile;
            let mut chunks: Vec<&mut [u32]> = Vec::with_capacity(proj.len());
            for &c in &counts {
                let (a, b) = std::mem::take(&mut rest).split_at_mut(c);
                chunks.push(a);
                rest = b;
            }
            chunks
                .into_par_iter()
                .zip(proj.par_iter())
                .for_each(|(dst, p)| {
                    if let Some((x0, x1, y0, y1)) = rect(p) {
                        let mut k = 0;
                        for ty in y0..=y1 {
                            for tx in x0..=x1 {
                                dst[k] = (ty * tbw + tx) as u32;
                                k += 1;
                            }
                        }
                    }
                });
        }
        let mut tile_count = vec![0usize; nt];
        for &t in &pairs_tile {
            tile_count[t as usize] += 1;
        }
        let mut offsets = vec![0u32; 2 * nt];
        let mut next = vec![0usize; nt];
        let mut at = 0usize;
        for t in 0..nt {
            let n = tile_count[t];
            if n > 0 {
                offsets[2 * t] = at as u32;
                offsets[2 * t + 1] = (at + n) as u32;
            } else {
                offsets[2 * t] = u32::MAX;
                offsets[2 * t + 1] = 0;
            }
            next[t] = at;
            at += n;
        }
        let mut sorted = vec![0u32; at];
        for (c, w) in starts.windows(2).enumerate() {
            for &t in &pairs_tile[w[0]..w[1]] {
                let b = &mut next[t as usize];
                sorted[*b] = c as u32;
                *b += 1;
            }
        }
        if let Some(&last) = starts.last() {
            let c = starts.len() - 1;
            for &t in &pairs_tile[last..] {
                let b = &mut next[t as usize];
                sorted[*b] = c as u32;
                *b += 1;
            }
        }
        let mut projected = vec![0.0f32; proj.len() * 9];
        projected
            .par_chunks_mut(9)
            .zip(proj.par_iter())
            .for_each(|(dst, p)| dst.copy_from_slice(&p.s));
        Self {
            img_w,
            img_h,
            bg,
            projected,
            sorted,
            offsets,
        }
    }

    pub fn num_splats(&self) -> usize {
        self.projected.len() / 9
    }

    fn range(&self, t: u32) -> Option<(usize, usize)> {
        let (s, e) = (
            self.offsets[2 * t as usize],
            self.offsets[2 * t as usize + 1],
        );
        (s != u32::MAX && e > s).then_some((s as usize, e as usize))
    }
}

/// `exp(x)` (Cephes `expf`), branch-free so the lane loops vectorize.
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
    const MAGIC: f32 = 12_582_912.0;
    let ni = (n + MAGIC).to_bits().wrapping_sub(MAGIC.to_bits());
    y * f32::from_bits(ni.wrapping_add(127) << 23)
}

/// The 8-lane halves of a tile row a splat can reach (see `cpu_raster`).
#[inline(always)]
fn row_lanes(
    level: Option<(f32, f32, f32, f32)>,
    dy: f32,
    mu_x: f32,
    x0: f32,
) -> Option<(usize, usize)> {
    let Some((c00, c01, c11, s2)) = level else {
        return Some((0, 16));
    };
    let disc = (c01 * dy) * (c01 * dy) - c00 * (c11 * dy * dy - s2);
    let half = if disc > 0.0 { disc.sqrt() / c00 } else { 0.0 };
    let xc = mu_x - c01 * dy / c00 - x0 - 0.5;
    let (lo, hi) = (xc - half - 1.0, xc + half + 1.0);
    let a = lo <= 7.0 && hi >= 0.0;
    let b = lo <= 15.0 && hi >= 8.0;
    match (a, b) {
        (true, true) => Some((0, 16)),
        (true, false) => Some((0, 8)),
        (false, true) => Some((8, 16)),
        (false, false) => None,
    }
}

/// A conic and its `alpha = 1/255` level: `(c00, c01, c11, s2)`.
type Level = (f32, f32, f32, f32);

/// Rows of a tile a splat can reach and its conic level, or `None`.
#[inline(always)]
fn splat_rows(p: &[f32], y0: u32, h: usize) -> Option<(usize, usize, Option<Level>)> {
    let (mu_y, c00, c01, c11, opac) = (p[1], p[2], p[3], p[4], p[5]);
    if opac.is_nan() || opac < 1.0 / 255.0 {
        return None;
    }
    let (mut ly0, mut ly1) = (0i64, h as i64 - 1);
    let det = c00 * c11 - c01 * c01;
    let mut level = None;
    if c00 > 0.0 && c11 > 0.0 && det > 0.0 {
        let s2 = 2.0 * (255.0 * opac).ln().max(0.0);
        let hy = (s2 * c00 / det).sqrt() + 1.0;
        ly0 = ly0.max((mu_y - hy - 0.5 - y0 as f32).floor() as i64);
        ly1 = ly1.min((mu_y + hy - 0.5 - y0 as f32).ceil() as i64);
        level = Some((c00, c01, c11, s2));
    }
    (ly0 <= ly1).then_some((ly0 as usize, ly1 as usize, level))
}

/// The forward pass's image (RGB, row-major) and per-pixel residuals.
pub struct Raster {
    pub image: Vec<f32>,
    pub final_t: Vec<f32>,
    pub last_idx: Vec<u32>,
}

/// Front-to-back alpha compositing of each tile's list.
pub fn forward(f: &Frame) -> Raster {
    const W: usize = TILE as usize;
    let (iw, ih) = (f.img_w as usize, f.img_h as usize);
    let tile_bw = f.img_w.div_ceil(TILE) as usize;
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
            for tx in 0..tile_bw {
                let x0 = tx * W;
                let w = (iw - x0).min(W);
                let tile = (ty * tile_bw + tx) as u32;
                let mut acc = [[[0.0f32; W]; W]; 3];
                let mut trans = [[1.0f32; W]; W];
                let mut last = [[0u32; W]; W];
                for row in trans.iter_mut().take(h) {
                    for t in row.iter_mut().skip(w) {
                        *t = 0.0;
                    }
                }
                for row in trans.iter_mut().skip(h) {
                    *row = [0.0; W];
                }
                if let Some((s, e)) = f.range(tile) {
                    let mut xs = [0.0f32; W];
                    for (lx, x) in xs.iter_mut().enumerate() {
                        *x = (x0 + lx) as f32 + 0.5;
                    }
                    let mut live_rows: u32 = (1u32 << h) - 1;
                    for j in s..e {
                        if live_rows == 0 {
                            break;
                        }
                        let c = f.sorted[j] as usize;
                        let p = &f.projected[c * 9..c * 9 + 9];
                        let Some((ly0, ly1, level)) = splat_rows(p, y0 as u32, h) else {
                            continue;
                        };
                        let (mu_x, mu_y, c00, c01, c11, opac) =
                            (p[0], p[1], p[2], p[3], p[4], p[5]);
                        let col = [p[6], p[7], p[8]];
                        let next = j as u32 + 1;
                        for ly in ly0..=ly1 {
                            if live_rows & (1 << ly) == 0 {
                                continue;
                            }
                            let dy = (y0 + ly) as f32 + 0.5 - mu_y;
                            let Some((k_lo, k_hi)) = row_lanes(level, dy, mu_x, x0 as f32) else {
                                continue;
                            };
                            for base in [0usize, 8] {
                                if base < k_lo || base >= k_hi {
                                    continue;
                                }
                                for kk in 0..8 {
                                    let k = base + kk;
                                    let dx = xs[k] - mu_x;
                                    let sigma =
                                        0.5 * (c00 * dx * dx + c11 * dy * dy) + c01 * dx * dy;
                                    let a = (opac * fast_exp(-sigma)).min(0.99);
                                    let t = trans[ly][k];
                                    let hit = t > 1e-4 && sigma >= 0.0 && a >= 1.0 / 255.0;
                                    let a = if hit { a } else { 0.0 };
                                    let ta = t * a;
                                    for ch in 0..3 {
                                        acc[ch][ly][k] += ta * col[ch];
                                    }
                                    trans[ly][k] = t * (1.0 - a);
                                    last[ly][k] = if hit { next } else { last[ly][k] };
                                }
                            }
                            if trans[ly].iter().all(|&t| t <= 1e-4) {
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

/// Intersections whose records are held at once.
const MAX_BATCH: usize = 1 << 20;

/// Per splat (compact id), the [`NB`] floats described at the top: the
/// gradient of `L = sum_px dot(d_image[px], C[px])`, and with
/// `gauss_newton`, the Gauss-Newton sums of `sum_px h_px |dC/d.|^2` with a
/// uniform per-pixel weight `h_px` (`2 / (3 N)` for a mean squared error);
/// without it those entries stay zero and the walk is cheaper.
///
/// `overlap` (0..=1) scales each pixel's Gauss-Newton term of a splat by
/// `(W / (alpha T))^overlap`, `W = 1 - T_final` the pixel's total weight:
/// at 1 the per-splat matrices majorize the joint one (Cauchy-Schwarz with
/// weights `alpha T`), so all splats can take their own steps at once and
/// the quadratic model still descends; at 0 it is plain Gauss-Newton, whose
/// simultaneous steps overshoot by about the number of splats that overlap.
pub fn backward(
    f: &Frame,
    r: &Raster,
    d_image: &[f32],
    h_px: f32,
    gauss_newton: bool,
    overlap: f32,
) -> Vec<[f32; NB]> {
    if gauss_newton {
        backward_impl::<true, NB>(f, r, d_image, h_px, overlap)
    } else {
        backward_impl::<false, 9>(f, r, d_image, h_px, overlap)
    }
}

/// [`backward`] with `NA` floats per intersection slot (see
/// [`tile_backward`]), so the gradient-only walk moves a third of the bytes.
fn backward_impl<const GN: bool, const NA: usize>(
    f: &Frame,
    r: &Raster,
    d_image: &[f32],
    h_px: f32,
    overlap: f32,
) -> Vec<[f32; NB]> {
    let nv = f.num_splats();
    let tile_bw = f.img_w.div_ceil(TILE);
    let tile_bh = f.img_h.div_ceil(TILE);
    let mut out = vec![[0.0f32; NB]; nv];
    let mut scratch: Vec<[f32; NA]> = Vec::new();
    let mut row = 0;
    while row < tile_bh {
        let first = (row * tile_bw..tile_bh * tile_bw).find_map(|t| f.range(t));
        let Some((lo, _)) = first else { break };
        let mut row_end = row;
        let mut hi = lo;
        while row_end < tile_bh {
            let row_hi = (row_end * tile_bw..(row_end + 1) * tile_bw)
                .filter_map(|t| f.range(t))
                .map(|x| x.1)
                .max()
                .unwrap_or(hi);
            if row_end > row && row_hi - lo > MAX_BATCH {
                break;
            }
            hi = hi.max(row_hi);
            row_end += 1;
        }
        scratch.clear();
        scratch.resize(hi - lo, [0.0; NA]);
        let mut jobs: Vec<(u32, usize, &mut [[f32; NA]])> = Vec::new();
        let mut rest: &mut [[f32; NA]] = &mut scratch;
        let mut at = lo;
        for t in row * tile_bw..row_end * tile_bw {
            if let Some((s, e)) = f.range(t) {
                let (_, tail) = std::mem::take(&mut rest).split_at_mut(s - at);
                let (mine, tail) = tail.split_at_mut(e - s);
                jobs.push((t, s, mine));
                rest = tail;
                at = e;
            }
        }
        jobs.par_iter_mut().for_each(|(t, s, slots)| {
            tile_backward::<GN, NA>(f, r, d_image, h_px, overlap, tile_bw, *t, *s, slots)
        });
        for (k, g) in scratch.iter().enumerate() {
            let c = f.sorted[lo + k] as usize;
            // The first NA of the NB floats (all of them with GN).
            for (d, v) in out[c].iter_mut().zip(g) {
                *d += v;
            }
        }
        row = row_end;
    }
    out
}

#[allow(clippy::too_many_arguments)]
/// `NA` accumulators per splat and slot: all [`NB`] with `GN`, else the
/// gradient and colour gradient (9).
fn tile_backward<const GN: bool, const NA: usize>(
    f: &Frame,
    r: &Raster,
    d_image: &[f32],
    h_px: f32,
    overlap: f32,
    tile_bw: u32,
    tile: u32,
    start: usize,
    slots: &mut [[f32; NA]],
) {
    const W: usize = TILE as usize;
    let end = start + slots.len();
    let x0 = (tile % tile_bw) * TILE;
    let y0 = (tile / tile_bw) * TILE;
    let w = ((x0 + TILE).min(f.img_w) - x0) as usize;
    let h = ((y0 + TILE).min(f.img_h) - y0) as usize;
    let mut t_cur = [[0.0f32; W]; W];
    let mut w_tot = [[0.0f32; W]; W];
    let mut last = [[0u32; W]; W];
    let mut gr = [[[0.0f32; W]; W]; 3];
    let mut be = [[[0.0f32; W]; W]; 3];
    let mut tile_last = 0u32;
    for ly in 0..h {
        for lx in 0..w {
            let pix = (y0 as usize + ly) * f.img_w as usize + x0 as usize + lx;
            t_cur[ly][lx] = r.final_t[pix];
            w_tot[ly][lx] = 1.0 - r.final_t[pix];
            last[ly][lx] = r.last_idx[pix];
            for (c, g) in gr.iter_mut().enumerate() {
                g[ly][lx] = d_image[pix * 3 + c];
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
    let mut j = stop;
    while j > start {
        j -= 1;
        let c = f.sorted[j] as usize;
        let p = &f.projected[c * 9..c * 9 + 9];
        let Some((ly0, ly1, level)) = splat_rows(p, y0, h) else {
            continue;
        };
        let (mu_x, mu_y, c00, c01, c11, opac) = (p[0], p[1], p[2], p[3], p[4], p[5]);
        let col = [p[6], p[7], p[8]];
        let ju = j as u32;
        let mut acc = [[0.0f32; W]; NA];
        let mut any = false;
        for ly in ly0..=ly1 {
            let dy = (y0 as usize + ly) as f32 + 0.5 - mu_y;
            let Some((k_lo, k_hi)) = row_lanes(level, dy, mu_x, x0 as f32) else {
                continue;
            };
            let mut row_any = 0u32;
            macro_rules! lanes8 {
                ($base:expr) => {
                    for kk in 0..8 {
                        let k = $base + kk;
                        let dx = xs[k] - mu_x;
                        let sigma = 0.5 * (c00 * dx * dx + c11 * dy * dy) + c01 * dx * dy;
                        let e = fast_exp(-sigma);
                        let raw = opac * e;
                        let a = raw.min(0.99);
                        let hit = ju < last[ly][k] && sigma >= 0.0 && a >= 1.0 / 255.0;
                        row_any |= hit as u32;
                        let a = if hit { a } else { 0.0 };
                        let t_before = t_cur[ly][k] / (1.0 - a);
                        let at = a * t_before;
                        let g = [gr[0][ly][k], gr[1][ly][k], gr[2][ly][k]];
                        let b = [be[0][ly][k], be[1][ly][k], be[2][ly][k]];
                        acc[B_COL][k] += at * g[0];
                        acc[B_COL + 1][k] += at * g[1];
                        acc[B_COL + 2][k] += at * g[2];
                        let dca = [
                            t_before * (col[0] - b[0]),
                            t_before * (col[1] - b[1]),
                            t_before * (col[2] - b[2]),
                        ];
                        let d_alpha = dca[0] * g[0] + dca[1] * g[1] + dca[2] * g[2];
                        for ch in 0..3 {
                            be[ch][ly][k] = a * col[ch] + (1.0 - a) * b[ch];
                        }
                        t_cur[ly][k] = t_before;
                        let live = hit && raw <= 0.99;
                        // dalpha/ds for s = (u, v, A, B, C, opacity).
                        let la = if live { a } else { 0.0 };
                        let ds = [
                            la * (c00 * dx + c01 * dy),
                            la * (c01 * dx + c11 * dy),
                            -la * 0.5 * dx * dx,
                            -la * dx * dy,
                            -la * 0.5 * dy * dy,
                            if live { e } else { 0.0 },
                        ];
                        for q in 0..6 {
                            acc[B_GRAD + q][k] += d_alpha * ds[q];
                        }
                        if GN {
                            let ratio = w_tot[ly][k] / at.max(1e-12);
                            let fo = if overlap == 1.0 {
                                ratio
                            } else if overlap == 0.0 {
                                1.0
                            } else if overlap == 0.5 {
                                ratio.sqrt()
                            } else {
                                ratio.powf(overlap)
                            };
                            let fo = if hit { fo.max(1.0) } else { 0.0 };
                            let hp = h_px * fo;
                            let wpx = hp * (dca[0] * dca[0] + dca[1] * dca[1] + dca[2] * dca[2]);
                            let mut n = B_H;
                            for q0 in 0..6 {
                                let wq = wpx * ds[q0];
                                for q1 in q0..6 {
                                    acc[n][k] += wq * ds[q1];
                                    n += 1;
                                }
                            }
                            acc[B_HC][k] += hp * at * at;
                        }
                    }
                };
            }
            if k_lo == 0 {
                lanes8!(0);
            }
            if k_hi == 16 {
                lanes8!(8);
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
