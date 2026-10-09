//! Batched GPU descriptor matching over a device-resident feature bank.
//!
//! [`GpuMatcher::match_pairs`] reproduces `BruteForceMatcher { ratio }`
//! (optionally wrapped in `CrossCheckMatcher`) for many image pairs in one
//! dispatch: the GPU computes per-row top-2 scores, the host applies the
//! ratio test and the cross-check. A bank uploaded with
//! [`FeatureBank::upload_u8`] matches COLMAP-style u8 descriptors with exact
//! integer dot products instead (about 1.6x faster on an NVIDIA L4).

use visloc_gsplat_render::GpuContext;
use visloc_vision::matching::DescriptorMatch;

use crate::extractor::{read_bytes, storage};
use crate::SiftGpuError;

/// All descriptors of a set of images, uploaded once.
pub struct FeatureBank {
    desc: wgpu::Buffer,
    norms: wgpu::Buffer,
    offsets: Vec<usize>,
    counts: Vec<usize>,
    dim: usize,
    norm_sq: Vec<f32>,
    /// Descriptors stored as COLMAP-style u8 (see [`FeatureBank::upload_u8`]).
    quantized: bool,
}

/// Scale of the u8 quantisation: byte = round(512 x), clamped to 255.
const U8_SCALE: f32 = 512.0;

impl FeatureBank {
    /// Upload one descriptor list per image. Every descriptor must have the
    /// same dimension, a multiple of 32 (SIFT 128, SuperPoint 256).
    pub fn upload(ctx: &GpuContext, images: &[&[Vec<f32>]]) -> Result<Self, SiftGpuError> {
        Self::upload_as(ctx, images, false)
    }

    /// Upload one descriptor list per image as u8, quantised like COLMAP's
    /// `FeatureDescriptorsToUnsignedByte` (round(512 x), clamped to 0..=255;
    /// unit-norm SIFT stays in range) and packed four per u32: a quarter of
    /// the memory of [`FeatureBank::upload`]. Matching then ranks by exact
    /// integer distances of the quantised descriptors (distances come back in
    /// the f32 descriptor scale). Dimension must be a multiple of 64.
    pub fn upload_u8(ctx: &GpuContext, images: &[&[Vec<f32>]]) -> Result<Self, SiftGpuError> {
        Self::upload_as(ctx, images, true)
    }

    fn upload_as(
        ctx: &GpuContext,
        images: &[&[Vec<f32>]],
        quantized: bool,
    ) -> Result<Self, SiftGpuError> {
        let dim = images
            .iter()
            .flat_map(|d| d.first())
            .map(Vec::len)
            .next()
            .unwrap_or(128);
        if dim == 0 || dim % if quantized { 64 } else { 32 } != 0 {
            return Err(SiftGpuError::Unsupported);
        }
        let mut offsets = Vec::with_capacity(images.len());
        let mut counts = Vec::with_capacity(images.len());
        // f32 bits, or four u8 per word.
        let mut words: Vec<u32> = Vec::new();
        let mut norm_sq: Vec<f32> = Vec::new();
        let q = |x: f32| (x * U8_SCALE).round().clamp(0.0, 255.0) as u32;
        for d in images {
            offsets.push(norm_sq.len());
            counts.push(d.len());
            for row in d.iter() {
                if row.len() != dim {
                    return Err(SiftGpuError::Unsupported);
                }
                if quantized {
                    let mut n = 0u32;
                    for c in row.chunks_exact(4) {
                        let b = [q(c[0]), q(c[1]), q(c[2]), q(c[3])];
                        n += b.iter().map(|v| v * v).sum::<u32>();
                        words.push(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
                    }
                    // At most 255^2 * dim: exact in f32 for dim < 258.
                    norm_sq.push(n as f32);
                } else {
                    words.extend(row.iter().map(|x| x.to_bits()));
                    norm_sq.push(row.iter().map(|x| x * x).sum());
                }
            }
        }
        let dev = &ctx.device;
        let use_ = wgpu::BufferUsages::empty();
        let desc = storage(dev, "bank-desc", (words.len() * 4) as u64, use_);
        ctx.queue
            .write_buffer(&desc, 0, bytemuck::cast_slice(&words));
        let norms = storage(dev, "bank-norms", (norm_sq.len() * 4) as u64, use_);
        ctx.queue
            .write_buffer(&norms, 0, bytemuck::cast_slice(&norm_sq));
        Ok(Self {
            desc,
            norms,
            offsets,
            counts,
            dim,
            norm_sq,
            quantized,
        })
    }

    pub fn num_images(&self) -> usize {
        self.counts.len()
    }
}

/// Top-2 result for one query row.
#[derive(Clone, Copy)]
struct Top2 {
    best: u32,
    s1: f32,
    s2: f32,
}

pub struct GpuMatcher {
    pipeline: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    /// Forward-only top-2 over u8 banks ([`FeatureBank::upload_u8`]).
    pipeline_u8: wgpu::ComputePipeline,
}

/// Rows of output per dispatch batch (16 B each).
const MAX_BATCH_ROWS: usize = 1 << 20;
/// Reverse partial records per dispatch batch (16 B each, 64 MiB).
const MAX_BATCH_PARTIALS: usize = 1 << 22;
const NONE: u32 = u32::MAX;

/// Forward (and, with cross-check, reverse) top-2 of one pair.
struct PairTops {
    forward: Vec<Top2>,
    reverse: Vec<Top2>,
}

impl GpuMatcher {
    pub fn new(ctx: &GpuContext) -> Self {
        let module = |label: &str, src: &str| {
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                })
        };
        let pipe = |module: &wgpu::ShaderModule, entry: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let f32_module = module("match", include_str!("shaders/match.wgsl"));
        let u8_module = module("match-u8", include_str!("shaders/match_u8.wgsl"));
        Self {
            pipeline: pipe(&f32_module, "top2"),
            reduce: pipe(&f32_module, "rev_reduce"),
            pipeline_u8: pipe(&u8_module, "top2"),
        }
    }

    /// Match image `i` (query) against image `j` (train) for every `(i, j)`
    /// in `pairs`; same result as `BruteForceMatcher { ratio }` and, with
    /// `cross_check`, `CrossCheckMatcher::new(BruteForceMatcher { ratio })`
    /// (up to f32 summation order in the dot products; for a u8 bank, of the
    /// quantised descriptors, exactly). The cross-check's reverse direction
    /// reuses the forward dot products (f32), or matches the swapped entries
    /// in the same dispatch (u8).
    pub fn match_pairs(
        &self,
        ctx: &GpuContext,
        bank: &FeatureBank,
        pairs: &[(usize, usize)],
        ratio: Option<f32>,
        cross_check: bool,
    ) -> Vec<Vec<DescriptorMatch>> {
        let tops = self.top2(ctx, bank, pairs, cross_check);
        pairs
            .iter()
            .zip(&tops)
            .map(|(&(i, j), t)| {
                let forward = self.finish(bank, i, j, &t.forward, ratio);
                if !cross_check || forward.is_empty() {
                    return forward;
                }
                let reverse = self.finish(bank, j, i, &t.reverse, ratio);
                let mut back = vec![usize::MAX; bank.counts[j]];
                for m in &reverse {
                    back[m.query_index] = m.train_index;
                }
                forward
                    .into_iter()
                    .filter(|m| back[m.train_index] == m.query_index)
                    .collect()
            })
            .collect()
    }

    fn finish(
        &self,
        bank: &FeatureBank,
        qi: usize,
        ti: usize,
        tops: &[Top2],
        ratio: Option<f32>,
    ) -> Vec<DescriptorMatch> {
        let nt = bank.counts[ti];
        if nt == 0 {
            return Vec::new();
        }
        let qn = &bank.norm_sq[bank.offsets[qi]..bank.offsets[qi] + bank.counts[qi]];
        let scale = if bank.quantized { U8_SCALE } else { 1.0 };
        let mut out = Vec::new();
        for (query_index, t) in tops.iter().enumerate() {
            let distance = (qn[query_index] + t.s1).max(0.0).sqrt() / scale;
            let second = (nt >= 2).then(|| (qn[query_index] + t.s2).max(0.0).sqrt() / scale);
            if let (Some(r), Some(s)) = (ratio, second) {
                if distance >= r * s {
                    continue;
                }
            }
            out.push(DescriptorMatch {
                query_index,
                train_index: t.best as usize,
                distance,
                second_best_distance: second,
                ratio: second.map(|s| distance / s),
                confidence: None,
            });
        }
        out
    }

    /// Per pair, the top-2 of every query row, plus (with `reverse`) the
    /// top-2 of every train row against the query image, from the same
    /// dot products.
    fn top2(
        &self,
        ctx: &GpuContext,
        bank: &FeatureBank,
        pairs: &[(usize, usize)],
        reverse: bool,
    ) -> Vec<PairTops> {
        if reverse && bank.quantized {
            // u8 dot products are exact integers, so the swapped entry gives
            // the reverse top-2 bit for bit; two forward passes beat the fused
            // reverse epilogue once the integer GEMM is this cheap.
            let both: Vec<(usize, usize)> =
                pairs.iter().flat_map(|&(i, j)| [(i, j), (j, i)]).collect();
            let mut tops = self.top2(ctx, bank, &both, false).into_iter();
            let mut result = Vec::with_capacity(pairs.len());
            while let (Some(f), Some(r)) = (tops.next(), tops.next()) {
                result.push(PairTops {
                    forward: f.forward,
                    reverse: r.forward,
                });
            }
            return result;
        }
        let mut result: Vec<PairTops> = Vec::with_capacity(pairs.len());
        let mut start = 0;
        while start < pairs.len() {
            // Batch so the outputs stay bounded and entries fit one dispatch.
            let mut end = start;
            let mut rows = 0usize;
            let mut parts = 0usize;
            while end < pairs.len() && end - start < 65535 {
                let (i, j) = pairs[end];
                let (r, pt) = if reverse {
                    (
                        bank.counts[i] + bank.counts[j],
                        bank.counts[i].div_ceil(128) * bank.counts[j],
                    )
                } else {
                    (bank.counts[i], 0)
                };
                if end > start && (rows + r > MAX_BATCH_ROWS || parts + pt > MAX_BATCH_PARTIALS) {
                    break;
                }
                rows += r;
                parts += pt;
                end += 1;
            }
            result.extend(self.top2_batch(ctx, bank, &pairs[start..end], reverse, rows, parts));
            start = end;
        }
        result
    }

    fn top2_batch(
        &self,
        ctx: &GpuContext,
        bank: &FeatureBank,
        pairs: &[(usize, usize)],
        reverse: bool,
        rows: usize,
        parts: usize,
    ) -> Vec<PairTops> {
        let dev = &ctx.device;
        let queue = &ctx.queue;
        let mut words: Vec<u32> = Vec::with_capacity(pairs.len() * 8);
        let mut offs = Vec::with_capacity(pairs.len());
        let mut out_off = 0usize;
        let mut part_off = 0usize;
        let mut max_q = 0usize;
        let mut max_t = 0usize;
        for &(qi, ti) in pairs {
            let (nq, nt) = (bank.counts[qi], bank.counts[ti]);
            let rev_out = out_off + nq;
            words.extend_from_slice(&[
                bank.offsets[qi] as u32,
                nq as u32,
                bank.offsets[ti] as u32,
                nt as u32,
                out_off as u32,
                part_off as u32,
                if reverse { rev_out as u32 } else { NONE },
                0,
            ]);
            offs.push((out_off, rev_out));
            out_off += nq;
            if reverse {
                out_off += nt;
                part_off += nq.div_ceil(128) * nt;
            }
            max_q = max_q.max(nq);
            max_t = max_t.max(nt);
        }
        if rows == 0 {
            return pairs
                .iter()
                .map(|_| PairTops {
                    forward: Vec::new(),
                    reverse: Vec::new(),
                })
                .collect();
        }
        let use_ = wgpu::BufferUsages::empty();
        let entries = storage(dev, "match-entries", (words.len() * 4) as u64, use_);
        queue.write_buffer(&entries, 0, bytemuck::cast_slice(&words));
        let out = storage(dev, "match-out", (rows * 16) as u64, use_);
        let rev_part = storage(dev, "match-rev-part", (parts * 16) as u64, use_);
        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("match-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // vec4s per descriptor row: of f32, or of u32 words of four bytes.
        let dim4 = if bank.quantized {
            bank.dim / 16
        } else {
            bank.dim / 4
        };
        let pw: [u32; 4] = [dim4 as u32, pairs.len() as u32, 0, 0];
        // The u8 kernel is forward only (`top2` sends cross-checks as
        // swapped forward entries) and has no reverse-partial binding.
        debug_assert!(!(bank.quantized && reverse));
        let top2 = if bank.quantized {
            &self.pipeline_u8
        } else {
            &self.pipeline
        };
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&pw));
        let bind = |pipeline: &wgpu::ComputePipeline, bindings: &[(u32, &wgpu::Buffer)]| {
            let entries: Vec<wgpu::BindGroupEntry> = bindings
                .iter()
                .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("match"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let all = [
            (0, &params),
            (1, &bank.desc),
            (2, &bank.norms),
            (3, &entries),
            (4, &out),
            (5, &rev_part),
        ];
        let bg = bind(top2, if bank.quantized { &all[..5] } else { &all[..] });
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("match"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("match"),
                timestamp_writes: None,
            });
            pass.set_pipeline(top2);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(max_q.div_ceil(128) as u32, pairs.len() as u32, 1);
        }
        if reverse {
            let bg = bind(&self.reduce, &[(3, &entries), (4, &out), (5, &rev_part)]);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("match-rev-reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reduce);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(max_t.div_ceil(64) as u32, pairs.len() as u32, 1);
        }
        queue.submit(Some(encoder.finish()));
        let raw: Vec<u32> = bytemuck::cast_slice(&read_bytes(dev, queue, &out, rows * 16)).to_vec();
        let read = |off: usize, n: usize| -> Vec<Top2> {
            (0..n)
                .map(|r| {
                    let w = &raw[(off + r) * 4..(off + r) * 4 + 4];
                    Top2 {
                        best: w[0],
                        s1: f32::from_bits(w[1]),
                        s2: f32::from_bits(w[2]),
                    }
                })
                .collect()
        };
        pairs
            .iter()
            .zip(&offs)
            .map(|(&(qi, ti), &(fwd, rev))| PairTops {
                forward: read(fwd, bank.counts[qi]),
                reverse: if reverse {
                    read(rev, bank.counts[ti])
                } else {
                    Vec::new()
                },
            })
            .collect()
    }
}
