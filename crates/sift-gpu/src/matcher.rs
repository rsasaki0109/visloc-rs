//! Batched GPU descriptor matching over a device-resident feature bank.
//!
//! [`GpuMatcher::match_pairs`] reproduces `BruteForceMatcher { ratio }`
//! (optionally wrapped in `CrossCheckMatcher`) for many image pairs in one
//! dispatch: the GPU computes per-row top-2 scores, the host applies the
//! ratio test and the cross-check.

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
}

impl FeatureBank {
    /// Upload one descriptor list per image. Every descriptor must have the
    /// same dimension, a multiple of 32 (SIFT 128, SuperPoint 256).
    pub fn upload(ctx: &GpuContext, images: &[&[Vec<f32>]]) -> Result<Self, SiftGpuError> {
        let dim = images
            .iter()
            .flat_map(|d| d.first())
            .map(Vec::len)
            .next()
            .unwrap_or(128);
        if dim == 0 || dim % 32 != 0 {
            return Err(SiftGpuError::Unsupported);
        }
        let mut offsets = Vec::with_capacity(images.len());
        let mut counts = Vec::with_capacity(images.len());
        let mut flat: Vec<f32> = Vec::new();
        let mut norm_sq: Vec<f32> = Vec::new();
        for d in images {
            offsets.push(norm_sq.len());
            counts.push(d.len());
            for row in d.iter() {
                if row.len() != dim {
                    return Err(SiftGpuError::Unsupported);
                }
                flat.extend_from_slice(row);
                norm_sq.push(row.iter().map(|x| x * x).sum());
            }
        }
        let dev = &ctx.device;
        let use_ = wgpu::BufferUsages::empty();
        let desc = storage(dev, "bank-desc", (flat.len() * 4) as u64, use_);
        ctx.queue
            .write_buffer(&desc, 0, bytemuck::cast_slice(&flat));
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
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("match"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/match.wgsl").into()),
            });
        let make = |entry: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        Self {
            pipeline: make("top2"),
            reduce: make("rev_reduce"),
        }
    }

    /// Match image `i` (query) against image `j` (train) for every `(i, j)`
    /// in `pairs`; same result as `BruteForceMatcher { ratio }` and, with
    /// `cross_check`, `CrossCheckMatcher::new(BruteForceMatcher { ratio })`
    /// (up to f32 summation order in the dot products). The cross-check's
    /// reverse direction reuses the forward dot products.
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
        let mut out = Vec::new();
        for (query_index, t) in tops.iter().enumerate() {
            let distance = (qn[query_index] + t.s1).max(0.0).sqrt();
            let second = (nt >= 2).then(|| (qn[query_index] + t.s2).max(0.0).sqrt());
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
                if end > start && (rows + r > MAX_BATCH_ROWS || parts + pt > MAX_BATCH_PARTIALS)
                {
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
        let pw: [u32; 4] = [(bank.dim / 4) as u32, pairs.len() as u32, 0, 0];
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
        let bg = bind(
            &self.pipeline,
            &[
                (0, &params),
                (1, &bank.desc),
                (2, &bank.norms),
                (3, &entries),
                (4, &out),
                (5, &rev_part),
            ],
        );
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("match"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("match"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
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
