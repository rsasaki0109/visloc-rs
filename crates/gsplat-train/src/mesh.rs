//! Mesh extraction from rendered depth: TSDF fusion + surface nets.
//!
//! The trained splat is rendered from every view as a median-depth map (see
//! `Renderer::render_depth`); the depths are fused into a truncated signed
//! distance field on a sparse grid of 8^3-voxel blocks, and the zero level set
//! is extracted with (naive) surface nets: one vertex per cell that the
//! surface crosses, placed at the mean of its edge crossings, and one quad per
//! grid edge with a sign change. No lookup tables, and the quads come out
//! consistently oriented (normals toward free space).

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;

use nalgebra::Vector3;
use rayon::prelude::*;
use visloc_gsplat_core::camera::CameraView;

const B: i32 = 8;
const BV: usize = (B * B * B) as usize;

#[derive(Clone)]
struct Block {
    sdf: [f32; BV],
    w: [f32; BV],
    col: [[f32; 3]; BV],
}

impl Block {
    fn new() -> Box<Self> {
        Box::new(Self {
            sdf: [1.0; BV],
            w: [0.0; BV],
            col: [[0.0; 3]; BV],
        })
    }
}

/// A depth frame to fuse: per-pixel camera z (0 = no surface) and colour.
pub struct DepthFrame<'a> {
    pub view: &'a CameraView,
    pub depth: &'a [f32],
    pub rgb: &'a [[f32; 3]],
}

/// Sparse TSDF volume.
pub struct Tsdf {
    voxel: f32,
    trunc: f32,
    max_depth: f32,
    /// Also carve free space along pixels with no surface (median depth 0,
    /// i.e. mostly transparent) or a surface beyond `max_depth`. Removes
    /// floaters that other views see through.
    pub carve: bool,
    /// Voxels fused from fewer observations are treated as unknown when
    /// extracting.
    pub min_weight: f32,
    blocks: HashMap<[i32; 3], Box<Block>>,
}

/// Triangle mesh with per-vertex colour.
#[derive(Default)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub colors: Vec<[u8; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

/// Max pyramid of `d + trunc` over a depth frame, +inf where the pixel
/// has no fused surface (no depth, or beyond `max_depth`): an upper bound,
/// over any pixel rectangle, of the depth behind which a voxel projecting
/// there is left unchanged.
struct FarPyramid {
    levels: Vec<(Vec<f32>, usize, usize)>,
}

impl FarPyramid {
    fn new(depth: &[f32], w: usize, h: usize, trunc: f32, max_depth: f32) -> Self {
        let base: Vec<f32> = depth
            .iter()
            .map(|&d| {
                if d > 0.0 && d <= max_depth {
                    d + trunc
                } else {
                    f32::INFINITY
                }
            })
            .collect();
        let mut levels = vec![(base, w, h)];
        while let Some((prev, pw, ph)) = levels.last().filter(|l| l.1 > 1 || l.2 > 1) {
            let (nw, nh) = (pw.div_ceil(2), ph.div_ceil(2));
            let mut next = vec![f32::NEG_INFINITY; nw * nh];
            for y in 0..*ph {
                for x in 0..*pw {
                    let c = &mut next[(y / 2) * nw + x / 2];
                    *c = c.max(prev[y * pw + x]);
                }
            }
            levels.push((next, nw, nh));
        }
        Self { levels }
    }

    /// Max over the pixels `u0..=u1` x `v0..=v1` (or an upper bound of it).
    fn max(&self, u0: usize, v0: usize, u1: usize, v1: usize) -> f32 {
        let mut l = 0;
        while l + 1 < self.levels.len() && ((u1 >> l) - (u0 >> l) > 1 || (v1 >> l) - (v0 >> l) > 1)
        {
            l += 1;
        }
        let (cells, lw, _) = &self.levels[l];
        let mut m = f32::NEG_INFINITY;
        for y in (v0 >> l)..=(v1 >> l) {
            for x in (u0 >> l)..=(u1 >> l) {
                m = m.max(cells[y * lw + x]);
            }
        }
        m
    }
}

fn floor_div(a: i32, b: i32) -> i32 {
    a.div_euclid(b)
}

impl Tsdf {
    /// `voxel`: edge length; `trunc`: truncation distance (a few voxels);
    /// depths beyond `max_depth` are ignored (background).
    pub fn new(voxel: f32, trunc: f32, max_depth: f32) -> Self {
        Self {
            voxel,
            trunc,
            max_depth,
            carve: true,
            min_weight: 1.0,
            blocks: HashMap::new(),
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn voxel_of(&self, p: Vector3<f32>) -> [i32; 3] {
        [
            (p.x / self.voxel).floor() as i32,
            (p.y / self.voxel).floor() as i32,
            (p.z / self.voxel).floor() as i32,
        ]
    }

    /// Fuse one depth frame.
    pub fn integrate(&mut self, f: &DepthFrame<'_>) {
        let cam = &f.view.camera;
        let (w, h) = (cam.width as usize, cam.height as usize);
        let rt = f.view.rotation.transpose();
        let t = f.view.translation;
        let center = f.view.camera_center();

        // Allocate every block the truncation band of a surface pixel touches
        // (samples along the ray no further apart than half a block).
        let step = (self.voxel * B as f32 * 0.5).min(self.trunc);
        let n_steps = (2.0 * self.trunc / step).ceil() as i32;
        // Only blocks not allocated yet: after the first views almost every
        // sample lands in an existing block, and collecting all of them into
        // a set cost more than the fusion itself.
        let blocks = &self.blocks;
        let needed: HashSet<[i32; 3]> = (0..h)
            .into_par_iter()
            .step_by(2)
            .flat_map_iter(|y| {
                let mut out = Vec::new();
                for x in (0..w).step_by(2) {
                    let z = f.depth[y * w + x];
                    if z <= 0.0 || z > self.max_depth {
                        continue;
                    }
                    let pc = Vector3::new(
                        (x as f32 + 0.5 - cam.cx) / cam.fx * z,
                        (y as f32 + 0.5 - cam.cy) / cam.fy * z,
                        z,
                    );
                    let p = rt * (pc - t);
                    let dir = (p - center).normalize();
                    let mut last = None;
                    for k in 0..=n_steps {
                        let q = p + dir * (-self.trunc + k as f32 * step);
                        let v = self.voxel_of(q);
                        let key = [floor_div(v[0], B), floor_div(v[1], B), floor_div(v[2], B)];
                        if last != Some(key) && !blocks.contains_key(&key) {
                            out.push(key);
                        }
                        last = Some(key);
                    }
                }
                out
            })
            .collect();
        for key in needed {
            self.blocks.entry(key).or_insert_with(Block::new);
        }

        let (voxel, trunc, max_depth, carve) = (self.voxel, self.trunc, self.max_depth, self.carve);
        let r = f.view.rotation;
        // Frustum culling: a voxel only changes when it projects into the
        // image in front of the camera, so a block whose bounding sphere lies
        // entirely outside one of the frustum's side / near planes is skipped
        // without changing the result (depth cannot be culled: free-space
        // carving reaches any distance behind empty pixels).
        let planes = {
            let (fx, fy, cx, cy) = (cam.fx, cam.fy, cam.cx, cam.cy);
            let (wf, hf) = (w as f32, h as f32);
            [
                Vector3::new(fx, 0.0, cx),       // u >= 0
                Vector3::new(-fx, 0.0, wf - cx), // u <= w
                Vector3::new(0.0, fy, cy),       // v >= 0
                Vector3::new(0.0, -fy, hf - cy), // v <= h
                Vector3::new(0.0, 0.0, 1.0),     // z >= 0
            ]
            .map(|n| n.normalize())
        };
        let half = 0.5 * B as f32 * voxel;
        let radius = half * 3.0f32.sqrt();
        // Occlusion culling: a voxel more than `trunc` behind the surface
        // depth of its pixel is left unchanged, so a block whose nearest
        // voxel lies behind every such depth over the pixels it projects to
        // is skipped, again without changing the result.
        let far = FarPyramid::new(f.depth, w, h, trunc, max_depth);
        let occluded = |key: &[i32; 3]| -> bool {
            let lo = Vector3::new(
                (key[0] * B) as f32 + 0.5,
                (key[1] * B) as f32 + 0.5,
                (key[2] * B) as f32 + 0.5,
            ) * voxel;
            let span = (B - 1) as f32 * voxel;
            let (mut umin, mut vmin, mut zmin) = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
            let (mut umax, mut vmax) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for c in 0..8 {
                let p = lo
                    + Vector3::new(
                        (c & 1) as f32 * span,
                        ((c >> 1) & 1) as f32 * span,
                        ((c >> 2) & 1) as f32 * span,
                    );
                let pc = r * p + t;
                if pc.z <= 1e-3 {
                    return false;
                }
                let (u, v) = (cam.fx * pc.x / pc.z + cam.cx, cam.fy * pc.y / pc.z + cam.cy);
                umin = umin.min(u);
                umax = umax.max(u);
                vmin = vmin.min(v);
                vmax = vmax.max(v);
                zmin = zmin.min(pc.z);
            }
            // Voxel centres project inside the corners' box (one pixel of
            // slack for rounding); voxels outside the image are skipped
            // anyway, so the box is clamped to it.
            let clamp = |x: f32, n: usize| (x.floor().max(0.0) as usize).min(n - 1);
            if umax < -1.0 || vmax < -1.0 || umin > w as f32 + 1.0 || vmin > h as f32 + 1.0 {
                return false;
            }
            let (u0, u1) = (clamp(umin - 1.0, w), clamp(umax + 1.0, w));
            let (v0, v1) = (clamp(vmin - 1.0, h), clamp(vmax + 1.0, h));
            zmin * (1.0 - 1e-4) - 1e-4 * voxel > far.max(u0, v0, u1, v1)
        };
        self.blocks.par_iter_mut().for_each(|(key, blk)| {
            let centre = Vector3::new(
                (key[0] * B) as f32 * voxel + half,
                (key[1] * B) as f32 * voxel + half,
                (key[2] * B) as f32 * voxel + half,
            );
            let cc = r * centre + t;
            if planes.iter().any(|n| n.dot(&cc) < -radius) || occluded(key) {
                return;
            }
            for i in 0..BV {
                let (lx, ly, lz) = (i as i32 % B, (i as i32 / B) % B, i as i32 / (B * B));
                let p = Vector3::new(
                    ((key[0] * B + lx) as f32 + 0.5) * voxel,
                    ((key[1] * B + ly) as f32 + 0.5) * voxel,
                    ((key[2] * B + lz) as f32 + 0.5) * voxel,
                );
                let pc = r * p + t;
                if pc.z <= 1e-6 {
                    continue;
                }
                let u = cam.fx * pc.x / pc.z + cam.cx;
                let v = cam.fy * pc.y / pc.z + cam.cy;
                if u < 0.0 || v < 0.0 || u >= w as f32 || v >= h as f32 {
                    continue;
                }
                let pix = v as usize * w + u as usize;
                let d = f.depth[pix];
                if d <= 0.0 || d > max_depth {
                    // Nothing opaque (or only background) along this pixel:
                    // the voxel is free space.
                    if carve && (d <= 0.0 || pc.z < d - trunc) {
                        let wn = blk.w[i] + 1.0;
                        blk.sdf[i] = (blk.sdf[i] * blk.w[i] + 1.0) / wn;
                        blk.w[i] = wn;
                    }
                    continue;
                }
                let sdf = d - pc.z;
                if sdf < -trunc {
                    continue;
                }
                let tsdf = (sdf / trunc).min(1.0);
                let wn = blk.w[i] + 1.0;
                blk.sdf[i] = (blk.sdf[i] * blk.w[i] + tsdf) / wn;
                let (c, w0) = (f.rgb[pix], blk.w[i]);
                for (acc, x) in blk.col[i].iter_mut().zip(c) {
                    *acc = (*acc * w0 + x) / wn;
                }
                blk.w[i] = wn;
            }
        });
    }

    fn sample(&self, v: [i32; 3]) -> Option<(f32, [f32; 3])> {
        let key = [floor_div(v[0], B), floor_div(v[1], B), floor_div(v[2], B)];
        let blk = self.blocks.get(&key)?;
        let l = [v[0] - key[0] * B, v[1] - key[1] * B, v[2] - key[2] * B];
        let i = (l[0] + l[1] * B + l[2] * B * B) as usize;
        (blk.w[i] >= self.min_weight.max(1e-6)).then(|| (blk.sdf[i], blk.col[i]))
    }

    /// Surface-nets extraction of the zero level set. Crossings between two
    /// truncated values (|tsdf| ~ 1, e.g. at occlusion edges) are skipped.
    pub fn extract(&self) -> Mesh {
        const CORNERS: [[i32; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        let crosses = |a: f32, b: f32| (a < 0.0) != (b < 0.0) && a.abs() < 0.999 && b.abs() < 0.999;

        // One vertex per cell (cell origin = its min-corner voxel).
        let cells: Vec<([i32; 3], [f32; 3], [f32; 3])> = self
            .blocks
            .par_iter()
            .flat_map_iter(|(key, _)| {
                let mut out = Vec::new();
                for i in 0..BV as i32 {
                    let o = [
                        key[0] * B + i % B,
                        key[1] * B + (i / B) % B,
                        key[2] * B + i / (B * B),
                    ];
                    let mut vals = [(0.0f32, [0.0f32; 3]); 8];
                    let mut ok = true;
                    for (c, d) in CORNERS.iter().enumerate() {
                        match self.sample([o[0] + d[0], o[1] + d[1], o[2] + d[2]]) {
                            Some(s) => vals[c] = s,
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let mut sum = Vector3::zeros();
                    let mut col = [0.0f32; 3];
                    let mut n = 0;
                    for &(a, b) in &EDGES {
                        let (sa, sb) = (vals[a].0, vals[b].0);
                        if !crosses(sa, sb) {
                            continue;
                        }
                        let t = sa / (sa - sb);
                        let pa = Vector3::from(CORNERS[a].map(|x| x as f32));
                        let pb = Vector3::from(CORNERS[b].map(|x| x as f32));
                        sum += pa + (pb - pa) * t;
                        for (ch, acc) in col.iter_mut().enumerate() {
                            *acc += vals[a].1[ch] * (1.0 - t) + vals[b].1[ch] * t;
                        }
                        n += 1;
                    }
                    if n == 0 {
                        continue;
                    }
                    let local = sum / n as f32;
                    let p = [
                        (o[0] as f32 + 0.5 + local.x) * self.voxel,
                        (o[1] as f32 + 0.5 + local.y) * self.voxel,
                        (o[2] as f32 + 0.5 + local.z) * self.voxel,
                    ];
                    out.push((o, p, col.map(|c| c / n as f32)));
                }
                out
            })
            .collect();

        let mut mesh = Mesh::default();
        let mut index: HashMap<[i32; 3], u32> = HashMap::with_capacity(cells.len());
        for (o, p, c) in &cells {
            index.insert(*o, mesh.vertices.len() as u32);
            mesh.vertices.push(*p);
            mesh.colors
                .push(c.map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8));
        }

        // One quad per grid edge (from voxel v along axis a) with a sign
        // change; the four cells around it are v minus {0,1} along the two
        // other axes.
        let quads: Vec<[u32; 4]> = cells
            .par_iter()
            .flat_map_iter(|(v, _, _)| {
                let mut out = Vec::new();
                let Some((s0, _)) = self.sample(*v) else {
                    return out;
                };
                for a in 0..3 {
                    let mut e = *v;
                    e[a] += 1;
                    let Some((s1, _)) = self.sample(e) else {
                        continue;
                    };
                    if !crosses(s0, s1) {
                        continue;
                    }
                    let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                    let cell = |db: i32, dc: i32| {
                        let mut k = *v;
                        k[b] -= db;
                        k[c] -= dc;
                        index.get(&k).copied()
                    };
                    let (Some(q0), Some(q1), Some(q2), Some(q3)) =
                        (cell(0, 0), cell(1, 0), cell(1, 1), cell(0, 1))
                    else {
                        continue;
                    };
                    // Outside (positive) -> inside: face toward the outside.
                    if s0 > 0.0 {
                        out.push([q0, q3, q2, q1]);
                    } else {
                        out.push([q0, q1, q2, q3]);
                    }
                }
                out
            })
            .collect();
        for q in quads {
            mesh.triangles.push([q[0], q[1], q[2]]);
            mesh.triangles.push([q[0], q[2], q[3]]);
        }
        mesh
    }
}

impl Mesh {
    /// Drop triangles with a vertex farther than `radius` from every point of
    /// `support` (e.g. the SfM points). Surfaces the splat invents where no
    /// feature was ever triangulated -- sky, mostly -- go; textured surfaces
    /// keep their SfM points nearby. Vertices are left in place.
    pub fn keep_supported(&mut self, support: &[[f32; 3]], radius: f32) {
        let cell = |p: [f32; 3]| p.map(|c| (c / radius).floor() as i32);
        let mut grid: HashMap<[i32; 3], Vec<[f32; 3]>> = HashMap::new();
        for &p in support {
            grid.entry(cell(p)).or_default().push(p);
        }
        let r2 = radius * radius;
        let supported: Vec<bool> = self
            .vertices
            .par_iter()
            .map(|&v| {
                let c = cell(v);
                (-1..=1).any(|dx| {
                    (-1..=1).any(|dy| {
                        (-1..=1).any(|dz| {
                            grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz])
                                .is_some_and(|ps| {
                                    ps.iter().any(|p| {
                                        (0..3).map(|k| (p[k] - v[k]).powi(2)).sum::<f32>() <= r2
                                    })
                                })
                        })
                    })
                })
            })
            .collect();
        self.triangles
            .retain(|t| t.iter().all(|&i| supported[i as usize]));
    }

    /// Keep only the connected components with at least `min_tris` triangles
    /// (drops floaters). Vertices are left in place.
    pub fn remove_small_components(&mut self, min_tris: usize) {
        let n = self.vertices.len();
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        for t in &self.triangles {
            for k in 1..3 {
                let (a, b) = (find(&mut parent, t[0]), find(&mut parent, t[k]));
                if a != b {
                    parent[a as usize] = b;
                }
            }
        }
        let mut count: HashMap<u32, usize> = HashMap::new();
        let roots: Vec<u32> = self
            .triangles
            .iter()
            .map(|t| find(&mut parent, t[0]))
            .collect();
        for r in &roots {
            *count.entry(*r).or_default() += 1;
        }
        let tris = std::mem::take(&mut self.triangles);
        self.triangles = tris
            .into_iter()
            .zip(roots)
            .filter(|(_, r)| count[r] >= min_tris)
            .map(|(t, _)| t)
            .collect();
    }

    /// Binary little-endian PLY with vertex colours; unreferenced vertices
    /// are dropped.
    pub fn write_ply(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut remap = vec![u32::MAX; self.vertices.len()];
        let mut order = Vec::new();
        for t in &self.triangles {
            for &v in t {
                if remap[v as usize] == u32::MAX {
                    remap[v as usize] = order.len() as u32;
                    order.push(v);
                }
            }
        }
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        write!(
            out,
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\n\
             property float x\nproperty float y\nproperty float z\n\
             property uchar red\nproperty uchar green\nproperty uchar blue\n\
             element face {}\nproperty list uchar int vertex_indices\nend_header\n",
            order.len(),
            self.triangles.len()
        )?;
        for &v in &order {
            for c in self.vertices[v as usize] {
                out.write_all(&c.to_le_bytes())?;
            }
            out.write_all(&self.colors[v as usize])?;
        }
        for t in &self.triangles {
            out.write_all(&[3u8])?;
            for &v in t {
                out.write_all(&(remap[v as usize] as i32).to_le_bytes())?;
            }
        }
        out.flush()
    }
}

/// Settings for [`extract_from_splat`]; lengths in voxels unless noted.
#[derive(Debug, Clone)]
pub struct MeshOptions {
    /// Voxel edge (world units); `None`: rig scale / 256.
    pub voxel: Option<f32>,
    pub trunc_voxels: f32,
    /// Ignore depths beyond this (world units); `None`: 2 x rig scale.
    pub max_depth: Option<f32>,
    pub carve: bool,
    pub min_weight: f32,
    /// Drop triangles farther than this from every support point (0: off).
    pub support_voxels: f32,
    pub min_component: usize,
}

impl Default for MeshOptions {
    fn default() -> Self {
        Self {
            voxel: None,
            trunc_voxels: 4.0,
            max_depth: None,
            carve: true,
            min_weight: 3.0,
            support_voxels: 10.0,
            min_component: 500,
        }
    }
}

/// Rig scale: median distance of the camera centres from their centroid.
pub fn rig_scale(views: &[&CameraView]) -> f32 {
    let centers: Vec<Vector3<f32>> = views.iter().map(|v| v.camera_center()).collect();
    let centroid = centers.iter().sum::<Vector3<f32>>() / centers.len().max(1) as f32;
    let mut d: Vec<f32> = centers.iter().map(|c| (c - centroid).norm()).collect();
    d.sort_by(f32::total_cmp);
    d.get(d.len() / 2).copied().unwrap_or(1.0).max(1e-6)
}

/// Render `scene`'s median depth from every view, fuse, extract, filter by
/// `support` (e.g. SfM points) and drop small components. Returns the mesh
/// and a one-line summary.
pub fn extract_from_splat(
    ctx: visloc_gsplat_render::GpuContext,
    scene: &visloc_gsplat_core::gaussian::Scene,
    views: &[&CameraView],
    support: &[[f32; 3]],
    opts: &MeshOptions,
) -> Result<(Mesh, String, visloc_gsplat_render::GpuContext), visloc_gsplat_render::GpuError> {
    let scale = rig_scale(views);
    let voxel = opts.voxel.unwrap_or(scale / 256.0);
    let max_depth = opts.max_depth.unwrap_or(2.0 * scale);
    let mut tsdf = Tsdf::new(voxel, voxel * opts.trunc_voxels, max_depth);
    tsdf.carve = opts.carve;
    tsdf.min_weight = opts.min_weight;
    let mut ctx = Some(ctx);
    let mut renderer: Option<(u32, u32, visloc_gsplat_render::Renderer)> = None;
    let t0 = std::time::Instant::now();
    let (mut render_s, mut integrate_s) = (0.0f64, 0.0f64);
    for view in views {
        let (w, h) = (view.camera.width, view.camera.height);
        if renderer.as_ref().map(|r| (r.0, r.1)) != Some((w, h)) {
            let c = match renderer.take() {
                Some((_, _, r)) => r.into_context(),
                None => ctx.take().expect("context"),
            };
            renderer = Some((w, h, visloc_gsplat_render::Renderer::new(c, scene, w, h)?));
        }
        let (_, _, r) = renderer.as_mut().expect("renderer set above");
        let t = std::time::Instant::now();
        let (image, depth, _) = r.render_depth(view, [0.0, 0.0, 0.0]);
        render_s += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        tsdf.integrate(&DepthFrame {
            view,
            depth: &depth,
            rgb: &image.rgb,
        });
        integrate_s += t.elapsed().as_secs_f64();
    }
    let fused = t0.elapsed().as_secs_f64();
    let mut mesh = tsdf.extract();
    let raw = mesh.triangles.len();
    if opts.support_voxels > 0.0 && !support.is_empty() {
        mesh.keep_supported(support, opts.support_voxels * voxel);
    }
    mesh.remove_small_components(opts.min_component);
    let summary = format!(
        "mesh: {} views fused in {fused:.1} s (render {render_s:.1} s, integrate {integrate_s:.1} s; voxel {voxel:.4}, {} blocks), {raw} -> {} triangles",
        views.len(),
        tsdf.num_blocks(),
        mesh.triangles.len()
    );
    let ctx = match renderer {
        Some((_, _, r)) => r.into_context(),
        None => ctx.take().expect("context"),
    };
    Ok((mesh, summary, ctx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;
    use visloc_gsplat_core::camera::PinholeCamera;

    #[test]
    fn far_pyramid_bounds_every_rectangle() {
        let (w, h) = (37usize, 23usize);
        let mut s = 0x9E37_79B9u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        // Some pixels empty (0) or beyond max_depth: +inf.
        let depth: Vec<f32> = (0..w * h)
            .map(|_| match rnd() {
                r if r < 0.1 => 0.0,
                r if r < 0.15 => 50.0,
                r => 10.0 * r,
            })
            .collect();
        let (trunc, max_depth) = (0.3, 20.0);
        let far = FarPyramid::new(&depth, w, h, trunc, max_depth);
        for _ in 0..2000 {
            let (a, b) = ((rnd() * w as f32) as usize, (rnd() * w as f32) as usize);
            let (c, d) = ((rnd() * h as f32) as usize, (rnd() * h as f32) as usize);
            let (u0, u1, v0, v1) = (a.min(b), a.max(b), c.min(d), c.max(d));
            let mut truth = f32::NEG_INFINITY;
            for v in v0..=v1 {
                for u in u0..=u1 {
                    let z = depth[v * w + u];
                    truth = truth.max(if z > 0.0 && z <= max_depth {
                        z + trunc
                    } else {
                        f32::INFINITY
                    });
                }
            }
            assert!(far.max(u0, v0, u1, v1) >= truth, "{u0}..{u1} x {v0}..{v1}");
        }
    }

    #[test]
    fn keeps_only_supported_triangles() {
        let mut m = Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 5.0, 5.0],
            ],
            colors: vec![[0; 3]; 4],
            triangles: vec![[0, 1, 2], [1, 2, 3]],
        };
        m.keep_supported(&[[0.1, 0.1, 0.0], [0.9, 0.2, 0.0]], 1.0);
        assert_eq!(m.triangles, vec![[0, 1, 2]]);
    }

    /// A sphere seen from six axis-aligned cameras fuses to a closed mesh
    /// whose vertices lie on the sphere.
    #[test]
    fn fuses_sphere() {
        let (w, h) = (128u32, 128u32);
        let cam = PinholeCamera::new(w, h, 100.0, 100.0, 64.0, 64.0);
        let radius = 1.0f32;
        let dist = 3.0f32;
        let mut tsdf = Tsdf::new(0.02, 0.08, 10.0);
        let dirs: [Vector3<f32>; 6] = [
            Vector3::x(),
            -Vector3::x(),
            Vector3::y(),
            -Vector3::y(),
            Vector3::z(),
            -Vector3::z(),
        ];
        for d in dirs {
            // Camera at dist * d looking at the origin.
            let fwd = -d;
            let up: Vector3<f32> = if d.y.abs() > 0.5 {
                Vector3::z()
            } else {
                Vector3::y()
            };
            let right = up.cross(&fwd).normalize();
            let down = fwd.cross(&right);
            let r = Matrix3::from_rows(&[right.transpose(), down.transpose(), fwd.transpose()]);
            let c = d * dist;
            let view = CameraView::new(r, -(r * c), cam);
            let mut depth = vec![0.0f32; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    let ray_c = Vector3::new(
                        (x as f32 + 0.5 - 64.0) / 100.0,
                        (y as f32 + 0.5 - 64.0) / 100.0,
                        1.0,
                    );
                    let ray = (r.transpose() * ray_c).normalize();
                    // |c + s ray|^2 = radius^2
                    let b = c.dot(&ray);
                    let disc = b * b - (c.norm_squared() - radius * radius);
                    if disc > 0.0 {
                        let s = -b - disc.sqrt();
                        let p = c + ray * s;
                        depth[(y * w + x) as usize] = (r * p + view.translation).z;
                    }
                }
            }
            let rgb = vec![[0.5f32; 3]; (w * h) as usize];
            tsdf.integrate(&DepthFrame {
                view: &view,
                depth: &depth,
                rgb: &rgb,
            });
        }
        let mesh = tsdf.extract();
        assert!(
            mesh.triangles.len() > 1000,
            "{} triangles",
            mesh.triangles.len()
        );
        let max_err = mesh
            .vertices
            .iter()
            .map(|v| (Vector3::from(*v).norm() - radius).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.02, "max radial error {max_err}");
        // Outward-facing: the first triangle's normal points away from the origin.
        let t = mesh.triangles[0];
        let [a, b, c] = t.map(|i| Vector3::from(mesh.vertices[i as usize]));
        let n = (b - a).cross(&(c - a));
        assert!(n.dot(&a) > 0.0, "normal points inward");
    }
}
