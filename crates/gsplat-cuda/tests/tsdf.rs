//! The TSDF fused on the device (`CudaGsplat::tsdf_surface`) against the
//! CPU `Tsdf` behind `fuse_views`, and across slab sizes. Skipped unless
//! `VISLOC_GSPLAT_CUDA_LIB` points at the library built by
//! `scripts/build_gsplat_cuda.sh`.

use std::collections::HashMap;

use nalgebra::{Quaternion, Rotation3, Vector3};
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::gaussian::{Gaussian, Scene};
use visloc_gsplat_cuda::mesh::{extract_mesh, extract_mesh_gpu, surface_mesh, MeshDepth};
use visloc_gsplat_cuda::{CudaGsplat, TsdfParams};
use visloc_gsplat_render::packing::PackedScene;
use visloc_gsplat_train::mesh::{Mesh, MeshOptions};

fn gsplat() -> Option<CudaGsplat> {
    match CudaGsplat::from_env() {
        None => {
            eprintln!("skipping: {} is not set", visloc_gsplat_cuda::LIBRARY_ENV);
            None
        }
        Some(r) => Some(r.expect("load the CUDA library")),
    }
}

/// Opaque, flattened splats tiling a bumpy wall at z of about 5, so every
/// pixel of the views below sees a surface.
fn wall() -> Scene {
    let mut gaussians = Vec::new();
    for j in -12..=12 {
        for i in -16..=16 {
            let (x, y) = (i as f32 * 0.2, j as f32 * 0.2);
            let z = 5.0 + 0.3 * (x * 1.3).sin() * (y * 0.9).cos();
            gaussians.push(Gaussian {
                mean: Vector3::new(x, y, z),
                scale_log: Vector3::new(0.16f32.ln(), 0.16f32.ln(), 0.01f32.ln()),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                opacity_logit: 4.0,
                sh_dc: [(x * 0.7).sin(), (y * 1.1).cos(), 0.3],
                sh_rest: Vec::new(),
                sh_degree: 0,
            });
        }
    }
    Scene::new(gaussians, 0)
}

fn view(yaw: f32, x: f32) -> CameraView {
    let (w, h) = (128u32, 96u32);
    let r = Rotation3::from_euler_angles(0.0, yaw, 0.0).into_inner();
    // Camera centre at (x, 0, 0): t = -R c.
    CameraView::new(
        r,
        -(r * Vector3::new(x, 0.0, 0.0)),
        PinholeCamera::new(w, h, 110.0, 110.0, 64.0, 48.0),
    )
}

fn options(voxel: f32) -> MeshOptions {
    MeshOptions {
        voxel: Some(voxel),
        trunc_voxels: 4.0,
        max_depth: Some(20.0),
        carve: true,
        min_weight: 1.0,
        support_voxels: 0.0,
        min_component: 0,
    }
}

/// Vertices bucketed by voxel, for nearest-vertex queries.
struct Grid<'a> {
    points: &'a [[f32; 3]],
    voxel: f32,
    buckets: HashMap<[i32; 3], Vec<u32>>,
}

impl<'a> Grid<'a> {
    fn new(points: &'a [[f32; 3]], voxel: f32) -> Self {
        let mut buckets: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (i, &p) in points.iter().enumerate() {
            buckets
                .entry(p.map(|c| (c / voxel).floor() as i32))
                .or_default()
                .push(i as u32);
        }
        Self {
            points,
            voxel,
            buckets,
        }
    }

    /// The nearest point within a voxel of `p` and its distance (max norm).
    fn nearest(&self, p: [f32; 3]) -> Option<(usize, f32)> {
        let c = p.map(|x| (x / self.voxel).floor() as i32);
        let mut best: Option<(usize, f32)> = None;
        for d in 0..27 {
            let k = [c[0] + d % 3 - 1, c[1] + (d / 3) % 3 - 1, c[2] + d / 9 - 1];
            for &i in self.buckets.get(&k).into_iter().flatten() {
                let q = self.points[i as usize];
                let dist = (0..3).map(|j| (p[j] - q[j]).abs()).fold(0.0f32, f32::max);
                if best.is_none_or(|(_, b)| dist < b) {
                    best = Some((i as usize, dist));
                }
            }
        }
        best
    }
}

/// Triangles as sorted triples of vertex ids under `map` (each rotated to
/// start at its smallest id, which keeps the winding).
fn triangles_under(m: &Mesh, map: &[usize]) -> Vec<[usize; 3]> {
    let mut tris: Vec<[usize; 3]> = m
        .triangles
        .iter()
        .map(|t| {
            let r = t.map(|i| map[i as usize]);
            let k = (0..3).min_by_key(|&k| r[k]).unwrap();
            [r[k], r[(k + 1) % 3], r[(k + 2) % 3]]
        })
        .collect();
    tris.sort_unstable();
    tris
}

/// With one view both fusions allocate every block before fusing, so the
/// device's mesh is the CPU's: the same vertices (to float rounding of the
/// projections), colours and triangles, from the planar median depth and
/// from the expected depth.
#[test]
fn one_view_matches_the_cpu_tsdf() {
    for depth in [MeshDepth::Planar, MeshDepth::Expected] {
        one_view(depth);
    }
}

fn one_view(depth: MeshDepth) {
    let Some(mut gs) = gsplat() else { return };
    let scene = wall();
    gs.set_scene(&PackedScene::from_scene(&scene))
        .expect("upload");
    let v = view(0.1, 0.3);
    let views = [&v];
    let voxel = 0.04;
    let opts = options(voxel);
    let (cpu, _) = extract_mesh(&mut gs, 0, depth, &views, &[], &opts).expect("cpu mesh");
    let (gpu, summary) =
        extract_mesh_gpu(&mut gs, 0, depth, &views, &[], &opts, 0).expect("gpu mesh");
    eprintln!(
        "{depth:?}: {summary}; cpu {} vertices, {} triangles",
        cpu.vertices.len(),
        cpu.triangles.len()
    );
    assert!(cpu.triangles.len() > 1000, "too small a surface to compare");
    assert_eq!(cpu.vertices.len(), gpu.vertices.len(), "vertex counts");
    // Each GPU vertex's CPU twin.
    let grid = Grid::new(&cpu.vertices, voxel);
    let mut twin = vec![usize::MAX; gpu.vertices.len()];
    let mut taken = vec![false; cpu.vertices.len()];
    let mut worst = 0.0f32;
    for (i, &p) in gpu.vertices.iter().enumerate() {
        let (j, d) = grid
            .nearest(p)
            .expect("a GPU vertex with no CPU vertex nearby");
        assert!(!taken[j], "two GPU vertices on one CPU vertex");
        taken[j] = true;
        twin[i] = j;
        worst = worst.max(d);
        let (a, b) = (gpu.colors[i], cpu.colors[j]);
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.abs_diff(*y) <= 1),
            "colours {a:?} / {b:?}"
        );
    }
    eprintln!("one view: worst vertex offset {:.2e} voxels", worst / voxel);
    assert!(worst < 1e-3 * voxel, "vertices off by {worst}");
    let identity: Vec<usize> = (0..cpu.vertices.len()).collect();
    assert_eq!(
        triangles_under(&gpu, &twin),
        triangles_under(&cpu, &identity),
        "triangles differ"
    );
}

/// Slabs only bound memory: every block sees every view in the same order
/// whatever slab it is fused in, so a tiny slab budget (many slabs, cells on
/// every slab face) gives exactly the surface of one slab. Against the CPU,
/// which fuses each block only from the view that first touched it on, the
/// surface agrees where the views agree.
#[test]
fn slabs_do_not_change_the_surface() {
    let Some(mut gs) = gsplat() else { return };
    let scene = wall();
    gs.set_scene(&PackedScene::from_scene(&scene))
        .expect("upload");
    let vs: Vec<CameraView> = (0..4)
        .map(|k| view(-0.15 + 0.1 * k as f32, -0.6 + 0.4 * k as f32))
        .collect();
    let views: Vec<&CameraView> = vs.iter().collect();
    let voxel = 0.03;
    let params = |slab_blocks, cache_mb| TsdfParams {
        voxel,
        trunc: 4.0 * voxel,
        max_depth: 20.0,
        min_weight: 1.0,
        carve: 1,
        free_space: 1,
        planar: 0,
        sh_active_degree: 0,
        slab_blocks,
        cache_mb,
        expected: 0,
    };
    let one = gs.tsdf_surface(&views, &params(0, 64)).expect("one slab");
    let many = gs.tsdf_surface(&views, &params(3, 64)).expect("many slabs");
    // Without the view cache every slab renders the views again.
    let rendered = gs.tsdf_surface(&views, &params(3, 0)).expect("no cache");
    eprintln!(
        "{} blocks: {} slab(s) / {} slabs; {} vertices, {} edges",
        one.blocks,
        one.slabs,
        many.slabs,
        one.cells.len(),
        one.edges.len()
    );
    assert_eq!(one.slabs, 1);
    assert!(many.slabs > 10, "{} slabs", many.slabs);
    let vertices = |s: &visloc_gsplat_cuda::TsdfSurface| {
        let mut v: Vec<([i32; 3], [u32; 3], [u8; 3])> = (0..s.cells.len())
            .map(|i| (s.cells[i], s.positions[i].map(f32::to_bits), s.colors[i]))
            .collect();
        v.sort_unstable();
        v
    };
    let edges = |s: &visloc_gsplat_cuda::TsdfSurface| {
        let mut e = s.edges.clone();
        e.sort_unstable();
        e
    };
    assert!(one.cells.len() > 1000, "too small a surface to compare");
    assert_eq!(vertices(&one), vertices(&many), "vertices differ");
    assert_eq!(edges(&one), edges(&many), "edges differ");
    assert_eq!(
        vertices(&one),
        vertices(&rendered),
        "vertices differ without the cache"
    );
    assert_eq!(
        edges(&one),
        edges(&rendered),
        "edges differ without the cache"
    );

    let gpu = surface_mesh(&one);
    let (cpu, _) = extract_mesh(&mut gs, 0, MeshDepth::Median, &views, &[], &options(voxel))
        .expect("cpu mesh");
    let grid = Grid::new(&cpu.vertices, voxel);
    let mut within = [0usize; 3];
    for &p in &gpu.vertices {
        let Some((_, d)) = grid.nearest(p) else {
            continue;
        };
        for (n, tol) in within.iter_mut().zip([0.01, 0.1, 0.5]) {
            *n += usize::from(d < tol * voxel);
        }
    }
    let frac = within.map(|n| n as f64 / gpu.vertices.len() as f64);
    eprintln!(
        "several views: {} gpu / {} cpu vertices, {} / {} triangles; within 0.01 / 0.1 / 0.5 voxel of a cpu vertex: {:.1}% / {:.1}% / {:.1}%",
        gpu.vertices.len(),
        cpu.vertices.len(),
        gpu.triangles.len(),
        cpu.triangles.len(),
        100.0 * frac[0],
        100.0 * frac[1],
        100.0 * frac[2]
    );
    assert!(
        frac[2] > 0.95,
        "only {:.1}% of the vertices agree with the CPU",
        100.0 * frac[2]
    );
}
