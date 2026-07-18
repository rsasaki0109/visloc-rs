//! `#[ignore]`-gated parity test for [`visloc_slam::dpvo_ba`] against
//! `E:/visloc_archive/dpvo_onnx_m1/fixtures/ba_fixture.npz` — Milestone M3 of
//! `docs/dpvo_droid_port_plan.md`. See `pipelines/slam/src/dpvo_patch_ba.rs`'s
//! module doc for the full provenance/convention writeup; this file only
//! loads the fixture and compares.
//!
//! # Why this file has its own tiny `.npz` reader
//!
//! `crates/vision/src/dpvo/npz.rs` already implements exactly this format
//! (uncompressed-ZIP-of-`.npy`), but it lives inside `crates/vision/src/dpvo`,
//! a module gated **whole** behind the `onnx-inference` feature (see that
//! module's own doc comment). `visloc-slam`'s `Cargo.toml` has no
//! `onnx-inference` feature at all — and the task's own placement constraint
//! for M3 ("must NOT be gated behind onnx-inference") means this crate must
//! stay that way. Rather than add a feature-gated dependency edge from a
//! math-only crate onto the ONNX-inference crate just to read one `.npz`
//! fixture in one `#[ignore]`-gated test, this file carries a small,
//! test-only, read-only copy of the same minimal parser (same design: only
//! `ZIP_STORED` entries, only little-endian `<f4`/`<i8` `.npy` payloads —
//! everything `numpy.savez` ever produces and everything these fixtures use).
//! This is intentional duplication of ~120 lines of single-purpose parsing
//! code to keep the production crate boundary clean, not an oversight.

use std::fs;
use std::path::Path;

use nalgebra::Vector2;
use visloc_core::geometry::SE3;
use visloc_slam::{
    dpvo_ba, se3_from_dpvo_pose, DpvoBaConfig, DpvoBaProblem, DpvoEdge, DpvoIntrinsics, DpvoPatch,
};

const FIXTURE_PATH: &str =
    "E:/visloc_archive/dpvo_onnx_m1/fixtures/ba_fixture.npz";

// ---------------------------------------------------------------------------
// Minimal read-only `.npz` (uncompressed ZIP of `.npy`) reader — see the
// module doc above for why this is a small test-local duplicate rather than
// a dependency on `visloc_vision::dpvo::npz`.
// ---------------------------------------------------------------------------

struct NpzArchive {
    bytes: Vec<u8>,
}

#[derive(Debug)]
enum NpyArray {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    I64 { shape: Vec<usize>, data: Vec<i64> },
}

impl NpzArchive {
    fn open(path: impl AsRef<Path>) -> Self {
        let bytes = fs::read(path.as_ref())
            .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.as_ref().display()));
        Self { bytes }
    }

    fn read_f32(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        match self.read_array(name) {
            NpyArray::F32 { shape, data } => (shape, data),
            NpyArray::I64 { .. } => panic!("{name}: expected <f4, found <i8"),
        }
    }

    fn read_i64(&self, name: &str) -> (Vec<usize>, Vec<i64>) {
        match self.read_array(name) {
            NpyArray::I64 { shape, data } => (shape, data),
            NpyArray::F32 { .. } => panic!("{name}: expected <i8, found <f4"),
        }
    }

    fn read_array(&self, name: &str) -> NpyArray {
        let entry_name = format!("{name}.npy");
        let data = self.locate_entry(&entry_name);
        parse_npy(data)
    }

    fn locate_entry(&self, entry_name: &str) -> &[u8] {
        let eocd = find_eocd(&self.bytes);
        let central_dir_offset =
            u32::from_le_bytes(self.bytes[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
        let central_dir_count =
            u16::from_le_bytes(self.bytes[eocd + 10..eocd + 12].try_into().unwrap()) as usize;

        let mut cursor = central_dir_offset;
        for _ in 0..central_dir_count {
            let record = &self.bytes[cursor..];
            let signature = u32::from_le_bytes(record[0..4].try_into().unwrap());
            assert_eq!(signature, 0x0201_4b50, "expected central directory signature");
            let compression_method = u16::from_le_bytes(record[10..12].try_into().unwrap());
            let compressed_size = u32::from_le_bytes(record[20..24].try_into().unwrap()) as usize;
            let file_name_len = u16::from_le_bytes(record[28..30].try_into().unwrap()) as usize;
            let extra_len = u16::from_le_bytes(record[30..32].try_into().unwrap()) as usize;
            let comment_len = u16::from_le_bytes(record[32..34].try_into().unwrap()) as usize;
            let local_header_offset =
                u32::from_le_bytes(record[42..46].try_into().unwrap()) as usize;
            let file_name = &record[46..46 + file_name_len];

            if file_name == entry_name.as_bytes() {
                assert_eq!(compression_method, 0, "only ZIP_STORED fixtures are supported");
                return self.read_stored_entry(local_header_offset, compressed_size);
            }
            cursor += 46 + file_name_len + extra_len + comment_len;
        }
        panic!("entry {entry_name} not found in fixture archive");
    }

    fn read_stored_entry(&self, local_header_offset: usize, compressed_size: usize) -> &[u8] {
        let header = &self.bytes[local_header_offset..];
        let signature = u32::from_le_bytes(header[0..4].try_into().unwrap());
        assert_eq!(signature, 0x0403_4b50, "expected local file header signature");
        let file_name_len = u16::from_le_bytes(header[26..28].try_into().unwrap()) as usize;
        let extra_len = u16::from_le_bytes(header[28..30].try_into().unwrap()) as usize;
        let data_offset = local_header_offset + 30 + file_name_len + extra_len;
        &self.bytes[data_offset..data_offset + compressed_size]
    }
}

fn find_eocd(bytes: &[u8]) -> usize {
    let search_start = bytes.len().saturating_sub(22 + 65535);
    for offset in (search_start..=bytes.len() - 22).rev() {
        if u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) == 0x0605_4b50 {
            return offset;
        }
    }
    panic!("End-Of-Central-Directory record not found");
}

fn parse_npy(bytes: &[u8]) -> NpyArray {
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "missing npy magic");
    let major_version = bytes[6];
    let (header_len, header_start) = if major_version == 1 {
        (u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize, 10)
    } else {
        (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
    };
    let header_end = header_start + header_len;
    let header_str = std::str::from_utf8(&bytes[header_start..header_end]).unwrap();

    let descr = header_str
        .split_once("'descr':")
        .and_then(|(_, rest)| rest.split_once('\''))
        .and_then(|(_, rest)| rest.split_once('\''))
        .map(|(value, _)| value)
        .expect("missing descr field");
    assert!(!header_str.contains("'fortran_order': True"), "fortran order not supported");
    let shape_str = header_str
        .split_once("'shape':")
        .and_then(|(_, rest)| rest.split_once('('))
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(value, _)| value)
        .expect("missing shape field");
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();

    let data_bytes = &bytes[header_end..];
    let element_count: usize = shape.iter().product::<usize>().max(if shape.is_empty() { 1 } else { 0 });

    match descr {
        "<f4" => {
            let data = data_bytes[..element_count * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            NpyArray::F32 { shape, data }
        }
        "<i8" => {
            let data = data_bytes[..element_count * 8]
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            NpyArray::I64 { shape, data }
        }
        other => panic!("unsupported dtype {other}"),
    }
}

// ---------------------------------------------------------------------------
// Fixture -> DpvoBaProblem loading.
// ---------------------------------------------------------------------------

fn load_poses(archive: &NpzArchive, key: &str) -> Vec<SE3> {
    let (shape, data) = archive.read_f32(key);
    assert_eq!(shape.len(), 3, "{key}: expected (1, n_frames, 7)");
    assert_eq!(shape[0], 1);
    assert_eq!(shape[2], 7);
    let n_frames = shape[1];
    (0..n_frames)
        .map(|f| {
            let mut row = [0.0f64; 7];
            for c in 0..7 {
                row[c] = data[f * 7 + c] as f64;
            }
            se3_from_dpvo_pose(row)
        })
        .collect()
}

/// Load `patches_in`-shaped `(1, n_patches, 3, P, P)` data into
/// `(x, y, inverse_depth)` triples, asserting the `P x P` replica is uniform
/// (see `dpvo_patch_ba.rs`'s module doc: only the center pixel of DPVO's
/// patch grid ever enters `BA()`'s math, and `dump_ba_fixture` broadcasts the
/// same scalar across the whole grid when constructing this fixture).
fn load_patches(archive: &NpzArchive, key: &str) -> Vec<DpvoPatch> {
    let (shape, data) = archive.read_f32(key);
    assert_eq!(shape.len(), 5, "{key}: expected (1, n_patches, 3, P, P)");
    let (n_patches, p) = (shape[1], shape[3]);
    assert_eq!(shape[2], 3);
    assert_eq!(shape[4], p);
    let stride_channel = p * p;
    let stride_patch = 3 * stride_channel;
    (0..n_patches)
        .map(|patch_index| {
            let base = patch_index * stride_patch;
            let channel = |c: usize| -> f64 {
                let values = &data[base + c * stride_channel..base + (c + 1) * stride_channel];
                let first = values[0] as f64;
                for &v in values {
                    assert!(
                        ((v as f64) - first).abs() < 1e-6,
                        "{key} patch {patch_index} channel {c}: non-uniform P x P replica \
                         ({v} vs {first}) -- BA only consumes the center pixel"
                    );
                }
                first
            };
            DpvoPatch {
                x: channel(0),
                y: channel(1),
                inverse_depth: channel(2),
            }
        })
        .collect()
}

fn load_intrinsics(archive: &NpzArchive, key: &str) -> Vec<DpvoIntrinsics> {
    let (shape, data) = archive.read_f32(key);
    assert_eq!(shape.len(), 3, "{key}: expected (1, n_frames, 4)");
    let n_frames = shape[1];
    (0..n_frames)
        .map(|f| DpvoIntrinsics {
            fx: data[f * 4] as f64,
            fy: data[f * 4 + 1] as f64,
            cx: data[f * 4 + 2] as f64,
            cy: data[f * 4 + 3] as f64,
        })
        .collect()
}

fn load_vec2(archive: &NpzArchive, key: &str) -> Vec<Vector2<f64>> {
    let (shape, data) = archive.read_f32(key);
    assert_eq!(shape.len(), 3, "{key}: expected (1, n_edges, 2)");
    let n_edges = shape[1];
    (0..n_edges)
        .map(|e| Vector2::new(data[e * 2] as f64, data[e * 2 + 1] as f64))
        .collect()
}

fn load_edges(archive: &NpzArchive) -> Vec<DpvoEdge> {
    let (_, ii) = archive.read_i64("ii");
    let (_, jj) = archive.read_i64("jj");
    let (_, kk) = archive.read_i64("kk");
    assert_eq!(ii.len(), jj.len());
    assert_eq!(ii.len(), kk.len());
    (0..ii.len())
        .map(|e| DpvoEdge {
            i: ii[e] as usize,
            j: jj[e] as usize,
            k: kk[e] as usize,
        })
        .collect()
}

fn max_abs_pose_diff(a: &[SE3], b: &[SE3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let dt = (x.translation - y.translation).norm();
            let dr = (x.rotation.to_rotation_matrix().into_inner()
                - y.rotation.to_rotation_matrix().into_inner())
            .abs()
            .max();
            dt.max(dr)
        })
        .fold(0.0_f64, f64::max)
}

fn max_abs_depth_diff(a: &[DpvoPatch], b: &[DpvoPatch]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.inverse_depth - y.inverse_depth).abs())
        .fold(0.0_f64, f64::max)
}

/// Parity target: `ba_fixture.npz`'s `poses_in`/`patches_in` run through
/// [`dpvo_ba`] with `config.iterations = 2` (matching `dump_ba_fixture`'s own
/// two chained `mini_ba` calls, `export_dpvo_onnx.py:745-746`) must match
/// `poses_after_iter2`/`patches_after_iter2` to a tight tolerance.
///
/// Tolerance: `1e-4` on the pose tangent (translation in meters, rotation
/// matrix entries dimensionless) and on relative inverse depth, per the
/// task's own target. This is fp32-PyTorch-forward vs f64-Rust-forward: the
/// fixture's *inputs* are `<f4` (fp32) and its *outputs* were produced by
/// fp32 PyTorch arithmetic (`mini_ba` runs in `torch.float32` throughout,
/// `export_dpvo_onnx.py` never casts to `float64` for the BA fixture itself
/// -- only the separate Jacobian finite-difference self-check upconverts).
/// This Rust port computes entirely in `f64`. The two therefore differ by
/// accumulated fp32 rounding over ~2 Gauss-Newton iterations each touching a
/// `12x12` dense solve, a handful of `6x6`/`2x6` matrix products, and a
/// Schur complement -- comfortably inside `1e-4` for a well-conditioned
/// (`ep=100`-damped) tiny system, matching this repo's own established
/// fp32-vs-f64/CPU-vs-reference parity bars elsewhere (M1/M2's `1e-4`
/// max-abs-diff threshold, `docs/dpvo_droid_port_plan.md`).
#[test]
#[ignore]
fn ba_fixture_two_iterations_matches_reference_within_1e_4() {
    let archive = NpzArchive::open(FIXTURE_PATH);

    let poses_in = load_poses(&archive, "poses_in");
    let patches_in = load_patches(&archive, "patches_in");
    let intrinsics = load_intrinsics(&archive, "intrinsics");
    let targets = load_vec2(&archive, "target");
    let weights = load_vec2(&archive, "weight");
    let edges = load_edges(&archive);
    let (lmbda_shape, lmbda_data) = archive.read_f32("lmbda");
    assert!(lmbda_shape.is_empty(), "lmbda expected to be a scalar in this fixture");
    let lmbda = lmbda_data[0] as f64;
    let (_, bounds_data) = archive.read_f32("bounds");
    let bounds = [
        bounds_data[0] as f64,
        bounds_data[1] as f64,
        bounds_data[2] as f64,
        bounds_data[3] as f64,
    ];

    let poses_after_iter2 = load_poses(&archive, "poses_after_iter2");
    let patches_after_iter2 = load_patches(&archive, "patches_after_iter2");

    let problem = DpvoBaProblem {
        poses: poses_in,
        patches: patches_in,
        intrinsics,
        edges,
        targets,
        weights,
        depth_damping: None,
    };
    let config = DpvoBaConfig {
        iterations: 2,
        fixedp: 1, // dump_ba_fixture's mini_ba calls use fixedp=1 (export_dpvo_onnx.py:745-746)
        lmbda,
        ep: 100.0, // mini_ba's own default, not overridden by dump_ba_fixture
        bounds,
    };

    // Rough per-primitive timing (release-build only, see the module doc's
    // "M2 results" precedent in docs/dpvo_droid_port_plan.md for why debug
    // numbers are not representative): repeat the full 2-iteration solve on
    // this fixture's own tiny (3 frames / 2 patches / 4 edges) problem size.
    let warmup = dpvo_ba(&problem, &config).expect("fixture problem is well-posed");
    let repeats = 2_000;
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        std::hint::black_box(dpvo_ba(&problem, &config).expect("fixture problem is well-posed"));
    }
    let elapsed = start.elapsed();
    println!(
        "[dpvo_ba fixture timing] {:.6} ms/call ({} repeats, 2 GN iterations, 3 frames / 2 patches / 4 edges)",
        elapsed.as_secs_f64() * 1000.0 / repeats as f64,
        repeats
    );

    let solved = warmup;
    let pose_diff = max_abs_pose_diff(&solved.poses, &poses_after_iter2);
    let depth_diff = max_abs_depth_diff(&solved.patches, &patches_after_iter2);

    println!(
        "[dpvo_ba fixture parity] max abs pose (translation/rotation-matrix) diff = {pose_diff:.6e}, \
         max abs inverse-depth diff = {depth_diff:.6e} (threshold 1e-4)"
    );

    assert!(pose_diff < 1e-4, "pose parity failed: max abs diff = {pose_diff:.6e}");
    assert!(depth_diff < 1e-4, "depth parity failed: max abs diff = {depth_diff:.6e}");
}

/// Same fixture, but only through one Gauss-Newton iteration
/// (`poses_after_iter1`/`patches_after_iter1`) -- an intermediate checkpoint
/// that isolates a single `dpvo_ba_step` call from the "two calls chained"
/// composition the primary test above exercises.
#[test]
#[ignore]
fn ba_fixture_one_iteration_matches_reference_within_1e_4() {
    let archive = NpzArchive::open(FIXTURE_PATH);

    let poses_in = load_poses(&archive, "poses_in");
    let patches_in = load_patches(&archive, "patches_in");
    let intrinsics = load_intrinsics(&archive, "intrinsics");
    let targets = load_vec2(&archive, "target");
    let weights = load_vec2(&archive, "weight");
    let edges = load_edges(&archive);
    let (_, lmbda_data) = archive.read_f32("lmbda");
    let lmbda = lmbda_data[0] as f64;
    let (_, bounds_data) = archive.read_f32("bounds");
    let bounds = [
        bounds_data[0] as f64,
        bounds_data[1] as f64,
        bounds_data[2] as f64,
        bounds_data[3] as f64,
    ];

    let poses_after_iter1 = load_poses(&archive, "poses_after_iter1");
    let patches_after_iter1 = load_patches(&archive, "patches_after_iter1");

    let problem = DpvoBaProblem {
        poses: poses_in,
        patches: patches_in,
        intrinsics,
        edges,
        targets,
        weights,
        depth_damping: None,
    };
    let config = DpvoBaConfig {
        iterations: 1,
        fixedp: 1,
        lmbda,
        ep: 100.0,
        bounds,
    };

    let solved = visloc_slam::dpvo_ba_step(&problem, &config).expect("fixture problem is well-posed");

    let pose_diff = max_abs_pose_diff(&solved.poses, &poses_after_iter1);
    let depth_diff = max_abs_depth_diff(&solved.patches, &patches_after_iter1);

    println!(
        "[dpvo_ba_step fixture parity, iter 1] max abs pose diff = {pose_diff:.6e}, \
         max abs inverse-depth diff = {depth_diff:.6e} (threshold 1e-4)"
    );

    assert!(pose_diff < 1e-4, "pose parity failed (iter 1): max abs diff = {pose_diff:.6e}");
    assert!(depth_diff < 1e-4, "depth parity failed (iter 1): max abs diff = {depth_diff:.6e}");
}
