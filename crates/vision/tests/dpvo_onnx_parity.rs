//! Fixture-based parity tests for Milestone M2 of
//! `docs/dpvo_droid_port_plan.md`'s DPVO port.
//!
//! Every test in this file is `#[ignore]`-gated: they all read `.npz`/`.onnx`
//! artifacts from `E:/visloc_archive/dpvo_onnx_m1` (M1's output — see that
//! plan doc section, and `scripts/export_dpvo_onnx.py`/
//! `scripts/check_dpvo_onnx_parity.py`), which are **not** part of this repo
//! (git-ignored, machine-local, regenerable from a DPVO checkpoint) and will
//! not exist in CI or on a fresh checkout. Run explicitly:
//!
//! ```text
//! ORT_DYLIB_PATH=E:/tools/onnxruntime-win-x64-1.23.2/lib/onnxruntime.dll \
//!   cargo test -p visloc-vision --features onnx-inference \
//!   --test dpvo_onnx_parity -- --ignored
//! ```
//!
//! The fixture directory can be overridden via the `DPVO_ONNX_M1_DIR`
//! environment variable (defaults to `E:/visloc_archive/dpvo_onnx_m1`,
//! matching every other reference to this path in this repo's docs/scripts
//! — not hardcoded into any *library* code, only into this test's default,
//! per the task's constraint).
//!
//! All numeric comparisons use max-abs-diff against a `1e-4` threshold,
//! matching `scripts/check_dpvo_onnx_parity.py`'s own PASS/FAIL convention
//! (see that script and the plan doc's M1 results for why max-abs-diff,
//! not max-rel-diff, is the metric that governs pass/fail here — relative
//! diff is meaningless on the near-saturated sigmoid `weight` output).

#![cfg(feature = "onnx-inference")]

use std::path::PathBuf;

use visloc_vision::dpvo::correlation::corr_cpu;
use visloc_vision::dpvo::native_cuda_correlation::NativeCudaCorrelation;
use visloc_vision::dpvo::npz::NpzArchive;
use visloc_vision::dpvo::onnx_session::DpvoOnnxSession;
use visloc_vision::dpvo::patchify::patchify_cpu;
use visloc_vision::dpvo::softagg::SoftAgg;
use visloc_vision::features::superpoint_onnx::OnnxBackend;

const PASS_THRESHOLD: f32 = 1e-4;

fn fixtures_dir() -> PathBuf {
    let base = std::env::var("DPVO_ONNX_M1_DIR")
        .unwrap_or_else(|_| "E:/visloc_archive/dpvo_onnx_m1".to_string());
    PathBuf::from(base).join("fixtures")
}

fn onnx_dir() -> PathBuf {
    let base = std::env::var("DPVO_ONNX_M1_DIR")
        .unwrap_or_else(|_| "E:/visloc_archive/dpvo_onnx_m1".to_string());
    PathBuf::from(base)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Rough `Instant`-based per-stage CPU timing, reported alongside the
/// parity numbers into `docs/dpvo_droid_port_plan.md`'s "M2 results"
/// section. Deliberately not a rigorous benchmark (no warmup-then-discard
/// separation, no criterion-style statistics) — just enough repeats to
/// smooth out scheduler noise on numbers that are themselves only meant to
/// be a rough order-of-magnitude sense of cost on this fixture's shapes,
/// consistent with the plan doc's own §3 CPU-feasibility estimates (which
/// this is the first chance to check against a real measurement).
fn time_repeated<T>(label: &str, repeats: u32, mut f: impl FnMut() -> T) -> T {
    let mut last = None;
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        last = Some(f());
    }
    let elapsed = start.elapsed();
    println!(
        "  [timing] {label}: {:.3} ms/call (avg over {repeats} calls)",
        elapsed.as_secs_f64() * 1000.0 / repeats as f64
    );
    last.expect("repeats > 0")
}

/// Scope note (see `crates/vision/src/dpvo/patchify.rs`'s module doc): this
/// checks agreement with `export_dpvo_onnx.py`'s own `patchify_cpu`
/// reference reimplementation, not with upstream's CUDA kernel (no pure
/// Python/CUDA-free reference exists upstream for this op at all).
#[test]
#[ignore = "reads E:/visloc_archive/dpvo_onnx_m1/fixtures — not present in CI, see module doc"]
fn patchify_parity_against_fixture() {
    let archive = NpzArchive::open(fixtures_dir().join("patchify_fixture.npz")).expect(
        "open patchify_fixture.npz (regenerate via scripts/export_dpvo_onnx.py if missing)",
    );

    let (fmap_shape, fmap_data) = archive.read_f32("fmap").unwrap();
    assert_eq!(fmap_shape.len(), 4, "fmap shape: {fmap_shape:?}");
    let (_, channels, height, width) = (fmap_shape[0], fmap_shape[1], fmap_shape[2], fmap_shape[3]);
    let fmap = ndarray::Array3::from_shape_vec((channels, height, width), fmap_data).unwrap();

    let (coords_shape, coords_data) = archive.read_f32("coords").unwrap();
    assert_eq!(coords_shape, vec![1, coords_shape[1], 2]);
    let num_patches = coords_shape[1];
    let coords: Vec<(f32, f32)> = coords_data
        .chunks_exact(2)
        .map(|xy| (xy[0], xy[1]))
        .collect();
    assert_eq!(coords.len(), num_patches);

    let (radius_shape, radius_data) = archive.read_i64("radius").unwrap();
    assert!(
        radius_shape.is_empty(),
        "radius should be a scalar, got shape {radius_shape:?}"
    );
    let radius = radius_data[0] as usize;

    let (expected_shape, expected_data) = archive.read_f32("patches").unwrap();
    assert_eq!(
        expected_shape,
        vec![1, num_patches, channels, 2 * radius + 1, 2 * radius + 1]
    );

    let got = time_repeated("patchify_cpu", 200, || {
        patchify_cpu(fmap.view(), &coords, radius)
    });
    assert_eq!(got.shape(), &expected_shape[1..]);

    let got_flat: Vec<f32> = got.iter().copied().collect();
    let diff = max_abs_diff(&got_flat, &expected_data);
    println!("patchify_cpu max abs diff vs fixture: {diff:.3e}");
    assert!(
        diff <= PASS_THRESHOLD,
        "patchify_cpu parity FAILED: max abs diff {diff:.3e}"
    );
}

/// The fixture is generated by the PyTorch port of upstream's directly
/// inspected CUDA correlation contract; native CUDA parity is checked again
/// below, including raw scale and upstream's flattened axis order.
#[test]
#[ignore = "reads E:/visloc_archive/dpvo_onnx_m1/fixtures — not present in CI, see module doc"]
fn correlation_parity_against_fixture() {
    let archive = NpzArchive::open(fixtures_dir().join("correlation_fixture.npz")).expect(
        "open correlation_fixture.npz (regenerate via scripts/export_dpvo_onnx.py if missing)",
    );

    let (anchor_shape, anchor_data) = archive.read_f32("anchor_patch_feats").unwrap();
    assert_eq!(anchor_shape.len(), 4);
    let (num_edges, channels, patch, patch2) = (
        anchor_shape[0],
        anchor_shape[1],
        anchor_shape[2],
        anchor_shape[3],
    );
    assert_eq!(patch, patch2);
    let anchor =
        ndarray::Array4::from_shape_vec((num_edges, channels, patch, patch), anchor_data).unwrap();

    let (target_shape, target_data) = archive.read_f32("target_fmap").unwrap();
    assert_eq!(
        target_shape[0], 1,
        "fixture's target_fmap keeps an unexpanded batch dim of 1"
    );
    let (t_channels, t_height, t_width) = (target_shape[1], target_shape[2], target_shape[3]);
    assert_eq!(t_channels, channels);
    let target =
        ndarray::Array3::from_shape_vec((t_channels, t_height, t_width), target_data).unwrap();

    let (center_shape, center_data) = archive.read_f32("coords_center").unwrap();
    assert_eq!(center_shape, vec![num_edges, patch, patch, 2]);
    let coords_center =
        ndarray::Array4::from_shape_vec((num_edges, patch, patch, 2), center_data).unwrap();

    let (radius_shape, radius_data) = archive.read_i64("radius").unwrap();
    assert!(radius_shape.is_empty());
    let radius = radius_data[0] as usize;
    let taps = 2 * radius + 1;

    let (expected_shape, expected_data) = archive.read_f32("corr_out").unwrap();
    assert_eq!(expected_shape, vec![num_edges, patch, patch, taps * taps]);

    let got = time_repeated("corr_cpu", 200, || {
        corr_cpu(anchor.view(), target.view(), coords_center.view(), radius)
    });
    assert_eq!(got.shape(), expected_shape.as_slice());

    let got_flat: Vec<f32> = got.iter().copied().collect();
    let diff = max_abs_diff(&got_flat, &expected_data);
    println!("corr_cpu max abs diff vs fixture: {diff:.3e}");
    assert!(
        diff <= PASS_THRESHOLD,
        "corr_cpu parity FAILED: max abs diff {diff:.3e}"
    );
}

/// Loads the two SoftAgg instances (`agg_kk`, `agg_ij`) from
/// `softagg_weights_fixture.npz` and checks that
/// `net_pre_agg + agg_kk(net_pre_agg, kk) + agg_ij(net_pre_agg, ii*12345+jj)`
/// matches `update_cell_fixture.npz`'s own stored `net_post_agg` — the
/// real-weight ground truth the plan doc's M1 results already validated
/// end-to-end in Python (`check_dpvo_onnx_parity.py`'s "host-side SoftAgg
/// step" section). This is the piece M1 flagged as needing a from-scratch
/// Rust port with no ONNX equivalent; see the `dpvo` module doc for the
/// weight-export gap this test's fixture had to close in M2.
#[test]
#[ignore = "reads E:/visloc_archive/dpvo_onnx_m1/fixtures — not present in CI, see module doc"]
fn softagg_parity_against_fixture() {
    let update_fixture = NpzArchive::open(fixtures_dir().join("update_cell_fixture.npz"))
        .expect("open update_cell_fixture.npz");
    let weights_fixture = NpzArchive::open(fixtures_dir().join("softagg_weights_fixture.npz"))
        .expect(
            "open softagg_weights_fixture.npz (regenerate via scripts/export_dpvo_onnx.py; \
             this fixture was added in M2 — see the dpvo module doc)",
        );

    let (net_pre_shape, net_pre_data) = update_fixture.read_f32("net_pre_agg").unwrap();
    assert_eq!(net_pre_shape.len(), 3);
    let (_, num_edges, dim) = (net_pre_shape[0], net_pre_shape[1], net_pre_shape[2]);
    let net_pre_agg_2d = ndarray::Array2::from_shape_vec((num_edges, dim), net_pre_data).unwrap();

    let (_, kk) = update_fixture.read_i64("kk").unwrap();
    let (_, ii) = update_fixture.read_i64("ii").unwrap();
    let (_, jj) = update_fixture.read_i64("jj").unwrap();

    let (expected_shape, expected_data) = update_fixture.read_f32("net_post_agg").unwrap();
    assert_eq!(expected_shape, vec![1, num_edges, dim]);

    let agg_kk = SoftAgg::load_from_npz(&weights_fixture, "agg_kk_").expect("load agg_kk weights");
    let agg_ij = SoftAgg::load_from_npz(&weights_fixture, "agg_ij_").expect("load agg_ij weights");

    let pair_key: Vec<i64> = ii
        .iter()
        .zip(jj.iter())
        .map(|(&i, &j)| i * 12345 + j)
        .collect();
    let net_post_agg_2d = time_repeated("SoftAgg host step (agg_kk + agg_ij)", 200, || {
        let agg_kk_out = agg_kk.forward(net_pre_agg_2d.view(), &kk);
        let agg_ij_out = agg_ij.forward(net_pre_agg_2d.view(), &pair_key);
        &net_pre_agg_2d + &agg_kk_out + &agg_ij_out
    });
    let got_flat: Vec<f32> = net_post_agg_2d.iter().copied().collect();
    let diff = max_abs_diff(&got_flat, &expected_data);
    println!("SoftAgg host step max abs diff vs fixture: {diff:.3e}");
    assert!(
        diff <= PASS_THRESHOLD,
        "SoftAgg parity FAILED: max abs diff {diff:.3e}"
    );
}

/// End-to-end: `dpvo_update_pre_agg.onnx` → host [`SoftAgg`] →
/// `dpvo_update_post_agg.onnx`, exercised through
/// [`DpvoOnnxSession::update_iteration`] — the full "one GRU update
/// iteration" M2 was scoped to deliver — checked against every stage
/// `update_cell_fixture.npz` recorded (`net_pre_agg`, `net_post_agg`,
/// `net_out`, `delta`, `weight`), not just the final output, so a
/// regression in any one of the three stages (pre-agg graph, host SoftAgg,
/// post-agg graph) is independently localized instead of only surfacing as
/// "the end-to-end number is off".
///
/// Requires `ORT_DYLIB_PATH` pointing at an ONNX Runtime 1.23.x shared
/// library (see this file's module doc for the exact invocation).
#[test]
#[ignore = "reads E:/visloc_archive/dpvo_onnx_m1 (.onnx + fixtures) and needs ORT_DYLIB_PATH — \
            not present in CI, see module doc"]
fn update_cell_end_to_end_parity_against_fixture() {
    let fixtures = fixtures_dir();
    let onnx = onnx_dir();

    let update_fixture =
        NpzArchive::open(fixtures.join("update_cell_fixture.npz")).expect("open fixture");
    let weights_fixture = NpzArchive::open(fixtures.join("softagg_weights_fixture.npz"))
        .expect("open softagg weights fixture");

    let (net_shape, net_data) = update_fixture.read_f32("net").unwrap();
    let (_, num_edges, dim) = (net_shape[0], net_shape[1], net_shape[2]);
    let net = ndarray::Array3::from_shape_vec((1, num_edges, dim), net_data).unwrap();
    let (_, inp_data) = update_fixture.read_f32("inp").unwrap();
    let inp = ndarray::Array3::from_shape_vec((1, num_edges, dim), inp_data).unwrap();
    let (corr_shape, corr_data) = update_fixture.read_f32("corr").unwrap();
    let corr_dim = corr_shape[2];
    let corr = ndarray::Array3::from_shape_vec((1, num_edges, corr_dim), corr_data).unwrap();

    let (_, kk) = update_fixture.read_i64("kk").unwrap();
    let (_, ii) = update_fixture.read_i64("ii").unwrap();
    let (_, jj) = update_fixture.read_i64("jj").unwrap();
    let (_, expected_ix) = update_fixture.read_i64("ix").unwrap();
    let (_, expected_jx) = update_fixture.read_i64("jx").unwrap();

    // `neighbors_cpu` (exercised indirectly via `update_iteration` below)
    // must reproduce the fixture's own `ix`/`jx` — check this explicitly
    // first so a mismatch here is not confused with an ONNX/SoftAgg bug.
    let (got_ix, got_jx) = visloc_vision::dpvo::softagg::neighbors_cpu(&kk, &jj);
    assert_eq!(got_ix, expected_ix, "neighbors_cpu ix mismatch");
    assert_eq!(got_jx, expected_jx, "neighbors_cpu jx mismatch");

    let agg_kk = SoftAgg::load_from_npz(&weights_fixture, "agg_kk_").expect("load agg_kk");
    let agg_ij = SoftAgg::load_from_npz(&weights_fixture, "agg_ij_").expect("load agg_ij");

    let session = DpvoOnnxSession::load_from_paths(
        onnx.join("fnet.onnx"),
        onnx.join("inet.onnx"),
        onnx.join("dpvo_update_pre_agg.onnx"),
        onnx.join("dpvo_update_post_agg.onnx"),
    )
    .expect("load the four DPVO ONNX graphs (check ORT_DYLIB_PATH and the onnx dir)");

    let (net_out, delta, weight) = time_repeated("update_iteration (end-to-end)", 20, || {
        session
            .update_iteration(
                net.view(),
                inp.view(),
                corr.view(),
                &kk,
                &ii,
                &jj,
                &agg_kk,
                &agg_ij,
            )
            .expect("update_iteration succeeds")
    });

    let (expected_out_shape, expected_net_out) = update_fixture.read_f32("net_out").unwrap();
    assert_eq!(net_out.shape(), expected_out_shape.as_slice());
    let diff_net_out = max_abs_diff(
        &net_out.iter().copied().collect::<Vec<_>>(),
        &expected_net_out,
    );

    let (expected_delta_shape, expected_delta) = update_fixture.read_f32("delta").unwrap();
    assert_eq!(delta.shape(), expected_delta_shape.as_slice());
    let diff_delta = max_abs_diff(&delta.iter().copied().collect::<Vec<_>>(), &expected_delta);

    let (expected_weight_shape, expected_weight) = update_fixture.read_f32("weight").unwrap();
    assert_eq!(weight.shape(), expected_weight_shape.as_slice());
    let diff_weight = max_abs_diff(
        &weight.iter().copied().collect::<Vec<_>>(),
        &expected_weight,
    );

    println!(
        "update_iteration max abs diff vs fixture: net_out={diff_net_out:.3e} \
         delta={diff_delta:.3e} weight={diff_weight:.3e}"
    );
    assert!(
        diff_net_out <= PASS_THRESHOLD,
        "net_out parity FAILED: {diff_net_out:.3e}"
    );
    assert!(
        diff_delta <= PASS_THRESHOLD,
        "delta parity FAILED: {diff_delta:.3e}"
    );
    assert!(
        diff_weight <= PASS_THRESHOLD,
        "weight parity FAILED: {diff_weight:.3e}"
    );
}

/// Smoke test for the ONNX session wrapper itself (M2 scope item #1):
/// `fnet`/`inet` load and run through `ort`'s CPU execution provider and
/// produce the documented output shape/stride with finite values. Exact
/// numeric agreement with PyTorch was already established in Python by
/// `scripts/check_dpvo_onnx_parity.py` (see the plan doc's M1 results
/// table); what this adds is confidence that `ort`, called from Rust,
/// reproduces the same graph — a well-trodden path already proven by
/// `superpoint_onnx.rs`/`lightglue_onnx.rs` in this crate, not new
/// territory, so shape/finiteness is an appropriately-scoped check here
/// rather than a fixture re-derivation.
#[test]
#[ignore = "reads E:/visloc_archive/dpvo_onnx_m1/*.onnx and needs ORT_DYLIB_PATH — not present in CI"]
fn fnet_inet_sessions_load_and_produce_finite_output_of_the_documented_shape() {
    let onnx = onnx_dir();
    let session = DpvoOnnxSession::load_from_paths(
        onnx.join("fnet.onnx"),
        onnx.join("inet.onnx"),
        onnx.join("dpvo_update_pre_agg.onnx"),
        onnx.join("dpvo_update_post_agg.onnx"),
    )
    .expect("load sessions");

    let height = 64;
    let width = 96;
    // Deterministic pseudo-image (no external RNG dependency): a simple
    // per-pixel formula in [0, 255].
    let mut image = ndarray::Array4::<f32>::zeros((1, 3, height, width));
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                image[(0, c, y, x)] = ((c * 37 + y * 7 + x) % 256) as f32;
            }
        }
    }

    let fmap = time_repeated("run_fnet (64x96 input)", 20, || {
        session.run_fnet(image.view()).expect("run_fnet")
    });
    assert_eq!(
        fmap.shape(),
        &[1, visloc_vision::dpvo::FNET_DIM, height / 4, width / 4]
    );
    assert!(
        fmap.iter().all(|v| v.is_finite()),
        "fnet produced a non-finite value"
    );

    let imap = time_repeated("run_inet (64x96 input)", 20, || {
        session.run_inet(image.view()).expect("run_inet")
    });
    assert_eq!(
        imap.shape(),
        &[1, visloc_vision::dpvo::DIM, height / 4, width / 4]
    );
    assert!(
        imap.iter().all(|v| v.is_finite()),
        "inet produced a non-finite value"
    );

    let (parallel_fmap, parallel_imap) =
        time_repeated("run_encoders concurrent (64x96 input)", 20, || {
            session.run_encoders(image.view()).expect("run_encoders")
        });
    let fmap_diff = max_abs_diff(
        &fmap.iter().copied().collect::<Vec<_>>(),
        &parallel_fmap.iter().copied().collect::<Vec<_>>(),
    );
    let imap_diff = max_abs_diff(
        &imap.iter().copied().collect::<Vec<_>>(),
        &parallel_imap.iter().copied().collect::<Vec<_>>(),
    );
    assert!(
        fmap_diff <= PASS_THRESHOLD && imap_diff <= PASS_THRESHOLD,
        "concurrent encoders changed outputs: fmap={fmap_diff:.3e}, imap={imap_diff:.3e}"
    );
}

#[test]
#[ignore = "reads the external ONNX bundle/fixtures and needs CUDA ORT"]
fn fused_correlation_level0_matches_frozen_fixture_under_strict_cuda() {
    let onnx = onnx_dir();
    let session = DpvoOnnxSession::load_from_paths_with_backend(
        onnx.join("fnet.onnx"),
        onnx.join("inet.onnx"),
        onnx.join("dpvo_update_pre_agg.onnx"),
        onnx.join("dpvo_update_post_agg.onnx"),
        OnnxBackend::Cuda,
    )
    .expect("load strict-CUDA sessions including dpvo_corr_pyramid.onnx");
    assert!(session.correlation_graph_enabled());

    let fixture = NpzArchive::open(fixtures_dir().join("correlation_fixture.npz")).unwrap();
    let (anchor_shape, anchor_data) = fixture.read_f32("anchor_patch_feats").unwrap();
    let (target_shape, target_data) = fixture.read_f32("target_fmap").unwrap();
    let (coords_shape, coords_data) = fixture.read_f32("coords_center").unwrap();
    let (expected_shape, expected_data) = fixture.read_f32("corr_out").unwrap();
    let anchor = ndarray::Array4::from_shape_vec(
        (
            anchor_shape[0],
            anchor_shape[1],
            anchor_shape[2],
            anchor_shape[3],
        ),
        anchor_data,
    )
    .unwrap();
    let target0 = ndarray::Array4::from_shape_vec(
        (
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
        ),
        target_data,
    )
    .unwrap();
    let target1 = ndarray::Array4::<f32>::zeros((
        1,
        target_shape[1],
        target_shape[2] / 4,
        target_shape[3] / 4,
    ));
    let coords = ndarray::Array4::from_shape_vec(
        (
            coords_shape[0],
            coords_shape[1],
            coords_shape[2],
            coords_shape[3],
        ),
        coords_data,
    )
    .unwrap();
    let got = session
        .run_correlation_pyramid(anchor.view(), target0.view(), target1.view(), coords.view())
        .expect("run fused correlation");
    assert_eq!(expected_shape, vec![anchor_shape[0], 3, 3, 49]);
    let mut got_level0 = Vec::with_capacity(expected_data.len());
    for edge in 0..anchor_shape[0] {
        for py in 0..3 {
            for px in 0..3 {
                for tap in 0..49 {
                    let assembled = ((tap * 3 + py) * 3 + px) * 2;
                    got_level0.push(got[(edge, assembled)]);
                }
            }
        }
    }
    let diff = max_abs_diff(&got_level0, &expected_data);
    assert!(
        diff <= PASS_THRESHOLD,
        "fused CUDA correlation level 0 differs from fixture: {diff:.3e}"
    );
}

#[test]
#[ignore = "reads the external correlation fixture and runtime CUDA DLL"]
fn native_cuda_correlation_matches_frozen_fixture() {
    let dll = std::env::var("DPVO_NATIVE_CUDA_DLL").unwrap_or_else(|_| {
        "E:/visloc_archive/dpvo_cuda_kernel_probe_20260723/visloc_dpvo_cuda.dll".into()
    });
    let mut runtime = NativeCudaCorrelation::load(dll).expect("load native CUDA DLL");
    let fixture = NpzArchive::open(fixtures_dir().join("correlation_fixture.npz")).unwrap();
    let (anchor_shape, anchor_data) = fixture.read_f32("anchor_patch_feats").unwrap();
    let (target_shape, target_data) = fixture.read_f32("target_fmap").unwrap();
    let (coords_shape, coords_data) = fixture.read_f32("coords_center").unwrap();
    let (_, expected_data) = fixture.read_f32("corr_out").unwrap();
    let anchor = ndarray::Array4::from_shape_vec(
        (
            anchor_shape[0],
            anchor_shape[1],
            anchor_shape[2],
            anchor_shape[3],
        ),
        anchor_data,
    )
    .unwrap();
    let target0 = ndarray::Array3::from_shape_vec(
        (target_shape[1], target_shape[2], target_shape[3]),
        target_data,
    )
    .unwrap();
    let target1 =
        ndarray::Array3::<f32>::zeros((target_shape[1], target_shape[2] / 4, target_shape[3] / 4));
    let coords = ndarray::Array4::from_shape_vec(
        (
            coords_shape[0],
            coords_shape[1],
            coords_shape[2],
            coords_shape[3],
        ),
        coords_data,
    )
    .unwrap();
    let targets = vec![0_i32; anchor_shape[0]];
    let (got, device_ms) = runtime
        .run(
            anchor.view(),
            &[&target0],
            &[&target1],
            coords.view(),
            &targets,
        )
        .expect("native CUDA correlation succeeds");
    let mut got_level0 = Vec::with_capacity(expected_data.len());
    for edge in 0..anchor_shape[0] {
        for py in 0..3 {
            for px in 0..3 {
                for tap in 0..49 {
                    got_level0.push(got[(edge, ((tap * 3 + py) * 3 + px) * 2)]);
                }
            }
        }
    }
    let diff = max_abs_diff(&got_level0, &expected_data);
    println!("native CUDA correlation device_ms={device_ms:.3}, max_abs={diff:.3e}");
    assert!(diff <= PASS_THRESHOLD);

    // Exercise the indexed multi-frame contract too: the original frozen
    // fixture has only one target, so alternate its original map with a
    // scaled copy and compare every selected row against the CPU primitive.
    let target0_scaled = target0.mapv(|value| value * 0.5);
    let target1_scaled = target1.clone();
    let indexed_targets: Vec<i32> = (0..anchor_shape[0])
        .map(|edge| (edge % 2) as i32)
        .collect();
    let (indexed, _) = runtime
        .run(
            anchor.view(),
            &[&target0, &target0_scaled],
            &[&target1, &target1_scaled],
            coords.view(),
            &indexed_targets,
        )
        .expect("indexed native CUDA correlation succeeds");
    let mut indexed_diff = 0.0_f32;
    for edge in 0..anchor_shape[0] {
        let target = if indexed_targets[edge] == 0 {
            &target0
        } else {
            &target0_scaled
        };
        let expected = corr_cpu(
            anchor.slice(ndarray::s![edge..edge + 1, .., .., ..]),
            target.view(),
            coords.slice(ndarray::s![edge..edge + 1, .., .., ..]),
            3,
        );
        for py in 0..3 {
            for px in 0..3 {
                for tap in 0..49 {
                    let expected_value = expected[(0, py, px, tap)];
                    let assembled = ((tap * 3 + py) * 3 + px) * 2;
                    indexed_diff =
                        indexed_diff.max((indexed[(edge, assembled)] - expected_value).abs());
                }
            }
        }
    }
    assert!(
        indexed_diff <= PASS_THRESHOLD,
        "indexed native CUDA correlation differs from CPU: {indexed_diff:.3e}"
    );

    let target1_pattern = ndarray::Array3::<f32>::from_shape_fn(target1.dim(), |(c, y, x)| {
        ((c * 17 + y * 11 + x * 5) % 257) as f32 / 257.0
    });
    let coords_l1 = coords.mapv(|value| value * 0.25);
    let expected_l1 = corr_cpu(anchor.view(), target1_pattern.view(), coords_l1.view(), 3);
    let (two_level, _) = runtime
        .run(
            anchor.view(),
            &[&target0],
            &[&target1_pattern],
            coords.view(),
            &targets,
        )
        .expect("two-level native CUDA correlation succeeds");
    let mut level1_diff = 0.0_f32;
    for edge in 0..anchor_shape[0] {
        for py in 0..3 {
            for px in 0..3 {
                for tap in 0..49 {
                    let assembled = ((tap * 3 + py) * 3 + px) * 2 + 1;
                    level1_diff = level1_diff.max(
                        (two_level[(edge, assembled)] - expected_l1[(edge, py, px, tap)]).abs(),
                    );
                }
            }
        }
    }
    assert!(
        level1_diff <= PASS_THRESHOLD,
        "native CUDA level-1 correlation differs from CPU: {level1_diff:.3e}"
    );

    let (cached_first, first_ms) = runtime
        .run_cached(
            anchor.view(),
            &[&target0, &target0_scaled],
            &[&target1, &target1_scaled],
            coords.view(),
            &indexed_targets,
            7,
        )
        .expect("first resident-map call uploads maps");
    let (cached_second, second_ms) = runtime
        .run_cached(
            anchor.view(),
            &[&target0, &target0_scaled],
            &[&target1, &target1_scaled],
            coords.view(),
            &indexed_targets,
            7,
        )
        .expect("second resident-map call reuses maps");
    assert_eq!(cached_first, cached_second);
    println!("native CUDA resident maps first={first_ms:.3}ms reused={second_ms:.3}ms");
    assert!(
        second_ms < first_ms,
        "resident reuse should avoid map upload: first={first_ms:.3}ms second={second_ms:.3}ms"
    );

    let (_, stable_first_ms) = runtime
        .run_stable(
            anchor.view(),
            &[&target0, &target0_scaled],
            &[&target1, &target1_scaled],
            coords.view(),
            &indexed_targets,
            &[10, 20],
        )
        .expect("first stable-slot call succeeds");
    let target0_new = target0.mapv(|value| value * 0.25);
    let (stable_second, stable_second_ms) = runtime
        .run_stable(
            anchor.view(),
            &[&target0_scaled, &target0_new],
            &[&target1, &target1_scaled],
            coords.view(),
            &indexed_targets,
            &[20, 30],
        )
        .expect("stable slots retain ID 20 and upload only ID 30");
    let mut stable_diff = 0.0_f32;
    for edge in 0..anchor_shape[0] {
        let target = if indexed_targets[edge] == 0 {
            &target0_scaled
        } else {
            &target0_new
        };
        let expected = corr_cpu(
            anchor.slice(ndarray::s![edge..edge + 1, .., .., ..]),
            target.view(),
            coords.slice(ndarray::s![edge..edge + 1, .., .., ..]),
            3,
        );
        for py in 0..3 {
            for px in 0..3 {
                for tap in 0..49 {
                    let assembled = ((tap * 3 + py) * 3 + px) * 2;
                    stable_diff = stable_diff.max(
                        (stable_second[(edge, assembled)] - expected[(0, py, px, tap)]).abs(),
                    );
                }
            }
        }
    }
    assert!(stable_diff <= PASS_THRESHOLD);
    println!(
        "native CUDA stable slots initial={stable_first_ms:.3}ms one-new={stable_second_ms:.3}ms"
    );

    let invalid_targets = vec![1_i32; anchor_shape[0]];
    let error = runtime
        .run(
            anchor.view(),
            &[&target0],
            &[&target1],
            coords.view(),
            &invalid_targets,
        )
        .expect_err("an out-of-range target index must fail before FFI");
    assert!(error.to_string().contains("outside 0..1"));
}
