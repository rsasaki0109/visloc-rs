//! Fixture-based parity tests for Milestone M6 of `docs/colmap_port_plan.md`
//! (LightGlue as an ONNX two-view matcher for the unordered-SfM path).
//!
//! Every test in this file is `#[ignore]`-gated: they read `.onnx`/`.npz`
//! artifacts from `E:/visloc_archive/lightglue_onnx_m6` (produced by
//! `scripts/export_lightglue_onnx.py` + `scripts/check_lightglue_onnx_parity.py`),
//! which are **not** part of this repo (git-ignored, machine-local,
//! regenerable) and will not exist in CI or on a fresh checkout. Run
//! explicitly:
//!
//! ```text
//! ORT_DYLIB_PATH=E:/tools/onnxruntime-win-x64-1.23.2/lib/onnxruntime.dll \
//!   cargo test -p visloc-vision --features onnx-inference \
//!   --test lightglue_onnx_parity -- --ignored
//! ```
//!
//! The fixture/model directory can be overridden via the
//! `LIGHTGLUE_ONNX_M6_DIR` environment variable (defaults to
//! `E:/visloc_archive/lightglue_onnx_m6`), mirroring the same convention
//! `crates/vision/tests/dpvo_onnx_parity.rs` uses for its own `DPVO_ONNX_M1_DIR`.
//!
//! Two fixtures, both produced from the *same* exported graph
//! (`models/lightglue_terrace_6205x4136.onnx`, baked for ETH3D `terrace`'s
//! 6205x4136 resolution — see the plan doc's M6 results for why one graph
//! per camera resolution is this milestone's chosen tradeoff, not a bug):
//! * `random_fixture.npz` — seeded-random keypoints/descriptors (M=512, N=480).
//! * `real_fixture.npz` — real SuperPoint descriptors from ETH3D `terrace`'s
//!   own cached feature files (`DSC_0259`/`DSC_0260`, an actually-adjacent
//!   pair), i.e. the exact kind of input the acceptance run in the plan doc
//!   feeds this matcher, not just a synthetic sanity check.
//!
//! Both fixtures store the *PyTorch* reference's own `matches0`/`mscores0` as
//! the expected values (written by `check_lightglue_onnx_parity.py` at the
//! same time it checked ONNX Runtime against PyTorch in Python — see that
//! script's own PASS/FAIL report for the Python-side numbers); this file
//! re-derives the same comparison through `ort`, called from Rust, via
//! [`LightGlueOnnxMatcher::match_features`] — the same "well-trodden path,
//! not new territory" scope `dpvo_onnx_parity.rs`'s own
//! `fnet_inet_sessions_load_and_produce_finite_output_of_the_documented_shape`
//! test already established for `superpoint_onnx.rs`/`lightglue_onnx.rs`.
//!
//! Comparison convention (matches `check_lightglue_onnx_parity.py`):
//! `mscores0` is compared by max-abs-diff against a `1e-4` threshold;
//! `matches0` is a discrete, argmax-derived index and is compared by exact
//! agreement rate (expected 100% — any floating-point noise between
//! PyTorch's own CPU kernels and `ort`'s `CPUExecutionProvider` would show up
//! as an occasional *disagreeing index*, not a small numeric delta, so
//! diffing it as if continuous would be meaningless).

#![cfg(feature = "onnx-inference")]

use std::path::PathBuf;

use visloc_vision::dpvo::npz::NpzArchive;
use visloc_vision::features::lightglue_onnx::LightGlueOnnxMatcher;

const PASS_THRESHOLD: f32 = 1e-4;

fn base_dir() -> PathBuf {
    let base = std::env::var("LIGHTGLUE_ONNX_M6_DIR")
        .unwrap_or_else(|_| "E:/visloc_archive/lightglue_onnx_m6".to_string());
    PathBuf::from(base)
}

fn model_path() -> PathBuf {
    base_dir().join("models/lightglue_terrace_6205x4136.onnx")
}

fn fixtures_dir() -> PathBuf {
    base_dir().join("fixtures")
}

/// Read one `(M, 2)` keypoint array out of an npz fixture into
/// `Vec<Point2<f64>>` (the type [`LightGlueOnnxMatcher::match_features`]
/// expects) — the fixture stores `f32` pixel coordinates (matching the ONNX
/// graph's own `f32` keypoint tensor dtype), widened to `f64` here the same
/// way `read_external_deep_features_txt` widens SuperPoint's own `f32`
/// keypoints for the rest of this repo's `f64`-based geometry stack.
fn read_keypoints(archive: &NpzArchive, name: &str) -> Vec<nalgebra::Point2<f64>> {
    let (shape, data) = archive.read_f32(name).unwrap();
    assert_eq!(shape.len(), 2, "{name} shape: {shape:?}");
    assert_eq!(shape[1], 2, "{name} shape: {shape:?}");
    data.chunks_exact(2)
        .map(|xy| nalgebra::Point2::new(xy[0] as f64, xy[1] as f64))
        .collect()
}

/// Read one `(M, 256)` descriptor array out of an npz fixture into
/// `Vec<Vec<f32>>` (the type [`LightGlueOnnxMatcher::match_features`] expects).
fn read_descriptors(archive: &NpzArchive, name: &str) -> Vec<Vec<f32>> {
    let (shape, data) = archive.read_f32(name).unwrap();
    assert_eq!(shape.len(), 2, "{name} shape: {shape:?}");
    let dim = shape[1];
    data.chunks_exact(dim).map(|d| d.to_vec()).collect()
}

/// Shared body for both fixtures: load `(kpts0, desc0, kpts1, desc1,
/// expected matches0, expected mscores0)`, run the fixture's own
/// `matches0`/`mscores0` expectations through
/// [`LightGlueOnnxMatcher::match_features`], and check parity.
fn check_fixture(name: &str) {
    let archive = NpzArchive::open(fixtures_dir().join(format!("{name}_fixture.npz")))
        .unwrap_or_else(|e| panic!("open {name}_fixture.npz (regenerate via scripts/check_lightglue_onnx_parity.py if missing): {e}"));

    let kpts0 = read_keypoints(&archive, "kpts0");
    let desc0 = read_descriptors(&archive, "desc0");
    let kpts1 = read_keypoints(&archive, "kpts1");
    let desc1 = read_descriptors(&archive, "desc1");

    let (_, expected_matches0) = archive.read_i64("matches0").unwrap();
    let (_, expected_mscores0) = archive.read_f32("mscores0").unwrap();
    assert_eq!(
        expected_matches0.len(),
        kpts0.len(),
        "matches0 length should equal M"
    );
    assert_eq!(
        expected_mscores0.len(),
        kpts0.len(),
        "mscores0 length should equal M"
    );

    let matcher =
        LightGlueOnnxMatcher::load_from_path(model_path()).expect("load lightglue ONNX model");
    let got = matcher
        .match_features(&kpts0, &desc0, &kpts1, &desc1)
        .expect("match_features");

    // Reconstruct dense (M,) matches0/mscores0 from the sparse `Vec<LightGlueMatch>`
    // `match_features` returns, so the comparison is apples-to-apples with the
    // fixture's own dense arrays (unmatched query keypoints are `-1`/`0.0`,
    // exactly `filter_matches`'s own convention baked into the ONNX graph).
    let mut got_matches0 = vec![-1i64; kpts0.len()];
    let mut got_mscores0 = vec![0.0f32; kpts0.len()];
    for m in &got {
        got_matches0[m.query_index] = m.train_index as i64;
        got_mscores0[m.query_index] = m.score;
    }

    let agree = expected_matches0
        .iter()
        .zip(got_matches0.iter())
        .filter(|(a, b)| a == b)
        .count();
    let agreement_rate = agree as f64 / expected_matches0.len() as f64;
    // `mscores0` is deliberately compared only at positions the fixture's
    // own `matches0` calls matched (>= 0): LightGlue's `filter_matches`
    // (`lightglue/lightglue.py:302-318`) sets `mscores0` from mutual-nearest-
    // neighbour agreement alone, *before* the `filter_threshold` cut that
    // decides `matches0` — so a "mutual but below-threshold" query keypoint
    // can carry a genuine nonzero `mscores0` in the fixture even though
    // `matches0` is `-1` there. [`LightGlueOnnxMatcher::match_features`]'s
    // public contract, by design, only returns scores for *actual* matches
    // (`matches0 >= 0`) — comparing the full dense array against a
    // reconstruction that zeros every non-returned slot would therefore
    // manufacture a "parity failure" out of an API-boundary filtering
    // choice, not a real PyTorch-vs-`ort` numeric disagreement. Restricting
    // the comparison to the matched subset is what this test is actually
    // able to check, and what the rest of this repo's pipeline actually
    // consumes.
    let matched_positions: Vec<usize> = (0..expected_matches0.len())
        .filter(|&i| expected_matches0[i] >= 0)
        .collect();
    let max_abs_score_diff = matched_positions
        .iter()
        .map(|&i| (expected_mscores0[i] - got_mscores0[i]).abs())
        .fold(0.0f32, f32::max);

    println!(
        "[{name}] M={} N={} matched(expected)={} matched(got)={} index_agreement={:.2}% \
         mscores0_max_abs={:.3e}",
        kpts0.len(),
        kpts1.len(),
        expected_matches0.iter().filter(|&&m| m >= 0).count(),
        got.len(),
        agreement_rate * 100.0,
        max_abs_score_diff,
    );
    assert_eq!(
        agree,
        expected_matches0.len(),
        "[{name}] matches0 index disagreement: {}/{} agree",
        agree,
        expected_matches0.len(),
    );
    assert!(
        max_abs_score_diff <= PASS_THRESHOLD,
        "[{name}] mscores0 parity FAILED: max abs diff {max_abs_score_diff:.3e}",
    );
}

#[test]
#[ignore = "reads E:/visloc_archive/lightglue_onnx_m6 and needs ORT_DYLIB_PATH — not present in CI, see module doc"]
fn random_fixture_matches_pytorch_reference() {
    check_fixture("random");
}

#[test]
#[ignore = "reads E:/visloc_archive/lightglue_onnx_m6 and needs ORT_DYLIB_PATH — not present in CI, see module doc"]
fn real_descriptor_fixture_matches_pytorch_reference() {
    check_fixture("real");
}
