#[test]
fn frame16_prior27_scalar_tail_fma_regression() {
    // Native frame16 iteration6 row24, from the pinned prior-stage capture.
    // The upper-triangular prior makes depths0..23 zero for this row.
    let mut j = DMatrix::<f64>::zeros(27, 27);
    let mut b = DVector::<f64>::zeros(27);
    let mut d = DVector::<f64>::zeros(27);
    let j_bits = [0xc2209eaa, 0x4009bc46, 0x41689bab];
    let d_bits = [0x3b8461b3, 0xb785cdd0, 0xbbf14604];
    for k in 0..3 {
        j[(24, 24 + k)] = f64::from(f32::from_bits(j_bits[k]));
        d[24 + k] = f64::from(f32::from_bits(d_bits[k]));
    }
    b[24] = f64::from(f32::from_bits(0xbf587d62));
    let actual = WindowProblem::prior_rhs_col_major_27_f32(&j, &b, &d);
    assert_eq!((actual[24] as f32).to_bits(), 0xbf8eb73b);
    let mut old = 0.0_f32;
    for col in 0..27 {
        old = WindowProblem::prior_mul_add_separate_f32(old, j[(24, col)] as f32, d[col] as f32);
    }
    assert_eq!(
        WindowProblem::prior_add_separate_f32(b[24] as f32, old).to_bits(),
        0xbf8eb73a
    );
}
#[test]
#[ignore = "requires M11_PRIOR_CAPTURE_ROOT pinned external capture"]
fn frame10_prior27_native_adjusted_rhs() {
    let root = std::path::PathBuf::from(std::env::var("M11_PRIOR_CAPTURE_ROOT").unwrap());
    let read = |name: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(root.join(name))
            .unwrap()
            .lines()
            .map(|x| serde_json::from_str(x).unwrap())
            .collect()
    };
    let inputs = read("prior_inputs.jsonl");
    let stages = read("prior_stages.jsonl");
    assert_eq!(inputs.len(), 8);
    assert_eq!(stages.len(), 32);
    let parse = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| {
                f64::from(f32::from_bits(
                    u32::from_str_radix(x.as_str().unwrap(), 16).unwrap(),
                ))
            })
            .collect()
    };
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(input["iteration"].as_u64(), Some(i as u64));
        let stage = |name: &str| -> Vec<f64> {
            parse(
                &stages
                    .iter()
                    .find(|s| {
                        s["iteration"].as_u64() == Some(i as u64)
                            && s["stage"].as_str() == Some(name)
                    })
                    .unwrap()["bits"],
            )
        };
        let j = DMatrix::from_column_slice(27, 27, &parse(&input["jacobian_bits"]));
        let b = DVector::from_vec(parse(&input["stored_rhs_bits"]));
        let delta = DVector::from_vec(stage("delta"));
        let expected = stage("adjusted_rhs");
        let actual = WindowProblem::prior_rhs_col_major_27_f32(&j, &b, &delta);
        for lane in 0..27 {
            assert_eq!(
                (actual[lane] as f32).to_bits(),
                (expected[lane] as f32).to_bits(),
                "iteration {i} lane {lane}"
            );
        }
        if i == 0 {
            let old = b + j * delta;
            assert_eq!(
                old.iter()
                    .zip(expected.iter())
                    .filter(|(a, b)| (**a as f32).to_bits() != (**b as f32).to_bits())
                    .count(),
                9
            );
        }
    }
}
#[test]
#[ignore = "requires external frame11 native and Rust reanchor captures"]
fn frame11_prior27_reanchor_candidate() {
    let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
    let trace: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            root.join("m11_trial_wired_frame10_20260908/r11_frame11_marg6/marg_trace.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let native_text = std::fs::read_to_string(
        root.join("m11_native_frame11_prior_stages_20260909/r1/prior_inputs.jsonl"),
    )
    .unwrap();
    let native: serde_json::Value =
        serde_json::from_str(native_text.lines().next().unwrap()).unwrap();
    let parse = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
            .collect()
    };
    assert_eq!(trace["event_ordinal"].as_u64(), Some(6));
    assert_eq!(trace["prior_jacobian_f32"]["bits"], native["jacobian_bits"]);
    let j = DMatrix::from_column_slice(27, 27, &parse(&native["jacobian_bits"]));
    let b = DVector::from_vec(parse(&trace["prior_rhs_before_delta_f32"]));
    let d = DVector::from_vec(parse(&trace["keep_delta_f32"]));
    let expected = parse(&native["stored_rhs_bits"]);
    assert_eq!((b.len(), d.len(), expected.len()), (27, 27, 27));
    let old = &b - &j * &d;
    assert_eq!(
        old.iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count(),
        2
    );
    let actual = WindowProblem::prior_reanchor_rhs_col_major_27_f32(&j, &b, &d);
    for row in 0..27 {
        assert_eq!(actual[row].to_bits(), expected[row].to_bits(), "lane {row}");
    }
}
#[test]
#[ignore = "requires external frame17 native and Rust reanchor captures"]
fn frame17_prior33_reanchor_candidate() {
    let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
    let trace: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            root.join("m11_trial_wired_frame10_20260908/r16_frame17_marg12/marg_trace.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let native_text = std::fs::read_to_string(
        root.join("m11_native_frame17_reanchor_abi_20260909/r1/prior_inputs.jsonl"),
    )
    .unwrap();
    let native: serde_json::Value =
        serde_json::from_str(native_text.lines().next().unwrap()).unwrap();
    let parse = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
            .collect()
    };
    assert_eq!(trace["event_ordinal"].as_u64(), Some(12));
    assert_eq!(trace["prior_jacobian_f32"]["bits"], native["jacobian_bits"]);
    let j = DMatrix::from_column_slice(33, 33, &parse(&native["jacobian_bits"]));
    let b = DVector::from_vec(parse(&trace["prior_rhs_before_delta_f32"]));
    let d = DVector::from_vec(parse(&trace["keep_delta_f32"]));
    let expected = parse(&native["stored_rhs_bits"]);
    assert_eq!((b.len(), d.len(), expected.len()), (33, 33, 33));
    let old = &b - &j * &d;
    assert_eq!(
        old.iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count(),
        5
    );
    let actual = WindowProblem::prior_reanchor_rhs_col_major_33_f32(&j, &b, &d);
    for row in 0..33 {
        assert_eq!(actual[row].to_bits(), expected[row].to_bits(), "lane {row}");
    }
}
#[test]
#[ignore = "requires validated native frame17 prior capture"]
fn frame24_prior39_reanchor_candidate() {
    let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
    let trace: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            root.join("m11_trial_wired_frame10_20260908/r21_frame24_marg19/marg_trace.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let native_text = std::fs::read_to_string(
        root.join("m11_native_frame24_reanchor_abi_20260909/r2/prior_inputs.jsonl"),
    )
    .unwrap();
    let native: serde_json::Value =
        serde_json::from_str(native_text.lines().next().unwrap()).unwrap();
    let parse = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
            .collect()
    };
    assert_eq!(trace["event_ordinal"].as_u64(), Some(19));
    assert_eq!(trace["prior_jacobian_f32"]["bits"], native["jacobian_bits"]);
    let j = DMatrix::from_column_slice(39, 39, &parse(&native["jacobian_bits"]));
    let b = parse(&trace["prior_rhs_before_delta_f32"]);
    let d = parse(&trace["keep_delta_f32"]);
    let expected = parse(&native["stored_rhs_bits"]);
    assert_eq!((b.len(), d.len(), expected.len()), (39, 39, 39));
    let actual = WindowProblem::prior_reanchor_rhs_col_major_39_f32(
        &j,
        &DVector::from_vec(b.clone()),
        &DVector::from_vec(d.clone()),
    );
    for row in 0..39 {
        assert_eq!(actual[row].to_bits(), expected[row].to_bits(), "lane {row}");
    }
    let old = DVector::from_vec(b) - j * DVector::from_vec(d);
    assert_eq!(
        old.iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count(),
        6
    );
}
#[test]
#[ignore = "requires validated native frame31 prior capture"]
fn frame31_prior45_reanchor_candidate() {
    let root = std::path::PathBuf::from(
        std::env::var("M11_FRAME31_REANCHOR_ROOT").expect("frame31 capture root"),
    );
    let trace: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            root.join("m11_frame31_marg26_r34_probe_20260912/r1/marg_trace.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let native: serde_json::Value = std::fs::read_to_string(
        root.join("m11_native_frame31_prior_stages_20260912/r1/events.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .find(|record: &serde_json::Value| {
        record["kind"].as_str() == Some("prior_input") && record["iteration"].as_u64() == Some(0)
    })
    .unwrap();
    let parse = |value: &serde_json::Value| -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| f32::from_bits(u32::from_str_radix(word.as_str().unwrap(), 16).unwrap()))
            .collect()
    };
    assert_eq!(trace["event_ordinal"].as_u64(), Some(26));
    assert_eq!(trace["prior_jacobian_f32"]["bits"], native["jacobian_bits"]);
    let j = DMatrix::from_column_slice(45, 45, &parse(&native["jacobian_bits"]));
    let before = DVector::from_vec(parse(&trace["prior_rhs_before_delta_f32"]));
    let delta = DVector::from_vec(parse(&trace["keep_delta_f32"]));
    let expected = parse(&native["stored_rhs_bits"]);
    let generic = &before - &j * &delta;
    assert_eq!(
        generic
            .iter()
            .zip(&expected)
            .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
            .count(),
        8
    );
    let candidate = WindowProblem::prior_reanchor_rhs_col_major_45_f32(&j, &before, &delta);
    for row in 0..45 {
        assert_eq!(
            candidate[row].to_bits(),
            expected[row].to_bits(),
            "lane {row}"
        );
    }
}
#[test]
#[ignore = "requires validated native frame38 prior and Rust event33 captures"]
fn frame38_prior51_reanchor_candidate() {
    let root = std::path::PathBuf::from(
        std::env::var("M11_FRAME38_PRIOR_ROOT").expect("frame38 capture root"),
    );
    let trace: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            root.join("m11_trial_wired_frame10_20260908/r40_frame38_marg33/marg_trace.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let native: serde_json::Value = std::fs::read_to_string(
        root.join("m11_native_frame38_strict_capture_20260913/r2/events.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .find(|record: &serde_json::Value| {
        record["kind"].as_str() == Some("prior_input") && record["iteration"].as_u64() == Some(0)
    })
    .unwrap();
    let parse = |value: &serde_json::Value| -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| f32::from_bits(u32::from_str_radix(word.as_str().unwrap(), 16).unwrap()))
            .collect()
    };
    assert_eq!(trace["event_ordinal"].as_u64(), Some(33));
    assert_eq!(trace["prior_jacobian_f32"]["bits"], native["jacobian_bits"]);
    let j = DMatrix::from_column_slice(51, 51, &parse(&native["jacobian_bits"]));
    let before = DVector::from_vec(parse(&trace["prior_rhs_before_delta_f32"]));
    let delta = DVector::from_vec(parse(&trace["keep_delta_f32"]));
    let expected = parse(&native["stored_rhs_bits"]);
    let generic = &before - &j * &delta;
    assert_eq!(
        generic
            .iter()
            .zip(&expected)
            .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
            .count(),
        15
    );
    let candidate = WindowProblem::prior_reanchor_rhs_col_major_51_f32(&j, &before, &delta);
    for row in 0..51 {
        assert_eq!(
            candidate[row].to_bits(),
            expected[row].to_bits(),
            "lane {row}"
        );
    }
}
#[test]
#[ignore = "requires validated native frame38 prior capture"]
fn frame38_prior51_adjusted_rhs_candidate() {
    let root = std::path::PathBuf::from(
        std::env::var("M11_FRAME38_PRIOR_ROOT").expect("frame38 capture root"),
    );
    let records: Vec<serde_json::Value> = std::fs::read_to_string(
        root.join("m11_native_frame38_strict_capture_20260913/r2/events.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .collect();
    let parse = |value: &serde_json::Value| -> Vec<f64> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| {
                f64::from(f32::from_bits(
                    u32::from_str_radix(word.as_str().unwrap(), 16).unwrap(),
                ))
            })
            .collect()
    };
    for iteration in 0..8 {
        let input = records
            .iter()
            .find(|record| {
                record["kind"].as_str() == Some("prior_input")
                    && record["iteration"].as_u64() == Some(iteration)
            })
            .unwrap();
        let stage = |name: &str| {
            records
                .iter()
                .find(|record| {
                    record["kind"].as_str() == Some("prior_stage")
                        && record["stage"].as_str() == Some(name)
                        && record["iteration"].as_u64() == Some(iteration)
                })
                .unwrap()
        };
        let j = DMatrix::from_column_slice(51, 51, &parse(&input["jacobian_bits"]));
        let stored = DVector::from_vec(parse(&input["stored_rhs_bits"]));
        let delta = DVector::from_vec(parse(&stage("delta")["bits"]));
        let expected = parse(&stage("adjusted_rhs")["bits"]);
        let candidate = WindowProblem::prior_rhs_col_major_51_f32(&j, &stored, &delta);
        for row in 0..51 {
            assert_eq!(
                (candidate[row] as f32).to_bits(),
                (expected[row] as f32).to_bits(),
                "iteration {iteration} lane {row}"
            );
        }
    }
}
#[test]
#[ignore = "requires validated native frame31 prior capture"]
fn frame31_prior45_adjusted_rhs_candidate() {
    let root = std::path::PathBuf::from(
        std::env::var("M11_FRAME31_REANCHOR_ROOT").expect("frame31 capture root"),
    );
    let records: Vec<serde_json::Value> = std::fs::read_to_string(
        root.join("m11_native_frame31_prior_stages_20260912/r1/events.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .collect();
    let parse = |value: &serde_json::Value| -> Vec<f64> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| {
                f64::from(f32::from_bits(
                    u32::from_str_radix(word.as_str().unwrap(), 16).unwrap(),
                ))
            })
            .collect()
    };
    for iteration in 0..8 {
        let input = records
            .iter()
            .find(|record| {
                record["kind"].as_str() == Some("prior_input")
                    && record["iteration"].as_u64() == Some(iteration)
            })
            .unwrap();
        let stage = |name: &str| {
            records
                .iter()
                .find(|record| {
                    record["kind"].as_str() == Some("prior_stage")
                        && record["stage"].as_str() == Some(name)
                        && record["iteration"].as_u64() == Some(iteration)
                })
                .unwrap()
        };
        let j = DMatrix::from_column_slice(45, 45, &parse(&input["jacobian_bits"]));
        let stored = DVector::from_vec(parse(&input["stored_rhs_bits"]));
        let delta = DVector::from_vec(parse(&stage("delta")["bits"]));
        let expected = parse(&stage("adjusted_rhs")["bits"]);
        let candidate = WindowProblem::prior_rhs_col_major_45_f32(&j, &stored, &delta);
        for row in 0..45 {
            assert_eq!(
                (candidate[row] as f32).to_bits(),
                (expected[row] as f32).to_bits(),
                "iteration {iteration} lane {row}"
            );
        }
    }
}
#[test]
#[ignore = "requires validated native frame17 prior capture"]
fn frame17_prior33_adjusted_rhs_candidate() {
    let root = std::path::PathBuf::from(std::env::var("M11_PRIOR_CAPTURE_ROOT").unwrap());
    let read = |name: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(root.join(name))
            .unwrap()
            .lines()
            .map(|x| serde_json::from_str(x).unwrap())
            .collect()
    };
    let inputs = read("prior_inputs.jsonl");
    let stages = read("prior_stages.jsonl");
    assert_eq!((inputs.len(), stages.len()), (8, 32));
    let parse = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| {
                f64::from(f32::from_bits(
                    u32::from_str_radix(x.as_str().unwrap(), 16).unwrap(),
                ))
            })
            .collect()
    };
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(input["iteration"].as_u64(), Some(i as u64));
        assert_eq!(
            (input["rows"].as_u64(), input["cols"].as_u64()),
            (Some(33), Some(33))
        );
        let stage = |name: &str| -> Vec<f64> {
            parse(
                &stages
                    .iter()
                    .find(|s| {
                        s["iteration"].as_u64() == Some(i as u64)
                            && s["stage"].as_str() == Some(name)
                    })
                    .unwrap()["bits"],
            )
        };
        let j = DMatrix::from_column_slice(33, 33, &parse(&input["jacobian_bits"]));
        let b = DVector::from_vec(parse(&input["stored_rhs_bits"]));
        let d = DVector::from_vec(stage("delta"));
        let expected = stage("adjusted_rhs");
        let actual = WindowProblem::prior_rhs_col_major_33_f32(&j, &b, &d);
        for row in 0..33 {
            assert_eq!(
                (actual[row] as f32).to_bits(),
                (expected[row] as f32).to_bits(),
                "iteration {i} lane {row}"
            );
        }
        if i == 0 {
            let old = b + j * d;
            assert_eq!(
                old.iter()
                    .zip(&expected)
                    .filter(|(a, b)| (**a as f32).to_bits() != (**b as f32).to_bits())
                    .count(),
                15
            );
        }
    }
}
#[test]
#[ignore = "requires validated native frame24 prior capture"]
fn frame24_prior39_adjusted_rhs_candidate() {
    let root = std::path::PathBuf::from(std::env::var("M11_PRIOR_CAPTURE_ROOT").unwrap());
    let read = |name: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(root.join(name))
            .unwrap()
            .lines()
            .map(|x| serde_json::from_str(x).unwrap())
            .collect()
    };
    let inputs = read("prior_inputs.jsonl");
    let stages = read("prior_stages.jsonl");
    assert_eq!((inputs.len(), stages.len()), (8, 32));
    let parse = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| {
                f64::from(f32::from_bits(
                    u32::from_str_radix(x.as_str().unwrap(), 16).unwrap(),
                ))
            })
            .collect()
    };
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(input["iteration"].as_u64(), Some(i as u64));
        assert_eq!(
            (input["rows"].as_u64(), input["cols"].as_u64()),
            (Some(39), Some(39))
        );
        let stage = |name: &str| -> Vec<f64> {
            parse(
                &stages
                    .iter()
                    .find(|s| {
                        s["iteration"].as_u64() == Some(i as u64)
                            && s["stage"].as_str() == Some(name)
                    })
                    .unwrap()["bits"],
            )
        };
        let j = DMatrix::from_column_slice(39, 39, &parse(&input["jacobian_bits"]));
        let b = DVector::from_vec(parse(&input["stored_rhs_bits"]));
        let d = DVector::from_vec(stage("delta"));
        let expected = stage("adjusted_rhs");
        let actual = WindowProblem::prior_rhs_col_major_39_f32(&j, &b, &d);
        for row in 0..39 {
            assert_eq!(
                (actual[row] as f32).to_bits(),
                (expected[row] as f32).to_bits(),
                "iteration {i} lane {row}"
            );
        }
        if i == 0 {
            let old = b + j * d;
            assert_eq!(
                old.iter()
                    .zip(&expected)
                    .filter(|(a, b)| (**a as f32).to_bits() != (**b as f32).to_bits())
                    .count(),
                20
            );
        }
    }
}
use super::*;
use crate::TimingStat;
use nalgebra::{Point3, Vector6};

#[test]
fn upstream_f32_householder_beta_uses_fused_pivot_square() {
    // Captured native event-1 column 20 operands.  The fused Eigen
    // association produces 4647bb33; the algebraically equivalent
    // non-fused product/add produces 4647bb34 on the pinned boundary.
    let c0 = f32::from_bits(0xc6452f66);
    let tail_sq_norm = f32::from_bits(0x4a7ca5af);
    let fused = super::upstream_f32_householder_beta(c0, tail_sq_norm);
    let product = std::hint::black_box(c0 * c0);
    let non_fused = std::hint::black_box(product + tail_sq_norm).sqrt();
    assert_eq!(fused.to_bits(), 0x4647bb33);
    assert_eq!(non_fused.to_bits(), 0x4647bb34);
}

#[test]
fn diagnostic_env_allowlist_keeps_harness_fast_and_fails_closed() {
    let canonical = [
        "VISLOC_BASALT_GT_FREE",
        "VISLOC_BASALT_PROFILE",
        "VISLOC_BASALT_SEQUENCE",
        "VISLOC_BASALT_INPUT_ROOT",
        "VISLOC_BASALT_OUTPUT_ROOT",
        "VISLOC_BASALT_SEED",
        "VISLOC_BASALT_TEMPORAL_SEED",
        "VISLOC_BASALT_THREADS",
        "VISLOC_BASALT_TIMING_BREAKDOWN",
        "VISLOC_BASALT_TRACE",
        "VISLOC_BASALT_M8D_SETUP_INPUT",
        "VISLOC_BASALT_M8D_SETUP_ORACLE",
        "VISLOC_BASALT_MH01_ROOT",
        "VISLOC_BASALT_CALIBRATION",
        "VISLOC_BASALT_CONFIG",
        "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACE",
        "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACKS",
        "VISLOC_BASALT_LIFECYCLE_TRACE",
    ]
    .into_iter()
    .map(std::ffi::OsString::from)
    .collect::<Vec<_>>();
    assert!(!diagnostic_env_active_for_keys(canonical));

    for key in [
        "visloc_basalt_gt_free",
        "VisLoc_Basalt_Profile",
        "vIsLoC_bAsAlT_tHrEaDs",
        "vIsLoC_bAsAlT_tImInG_bReAkDoWn",
        "VISLOC_basalt_mh01_root",
        "visloc_BASALT_config",
    ] {
        assert!(!diagnostic_env_active_for_keys([std::ffi::OsString::from(
            key
        )]));
    }

    for key in [
        "VISLOC_BASALT_DETAIL_ITERATIONS",
        "VISLOC_BASALT_DIAGNOSTIC_IMU_HB",
        "VISLOC_BASALT_FUTURE_KEY",
    ] {
        assert!(diagnostic_env_active_for_keys([std::ffi::OsString::from(
            key
        )]));
    }
    for key in [
        "visloc_basalt_detail_iterations",
        "VisLoc_Basalt_Diagnostic_Imu_Hb",
        "vIsLoC_bAsAlT_fUtUrE_kEy",
    ] {
        assert!(diagnostic_env_active_for_keys([std::ffi::OsString::from(
            key
        )]));
    }
    assert!(!diagnostic_env_active_for_keys([std::ffi::OsString::from(
        "UNRELATED_ENV"
    )]));
}

#[test]
fn visual_prefix_selector_snapshot_parses_typed_values_once_and_fails_closed() {
    let mut values = HashMap::new();
    values.insert("VISLOC_BASALT_VISUAL_PREFIX_FRAME_ID", OsString::from("12"));
    values.insert("VISLOC_BASALT_VISUAL_PREFIX_ITERATION", OsString::from("3"));
    values.insert("VISLOC_BASALT_VISUAL_PREFIX_TRIAL", OsString::from("0"));
    assert_eq!(
        cached_selector_u64(&values, "VISLOC_BASALT_VISUAL_PREFIX_FRAME_ID"),
        Some(Ok(12))
    );
    assert_eq!(
        cached_selector_usize(&values, "VISLOC_BASALT_VISUAL_PREFIX_ITERATION"),
        Some(Ok(3))
    );
    values.insert(
        "VISLOC_BASALT_VISUAL_PREFIX_TRIAL",
        OsString::from("not-a-number"),
    );
    assert_eq!(
        cached_selector_usize(&values, "VISLOC_BASALT_VISUAL_PREFIX_TRIAL"),
        Some(Err(()))
    );
}

#[test]
fn diagnostic_env_snapshot_is_observed_in_fresh_processes() {
    fn run_child(diagnostic_key: Option<&str>, expected: bool) {
        let executable = std::env::current_exe().expect("test executable path");
        let mut command = std::process::Command::new(executable);
        // Do not let the parent test environment decide the child result.
        // Keep ordinary process variables intact, but remove every
        // VISLOC_BASALT_* key before installing the one case under test.
        for (key, _) in std::env::vars_os() {
            if key
                .to_string_lossy()
                .get(.."VISLOC_BASALT_".len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("VISLOC_BASALT_"))
            {
                command.env_remove(key);
            }
        }
        command
            .env("VISLOC_DIAGNOSTIC_SNAPSHOT_CHILD", "1")
            .env(
                "VISLOC_DIAGNOSTIC_SNAPSHOT_EXPECTED",
                if expected { "true" } else { "false" },
            )
            .args(["diagnostic_env_snapshot_child", "--nocapture"]);
        if let Some(key) = diagnostic_key {
            command.env(key, "1");
        }
        let output = command.output().expect("spawn fresh diagnostic child");
        assert!(
            output.status.success(),
            "child failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Each child snapshots its environment on its first call.  Separate
    // processes make this independent of unit-test ordering and avoid
    // mutating the parent's environment.
    run_child(Some("VISLOC_BASALT_SNAPSHOT_UNKNOWN"), true);
    run_child(Some("vIsLoC_bAsAlT_PrOfIlE"), false);
    run_child(None, false);
}

#[test]
fn diagnostic_env_snapshot_child() {
    if std::env::var_os("VISLOC_DIAGNOSTIC_SNAPSHOT_CHILD").is_none() {
        return;
    }
    let expected = std::env::var("VISLOC_DIAGNOSTIC_SNAPSHOT_EXPECTED")
        .expect("child expected value")
        == "true";
    assert_eq!(diagnostic_env_active(), expected);
    // A second read must return the same process-lifetime value.
    assert_eq!(diagnostic_env_active(), expected);
}

#[test]
fn imu_f64_observer_scope_serializes_actual_product_without_recomputing() {
    assert!(!initial_imu_f64_capture_scope_active());
    let from = nav(10, 0.0);
    let mut to = nav(11, 0.2);
    to.nav.imu_to_world.rotation =
        UnitQuaternion::from_scaled_axis(Vector3::new(0.013, -0.027, 0.041));
    let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
    delta.delta_time = 0.05;
    delta.delta_rotation = UnitQuaternion::from_scaled_axis(Vector3::new(-0.009, 0.017, 0.023));

    let (_, rotation, _, product) =
        imu_unwhitened_residual(&from.nav, &to.nav, &delta, Vector3::new(0.0, 0.0, -9.81));
    let record = build_imu_unwhitened_residual_f64_capture(
        Some(to.frame_id),
        &from,
        &to,
        0,
        0,
        1,
        None,
        &product,
        rotation,
    )
    .expect("active frame is required for an inline capture");

    let bits = |value: f64| format!("{:016x}", value.to_bits());
    let q = product.quaternion();
    assert_eq!(
        record["product_xyzw_f64_bits"],
        json!([bits(q.i), bits(q.j), bits(q.k), bits(q.w)])
    );
    assert_eq!(
        record["observed_r_R_f64_bits"],
        json!([bits(rotation.x), bits(rotation.y), bits(rotation.z)])
    );
    assert_eq!(record["capture_proof"]["product_same_call"], json!(true));
    assert_eq!(record["capture_proof"]["result_same_call"], json!(true));
    assert_eq!(record["capture_proof"]["norm_captured"], json!(false));
    assert_eq!(record["capture_proof"]["angle_captured"], json!(false));
    assert!(record.get("norm_f64_bits").is_none());
    assert!(record.get("candidate").is_none());
    assert!(record.get("native_oracle").is_none());
    assert!(!initial_imu_f64_capture_scope_active());

    let scope = InitialImuF64CaptureScope::enter();
    assert!(initial_imu_f64_capture_scope_active());
    let first_link = InitialImuF64CaptureLink::set(4);
    assert_eq!(initial_imu_f64_capture_link_index(), Some(4));
    {
        let second_link = InitialImuF64CaptureLink::set(9);
        assert_eq!(initial_imu_f64_capture_link_index(), Some(9));
        drop(second_link);
    }
    assert_eq!(initial_imu_f64_capture_link_index(), Some(4));
    drop(first_link);
    assert_eq!(initial_imu_f64_capture_link_index(), None);
    drop(scope);
    assert!(!initial_imu_f64_capture_scope_active());

    // Constructing the observation must not replace or perturb the
    // production helper's already-computed f64 values.
    let (_, rotation_again, _, product_again) =
        imu_unwhitened_residual(&from.nav, &to.nav, &delta, Vector3::new(0.0, 0.0, -9.81));
    let q_again = product_again.quaternion();
    assert_eq!(q.i.to_bits(), q_again.i.to_bits());
    assert_eq!(q.j.to_bits(), q_again.j.to_bits());
    assert_eq!(q.k.to_bits(), q_again.k.to_bits());
    assert_eq!(q.w.to_bits(), q_again.w.to_bits());
    for (lhs, rhs) in rotation.iter().zip(rotation_again.iter()) {
        assert_eq!(lhs.to_bits(), rhs.to_bits());
    }
}

fn nav(frame: u64, x: f64) -> WindowState {
    WindowState {
        frame_id: frame,
        timestamp_ns: frame as i64 + 1,
        nav: BasaltNavState {
            imu_to_world: SE3::new(UnitQuaternion::identity(), Vector3::new(x, 0.0, 0.0)),
            ..BasaltNavState::default()
        },
        stored_current_nav: BasaltNavState {
            imu_to_world: SE3::new(UnitQuaternion::identity(), Vector3::new(x, 0.0, 0.0)),
            ..BasaltNavState::default()
        },
        linearized_nav: BasaltNavState {
            imu_to_world: SE3::new(UnitQuaternion::identity(), Vector3::new(x, 0.0, 0.0)),
            ..BasaltNavState::default()
        },
        linearized_delta: DVector::zeros(NAV_STATE_DOF),
        is_keyframe: frame == 0,
        is_latest: frame == 2,
        linearized: frame == 0,
    }
}

fn test_window(states: Vec<WindowState>, landmarks: Vec<WindowLandmark>) -> WindowProblem {
    // These fresh fixtures insert hosts in landmark order, with no prior
    // erase/rehash history. Production carries its persistent order.
    let mut host_order = super::super::landmarks::NativeHostOrder::default();
    for landmark in &landmarks {
        if !landmark.observations.is_empty() {
            host_order.insert(
                states[landmark.anchor_state_index].timestamp_ns as u64,
                landmark.anchor_camera_id,
            );
        }
    }
    WindowProblem {
        trial_host_order: host_order.keys().collect(),
        camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap(),
        cameras: Vec::new(),
        t_imu_cam: Vec::new(),
        poses: Vec::new(),
        states,
        landmarks,
        imu_links: Vec::new(),
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0,
        initial_accel_bias_weight: 1.0,
        initial_gyro_bias_weight: 1.0,
        prior: None,
        anchor_point: None,
        gravity_world: Vector3::zeros(),
        scalar_mode: ScalarMode::UpstreamF32,
        factor_config: FactorConfig::default(),
        visual_only: None,
    }
}

fn assert_window_dynamic_eq(lhs: &WindowProblem, rhs: &WindowProblem) {
    assert_eq!(lhs.poses, rhs.poses);
    assert_eq!(lhs.states, rhs.states);
    assert_eq!(lhs.landmarks, rhs.landmarks);
}

fn assert_token_rejected_without_mutation(
    candidate: &mut WindowProblem,
    state: &DVector<f64>,
    step: &DVector<f64>,
    token: LmTrialToken,
) {
    let before = candidate.clone();
    let result = candidate.accept_step_with_token(state, step, token);
    assert!(matches!(result, Err(LmFailure::LinearSolve)));
    assert_window_dynamic_eq(candidate, &before);
    assert_eq!(candidate.prior, before.prior);
    assert_eq!(candidate.anchor_point, before.anchor_point);
    assert_eq!(candidate.scalar_mode, before.scalar_mode);
}

fn token_binding_window() -> WindowProblem {
    let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
    test_window(
        vec![nav(0, 0.0), nav(1, 0.25)],
        vec![WindowLandmark {
            track_id: 23,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction,
            inverse_distance: 0.75,
            observations: Vec::new(),
        }],
    )
}

fn visual_trial_fixture() -> (
    WindowProblem,
    DVector<f64>,
    DVector<f64>,
    DVector<f64>,
    Vec<Option<Vector3<f64>>>,
) {
    let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
    let landmark = WindowLandmark {
        track_id: 17,
        anchor_state_index: 0,
        anchor_camera_id: 0,
        direction,
        inverse_distance: 0.75,
        observations: vec![
            WindowObservation {
                state_index: 1,
                camera_id: 0,
                pixel: Point2::new(320.0, 240.0),
            },
            WindowObservation {
                state_index: 2,
                camera_id: 0,
                pixel: Point2::new(319.0, 241.0),
            },
        ],
    };
    let problem = test_window(vec![nav(0, 0.0), nav(1, 0.25), nav(2, 0.5)], vec![landmark]);
    let state = problem.initial_state();
    let mut step = DVector::zeros(problem.state_dof());
    step[15] = 0.015625;
    step[20] = -0.03125;
    step[30] = -0.0234375;
    step[35] = 0.046875;
    let trial = problem.apply_step(&state, &step);
    let landmark_steps = problem.landmark_steps(&state, &step);
    (problem, state, step, trial, landmark_steps)
}

#[test]
fn clean_trial_view_shares_topology_and_preserves_accept_reject_semantics() {
    let problem = test_window(vec![nav(1, 0.0)], Vec::new());
    let before = problem.clone();
    let state = problem.initial_state();
    let mut step = DVector::zeros(NAV_STATE_DOF);
    step[0] = 1.25;
    step[5] = 0.03125;
    let trial = problem.apply_step(&state, &step);
    let view = WindowTrialView::from_step(&problem, &state, &step, &trial);

    assert!(std::ptr::eq(view.base, &problem));
    assert_window_dynamic_eq(&problem, &before);
    assert_ne!(
        view.states[0].nav.imu_to_world.translation.x.to_bits(),
        problem.states[0].nav.imu_to_world.translation.x.to_bits()
    );

    // A rejected trial is dropped without touching the accepted sidecar;
    // an accepted step follows the same candidate manifold values.
    drop(view);
    assert_window_dynamic_eq(&problem, &before);
    let mut accepted = problem.clone();
    accepted.accept_step(&state, &step).unwrap();
    assert_eq!(
        accepted.states[0].nav.imu_to_world.translation.x.to_bits(),
        trial[0].to_bits()
    );
}

#[test]
fn timed_trial_view_subbuckets_preserve_values_and_disabled_path_is_empty() {
    let problem = token_binding_window();
    let state = problem.initial_state();
    let mut step = DVector::zeros(problem.state_dof());
    step[15] = 0.25;
    step[20] = -0.03125;
    let trial = problem.apply_step(&state, &step);
    let baseline = WindowTrialView::from_step(&problem, &state, &step, &trial);

    let mut disabled = TimingBreakdown::default();
    let disabled_view =
        WindowTrialView::from_step_timed(&problem, &state, &step, &trial, &mut disabled);
    assert_eq!(disabled_view.poses, baseline.poses);
    assert_eq!(disabled_view.states, baseline.states);
    assert_eq!(disabled_view.landmarks, baseline.landmarks);
    assert_eq!(disabled_view.landmark_steps, baseline.landmark_steps);
    assert_eq!(disabled_view.prior_fej_delta, baseline.prior_fej_delta);
    assert_eq!(disabled.lm_trial_construct_step, TimingStat::default());
    assert_eq!(disabled.lm_trial_landmark_recovery, TimingStat::default());
    assert_eq!(disabled.lm_trial_apply_step_full, TimingStat::default());
    assert_eq!(
        disabled.lm_trial_landmark_materialization,
        TimingStat::default()
    );
    assert_eq!(disabled.lm_trial_sidecar_other, TimingStat::default());

    let mut timing = TimingBreakdown::enabled_for_test();
    let timed = WindowTrialView::from_step_timed(&problem, &state, &step, &trial, &mut timing);
    assert_eq!(timed.poses, baseline.poses);
    assert_eq!(timed.states, baseline.states);
    assert_eq!(timed.landmarks, baseline.landmarks);
    assert_eq!(timed.landmark_steps, baseline.landmark_steps);
    assert_eq!(timed.prior_fej_delta, baseline.prior_fej_delta);
    assert_eq!(
        timed.cost().unwrap().to_bits(),
        baseline.cost().unwrap().to_bits()
    );

    assert_eq!(timing.lm_trial_construct_step.count, 1);
    assert_eq!(timing.lm_trial_landmark_recovery.count, 1);
    assert_eq!(timing.lm_trial_apply_step_full.count, 1);
    assert_eq!(timing.lm_trial_landmark_materialization.count, 1);
    // The faithful source order places pose/state sidecars before the
    // landmark loop and view/prior sidecars after it, hence two sequential
    // samples in this one bucket.
    assert_eq!(timing.lm_trial_sidecar_other.count, 2);
    assert!(timing.lm_trial_construct_subbuckets_ns() <= timing.lm_trial_construct_step.total_ns);
}

#[test]
fn clean_trial_token_matches_recomputed_landmark_recovery_exactly() {
    let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
    let landmark = WindowLandmark {
        track_id: 17,
        anchor_state_index: 0,
        anchor_camera_id: 0,
        direction,
        inverse_distance: 0.75,
        observations: vec![
            WindowObservation {
                state_index: 1,
                camera_id: 0,
                pixel: Point2::new(320.0, 240.0),
            },
            WindowObservation {
                state_index: 2,
                camera_id: 0,
                pixel: Point2::new(319.0, 241.0),
            },
        ],
    };
    let problem = test_window(vec![nav(0, 0.0), nav(1, 0.25), nav(2, 0.5)], vec![landmark]);
    let state = problem.initial_state();
    let mut step = DVector::zeros(problem.state_dof());
    step[15] = 0.015625;
    step[20] = -0.03125;
    step[30] = -0.0234375;
    step[35] = 0.046875;
    let trial = problem.apply_step(&state, &step);
    let expected_steps = problem.landmark_steps(&state, &step);
    assert!(expected_steps.iter().any(Option::is_some));

    let mut optimized = problem.clone();
    let before = optimized.clone();
    let binding = optimized.lm_trial_binding(&state, &step);
    let (optimized_cost, token) = optimized
        .trial_cost_timed_with_token(&state, &step, &trial, &mut TimingBreakdown::default())
        .unwrap();
    let token_steps = token.take_landmark_steps().expect("clean trial token");
    assert_eq!(token_steps.len(), expected_steps.len());
    for (token_step, expected_step) in token_steps.iter().zip(&expected_steps) {
        match (token_step, expected_step) {
            (Some(token_step), Some(expected_step)) => {
                assert_eq!(token_step.x.to_bits(), expected_step.x.to_bits());
                assert_eq!(token_step.y.to_bits(), expected_step.y.to_bits());
                assert_eq!(token_step.z.to_bits(), expected_step.z.to_bits());
            }
            (None, None) => {}
            mismatch => panic!("landmark token mismatch: {mismatch:?}"),
        }
    }
    let token = LmTrialToken::with_bound_landmark_steps(token_steps, binding);
    optimized
        .accept_step_with_token(&state, &step, token)
        .unwrap();

    let mut legacy = problem.clone();
    let legacy_cost = WindowTrialView::from_step(&problem, &state, &step, &trial)
        .cost()
        .unwrap();
    legacy.accept_step(&state, &step).unwrap();
    assert_eq!(optimized_cost.to_bits(), legacy_cost.to_bits());
    assert_window_dynamic_eq(&optimized, &legacy);
    assert_eq!(optimized.prior, legacy.prior);
    assert_eq!(optimized.anchor_point, legacy.anchor_point);
    assert_window_dynamic_eq(&before, &problem);
}

#[test]
fn trial_host_order_changes_token_generation_and_factor_plan() {
    let mut problem = token_binding_window();
    problem.trial_host_order = vec![(100, 0), (200, 1)];
    let generation = lm_token_window_generation(&problem);
    let plan = lm_token_factor_plan_fingerprint(&problem);
    problem.trial_host_order.reverse();
    assert_ne!(generation, lm_token_window_generation(&problem));
    assert_ne!(plan, lm_token_factor_plan_fingerprint(&problem));
    problem.trial_host_order.reverse();
    assert_eq!(generation, lm_token_window_generation(&problem));
    assert_eq!(plan, lm_token_factor_plan_fingerprint(&problem));
    problem.trial_host_order[0].1 = 1;
    assert_ne!(generation, lm_token_window_generation(&problem));
    assert_ne!(plan, lm_token_factor_plan_fingerprint(&problem));
}

#[test]
fn prepared_trial_view_matches_legacy_view_and_cost_bitwise() {
    let (problem, state, step, trial, landmark_steps) = visual_trial_fixture();
    let legacy = WindowTrialView::from_step(&problem, &state, &step, &trial);
    let prepared = WindowTrialView::from_step_with_landmark_steps(
        &problem,
        &state,
        &step,
        &trial,
        landmark_steps.clone(),
    );

    assert_eq!(prepared.poses, legacy.poses);
    assert_eq!(prepared.states, legacy.states);
    assert_eq!(prepared.landmarks, legacy.landmarks);
    assert_eq!(prepared.landmark_steps, legacy.landmark_steps);
    assert_eq!(prepared.prior_fej_delta, legacy.prior_fej_delta);
    assert_eq!(
        prepared.cost().unwrap().to_bits(),
        legacy.cost().unwrap().to_bits()
    );

    let preparation = LmTrialPreparation::from_test_entries_for_state_step(
        &state,
        &step,
        1e-10,
        vec![(0, problem.landmarks[0].track_id, landmark_steps[0])],
    );
    let mut candidate = problem.clone();
    let (prepared_cost, token) = candidate
        .trial_cost_timed_with_preparation(
            &state,
            &step,
            &trial,
            Some(preparation),
            &mut TimingBreakdown::default(),
        )
        .unwrap();
    assert_eq!(prepared_cost.to_bits(), legacy.cost().unwrap().to_bits());
    candidate
        .accept_step_with_token(&state, &step, token)
        .unwrap();

    let mut expected = problem.clone();
    expected.accept_step(&state, &step).unwrap();
    assert_window_dynamic_eq(&candidate, &expected);
}

#[test]
fn prepared_trial_timing_keeps_recovery_inside_existing_parent_bucket() {
    let (problem, state, step, trial, landmark_steps) = visual_trial_fixture();
    let preparation = LmTrialPreparation::from_test_entries_for_state_step(
        &state,
        &step,
        1e-10,
        vec![(0, problem.landmarks[0].track_id, landmark_steps[0])],
    );
    let mut timing = TimingBreakdown::enabled_for_test();
    let (_, token) = problem
        .trial_cost_timed_with_preparation(&state, &step, &trial, Some(preparation), &mut timing)
        .unwrap();
    drop(token);
    assert_eq!(timing.lm_trial_construct_step.count, 1);
    assert_eq!(timing.lm_trial_landmark_recovery.count, 1);
    assert_eq!(timing.lm_trial_apply_step_full.count, 1);
    assert_eq!(timing.lm_trial_landmark_materialization.count, 1);
    assert_eq!(timing.lm_trial_sidecar_other.count, 2);
    assert!(timing.lm_trial_construct_subbuckets_ns() <= timing.lm_trial_construct_step.total_ns);
}

#[test]
fn missing_preparation_uses_the_legacy_window_token_path() {
    let (problem, state, step, trial, _) = visual_trial_fixture();
    let mut prepared_timing = TimingBreakdown::default();
    let prepared = problem
        .trial_cost_timed_with_preparation(&state, &step, &trial, None, &mut prepared_timing)
        .unwrap();
    let mut legacy_timing = TimingBreakdown::default();
    let legacy = problem
        .trial_cost_timed_with_token(&state, &step, &trial, &mut legacy_timing)
        .unwrap();
    assert_eq!(prepared.0.to_bits(), legacy.0.to_bits());
    assert_eq!(
        prepared.1.take_landmark_steps().unwrap(),
        legacy.1.take_landmark_steps().unwrap()
    );
}

#[test]
fn prepared_landmark_mapping_rejects_bad_inputs_before_mutation() {
    let (problem, state, step, trial, landmark_steps) = visual_trial_fixture();
    let make_preparation = |tolerance: f64, entries: Vec<(usize, u64)>| {
        LmTrialPreparation::from_test_entries_for_state_step(
            &state,
            &step,
            tolerance,
            entries
                .into_iter()
                .map(|(index, track_id)| (index, track_id, landmark_steps[0]))
                .collect(),
        )
    };
    let cases = [
        ("tolerance", make_preparation(1e-9, vec![(0, 17)])),
        ("index", make_preparation(1e-10, vec![(1, 17)])),
        ("track", make_preparation(1e-10, vec![(0, 18)])),
    ];
    for (name, preparation) in cases {
        let candidate = problem.clone();
        let before = candidate.clone();
        let result = candidate.trial_cost_timed_with_preparation(
            &state,
            &step,
            &trial,
            Some(preparation),
            &mut TimingBreakdown::default(),
        );
        assert!(matches!(result, Err(LmFailure::LinearSolve)), "{name}");
        assert_window_dynamic_eq(&candidate, &before);
    }

    let mut stale_state = state.clone();
    stale_state[0] += 0.125;
    let stale_state_preparation = LmTrialPreparation::from_test_entries_for_state_step(
        &stale_state,
        &step,
        1e-10,
        vec![(0, 17, landmark_steps[0])],
    );
    let candidate = problem.clone();
    let before = candidate.clone();
    let result = candidate.trial_cost_timed_with_preparation(
        &state,
        &step,
        &trial,
        Some(stale_state_preparation),
        &mut TimingBreakdown::default(),
    );
    assert!(
        matches!(result, Err(LmFailure::LinearSolve)),
        "state fingerprint"
    );
    assert_window_dynamic_eq(&candidate, &before);

    let mut stale_step = step.clone();
    stale_step[0] += 0.125;
    let stale_step_preparation = LmTrialPreparation::from_test_entries_for_state_step(
        &state,
        &stale_step,
        1e-10,
        vec![(0, 17, landmark_steps[0])],
    );
    let candidate = problem.clone();
    let before = candidate.clone();
    let result = candidate.trial_cost_timed_with_preparation(
        &state,
        &step,
        &trial,
        Some(stale_step_preparation),
        &mut TimingBreakdown::default(),
    );
    assert!(
        matches!(result, Err(LmFailure::LinearSolve)),
        "step fingerprint"
    );
    assert_window_dynamic_eq(&candidate, &before);

    let mut duplicate_problem = problem;
    duplicate_problem
        .landmarks
        .push(duplicate_problem.landmarks[0].clone());
    duplicate_problem.landmarks[1].track_id = 18;
    let duplicate = LmTrialPreparation::from_test_entries_for_state_step(
        &state,
        &step,
        1e-10,
        vec![(0, 17, landmark_steps[0]), (0, 17, landmark_steps[0])],
    );
    let before = duplicate_problem.clone();
    let result = duplicate_problem.trial_cost_timed_with_preparation(
        &state,
        &step,
        &trial,
        Some(duplicate),
        &mut TimingBreakdown::default(),
    );
    assert!(matches!(result, Err(LmFailure::LinearSolve)));
    assert_window_dynamic_eq(&duplicate_problem, &before);
}

#[test]
fn prepared_mixed_prior_visual_imu_matches_legacy_coverage_and_accept() {
    let (mut problem, _, _, _, _) = visual_trial_fixture();
    // Keep the initial anchor row and one chronological IMU+bias pair in
    // the same trial as the visual landmark.  This exercises the full
    // factor order that the clean reducer hands to the preparation hook.
    problem.anchor_point = Some(DVector::zeros(NAV_STATE_DOF));
    let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
    delta.delta_time = 1.0;
    problem.imu_links.push(WindowImuLink {
        from_index: 0,
        to_index: 1,
        delta,
    });

    let state = problem.initial_state();
    let mut step = DVector::zeros(problem.state_dof());
    step[15] = 0.015625;
    step[20] = -0.03125;
    step[30] = -0.0234375;
    step[35] = 0.046875;
    let trial = problem.apply_step(&state, &step);
    let landmark_steps = problem.landmark_steps(&state, &step);
    let factors = problem.linearize_snapshot(&state).unwrap();
    assert_eq!(
        factors
            .iter()
            .filter(|factor| factor.kind == FactorKind::Prior)
            .count(),
        1
    );
    assert_eq!(
        factors
            .iter()
            .filter(|factor| factor.kind == FactorKind::Visual)
            .count(),
        1
    );
    assert_eq!(
        factors
            .iter()
            .filter(|factor| factor.kind == FactorKind::Imu)
            .count(),
        1
    );
    assert_eq!(
        factors
            .iter()
            .filter(|factor| factor.kind == FactorKind::Bias)
            .count(),
        1
    );

    let preparation = LmTrialPreparation::from_test_entries_for_state_step(
        &state,
        &step,
        1e-10,
        vec![(0, problem.landmarks[0].track_id, landmark_steps[0])],
    );
    let mut prepared = problem.clone();
    let (prepared_cost, prepared_token) = prepared
        .trial_cost_timed_with_preparation(
            &state,
            &step,
            &trial,
            Some(preparation),
            &mut TimingBreakdown::default(),
        )
        .unwrap();
    let mut legacy = problem.clone();
    let (legacy_cost, legacy_token) = legacy
        .trial_cost_timed_with_token(&state, &step, &trial, &mut TimingBreakdown::default())
        .unwrap();
    assert_eq!(prepared_cost.to_bits(), legacy_cost.to_bits());

    prepared
        .accept_step_with_token(&state, &step, prepared_token)
        .unwrap();
    legacy
        .accept_step_with_token(&state, &step, legacy_token)
        .unwrap();
    assert_window_dynamic_eq(&prepared, &legacy);
    assert_eq!(prepared.prior, legacy.prior);
    assert_eq!(prepared.anchor_point, legacy.anchor_point);
}

#[test]
fn compact_reducer_preparation_mixed_window_path_matches_legacy_end_to_end() {
    let (mut problem, _, _, _, _) = visual_trial_fixture();
    problem.anchor_point = Some(DVector::zeros(NAV_STATE_DOF));
    let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
    delta.delta_time = 1.0;
    problem.imu_links.push(WindowImuLink {
        from_index: 0,
        to_index: 1,
        delta,
    });

    let state = problem.initial_state();
    let mut step = DVector::zeros(problem.state_dof());
    step[15] = 0.015625;
    step[20] = -0.03125;
    step[30] = -0.0234375;
    step[35] = 0.046875;
    let trial = problem.apply_step(&state, &step);
    let factors = problem.linearize_snapshot(&state).unwrap();
    let preparation = compact_trial_preparation_for_test(&factors, &state, &step, 1e-10)
        .expect("mixed compact reducer must produce Window preparation");

    let mut prepared = problem.clone();
    let (prepared_cost, prepared_token) = prepared
        .trial_cost_timed_with_preparation(
            &state,
            &step,
            &trial,
            Some(preparation),
            &mut TimingBreakdown::default(),
        )
        .unwrap();
    let mut legacy = problem.clone();
    let (legacy_cost, legacy_token) = legacy
        .trial_cost_timed_with_token(&state, &step, &trial, &mut TimingBreakdown::default())
        .unwrap();
    assert_eq!(prepared_cost.to_bits(), legacy_cost.to_bits());

    prepared
        .accept_step_with_token(&state, &step, prepared_token)
        .unwrap();
    legacy
        .accept_step_with_token(&state, &step, legacy_token)
        .unwrap();
    assert_window_dynamic_eq(&prepared, &legacy);
    assert_eq!(prepared.prior, legacy.prior);
    assert_eq!(prepared.anchor_point, legacy.anchor_point);
}

#[test]
fn clean_trial_token_drop_and_malformed_token_leave_window_unchanged() {
    let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
    let landmark = WindowLandmark {
        track_id: 19,
        anchor_state_index: 0,
        anchor_camera_id: 0,
        direction,
        inverse_distance: 0.75,
        observations: Vec::new(),
    };
    let problem = test_window(vec![nav(0, 0.0)], vec![landmark]);
    let state = problem.initial_state();
    let step = DVector::zeros(problem.state_dof());

    let dropped = problem.clone();
    let trial = dropped.apply_step(&state, &step);
    let (_, token) = dropped
        .trial_cost_timed_with_token(&state, &step, &trial, &mut TimingBreakdown::default())
        .unwrap();
    drop(token);
    assert_window_dynamic_eq(&dropped, &problem);

    let mut malformed = problem.clone();
    let result = malformed.accept_step_with_token(
        &state,
        &step,
        LmTrialToken::with_landmark_steps(Vec::new()),
    );
    assert!(matches!(result, Err(LmFailure::LinearSolve)));
    assert_window_dynamic_eq(&malformed, &problem);
    assert_eq!(malformed.prior, problem.prior);
    assert_eq!(malformed.anchor_point, problem.anchor_point);
}

#[test]
fn empty_trial_token_preserves_legacy_accept_fallback() {
    let problem = test_window(vec![nav(0, 0.0), nav(1, 0.25)], Vec::new());
    let state = problem.initial_state();
    let mut step = DVector::zeros(problem.state_dof());
    step[15] = 0.25;
    step[20] = 0.03125;

    let mut fallback = problem.clone();
    fallback
        .accept_step_with_token(&state, &step, LmTrialToken::default())
        .unwrap();
    let mut legacy = problem;
    legacy.accept_step(&state, &step).unwrap();
    assert_window_dynamic_eq(&fallback, &legacy);
}

#[test]
fn clean_trial_token_rejects_cross_context_bindings_before_mutation() {
    let owner = token_binding_window();
    let state = owner.initial_state();
    let mut step = DVector::zeros(owner.state_dof());
    step[15] = 0.25;
    let binding = owner.lm_trial_binding(&state, &step);
    let token_steps = vec![None; owner.landmarks.len()];

    // A clone has the same topology but is a different WindowProblem
    // identity, so a token from the owner must not write into it.
    let mut cross_window = token_binding_window();
    assert_token_rejected_without_mutation(
        &mut cross_window,
        &state,
        &step,
        LmTrialToken::with_bound_landmark_steps(token_steps.clone(), binding),
    );

    let mut state_candidate = token_binding_window();
    let state_candidate_state = state_candidate.initial_state();
    let state_binding = state_candidate.lm_trial_binding(&state_candidate_state, &step);
    let mut changed_state = state_candidate_state;
    changed_state[0] += 0.125;
    assert_token_rejected_without_mutation(
        &mut state_candidate,
        &changed_state,
        &step,
        LmTrialToken::with_bound_landmark_steps(token_steps.clone(), state_binding),
    );

    let mut changed_step = step.clone();
    changed_step[15] += 0.125;
    let mut step_candidate = token_binding_window();
    let step_candidate_state = step_candidate.initial_state();
    let step_binding = step_candidate.lm_trial_binding(&step_candidate_state, &step);
    assert_token_rejected_without_mutation(
        &mut step_candidate,
        &step_candidate_state,
        &changed_step,
        LmTrialToken::with_bound_landmark_steps(token_steps.clone(), step_binding),
    );

    let mut topology_candidate = token_binding_window();
    let topology_state = topology_candidate.initial_state();
    let topology_binding = topology_candidate.lm_trial_binding(&topology_state, &step);
    topology_candidate
        .landmarks
        .push(topology_candidate.landmarks[0].clone());
    assert_token_rejected_without_mutation(
        &mut topology_candidate,
        &topology_state,
        &step,
        LmTrialToken::with_bound_landmark_steps(token_steps.clone(), topology_binding),
    );

    let mut scalar_candidate = token_binding_window();
    let scalar_state = scalar_candidate.initial_state();
    let scalar_binding = scalar_candidate.lm_trial_binding(&scalar_state, &step);
    scalar_candidate.scalar_mode = ScalarMode::ExtendedF64;
    assert_token_rejected_without_mutation(
        &mut scalar_candidate,
        &scalar_state,
        &step,
        LmTrialToken::with_bound_landmark_steps(token_steps, scalar_binding),
    );
}

#[test]
fn failed_landmark_increment_keeps_the_previous_trial_value() {
    let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
    let landmark = WindowLandmark {
        track_id: 7,
        anchor_state_index: 0,
        anchor_camera_id: 0,
        direction,
        inverse_distance: 0.75,
        observations: Vec::new(),
    };
    let problem = test_window(vec![nav(0, 0.0)], vec![landmark]);
    let original = TrialLandmarkValue {
        direction,
        inverse_distance: 0.75,
    };

    let failed = problem.trial_landmark_value(0, Vector3::new(f64::NAN, 0.0, 0.0));
    assert!(failed.is_none());
    let materialized = failed.unwrap_or(original);
    assert_eq!(materialized, original);

    let state = problem.initial_state();
    let step = DVector::zeros(problem.state_dof());
    let trial = problem.apply_step(&state, &step);
    let view = WindowTrialView::from_step(&problem, &state, &step, &trial);
    let view_value = view.landmark_parameter(0).unwrap();
    assert_eq!(view_value.direction, original.direction);
    assert_eq!(view_value.inverse_distance, original.inverse_distance);

    let accepted = problem
        .trial_landmark_value(0, Vector3::new(0.015625, -0.03125, 0.125))
        .unwrap();
    assert_ne!(accepted, original);
    assert_eq!(problem.landmarks[0].direction, original.direction);
    assert_eq!(
        problem.landmarks[0].inverse_distance,
        original.inverse_distance
    );
}

#[test]
fn diagnostic_trial_path_retains_the_owned_clone_fallback() {
    assert!(diagnostic_env_active_for_keys([std::ffi::OsString::from(
        "VISLOC_BASALT_DETAIL_ITERATIONS"
    )]));
    let problem = test_window(vec![nav(1, 0.0)], Vec::new());
    let before = problem.clone();
    let state = problem.initial_state();
    let mut step = DVector::zeros(NAV_STATE_DOF);
    step[0] = 0.5;
    let owned = problem.trial_problem_with_step_owned(&state, &step);

    assert_window_dynamic_eq(&problem, &before);
    assert!(!std::ptr::eq(&problem, &owned));
    assert_ne!(
        owned.states[0].nav.imu_to_world.translation.x.to_bits(),
        problem.states[0].nav.imu_to_world.translation.x.to_bits()
    );
}

#[test]
fn malformed_trial_chart_is_rejected_before_view_construction() {
    let problem = test_window(vec![nav(1, 0.0)], Vec::new());
    let state = problem.initial_state();
    let step = DVector::zeros(problem.state_dof());
    let malformed = DVector::zeros(problem.state_dof() - 1);
    let result = problem.trial_cost(&state, &step, &malformed);
    assert!(matches!(result, Err(LmFailure::LinearSolve)));
}

#[test]
#[ignore = "requires external pinned-native frame8 capture"]
fn m11_frame8_prior_error_native_inputs_probe() {
    let root = std::env::var("M11_PRIOR_INPUT_ROOT").expect("capture root");
    let inputs = std::fs::read_to_string(format!("{root}/prior_inputs.jsonl")).unwrap();
    let outputs = std::fs::read_to_string(format!("{root}/lm_eval.jsonl")).unwrap();
    let oracle: Vec<serde_json::Value> = outputs
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let parse = |value: &serde_json::Value| {
        f32::from_bits(u32::from_str_radix(value.as_str().unwrap(), 16).unwrap()) as f64
    };
    let mut count = 0;
    for line in inputs.lines() {
        let input: serde_json::Value = serde_json::from_str(line).unwrap();
        let iteration = input["iteration"].as_u64().unwrap() as usize;
        let rows = input["rows"].as_u64().unwrap() as usize;
        let cols = input["cols"].as_u64().unwrap() as usize;
        assert_eq!((rows, cols), (21, 21));
        assert_eq!(iteration, count);
        let j = DMatrix::from_iterator(
            rows,
            cols,
            input["J_column_major"]
                .as_array()
                .unwrap()
                .iter()
                .map(&parse),
        );
        let b = DVector::from_iterator(rows, input["b"].as_array().unwrap().iter().map(&parse));
        let delta =
            DVector::from_iterator(cols, input["delta"].as_array().unwrap().iter().map(&parse));
        let actual = upstream_marginal_prior_error(&j, &b, &delta, ScalarMode::UpstreamF32) as f32;
        let expected = u32::from_str_radix(
            oracle[iteration]["after_update_marg_prior_error"]["f32_bits"]
                .as_str()
                .unwrap(),
            16,
        )
        .unwrap();
        println!(
            "M11_PRIOR_ERROR iter={iteration} native={expected:08x} rust={:08x} exact={}",
            actual.to_bits(),
            actual.to_bits() == expected
        );
        assert_eq!(
            actual.to_bits(),
            expected,
            "native prior cost iteration {iteration}"
        );
        for packet_gemv in [false, true] {
            let jf = j.map(|v| v as f32);
            let df = delta.map(|v| v as f32);
            let jd = if packet_gemv {
                WindowProblem::prior_rhs_col_major_21_f32(&j, &DVector::zeros(21), &delta)
                    .map(|v| v as f32)
            } else {
                &jf * &df
            };
            let mut products = [0.0_f32; 16];
            for k in 0..16 {
                products[k] = (b[k] as f32 + 0.5_f32 * jd[k]) * jd[k];
            }
            let lanes: [f32; 8] = std::array::from_fn(|k| products[k] + products[k + 8]);
            let half: [f32; 4] = std::array::from_fn(|k| lanes[k] + lanes[k + 4]);
            let mut sum = (half[0] + half[2]) + (half[1] + half[3]);
            for k in 16..21 {
                sum = (0.5_f32 * jd[k] + b[k] as f32).mul_add(jd[k], sum);
            }
            println!("M11_PRIOR_DOT_CANDIDATE iter={iteration} packet_gemv={packet_gemv} native={expected:08x} candidate={:08x} exact={}", sum.to_bits(), sum.to_bits() == expected);
        }
        count += 1;
    }
    assert_eq!(count, 8);
    // Every captured native output is a mandatory bitwise oracle.
}

#[test]
#[ignore = "requires external pinned-native frame18 prior cost capture"]
fn m11_frame18_prior33_error_same_inputs_probe() {
    let root = std::env::var("M11_PRIOR_INPUT_ROOT").expect("capture root");
    let records = |name: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(format!("{root}/{name}"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let inputs = records("prior_inputs.jsonl");
    let oracle = records("lm_eval.jsonl");
    assert_eq!(inputs.len(), 8);
    assert_eq!(oracle.len(), 8);
    let parse = |v: &serde_json::Value| {
        f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap()) as f64
    };
    let mut exact = 0;
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(input["iteration"].as_u64(), Some(i as u64));
        assert_eq!(oracle[i]["iteration"].as_u64(), Some(i as u64));
        assert_eq!(input["rows"].as_u64(), Some(33));
        assert_eq!(input["cols"].as_u64(), Some(33));
        assert_eq!(input["is_sqrt"].as_bool(), Some(true));
        let j = DMatrix::from_iterator(
            33,
            33,
            input["J_column_major"]
                .as_array()
                .unwrap()
                .iter()
                .map(&parse),
        );
        let b = DVector::from_iterator(33, input["b"].as_array().unwrap().iter().map(&parse));
        let delta =
            DVector::from_iterator(33, input["delta"].as_array().unwrap().iter().map(&parse));
        let actual = upstream_marginal_prior_error(&j, &b, &delta, ScalarMode::UpstreamF32) as f32;
        let expected =
            u32::from_str_radix(oracle[i]["after_prior"]["f32_bits"].as_str().unwrap(), 16)
                .unwrap();
        exact += usize::from(actual.to_bits() == expected);
        println!(
            "M11_PRIOR33_SAME_INPUT iter={i} native={expected:08x} rust={:08x} exact={}",
            actual.to_bits(),
            actual.to_bits() == expected
        );
    }
    // Diagnostic probe: report all cases before enforcing the native oracle.
    assert_eq!(exact, 8, "native prior33 same-input cost parity");
}

#[test]
#[ignore = "requires external pinned-native frame24 prior cost capture"]
fn m11_frame24_prior39_error_same_inputs_probe() {
    let root = std::env::var("M11_PRIOR_INPUT_ROOT").expect("capture root");
    let records = |name: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(format!("{root}/{name}"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let inputs = records("prior_inputs.jsonl");
    let oracle = records("lm_eval.jsonl");
    assert_eq!(inputs.len(), 8);
    assert_eq!(oracle.len(), 8);
    let parse = |v: &serde_json::Value| {
        f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap()) as f64
    };
    let mut exact = 0;
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(input["iteration"].as_u64(), Some(i as u64));
        assert_eq!(oracle[i]["iteration"].as_u64(), Some(i as u64));
        assert_eq!(input["rows"].as_u64(), Some(39));
        assert_eq!(input["cols"].as_u64(), Some(39));
        assert_eq!(input["is_sqrt"].as_bool(), Some(true));
        let j = DMatrix::from_iterator(
            39,
            39,
            input["J_column_major"]
                .as_array()
                .unwrap()
                .iter()
                .map(&parse),
        );
        let b = DVector::from_iterator(39, input["b"].as_array().unwrap().iter().map(&parse));
        let delta =
            DVector::from_iterator(39, input["delta"].as_array().unwrap().iter().map(&parse));
        let actual = upstream_marginal_prior_error(&j, &b, &delta, ScalarMode::UpstreamF32) as f32;
        let expected =
            u32::from_str_radix(oracle[i]["after_prior"]["f32_bits"].as_str().unwrap(), 16)
                .unwrap();
        exact += usize::from(actual.to_bits() == expected);
        println!(
            "M11_PRIOR39_SAME_INPUT iter={i} native={expected:08x} rust={:08x} exact={}",
            actual.to_bits(),
            actual.to_bits() == expected
        );
    }
    // Diagnostic probe: report all cases before enforcing the native oracle.
    assert_eq!(exact, 8, "native prior39 same-input cost parity");
}

#[test]
#[ignore = "requires external pinned-native frame32 prior cost capture"]
fn m11_frame32_prior45_error_same_inputs_probe() {
    let root = std::env::var("M11_FRAME32_PRIOR_ROOT").expect("capture root");
    let events: Vec<serde_json::Value> = std::fs::read_to_string(format!("{root}/events.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let inputs: Vec<_> = events
        .iter()
        .filter(|event| event["kind"].as_str() == Some("prior_input"))
        .collect();
    let deltas: Vec<_> = events
        .iter()
        .filter(|event| {
            event["kind"].as_str() == Some("prior_stage")
                && event["stage"].as_str() == Some("delta")
        })
        .collect();
    let oracle: Vec<_> = events
        .iter()
        .filter(|event| event["kind"].as_str() == Some("lm_eval"))
        .collect();
    assert_eq!(inputs.len(), 8);
    assert_eq!(deltas.len(), 8);
    assert_eq!(oracle.len(), 8);
    let parse = |v: &serde_json::Value| {
        f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap()) as f64
    };
    let mut exact = 0;
    // The native prior probe fires at the accepted state entering the next
    // iteration.  LM evaluation N therefore corresponds to prior input N+1
    // only when candidate N was accepted.  Candidate 4 was rejected and
    // candidate 7 has no following iteration in this bounded capture.
    for candidate_iteration in [0_u64, 1, 2, 3, 5, 6] {
        let input_iteration = candidate_iteration + 1;
        let input = inputs
            .iter()
            .find(|event| event["iteration"].as_u64() == Some(input_iteration))
            .unwrap();
        let delta_event = deltas
            .iter()
            .find(|event| event["iteration"].as_u64() == Some(input_iteration))
            .unwrap();
        let expected_event = oracle
            .iter()
            .find(|event| event["iteration"].as_u64() == Some(candidate_iteration))
            .unwrap();
        assert_eq!(input["rows"].as_u64(), Some(45));
        assert_eq!(input["cols"].as_u64(), Some(45));
        let j = DMatrix::from_iterator(
            45,
            45,
            input["jacobian_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(&parse),
        );
        let b = DVector::from_iterator(
            45,
            input["stored_rhs_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(&parse),
        );
        let delta = DVector::from_iterator(
            45,
            delta_event["bits"].as_array().unwrap().iter().map(&parse),
        );
        let actual = upstream_marginal_prior_error(&j, &b, &delta, ScalarMode::UpstreamF32) as f32;
        let expected = u32::from_str_radix(
            expected_event["after_prior"]["f32_bits"].as_str().unwrap(),
            16,
        )
        .unwrap();
        exact += usize::from(actual.to_bits() == expected);
        println!(
            "M11_PRIOR45_SAME_INPUT candidate_iter={candidate_iteration} input_iter={input_iteration} native={expected:08x} rust={:08x} exact={}",
            actual.to_bits(),
            actual.to_bits() == expected
        );
    }
    assert_eq!(exact, 6, "native prior45 accepted same-input cost parity");
}

#[test]
#[ignore = "requires validated native frame38 prior capture"]
fn m11_frame38_prior51_error_same_inputs_probe() {
    let root = std::path::PathBuf::from(
        std::env::var("M11_FRAME38_PRIOR_ROOT").expect("frame38 capture root"),
    );
    let events: Vec<serde_json::Value> = std::fs::read_to_string(
        root.join("m11_native_frame38_strict_capture_20260913/r2/events.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .collect();
    let parse = |value: &serde_json::Value| -> Vec<f64> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| {
                f64::from(f32::from_bits(
                    u32::from_str_radix(word.as_str().unwrap(), 16).unwrap(),
                ))
            })
            .collect()
    };
    for candidate_iteration in [0_u64, 1, 2, 3, 4, 6] {
        let input_iteration = candidate_iteration + 1;
        let input = events
            .iter()
            .find(|event| {
                event["kind"].as_str() == Some("prior_input")
                    && event["iteration"].as_u64() == Some(input_iteration)
            })
            .unwrap();
        let delta = events
            .iter()
            .find(|event| {
                event["kind"].as_str() == Some("prior_stage")
                    && event["stage"].as_str() == Some("delta")
                    && event["iteration"].as_u64() == Some(input_iteration)
            })
            .unwrap();
        let oracle = events
            .iter()
            .find(|event| {
                event["kind"].as_str() == Some("lm_eval")
                    && event["iteration"].as_u64() == Some(candidate_iteration)
            })
            .unwrap();
        let j = DMatrix::from_column_slice(51, 51, &parse(&input["jacobian_bits"]));
        let rhs = DVector::from_vec(parse(&input["stored_rhs_bits"]));
        let delta = DVector::from_vec(parse(&delta["bits"]));
        let actual = upstream_marginal_prior_error_51_candidate(&j, &rhs, &delta) as f32;
        let expected =
            u32::from_str_radix(oracle["after_prior"]["f32_bits"].as_str().unwrap(), 16).unwrap();
        assert_eq!(
            actual.to_bits(),
            expected,
            "candidate iteration {candidate_iteration}"
        );
    }
}

#[test]
#[ignore = "requires validated native frame45 prior capture"]
fn m11_frame45_prior57_same_inputs_probe() {
    let root = std::path::PathBuf::from(
        std::env::var("M11_FRAME45_PRIOR_ROOT").expect("frame45 capture root"),
    );
    let events: Vec<serde_json::Value> = std::fs::read_to_string(
        root.join("m11_native_frame45_strict_capture_20260913/r1/events.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .collect();
    let parse = |value: &serde_json::Value| -> Vec<f64> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| {
                f64::from(f32::from_bits(
                    u32::from_str_radix(word.as_str().unwrap(), 16).unwrap(),
                ))
            })
            .collect()
    };
    for candidate_iteration in 0..7_u64 {
        let iteration = candidate_iteration + 1;
        let input = events
            .iter()
            .find(|event| {
                event["kind"].as_str() == Some("prior_input")
                    && event["iteration"].as_u64() == Some(iteration)
            })
            .unwrap();
        let stage = |name: &str| {
            events
                .iter()
                .find(|event| {
                    event["kind"].as_str() == Some("prior_stage")
                        && event["stage"].as_str() == Some(name)
                        && event["iteration"].as_u64() == Some(iteration)
                })
                .unwrap()
        };
        let oracle = events
            .iter()
            .find(|event| {
                event["kind"].as_str() == Some("lm_eval")
                    && event["iteration"].as_u64() == Some(candidate_iteration)
            })
            .unwrap();
        let j = DMatrix::from_column_slice(57, 57, &parse(&input["jacobian_bits"]));
        let rhs = DVector::from_vec(parse(&input["stored_rhs_bits"]));
        let delta = DVector::from_vec(parse(&stage("delta")["bits"]));
        let adjusted_expected = parse(&stage("adjusted_rhs")["bits"]);
        let adjusted = WindowProblem::prior_rhs_col_major_57_f32(&j, &rhs, &delta);
        for lane in 0..57 {
            assert_eq!(
                (adjusted[lane] as f32).to_bits(),
                (adjusted_expected[lane] as f32).to_bits(),
                "adjusted RHS input iteration {iteration} lane {lane}"
            );
        }
        let actual = upstream_marginal_prior_error_57_candidate(&j, &rhs, &delta) as f32;
        let expected =
            u32::from_str_radix(oracle["after_prior"]["f32_bits"].as_str().unwrap(), 16).unwrap();
        assert_eq!(
            actual.to_bits(),
            expected,
            "prior model candidate {candidate_iteration} from input {iteration}"
        );
    }
}

#[test]
fn clean_trial_view_prior_cost_is_bit_exact_to_diagnostic_clone() {
    let mut problem = test_window(vec![nav(0, 0.0)], Vec::new());
    problem.prior = Some(WindowPrior {
        frame_ids: vec![0],
        block_kinds: vec![WindowBlockKind::State],
        jacobian: DMatrix::identity(NAV_STATE_DOF, NAV_STATE_DOF),
        rhs: DVector::from_iterator(
            NAV_STATE_DOF,
            (0..NAV_STATE_DOF).map(|index| (index as f64 - 4.0) * 0.125),
        ),
        fej_point: DVector::zeros(NAV_STATE_DOF),
    });
    let state = problem.initial_state();
    let mut step = DVector::zeros(NAV_STATE_DOF);
    step[0] = 0.25;
    step[3] = 0.03125;
    step[9] = -0.0625;
    let trial = problem.apply_step(&state, &step);

    let owned = problem.trial_problem_with_step_owned(&state, &step);
    let owned_cost = owned.owned_trial_objective(&trial).unwrap();
    let view_cost = WindowTrialView::from_step(&problem, &state, &step, &trial)
        .cost()
        .unwrap();
    assert_eq!(view_cost.to_bits(), owned_cost.to_bits());
}

#[test]
fn clean_trial_view_prior_state_index_skips_pose_prefix() {
    let mut problem = test_window(vec![nav(1, 0.0)], Vec::new());
    problem.poses.push(WindowPose {
        frame_id: 0,
        timestamp_ns: 1,
        pose: SE3::identity(),
        stored_current_pose: SE3::identity(),
        linearized_pose: SE3::identity(),
        linearized_delta: DVector::zeros(POSE_DOF),
        is_keyframe: true,
    });
    // The carried prior names the full state block after the pose-only
    // prefix.  Marking it FEJ-linearized makes the trial increment enter
    // the prior current point, exposing the absolute-vs-local index
    // conversion that differs when a pose prefix is present.
    problem.states[0].linearized = true;
    problem.prior = Some(WindowPrior {
        frame_ids: vec![1],
        block_kinds: vec![WindowBlockKind::State],
        jacobian: DMatrix::identity(NAV_STATE_DOF, NAV_STATE_DOF),
        rhs: DVector::from_iterator(
            NAV_STATE_DOF,
            (0..NAV_STATE_DOF).map(|index| (index as f64 - 2.0) * 0.25),
        ),
        fej_point: DVector::zeros(NAV_STATE_DOF),
    });
    let state = problem.initial_state();
    let mut step = DVector::zeros(problem.state_dof());
    step[6] = 0.25;
    step[9] = -0.03125;
    let trial = problem.apply_step(&state, &step);

    let owned = problem.trial_problem_with_step_owned(&state, &step);
    let owned_cost = owned.owned_trial_objective(&trial).unwrap();
    let view_cost = WindowTrialView::from_step(&problem, &state, &step, &trial)
        .cost()
        .unwrap();
    assert_eq!(view_cost.to_bits(), owned_cost.to_bits());
}

#[test]
fn clean_trial_view_visual_factor_is_bit_exact_to_diagnostic_clone() {
    let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
    let landmark = WindowLandmark {
        track_id: 11,
        anchor_state_index: 0,
        anchor_camera_id: 0,
        direction,
        inverse_distance: 0.75,
        observations: vec![WindowObservation {
            state_index: 1,
            camera_id: 0,
            pixel: Point2::new(320.0, 240.0),
        }],
    };
    let problem = test_window(vec![nav(0, 0.0), nav(1, 0.0)], vec![landmark]);
    let state = problem.initial_state();
    let mut step = DVector::zeros(problem.state_dof());
    step[15] = 0.015625;
    step[20] = -0.03125;
    let trial = problem.apply_step(&state, &step);
    let owned = problem.trial_problem_with_step_owned(&state, &step);
    let owned_cost = owned.owned_trial_objective(&trial).unwrap();
    let view_cost = WindowTrialView::from_step(&problem, &state, &step, &trial)
        .cost()
        .unwrap();
    assert_eq!(view_cost.to_bits(), owned_cost.to_bits());
}

#[test]
fn trial_objective_rejects_missing_host_history_without_mutation() {
    let (mut problem, state, step, trial, _) = visual_trial_fixture();
    problem.trial_host_order.clear();
    let before = problem.clone();
    let owned = problem.trial_problem_with_step_owned(&state, &step);
    assert_eq!(
        owned.owned_trial_objective(&trial),
        Err(LmFailure::LinearSolve)
    );
    assert_eq!(
        WindowTrialView::from_step(&problem, &state, &step, &trial).cost(),
        Err(LmFailure::LinearSolve)
    );
    assert_window_dynamic_eq(&problem, &before);
    assert_eq!(problem.trial_host_order, before.trial_host_order);
}

#[test]
fn clean_trial_view_imu_factor_is_bit_exact_to_diagnostic_clone() {
    let mut problem = test_window(vec![nav(0, 0.0), nav(1, 0.1)], Vec::new());
    let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
    delta.delta_time = 0.01;
    for axis in 0..9 {
        delta.covariance[(axis, axis)] = 1.0;
    }
    problem.imu_links.push(WindowImuLink {
        from_index: 0,
        to_index: 1,
        delta,
    });
    let state = problem.initial_state();
    let mut step = DVector::zeros(problem.state_dof());
    step[15] = 0.015625;
    step[27] = -0.03125;
    let trial = problem.apply_step(&state, &step);
    let owned = problem.trial_problem_with_step_owned(&state, &step);
    let owned_cost = owned.owned_trial_objective(&trial).unwrap();
    let view_cost = WindowTrialView::from_step(&problem, &state, &step, &trial)
        .cost()
        .unwrap();
    assert_eq!(view_cost.to_bits(), owned_cost.to_bits());
}

#[test]
fn global_state_layout_is_fifteen_columns_per_state() {
    let states = vec![nav(0, 0.0), nav(1, 1.0), nav(2, 2.0)];
    let x = flatten_states(&states);
    assert_eq!(x.len(), 45);
    assert_eq!(x[15], 1.0);
    assert_eq!(x[30], 2.0);
}

#[test]
fn upstream_f32_window_state_roundtrip_preserves_so3_components() {
    let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
        0.5944822430610657_f64,
        -0.052778493613004684_f64,
        -0.8023747801780701_f64,
        0.0,
    ));
    let state = BasaltNavState {
        imu_to_world: SE3::new(rotation, Vector3::zeros()),
        ..BasaltNavState::default()
    };
    let encoded = flatten_nav_with_mode(&state, ScalarMode::UpstreamF32);
    let decoded = decode_state_with_mode(&encoded, ScalarMode::UpstreamF32);
    let before = state.imu_to_world.rotation.quaternion();
    let after = decoded.imu_to_world.rotation.quaternion();
    assert_eq!(after.w, before.w);
    assert_eq!(after.i, before.i);
    assert_eq!(after.j, before.j);
    assert_eq!(after.k, before.k);
}

#[test]
fn mixed_pose_then_state_layout_is_six_plus_fifteen() {
    let states = vec![nav(7, 1.0), nav(8, 2.0)];
    let poses = vec![WindowPose {
        frame_id: 0,
        timestamp_ns: 1,
        pose: SE3::identity(),
        stored_current_pose: SE3::identity(),
        linearized_pose: SE3::identity(),
        linearized_delta: DVector::zeros(POSE_DOF),
        is_keyframe: true,
    }];
    let problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap(),
        cameras: Vec::new(),
        t_imu_cam: Vec::new(),
        poses,
        states,
        landmarks: Vec::new(),
        imu_links: Vec::new(),
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0,
        initial_accel_bias_weight: 1.0,
        initial_gyro_bias_weight: 1.0,
        prior: None,
        anchor_point: None,
        gravity_world: Vector3::zeros(),
        scalar_mode: ScalarMode::ExtendedF64,
        factor_config: FactorConfig::default(),
        visual_only: None,
    };
    assert_eq!(problem.state_dof(), 36);
    let x = problem.initial_state();
    assert_eq!(x.len(), 36);
    assert_eq!(problem.state_offset(0), 6);
    assert_eq!(problem.state_offset(1), 21);
    assert_eq!(problem.block_frame_id(0), Some(0));
    assert_eq!(problem.block_frame_id(1), Some(7));
}

#[test]
fn lm_pose_step_matches_upstream_additive_translation_left_rotation() {
    let mut base = nav(0, 2.0);
    base.nav.imu_to_world.rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.4));
    let problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap(),
        cameras: Vec::new(),
        t_imu_cam: Vec::new(),
        poses: Vec::new(),
        states: vec![base.clone()],
        landmarks: Vec::new(),
        imu_links: Vec::new(),
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0,
        initial_accel_bias_weight: 1.0,
        initial_gyro_bias_weight: 1.0,
        prior: None,
        anchor_point: None,
        gravity_world: Vector3::zeros(),
        scalar_mode: ScalarMode::ExtendedF64,
        factor_config: FactorConfig::default(),
        visual_only: None,
    };
    let mut step = DVector::zeros(NAV_STATE_DOF);
    step[0] = 0.3;
    step[5] = 0.2;
    let updated = problem.apply_step(&problem.initial_state(), &step);
    let decoded = decode_state(updated.rows(0, NAV_STATE_DOF));
    let expected_rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.2))
        * base.nav.imu_to_world.rotation;
    assert_eq!(
        decoded.imu_to_world.translation,
        base.nav.imu_to_world.translation + Vector3::new(0.3, 0.0, 0.0)
    );
    assert!((decoded.imu_to_world.rotation.inverse() * expected_rotation).angle() < 1e-12);
}

#[test]
fn upstream_f32_pose_apply_reconstructs_from_linearized_delta() {
    // Pose-only keyframes use Basalt's PoseStateWithLin contract: the
    // accepted increments are accumulated in a binary32 tangent and the
    // candidate pose is rebuilt from the frozen linearization pose.  A
    // direct update of the current pose is observably different once the
    // second rotational increment is applied.
    let linearized_pose = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.17, -0.23, 0.31)),
        Vector3::new(3.8816469, -2.5, 3.75),
    );
    let first_increment = DVector::from_column_slice(&[
        0.8826645_f64,
        -0.375_f64,
        0.625_f64,
        0.013_f64,
        -0.021_f64,
        0.034_f64,
    ]);
    let first_translation = vec3_f32(linearized_pose.translation)
        + Vector3::from_iterator(
            first_increment
                .rows(0, 3)
                .iter()
                .copied()
                .map(|value| value as f32),
        );
    let first_rotation = sophus_so3_product(
        so3_exp_f32(Vector3::from_iterator(
            first_increment
                .rows(3, 3)
                .iter()
                .copied()
                .map(|value| value as f32),
        )),
        quat_f32_from_f64(&linearized_pose.rotation),
    );
    let pose = WindowPose {
        frame_id: 0,
        timestamp_ns: 1,
        pose: SE3::new(
            quat_f64_from_f32(&first_rotation),
            vec3_f64(first_translation),
        ),
        stored_current_pose: SE3::new(
            quat_f64_from_f32(&first_rotation),
            vec3_f64(first_translation),
        ),
        linearized_pose: linearized_pose.clone(),
        linearized_delta: first_increment.clone(),
        is_keyframe: true,
    };
    let problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap(),
        cameras: Vec::new(),
        t_imu_cam: Vec::new(),
        poses: vec![pose.clone()],
        states: Vec::new(),
        landmarks: Vec::new(),
        imu_links: Vec::new(),
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0,
        initial_accel_bias_weight: 1.0,
        initial_gyro_bias_weight: 1.0,
        prior: None,
        anchor_point: None,
        gravity_world: Vector3::zeros(),
        scalar_mode: ScalarMode::UpstreamF32,
        factor_config: FactorConfig::default(),
        visual_only: None,
    };
    let second_increment = DVector::from_column_slice(&[
        0.8693182_f64,
        0.1875_f64,
        -0.3125_f64,
        -0.019_f64,
        0.027_f64,
        -0.011_f64,
    ]);
    let (updated, _) = problem.apply_step_full(&problem.initial_state(), &second_increment);
    let actual = &updated[0];

    let mut accumulated = first_increment;
    for (lhs, rhs) in accumulated.iter_mut().zip(second_increment.iter()) {
        *lhs = ((*lhs as f32) + (*rhs as f32)) as f64;
    }
    let expected_translation = vec3_f32(linearized_pose.translation)
        + Vector3::from_iterator(
            accumulated
                .rows(0, 3)
                .iter()
                .copied()
                .map(|value| value as f32),
        );
    let expected_rotation = sophus_so3_product(
        so3_exp_f32(Vector3::from_iterator(
            accumulated
                .rows(3, 3)
                .iter()
                .copied()
                .map(|value| value as f32),
        )),
        quat_f32_from_f64(&linearized_pose.rotation),
    );
    let actual_q = actual.rotation.quaternion();
    let expected_q = expected_rotation.quaternion();
    assert_eq!(
        actual.translation.map(|value| (value as f32).to_bits()),
        expected_translation.map(|value| value.to_bits())
    );
    assert_eq!(
        [actual_q.i, actual_q.j, actual_q.k, actual_q.w].map(|value| (value as f32).to_bits()),
        [expected_q.i, expected_q.j, expected_q.k, expected_q.w].map(|value| value.to_bits())
    );

    // This is the pre-fix behavior: apply only the new increment to the
    // already-updated pose.  Keep the assertion explicit so a future
    // simplification cannot silently lose the accumulated-delta rule.
    let naive_translation = vec3_f32(pose.pose.translation)
        + Vector3::from_iterator(
            second_increment
                .rows(0, 3)
                .iter()
                .copied()
                .map(|value| value as f32),
        );
    let naive_rotation = sophus_so3_product(
        so3_exp_f32(Vector3::from_iterator(
            second_increment
                .rows(3, 3)
                .iter()
                .copied()
                .map(|value| value as f32),
        )),
        quat_f32_from_f64(&pose.pose.rotation),
    );
    let naive_q = naive_rotation.quaternion();
    assert_ne!(
        [actual_q.i, actual_q.j, actual_q.k, actual_q.w].map(|value| (value as f32).to_bits()),
        [naive_q.i, naive_q.j, naive_q.k, naive_q.w].map(|value| value.to_bits())
    );
    assert_ne!(
        actual.translation.map(|value| (value as f32).to_bits()),
        naive_translation.map(|value| value.to_bits())
    );
}

#[test]
fn pose_only_so3_exp_matches_frame174_portable_oracle() {
    // MH_01 frame 174, iteration 0, pose-only block 112. Both targets
    // receive these exact FEJ-base and accumulated-tangent bits.
    let f = f32::from_bits;
    let before = UnitQuaternion::new_unchecked(Quaternion::new(
        f(0x3f16_aba4),
        f(0xbd94_e019),
        f(0xbf4d_7c08),
        f(0xbd82_0a27),
    ));
    let omega = Vector3::new(f(0xbba4_3bd6), f(0xbc41_ea40), f(0xb91b_19cc));
    let delta = so3_exp_f32(omega);
    assert_eq!(
        [delta.i, delta.j, delta.k, delta.w].map(f32::to_bits),
        [0xbb24_3b8c, 0xbbc1_e9e9, 0xb89b_1986, 0x3f7f_fea6]
    );
    let updated = sophus_so3_product(delta, before);
    assert_eq!(
        [updated.i, updated.j, updated.k, updated.w].map(f32::to_bits),
        [0xbd97_3eb7, 0xbf4e_6948, 0xbd7d_c672, 0x3f15_674c]
    );
}

#[test]
fn upstream_f32_state_update_quaternion_matches_native_sophus_bits() {
    let cases = [
        (
            [0xbd5cdea6, 0xbf4d341f, 0xbb2aa78b, 0x3f186f67],
            [0xb92b55aa, 0xb9418699, 0x38359f2f],
            [0xb8ab55aa, 0xb8c18699, 0x37b59f2f, 0x3f800000],
            [0xbd5cff35, 0xbf4d37d1, 0xbb25d808, 0x3f186a45],
        ),
        (
            [0xbd5cf5f2, 0xbf4d97c0, 0xbbac4997, 0x3f17e7a6],
            [0xb84253bc, 0xb99a78ae, 0x387d38e4],
            [0xb7c253bc, 0xb91a78ae, 0x37fd38e4, 0x3f800000],
            [0xbd5cea20, 0xbf4d9d98, 0xbbab59ef, 0x3f17dfd3],
        ),
        (
            [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
            [0x38781aee, 0xb9dc2926, 0x38c40d56],
            [0x37f81aee, 0xb95c2926, 0x38440d56, 0x3f800000],
            [0xbd5e3af5, 0xbf4e66c7, 0xbbc319ff, 0x3f16cb91],
        ),
        (
            [0xbd5f79e6, 0xbf4f45a2, 0xbbb53f68, 0x3f159719],
            [0x3910e3ec, 0xba0d5f2c, 0x3903baf9],
            [0x3890e3ec, 0xb98d5f2c, 0x3883baf9, 0x3f7fffff],
            [0xbd5f18ac, 0xbf4f5028, 0xbbb65c29, 0x3f15890f],
        ),
    ];
    for (q_bits, omega_bits, exp_bits, result_bits) in cases {
        let q = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(q_bits[3]),
            f32::from_bits(q_bits[0]),
            f32::from_bits(q_bits[1]),
            f32::from_bits(q_bits[2]),
        ));
        let omega = Vector3::new(
            f32::from_bits(omega_bits[0]),
            f32::from_bits(omega_bits[1]),
            f32::from_bits(omega_bits[2]),
        );
        let delta = so3_exp_f32(omega);
        let result = sophus_so3_product(delta, q);
        let bits = |value: &UnitQuaternion<f32>| {
            [
                value.i.to_bits(),
                value.j.to_bits(),
                value.k.to_bits(),
                value.w.to_bits(),
            ]
        };
        assert_eq!(bits(&delta), exp_bits);
        assert_eq!(bits(&result), result_bits);
    }
}

#[test]
fn upstream_f32_frame3_iter0_apply_step_matches_native_accepted_state() {
    let f64_from_bits = |bits: u32| f32::from_bits(bits) as f64;
    let pre = BasaltNavState {
        imu_to_world: SE3::new(
            UnitQuaternion::new_unchecked(Quaternion::new(
                f64_from_bits(0x3f16d68a),
                f64_from_bits(0xbd5e760f),
                f64_from_bits(0xbf4e5e85),
                f64_from_bits(0xbbc2d95f),
            )),
            Vector3::new(
                f64_from_bits(0x3b8ebfa0),
                f64_from_bits(0xba9428f4),
                f64_from_bits(0xbc857642),
            ),
        ),
        velocity_world_m_s: Vector3::new(
            f64_from_bits(0x3d850448),
            f64_from_bits(0xbc998d12),
            f64_from_bits(0xbe4a560c),
        ),
        ..BasaltNavState::default()
    };
    let increment_bits = [
        0xbae43978, 0xbb07ab82, 0xbd09c073, 0x38781648, 0xb9dc3846, 0x38c40d43, 0xbc475478,
        0xbc60489e, 0xbe67ad09, 0x38e89ae4, 0x3ab2b6ec, 0x3ab509db, 0x3a29b4c8, 0xb8b7e658,
        0xb9d1c9ee,
    ];
    let expected_bits = [
        0x3b2b6284, 0xbb51bffc, 0xbd4c7b94, 0xbd5e3af5, 0xbf4e66c7, 0xbbc31a02, 0x3f16cb91,
        0x3d583372, 0xbd04d8b0, 0xbed9018a, 0x38e89ae4, 0x3ab2b6ec, 0x3ab509db, 0x3a29b4c8,
        0xb8b7e658, 0xb9d1c9ee,
    ];
    let state = WindowState {
        frame_id: 3,
        timestamp_ns: 1,
        nav: pre.clone(),
        stored_current_nav: pre.clone(),
        linearized_nav: pre,
        linearized_delta: DVector::zeros(NAV_STATE_DOF),
        is_keyframe: false,
        is_latest: true,
        linearized: false,
    };
    let problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap(),
        cameras: Vec::new(),
        t_imu_cam: Vec::new(),
        poses: Vec::new(),
        states: vec![state],
        landmarks: Vec::new(),
        imu_links: Vec::new(),
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0,
        initial_accel_bias_weight: 1.0,
        initial_gyro_bias_weight: 1.0,
        prior: None,
        anchor_point: None,
        gravity_world: Vector3::zeros(),
        scalar_mode: ScalarMode::UpstreamF32,
        factor_config: FactorConfig::default(),
        visual_only: None,
    };
    let step = DVector::from_iterator(NAV_STATE_DOF, increment_bits.into_iter().map(f64_from_bits));
    let (_, updated_states) = problem.apply_step_full(&problem.initial_state(), &step);
    let updated = &updated_states[0];
    let q = updated.imu_to_world.rotation.quaternion();
    let mut actual = Vec::with_capacity(16);
    actual.extend(
        updated
            .imu_to_world
            .translation
            .iter()
            .map(|x| (*x as f32).to_bits()),
    );
    actual.extend([q.i, q.j, q.k, q.w].map(|x| (x as f32).to_bits()));
    actual.extend(
        updated
            .velocity_world_m_s
            .iter()
            .map(|x| (*x as f32).to_bits()),
    );
    actual.extend(
        updated
            .gyro_bias_rad_s
            .iter()
            .map(|x| (*x as f32).to_bits()),
    );
    actual.extend(
        updated
            .accel_bias_m_s2
            .iter()
            .map(|x| (*x as f32).to_bits()),
    );
    assert_eq!(actual, expected_bits);
}

#[test]
fn upstream_f32_current_pose_zero_inc_normalizes_accepted_state_bits() {
    // `BundleAdjustmentBase::getPoseStateWithLin` materializes a
    // non-linearized navigation state as a temporary PoseStateWithLin.
    // Its constructor calls PoseState::incPose with a zero tangent; the
    // Sophus product still crosses the float quaternion normalization
    // boundary.  The accepted compact state below is authoritative
    // frame 3 / frame-4 iter0 from the native GDB capture.
    let accepted = UnitQuaternion::new_unchecked(Quaternion::new(
        f32::from_bits(0x3f16cb91),
        f32::from_bits(0xbd5e3af5),
        f32::from_bits(0xbf4e66c7),
        f32::from_bits(0xbbc31a02),
    ));
    let current = sophus_so3_product(so3_exp_f32(Vector3::zeros()), accepted);
    let q = current.quaternion();
    assert_eq!(
        [q.i, q.j, q.k, q.w].map(f32::to_bits),
        [0xbd5e3af6, 0xbf4e66c8, 0xbbc31a03, 0x3f16cb92]
    );
}

#[test]
fn m11_frame34_boundary_state_history_pose_matches_native_zero_inc_bits() {
    // Native frame34 LM writeback and Rust agree on this raw frame33
    // state.  At the following frame35 measure(), frame33 has become the
    // FEJ boundary and getPoseStateWithLin() applies its zero delta before
    // delayed triangulation.  The resulting native operand was captured
    // at libbasalt 0x4e4960 for track1821.
    let raw_pose = SE3::new(
        UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f1b6492) as f64,
            f32::from_bits(0xbd52b57d) as f64,
            f32::from_bits(0xbf4ad7a3) as f64,
            f32::from_bits(0xbd06774d) as f64,
        )),
        Vector3::new(
            f32::from_bits(0xbc8eaa38) as f64,
            f32::from_bits(0xbc2aec38) as f64,
            f32::from_bits(0x3e1236c5) as f64,
        ),
    );
    let state = WindowState {
        frame_id: 33,
        timestamp_ns: 1403636581413555456,
        nav: BasaltNavState {
            imu_to_world: raw_pose.clone(),
            ..BasaltNavState::default()
        },
        stored_current_nav: BasaltNavState {
            imu_to_world: raw_pose.clone(),
            ..BasaltNavState::default()
        },
        linearized_nav: BasaltNavState {
            imu_to_world: raw_pose,
            ..BasaltNavState::default()
        },
        linearized_delta: DVector::zeros(NAV_STATE_DOF),
        is_keyframe: true,
        is_latest: false,
        linearized: true,
    };
    let effective = visual_state_pose_with_mode(&state, ScalarMode::UpstreamF32);
    let q = effective.rotation.quaternion();
    assert_eq!(
        [q.i, q.j, q.k, q.w].map(|value| (value as f32).to_bits()),
        [0xbd52b57b, 0xbf4ad7a1, 0xbd06774c, 0x3f1b6491]
    );
    assert_eq!(
        effective.translation.map(|value| (value as f32).to_bits()),
        Vector3::new(0xbc8eaa38, 0xbc2aec38, 0x3e1236c5)
    );
}

#[test]
fn upstream_f32_visual_get_pose_reconstructs_native_linearized_current() {
    // The pinned native `getPoseStateWithLin()` constructor always
    // applies the stored delta to `pose_linearized`, including a zero
    // delta.  These two cases are copied from
    // `m7im15_gdb_frame6_getpose_frame4_20260828.json`: the first is the
    // frame-6/iter-0 frame-4 state and the second is a later nonzero
    // frame-4 delta.  This test exercises the actual visual accessor,
    // rather than only testing the quaternion helper in isolation.
    let pose_from_bits = |translation: [u32; 3], q_xyzw: [u32; 4]| {
        SE3::new(
            UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q_xyzw[3]) as f64,
                f32::from_bits(q_xyzw[0]) as f64,
                f32::from_bits(q_xyzw[1]) as f64,
                f32::from_bits(q_xyzw[2]) as f64,
            )),
            Vector3::from_iterator(
                translation
                    .into_iter()
                    .map(|bits| f32::from_bits(bits) as f64),
            ),
        )
    };
    let make_problem = |linearized_pose: SE3, linearized: bool, delta_bits: [u32; 6]| {
        let nav = BasaltNavState {
            imu_to_world: linearized_pose.clone(),
            ..BasaltNavState::default()
        };
        let mut delta = DVector::zeros(NAV_STATE_DOF);
        for (slot, bits) in delta_bits.into_iter().enumerate() {
            delta[slot] = f32::from_bits(bits) as f64;
        }
        WindowProblem {
            trial_host_order: Vec::new(),
            camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap(),
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses: Vec::new(),
            states: vec![WindowState {
                frame_id: 4,
                timestamp_ns: 1403636579963555584,
                stored_current_nav: nav.clone(),
                nav,
                linearized_nav: BasaltNavState {
                    imu_to_world: linearized_pose,
                    ..BasaltNavState::default()
                },
                linearized_delta: delta,
                is_keyframe: false,
                is_latest: true,
                linearized,
            }],
            landmarks: Vec::new(),
            imu_links: Vec::new(),
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0,
            initial_accel_bias_weight: 1.0,
            initial_gyro_bias_weight: 1.0,
            prior: None,
            anchor_point: None,
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::UpstreamF32,
            factor_config: FactorConfig::default(),
            visual_only: None,
        }
    };
    let raw_pose = pose_from_bits(
        [0xbb21be47, 0xbb8635a5, 0xbdb353f2],
        [0xbd5d86c4, 0xbf511117, 0xbbd647bb, 0x3f13148c],
    );

    let zero_problem = make_problem(raw_pose.clone(), true, [0; 6]);
    let zero_current = zero_problem
        .block_nav_visual_current(&DVector::zeros(NAV_STATE_DOF), 0)
        .unwrap();
    let zero_q = zero_current.imu_to_world.rotation.quaternion();
    assert_eq!(
        zero_current
            .imu_to_world
            .translation
            .map(|value| (value as f32).to_bits()),
        Vector3::new(0xbb21be47_u32, 0xbb8635a5_u32, 0xbdb353f2_u32)
    );
    assert_eq!(
        [zero_q.i, zero_q.j, zero_q.k, zero_q.w].map(|value| (value as f32).to_bits()),
        [0xbd5d86c2, 0xbf511115, 0xbbd647b9, 0x3f13148b]
    );

    let nonzero_problem = make_problem(
        raw_pose,
        true,
        [
            0x37bd51ec, 0xb49be800, 0x39cee3c6, 0x3a42b20c, 0xb98b69f0, 0x37b3f87b,
        ],
    );
    let nonzero_current = nonzero_problem
        .block_nav_visual_current(&DVector::zeros(NAV_STATE_DOF), 0)
        .unwrap();
    let nonzero_q = nonzero_current.imu_to_world.rotation.quaternion();
    assert_eq!(
        nonzero_current
            .imu_to_world
            .translation
            .map(|value| (value as f32).to_bits()),
        Vector3::new(0xbb2043a3_u32, 0xbb863815_u32, 0xbdb2850e_u32)
    );
    assert_eq!(
        [nonzero_q.i, nonzero_q.j, nonzero_q.k, nonzero_q.w].map(|value| (value as f32).to_bits()),
        [0xbd5c9cf2, 0xbf5115f9, 0xbbe0405f, 0x3f130ebf]
    );

    // The non-linearized branch is deliberately still the raw frozen
    // pose, even if a stale delta happens to be present in a fixture.
    let false_problem = make_problem(
        pose_from_bits(
            [0xbb21be47, 0xbb8635a5, 0xbdb353f2],
            [0xbd5d86c4, 0xbf511117, 0xbbd647bb, 0x3f13148c],
        ),
        false,
        [
            0x37bd51ec, 0xb49be800, 0x39cee3c6, 0x3a42b20c, 0xb98b69f0, 0x37b3f87b,
        ],
    );
    let false_current = false_problem
        .block_nav_visual_current(&DVector::zeros(NAV_STATE_DOF), 0)
        .unwrap();
    let false_q = false_current.imu_to_world.rotation.quaternion();
    assert_eq!(
        [false_q.i, false_q.j, false_q.k, false_q.w].map(|value| (value as f32).to_bits()),
        [0xbd5d86c4, 0xbf511117, 0xbbd647bb, 0x3f13148c]
    );
}

#[test]
fn fej_local_boxminus_is_zero_and_first_order_consistent() {
    let reference = BasaltNavState {
        imu_to_world: SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.1, -0.2, 0.3)),
            Vector3::new(1.0, -2.0, 0.5),
        ),
        velocity_world_m_s: Vector3::new(0.2, 0.3, -0.1),
        ..BasaltNavState::default()
    };
    let zero = local_state_difference(&reference, &reference);
    assert!(zero.norm() < 1e-12);
    let increment = Vector6::new(1e-4, -2e-4, 3e-4, 2e-5, -3e-5, 4e-5);
    let mut moved = reference.clone();
    moved.imu_to_world.translation += Vector3::new(increment[0], increment[1], increment[2]);
    moved.imu_to_world.rotation =
        UnitQuaternion::from_scaled_axis(Vector3::new(increment[3], increment[4], increment[5]))
            * moved.imu_to_world.rotation;
    let recovered = local_state_difference(&moved, &reference);
    assert!((recovered.rows(0, 6) - increment).norm() < 1e-10);
}

#[test]
fn marginal_prior_error_drops_constant_and_allows_negative_values() {
    let jacobian = DMatrix::from_diagonal(&DVector::from_column_slice(&[2.0, 3.0]));
    let rhs = DVector::from_column_slice(&[-10.0, 1.0]);
    let delta = DVector::from_column_slice(&[1.0, 0.0]);

    // J*delta + r has a positive complete squared cost, while Basalt's
    // linearized error intentionally drops 0.5*rᵀr and is negative.
    let complete = 0.5
        * (&jacobian * &delta + &rhs)
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
    assert_eq!(complete, 32.5);
    assert_eq!(
        upstream_marginal_prior_error(&jacobian, &rhs, &delta, ScalarMode::ExtendedF64,),
        -18.0
    );
    assert_eq!(
        upstream_marginal_prior_error(&jacobian, &rhs, &delta, ScalarMode::UpstreamF32,).to_bits(),
        (-18.0_f32 as f64).to_bits()
    );
}

#[test]
fn upstream_sqrt_marginalization_uses_marg_first_order_and_sqrt_eps_cutoff() {
    let jacobian = DMatrix::from_row_slice(
        4,
        3,
        &[
            1.0, 0.2, 2.0, -0.5, 1.5, 0.3, 0.7, -0.4, 1.2, 0.1, 0.8, -0.9,
        ],
    );
    let rhs = DVector::from_column_slice(&[0.4, -1.2, 0.7, 2.1]);
    // Deliberately non-sorted input exercises the same set-order contract
    // as upstream's std::set indices after the caller has built AOM lists.
    let prior = sqrt_marginalize_upstream(
        &jacobian,
        &rhs,
        &[2],
        &[1, 0],
        DVector::from_element(1, 3.0),
    )
    .expect("full-rank synthetic prior");
    assert_eq!(prior.jacobian.ncols(), 1);
    assert_eq!(prior.jacobian.nrows(), 1);
    assert_eq!(prior.fej_point, DVector::from_element(1, 3.0));

    // Compare the emitted square-root prior's normal terms against the
    // Schur complement of the same un-squared rows. The sign of a
    // Householder row is intentionally free, so compare Gram products.
    let h = jacobian.transpose() * &jacobian;
    let b = jacobian.transpose() * rhs;
    let hmm = h.view((0, 0), (2, 2)).into_owned();
    let hmk = h.view((0, 2), (2, 1)).into_owned();
    let hkm = h.view((2, 0), (1, 2)).into_owned();
    let hkk = h[(2, 2)];
    let bm = b.rows(0, 2).into_owned();
    let bk = b[2];
    let schur_h = hkk - (hkm.clone() * hmm.clone().try_inverse().unwrap() * hmk)[(0, 0)];
    let schur_b = bk - (hkm * hmm.try_inverse().unwrap() * bm)[0];
    assert!((prior.jacobian[(0, 0)].powi(2) - schur_h).abs() < 1e-10);
    assert!((prior.jacobian[(0, 0)] * prior.rhs[0] - schur_b).abs() < 1e-10);

    let tiny_marginal = DMatrix::from_column_slice(3, 1, &[0.5e-8, 0.0, 0.0]);
    let tiny_keep = DMatrix::from_column_slice(3, 1, &[0.0, 2.0, 3.0]);
    let mut stacked = DMatrix::zeros(3, 2);
    stacked.column_mut(0).copy_from(&tiny_marginal.column(0));
    stacked.column_mut(1).copy_from(&tiny_keep.column(0));
    let tiny =
        sqrt_marginalize_upstream(&stacked, &DVector::zeros(3), &[1], &[0], DVector::zeros(1))
            .expect("rank-deficient marginal block is retained as an explicit zero row");
    assert_eq!(tiny.jacobian.nrows(), 1);
    assert!((tiny.jacobian[(0, 0)].abs() - tiny_keep.column(0).norm()).abs() < 1e-12);
}

fn assert_f32_golden(actual: f64, expected_bits: u32) {
    let actual_bits = (actual as f32).to_bits();
    assert_eq!(
        actual_bits, expected_bits,
        "actual={actual:.9e} bits=0x{actual_bits:08x}, expected=0x{expected_bits:08x}"
    );
}

#[test]
fn upstream_f32_sqrt_marginalization_matches_eigen_golden_boundaries() {
    // These fixtures are emitted by
    // benchmarks/basalt/upstream_sqrt_marginalization_oracle.cpp against
    // the pinned Basalt Eigen 5.0.1 checkout.  Compare float bit patterns,
    // not widened decimal text, so a change in reflector storage, sign,
    // accumulation precision, or rank cutoff is visible.  The expected
    // values are bound to the pinned clean native O3/AVX build
    // (GCC 11.4, -O3 -march=native, Eigen 5.0.1); the older default
    // SSE2/Packet4 fixture is retained as historical evidence only.
    let cases = [
        (
            "normal",
            DMatrix::from_row_slice(
                5,
                4,
                &[
                    0.75, -1.2, 0.25, -0.7, -0.35, 0.8, 1.3, 0.2, 1.1, 0.15, -0.6, 0.9, -0.45,
                    -0.55, 0.7, -1.4, 0.2, 1.05, -0.9, 0.35,
                ],
            ),
            DVector::from_column_slice(&[0.4, -1.2, 0.7, 2.1, -0.3]),
            vec![0, 3],
            vec![1, 2],
            (2, 2),
            vec![3214139916, 3208334584, 0, 1064423251],
            vec![1048055211, 3215223833],
        ),
        (
            "near_rank",
            DMatrix::from_row_slice(
                4,
                3,
                &[
                    1.0e-4, 0.4, -0.7, 2.0e-4, -0.2, 0.9, -1.0e-4, 0.8, 0.1, 0.0, -0.5, 0.3,
                ],
            ),
            DVector::from_column_slice(&[0.25, -0.5, 0.75, -1.0]),
            vec![0, 2],
            vec![1],
            (1, 2),
            vec![0, 3213413843],
            vec![1045480110],
        ),
        (
            "sign",
            DMatrix::from_row_slice(4, 2, &[-0.8, 0.3, 0.25, -1.1, -0.45, 0.6, 0.7, 0.2]),
            DVector::from_column_slice(&[0.6, -0.4, 1.2, 0.1]),
            vec![1],
            vec![0],
            (1, 1),
            vec![1066896430],
            vec![1060968209],
        ),
        (
            "zero",
            DMatrix::from_row_slice(3, 2, &[0.0, 0.5, 0.0, -1.0, 0.0, 0.25]),
            DVector::from_column_slice(&[0.2, -0.3, 0.4]),
            vec![1],
            vec![0],
            (1, 1),
            vec![3214058614],
            vec![3202315395],
        ),
        (
            "denormal",
            DMatrix::from_row_slice(
                3,
                2,
                &[f64::from(f32::from_bits(1)), 0.5, 0.0, -1.0, 0.0, 0.25],
            ),
            DVector::from_column_slice(&[0.2, -0.3, 0.4]),
            vec![1],
            vec![0],
            (1, 1),
            vec![3214058614],
            vec![3202315395],
        ),
    ];

    for (
        name,
        jacobian,
        rhs,
        keep,
        marg,
        (expected_rows, expected_cols),
        expected_j,
        expected_rhs,
    ) in cases
    {
        let prior = sqrt_marginalize_upstream_f32(
            &jacobian,
            &rhs,
            &keep,
            &marg,
            DVector::zeros(keep.len()),
        )
        .unwrap_or_else(|| panic!("{name} fixture unexpectedly rejected"));
        assert_eq!(
            (prior.jacobian.nrows(), prior.jacobian.ncols()),
            (expected_rows, expected_cols),
            "{name} dimensions"
        );
        assert_eq!(prior.rhs.len(), expected_rhs.len(), "{name} rhs dimensions");
        for row in 0..expected_rows {
            for column in 0..expected_cols {
                assert_f32_golden(
                    prior.jacobian[(row, column)],
                    expected_j[row * expected_cols + column],
                );
            }
        }
        for (actual, expected) in prior.rhs.iter().zip(expected_rhs) {
            assert_f32_golden(*actual, expected);
        }
    }
}

#[test]
fn upstream_f32_packet8_squared_norm_matches_m7im15_k12_eigen_bits() {
    // Q2 column 12 at the exact 96x60 M7IM15 boundary.  The production
    // Householder call norms the 83-element tail (excluding c0), while
    // the native diagnostic norms the full 84-element column segment.
    let tail_bits: [u32; 83] = [
        0xc3667f7b, 0xc3ab921f, 0xc3878255, 0xc3aaf1f2, 0x4295b589, 0x42a1a78d, 0x41f80a43,
        0x42a054b4, 0x4291e512, 0x428a9698, 0x00000000, 0x00000000, 0x00000000, 0xc30220a8,
        0xc3150c1c, 0x43b06b4d, 0xc3647953, 0x43b0f9bf, 0x41d51baa, 0xc31fd311, 0x00000000,
        0x00000000, 0x00000000, 0xc29dce23, 0xc0fa82ee, 0xc33a4aef, 0xc3462aef, 0xc191cd1f,
        0xc3a20a37, 0xc4aa2e5e, 0x466b58f6, 0x4594b741, 0x43bb27b6, 0x439bd043, 0x432bda22,
        0xc2b7f5db, 0xc2358289, 0xc2b4c9d9, 0x42eb8178, 0x40faa3ee, 0x412a6e92, 0x426e3b4e,
        0x40aefdd3, 0x3e5605f7, 0x44b480e6, 0xc66a71e4, 0xc59a0497, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x3cd8d2c8, 0x3cb4836f, 0x3c47182d, 0xbe529c20, 0xbdcfcdec, 0xbe4efa7d,
    ];
    let tail = tail_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect::<Vec<_>>();
    assert_eq!(
        upstream_f32_packet8_squared_norm(&tail, 0, tail.len()).to_bits(),
        0x4df06330
    );

    let mut full = Vec::with_capacity(tail.len() + 1);
    full.push(f32::from_bits(0x42c19836));
    full.extend_from_slice(&tail);
    assert_eq!(
        upstream_f32_packet8_squared_norm(&full, 0, full.len()).to_bits(),
        0x4df06454
    );
    // Keep this comparison explicitly non-contracted; it documents why
    // the source-level scalar expression is not the Eigen evaluator's
    // packet/FMA result on this boundary.
    let c0_sq = std::hint::black_box(full[0] * full[0]);
    let tail_sum = tail
        .iter()
        .fold(std::hint::black_box(0.0_f32), |sum, value| {
            let square = std::hint::black_box(*value * *value);
            std::hint::black_box(sum + square)
        });
    assert_eq!(std::hint::black_box(c0_sq + tail_sum).to_bits(), 0x4df06456);
}

#[test]
fn upstream_f32_packet8_squared_norm_matches_m7im15_all_reflector_tails() {
    // Each fixture row is the exact pre-reflector essential tail from the
    // 96x60 M7IM15 Q2 replay, followed by Eigen's tail.squaredNorm() bits.
    // Keeping all 60 rows catches packet-unroll and scalar-remainder
    // changes that happen to be invisible on the k12 tail alone.
    let mut lines =
        include_str!("../../../../../benchmarks/basalt/m7im15_sqrt_norm_fixture_v1.txt").lines();
    assert_eq!(lines.next(), Some("M7IM15_SQRT_NORM_V1"));
    let dimensions = lines
        .next()
        .expect("M7IM15 norm fixture dimensions")
        .split_whitespace()
        .map(|token| token.parse::<usize>().expect("fixture dimension"))
        .collect::<Vec<_>>();
    assert_eq!(dimensions, vec![60, 96, 60]);

    let mut count = 0usize;
    for line in lines {
        let mut tokens = line.split_whitespace();
        let column = tokens
            .next()
            .expect("fixture column")
            .parse::<usize>()
            .expect("fixture column number");
        let tail_len = tokens
            .next()
            .expect("fixture tail length")
            .parse::<usize>()
            .expect("fixture tail length number");
        let expected = u32::from_str_radix(tokens.next().expect("fixture expected norm"), 16)
            .expect("fixture expected norm bits");
        let tail = tokens
            .map(|token| f32::from_bits(u32::from_str_radix(token, 16).expect("fixture tail bits")))
            .collect::<Vec<_>>();
        assert_eq!(tail.len(), tail_len, "M7IM15 k{column} tail length");
        assert_eq!(
            upstream_f32_packet8_squared_norm(&tail, 0, tail_len).to_bits(),
            expected,
            "M7IM15 k{column} squared norm"
        );
        count += 1;
    }
    assert_eq!(count, dimensions[0]);
}

#[test]
fn upstream_f32_real_m6_packet_slice_matches_default_eigen() {
    // This is the complete real 69-row, 72-column
    // deterministic slice of the actual MH_01 frame-51 packet
    // (`m6_actual_packet_oracle_v1`, Rust `frame_000051.json`).  The
    // selected rows preserve all 15 leaving columns and reach the same
    // 54-row retained rank as the standalone pinned native O3 Eigen 5.0.1
    // oracle (GCC 11.4, -march=native, AVX Packet8 on the capture host).
    // The production `UpstreamF32` path is a safe packet-order emulation,
    // so this test compares strict f32 bits rather than decimal values.
    // The pinned C++ MargData packet contains only abs_H/abs_b (not sqrt
    // J/r); this fixture is the authoritative standalone SqrtToSqrt
    // boundary for the real packet's AOM rows.
    let mut tokens =
        include_str!("../../../../../benchmarks/basalt/m6_sqrt_boundary_frame51_v1.txt")
            .split_whitespace();
    assert_eq!(tokens.next(), Some("M6_FIXTURE_V1"));
    let rows = tokens.next().unwrap().parse::<usize>().unwrap();
    let cols = tokens.next().unwrap().parse::<usize>().unwrap();
    let marg_count = tokens.next().unwrap().parse::<usize>().unwrap();
    let marg = (0..marg_count)
        .map(|_| tokens.next().unwrap().parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let keep_count = tokens.next().unwrap().parse::<usize>().unwrap();
    let keep = (0..keep_count)
        .map(|_| tokens.next().unwrap().parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let selected_count = tokens.next().unwrap().parse::<usize>().unwrap();
    for _ in 0..selected_count {
        let _ = tokens.next().unwrap().parse::<usize>().unwrap();
    }
    let jacobian_values = (0..rows * cols)
        .map(|_| tokens.next().unwrap().parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    let jacobian = DMatrix::from_row_slice(rows, cols, &jacobian_values);
    let rhs = DVector::from_fn(rows, |_, _| tokens.next().unwrap().parse::<f64>().unwrap());
    assert!(tokens.next().is_none(), "fixture has trailing tokens");

    let prior =
        sqrt_marginalize_upstream_f32(&jacobian, &rhs, &keep, &marg, DVector::zeros(keep.len()))
            .expect("actual frame-51 slice must produce a retained prior");
    assert_eq!((prior.jacobian.nrows(), prior.jacobian.ncols()), (54, 57));
    assert_eq!(prior.rhs.len(), 54);

    let expected_j = [
        ((0, 0), 1_176_616_427),
        ((0, 1), 1_127_794_637),
        ((0, 56), 3_212_872_857),
        ((1, 0), 0),
        ((27, 28), 3_310_529_033),
        ((53, 56), 3_242_393_197),
    ];
    for &((row, column), bits) in &expected_j {
        assert_f32_golden(prior.jacobian[(row, column)], bits);
    }
    let expected_rhs = [(0, 3_215_659_848), (27, 1_080_990_045), (53, 3_217_203_652)];
    for &(row, bits) in &expected_rhs {
        assert_f32_golden(prior.rhs[row], bits);
    }

    // Hash every retained raw f32 bit, in row-major J order followed by
    // rhs order.  This keeps the full 3132-value gate compact while the
    // selected probes above make a first mismatch easy to diagnose.
    let mut digest = 0xcbf29ce484222325_u64;
    let mut value_count = 0usize;
    for row in 0..prior.jacobian.nrows() {
        for column in 0..prior.jacobian.ncols() {
            for byte in (prior.jacobian[(row, column)] as f32)
                .to_bits()
                .to_le_bytes()
            {
                digest ^= u64::from(byte);
                digest = digest.wrapping_mul(0x100000001b3);
            }
            value_count += 1;
        }
    }
    for value in prior.rhs.iter() {
        for byte in (*value as f32).to_bits().to_le_bytes() {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(0x100000001b3);
        }
        value_count += 1;
    }
    assert_eq!(value_count, 3132);
    assert_eq!(digest, 0xe4fa3a57cf2f00a1);
}

#[test]
fn imu_residual_has_cross_state_jacobian_blocks() {
    let states = vec![nav(0, 0.0), nav(1, 1.0)];
    let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
    delta.delta_time = 0.01;
    let problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap(),
        cameras: Vec::new(),
        t_imu_cam: Vec::new(),
        poses: Vec::new(),
        states,
        landmarks: Vec::new(),
        imu_links: vec![WindowImuLink {
            from_index: 0,
            to_index: 1,
            delta,
        }],
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0e8,
        initial_accel_bias_weight: 1.0e1,
        initial_gyro_bias_weight: 1.0e2,
        prior: None,
        anchor_point: Some(DVector::zeros(15)),
        gravity_world: Vector3::zeros(),
        scalar_mode: ScalarMode::ExtendedF64,
        factor_config: FactorConfig::default(),
        visual_only: None,
    };
    let x = problem.initial_state();
    let factors = problem.linearize(&x).unwrap().factors;
    let imu = factors.iter().find(|factor| factor.rows() == 9).unwrap();
    assert!(imu.state_jacobian.columns(0, 15).norm() > 0.0);
    assert!(imu.state_jacobian.columns(15, 15).norm() > 0.0);
}

#[test]
fn initial_prior_matches_basalt_pose_yaw_and_bias_columns() {
    let mut problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap(),
        cameras: Vec::new(),
        t_imu_cam: Vec::new(),
        poses: Vec::new(),
        states: vec![nav(0, 0.0)],
        landmarks: Vec::new(),
        imu_links: Vec::new(),
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0e8,
        initial_accel_bias_weight: 1.0e1,
        initial_gyro_bias_weight: 1.0e2,
        prior: None,
        anchor_point: Some(DVector::zeros(15)),
        gravity_world: Vector3::new(0.0, 0.0, -9.81),
        scalar_mode: ScalarMode::ExtendedF64,
        factor_config: FactorConfig::default(),
        visual_only: None,
    };
    let factors = problem.linearize(&problem.initial_state()).unwrap().factors;
    let anchor = factors.iter().find(|factor| factor.rows() == 15).unwrap();
    let expected_pose = 1.0e4;
    let expected_accel_bias = 10.0_f64.sqrt();
    let expected_gyro_bias = 1.0e2_f64.sqrt();
    for axis in 0..3 {
        assert!((anchor.state_jacobian[(axis, axis)] - expected_pose).abs() < 1e-12);
        assert!((anchor.state_jacobian[(9 + axis, 9 + axis)] - expected_accel_bias).abs() < 1e-12);
        assert!((anchor.state_jacobian[(12 + axis, 12 + axis)] - expected_gyro_bias).abs() < 1e-12);
    }
    assert!((anchor.state_jacobian[(5, 5)] - expected_pose).abs() < 1e-12);
    for column in [3, 4, 6, 7, 8] {
        assert_eq!(anchor.state_jacobian[(column, column)], 0.0);
    }
    // Keep this mutable binding so this test also exercises the same
    // linearization path used by LM callers without changing the public
    // constructor contract.
    problem.states[0].is_latest = true;
}

#[test]
fn window_imu_rows_are_whitened_and_bias_walk_uses_both_state_blocks() {
    let states = vec![nav(0, 0.0), nav(1, 1.0)];
    let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
    delta.delta_time = 0.1;
    delta.covariance = nalgebra::SMatrix::<f64, 9, 9>::identity() * 4.0;
    let mut problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap(),
        cameras: Vec::new(),
        t_imu_cam: Vec::new(),
        poses: Vec::new(),
        states,
        landmarks: Vec::new(),
        imu_links: vec![WindowImuLink {
            from_index: 0,
            to_index: 1,
            delta,
        }],
        imu_noise: ImuNoiseModel {
            gyro_density: 2.0,
            accel_density: 3.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 2.0,
            accel_density: 4.0,
        },
        initial_pose_weight: 1.0e8,
        initial_accel_bias_weight: 1.0e1,
        initial_gyro_bias_weight: 1.0e2,
        prior: None,
        anchor_point: Some(DVector::zeros(15)),
        gravity_world: Vector3::zeros(),
        scalar_mode: ScalarMode::ExtendedF64,
        factor_config: FactorConfig::default(),
        visual_only: None,
    };
    let initial = problem.initial_state();
    let linearization = problem.linearize(&initial).unwrap();
    let imu = linearization
        .factors
        .iter()
        .find(|factor| factor.rows() == 9)
        .expect("one whitened 9-row IMU factor");
    assert_eq!(imu.state_jacobian.ncols(), 30);
    assert!(imu.state_jacobian.columns(0, 15).norm() > 0.0);
    assert!(imu.state_jacobian.columns(15, 15).norm() > 0.0);
    assert!(imu.residual.iter().all(|value| value.is_finite()));
    // The upstream row contract is [position, rotation, velocity], and
    // the identity delta leaves the one-metre state displacement in
    // position.
    assert!(imu.residual.rows(0, 3).norm() > 0.1);
    assert!(imu.residual.rows(3, 3).norm() < 1e-12);
    assert!(imu.residual.rows(6, 3).norm() < 1e-12);
    // The IMU factor's left-local rotation convention is owned by
    // `imu::factors`; retain the cross-state contract without pinning the
    // pre-m7z tangent coefficient here.
    assert!(imu.state_jacobian[(3, 18)].abs() > 0.0);

    let bias = linearization
        .factors
        .iter()
        .find(|factor| factor.rows() == 6)
        .expect("one whitened 6-row bias walk factor");
    let gyro_weight = 2.0 / 0.1_f64.sqrt();
    let accel_weight = 4.0 / 0.1_f64.sqrt();
    assert!((bias.state_jacobian[(0, 9)] - gyro_weight).abs() < 1e-12);
    assert!((bias.state_jacobian[(0, 24)] + gyro_weight).abs() < 1e-12);
    assert!((bias.state_jacobian[(3, 12)] - accel_weight).abs() < 1e-12);
    assert!((bias.state_jacobian[(3, 27)] + accel_weight).abs() < 1e-12);

    let (_, trace) = problem
        .imu_factor_with_trace(&initial, problem.imu_links[0].clone())
        .expect("IMU adapter trace");
    assert!((trace.delta_time - 0.1).abs() < 1e-12);
    assert!((trace.residual_position[0] - 1.0).abs() < 1e-12);
    assert!(trace
        .residual_rotation
        .iter()
        .all(|value| value.abs() < 1e-12));
    assert!(trace
        .residual_velocity
        .iter()
        .all(|value| value.abs() < 1e-12));
    assert!((trace.whitened_norm - 0.5).abs() < 1e-12);
    assert!((trace.covariance_eigen_min - 4.0).abs() < 1e-12);
    assert!((trace.covariance_eigen_max - 4.0).abs() < 1e-12);

    let solved = problem.solve(initial, LmConfig::default()).unwrap();
    assert!(solved.state.iter().all(|value| value.is_finite()));
    assert!(solved.cost.is_finite());
}

#[test]
fn lean_with_factors_matches_retained_diagnostics_exactly() {
    // The online-mapper/MargData fast path must produce the same accepted
    // state and the same retained factor snapshot as the fully diagnostic
    // implementation, because MargData bytes depend on both.  Build a
    // visual landmark problem with observable geometry so the LM actually
    // iterates, then compare the two solve entry points field by field.
    let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
    let true_point = Point3::new(0.2, -0.1, 1.5);
    let mut landmarks = Vec::new();
    for track_id in 0..6u64 {
        landmarks.push(WindowLandmark {
            track_id,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction: StereographicDirection::from_bearing(Vector3::new(
                0.1 * track_id as f64,
                -0.05,
                1.0,
            ))
            .unwrap(),
            inverse_distance: 0.9,
            observations: vec![
                WindowObservation {
                    state_index: 0,
                    camera_id: 0,
                    pixel: camera.project(&true_point).unwrap(),
                },
                WindowObservation {
                    state_index: 1,
                    camera_id: 0,
                    pixel: camera.project(&true_point).unwrap(),
                },
            ],
        });
    }
    let make = || test_window(vec![nav(0, 0.0), nav(1, 0.2)], landmarks.clone());

    let initial = make().initial_state();
    let config = LmConfig::default();
    let mut retained_timing = TimingBreakdown::default();
    let mut lean_timing = TimingBreakdown::default();
    let retained = make()
        .solve_with_timing(initial.clone(), config, &mut retained_timing)
        .unwrap();
    let lean = make()
        .solve_lean_with_factors_with_timing(initial, config, &mut lean_timing)
        .unwrap();

    assert_eq!(lean.state, retained.state);
    assert_eq!(lean.cost, retained.cost);
    assert_eq!(lean.iterations, retained.iterations);
    assert_eq!(lean.factors.len(), retained.factors.len());
    for (lean_factor, retained_factor) in lean.factors.iter().zip(&retained.factors) {
        assert_eq!(
            lean_factor.state_jacobian.as_slice(),
            retained_factor.state_jacobian.as_slice()
        );
        assert_eq!(
            lean_factor.landmark_jacobian.as_slice(),
            retained_factor.landmark_jacobian.as_slice()
        );
        assert_eq!(
            lean_factor.residual.as_slice(),
            retained_factor.residual.as_slice()
        );
    }
    // The row counters emit_marg reads stay populated on the lean path
    // even though the pre-solve diagnostic linearization is skipped.
    assert_eq!(
        lean.diagnostics.visual_factor_rows,
        retained.diagnostics.visual_factor_rows
    );
    assert_eq!(
        lean.diagnostics.factor_count,
        retained.diagnostics.factor_count
    );
    assert_eq!(
        lean.diagnostics.factor_rows,
        retained.diagnostics.factor_rows
    );
}

#[test]
fn grouped_landmark_qr_backsubstitutes_a_point_update() {
    let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
    let true_point = Point3::new(0.1, -0.05, 2.0);
    let states = vec![nav(0, 0.0), nav(1, 0.2)];
    let observations = vec![
        WindowObservation {
            state_index: 0,
            camera_id: 0,
            pixel: camera.project(&true_point).unwrap(),
        },
        WindowObservation {
            state_index: 1,
            camera_id: 0,
            pixel: camera
                .project(
                    &states[1]
                        .nav
                        .imu_to_world
                        .inverse()
                        .transform_point(&true_point),
                )
                .unwrap(),
        },
    ];
    let initial_point = true_point + Vector3::new(0.002, -0.001, 0.003);
    let initial_distance = initial_point.coords.norm();
    let mut problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera,
        cameras: Vec::new(),
        t_imu_cam: Vec::new(),
        poses: Vec::new(),
        states,
        landmarks: vec![WindowLandmark {
            track_id: 7,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction: StereographicDirection::from_bearing(initial_point.coords).unwrap(),
            inverse_distance: 1.0 / initial_distance,
            observations,
        }],
        imu_links: Vec::new(),
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0e8,
        initial_accel_bias_weight: 1.0e1,
        initial_gyro_bias_weight: 1.0e2,
        prior: None,
        anchor_point: Some(DVector::zeros(15)),
        gravity_world: Vector3::zeros(),
        scalar_mode: ScalarMode::ExtendedF64,
        factor_config: FactorConfig::default(),
        visual_only: None,
    };
    let before = problem.landmarks[0].direction.bearing() / problem.landmarks[0].inverse_distance;
    let initial = problem.initial_state();
    let solved = problem.solve(initial, LmConfig::default()).unwrap();
    assert!(solved.diagnostics.visual_factor_rows > 0);
    assert_eq!(
        solved.diagnostics.prior_factor_rows
            + solved.diagnostics.visual_factor_rows
            + solved.diagnostics.imu_factor_rows
            + solved.diagnostics.bias_factor_rows,
        solved.diagnostics.factor_rows
    );
    let after = problem.landmarks[0].direction.bearing() / problem.landmarks[0].inverse_distance;
    assert!((after - before).norm() > 1e-8);
    assert!((after - true_point.coords).norm() < (before - true_point.coords).norm());
}

#[test]
fn cam1_hosted_landmark_rows_use_host_and_target_extrinsics() {
    // Divergent rig like Project Aria's SLAM pair: cam1 = cam0 rotated 75
    // degrees about x with a 0.138 m baseline.  A landmark hosted in cam1
    // is observed by cam1 (host frame and a later frame) and by cam0 at
    // the later frame.  Built from the true point, every visual row must
    // vanish: the window has to use T_imu_cam1 for the host and the
    // observing camera's extrinsic for each target.
    let camera = DoubleSphereCamera::new(241.6, 241.6, 379.0, 286.0, 0.0, 0.0, 758, 572).unwrap();
    let t_imu_cam = vec![
        SE3::identity(),
        SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(75_f64.to_radians(), 0.0, 0.0)),
            Vector3::new(0.004, -0.109, -0.085),
        ),
    ];
    let states = vec![nav(0, 0.0), nav(1, 0.3)];
    let point_host = Vector3::new(0.1, 1.5, 2.0);
    let point_world = states[0]
        .nav
        .imu_to_world
        .compose(&t_imu_cam[1])
        .transform_point(&Point3::from(point_host));
    let observe = |state_index: usize, camera_id: u16| {
        let pose = states[state_index]
            .nav
            .imu_to_world
            .compose(&t_imu_cam[camera_id as usize]);
        let pixel = camera
            .project(&pose.inverse().transform_point(&point_world))
            .unwrap();
        assert!(camera.contains_pixel(&pixel));
        WindowObservation {
            state_index,
            camera_id,
            pixel,
        }
    };
    let landmark = WindowLandmark {
        track_id: 7,
        anchor_state_index: 0,
        anchor_camera_id: 1,
        direction: StereographicDirection::from_bearing(point_host.normalize()).unwrap(),
        inverse_distance: 1.0 / point_host.norm(),
        observations: vec![observe(0, 1), observe(1, 1), observe(1, 0)],
    };
    let visual_rows = |anchor_camera_id: u16, scalar_mode: ScalarMode| {
        let mut landmark = landmark.clone();
        landmark.anchor_camera_id = anchor_camera_id;
        let mut problem = test_window(states.clone(), vec![landmark]);
        problem.camera = camera;
        problem.cameras = vec![camera, camera];
        problem.t_imu_cam = t_imu_cam.clone();
        problem.scalar_mode = scalar_mode;
        problem
            .linearize(&problem.initial_state())
            .unwrap()
            .factors
            .into_iter()
            .find(|factor| factor.kind == FactorKind::Visual)
            .expect("one grouped visual factor")
    };
    for (scalar_mode, tolerance) in [
        (ScalarMode::ExtendedF64, 1e-6),
        (ScalarMode::UpstreamF32, 2e-2),
    ] {
        let rows = visual_rows(1, scalar_mode);
        // Two rows per stored observation (the host identity included).
        assert_eq!(rows.residual.len(), 6, "{scalar_mode:?}");
        assert!(
            rows.residual.norm() < tolerance,
            "{scalar_mode:?}: residual {:?}",
            rows.residual
        );
        assert!(rows.state_jacobian.norm() > 1.0);
        assert!(rows.landmark_jacobian.norm() > 1.0);
        // Negative control: the same parameters read in cam0 are wrong.
        let wrong = visual_rows(0, scalar_mode);
        assert!(wrong.residual.norm() > 10.0, "{scalar_mode:?}");
    }
}

#[test]
fn same_timestamp_stereo_rows_accumulate_both_pose_blocks() {
    // Same-camera identity remains zero, while the stereo observation
    // contributes two relative camera terms that cancel on the shared
    // navigation block. The exact zero is a useful guard against
    // independently rounded anchor/target products leaking into AOM.
    let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
    let point = Vector3::new(0.1, -0.05, 2.0);
    let direction = StereographicDirection::from_bearing(point).unwrap();
    let observations = vec![
        WindowObservation {
            state_index: 0,
            camera_id: 0,
            pixel: Point2::new(320.0, 240.0),
        },
        WindowObservation {
            state_index: 0,
            camera_id: 1,
            pixel: Point2::new(321.0, 241.0),
        },
    ];
    let problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera,
        cameras: vec![camera, camera],
        t_imu_cam: vec![
            SE3::identity(),
            SE3::new(
                UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.02, 0.03)),
                Vector3::new(0.2, -0.1, 0.05),
            ),
        ],
        poses: Vec::new(),
        states: vec![nav(0, 0.0)],
        landmarks: vec![WindowLandmark {
            track_id: 42,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction,
            inverse_distance: 1.0 / point.norm(),
            observations,
        }],
        imu_links: Vec::new(),
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0,
        initial_accel_bias_weight: 1.0,
        initial_gyro_bias_weight: 1.0,
        prior: None,
        anchor_point: None,
        gravity_world: Vector3::zeros(),
        scalar_mode: ScalarMode::UpstreamF32,
        factor_config: FactorConfig::default(),
        visual_only: None,
    };
    let factor = problem
        .linearize(&problem.initial_state())
        .unwrap()
        .factors
        .into_iter()
        .next()
        .expect("one grouped stereo landmark factor");
    assert_eq!(factor.rows(), 4);
    assert!(factor.state_jacobian.rows(0, 2).columns(0, 6).norm() < 1e-6);
    let landmark = &problem.landmarks[0];
    let nav_state = &problem.states[0].nav;
    let stereo = anchored_visual_reprojection_factor_f32_with_time_cam(
        &problem.cameras[1],
        &nav_state.imu_to_world,
        &problem.t_imu_cam[0],
        &nav_state.imu_to_world,
        &problem.t_imu_cam[1],
        &landmark.parameter(0),
        landmark.observations[1].pixel,
        true,
        false,
        FactorConfig::default(),
    )
    .expect("same-timestamp stereo factor");
    assert!(stereo.anchor_pose_jacobian.norm() > 1e-3);
    assert!(stereo.target_pose_jacobian.norm() > 1e-3);
    assert!(factor
        .state_jacobian
        .rows(2, 2)
        .columns(0, 6)
        .iter()
        .all(|value| (*value as f32).to_bits() == 0));
    assert!(factor.landmark_jacobian.norm() > 1.0);
}

#[test]
fn pre_marginal_aom_holes_keep_native_zero_rows_in_observation_order() {
    // Native LandmarkBlockAbsDynamic reserves two rows for every stored
    // observation even when the pre-marginal AOM has truncated the target
    // state and its pose_lin entry is null.  The zero slot is part of the
    // Householder schedule, so it must remain between the two valid
    // observations rather than being filtered out.
    let camera = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
    let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.05, 2.0)).unwrap();
    let problem = WindowProblem {
        trial_host_order: Vec::new(),
        camera,
        cameras: Vec::new(),
        t_imu_cam: Vec::new(),
        poses: Vec::new(),
        states: vec![nav(0, 0.0), nav(1, 0.25)],
        landmarks: vec![WindowLandmark {
            track_id: 9001,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction,
            inverse_distance: 0.5,
            observations: vec![
                WindowObservation {
                    state_index: 0,
                    camera_id: 0,
                    pixel: Point2::new(320.0, 240.0),
                },
                WindowObservation {
                    state_index: 2,
                    camera_id: 0,
                    pixel: Point2::new(319.0, 241.0),
                },
                WindowObservation {
                    state_index: 1,
                    camera_id: 0,
                    pixel: Point2::new(321.0, 241.0),
                },
            ],
        }],
        imu_links: Vec::new(),
        imu_noise: ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        bias_walk_noise: BiasRandomWalkNoise {
            gyro_density: 1.0,
            accel_density: 1.0,
        },
        initial_pose_weight: 1.0,
        initial_accel_bias_weight: 1.0,
        initial_gyro_bias_weight: 1.0,
        prior: None,
        anchor_point: None,
        gravity_world: Vector3::zeros(),
        scalar_mode: ScalarMode::UpstreamF32,
        factor_config: FactorConfig::default(),
        visual_only: None,
    };
    let state = problem.initial_state();
    let factor = problem
        .linearize(&state)
        .unwrap()
        .factors
        .into_iter()
        .next()
        .expect("one grouped visual factor");
    assert_eq!(factor.rows(), 6);
    assert!(factor.landmark_jacobian.rows(0, 2).norm() > 0.0);
    assert!(factor.state_jacobian.rows(4, 2).norm() > 0.0);
    assert!(factor
        .state_jacobian
        .rows(2, 2)
        .iter()
        .all(|value| (*value as f32).to_bits() == 0));
    assert!(factor
        .landmark_jacobian
        .rows(2, 2)
        .iter()
        .all(|value| (*value as f32).to_bits() == 0));
    assert!(factor
        .residual
        .rows(2, 2)
        .iter()
        .all(|value| (*value as f32).to_bits() == 0));

    let mut without_hole = problem;
    without_hole.landmarks[0].observations.remove(1);
    let compact_factor = without_hole
        .linearize(&without_hole.initial_state())
        .unwrap()
        .factors
        .into_iter()
        .next()
        .expect("compact grouped visual factor");
    assert_eq!(
        factor.objective_cost.to_bits(),
        compact_factor.objective_cost.to_bits()
    );
}

#[test]
fn normal_system_stage_checkpoint_excludes_lm_solver_damping() {
    // `get_dense_H_b` returns the reduced normal system before the LM
    // driver constructs H_copy.  Keep a non-zero prior and a deliberately
    // unrelated value in the test fixture so accidentally adding the
    // solver's damping diagonal to stage 3 is observable at the bit level.
    let mut visual_h = DMatrix::<f64>::zeros(2, 2);
    visual_h[(0, 0)] = 1.0;
    let mut visual_imu_h = DMatrix::<f64>::zeros(2, 2);
    visual_imu_h[(0, 0)] = 1.5;
    visual_imu_h[(1, 1)] = -2.0;
    let mut prior_h = DMatrix::<f64>::zeros(2, 2);
    prior_h[(0, 0)] = 2.25;
    prior_h[(1, 1)] = 0.25;
    let visual_b = DVector::from_vec(vec![1.0, 2.0]);
    let visual_imu_b = DVector::from_vec(vec![3.0, 4.0]);
    let prior_b = DVector::from_vec(vec![5.0, 6.0]);
    let reduced = ReducedNormalSystem {
        h: visual_imu_h.clone(),
        b: visual_imu_b.clone(),
        back_substitution: Vec::new(),
        diagnostic_stages: Some(super::super::aom::DiagnosticNormalSystem {
            visual_h,
            visual_b,
            visual_imu_h,
            visual_imu_b,
            prior_h,
            prior_b,
        }),
    };
    let payload =
        diagnostic_normal_system_stages(&reduced, &|value: f32| format!("{:08x}", value.to_bits()));
    assert_eq!(payload["plus_pose_damping"]["h"]["bits"][0], "3fc00000");
    assert_eq!(payload["plus_pose_damping"]["h"]["bits"][3], "c0000000");
    assert_eq!(payload["plus_marginal_prior"]["h"]["bits"][0], "40700000");
    assert_eq!(payload["plus_marginal_prior"]["h"]["bits"][3], "bfe00000");
    assert_eq!(payload["plus_pose_damping"]["b"][0], "40400000");
    assert_eq!(payload["plus_marginal_prior"]["b"][0], "41000000");
}

fn read_m7im15_marg_capture_for_sqrt_trace(
    path: &std::path::Path,
) -> (DMatrix<f64>, DVector<f64>, Vec<usize>, Vec<usize>) {
    let bytes = fs::read(path).expect("M7IM15 MARG capture");
    assert!(bytes.len() >= 11);
    assert_eq!(&bytes[..11], b"M7IM15MARG1");
    let mut offset = 11usize;
    let mut read_u32 = || {
        let end = offset + 4;
        assert!(end <= bytes.len());
        let value = u32::from_le_bytes(bytes[offset..end].try_into().unwrap());
        offset = end;
        value
    };
    assert_eq!(read_u32(), 1);
    let keep_count = read_u32() as usize;
    let marg_count = read_u32() as usize;
    let mut keep = Vec::with_capacity(keep_count);
    for _ in 0..keep_count {
        keep.push(read_u32() as usize);
    }
    let mut marg = Vec::with_capacity(marg_count);
    for _ in 0..marg_count {
        marg.push(read_u32() as usize);
    }
    assert_eq!(read_u32(), 1);
    let rows = read_u32() as usize;
    let cols = read_u32() as usize;
    let mut jacobian = vec![0.0_f64; rows * cols];
    for column in 0..cols {
        for row in 0..rows {
            jacobian[column * rows + row] = f32::from_bits(read_u32()) as f64;
        }
    }
    let rhs_size = read_u32() as usize;
    assert_eq!(rhs_size, rows);
    let mut rhs = Vec::with_capacity(rhs_size);
    for _ in 0..rhs_size {
        rhs.push(f32::from_bits(read_u32()) as f64);
    }
    assert_eq!(offset, bytes.len());
    (
        DMatrix::from_column_slice(rows, cols, &jacobian),
        DVector::from_vec(rhs),
        keep,
        marg,
    )
}

fn read_m7im15_k12_focus_fixture() -> (Vec<f32>, Vec<f32>) {
    // Cargo runs package tests with the package directory as cwd when this
    // crate is selected from the workspace, while the captured diagnostic
    // lives in the workspace-level target directory.  Keep the fixture
    // lookup test-only and accept either invocation cwd.
    let relative_paths = [
        std::path::Path::new("target/m7im15_eigen_event2_focus_20260828.stdout.log"),
        std::path::Path::new("../../target/m7im15_eigen_event2_focus_20260828.stdout.log"),
    ];
    let (_path, text) = relative_paths
        .iter()
        .find_map(|path| fs::read_to_string(path).ok().map(|text| (*path, text)))
        .expect("M7IM15 k12 Eigen focus fixture");
    let mut essential = None;
    let mut bottom = None;
    for line in text.lines() {
        let words = line.split_whitespace().collect::<Vec<_>>();
        match words.first().copied() {
            Some("focus_essential") => {
                essential = Some(
                    words[1..]
                        .iter()
                        .map(|value| f32::from_bits(u32::from_str_radix(value, 16).unwrap()))
                        .collect::<Vec<_>>(),
                );
            }
            Some("focus_bottom") if words.get(1) == Some(&"0") => {
                bottom = Some(
                    words[3..]
                        .iter()
                        .map(|value| f32::from_bits(u32::from_str_radix(value, 16).unwrap()))
                        .collect::<Vec<_>>(),
                );
            }
            _ => {}
        }
    }
    (
        essential.expect("k12 essential"),
        bottom.expect("k12 column 13"),
    )
}

fn m7im15_k12_schedule_value(
    essential: &[f32],
    bottom: &[f32],
    packet_fma: bool,
    tail_fma: bool,
    reduction: u8,
    tail_first: bool,
) -> f32 {
    assert_eq!(essential.len(), 23);
    assert_eq!(bottom.len(), 23);
    let mut packets = [0.0_f32; 8];
    for offset in [0usize, 8] {
        for lane in 0..8 {
            let lhs = bottom[offset + lane];
            let rhs = essential[offset + lane];
            packets[lane] = if packet_fma {
                lhs.mul_add(rhs, packets[lane])
            } else {
                packets[lane] + upstream_f32_gemv_scalar_product(lhs, rhs)
            };
        }
    }
    let reduce = |packet: [f32; 8], mode: u8| -> f32 {
        match mode {
            // Eigen AVX `predux(Packet8f)` low/high then even/odd.
            0 => {
                let p0 = packet[0] + packet[4];
                let p1 = packet[1] + packet[5];
                let p2 = packet[2] + packet[6];
                let p3 = packet[3] + packet[7];
                (p0 + p2) + (p1 + p3)
            }
            // Scalar left fold over the packet lanes.
            1 => packet.into_iter().fold(0.0, |sum, value| sum + value),
            // Horizontal pair tree with adjacent lanes.
            2 => {
                let p0 = packet[0] + packet[1];
                let p1 = packet[2] + packet[3];
                let p2 = packet[4] + packet[5];
                let p3 = packet[6] + packet[7];
                (p0 + p1) + (p2 + p3)
            }
            // Reverse scalar fold, useful as a control for ordering.
            _ => packet.into_iter().rev().fold(0.0, |sum, value| sum + value),
        }
    };
    let mut result = 0.0_f32;
    let add_tail = |sum: &mut f32, indices: &[usize]| {
        for &offset in indices {
            let lhs = bottom[offset];
            let rhs = essential[offset];
            *sum = if tail_fma {
                lhs.mul_add(rhs, *sum)
            } else {
                *sum + upstream_f32_gemv_scalar_product(lhs, rhs)
            };
        }
    };
    if tail_first {
        add_tail(&mut result, &[16, 17, 18, 19, 20, 21, 22]);
        result += reduce(packets, reduction);
    } else {
        result = reduce(packets, reduction);
        add_tail(&mut result, &[16, 17, 18, 19, 20, 21, 22]);
    }
    result
}

#[test]
#[ignore = "requires pinned external k12 Eigen workspace capture"]
fn enumerate_m7im15_k12_gemv_workspace_schedules() {
    let (essential, bottom) = read_m7im15_k12_focus_fixture();
    let target = 0x35bfb5b6_u32;
    let candidates = [
        ("eigen_packet_fma_tail_fma", true, true, 0, false),
        ("eigen_packet_fma_tail_mul_add", true, false, 0, false),
        ("packet_mul_add_tail_fma", false, true, 0, false),
        ("packet_mul_add_tail_mul_add", false, false, 0, false),
        ("ltr_packet_fma_tail_fma", true, true, 1, false),
        ("adjacent_packet_fma_tail_mul_add", true, false, 2, false),
        ("reverse_packet_fma_tail_mul_add", true, false, 3, false),
        ("tail_first_packet_fma_tail_mul_add", true, false, 0, true),
        ("ltr_packet_mul_add_tail_mul_add", false, false, 1, false),
        (
            "adjacent_packet_mul_add_tail_mul_add",
            false,
            false,
            2,
            false,
        ),
    ];
    let mut best = None;
    for (name, packet_fma, tail_fma, reduction, tail_first) in candidates {
        let value = m7im15_k12_schedule_value(
            &essential, &bottom, packet_fma, tail_fma, reduction, tail_first,
        );
        let bits = value.to_bits();
        let distance = bits.abs_diff(target);
        println!("m7im15_k12_candidate {name} {bits:08x} distance={distance}");
        if best.map_or(true, |(_, best_distance)| distance < best_distance) {
            best = Some((name, distance));
        }
    }
    let (best_name, best_distance) = best.expect("schedule candidates");
    println!("m7im15_k12_best {best_name} distance={best_distance}");
    assert_eq!(target, 0x35bfb5b6);
}

#[test]
#[ignore = "requires external native marginalization column captures"]
fn m11_marg8_single_column_workspace_native() {
    let root = std::path::PathBuf::from(
        std::env::var_os("M11_QR_COLUMNS_ROOT").expect("native capture root"),
    );
    let read = |name: &str| -> Vec<f32> {
        let bytes = std::fs::read(root.join(name)).expect("capture");
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let before = read("event8_loop_k40_J190x42.f32");
    let after = read("event8_loop_k41_J190x42.f32");
    assert_eq!(before.len(), 190 * 42);
    assert_eq!(after.len(), before.len());
    let hh = std::fs::read_to_string(root.join("householder.tsv")).unwrap();
    let fields: Vec<_> = hh
        .lines()
        .find(|line| line.starts_with("40\t"))
        .unwrap()
        .split('\t')
        .collect();
    assert_eq!(fields[1], "40");
    let beta = f32::from_bits(u32::from_str_radix(fields[2], 16).unwrap());
    let tau = f32::from_bits(u32::from_str_radix(fields[3], 16).unwrap());
    let denominator = before[40 * 190 + 40] - beta;
    let essential: Vec<_> = before[40 * 190 + 41..41 * 190]
        .iter()
        .map(|v| *v / denominator)
        .collect();
    let scaled = upstream_f32_packet_scale(&essential, tau);
    let generic = upstream_f32_packet8_gemv_workspace(&essential, &before, 190, 41, 41, 1)[0];
    let candidate = upstream_f32_packet8_rhs_dot(&essential, &before, 41 * 190 + 41, 149);
    for (name, dot, expected_exact) in [
        ("generic", generic, 188),
        ("native_single_column", candidate, 190),
    ] {
        let mut result = before.clone();
        let workspace = dot + before[41 * 190 + 40];
        result[41 * 190 + 40] = (-tau).mul_add(workspace, before[41 * 190 + 40]);
        upstream_f32_packet8_sub_scaled(&mut result, 41 * 190 + 41, &scaled, workspace);
        let exact = result[41 * 190..42 * 190]
            .iter()
            .zip(&after[41 * 190..42 * 190])
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        println!(
            "{name} dot={:08x} workspace={:08x} exact={exact}/190",
            dot.to_bits(),
            workspace.to_bits()
        );
        assert_eq!(exact, expected_exact, "{name}");
    }
}

#[test]
fn dump_m7im15_sqrt_qtrace_from_exact_marg_capture() {
    let Some(input) = std::env::var_os("VISLOC_BASALT_SQRT_TRACE_INPUT") else {
        return;
    };
    let Some(_dump) = std::env::var_os("VISLOC_BASALT_SQRT_DUMP_QJ") else {
        panic!("exact QR replay requires qtrace output");
    };
    let (jacobian, rhs, keep, marg) =
        read_m7im15_marg_capture_for_sqrt_trace(std::path::Path::new(&input));
    let prior = sqrt_marginalize_upstream_f32_packet(
        &jacobian,
        &rhs,
        &keep,
        &marg,
        DVector::zeros(keep.len()),
    )
    .expect("exact M7IM15 Q2 capture should produce a prior");
    assert_eq!(prior.jacobian.ncols(), keep.len());
    assert!(prior.jacobian.nrows() > 0);
}

mod m11_imu_r_f64_scaled_axis_candidate_20260908 {
    use super::*;
    use nalgebra::{Quaternion, UnitQuaternion, Vector3};
    use serde_json::Value;
    use std::collections::HashSet;

    const FIXTURE: &str =
        include_str!("../../../tests/fixtures/m11_imu_r_f64_scaled_axis_candidate_20260908.json");

    fn string_field<'a>(value: &'a Value, key: &str) -> &'a str {
        value
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("missing string field {key}"))
    }

    fn integer_field(value: &Value, key: &str) -> u64 {
        value
            .get(key)
            .and_then(Value::as_u64)
            .unwrap_or_else(|| panic!("missing integer field {key}"))
    }

    fn vector_bits(value: &Value, key: &str, expected_len: usize) -> Vec<u64> {
        let values = value
            .get(key)
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("missing array field {key}"));
        assert_eq!(values.len(), expected_len);
        values
            .iter()
            .map(|entry| {
                let bits = entry.as_str().expect("f64 bit string");
                assert_eq!(bits.len(), 16);
                u64::from_str_radix(bits, 16).expect("f64 bits")
            })
            .collect()
    }

    fn raw_rotation(product_xyzw: [f64; 4]) -> UnitQuaternion<f64> {
        UnitQuaternion::new_unchecked(Quaternion::new(
            product_xyzw[3],
            product_xyzw[0],
            product_xyzw[1],
            product_xyzw[2],
        ))
    }

    fn scaled_axis_bits(value: Vector3<f64>) -> [u64; 3] {
        [value.x.to_bits(), value.y.to_bits(), value.z.to_bits()]
    }

    #[test]
    fn actual_154_quaternions_match_shared_grouped_candidate_and_native_fallback() {
        let fixture: Value = serde_json::from_str(FIXTURE).expect("valid candidate fixture");
        assert_eq!(
            string_field(&fixture, "schema_id"),
            "visloc.basalt.m11.imu_r_f64.scaled_axis_candidate_regression.v1"
        );
        assert_eq!(integer_field(&fixture, "schema_version"), 1);
        assert_eq!(
            string_field(&fixture, "expected_source"),
            "RustLinuxDiagnostic"
        );
        assert!(!fixture["expected_source_is_native_oracle"]
            .as_bool()
            .unwrap());
        assert!(!fixture["native_basalt_oracle_used"].as_bool().unwrap());
        assert_eq!(
            string_field(&fixture, "candidate_scope"),
            "diagnostic_f64_raw_rotation_only"
        );
        assert!(!fixture["production_path_unchanged"].as_bool().unwrap());

        let helper = &fixture["helper_source"];
        assert_eq!(
            string_field(helper, "path"),
            "work/m11_compensated_f64_atan_small_candidate_scale_only_grouped_20260907.rs"
        );
        assert_eq!(
            string_field(helper, "sha256"),
            "5B3DF98640950912EDA558743BD58256862FC29575BDD4F8C9EF4BF2D8E12246"
        );
        let source_hashes = &fixture["source_hashes"];
        for key in [
            "window_rs",
            "linux_actual_capture",
            "msvc_actual_capture",
            "test_module_rs",
        ] {
            let hash = string_field(source_hashes, key);
            assert_eq!(hash.len(), 64, "source hash {key}");
            assert!(hash.chars().all(|character| character.is_ascii_hexdigit()));
        }
        assert_eq!(
            string_field(source_hashes, "linux_actual_capture"),
            "51DFB68440D9393842F3FF98E1CE7F391F15C88CA069F36D8365B85C0B997E83"
        );
        assert_eq!(
            string_field(source_hashes, "msvc_actual_capture"),
            "34D32C64177DCD26200BF165C33EDB3A2C1ECA9E4C68D7F8121F04C7095C3910"
        );

        let capture = &fixture["capture_contract"];
        assert_eq!(string_field(capture, "q_component_order"), "xyzw");
        assert_eq!(string_field(capture, "expected_r_R_component_order"), "xyz");
        assert_eq!(
            string_field(capture, "expected_r_R_source"),
            "RustLinuxDiagnostic"
        );
        assert_eq!(integer_field(capture, "record_count"), 154);
        assert_eq!(integer_field(capture, "composite_link_count"), 154);
        assert!(capture["cross_os_q_f64_bits_equal"].as_bool().unwrap());
        assert!(capture["input_scalars_finite"].as_bool().unwrap());
        assert!(capture["input_w_positive"].as_bool().unwrap());

        let contract = &fixture["candidate_contract"];
        assert_eq!(string_field(contract, "ratio_threshold"), "2^-20");
        assert_eq!(
            string_field(contract, "ratio_threshold_f64_bits"),
            "3eb0000000000000"
        );
        assert_eq!(integer_field(contract, "inside_count"), 79);
        assert_eq!(integer_field(contract, "outside_count"), 75);
        assert_eq!(string_field(contract, "inside_route"), "grouped_scale_only");
        assert_eq!(
            string_field(contract, "outside_route"),
            "platform_std_fallback"
        );

        let records = fixture["records"].as_array().expect("fixture records");
        assert_eq!(records.len(), 154);
        let mut grouped_count = 0;
        let mut fallback_count = 0;
        let mut seen_links = HashSet::new();
        for (ordinal, record) in records.iter().enumerate() {
            assert_eq!(integer_field(record, "ordinal"), ordinal as u64);
            let key = (
                integer_field(record, "active_frame_id"),
                integer_field(record, "from_frame_id"),
                integer_field(record, "to_frame_id"),
                integer_field(record, "link_index"),
            );
            assert!(seen_links.insert(key));
            let q_bits = vector_bits(record, "product_xyzw_f64_bits", 4);
            let product = [
                f64::from_bits(q_bits[0]),
                f64::from_bits(q_bits[1]),
                f64::from_bits(q_bits[2]),
                f64::from_bits(q_bits[3]),
            ];
            assert!(product.iter().all(|value| value.is_finite()));
            assert!(product[3] > 0.0);
            let norm_squared =
                product[0] * product[0] + product[1] * product[1] + product[2] * product[2];
            let route = if norm_squared.sqrt() / product[3] <= IMU_F64_ATAN_SMALL_MAX_RATIO {
                grouped_count += 1;
                "grouped_scale_only"
            } else {
                fallback_count += 1;
                "platform_std_fallback"
            };
            assert_eq!(string_field(record, "candidate_route"), route);
            let expected_bits = vector_bits(record, "expected_r_R_xyz_f64_bits", 3);
            assert_eq!(
                scaled_axis_bits(imu_f64_scaled_axis_candidate(&raw_rotation(product))).as_slice(),
                expected_bits.as_slice(),
                "candidate mismatch at ordinal {ordinal}"
            );
        }
        assert_eq!(seen_links.len(), 154);
        assert_eq!(grouped_count, 79);
        assert_eq!(fallback_count, 75);
    }

    #[test]
    fn shared_candidate_preserves_axis_sign_zero_axis_and_helper_domain_fallback() {
        assert_eq!(
            scaled_axis_bits(imu_f64_scaled_axis_candidate(&raw_rotation([
                0.0, 0.0, 0.0, 1.0,
            ]))),
            [0.0_f64.to_bits(); 3]
        );
        assert_eq!(
            scaled_axis_bits(imu_f64_scaled_axis_candidate(&raw_rotation([
                0.0, 0.0, 0.0, -1.0,
            ]))),
            [0.0_f64.to_bits(); 3]
        );

        let positive_w = [1.0e-8, -2.0e-8, 3.0e-8, 1.0];
        let negative_w = positive_w.map(|value| -value);
        assert_eq!(
            scaled_axis_bits(imu_f64_scaled_axis_candidate(&raw_rotation(positive_w))),
            scaled_axis_bits(imu_f64_scaled_axis_candidate(&raw_rotation(negative_w)))
        );

        assert_eq!(
            imu_f64_atan_small_grouped_positive(0.0, 1.0)
                .expect("zero angle")
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            imu_f64_atan_small_grouped_positive(-0.0, 1.0)
                .expect("negative zero angle")
                .to_bits(),
            (-0.0_f64).to_bits()
        );
        for (y, x) in [
            (-1.0, 1.0),
            (1.0, 0.0),
            (1.0, -0.0),
            (1.0, -1.0),
            (f64::INFINITY, 1.0),
            (1.0, f64::INFINITY),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
        ] {
            assert!(
                imu_f64_atan_small_grouped_positive(y, x).is_none(),
                "accepted invalid helper domain y={y:?} x={x:?}"
            );
        }
        let threshold_bits = IMU_F64_ATAN_SMALL_MAX_RATIO.to_bits();
        assert!(
            imu_f64_atan_small_grouped_positive(f64::from_bits(threshold_bits - 1), 1.0).is_some()
        );
        assert!(imu_f64_atan_small_grouped_positive(IMU_F64_ATAN_SMALL_MAX_RATIO, 1.0).is_some());
        assert!(
            imu_f64_atan_small_grouped_positive(f64::from_bits(threshold_bits + 1), 1.0).is_none()
        );

        let invalid = raw_rotation([f64::NAN, 0.0, 0.0, 1.0]);
        assert_eq!(
            scaled_axis_bits(imu_f64_scaled_axis_candidate(&invalid)),
            scaled_axis_bits(invalid.scaled_axis())
        );
        for scalar in [0.0, -0.0] {
            let product = raw_rotation([0.25, -0.125, 0.0, scalar]);
            assert_eq!(
                scaled_axis_bits(imu_f64_scaled_axis_candidate(&product)),
                scaled_axis_bits(product.scaled_axis())
            );
        }
    }
}

// Test-only reanchor schedule candidate.  The production path remains
// untouched: this isolates the captured 21-row stored-RHS update while
// its native pre-b/delta operand binding is still incomplete.
mod m11_prior_reanchor_rhs_schedule_20260908 {
    use super::{DMatrix, DVector, WindowProblem};
    use serde_json::Value;

    const FIXTURE: &str =
        include_str!("../../../tests/fixtures/m11_prior_reanchor_rhs_schedule_20260908.json");

    fn bits<const N: usize>(value: &Value, key: &str) -> [u32; N] {
        let values = value
            .get(key)
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("missing bit array {key}"));
        assert_eq!(values.len(), N, "bit array length for {key}");
        values
            .iter()
            .map(|entry| {
                u32::from_str_radix(
                    entry
                        .as_str()
                        .unwrap_or_else(|| panic!("non-string bit entry in {key}")),
                    16,
                )
                .unwrap_or_else(|error| panic!("invalid bit entry in {key}: {error}"))
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("wrong bit array length for {key}"))
    }

    // Decode the captured packet and exercise the same private helper as
    // the UpstreamF32 production re-anchor.  The independent witnesses
    // below remain test-local so association changes are visible.
    fn prior_reanchor_rhs_col_major_21_f32_candidate(
        jacobian_bits_col_major: &[u32; 441],
        stored_rhs_bits: &[u32; 21],
        keep_delta_bits: &[u32; 21],
    ) -> [u32; 21] {
        let jacobian = DMatrix::from_column_slice(
            21,
            21,
            &jacobian_bits_col_major
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect::<Vec<_>>(),
        );
        let stored_rhs =
            DVector::from_iterator(21, stored_rhs_bits.iter().copied().map(f32::from_bits));
        let keep_delta =
            DVector::from_iterator(21, keep_delta_bits.iter().copied().map(f32::from_bits));
        WindowProblem::prior_reanchor_rhs_col_major_21_f32(&jacobian, &stored_rhs, &keep_delta)
            .iter()
            .copied()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
            .try_into()
            .expect("21-row candidate output")
    }

    // Algebraically equivalent, but not the native association: round
    // every depth product before the ordinary subtraction from b.
    #[inline(never)]
    fn separate_depth_reanchor_rhs(
        jacobian_bits_col_major: &[u32; 441],
        stored_rhs_bits: &[u32; 21],
        keep_delta_bits: &[u32; 21],
    ) -> [u32; 21] {
        let mut result = [0_u32; 21];
        for row in 0..21 {
            let mut value = 0.0_f32;
            for column in 0..21 {
                let product = f32::from_bits(jacobian_bits_col_major[column * 21 + row])
                    * f32::from_bits(keep_delta_bits[column]);
                value += product;
            }
            result[row] = (f32::from_bits(stored_rhs_bits[row]) - value).to_bits();
        }
        result
    }

    // Another plausible expression starts the accumulator at b and fuses
    // each negative product into it.  Keep it as a negative witness so a
    // future change cannot collapse the tested association to this form.
    #[inline(never)]
    fn negative_fma_into_stored_rhs(
        jacobian_bits_col_major: &[u32; 441],
        stored_rhs_bits: &[u32; 21],
        keep_delta_bits: &[u32; 21],
    ) -> [u32; 21] {
        let mut result = [0_u32; 21];
        for row in 0..20 {
            let mut value = f32::from_bits(stored_rhs_bits[row]);
            for column in 0..21 {
                value = f32::from_bits(jacobian_bits_col_major[column * 21 + row])
                    .mul_add(-f32::from_bits(keep_delta_bits[column]), value);
            }
            result[row] = value.to_bits();
        }

        let mut value = 0.0_f32;
        for column in 0..21 {
            let product = f32::from_bits(jacobian_bits_col_major[column * 21 + 20])
                * f32::from_bits(keep_delta_bits[column]);
            value += product;
        }
        result[20] = (f32::from_bits(stored_rhs_bits[20]) - value).to_bits();
        result
    }

    fn mismatch_indices(actual: &[u32; 21], expected: &[u32; 21]) -> Vec<usize> {
        actual
            .iter()
            .zip(expected)
            .enumerate()
            .filter_map(|(index, (actual, expected))| (actual != expected).then_some(index))
            .collect()
    }

    #[test]
    fn captured_prior_reanchor_schedule_has_native_and_negative_witnesses() {
        let fixture: Value = serde_json::from_str(FIXTURE).expect("valid reanchor fixture");
        assert_eq!(fixture["shape"]["rows"].as_u64(), Some(21));
        assert_eq!(fixture["shape"]["cols"].as_u64(), Some(21));
        assert_eq!(
            fixture["source_basis"]["matrix_layout"].as_str(),
            Some("column_major_f32")
        );

        let jacobian = bits::<441>(&fixture["inputs"], "jacobian_f32_bits_col_major");
        let stored_rhs = bits::<21>(&fixture["inputs"], "stored_rhs_f32_bits");
        let keep_delta = bits::<21>(&fixture["inputs"], "keep_delta_f32_bits");
        let expected = bits::<21>(&fixture["expected"], "native_mld_b_after_f32_bits");

        let actual =
            prior_reanchor_rhs_col_major_21_f32_candidate(&jacobian, &stored_rhs, &keep_delta);
        assert_eq!(actual, expected, "candidate schedule must match captured b");
        assert_eq!(actual[0], 0xbf4b05b0);
        assert_eq!(actual[2], 0x3ee2e4f5);

        let separated = separate_depth_reanchor_rhs(&jacobian, &stored_rhs, &keep_delta);
        assert_eq!(
            mismatch_indices(&separated, &expected),
            vec![0, 2],
            "separate product/add witness changed unexpectedly"
        );
        assert_eq!(separated[0], 0xbf4b05af);
        assert_eq!(separated[2], 0x3ee2e4f7);

        let negative_fma = negative_fma_into_stored_rhs(&jacobian, &stored_rhs, &keep_delta);
        assert_eq!(
            mismatch_indices(&negative_fma, &expected),
            vec![3],
            "negative-FMA-into-b witness changed unexpectedly"
        );
        assert_eq!(negative_fma[3], 0x41841148);
    }

    #[test]
    fn prior_reanchor_candidate_keeps_scalar_row20_tail_contract() {
        let mut jacobian = [0_u32; 441];
        let mut keep_delta = [0_u32; 21];
        let stored_rhs = [0_u32; 21];
        // A cancellation witness: separate f32 products round the
        // second product to one and produce zero, while an all-FMA tail
        // retains -2^-46.  The scalar row must take the former path.
        jacobian[20] = 1.0_f32.to_bits();
        jacobian[21 + 20] = 0x3f800001;
        keep_delta[0] = (-1.0_f32).to_bits();
        keep_delta[1] = 0x3f7ffffe;

        let actual =
            prior_reanchor_rhs_col_major_21_f32_candidate(&jacobian, &stored_rhs, &keep_delta);
        assert_eq!(actual[20], 0.0_f32.to_bits());

        let mut all_fma_sum = 0.0_f32;
        for column in 0..21 {
            all_fma_sum = f32::from_bits(jacobian[column * 21 + 20])
                .mul_add(f32::from_bits(keep_delta[column]), all_fma_sum);
        }
        assert_eq!(all_fma_sum.to_bits(), 0xa8800000);
        assert_eq!(
            all_fma_sum
                .mul_add(-1.0_f32, f32::from_bits(stored_rhs[20]))
                .to_bits(),
            0x28800000
        );
    }
}
