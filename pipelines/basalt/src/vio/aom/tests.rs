#[test]
#[ignore = "requires M11_VISUAL_CAPTURE_ROOT pinned frame17 capture"]
fn frame17_visual_packet_tail_native_full_matrix() {
    let root = std::path::PathBuf::from(std::env::var("M11_VISUAL_CAPTURE_ROOT").unwrap());
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("factor_storage.json")).unwrap()).unwrap();
    let native = std::fs::read(root.join("frame17_visual_total_iter0.bin"))
        .or_else(|_| {
            // Historical call26 capture is labelled iter0; dense_call above
            // binds its actual iteration rather than trusting the filename.
            std::fs::read(root.join("frame7_visual_total_iter0.bin"))
        })
        .unwrap();
    let width = u64::from_le_bytes(native[0..8].try_into().unwrap()) as usize;
    assert!(matches!(
        (meta["dense_call"].as_u64().unwrap(), width),
        (100, 63) | (25, 51) | (26, 51)
    ));
    let factors = meta["factors"].as_array().unwrap();
    assert_eq!(factors.len(), meta["count"].as_u64().unwrap() as usize);
    let mut old = DMatrix::<f32>::zeros(width, width);
    let mut candidate = old.clone();
    for factor in factors {
        let rows = factor["rows"].as_u64().unwrap() as usize;
        let cols = factor["cols"].as_u64().unwrap() as usize;
        let used = factor["num_rows"].as_u64().unwrap() as usize;
        assert_eq!(factor["padding_idx"].as_u64(), Some(width as u64));
        let bytes = std::fs::read(root.join(factor["storage"].as_str().unwrap())).unwrap();
        assert_eq!(bytes.len(), rows * cols * 4);
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let j = DMatrix::from_fn(used - 3, width, |r, c| values[(r + 3) * cols + c]);
        old += j.transpose() * &j;
        candidate += eigen_visual_gram_packet_tail_f32(&j);
    }
    assert_eq!(native.len(), 24 + (width * width + width) * 4);
    for offset in [0, 8, 16] {
        assert_eq!(
            u64::from_le_bytes(native[offset..offset + 8].try_into().unwrap()),
            width as u64
        );
    }
    let expected: Vec<u32> = native[24..24 + width * width * 4]
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    if width == 63 {
        assert_eq!(
            old.iter()
                .zip(&expected)
                .filter(|(x, y)| x.to_bits() == **y)
                .count(),
            3864
        );
    }
    for (i, (actual, expected)) in candidate.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected, "lane {i}");
    }
}

#[test]
#[ignore = "requires M11_PRIOR_CAPTURE_ROOT pinned external capture"]
fn frame10_prior27_native_gram_and_normal_rhs() {
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
    let parse = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
            .collect()
    };
    for (iteration, input) in inputs.iter().enumerate() {
        assert_eq!(input["iteration"].as_u64().unwrap(), iteration as u64);
        assert_eq!(input["rows"].as_u64().unwrap(), 27);
        assert_eq!(input["cols"].as_u64().unwrap(), 27);
        let j = DMatrix::from_column_slice(27, 27, &parse(&input["jacobian_bits"]));
        let gram = stages
            .iter()
            .find(|x| {
                x["iteration"].as_u64() == Some(iteration as u64)
                    && x["stage"].as_str() == Some("gram")
            })
            .unwrap();
        let expected = parse(&gram["bits"]);
        assert_eq!(expected.len(), 729);
        let actual = eigen_prior_compact_gram_f32(&j);
        for i in 0..729 {
            assert_eq!(
                actual.as_slice()[i].to_bits(),
                expected[i].to_bits(),
                "iteration {iteration}, lane {i}"
            );
        }
        let old = j.transpose() * &j;
        assert_eq!(
            old.iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count(),
            41
        );
        let stage = |name: &str| -> Vec<f32> {
            parse(
                &stages
                    .iter()
                    .find(|x| {
                        x["iteration"].as_u64() == Some(iteration as u64)
                            && x["stage"].as_str() == Some(name)
                    })
                    .unwrap()["bits"],
            )
        };
        let residual = DVector::from_vec(stage("adjusted_rhs"));
        let normal_expected = stage("normal_rhs");
        let jt = j.transpose().into_owned();
        let normal = eigen_prior_row_major_gemv_f32(&jt, &residual);
        for lane in 0..27 {
            assert_eq!(
                normal[lane].to_bits(),
                normal_expected[lane].to_bits(),
                "normal RHS iteration {iteration} lane {lane}"
            );
        }
        if iteration == 0 {
            let old_normal = eigen_row_major_gemv_f32(&jt, &residual);
            assert_eq!(
                old_normal
                    .iter()
                    .zip(&normal_expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count(),
                3
            );
        }
    }
}
#[test]
#[ignore = "requires validated external native prior capture"]
fn prior_packet_tail_candidate_native_gram() {
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
    assert!(!inputs.is_empty());
    let parse = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
            .collect()
    };
    for (i, input) in inputs.iter().enumerate() {
        let size = input["rows"].as_u64().unwrap() as usize;
        assert_eq!(input["cols"].as_u64(), Some(size as u64));
        assert!([21, 27, 33, 39].contains(&size));
        assert_eq!(input["iteration"].as_u64(), Some(i as u64));
        let j = DMatrix::from_column_slice(size, size, &parse(&input["jacobian_bits"]));
        let gram = stages
            .iter()
            .find(|s| {
                s["iteration"].as_u64() == Some(i as u64) && s["stage"].as_str() == Some("gram")
            })
            .unwrap();
        let expected = parse(&gram["bits"]);
        assert_eq!(expected.len(), size * size);
        let actual = eigen_prior_packet_tail_gram_candidate_f32(&j);
        for (lane, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "size {size} iteration {i} lane {lane}"
            );
        }
        if size == 33 {
            let old = j.transpose() * &j;
            assert_eq!(
                old.iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count(),
                191
            );
        }
        if size == 39 {
            let production = eigen_prior_compact_gram_f32(&j);
            assert!(production
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            let old = j.transpose() * &j;
            assert_eq!(
                old.iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count(),
                339
            );
        }
    }
}

#[test]
#[ignore = "requires external frame31 prior input and native stage capture"]
fn m11_frame31_prior45_candidate_probe() {
    let root = std::path::PathBuf::from(
        std::env::var("M11_FRAME31_PRIOR_ROOT").expect("frame31 capture root"),
    );
    let input_path =
        root.join("m11_trial_wired_frame10_20260908/r34_frame31_prior/prior_inputs.jsonl");
    let native = root.join("m11_native_frame31_strict_capture_20260912/r1");
    let records: Vec<serde_json::Value> = std::fs::read_to_string(input_path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .filter(|record: &serde_json::Value| {
            record["rows"].as_u64() == Some(45) && record["global_cols"].as_u64() == Some(75)
        })
        .collect();
    assert_eq!(records.len(), 8);
    let native_prior_stages: Vec<serde_json::Value> = std::fs::read_to_string(
        root.join("m11_native_frame31_prior_stages_20260912/r1/events.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .collect();
    let native_model_parts: Vec<serde_json::Value> = std::fs::read_to_string(
        root.join("m11_native_frame31_model_parts_20260912/r1/events.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .collect();
    let bits = |value: &serde_json::Value| -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| f32::from_bits(u32::from_str_radix(word.as_str().unwrap(), 16).unwrap()))
            .collect()
    };
    let dense = |path: &std::path::Path| -> (DMatrix<f32>, DVector<f32>) {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len(), 22_824);
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 75);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 75);
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 75);
        let values: Vec<f32> = bytes[24..]
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        (
            DMatrix::from_column_slice(75, 75, &values[..75 * 75]),
            DVector::from_column_slice(&values[75 * 75..]),
        )
    };
    for (iteration, record) in records.iter().enumerate() {
        assert_eq!(record["call"].as_u64(), Some((209 + iteration) as u64));
        let columns: Vec<usize> = record["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(columns.len(), 45);
        let j = DMatrix::from_column_slice(45, 45, &bits(&record["jacobian_compact_bits"]));
        let residual = DVector::from_vec(bits(&record["residual_bits"]));
        let (before_h, before_b) =
            dense(&native.join(format!("frame31_prior_before_iter{iteration}.bin")));
        let (expected_h, expected_b) =
            dense(&native.join(format!("frame31_prior_after_iter{iteration}.bin")));
        let mut candidate_h = before_h.clone();
        let gram = eigen_prior_packet_tail_gram_candidate_f32(&j);
        for local_row in 0..45 {
            for local_column in 0..45 {
                candidate_h[(columns[local_row], columns[local_column])] +=
                    gram[(local_row, local_column)];
            }
        }
        let mut candidate_b = before_b.clone();
        let transpose = j.transpose().into_owned();
        let normal = eigen_prior_row_major_gemv_45_f32(&transpose, &residual);
        for (local, &global) in columns.iter().enumerate() {
            candidate_b[global] += normal[local];
        }
        let h_exact = candidate_h
            .iter()
            .zip(expected_h.iter())
            .filter(|(actual, expected)| actual.to_bits() == expected.to_bits())
            .count();
        let b_exact = candidate_b
            .iter()
            .zip(expected_b.iter())
            .filter(|(actual, expected)| actual.to_bits() == expected.to_bits())
            .count();
        let native_stage = |name: &str| {
            native_prior_stages
                .iter()
                .find(|entry| {
                    entry["kind"].as_str() == Some("prior_stage")
                        && entry["stage"].as_str() == Some(name)
                        && entry["iteration"].as_u64() == Some(iteration as u64)
                })
                .unwrap()
        };
        let native_adjusted = DVector::from_vec(bits(&native_stage("adjusted_rhs")["bits"]));
        let expected_normal = bits(&native_stage("normal_rhs")["bits"]);
        let native_normal = eigen_prior_row_major_gemv_45_f32(&transpose, &native_adjusted);
        let normal_exact = native_normal
            .iter()
            .zip(&expected_normal)
            .filter(|(actual, expected)| actual.to_bits() == expected.to_bits())
            .count();
        let normal_mismatch = native_normal
            .iter()
            .zip(&expected_normal)
            .enumerate()
            .filter(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
            .map(|(lane, (actual, expected))| {
                format!("{lane}:{:08x}/{:08x}", actual.to_bits(), expected.to_bits())
            })
            .collect::<Vec<_>>();
        let step_bytes =
            std::fs::read(native.join(format!("frame31_inc_entry_iter{iteration}.f32"))).unwrap();
        assert_eq!(step_bytes.len(), 75 * 4);
        let global_step: Vec<f32> = step_bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        let compact_step =
            DVector::from_iterator(45, columns.iter().map(|&column| global_step[column]));
        let model = prior45_model_f32(&j, &native_adjusted, &compact_step);
        let expected_model = native_model_parts
            .iter()
            .find(|entry| {
                entry["kind"].as_str() == Some("model_part")
                    && entry["stage"].as_str() == Some("prior")
                    && entry["iteration"].as_u64() == Some(iteration as u64)
            })
            .unwrap()["f32_bits"]
            .as_str()
            .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
            .unwrap();
        println!(
            "M11_FRAME31_PRIOR45 iter={iteration} H={h_exact}/5625 b={b_exact}/75 normal={normal_exact}/45 model={:08x}/{:08x} mismatch={normal_mismatch:?}",
            model.to_bits(),
            expected_model.to_bits(),
        );
        assert_eq!(
            model.to_bits(),
            expected_model.to_bits(),
            "model iteration {iteration}"
        );
        let mut global_j = DMatrix::<f64>::zeros(45, 75);
        for (local_column, &global_column) in columns.iter().enumerate() {
            for row in 0..45 {
                global_j[(row, global_column)] = j[(row, local_column)] as f64;
            }
        }
        let factor = WhitenedFactorRowStack::new(
            global_j,
            DMatrix::zeros(45, 0),
            native_adjusted.map(|value| value as f64),
        )
        .unwrap()
        .with_kind(FactorKind::Prior)
        .with_prior_state_columns(columns.clone());
        let dispatched = model_cost_decrease_f32(
            &[factor],
            &DVector::from_iterator(75, global_step.iter().map(|&value| value as f64)),
            1e-10,
        )
        .unwrap() as f32;
        assert_eq!(
            dispatched.to_bits(),
            expected_model.to_bits(),
            "production dispatch iteration {iteration}"
        );
    }
}

#[test]
#[ignore = "requires validated native frame38 prior capture"]
fn m11_frame38_prior51_gram_and_normal_candidate() {
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
    let bits = |value: &serde_json::Value| -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| f32::from_bits(u32::from_str_radix(word.as_str().unwrap(), 16).unwrap()))
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
        let j = DMatrix::from_column_slice(51, 51, &bits(&input["jacobian_bits"]));
        let gram_expected = bits(&stage("gram")["bits"]);
        let gram = eigen_prior_packet_tail_gram_candidate_f32(&j);
        for lane in 0..51 * 51 {
            assert_eq!(
                gram.as_slice()[lane].to_bits(),
                gram_expected[lane].to_bits(),
                "Gram iteration {iteration} lane {lane}"
            );
        }

        let adjusted = DVector::from_vec(bits(&stage("adjusted_rhs")["bits"]));
        let normal_expected = bits(&stage("normal_rhs")["bits"]);
        let normal = eigen_prior_row_major_gemv_51_f32(&j.transpose(), &adjusted);
        for lane in 0..51 {
            assert_eq!(
                normal[lane].to_bits(),
                normal_expected[lane].to_bits(),
                "normal RHS iteration {iteration} lane {lane}"
            );
        }
    }
}

#[test]
#[ignore = "requires validated native frame45 prior capture"]
fn m11_frame45_prior57_gram_and_normal_candidate() {
    let root = std::path::PathBuf::from(
        std::env::var("M11_FRAME45_PRIOR_ROOT").expect("frame45 capture root"),
    );
    let records: Vec<serde_json::Value> = std::fs::read_to_string(
        root.join("m11_native_frame45_strict_capture_20260913/r1/events.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .collect();
    let bits = |value: &serde_json::Value| -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| f32::from_bits(u32::from_str_radix(word.as_str().unwrap(), 16).unwrap()))
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
        let j = DMatrix::from_column_slice(57, 57, &bits(&input["jacobian_bits"]));
        let gram_expected = bits(&stage("gram")["bits"]);
        let gram = eigen_prior_packet_tail_gram_candidate_f32(&j);
        for lane in 0..57 * 57 {
            assert_eq!(
                gram.as_slice()[lane].to_bits(),
                gram_expected[lane].to_bits(),
                "Gram iteration {iteration} lane {lane}"
            );
        }

        let adjusted = DVector::from_vec(bits(&stage("adjusted_rhs")["bits"]));
        let normal_expected = bits(&stage("normal_rhs")["bits"]);
        let normal = eigen_prior_row_major_gemv_57_f32(&j.transpose(), &adjusted);
        for lane in 0..57 {
            assert_eq!(
                normal[lane].to_bits(),
                normal_expected[lane].to_bits(),
                "normal RHS iteration {iteration} lane {lane}"
            );
        }
    }
}
#[test]
#[ignore = "requires validated external native frame17 prior capture"]
fn frame17_prior33_normal_rhs_candidate() {
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
    let parse = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
            .collect()
    };
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(input["iteration"].as_u64(), Some(i as u64));
        assert_eq!(
            (input["rows"].as_u64(), input["cols"].as_u64()),
            (Some(33), Some(33))
        );
        let j = DMatrix::from_column_slice(33, 33, &parse(&input["jacobian_bits"]));
        let stage = |name: &str| -> Vec<f32> {
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
        let r = DVector::from_vec(stage("adjusted_rhs"));
        let expected = stage("normal_rhs");
        let jt = j.transpose().into_owned();
        let actual = eigen_prior_row_major_gemv_33(&jt, &r);
        for lane in 0..33 {
            assert_eq!(
                actual[lane].to_bits(),
                expected[lane].to_bits(),
                "iteration {i} lane {lane}"
            );
        }
        let old = eigen_row_major_gemv_f32(&jt, &r);
        assert_eq!(
            old.iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count(),
            usize::from([0, 5, 6, 7].contains(&i))
        );
    }
}
#[test]
#[ignore = "requires validated native trial computeRelPose capture on E"]
fn frame24_prior39_model_same_inputs_candidate() {
    let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
    let input_root = root.join("m11_native_frame24_prior_stages_20260909/r1");
    let model_root = root.join("m11_native_frame24_model_parts_20260909/r1");
    let records = |name: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(input_root.join(name))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let inputs = records("prior_inputs.jsonl");
    let stages = records("prior_stages.jsonl");
    let oracle: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model_root.join("model_parts_validated.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(oracle["status"], "PASS");
    assert_eq!(inputs.len(), 8);
    let parse = |x: &serde_json::Value| {
        f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap())
    };
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(input["iteration"].as_u64(), Some(i as u64));
        assert_eq!(input["rows"].as_u64(), Some(39));
        assert_eq!(input["cols"].as_u64(), Some(39));
        let j = DMatrix::from_iterator(
            39,
            39,
            input["jacobian_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(&parse),
        );
        let matching: Vec<_> = stages
            .iter()
            .filter(|x| x["iteration"].as_u64() == Some(i as u64) && x["stage"] == "adjusted_rhs")
            .collect();
        assert_eq!(matching.len(), 1);
        let rhs = DVector::from_iterator(
            39,
            matching[0]["bits"].as_array().unwrap().iter().map(&parse),
        );
        let name = format!("frame24_inc_entry_iter{i}.f32");
        let bytes = std::fs::read(input_root.join(&name)).unwrap();
        assert_eq!(bytes, std::fs::read(model_root.join(&name)).unwrap());
        assert_eq!(bytes.len(), 69 * 4);
        let step = DVector::from_iterator(
            39,
            bytes
                .chunks_exact(4)
                .take(39)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap())),
        );
        let actual = prior39_model_f32(&j, &rhs, &step);
        let expected = parse(&oracle["iterations"][i]["prior"]);
        println!(
            "M11_PRIOR39_MODEL iter={i} native={:08x} candidate={:08x}",
            expected.to_bits(),
            actual.to_bits()
        );
        assert_eq!(actual.to_bits(), expected.to_bits(), "iteration {i}");
        let mut global_j = DMatrix::<f64>::zeros(39, 69);
        global_j.columns_mut(0, 39).copy_from(&j.map(|x| x as f64));
        let factor =
            WhitenedFactorRowStack::new(global_j, DMatrix::zeros(39, 0), rhs.map(|x| x as f64))
                .unwrap()
                .with_kind(FactorKind::Prior)
                .with_prior_state_columns((0..39).collect());
        let global_step = DVector::from_iterator(
            69,
            bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()) as f64),
        );
        let dispatched = model_cost_decrease_f32(&[factor], &global_step, 1e-10).unwrap() as f32;
        assert_eq!(
            dispatched.to_bits(),
            expected.to_bits(),
            "production dispatch iteration {i}"
        );
    }
}

#[test]
#[ignore = "requires validated native frame24 visual capture and r28 detail on E"]
fn frame24_track1259_model_qr_boundary_probe() {
    use std::io::BufRead;
    let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
    let native = root.join("m11_native_frame24_visual_model_callsite_20260909/r2");
    for gate in [
        "engine.rc",
        "gdb.wrapper.rc",
        "probe.validation.rc",
        "binding.pre_post.cmp.rc",
    ] {
        assert_eq!(
            std::fs::read_to_string(native.join(gate)).unwrap().trim(),
            "0"
        );
    }
    let input = std::fs::File::open(
        root.join("m11_trial_wired_frame10_20260908/r28_frame24/detail_iterations.jsonl"),
    )
    .unwrap();
    let mut seen = [false; 8];
    for line in std::io::BufReader::new(input).lines() {
        let record: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        if record["frame_id"] != 24 || record["phase"] != "iteration_start" {
            continue;
        }
        let i = record["iteration"].as_u64().unwrap() as usize;
        assert!(i < 8 && !seen[i]);
        seen[i] = true;
        let factors: Vec<_> = record["landmark_factors"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["track_id"] == 1259)
            .collect();
        assert_eq!(factors.len(), 1);
        let f = factors[0];
        let matrix = |key: &str, cols: usize| {
            DMatrix::<f32>::from_fn(10, cols, |r, c| f[key][r][c].as_f64().unwrap() as f32)
        };
        let state = matrix("state_jacobian", 69);
        let landmark = matrix("landmark_jacobian", 3);
        let residual = DVector::from_iterator(
            10,
            f["residual"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32),
        );
        let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual).unwrap();
        let bytes =
            std::fs::read(native.join(format!("visual_iter{i}_ordinal0_storage.f32"))).unwrap();
        assert_eq!(bytes.len(), 13 * 76 * 4);
        let words: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let transformed = qr.transformed_state();
        let qres = qr.transformed_residual();
        let upper = qr.upper_r();
        let mut mismatch = Vec::new();
        for row in 0..10 {
            for col in 0..69 {
                if transformed[(row, col)].to_bits() != words[row * 76 + col].to_bits() {
                    mismatch.push(format!(
                        "state[{row},{col}] native={:08x} rust={:08x}",
                        words[row * 76 + col].to_bits(),
                        transformed[(row, col)].to_bits()
                    ));
                }
            }
            if qres[row].to_bits() != words[row * 76 + 75].to_bits() {
                mismatch.push(format!("residual[{row}]"));
            }
        }
        for row in 0..3 {
            for col in row..3 {
                if upper[(row, col)].to_bits() != words[row * 76 + 72 + col].to_bits() {
                    mismatch.push(format!("R[{row},{col}]"));
                }
            }
        }
        println!(
            "M11_TRACK1259_QR iter={i} mismatch={} first={:?}",
            mismatch.len(),
            mismatch.first()
        );
        assert!(mismatch.is_empty(), "QR boundary iteration {i}");
        let step_bytes =
            std::fs::read(native.join(format!("frame24_inc_entry_iter{i}.f32"))).unwrap();
        assert_eq!(step_bytes.len(), 69 * 4);
        let step = DVector::from_iterator(
            69,
            step_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
        let blocks: Vec<serde_json::Value> =
            std::fs::read_to_string(native.join("visual_model_blocks.jsonl"))
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
        let expected = blocks
            .iter()
            .find(|b| b["iteration"] == i && b["ordinal"] == 0)
            .unwrap();
        assert_eq!(expected["before_bits"], "00000000");
        let expected = u32::from_str_radix(expected["after_bits"].as_str().unwrap(), 16).unwrap();
        for mode in 0..16 {
            let mut inc = if mode & 1 != 0 {
                eigen_row_major_gemv_f32(&transformed, &step)
            } else {
                &transformed * &step
            };
            let head = if mode & 2 != 0 {
                eigen_row_major_gemv_f32(&transformed.rows(0, 3).into_owned(), &step)
            } else {
                inc.rows(0, 3).into_owned()
            };
            let rhs = qres.rows(0, 3).into_owned() + head;
            let mut lm = DVector::<f32>::zeros(3);
            if mode & 4 != 0 {
                let x2 = rhs[2] / upper[(2, 2)];
                let x1 = (-upper[(1, 2)]).mul_add(x2, rhs[1]) / upper[(1, 1)];
                let dot = upper[(0, 2)].mul_add(x2, upper[(0, 1)] * x1);
                let x0 = (rhs[0] - dot) / upper[(0, 0)];
                lm[0] = -x0;
                lm[1] = -x1;
                lm[2] = -x2;
            } else {
                for row in (0..3).rev() {
                    let mut value = -rhs[row];
                    for col in row + 1..3 {
                        value -= upper[(row, col)] * lm[col];
                    }
                    lm[row] = value / upper[(row, row)];
                }
            }
            let extra = &upper * lm;
            for row in 0..3 {
                inc[row] += extra[row];
            }
            let actual = if mode & 8 != 0 {
                // Candidate only: 8-lane product reduction followed by scalar FMA tail.
                let products: [f32; 8] =
                    std::array::from_fn(|k| -inc[k] * 0.5_f32.mul_add(inc[k], qres[k]));
                let half: [f32; 4] = std::array::from_fn(|k| products[k] + products[k + 4]);
                let mut sum = (half[0] + half[2]) + (half[1] + half[3]);
                for k in 8..10 {
                    sum = (-inc[k]).mul_add(0.5_f32.mul_add(inc[k], qres[k]), sum);
                }
                sum
            } else {
                -inc.dot(&(0.5_f32 * &inc + &qres))
            };
            println!("M11_TRACK1259_MODEL iter={i} mode={mode} native={expected:08x} actual={:08x} exact={}", actual.to_bits(), actual.to_bits()==expected);
        }
    }
    assert!(seen.into_iter().all(|x| x));
}

#[test]
#[ignore = "requires validated native frame24 visual capture and r28 detail on E"]
fn frame24_all_visual_model_packet_schedule_exact() {
    use std::collections::HashMap;
    use std::io::BufRead;

    let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
    let native = root.join("m11_native_frame24_visual_model_callsite_20260909/r2");
    for gate in [
        "engine.rc",
        "gdb.wrapper.rc",
        "probe.validation.rc",
        "binding.pre_post.cmp.rc",
    ] {
        assert_eq!(
            std::fs::read_to_string(native.join(gate)).unwrap().trim(),
            "0"
        );
    }

    let blocks: Vec<serde_json::Value> =
        std::fs::read_to_string(native.join("visual_model_blocks.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
    assert_eq!(blocks.len(), 65 * 8);
    let mut by_iteration: [Vec<&serde_json::Value>; 8] = std::array::from_fn(|_| Vec::new());
    for block in &blocks {
        let iteration = block["iteration"].as_u64().unwrap() as usize;
        assert!(iteration < 8);
        by_iteration[iteration].push(block);
    }
    for entries in &mut by_iteration {
        entries.sort_by_key(|block| block["ordinal"].as_u64().unwrap());
        assert_eq!(entries.len(), 65);
    }

    let input = std::fs::File::open(
        root.join("m11_trial_wired_frame10_20260908/r28_frame24/detail_iterations.jsonl"),
    )
    .unwrap();
    let mut seen = [false; 8];
    for line in std::io::BufReader::new(input).lines() {
        let record: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        if record["frame_id"] != 24 || record["phase"] != "iteration_start" {
            continue;
        }
        let iteration = record["iteration"].as_u64().unwrap() as usize;
        assert!(iteration < 8 && !seen[iteration]);
        seen[iteration] = true;

        let factors: HashMap<(u64, u64, [u32; 2], u32), &serde_json::Value> = record
            ["landmark_factors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|factor| {
                let direction: [u32; 2] = std::array::from_fn(|index| {
                    (factor["direction"][index].as_f64().unwrap() as f32).to_bits()
                });
                (
                    (
                        factor["host_timestamp_ns"].as_u64().unwrap(),
                        factor["host_cam"].as_u64().unwrap(),
                        direction,
                        (factor["rho"].as_f64().unwrap() as f32).to_bits(),
                    ),
                    factor,
                )
            })
            .collect();
        assert_eq!(factors.len(), 65);

        let step_bytes =
            std::fs::read(native.join(format!("frame24_inc_entry_iter{iteration}.f32"))).unwrap();
        assert_eq!(step_bytes.len(), 69 * 4);
        let step = DVector::from_iterator(
            69,
            step_bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())),
        );

        let mut accumulator = 0.0_f32;
        let mut production_factors = Vec::with_capacity(65);
        for block in &by_iteration[iteration] {
            let ordinal = block["ordinal"].as_u64().unwrap() as usize;
            let hex = |value: &serde_json::Value| {
                u32::from_str_radix(value.as_str().unwrap(), 16).unwrap()
            };
            let key = (
                block["host_timestamp_ns"].as_u64().unwrap(),
                block["host_camera"].as_u64().unwrap(),
                [
                    hex(&block["direction_bits"][0]),
                    hex(&block["direction_bits"][1]),
                ],
                hex(&block["inverse_distance_bits"]),
            );
            let factor = factors[&key];
            let rows = factor["residual"].as_array().unwrap().len();
            let matrix = |key: &str, columns: usize| {
                DMatrix::<f32>::from_fn(rows, columns, |row, column| {
                    factor[key][row][column].as_f64().unwrap() as f32
                })
            };
            let state = matrix("state_jacobian", 69);
            let landmark = matrix("landmark_jacobian", 3);
            let residual = DVector::from_iterator(
                rows,
                factor["residual"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32),
            );
            production_factors.push(
                WhitenedFactorRowStack::new(
                    state.map(|value| value as f64),
                    landmark.map(|value| value as f64),
                    residual.map(|value| value as f64),
                )
                .unwrap()
                .with_kind(FactorKind::Visual),
            );
            let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual).unwrap();
            assert_eq!(
                qr.pivots
                    .iter()
                    .filter(|pivot| pivot.abs() > 1.0e-10_f32)
                    .count(),
                3,
                "iteration {iteration} ordinal {ordinal} rank"
            );
            let transformed_state = qr.transformed_state();
            let transformed_residual = qr.transformed_residual();
            let upper = qr.upper_r();
            let storage_bytes = std::fs::read(native.join(format!(
                "visual_iter{iteration}_ordinal{ordinal}_storage.f32"
            )))
            .unwrap();
            assert_eq!(storage_bytes.len(), (rows + 3) * 76 * 4);
            let storage: Vec<f32> = storage_bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            for row in 0..rows {
                for column in 0..69 {
                    assert_eq!(
                        transformed_state[(row, column)].to_bits(),
                        storage[row * 76 + column].to_bits(),
                        "iteration {iteration} ordinal {ordinal} QJ[{row},{column}]"
                    );
                }
                assert_eq!(
                    transformed_residual[row].to_bits(),
                    storage[row * 76 + 75].to_bits(),
                    "iteration {iteration} ordinal {ordinal} Qr[{row}]"
                );
            }
            for row in 0..3 {
                for column in row..3 {
                    assert_eq!(
                        upper[(row, column)].to_bits(),
                        storage[row * 76 + 72 + column].to_bits(),
                        "iteration {iteration} ordinal {ordinal} R[{row},{column}]"
                    );
                }
            }
            let mut increment = eigen_row_major_gemv_f32(&transformed_state, &step);
            let rhs =
                transformed_residual.rows(0, 3).into_owned() + increment.rows(0, 3).into_owned();
            let mut landmark_increment = DVector::<f32>::zeros(3);
            let x2 = rhs[2] / upper[(2, 2)];
            let x1 = (-upper[(1, 2)]).mul_add(x2, rhs[1]) / upper[(1, 1)];
            let upper_dot = upper[(0, 2)].mul_add(x2, upper[(0, 1)] * x1);
            let x0 = (rhs[0] - upper_dot) / upper[(0, 0)];
            landmark_increment[0] = -x0;
            landmark_increment[1] = -x1;
            landmark_increment[2] = -x2;
            let q1_increment = &upper * landmark_increment;
            for row in 0..3 {
                increment[row] += q1_increment[row];
            }

            let before = u32::from_str_radix(block["before_bits"].as_str().unwrap(), 16).unwrap();
            let after = u32::from_str_radix(block["after_bits"].as_str().unwrap(), 16).unwrap();
            assert_eq!(
                accumulator.to_bits(),
                before,
                "iteration {iteration} ordinal {ordinal} before"
            );
            accumulator -= eigen_visual_model_dot_f32(&increment, &transformed_residual);
            assert_eq!(
                accumulator.to_bits(),
                after,
                "iteration {iteration} ordinal {ordinal} after"
            );
        }
        let production_step = step.map(|value| value as f64);
        let production = model_cost_decrease_f32(&production_factors, &production_step, 1.0e-10_f64)
            .unwrap() as f32;
        assert_eq!(
            production.to_bits(),
            accumulator.to_bits(),
            "iteration {iteration} production visual total"
        );
    }
    assert!(seen.into_iter().all(|value| value));
}

#[test]
#[ignore = "requires external pinned-native frame24 prior capture"]
fn frame24_prior39_normal_rhs_candidate() {
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
    let parse = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
            .collect()
    };
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(input["iteration"].as_u64(), Some(i as u64));
        assert_eq!(
            (input["rows"].as_u64(), input["cols"].as_u64()),
            (Some(39), Some(39))
        );
        let j = DMatrix::from_column_slice(39, 39, &parse(&input["jacobian_bits"]));
        let stage = |name: &str| -> Vec<f32> {
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
        let r = DVector::from_vec(stage("adjusted_rhs"));
        let expected = stage("normal_rhs");
        let jt = j.transpose().into_owned();
        let actual = eigen_prior_row_major_gemv_39_f32(&jt, &r);
        for lane in 0..39 {
            assert_eq!(
                actual[lane].to_bits(),
                expected[lane].to_bits(),
                "iteration {i} lane {lane}"
            );
        }
        let old = eigen_row_major_gemv_f32(&jt, &r);
        assert_eq!(
            old.iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count(),
            [3, 4, 3, 2, 3, 4, 2, 5][i]
        );
    }
}
#[test]
#[ignore = "requires validated native trial computeRelPose capture on E"]
fn m11_native_trial_full_transform_capture_exact() {
    use std::io::BufRead;
    let path = std::env::var("M11_NATIVE_RELPOSE_STAGES").unwrap();
    let mut counts = [[0_usize; 5]; 8];
    let mut calls = [0_usize; 8];
    let mut first = None;
    for line in std::io::BufReader::new(std::fs::File::open(path).unwrap()).lines() {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let word = |v: &serde_json::Value| u32::from_str_radix(v.as_str().unwrap(), 16).unwrap();
        let pose = |v: &serde_json::Value| {
            let values = v
                .as_array()
                .unwrap()
                .iter()
                .map(|x| f32::from_bits(word(x)))
                .collect::<Vec<_>>();
            assert_eq!(values.len(), 7);
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    values[3], values[0], values[1], values[2],
                )),
                translation: Vector3::new(values[4], values[5], values[6]),
            }
        };
        let host = pose(&row["inputs"]["host_imu"]);
        let target = pose(&row["inputs"]["target_imu"]);
        let host_camera = pose(&row["inputs"]["host_camera"]);
        let target_camera = pose(&row["inputs"]["target_camera"]);
        let (tmp2, inverse, relative, prefix, result) =
            super::upstream_trial_transform_stages_f32(host, target, host_camera, target_camera);
        let stage1 = &row["stages"][0];
        let stage2 = &row["stages"][1];
        assert_eq!(stage1["elf_pc"], "394998");
        assert_eq!(stage2["elf_pc"], "394b07");
        let words =
            |v: &serde_json::Value| v.as_array().unwrap().iter().map(word).collect::<Vec<_>>();
        let stack1 = words(&stage1["stack_80"]);
        let stack2 = words(&stage2["stack_80"]);
        let mut relative_expected = words(&stage1["xmm"]["3"]);
        relative_expected.extend([stack1[12], stack1[13], word(&stage1["stack_210"][2])]);
        let mut prefix_expected = stack2[16..20].to_vec();
        prefix_expected.extend([word(&stage2["xmm"]["6"][0]), word(&stage2["xmm"]["7"][0])]);
        let prefix_values = visual_chain_pose_snapshot(prefix);
        let inverse_values = visual_chain_pose_snapshot(F32Pose {
            rotation: inverse,
            translation: Vector3::zeros(),
        });
        let actual = [
            visual_chain_pose_snapshot(tmp2).to_vec(),
            inverse_values[..4].to_vec(),
            visual_chain_pose_snapshot(relative).to_vec(),
            prefix_values[..4]
                .iter()
                .chain(prefix_values[5..7].iter())
                .copied()
                .collect(),
            visual_chain_pose_snapshot(result).to_vec(),
        ];
        let expected = [
            stack1[..7].to_vec(),
            words(&stage1["stack_180"])[..4].to_vec(),
            relative_expected,
            prefix_expected,
            words(&row["result"]),
        ];
        let iteration = row["iteration"].as_u64().unwrap() as usize;
        for stage in 0..5 {
            assert_eq!(actual[stage].len(), expected[stage].len());
            for lane in 0..actual[stage].len() {
                if actual[stage][lane].to_bits() == expected[stage][lane] {
                    counts[iteration][stage] += 1;
                } else if first.is_none() {
                    first = Some(
                        serde_json::json!({"iteration":iteration,"call":calls[iteration],"stage":stage,"lane":lane,
                        "native":format!("{:08x}",expected[stage][lane]),"rust":format!("{:08x}",actual[stage][lane].to_bits())}),
                    );
                }
            }
        }
        calls[iteration] += 1;
    }
    eprintln!(
        "M11_TRIAL_TRANSFORM {}",
        serde_json::json!({"stages":["camera_inverse","target_inverse_q","relative","prefix_q_y_z","result"],"exact_lanes":counts,"calls":calls,"first_mismatch":first})
    );
    assert_eq!(calls, [13; 8]);
    assert_eq!(counts, [[91, 52, 91, 78, 91]; 8]);
}

#[test]
fn m11_native_trial_projection_domain_matches_source() {
    let camera = DoubleSphereCamera::new(100.0, 100.0, 0.0, 0.0, 0.0, 1.0, 640, 480).unwrap();
    // alpha=1, xi=0 makes w2 exactly zero, so the z boundary is strict.
    assert!(super::upstream_trial_project_f32(&camera, Vector3::new(1.0, 0.0, 0.0)).is_none());
    assert!(super::upstream_trial_project_f32(&camera, Vector3::new(1.0, 0.0, 1e-20)).is_some());
    assert!(super::upstream_trial_project_f32(&camera, Vector3::new(1.0, 0.0, -1e-20)).is_none());
    assert!(super::upstream_trial_project_f32(&camera, Vector3::zeros()).is_none());
    assert!(super::upstream_trial_project_f32(&camera, Vector3::new(f32::NAN, 0.0, 1.0)).is_none());
    let pinhole = DoubleSphereCamera::new(100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 640, 480).unwrap();
    assert!(super::upstream_trial_project_f32(&pinhole, Vector3::new(0.0, 0.0, 1e-12)).is_some());
    let outside =
        super::upstream_trial_project_f32(&pinhole, Vector3::new(10.0, 0.0, 1.0)).unwrap();
    assert_eq!(outside.x, 1000.0);
}

#[test]
#[ignore = "requires validated native point capture and matching Rust observation inputs on E"]
fn m11_native_trial_point_and_projection_all_observations_exact() {
    use std::io::BufRead;
    let native = std::env::var("M11_NATIVE_VISUAL_OBSERVATIONS").unwrap();
    let detail = std::env::var("M11_RUST_TRIAL_DETAIL").unwrap();
    let calibration = std::env::var("M11_NATIVE_CALIBRATION").unwrap();
    let calibration: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(calibration).unwrap()).unwrap();
    let cameras = calibration["value0"]["intrinsics"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            assert_eq!(entry["camera_type"], "ds");
            let c = &entry["intrinsics"];
            let resolution = &calibration["value0"]["resolution"][index];
            DoubleSphereCamera::new(
                c["fx"].as_f64().unwrap(),
                c["fy"].as_f64().unwrap(),
                c["cx"].as_f64().unwrap(),
                c["cy"].as_f64().unwrap(),
                c["xi"].as_f64().unwrap(),
                c["alpha"].as_f64().unwrap(),
                resolution[0].as_u64().unwrap() as u32,
                resolution[1].as_u64().unwrap() as u32,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut pixels = std::collections::BTreeMap::new();
    for line in std::io::BufReader::new(std::fs::File::open(detail).unwrap()).lines() {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        if row["phase"] != "trial" {
            continue;
        }
        assert_eq!(row["iteration"], 0);
        assert_eq!(
            row["trial_visual_costs"]["state_boundary"],
            "candidate_after_step"
        );
        for landmark in row["landmarks"].as_array().unwrap() {
            for observation in landmark["observations"].as_array().unwrap() {
                let key = (
                    landmark["track_id"].as_u64().unwrap(),
                    observation["target_timestamp_ns"].as_u64().unwrap(),
                    observation["target_cam"].as_u64().unwrap(),
                );
                let pixel = Vector2::new(
                    observation["pixel"][0].as_f64().unwrap() as f32,
                    observation["pixel"][1].as_f64().unwrap() as f32,
                );
                assert!(pixels.insert(key, pixel).is_none());
            }
        }
        break;
    }
    assert_eq!(pixels.len(), 739);
    let mut counts = [0_usize; 8];
    for line in std::io::BufReader::new(std::fs::File::open(native).unwrap()).lines() {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let parse = |v: &serde_json::Value| {
            f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap())
        };
        let key = (
            row["track_id"].as_u64().unwrap(),
            row["target_timestamp_ns"].as_u64().unwrap(),
            row["target_cam"].as_u64().unwrap(),
        );
        let parameters = &row["landmark_parameter"];
        let bearing = super::upstream_trial_bearing_f32(Vector2::new(
            parse(&parameters[0]),
            parse(&parameters[1]),
        ));
        let matrix = SMatrix::<f32, 4, 4>::from_iterator(
            row["T_t_h_column_major"]
                .as_array()
                .unwrap()
                .iter()
                .map(parse),
        );
        let point =
            super::eigen_homogeneous_point_product_f32(matrix, bearing, parse(&parameters[2]));
        for lane in 0..4 {
            assert_eq!(
                point[lane].to_bits(),
                parse(&row["point4_target"][lane]).to_bits(),
                "point {key:?} lane{lane}"
            );
        }
        let (raw, cost) = super::upstream_trial_observation_f32(
            &cameras[key.2 as usize],
            matrix,
            Vector3::new(
                parse(&parameters[0]),
                parse(&parameters[1]),
                parse(&parameters[2]),
            ),
            pixels[&key],
            FactorConfig::default(),
        )
        .expect("native captured observation is valid");
        assert_eq!(
            cost.to_bits(),
            parse(&row["objective"]).to_bits(),
            "cost {key:?}"
        );
        for lane in 0..2 {
            assert_eq!(
                raw[lane].to_bits(),
                parse(&row["raw_residual"][lane]).to_bits(),
                "residual {key:?} lane{lane}"
            );
        }
        counts[row["iteration"].as_u64().unwrap() as usize] += 1;
    }
    assert_eq!(counts, [739; 8]);
}

#[test]
#[ignore = "requires validated native frame8 observation capture on E"]
fn m11_native_trial_visual_cost_all_observations_exact() {
    use std::io::BufRead;
    let path = std::env::var("M11_NATIVE_VISUAL_OBSERVATIONS")
        .expect("explicit validated native observation path required");
    let file = std::fs::File::open(path).unwrap();
    let mut counts = [0_usize; 8];
    for line in std::io::BufReader::new(file).lines() {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let iteration = row["iteration"].as_u64().unwrap() as usize;
        let word = |v: &serde_json::Value| u32::from_str_radix(v.as_str().unwrap(), 16).unwrap();
        let x = f32::from_bits(word(&row["raw_residual"][0]));
        let y = f32::from_bits(word(&row["raw_residual"][1]));
        let actual = super::upstream_trial_visual_cost_f32(x, y, 0.5, 1.0);
        assert_eq!(
            actual.to_bits(),
            word(&row["objective"]),
            "iteration {iteration}, track {}",
            row["track_id"]
        );
        counts[iteration] += 1;
    }
    assert_eq!(counts, [739; 8]);
}
#[test]
#[ignore = "requires external frame7 native chain fixture on E"]
fn m11_frame7_current_fej_matrix_probe() {
    let root = std::env::var("M11_TRI_PROBE_ROOT").unwrap();
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!(
            "{root}/m11_native_frame7_iter1_relpose_20260908/r1/current_chain_fixture.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let calibration: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("../../target/euroc_ds_calib.json").unwrap())
            .unwrap();
    let parse = |v: &serde_json::Value| {
        let f = |i: usize| f32::from_bits(u32::from_str_radix(v[i].as_str().unwrap(), 16).unwrap());
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(f(3), f(0), f(1), f(2))),
            translation: Vector3::new(f(4), f(5), f(6)),
        }
    };
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 15);
    for case in cases {
        let chain = &case["rust_chain"];
        let current = parse(&chain["target_camera_from_anchor_camera_f32_bits"]);
        let prefix = parse(&chain["target_camera_from_anchor_imu_f32_bits"]);
        let fej = parse(&chain["target_camera_from_anchor_imu_fej_f32_bits"]);
        let host_cam = case["endpoint"][1].as_u64().unwrap() as usize;
        let e = &calibration["value0"]["T_imu_cam"][host_cam];
        let t = Vector3::new(
            e["px"].as_f64().unwrap() as f32,
            e["py"].as_f64().unwrap() as f32,
            e["pz"].as_f64().unwrap() as f32,
        );
        let alternative_t = sophus_rotate_f32(fej.rotation, t) + fej.translation;
        let native: Vec<u32> = case["native"]["T_t_h"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u32::from_str_radix(v.as_str().unwrap(), 16).unwrap())
            .collect();
        let rotation = eigen_quaternion_matrix_f32(current.rotation);
        let mut matrix = [0u32; 16];
        for c in 0..3 {
            for r in 0..3 {
                matrix[c * 4 + r] = rotation[(r, c)].to_bits();
            }
        }
        for i in 0..3 {
            matrix[12 + i] = current.translation[i].to_bits();
        }
        matrix[15] = 1.0f32.to_bits();
        let current_exact = matrix.iter().zip(&native).filter(|(a, b)| a == b).count();
        for i in 0..3 {
            matrix[12 + i] = alternative_t[i].to_bits();
        }
        let alternative_exact = matrix.iter().zip(&native).filter(|(a, b)| a == b).count();
        println!(
            "M11_FEJ_MATRIX {}",
            serde_json::json!({"endpoint":case["endpoint"],"host":case["host_frame"],"target":case["target_frame"],"current_exact":current_exact,"fej_translation_only_exact":alternative_exact,"prefix_rotation_equal":prefix.rotation.coords.iter().zip(fej.rotation.coords.iter()).all(|(a,b)|a.to_bits()==b.to_bits())})
        );
    }
}
use super::*;
use crate::vio::landmarks::StereographicDirection;
use nalgebra::{UnitQuaternion, Vector2};
use std::cell::Cell;
fn factor(
    js: &[f64],
    jl: &[f64],
    r: &[f64],
    rows: usize,
    state: usize,
    ld: usize,
) -> WhitenedFactorRowStack {
    WhitenedFactorRowStack::new(
        DMatrix::from_row_slice(rows, state, js),
        DMatrix::from_row_slice(rows, ld, jl),
        DVector::from_column_slice(r),
    )
    .unwrap()
}

fn assert_f32_bits(actual: &[f32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&value, &bits)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            value.to_bits(),
            bits,
            "f32 lane {index}: got {:08x}, expected {bits:08x}",
            value.to_bits()
        );
    }
}

fn assert_matrix_f32_bitwise_equal(actual: &DMatrix<f32>, expected: &DMatrix<f32>, label: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{label} shape");
    for (index, (&actual, &expected)) in actual
        .as_slice()
        .iter()
        .zip(expected.as_slice())
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} lane {index}: got {:08x}, expected {:08x}",
            actual.to_bits(),
            expected.to_bits()
        );
    }
}

fn assert_vector_f32_bitwise_equal(actual: &DVector<f32>, expected: &DVector<f32>, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual
        .as_slice()
        .iter()
        .zip(expected.as_slice())
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} lane {index}: got {:08x}, expected {:08x}",
            actual.to_bits(),
            expected.to_bits()
        );
    }
}

fn assert_landmark_qr_bitwise_equal(
    actual: &LandmarkHouseholderF32,
    expected: &LandmarkHouseholderF32,
    label: &str,
) {
    assert_eq!(actual.rows, expected.rows, "{label} rows");
    assert_eq!(actual.state_cols, expected.state_cols, "{label} state cols");
    assert_eq!(
        actual.landmark_cols, expected.landmark_cols,
        "{label} landmark cols"
    );
    assert_eq!(
        actual.landmark_offset, expected.landmark_offset,
        "{label} landmark offset"
    );
    assert_eq!(
        actual.residual_offset, expected.residual_offset,
        "{label} residual offset"
    );
    assert_f32_bits(
        &actual.storage,
        &expected
            .storage
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
    assert_f32_bits(
        &actual.pivots,
        &expected
            .pivots
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
    assert_f32_bits(
        &actual.tau,
        &expected
            .tau
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
    let _ = label;
}

fn assert_compact_payload_bitwise_equal(
    materialized: &CompactLandmarkBackSubstitutionF32,
    arena: &[f32],
    entry: &CompactLandmarkBackSubstitutionEntryF32,
    label: &str,
) {
    assert_eq!(
        materialized.landmark_index, entry.landmark_index,
        "{label} landmark index"
    );
    assert_eq!(materialized.track_id, entry.track_id, "{label} track id");
    assert_eq!(
        materialized.state_cols, entry.state_cols,
        "{label} state cols"
    );
    assert_eq!(
        materialized.landmark_cols, entry.landmark_cols,
        "{label} landmark cols"
    );
    assert_eq!(materialized.rank, entry.rank, "{label} rank");
    assert_eq!(materialized.eligible, entry.eligible, "{label} eligible");
    let end = entry
        .storage_offset
        .checked_add(materialized.storage.len())
        .expect("compact payload offset overflow");
    assert!(end <= arena.len(), "{label} arena bounds");
    assert_f32_bits(
        &materialized.storage,
        &arena[entry.storage_offset..end]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
}

/// Test-only spelling of the native quaternion-to-matrix diagonal path.
///
/// The pinned Eigen/Sophus codegen contracts the two diagonal products
/// before subtracting from one (`fma(x, 2*x, yy/zz)`).  The production
/// helper above intentionally remains unchanged until this candidate has
/// passed every captured relative-pose fixture.  The inverse is supplied
/// by `sophus_so3_inverse`, whose packet implementation already performs
/// the native pairwise normalization.
fn m7_candidate_quaternion_matrix_f32(rotation: UnitQuaternion<f32>) -> Matrix3<f32> {
    let q = rotation.quaternion();
    let tx = 2.0_f32 * q.i;
    let ty = 2.0_f32 * q.j;
    let tz = 2.0_f32 * q.k;
    let txy = ty * q.i;
    let txz = tz * q.i;
    let tyy = ty * q.j;
    let tyz = tz * q.j;
    let tzz = tz * q.k;
    Matrix3::new(
        1.0_f32 - (tyy + tzz),
        (-tz).mul_add(q.w, txy),
        ty.mul_add(q.w, txz),
        tz.mul_add(q.w, txy),
        1.0_f32 - q.i.mul_add(tx, tzz),
        (-tx).mul_add(q.w, tyz),
        (-ty).mul_add(q.w, txz),
        tx.mul_add(q.w, tyz),
        1.0_f32 - q.i.mul_add(tx, tyy),
    )
}

/// Test-only candidate retaining the production signed-left-operand
/// boundary and increasing-k six-term Eigen reduction.
fn m7_candidate_adjoint_times_rotation_blocks_f32(
    pose: F32Pose,
    rotation: Matrix3<f32>,
    left_sign: f32,
) -> SMatrix<f32, 6, 6> {
    let pose_rotation = m7_candidate_quaternion_matrix_f32(pose.rotation);
    let cross_rotation = eigen_matrix_product_3x3_f32(skew3_f32(pose.translation), pose_rotation);
    let mut adjoint = SMatrix::<f32, 6, 6>::zeros();
    adjoint
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&pose_rotation);
    adjoint
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&cross_rotation);
    adjoint
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&pose_rotation);
    let mut rotation_blocks = SMatrix::<f32, 6, 6>::zeros();
    rotation_blocks
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation);
    rotation_blocks
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&rotation);
    eigen_matrix_product_6x6_f32(adjoint * left_sign, rotation_blocks)
}

fn tagged_imu_bias_pair(
    state_dof: usize,
    imu_offsets: Option<ImuLinkOffsets>,
    bias_offsets: Option<ImuLinkOffsets>,
    unexpected_column: Option<usize>,
) -> (WhitenedFactorRowStack, WhitenedFactorRowStack) {
    let mut imu_jacobian = DMatrix::<f32>::zeros(9, state_dof);
    let mut bias_jacobian = DMatrix::<f32>::zeros(6, state_dof);
    if let Some(offsets) = imu_offsets {
        for row in 0..9 {
            for column in 0..AOM_NAV_DOF * 2 {
                let block = column / AOM_NAV_DOF;
                let global_offset = if block == 0 {
                    offsets.start
                } else {
                    offsets.end
                };
                let global_column = global_offset + column % AOM_NAV_DOF;
                if global_column < state_dof {
                    imu_jacobian[(row, global_column)] =
                        1.0 + (row * AOM_NAV_DOF * 2 + column) as f32 * 0.03125;
                }
            }
        }
        for row in 0..6 {
            for column in 0..AOM_NAV_DOF * 2 {
                let block = column / AOM_NAV_DOF;
                let global_offset = if block == 0 {
                    offsets.start
                } else {
                    offsets.end
                };
                let global_column = global_offset + column % AOM_NAV_DOF;
                if global_column < state_dof {
                    bias_jacobian[(row, global_column)] =
                        -0.5 + (row * AOM_NAV_DOF * 2 + column) as f32 * 0.0625;
                }
            }
        }
    }
    if let Some(column) = unexpected_column {
        imu_jacobian[(0, column)] = 7.0;
    }
    let imu_residual = DVector::from_fn(9, |row, _| 0.25 + row as f32 * 0.125);
    let bias_residual = DVector::from_fn(6, |row, _| -0.5 + row as f32 * 0.03125);
    let imu = WhitenedFactorRowStack::with_objective_cost_kind(
        imu_jacobian.map(f64::from),
        DMatrix::zeros(9, 0),
        imu_residual.map(f64::from),
        0.0,
        FactorKind::Imu,
    )
    .unwrap();
    let bias = WhitenedFactorRowStack::with_objective_cost_kind(
        bias_jacobian.map(f64::from),
        DMatrix::zeros(6, 0),
        bias_residual.map(f64::from),
        0.0,
        FactorKind::Bias,
    )
    .unwrap();
    let imu = if let Some(offsets) = imu_offsets {
        imu.with_imu_link_offsets(offsets.start, offsets.end)
    } else {
        imu
    };
    let bias = if let Some(offsets) = bias_offsets {
        bias.with_imu_link_offsets(offsets.start, offsets.end)
    } else {
        bias
    };
    (imu, bias)
}

fn fixture_f32_bits(value: &serde_json::Value, field: &str) -> Vec<u32> {
    value[field]
        .as_array()
        .unwrap_or_else(|| panic!("fixture field {field} is not an array"))
        .iter()
        .map(|word| {
            let word = word
                .as_str()
                .unwrap_or_else(|| panic!("fixture field {field} contains a non-string"));
            u32::from_str_radix(word, 16)
                .unwrap_or_else(|error| panic!("invalid fixture f32 word {word}: {error}"))
        })
        .collect()
}

fn fixture_matrix_bits(value: &serde_json::Value) -> Vec<u32> {
    let rows = value["rows"].as_u64().unwrap() as usize;
    let cols = value["cols"].as_u64().unwrap() as usize;
    let bits = fixture_f32_bits(value, "bits_row_major");
    assert_eq!(bits.len(), rows * cols);
    bits
}

#[test]
#[ignore = "requires pinned external m7im15 full-product capture"]
fn m7im15_local_product_matches_full_900_30_oracle_bits() {
    let fixture_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/m7im15_test_new.json");
    let fixture_text = std::fs::read_to_string(&fixture_path).unwrap_or_else(|error| {
        panic!(
            "m7im15 oracle fixture {} is required for this bit gate: {error}",
            fixture_path.display()
        )
    });
    let fixture: serde_json::Value = serde_json::from_str(&fixture_text).unwrap();
    let input = &fixture["inputs"]["Jp"];
    assert_eq!(input["rows"].as_u64(), Some(IMU_LOCAL_ROWS as u64));
    assert_eq!(input["cols"].as_u64(), Some(IMU_LOCAL_COLS as u64));
    let jacobian_bits = fixture_matrix_bits(input);
    let residual_bits = fixture_f32_bits(&fixture["inputs"]["r"], "bits");
    let jacobian = DMatrix::from_row_slice(
        IMU_LOCAL_ROWS,
        IMU_LOCAL_COLS,
        &jacobian_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>(),
    );
    let residual = DVector::from_column_slice(
        &residual_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>(),
    );
    let (actual_h, actual_b) = local_imu_h_b_15x30(&jacobian, &residual);

    let expected_h = fixture_matrix_bits(&fixture["native_eigen"]["H"]);
    let expected_b = fixture_f32_bits(&fixture["native_eigen"]["b"], "bits");
    let actual_h_ref = &actual_h;
    let actual_h_row_major = (0..IMU_LOCAL_COLS)
        .flat_map(|row| (0..IMU_LOCAL_COLS).map(move |column| actual_h_ref[(row, column)]))
        .collect::<Vec<_>>();
    assert_f32_bits(&actual_h_row_major, &expected_h);
    assert_f32_bits(actual_b.as_slice(), &expected_b);
    assert_eq!(expected_h.len(), 900);
    assert_eq!(expected_b.len(), 30);
}

#[test]
fn m7im15_scatter_at_nonzero_offsets_preserves_cross_block_order() {
    let mut accumulator_h = DMatrix::<f32>::zeros(40, 40);
    let mut accumulator_b = DVector::<f32>::zeros(40);
    let local_h = DMatrix::from_fn(IMU_LOCAL_COLS, IMU_LOCAL_COLS, |row, column| {
        (row * IMU_LOCAL_COLS + column) as f32 + 0.25
    });
    let local_b = DVector::from_fn(IMU_LOCAL_COLS, |row, _| row as f32 + 0.5);
    scatter_local_imu_h_b_15x30(
        &mut accumulator_h,
        &mut accumulator_b,
        &local_h,
        &local_b,
        ImuLinkOffsets { start: 6, end: 21 },
    )
    .unwrap();

    for row in 0..IMU_LOCAL_COLS {
        for column in 0..IMU_LOCAL_COLS {
            let (global_row, global_column) = if row < AOM_NAV_DOF {
                if column < AOM_NAV_DOF {
                    (6 + row, 6 + column)
                } else {
                    (6 + row, 21 + column - AOM_NAV_DOF)
                }
            } else if column < AOM_NAV_DOF {
                (21 + row - AOM_NAV_DOF, 6 + column)
            } else {
                (21 + row - AOM_NAV_DOF, 21 + column - AOM_NAV_DOF)
            };
            assert_eq!(
                accumulator_h[(global_row, global_column)].to_bits(),
                local_h[(row, column)].to_bits(),
                "scatter mismatch local ({row},{column}) -> global ({global_row},{global_column})"
            );
        }
    }
    for row in 0..AOM_NAV_DOF {
        assert_eq!(accumulator_b[6 + row].to_bits(), local_b[row].to_bits());
        assert_eq!(
            accumulator_b[21 + row].to_bits(),
            local_b[AOM_NAV_DOF + row].to_bits()
        );
    }
    // Distinct cross blocks make an accidental transpose immediately
    // visible.  H10/H01 are copied in their native local orientation.
    assert_eq!(
        accumulator_h[(21 + 2, 6 + 4)].to_bits(),
        local_h[(17, 4)].to_bits()
    );
    assert_eq!(
        accumulator_h[(6 + 2, 21 + 4)].to_bits(),
        local_h[(2, 19)].to_bits()
    );

    let mut too_small_h = DMatrix::<f32>::zeros(35, 35);
    let mut too_small_b = DVector::<f32>::zeros(35);
    assert!(scatter_local_imu_h_b_15x30(
        &mut too_small_h,
        &mut too_small_b,
        &local_h,
        &local_b,
        ImuLinkOffsets { start: 6, end: 21 },
    )
    .is_err());
}

#[test]
fn m7im15_diagnostic_uses_global_width_for_absolute_offsets() {
    let offsets = ImuLinkOffsets { start: 6, end: 21 };
    let local_jacobian = DMatrix::from_fn(IMU_LOCAL_ROWS, IMU_LOCAL_COLS, |row, column| {
        0.25_f32 + (row * IMU_LOCAL_COLS + column) as f32 * 0.03125
    });
    let residual = DVector::from_fn(IMU_LOCAL_ROWS, |row, _| -0.75_f32 + row as f32 * 0.0625);
    let mut global_jacobian = DMatrix::<f32>::zeros(IMU_LOCAL_ROWS, 40);
    for row in 0..IMU_LOCAL_ROWS {
        for column in 0..IMU_LOCAL_COLS {
            let global_column = if column < AOM_NAV_DOF {
                offsets.start + column
            } else {
                offsets.end + column - AOM_NAV_DOF
            };
            global_jacobian[(row, global_column)] = local_jacobian[(row, column)];
        }
    }

    // The local product must remain 30x30 even though its active columns
    // live at absolute offsets 6 and 21 in the 40-column source stack.
    // The old helper indexed a 30x30 product with offset 21 and panicked
    // as soon as the second 15-column block was reached.
    let diagnostic =
        imu_local_block_diagnostic(&local_jacobian, &residual, &global_jacobian, offsets, None);
    assert_eq!(diagnostic.active_offsets, vec![6, 21]);
    assert_eq!(
        diagnostic.local_jacobian.shape(),
        (IMU_LOCAL_ROWS, IMU_LOCAL_COLS)
    );
    assert_eq!(diagnostic.residual.len(), IMU_LOCAL_ROWS);
    assert_eq!(diagnostic.local_jacobian[(0, 0)].to_bits(), 0x3e80_0000);
    assert_eq!(diagnostic.local_jacobian[(14, 29)].to_bits(), 0x4164_8000);
    assert_eq!(diagnostic.residual[0].to_bits(), 0xbf40_0000);
    assert_eq!(diagnostic.residual[14].to_bits(), 0x3e00_0000);
    assert_eq!(diagnostic.local_h.shape(), (IMU_LOCAL_COLS, IMU_LOCAL_COLS));
    assert_eq!(diagnostic.local_b.len(), IMU_LOCAL_COLS);
    assert_eq!(diagnostic.local_vs_global_h_mismatches, 0);
    assert_eq!(diagnostic.local_vs_global_b_mismatches, 0);
    assert_eq!(diagnostic.local_vs_global_padded_h_mismatches, 0);
    assert_eq!(diagnostic.local_vs_global_padded_b_mismatches, 0);
}

fn assert_pose_f32_bits(
    pose: F32Pose,
    expected_translation: [u32; 3],
    expected_quaternion_xyzw: [u32; 4],
) {
    assert_f32_bits(
        &[pose.translation.x, pose.translation.y, pose.translation.z],
        &expected_translation,
    );
    let q = pose.rotation.quaternion();
    assert_f32_bits(&[q.i, q.j, q.k, q.w], &expected_quaternion_xyzw);
}

fn m7_fixed_relative_point() -> (UnitQuaternion<f32>, Vector3<f32>, Vector3<f32>, f32) {
    let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
        f32::from_bits(0x3f7ffd31),
        f32::from_bits(0xbc0e2956),
        f32::from_bits(0xbb507eff),
        f32::from_bits(0xba002940),
    ));
    let translation = Vector3::new(
        f32::from_bits(0xbde1e254),
        f32::from_bits(0xbaf1ecc0),
        f32::from_bits(0xba884364),
    );
    let bearing = Vector3::new(
        f32::from_bits(0xbf2a746e),
        f32::from_bits(0xbe9aed26),
        f32::from_bits(0x3f2e9644),
    );
    (rotation, translation, bearing, f32::from_bits(0x3e160ef3))
}

fn m7_fixed_camera() -> DoubleSphereCamera {
    DoubleSphereCamera::new(
        361.6713883800533,
        360.5856493689301,
        379.40818394080869,
        255.9772968522045,
        -0.21300835384809328,
        0.5767008625037023,
        752,
        480,
    )
    .unwrap()
}

fn m7_fixed_factor_inputs() -> (
    DoubleSphereCamera,
    SE3,
    SE3,
    SE3,
    SE3,
    InverseDistanceLandmark,
    Point2<f64>,
) {
    let anchor_pose = SE3::new(
        UnitQuaternion::new_unchecked(Quaternion::new(
            0.5944822430610657,
            -0.052778493613004684,
            -0.8023747801780701,
            0.0,
        )),
        Vector3::zeros(),
    );
    let target_pose = SE3::new(
        UnitQuaternion::new_unchecked(Quaternion::new(
            0.5954498648643494,
            -0.05392327159643173,
            -0.801576554775238,
            -0.002603980479761958,
        )),
        Vector3::new(
            0.0003828657791018486,
            -9.85765946097672e-5,
            -0.002031802199780941,
        ),
    );
    let anchor_extrinsic = SE3::new(
        UnitQuaternion::new_normalize(Quaternion::new(
            0.7123125505904486,
            -0.007239825785317818,
            0.007541278561558601,
            0.7017845426564943,
        )),
        Vector3::new(
            -0.016774788924641534,
            -0.068938940687127,
            0.005139123188382424,
        ),
    );
    let target_extrinsic = SE3::new(
        UnitQuaternion::new_normalize(Quaternion::new(
            0.7115930283929829,
            -0.0023360576185881625,
            0.013000769689092388,
            0.7024677108343111,
        )),
        Vector3::new(
            -0.01507436282032619,
            0.0412627204046637,
            0.00316287258752953,
        ),
    );
    let landmark = InverseDistanceLandmark {
        anchor_pose: 0,
        anchor_camera_id: 0,
        direction: StereographicDirection {
            xy: Point2::new(-0.39586612582206726, -0.1799013614654541),
        },
        inverse_distance: 0.14654140174388885,
    };
    (
        m7_fixed_camera(),
        anchor_pose,
        anchor_extrinsic,
        target_pose,
        target_extrinsic,
        landmark,
        Point2::new(27.31320571899414, 106.39038848876953),
    )
}

#[test]
fn m7_eigen_quaternion_matrix_matches_pinned_lanes() {
    let (rotation, _, _, _) = m7_fixed_relative_point();
    let matrix = eigen_quaternion_matrix_f32(rotation);
    assert_f32_bits(
        matrix.as_slice(),
        &[
            0x3f7ffea4, 0xba71d6ad, 0x3bd0c3e1, 0x3a87645a, 0x3f7ff61a, 0xbc8e2141, 0xbbd0358a,
            0x3c8e2e4d, 0x3f7ff4ce,
        ],
    );
}

#[test]
fn m7_frame4_iter1_relative_pose_matrix_and_translation_match_native() {
    // Native Basalt frame-4/iter-1 track-120 observations, host
    // frame0/cam0 -> target frame3/cam0 and cam1.  These endpoint
    // quaternions are captured immediately before `linearizePoint`;
    // checking the complete Eigen column-major matrix guards the native
    // f32 conversion schedule used by the current value path.
    let cases = [
        (
            [0x3bcfedd3, 0xbbc0c558, 0x3b389897, 0x3f7ffd4a],
            [
                0x3f7ffa6d, 0x3bb62458, 0x3c41593c, 0xbbbb08ed, 0x3f7ff9af, 0x3c4f609f, 0xbc402d5f,
                0xbc5076a0, 0x3f7ff630,
            ],
        ),
        (
            [0xba7fd817, 0xbbcefd9a, 0x3aeb32ff, 0x3f7ffe8f],
            [
                0x3f7ffa59, 0x3b6c0089, 0x3c4eedbf, 0xbb6a62cf, 0x3f7fff74, 0xbb0167ab, 0xbc4f0b21,
                0x3afcddf6, 0x3f7ffaa5,
            ],
        ),
    ];
    for (q_xyzw, expected_matrix) in cases {
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(q_xyzw[3]),
            f32::from_bits(q_xyzw[0]),
            f32::from_bits(q_xyzw[1]),
            f32::from_bits(q_xyzw[2]),
        ));
        let matrix = eigen_quaternion_matrix_native_f32(rotation);
        assert_f32_bits(matrix.as_slice(), &expected_matrix);
    }
}

#[test]
fn m7_probe_relpose_candidate_vs_generic_iter1_frame3_cam1() {
    fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
        let q = std::hint::black_box(q);
        let t = std::hint::black_box(t);
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        }
    }
    fn drel(pose: F32Pose, host: F32Pose) -> SMatrix<f32, 6, 6> {
        eigen_adjoint_times_rotation_blocks_f32(
            pose,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
            1.0,
        )
    }
    let host = pose([0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd], [0, 0, 0]);
    let target = pose(
        [0xbd5e3af6, 0xbf4e66c7, 0xbbc319f3, 0x3f16cb91],
        [0x3b2b6242, 0xbb51bff3, 0xbd4c7b9e],
    );
    let target_ext = pose(
        [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
        [0xbc76fa76, 0x3d290319, 0x3b4f4832],
    );
    let target_camera_from_imu = std::hint::black_box(target_ext.inverse());
    let relative = sophus_relative_imu_f32(target, host);
    let generic = std::hint::black_box(F32Pose {
        rotation: sophus_quat_product_f32(target_camera_from_imu.rotation, relative.rotation),
        translation: sophus_rotate_f32(target_camera_from_imu.rotation, relative.translation)
            + target_camera_from_imu.translation,
    });
    let candidate = std::hint::black_box(sophus_compute_relpose_tmp_out_of_line_f32(
        target_camera_from_imu,
        target,
        host,
    ));
    let candidate_raw =
        m7_relpose_packet_product_raw_for_test(target_camera_from_imu.rotation, relative.rotation);
    let generic_raw =
        m7_generic_packet_product_raw_for_test(target_camera_from_imu.rotation, relative.rotation);
    let candidate_norm = m7_normalize_packet_product_for_test(candidate_raw);
    let generic_norm = m7_normalize_packet_product_for_test(generic_raw);
    let candidate_drel = drel(candidate, host);
    let generic_drel = drel(generic, host);
    let d_diffs = candidate_drel
        .as_slice()
        .iter()
        .zip(generic_drel.as_slice())
        .enumerate()
        .filter_map(|(index, (actual, expected))| {
            (actual.to_bits() != expected.to_bits()).then_some(format!(
                "{index}:{:08x}/{:08x}",
                actual.to_bits(),
                expected.to_bits()
            ))
        })
        .collect::<Vec<_>>();
    // First actual_eigen_site record in
    // target/m7im15_native_actual_eigen_site_iter1_track119_20260827.jsonl:
    // iteration 1, track 119, target frame 3/cam 1.  The native right36
    // is Eigen's column-major memory order, matching SMatrix::as_slice().
    let native_right36 = [
        0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0x00000000, 0x00000000, 0x00000000, 0x3f7e3f01,
        0xbd87ecbc, 0x3dc4f985, 0x00000000, 0x00000000, 0x00000000, 0xbd118207, 0xbf74a0e5,
        0xbe95cd73, 0x00000000, 0x00000000, 0x00000000, 0x3d8687dc, 0xbd228425, 0xbb8c8d8a,
        0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0xbbecbb09, 0xbc3f8e51, 0x3d8841e8, 0x3f7e3f01,
        0xbd87ecbc, 0x3dc4f985, 0x3b7e5e61, 0xbc360bff, 0x3d12b628, 0xbd118207, 0xbf74a0e5,
        0xbe95cd73,
    ];
    let drel_bits = |matrix: &SMatrix<f32, 6, 6>| {
        matrix
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    let candidate_bits = drel_bits(&candidate_drel);
    let generic_bits = drel_bits(&generic_drel);
    let candidate_native_exact = candidate_bits
        .iter()
        .zip(native_right36)
        .filter(|(actual, expected)| **actual == *expected)
        .count();
    let generic_native_exact = generic_bits
        .iter()
        .zip(native_right36)
        .filter(|(actual, expected)| **actual == *expected)
        .count();
    let format_bits = |bits: &[u32]| {
        bits.iter()
            .map(|bits| format!("{bits:08x}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    println!(
        "iter1=3/1 drel36_native={} drel36_candidate={} drel36_generic={} drel36_exact_candidate={}/36 drel36_exact_generic={}/36",
        format_bits(&native_right36),
        format_bits(&candidate_bits),
        format_bits(&generic_bits),
        candidate_native_exact,
        generic_native_exact,
    );
    assert_eq!(
        generic_native_exact, 24,
        "generic iter1 frame3/cam1 drel must match native right36 in 24/36 lanes"
    );
    assert_eq!(
        candidate_native_exact, 18,
        "candidate iter1 frame3/cam1 drel must match native right36 in 18/36 lanes"
    );
    let q = |pose: F32Pose| {
        [
            pose.rotation.i.to_bits(),
            pose.rotation.j.to_bits(),
            pose.rotation.k.to_bits(),
            pose.rotation.w.to_bits(),
        ]
    };
    let t = |pose: F32Pose| {
        [
            pose.translation.x.to_bits(),
            pose.translation.y.to_bits(),
            pose.translation.z.to_bits(),
        ]
    };
    println!(
        "iter1=3/1 candidate_q={:08x},{:08x},{:08x},{:08x} generic_q={:08x},{:08x},{:08x},{:08x} candidate_t={:08x},{:08x},{:08x} generic_t={:08x},{:08x},{:08x} drel_diffs=[{}]",
        q(candidate)[0],
        q(candidate)[1],
        q(candidate)[2],
        q(candidate)[3],
        q(generic)[0],
        q(generic)[1],
        q(generic)[2],
        q(generic)[3],
        t(candidate)[0],
        t(candidate)[1],
        t(candidate)[2],
        t(generic)[0],
        t(generic)[1],
        t(generic)[2],
        d_diffs.join(","),
    );
    println!(
        "iter1=3/1 raw_candidate={:08x},{:08x},{:08x},{:08x} raw_generic={:08x},{:08x},{:08x},{:08x} norm_candidate={:08x},{:08x},{:08x},{:08x} norm_generic={:08x},{:08x},{:08x},{:08x}",
        candidate_raw[0].to_bits(),
        candidate_raw[1].to_bits(),
        candidate_raw[2].to_bits(),
        candidate_raw[3].to_bits(),
        generic_raw[0].to_bits(),
        generic_raw[1].to_bits(),
        generic_raw[2].to_bits(),
        generic_raw[3].to_bits(),
        candidate_norm[0].to_bits(),
        candidate_norm[1].to_bits(),
        candidate_norm[2].to_bits(),
        candidate_norm[3].to_bits(),
        generic_norm[0].to_bits(),
        generic_norm[1].to_bits(),
        generic_norm[2].to_bits(),
        generic_norm[3].to_bits(),
    );
}

#[test]
fn m7im15_pose_lin_generic_drel_matches_tied_native_and_current_is_negative() {
    let packet: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7im15_native_iter1_track119_packet_inputs_20260827.json"
    ))
    .expect("packet-input fixture must be valid JSON");

    fn pose(packet: &serde_json::Value, field: &str) -> F32Pose {
        let pose = &packet[field];
        let q = pose["q_xyzw"]
            .as_array()
            .expect("pose quaternion must be an array")
            .iter()
            .map(|value| {
                u32::from_str_radix(value.as_str().expect("pose bits must be strings"), 16)
                    .expect("pose bits must be hexadecimal")
            })
            .collect::<Vec<_>>();
        let t = pose["t_xyz"]
            .as_array()
            .expect("pose translation must be an array")
            .iter()
            .map(|value| {
                u32::from_str_radix(value.as_str().expect("pose bits must be strings"), 16)
                    .expect("pose bits must be hexadecimal")
            })
            .collect::<Vec<_>>();
        assert_eq!(q.len(), 4);
        assert_eq!(t.len(), 3);
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        }
    }

    fn drel(packet_pose: F32Pose, host: F32Pose) -> SMatrix<f32, 6, 6> {
        eigen_adjoint_times_rotation_blocks_f32(
            packet_pose,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
            1.0,
        )
    }

    fn candidate_drel(packet_pose: F32Pose, host: F32Pose) -> SMatrix<f32, 6, 6> {
        m7_candidate_adjoint_times_rotation_blocks_f32(
            packet_pose,
            m7_candidate_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
            1.0,
        )
    }

    fn generic_drel(
        host: F32Pose,
        target: F32Pose,
        target_extrinsic: F32Pose,
    ) -> SMatrix<f32, 6, 6> {
        let target_camera_from_imu = target_extrinsic.inverse();
        let relative = sophus_relative_imu_f32(target, host);
        let target_camera_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                relative.rotation,
            ),
            translation: sophus_rotate_f32(target_camera_from_imu.rotation, relative.translation)
                + target_camera_from_imu.translation,
        };
        drel(target_camera_from_anchor_imu, host)
    }

    fn expected(packet: &serde_json::Value, field: &str) -> Vec<u32> {
        packet[field]
            .as_array()
            .expect("drel bits must be an array")
            .iter()
            .map(|value| {
                u32::from_str_radix(value.as_str().expect("drel bits must be strings"), 16)
                    .expect("drel bits must be hexadecimal")
            })
            .collect()
    }

    let host_lin = pose(&packet, "host_pose_lin");
    let target_lin = pose(&packet, "target_pose_lin");
    let host_current = pose(&packet, "host_pose_current");
    let target_current = pose(&packet, "target_pose_current");
    let target_extrinsic = pose(&packet, "target_extrinsic");
    let native_h = expected(&packet, "d_rel_d_h36");
    let native_t = expected(&packet, "d_rel_d_t36");
    assert_eq!(native_h.len(), 36);
    assert_eq!(native_t.len(), 36);

    let lin_h = generic_drel(host_lin, target_lin, target_extrinsic);
    let candidate_lin_h = {
        let target_camera_from_imu = target_extrinsic.inverse();
        let relative = sophus_relative_imu_f32(target_lin, host_lin);
        let target_camera_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                relative.rotation,
            ),
            translation: sophus_rotate_f32(target_camera_from_imu.rotation, relative.translation)
                + target_camera_from_imu.translation,
        };
        candidate_drel(target_camera_from_anchor_imu, host_lin)
    };
    let target_camera_from_imu = target_extrinsic.inverse();
    let lin_t = eigen_adjoint_times_rotation_blocks_f32(
        target_camera_from_imu,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(target_lin.rotation)),
        -1.0,
    );
    let candidate_lin_t = m7_candidate_adjoint_times_rotation_blocks_f32(
        target_camera_from_imu,
        m7_candidate_quaternion_matrix_f32(sophus_so3_inverse(target_lin.rotation)),
        -1.0,
    );
    let current_h = generic_drel(host_current, target_current, target_extrinsic);
    let current_t = eigen_adjoint_times_rotation_blocks_f32(
        target_camera_from_imu,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(target_current.rotation)),
        -1.0,
    );
    let candidate_current_t = m7_candidate_adjoint_times_rotation_blocks_f32(
        target_camera_from_imu,
        m7_candidate_quaternion_matrix_f32(sophus_so3_inverse(target_current.rotation)),
        -1.0,
    );

    let exact = |actual: &SMatrix<f32, 6, 6>, expected: &[u32]| {
        actual
            .as_slice()
            .iter()
            .zip(expected.iter())
            .filter(|(actual, expected)| actual.to_bits() == **expected)
            .count()
    };
    let lin_h_exact = exact(&lin_h, &native_h);
    let lin_t_exact = exact(&lin_t, &native_t);
    let candidate_lin_h_exact = exact(&candidate_lin_h, &native_h);
    let candidate_lin_t_exact = exact(&candidate_lin_t, &native_t);
    let current_h_exact = exact(&current_h, &native_h);
    let current_t_exact = exact(&current_t, &native_t);
    let candidate_current_t_exact = exact(&candidate_current_t, &native_t);
    eprintln!(
        "m7im15 poseLin generic drel exact h={lin_h_exact}/36 candidate_h={candidate_lin_h_exact}/36 t={lin_t_exact}/36 candidate_t={candidate_lin_t_exact}/36; current-input negative h={current_h_exact}/36 t={current_t_exact}/36 candidate_t={candidate_current_t_exact}/36"
    );
    assert_eq!(candidate_lin_h_exact, 36);
    assert_eq!(candidate_lin_t_exact, 36);
    assert_eq!(
        lin_h_exact, 36,
        "poseLin generic drel_h must match tied native"
    );
    assert_eq!(
        lin_t_exact, 36,
        "poseLin generic drel_t must match tied native"
    );
    assert!(
        current_h_exact < 36 || current_t_exact < 36,
        "current-input generic drel must remain a negative witness"
    );
}

#[test]
fn m7im15_frame6_iter4_track120_obs6_target_drel_matches_native_call8() {
    // Native `computeRelPose` call 8 is the target absolute-pose derivative
    // packet for frame 6/iter 4 track 120 observation 6.  Keep the
    // comparison at the raw 6x6 boundary: this distinguishes a d_rel
    // producer mismatch from the subsequent 2x6*6x6 Eigen assignment.
    fn pose(q_xyzw: [u32; 4], t_xyz: [u32; 3]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q_xyzw[3]),
                f32::from_bits(q_xyzw[0]),
                f32::from_bits(q_xyzw[1]),
                f32::from_bits(q_xyzw[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t_xyz[0]),
                f32::from_bits(t_xyz[1]),
                f32::from_bits(t_xyz[2]),
            ),
        }
    }

    let target_camera_from_imu = pose(
        [0x3bed3c0f, 0xbbf71cd4, 0xbf33a827, 0x3f365a1e],
        [0x3d8ddf2e, 0xbc810144, 0xbb71aa03],
    );
    let target_pose_fej = pose(
        [0xbd61da53, 0xbf524aba, 0xbbd8d37a, 0x3f114c04],
        [0xba89dd02, 0xbbe8ec4f, 0xbe135a1f],
    );
    let target_inverse = sophus_so3_inverse(target_pose_fej.rotation);
    let r_w_i_target_inv = eigen_quaternion_matrix_f32(target_inverse);
    let candidate_r_w_i_target_inv = m7_candidate_quaternion_matrix_f32(target_inverse);
    eprintln!(
        "target_inverse_q={:08x} {:08x} {:08x} {:08x} matrix={} candidate_matrix={}",
        target_inverse.i.to_bits(),
        target_inverse.j.to_bits(),
        target_inverse.k.to_bits(),
        target_inverse.w.to_bits(),
        r_w_i_target_inv
            .as_slice()
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join(" "),
        candidate_r_w_i_target_inv
            .as_slice()
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let actual =
        eigen_adjoint_times_rotation_blocks_f32(target_camera_from_imu, r_w_i_target_inv, -1.0);
    let candidate = m7_candidate_adjoint_times_rotation_blocks_f32(
        target_camera_from_imu,
        candidate_r_w_i_target_inv,
        -1.0,
    );
    let expected = [
        0xbde613b7, 0xbeb39f91, 0x3f6dff58, 0x00000000, 0x00000000, 0x00000000, 0xbf7e42b2,
        0x3d8bc630, 0xbdc10d9f, 0x00000000, 0x00000000, 0x80000000, 0x3cf8ddd0, 0x3f6f175d,
        0x3eb65413, 0x00000000, 0x00000000, 0x00000000, 0xbc8287e0, 0xbd830bee, 0xbcd59519,
        0xbde613b7, 0xbeb39f91, 0x3f6dff58, 0x3ae38e3c, 0x3c26fe37, 0xbc32cbb7, 0xbf7e42b2,
        0x3d8bc630, 0xbdc10d9f, 0xbb0dd14b, 0xbccb0173, 0x3d857b21, 0x3cf8ddd0, 0x3f6f175d,
        0x3eb65413,
    ];
    assert_f32_bits(actual.as_slice(), &expected);
    assert_f32_bits(candidate.as_slice(), &expected);
}

#[test]
fn m7im15_historical_packet_tmp_drel_matches_variant_fixture() {
    // This is a historical step-only call-15 packet. The helper under
    // test is a test-only reconstruction of that out-of-line boundary;
    // it is not the active anchored_visual_reprojection_factor_f32 path.
    fn bits(value: &serde_json::Value) -> u32 {
        u32::from_str_radix(
            value
                .as_str()
                .expect("f32 fixture bits must be hexadecimal strings"),
            16,
        )
        .expect("f32 fixture bits must be hexadecimal")
    }

    fn pose(packet: &serde_json::Value, field: &str) -> F32Pose {
        let value = &packet[field];
        let q = value["q_xyzw"]
            .as_array()
            .expect("pose quaternion must be an array");
        let t = value["t_xyz"]
            .as_array()
            .expect("pose translation must be an array");
        assert_eq!(q.len(), 4);
        assert_eq!(t.len(), 3);
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(bits(&q[3])),
                f32::from_bits(bits(&q[0])),
                f32::from_bits(bits(&q[1])),
                f32::from_bits(bits(&q[2])),
            )),
            translation: Vector3::new(
                f32::from_bits(bits(&t[0])),
                f32::from_bits(bits(&t[1])),
                f32::from_bits(bits(&t[2])),
            ),
        }
    }

    fn pose_bits(pose: F32Pose) -> [u32; 7] {
        [
            pose.rotation.i.to_bits(),
            pose.rotation.j.to_bits(),
            pose.rotation.k.to_bits(),
            pose.rotation.w.to_bits(),
            pose.translation.x.to_bits(),
            pose.translation.y.to_bits(),
            pose.translation.z.to_bits(),
        ]
    }

    fn matrix_bits(packet: &serde_json::Value, field: &str) -> Vec<u32> {
        packet[field]
            .as_array()
            .expect("matrix fixture must be an array")
            .iter()
            .map(bits)
            .collect()
    }

    let packet: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7im15_native_iter1_track119_packet_inputs_20260827.json"
    ))
    .expect("packet-input fixture must be valid JSON");
    assert_eq!(
        packet["schema"].as_str(),
        Some("basalt.relpose_packet_inputs.v1")
    );
    assert_eq!(packet["record"].as_str(), Some("relpose_packet_inputs"));
    assert_eq!(packet["iteration"].as_u64(), Some(1));
    assert_eq!(packet["trial"].as_u64(), Some(0));
    assert_eq!(packet["host_cam"].as_u64(), Some(0));
    assert_eq!(packet["target_cam"].as_u64(), Some(1));
    assert_eq!(packet["host_is_linearized"].as_bool(), Some(true));
    assert_eq!(packet["target_is_linearized"].as_bool(), Some(false));

    let host = pose(&packet, "host_pose_lin");
    let target = pose(&packet, "target_pose_lin");
    let target_camera_from_imu = pose(&packet, "target_extrinsic").inverse();
    let tmp = sophus_compute_relpose_tmp_out_of_line_f32(target_camera_from_imu, target, host);
    assert_eq!(
        pose_bits(tmp),
        pose_bits(pose(&packet, "tmp_for_jacobian")),
        "captured out-of-line variant relpose must reproduce the packet pose"
    );

    let d_rel_d_h = eigen_adjoint_times_rotation_blocks_f32(
        tmp,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
        1.0,
    );
    let d_rel_d_t = eigen_adjoint_times_rotation_blocks_f32(
        target_camera_from_imu,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(target.rotation)),
        -1.0,
    );
    let expected_h = matrix_bits(&packet, "d_rel_d_h36");
    let expected_t = matrix_bits(&packet, "d_rel_d_t36");
    assert_eq!(expected_h.len(), 36);
    assert_eq!(expected_t.len(), 36);
    assert_f32_bits(d_rel_d_h.as_slice(), &expected_h);
    assert_f32_bits(d_rel_d_t.as_slice(), &expected_t);

    // Keep the captured current-value chain as a negative witness: the
    // Jacobian boundary is tied to the linearized packet, not an arbitrary
    // current-pose recomputation.
    let current_host = pose(&packet, "host_pose_current");
    let current_target = pose(&packet, "target_pose_current");
    let current_tmp = sophus_compute_relpose_tmp_out_of_line_f32(
        target_camera_from_imu,
        current_target,
        current_host,
    );
    let current_h = eigen_adjoint_times_rotation_blocks_f32(
        current_tmp,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(current_host.rotation)),
        1.0,
    );
    let current_t = eigen_adjoint_times_rotation_blocks_f32(
        target_camera_from_imu,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(current_target.rotation)),
        -1.0,
    );
    let current_matches = current_h
        .as_slice()
        .iter()
        .zip(expected_h.iter())
        .all(|(actual, expected)| actual.to_bits() == *expected)
        && current_t
            .as_slice()
            .iter()
            .zip(expected_t.iter())
            .all(|(actual, expected)| actual.to_bits() == *expected);
    assert!(
        !current_matches,
        "current-value relpose must remain a negative witness"
    );
}

#[test]
fn m7im15_found_tmp_translation_reproduces_native_drel_cross() {
    // Standalone-search witness for the exact native cross block.  The
    // temporary rotation is the tied true-pose operand; only translation
    // is replaced by the ULP-search hit.  Keep this focused regression
    // test-only until the authoritative step-only producer is recovered.
    fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        }
    }

    let found_tmp = pose(
        [0x3c342b3c, 0xbc4f602d, 0xbf3355a4, 0x3f36a35f],
        [0xbd235394, 0xbd83bd75, 0xbc801294],
    );
    let host = pose([0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd], [0, 0, 0]);
    let expected = [
        0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0x00000000, 0x00000000, 0x00000000, 0x3f7e3f01,
        0xbd87ecbc, 0x3dc4f985, 0x00000000, 0x00000000, 0x00000000, 0xbd118207, 0xbf74a0e5,
        0xbe95cd73, 0x00000000, 0x00000000, 0x00000000, 0x3d8687db, 0xbd228425, 0xbb8c8d8c,
        0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0xbbecbb07, 0xbc3f8e51, 0x3d8841e7, 0x3f7e3f01,
        0xbd87ecbc, 0x3dc4f985, 0x3b7e5e58, 0xbc360bff, 0x3d12b627, 0xbd118207, 0xbf74a0e5,
        0xbe95cd73,
    ];
    let right = eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation));
    let actual = eigen_adjoint_times_rotation_blocks_f32(found_tmp, right, 1.0);
    assert_eq!(
        actual
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected,
        "ULP-search translation must reproduce native drel 36/36"
    );
}

#[test]
fn m7_clean_frame4_iter1_cam1_fej_tmp_and_drel_h_match_native() {
    fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        }
    }
    let host = pose([0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd], [0, 0, 0]);
    let _host_ext = pose(
        [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
        [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
    );
    let target = pose(
        [0xbd5e3af6, 0xbf4e66c7, 0xbbc319f3, 0x3f16cb91],
        [0x3b2b6242, 0xbb51bff3, 0xbd4c7b9e],
    );
    let target_ext = pose(
        [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
        [0xbc76fa76, 0x3d290319, 0x3b4f4832],
    );
    let tmp2 = target_ext.inverse();
    let tmp = sophus_compute_relpose_tmp_out_of_line_f32(tmp2, target, host);
    assert_pose_f32_bits(
        tmp,
        [0xbd235393, 0xbd83bd7a, 0xbc8012a4],
        [0x3c342b39, 0xbc4f6029, 0xbf3355a5, 0x3f36a35f],
    );
    let drel = eigen_adjoint_times_rotation_blocks_f32(
        tmp,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
        1.0,
    );
    assert_f32_bits(
        drel.as_slice(),
        &[
            0x3de3fe94, 0x3e9306e1, 0xbf738e5b, 0x00000000, 0x00000000, 0x00000000, 0x3f7e3f02,
            0xbd87ecd4, 0x3dc4f984, 0x00000000, 0x00000000, 0x00000000, 0xbd118236, 0xbf74a0e5,
            0xbe95cd73, 0x00000000, 0x00000000, 0x00000000, 0x3d8687e1, 0xbd228425, 0xbb8c8d7c,
            0x3de3fe94, 0x3e9306e1, 0xbf738e5b, 0xbbecbb19, 0xbc3f8e72, 0x3d8841ee, 0x3f7e3f02,
            0xbd87ecd4, 0x3dc4f984, 0x3b7e5e03, 0xbc360bf9, 0x3d12b625, 0xbd118236, 0xbf74a0e5,
            0xbe95cd73,
        ],
    );
}

#[test]
fn m7_clean_frame4_iter0_cam1_fej_tmp_and_drel_h_match_native() {
    fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        }
    }
    let host = pose(
        [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
        [0x00000000, 0x00000000, 0x00000000],
    );
    let target = pose(
        [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
        [0x3b8ebfa0, 0xba9428f4, 0xbc857642],
    );
    let target_ext = pose(
        [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
        [0xbc76fa76, 0x3d290319, 0x3b4f4832],
    );
    let target_camera_from_imu = target_ext.inverse();
    let target_inverse_rotation = sophus_so3_inverse(target.rotation);
    let relative_rotation = sophus_quat_product_f32(target_inverse_rotation, host.rotation);
    assert_f32_bits(
        &[
            relative_rotation.i,
            relative_rotation.j,
            relative_rotation.k,
            relative_rotation.w,
        ],
        &[0x3bc353bb, 0x3bc9740e, 0x3b240734, 0x3f7ffd65],
    );
    let tmp = sophus_compute_relpose_tmp_out_of_line_f32(target_camera_from_imu, target, host);
    assert_pose_f32_bits(
        tmp,
        [0xbd27a81a, 0xbd05555f, 0xbb8ddf30],
        [0x3c31fe46, 0xbc520562, 0xbf33585b, 0x3f36a0a7],
    );
    let drel = eigen_adjoint_times_rotation_blocks_f32(
        tmp,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
        1.0,
    );
    assert_f32_bits(
        drel.as_slice(),
        &[
            0x3de4276f, 0x3e92d17a, 0xbf7395cf, 0x00000000, 0x00000000, 0x00000000, 0x3f7e3e30,
            0xbd88188a, 0x3dc51f14, 0x00000000, 0x00000000, 0x00000000, 0xbd11f0e1, 0xbf74a889,
            0xbe9599d5, 0x00000000, 0x00000000, 0x00000000, 0x3d03f3e6, 0xbd21806f, 0xbc04e3d5,
            0x3de4276f, 0x3e92d17a, 0xbf7395cf, 0xbb6030cf, 0xb9bcd31b, 0x3d0f8f43, 0x3f7e3e30,
            0xbd88188a, 0x3dc51f14, 0x3bb0151e, 0xbc416c25, 0x3d1b7a6d, 0xbd11f0e1, 0xbf74a889,
            0xbe9599d5,
        ],
    );
}

#[test]
fn m7_clean_frame4_iter0_cam0_fej_tmp_and_drel_h_match_native() {
    fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        }
    }
    let host = pose(
        [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
        [0x00000000, 0x00000000, 0x00000000],
    );
    let target = pose(
        [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
        [0x3b8ebfa0, 0xba9428f4, 0xbc857642],
    );
    let target_ext = pose(
        [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
        [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
    );
    let target_camera_from_imu = target_ext.inverse();
    let target_inverse_rotation = sophus_so3_inverse(target.rotation);
    let relative_rotation = sophus_quat_product_f32(target_inverse_rotation, host.rotation);
    let relative_translation = sophus_rotate_difference_f32(
        target_inverse_rotation,
        host.translation,
        target.translation,
    );
    assert_f32_bits(
        &[
            relative_rotation.i,
            relative_rotation.j,
            relative_rotation.k,
            relative_rotation.w,
        ],
        &[0x3bc353bb, 0x3bc9740e, 0x3b240734, 0x3f7ffd65],
    );
    assert_f32_bits(
        relative_translation.as_slice(),
        &[0x3c8a503f, 0xb9376126, 0xba473200],
    );
    let relative_product =
        m7_relpose_packet_product_raw_for_test(target_camera_from_imu.rotation, relative_rotation);
    eprintln!(
        "m7 cam0 raw_product={:08x} {:08x} {:08x} {:08x}",
        relative_product[0].to_bits(),
        relative_product[1].to_bits(),
        relative_product[2].to_bits(),
        relative_product[3].to_bits(),
    );
    let rotated_translation = sophus_rotate_relpose_out_of_line_f32(
        target_camera_from_imu.rotation,
        relative_translation,
    );
    // Keep the packet-stage value visible while auditing the native
    // target-frame-3/cam0 capture; the final add below is the value
    // materialized at the computeRelPose return boundary.
    eprintln!(
        "m7 cam0 rel_t={:08x} {:08x} {:08x} rotated_t={:08x} {:08x} {:08x}",
        relative_translation.x.to_bits(),
        relative_translation.y.to_bits(),
        relative_translation.z.to_bits(),
        rotated_translation.x.to_bits(),
        rotated_translation.y.to_bits(),
        rotated_translation.z.to_bits(),
    );
    let tmp = sophus_compute_relpose_tmp_out_of_line_f32(target_camera_from_imu, target, host);
    assert_pose_f32_bits(
        tmp,
        [0x3d8e0f98, 0xbd05a9be, 0xbb91861b],
        [0x3c814783, 0xbbf146bb, 0xbf332b9e, 0x3f36cb95],
    );
    let drel = eigen_adjoint_times_rotation_blocks_f32(
        tmp,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
        1.0,
    );
    assert_f32_bits(
        drel.as_slice(),
        &[
            0x3de12055, 0x3e9a0cd6, 0xbf7282a1, 0x00000000, 0x00000000, 0x00000000, 0x3f7e4d08,
            0xbd869b07, 0x3dc151a6, 0x00000000, 0x00000000, 0x00000000, 0xbd0ab1a0, 0xbf738e9a,
            0xbe9cba11, 0x00000000, 0x00000000, 0x00000000, 0x3d0417c9, 0x3d859348, 0x3cc85bc6,
            0x3de12055, 0x3e9a0cd6, 0xbf7282a1, 0xbb5d0047, 0xbc338e7d, 0x3ce4342e, 0x3f7e4d08,
            0xbd869b07, 0x3dc151a6, 0x3bbcdefd, 0x3caf2cdf, 0xbd896b41, 0xbd0ab1a0, 0xbf738e9a,
            0xbe9cba11,
        ],
    );
}

#[cfg(test)]
fn m7_relpose_packet_product_raw_for_test(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> [f32; 4] {
    let a = first.quaternion();
    let b = second.quaternion();
    let a24 = [a.i, a.j, a.k, a.i];
    let a49 = [a.j, a.k, a.i, a.j];
    let a92 = [a.k, a.i, a.j, a.k];
    let b3f = [b.w, b.w, b.w, b.i];
    let b52 = [b.k, b.i, b.j, b.j];
    let b89 = [b.j, b.k, b.i, b.k];
    let b_times_aw = [b.i * a.w, b.j * a.w, b.k * a.w, b.w * a.w];
    let mut first_negative = [0.0_f32; 4];
    let mut first_positive = [0.0_f32; 4];
    for lane in 0..4 {
        first_negative[lane] = (-b3f[lane]).mul_add(a24[lane], b_times_aw[lane]);
        first_positive[lane] = b3f[lane].mul_add(a24[lane], b_times_aw[lane]);
    }
    first_positive[3] = first_negative[3];
    let mut second_negative = [0.0_f32; 4];
    let mut second_positive = [0.0_f32; 4];
    for lane in 0..4 {
        second_negative[lane] = (-b52[lane]).mul_add(a49[lane], first_positive[lane]);
        second_positive[lane] = b52[lane].mul_add(a49[lane], first_positive[lane]);
    }
    second_positive[3] = second_negative[3];
    let mut product = [0.0_f32; 4];
    for lane in 0..4 {
        product[lane] = (-a92[lane]).mul_add(b89[lane], second_positive[lane]);
    }
    product
}

#[cfg(test)]
fn m7_generic_packet_product_raw_for_test(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> [f32; 4] {
    let a = first.quaternion();
    let b = second.quaternion();
    [
        (-a.k).mul_add(b.j, a.j.mul_add(b.k, a.w.mul_add(b.i, a.i * b.w))),
        (-a.i).mul_add(b.k, a.k.mul_add(b.i, a.w.mul_add(b.j, a.j * b.w))),
        (-a.j).mul_add(b.i, a.i.mul_add(b.j, a.w.mul_add(b.k, a.k * b.w))),
        (-a.k).mul_add(b.k, (-a.j).mul_add(b.j, b.w.mul_add(a.w, -(b.i * a.i)))),
    ]
}

#[cfg(test)]
fn m7_normalize_packet_product_for_test(raw: [f32; 4]) -> [f32; 4] {
    let x2 = raw[0] * raw[0];
    let z2 = raw[2] * raw[2];
    let y2 = raw[1] * raw[1];
    let w2 = raw[3] * raw[3];
    let norm = (x2 + z2 + (y2 + w2)).sqrt();
    [raw[0] / norm, raw[1] / norm, raw[2] / norm, raw[3] / norm]
}

#[test]
fn m7_relative_pose_chain_matches_pinned_lanes() {
    let (_, anchor_pose, anchor_extrinsic, target_pose, target_extrinsic, _, _) =
        m7_fixed_factor_inputs();
    let anchor_pose = F32Pose::from_se3(&anchor_pose);
    let target_pose = F32Pose::from_se3(&target_pose);
    let anchor_extrinsic = F32Pose::from_se3(&anchor_extrinsic);
    let target_extrinsic = F32Pose::from_se3(&target_extrinsic);
    let target_camera_from_imu = target_extrinsic.inverse();
    let target_imu_from_anchor_imu = target_pose.inverse().compose(anchor_pose);
    let target_camera_from_anchor_imu = F32Pose {
        rotation: sophus_quat_product_camera_prefix_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.translation,
        ) + target_camera_from_imu.translation,
    };
    let target_camera_from_anchor_camera = F32Pose {
        rotation: sophus_quat_product_camera_suffix_f32(
            target_camera_from_anchor_imu.rotation,
            anchor_extrinsic.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_anchor_imu.rotation,
            anchor_extrinsic.translation,
        ) + target_camera_from_anchor_imu.translation,
    };

    assert_pose_f32_bits(
        target_camera_from_imu,
        [0xbd27e3b1, 0xbc8044de, 0xbb7a8e6e],
        [0x3b19188b, 0xbc55012e, 0xbf33d4ed, 0x3f362af6],
    );
    assert_pose_f32_bits(
        target_imu_from_anchor_imu,
        [0x3b06d6cc, 0xb8746f41, 0xb9657ee0],
        [0x3b322ebb, 0xbab5f931, 0x3a19f84f, 0x3f7fffaf],
    );
    assert_pose_f32_bits(
        target_camera_from_anchor_imu,
        [0xbd28004c, 0xbc912752, 0xbb837666],
        [0x3b577913, 0xbc82408f, 0xbf33b736, 0x3f36442d],
    );
    assert_pose_f32_bits(
        target_camera_from_anchor_camera,
        [0xbde1e255, 0xbaf1ec60, 0xba884368],
        [0xbc0e2957, 0xbb507f02, 0xba002d40, 0x3f7ffd31],
    );
}

#[test]
fn m7dx_clean_track1_relative_pose_jacobians_are_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        factor: String,
        host_pose_q_xyzw: [String; 4],
        host_pose_t_xyz: [String; 3],
        target_pose_q_xyzw: [String; 4],
        target_pose_t_xyz: [String; 3],
        host_extrinsic_q_xyzw: [String; 4],
        host_extrinsic_t_xyz: [String; 3],
        target_extrinsic_q_xyzw: [String; 4],
        target_extrinsic_t_xyz: [String; 3],
        d_rel_d_h_bits_column_major: Vec<String>,
        d_rel_d_t_bits_column_major: Vec<String>,
    }

    fn bits(value: &str) -> u32 {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    }

    fn pose(q_xyzw: &[String; 4], t_xyz: &[String; 3]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(bits(&q_xyzw[3])),
                f32::from_bits(bits(&q_xyzw[0])),
                f32::from_bits(bits(&q_xyzw[1])),
                f32::from_bits(bits(&q_xyzw[2])),
            )),
            translation: Vector3::new(
                f32::from_bits(bits(&t_xyz[0])),
                f32::from_bits(bits(&t_xyz[1])),
                f32::from_bits(bits(&t_xyz[2])),
            ),
        }
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7dx_clean_track1_relpose_jacobian.json"
    ))
    .expect("m7dx clean relative-pose Jacobian fixture must parse");
    assert_eq!(fixture.d_rel_d_h_bits_column_major.len(), 36);
    assert_eq!(fixture.d_rel_d_t_bits_column_major.len(), 36);

    let host_pose = pose(&fixture.host_pose_q_xyzw, &fixture.host_pose_t_xyz);
    let target_pose = pose(&fixture.target_pose_q_xyzw, &fixture.target_pose_t_xyz);
    let host_extrinsic = pose(
        &fixture.host_extrinsic_q_xyzw,
        &fixture.host_extrinsic_t_xyz,
    );
    let target_extrinsic = pose(
        &fixture.target_extrinsic_q_xyzw,
        &fixture.target_extrinsic_t_xyz,
    );
    let target_camera_from_imu = target_extrinsic.inverse();
    let target_imu_from_anchor_imu = sophus_relative_imu_f32(target_pose, host_pose);
    let target_camera_from_anchor_imu = F32Pose {
        rotation: sophus_quat_product_camera_prefix_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.translation,
        ) + target_camera_from_imu.translation,
    };
    // Keep the complete clean factor chain in the fixture exercise, even
    // though the raw d_rel buffers stop before this host-extrinsic suffix.
    let _relative_camera = F32Pose {
        rotation: sophus_quat_product_camera_suffix_f32(
            target_camera_from_anchor_imu.rotation,
            host_extrinsic.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_anchor_imu.rotation,
            host_extrinsic.translation,
        ) + target_camera_from_anchor_imu.translation,
    };

    let r_w_i_anchor_inv = eigen_quaternion_matrix_f32(host_pose.rotation.inverse());
    let r_w_i_target_inv = eigen_quaternion_matrix_f32(target_pose.rotation.inverse());
    let d_rel_d_h = eigen_adjoint_times_rotation_blocks_f32(
        target_camera_from_anchor_imu,
        r_w_i_anchor_inv,
        1.0,
    );
    let d_rel_d_t =
        eigen_adjoint_times_rotation_blocks_f32(target_camera_from_imu, r_w_i_target_inv, -1.0);
    let expected_h = fixture
        .d_rel_d_h_bits_column_major
        .iter()
        .map(|value| bits(value))
        .collect::<Vec<_>>();
    let expected_t = fixture
        .d_rel_d_t_bits_column_major
        .iter()
        .map(|value| bits(value))
        .collect::<Vec<_>>();
    assert_eq!(
        expected_t
            .iter()
            .filter(|&&value| value == 0x8000_0000)
            .count(),
        1
    );
    assert_f32_bits(d_rel_d_h.as_slice(), &expected_h);
    assert_f32_bits(d_rel_d_t.as_slice(), &expected_t);
}

#[test]
fn m7ga_weighted_pose_product_matches_native_packet_tree() {
    // Retained native M7aq pose-boundary probe (track 1, frame 0 cam0 ->
    // frame 1 cam1).  The probe records the post-whitening relative
    // Jacobian and both post-product absolute pose blocks.  These words
    // are deliberately kept in Eigen column-major order, including the
    // signed zero in the target chain.
    fn matrix<const R: usize, const C: usize>(words: &[u32]) -> SMatrix<f32, R, C> {
        assert_eq!(words.len(), R * C);
        let values = words
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
        SMatrix::from_column_slice(&values)
    }

    let relative = matrix(&[
        0x4247c89d, 0xc129fef8, 0xc12a8201, 0x428cdaf2, 0x4236ccb5, 0x419a2414, 0xc2239d4b,
        0xc3b7255f, 0x43df6969, 0x42231f8c, 0x4314e606, 0xc3af8621,
    ]);
    let d_rel_d_h = matrix(&[
        0x3dda79ad, 0x3e8b3905, 0xbf74d5e7, 0x00000000, 0x00000000, 0x00000000, 0x3f7e5131,
        0xbd8df5f8, 0x3dba92eb, 0x00000000, 0x00000000, 0x00000000, 0xbd2a12ca, 0xbf75b6c8,
        0xbe8e17f1, 0x00000000, 0x00000000, 0x00000000, 0x3c93c295, 0xbd226d6d, 0xbc17c30c,
        0x3dda79ad, 0x3e8b3905, 0xbf74d5e7, 0xbaf80701, 0xb9828894, 0x3ca77d72, 0x3f7e5131,
        0xbd8df5f8, 0x3dba92eb, 0x3a8bd267, 0xbc37c50e, 0x3d1e3cc6, 0xbd2a12ca, 0xbf75b6c8,
        0xbe8e17f1,
    ]);
    let d_rel_d_t = matrix(&[
        0xbdda79af, 0xbe8b3907, 0x3f74d5e9, 0x00000000, 0x00000000, 0x00000000, 0xbf7e5131,
        0x3d8df601, 0xbdba92eb, 0x00000000, 0x00000000, 0x80000000, 0x3d2a12d7, 0x3f75b6ca,
        0x3e8e17f1, 0x00000000, 0x00000000, 0x00000000, 0xbc833104, 0x3d223cf7, 0x3c1b3e27,
        0xbdda79af, 0xbe8b3907, 0x3f74d5e9, 0x3addb39c, 0x388626ba, 0xbc96b372, 0xbf7e5131,
        0x3d8df601, 0xbdba92eb, 0xba312e45, 0x3c37c62b, 0xbd1e7b0f, 0x3d2a12d7, 0x3f75b6ca,
        0x3e8e17f1,
    ]);
    let sqrt_weight = f32::from_bits(0x3ff04066);
    let expected_anchor = [
        0xc29af308, 0xbf450fd7, 0x42cca9a4, 0xc1cd6fc5, 0xc107fd35, 0xc3081664, 0xc236f40c,
        0x440eed0a, 0xc2d6b96d, 0xc43ae580, 0xc45aed81, 0x4309d5bd,
    ];
    let expected_target = [
        0x429af30a, 0x3f450fa6, 0xc2cca9a4, 0x41cd6fca, 0x4107fd38, 0x43081665, 0x4237c9cc,
        0xc40eef86, 0x42d70bb8, 0x443ae8ef, 0x445aef88, 0xc309d848,
    ];

    let anchor = eigen_weighted_pose_jacobian_f32(relative, d_rel_d_h, sqrt_weight);
    let target = eigen_weighted_pose_jacobian_f32(relative, d_rel_d_t, sqrt_weight);
    assert_f32_bits(anchor.as_slice(), &expected_anchor);
    assert_f32_bits(target.as_slice(), &expected_target);
}

#[test]
fn m7fx_clean_same_timestamp_stereo_relative_pose_jacobians_are_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        factor: String,
        #[allow(dead_code)]
        provenance: String,
        host_pose_q_xyzw: [String; 4],
        host_pose_t_xyz: [String; 3],
        target_pose_q_xyzw: [String; 4],
        target_pose_t_xyz: [String; 3],
        host_extrinsic_q_xyzw: [String; 4],
        host_extrinsic_t_xyz: [String; 3],
        target_extrinsic_q_xyzw: [String; 4],
        target_extrinsic_t_xyz: [String; 3],
        d_rel_d_h_bits_column_major: Vec<String>,
        d_rel_d_t_bits_column_major: Vec<String>,
    }

    fn bits(value: &str) -> u32 {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    }

    fn pose(q_xyzw: &[String; 4], t_xyz: &[String; 3]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(bits(&q_xyzw[3])),
                f32::from_bits(bits(&q_xyzw[0])),
                f32::from_bits(bits(&q_xyzw[1])),
                f32::from_bits(bits(&q_xyzw[2])),
            )),
            translation: Vector3::new(
                f32::from_bits(bits(&t_xyz[0])),
                f32::from_bits(bits(&t_xyz[1])),
                f32::from_bits(bits(&t_xyz[2])),
            ),
        }
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7fx_same_timestamp_clean_relpose_jacobian.json"
    ))
    .expect("m7fx same-timestamp relative-pose Jacobian fixture must parse");
    assert_eq!(fixture.d_rel_d_h_bits_column_major.len(), 36);
    assert_eq!(fixture.d_rel_d_t_bits_column_major.len(), 36);

    let host_pose = pose(&fixture.host_pose_q_xyzw, &fixture.host_pose_t_xyz);
    let target_pose = pose(&fixture.target_pose_q_xyzw, &fixture.target_pose_t_xyz);
    let host_extrinsic = pose(
        &fixture.host_extrinsic_q_xyzw,
        &fixture.host_extrinsic_t_xyz,
    );
    let target_extrinsic = pose(
        &fixture.target_extrinsic_q_xyzw,
        &fixture.target_extrinsic_t_xyz,
    );
    let target_camera_from_imu = target_extrinsic.inverse();
    let target_imu_from_anchor_imu = sophus_relative_imu_f32(target_pose, host_pose);
    let target_camera_from_anchor_imu = F32Pose {
        rotation: sophus_quat_product_camera_prefix_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.translation,
        ) + target_camera_from_imu.translation,
    };
    // The clean relation is a stereo call: host and target share the
    // state timestamp but differ in camera id, so no identity shortcut
    // applies to either relative-pose Jacobian.
    let _relative_camera = F32Pose {
        rotation: sophus_quat_product_camera_suffix_f32(
            target_camera_from_anchor_imu.rotation,
            host_extrinsic.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_anchor_imu.rotation,
            host_extrinsic.translation,
        ) + target_camera_from_anchor_imu.translation,
    };

    let r_w_i_anchor_inv = eigen_quaternion_matrix_f32(host_pose.rotation.inverse());
    let r_w_i_target_inv = eigen_quaternion_matrix_f32(target_pose.rotation.inverse());
    let d_rel_d_h = eigen_adjoint_times_rotation_blocks_f32(
        target_camera_from_anchor_imu,
        r_w_i_anchor_inv,
        1.0,
    );
    let d_rel_d_t =
        eigen_adjoint_times_rotation_blocks_f32(target_camera_from_imu, r_w_i_target_inv, -1.0);
    let expected_h = fixture
        .d_rel_d_h_bits_column_major
        .iter()
        .map(|value| bits(value))
        .collect::<Vec<_>>();
    let expected_t = fixture
        .d_rel_d_t_bits_column_major
        .iter()
        .map(|value| bits(value))
        .collect::<Vec<_>>();
    assert_f32_bits(d_rel_d_h.as_slice(), &expected_h);
    assert_f32_bits(d_rel_d_t.as_slice(), &expected_t);
}

#[test]
fn m7gh_clean_all_relative_pose_jacobians_are_bitwise_exact() {
    // Replay all nine non-identity TimeCam relations from the clean M7fw
    // capture with the clean m7db f32 states and pinned EuRoC calibration
    // words. Compare the raw d_rel buffers before any residual/Jacobian
    // product, so a failure identifies the relative-pose boundary.
    fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        }
    }
    fn bits(value: &serde_json::Value) -> u32 {
        u32::from_str_radix(value.as_str().unwrap(), 16).unwrap()
    }
    fn assert_matrix_bits(actual: &SMatrix<f32, 6, 6>, expected: &serde_json::Value, label: &str) {
        let expected = expected["f32_bits_column_major"].as_array().unwrap();
        assert_eq!(expected.len(), 36);
        for (index, value) in expected.iter().enumerate() {
            let got = actual.as_slice()[index].to_bits();
            let want = bits(value);
            assert_eq!(got, want, "{label} lane {index}: {got:08x} != {want:08x}");
        }
    }
    fn assert_pose_bits(actual: F32Pose, expected: &serde_json::Value, label: &str) {
        let expected = expected["q_t"]["f32_bits"].as_array().unwrap();
        assert_eq!(expected.len(), 7);
        let actual = [
            actual.rotation.i.to_bits(),
            actual.rotation.j.to_bits(),
            actual.rotation.k.to_bits(),
            actual.rotation.w.to_bits(),
            actual.translation.x.to_bits(),
            actual.translation.y.to_bits(),
            actual.translation.z.to_bits(),
        ];
        for (index, value) in expected.iter().enumerate() {
            let want = bits(value);
            assert_eq!(
                actual[index], want,
                "{label} lane {index}: {:08x} != {want:08x}",
                actual[index]
            );
        }
    }

    let states = [
        pose(
            [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
            [0x00000000, 0x00000000, 0x00000000],
        ),
        pose(
            [0xbd5cdea6, 0xbf4d341f, 0xbb2aa78b, 0x3f186f67],
            [0x39c8bb60, 0xb8cebae8, 0xbb0527fc],
        ),
        pose(
            [0xbd5cf5f2, 0xbf4d97c0, 0xbbac4997, 0x3f17e7a6],
            [0x3af3e90a, 0xb9fd97e2, 0xbbfc3924],
        ),
        pose(
            [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
            [0x3b8ebfa0, 0xba9428f4, 0xbc857642],
        ),
        pose(
            [0xbd5f79e6, 0xbf4f45a2, 0xbbb53f68, 0x3f159719],
            [0x3c02c75a, 0xbb0c3bda, 0xbcdc2b7f],
        ),
    ];
    let extrinsics = [
        pose(
            [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
            [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
        ),
        pose(
            [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
            [0xbc76fa76, 0x3d290319, 0x3b4f4832],
        ),
    ];
    let clean: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7gh_clean_all_relpose_jacobians.json"
    ))
    .unwrap();
    assert_eq!(
        clean["schema"].as_str(),
        Some("visloc-rs.basalt.m7gh.clean-all-relpose-jacobians.v1")
    );
    assert_eq!(
        clean["source"]["upstream_commit"].as_str(),
        Some("0f3b2b52c807f70ff4e2973ce253c73329eea7bc")
    );
    assert_eq!(clean["relation_count"].as_u64(), Some(9));
    for target_frame in 0..5 {
        for target_cam in 0..2 {
            if target_frame == 0 && target_cam == 0 {
                continue;
            }
            let record = clean["records"]
                .as_array()
                .unwrap()
                .iter()
                .find(|record| {
                    record["caller"]["host_frame_id"] == 0
                        && record["caller"]["host_cam"] == 0
                        && record["caller"]["target_frame_id"] == target_frame
                        && record["caller"]["target_cam"] == target_cam
                })
                .unwrap_or_else(|| panic!("missing relation {target_frame}/{target_cam}"));
            let target_camera_from_imu = extrinsics[target_cam].inverse();
            let target_inverse_rotation = states[target_frame].rotation.inverse();
            let target_imu_from_anchor_imu = F32Pose {
                rotation: sophus_quat_product_f32(target_inverse_rotation, states[0].rotation),
                translation: sophus_rotate_difference_f32(
                    target_inverse_rotation,
                    states[0].translation,
                    states[target_frame].translation,
                ),
            };
            let target_camera_from_anchor_imu = F32Pose {
                rotation: sophus_quat_product_camera_prefix_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_anchor_imu.rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_anchor_imu.translation,
                ) + target_camera_from_imu.translation,
            };
            let relative_camera = F32Pose {
                rotation: sophus_quat_product_camera_suffix_f32(
                    target_camera_from_anchor_imu.rotation,
                    extrinsics[0].rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_anchor_imu.rotation,
                    extrinsics[0].translation,
                ) + target_camera_from_anchor_imu.translation,
            };
            assert_pose_bits(
                relative_camera,
                record,
                &format!("target={target_frame}/{target_cam} q_t"),
            );
            let anchor_rotation = eigen_quaternion_matrix_f32(states[0].rotation.inverse());
            let target_rotation =
                eigen_quaternion_matrix_f32(states[target_frame].rotation.inverse());
            let actual_anchor = eigen_adjoint_times_rotation_blocks_f32(
                target_camera_from_anchor_imu,
                anchor_rotation,
                1.0,
            );
            let actual_target = eigen_adjoint_times_rotation_blocks_f32(
                target_camera_from_imu,
                target_rotation,
                -1.0,
            );
            assert_matrix_bits(
                &actual_anchor,
                &record["d_rel_d_h"],
                &format!("target={target_frame}/{target_cam} anchor"),
            );
            assert_matrix_bits(
                &actual_target,
                &record["d_rel_d_t"],
                &format!("target={target_frame}/{target_cam} target"),
            );
        }
    }
}

#[test]
fn m7gu_clean_frame4_point_projection_and_relative_jacobians_are_bitwise_exact() {
    // Keep this fixture bounded to the eleven frame-4 ordinals which still
    // differ after the absolute-pose chain was made exact.  The native
    // capture is the raw `linearizePoint` boundary: target_point4,
    // projection, residual, and d_res_d_xi are checked independently from
    // the caller-owned landmark/absolute-pose buffers.  The selected
    // records live under tests/fixtures so this contract does not depend
    // on a developer-local raw capture.
    let clean: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7gu_clean_frame4_visual.json"
    ))
    .unwrap();
    assert_eq!(
        clean["schema"].as_str(),
        Some("visloc-rs.basalt.m7gu.clean-frame4-visual.v1")
    );
    assert_eq!(
        clean["source"]["upstream_commit"].as_str(),
        Some("0f3b2b52c807f70ff4e2973ce253c73329eea7bc")
    );
    let ordinals = [75_usize, 79, 162, 257, 267, 283, 306, 478, 494, 515, 579];
    assert_eq!(clean["records"].as_array().unwrap().len(), ordinals.len());
    let bits =
        |value: &serde_json::Value| u32::from_str_radix(value.as_str().unwrap(), 16).unwrap();
    let lane_bits = |record: &serde_json::Value, key: &str| {
        record[key]
            .as_array()
            .unwrap()
            .iter()
            .map(bits)
            .collect::<Vec<_>>()
    };
    for ordinal in ordinals {
        let record = clean["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|record| record["ordinal"].as_u64() == Some(ordinal as u64))
            .unwrap_or_else(|| panic!("missing bounded visual ordinal {ordinal}"));
        let transform_bits = record["T_t_h_f32_bits"].as_array().unwrap();
        let transform = SMatrix::<f32, 4, 4>::from_column_slice(
            &transform_bits
                .iter()
                .map(bits)
                .map(f32::from_bits)
                .collect::<Vec<_>>(),
        );
        let direction = record["direction2_f32_bits"].as_array().unwrap();
        let u = f32::from_bits(bits(&direction[0]));
        let v = f32::from_bits(bits(&direction[1]));
        let r2 = u * u + v * v;
        let scale = 2.0_f32 / (1.0_f32 + r2);
        let bearing = Vector3::new(u * scale, v * scale, scale - 1.0_f32);
        let rho = f32::from_bits(bits(&record["inv_dist_f32_bits"][0]));
        let point4 = eigen_homogeneous_point_product_f32(transform, bearing, rho);
        let point = point4.fixed_rows::<3>(0).into_owned();
        // The bounded ordinals span both target cameras.  M7ef stores the
        // native camera pointer but not its compact id in each record;
        // retain the keyed target-cam mapping from the clean relation
        // audit here so projection/Jxi are evaluated against the right
        // calibration rather than silently using cam1 for cam0 rows.
        let target_cam = match ordinal {
            79 | 257 | 306 | 494 => 1,
            _ => 0,
        };
        let camera = if target_cam == 0 {
            DoubleSphereCamera::new(
                349.7560023050409,
                348.72454229977037,
                365.89440762590149,
                249.32995565708704,
                -0.2409573942178872,
                0.566996899163044,
                752,
                480,
            )
            .unwrap()
        } else {
            m7_fixed_camera()
        };
        let (projection, projection_jacobian) =
            project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
        let pixel = record["pixel2_f32_bits"].as_array().unwrap();
        let observation = Vector2::new(
            f32::from_bits(bits(&pixel[0])),
            f32::from_bits(bits(&pixel[1])),
        );
        let raw = projection - observation;
        let mut point_wrt_relative_pose = SMatrix::<f32, 3, 6>::zeros();
        point_wrt_relative_pose
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(SMatrix::<f32, 3, 3>::identity() * rho));
        point_wrt_relative_pose
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-skew3_f32(point)));
        let d_xi = eigen_relative_pose_jacobian_f32(projection_jacobian, point_wrt_relative_pose);
        for (name, actual, expected) in [
            (
                "point4",
                point4.as_slice().to_vec(),
                lane_bits(record, "target_point4_f32_bits"),
            ),
            (
                "projection",
                projection.as_slice().to_vec(),
                lane_bits(record, "projection2_f32_bits"),
            ),
            (
                "raw",
                raw.as_slice().to_vec(),
                lane_bits(record, "raw_residual2_f32_bits"),
            ),
            (
                "d_res_d_xi",
                d_xi.as_slice().to_vec(),
                lane_bits(record, "d_res_d_xi12_f32_bits"),
            ),
        ] {
            assert_eq!(
                actual.len(),
                expected.len(),
                "ordinal {ordinal} {name} length"
            );
            for (lane, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want,
                    "ordinal {ordinal} {name} lane {lane}: {:08x} != {want:08x}",
                    got.to_bits()
                );
            }
        }
    }
}

#[test]
fn m7_homogeneous_relative_point_matches_pinned_lanes() {
    let (rotation, translation, bearing, inverse_distance) = m7_fixed_relative_point();
    let point = sophus_homogeneous_point_f32(rotation, translation, bearing, inverse_distance);
    assert_f32_bits(point.as_slice(), &[0xbf2fc73d, 0xbe94aaaa, 0x3f2ec6b2]);
}

#[test]
fn m7_double_sphere_boundary_matches_pinned_lanes() {
    let (_, _, _, point_inverse_distance) = m7_fixed_relative_point();
    let point = Vector3::new(
        f32::from_bits(0xbf2fc73d),
        f32::from_bits(0xbe94aaaa),
        f32::from_bits(0x3f2ec6b2),
    );
    let (predicted, jacobian) =
        project_double_sphere_with_jacobian_f32(&m7_fixed_camera(), point).unwrap();
    assert_f32_bits(predicted.as_slice(), &[0x41da6d22, 0x42d70d2e]);
    assert_f32_bits(
        jacobian.as_slice(),
        &[
            0x43aa6a6b, 0xc29101bc, 0xc2917182, 0x43f04ca7, 0x439beda6, 0x43037b7f,
        ],
    );
    assert_eq!(point_inverse_distance.to_bits(), 0x3e160ef3);
    let raw = predicted - Vector2::new(27.31320571899414_f32, 106.39038848876953_f32);
    assert_f32_bits(raw.as_slice(), &[0xbc228000, 0x3f915340]);
}

#[test]
fn m7_anchored_factor_matches_pinned_projection_and_raw() {
    let (
        camera,
        anchor_pose,
        anchor_extrinsic,
        target_pose,
        target_extrinsic,
        landmark,
        observation,
    ) = m7_fixed_factor_inputs();
    let factor = anchored_visual_reprojection_factor_f32(
        &camera,
        &anchor_pose,
        &anchor_extrinsic,
        &target_pose,
        &target_extrinsic,
        &landmark,
        observation,
        false,
        FactorConfig::default(),
    )
    .unwrap();
    let projection = factor.projection.map(|value| value as f32);
    let raw = factor.raw_residual.map(|value| value as f32);
    assert_f32_bits(projection.as_slice(), &[0x41da6d17, 0x42d70d32]);
    assert_f32_bits(raw.as_slice(), &[0xbc22d800, 0x3f915440]);
}

#[test]
fn m7ec_clean_track2_stereo_factor_is_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        factor: String,
        direction_xy_f32_bits: [String; 2],
        inverse_distance_f32_bits: String,
        observation_f32_bits: [String; 2],
        q_xyzw_f32_bits: [String; 4],
        t_xyz_f32_bits: [String; 3],
        target_point_xyz_f32_bits: [String; 3],
        projection_uv_f32_bits: [String; 2],
        raw_residual_uv_f32_bits: [String; 2],
        landmark_jacobian_column_major_f32_bits: [String; 6],
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7ec_clean_track2_stereo_factor.json"
    ))
    .expect("m7ec clean track-2 stereo fixture must parse");
    let bits = |value: &str| {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    };
    let (camera, anchor_pose, anchor_extrinsic, _target_pose, target_extrinsic, _, _) =
        m7_fixed_factor_inputs();
    let landmark = InverseDistanceLandmark {
        anchor_pose: 0,
        anchor_camera_id: 0,
        direction: StereographicDirection {
            xy: Point2::new(
                f32::from_bits(bits(&fixture.direction_xy_f32_bits[0])) as f64,
                f32::from_bits(bits(&fixture.direction_xy_f32_bits[1])) as f64,
            ),
        },
        inverse_distance: f32::from_bits(bits(&fixture.inverse_distance_f32_bits)) as f64,
    };
    let anchor_pose_f32 = F32Pose::from_se3(&anchor_pose);
    let target_pose_f32 = F32Pose::from_se3(&anchor_pose);
    let anchor_extrinsic_f32 = F32Pose::from_se3(&anchor_extrinsic);
    let target_extrinsic_f32 = F32Pose::from_se3(&target_extrinsic);
    let target_camera_from_imu = target_extrinsic_f32.inverse();
    let target_imu_from_anchor_imu = sophus_relative_imu_f32(target_pose_f32, anchor_pose_f32);
    let target_camera_from_anchor_imu = F32Pose {
        rotation: sophus_quat_product_camera_prefix_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.translation,
        ) + target_camera_from_imu.translation,
    };
    let relative_camera = F32Pose {
        rotation: sophus_quat_product_camera_suffix_f32(
            target_camera_from_anchor_imu.rotation,
            anchor_extrinsic_f32.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_anchor_imu.rotation,
            anchor_extrinsic_f32.translation,
        ) + target_camera_from_anchor_imu.translation,
    };
    assert_pose_f32_bits(
        relative_camera,
        std::array::from_fn(|index| bits(&fixture.t_xyz_f32_bits[index])),
        std::array::from_fn(|index| bits(&fixture.q_xyzw_f32_bits[index])),
    );
    let point_target = sophus_homogeneous_point_normalized_rotation_f32(
        relative_camera.rotation,
        relative_camera.translation,
        landmark.direction.bearing_f32(),
        landmark.inverse_distance as f32,
    );
    let expected_point = [
        bits(&fixture.target_point_xyz_f32_bits[0]),
        bits(&fixture.target_point_xyz_f32_bits[1]),
        bits(&fixture.target_point_xyz_f32_bits[2]),
    ];
    assert_f32_bits(point_target.as_slice(), &expected_point);
    let observation = Point2::new(
        f32::from_bits(bits(&fixture.observation_f32_bits[0])) as f64,
        f32::from_bits(bits(&fixture.observation_f32_bits[1])) as f64,
    );
    let factor = anchored_visual_reprojection_factor_f32_with_time_cam(
        &camera,
        &anchor_pose,
        &anchor_extrinsic,
        &anchor_pose,
        &target_extrinsic,
        &landmark,
        observation,
        true,
        false,
        FactorConfig::default(),
    )
    .unwrap();
    let projection = factor.projection.map(|value| value as f32);
    let raw = factor.raw_residual.map(|value| value as f32);
    let jl = factor
        .landmark_jacobian
        .map(|value| (value as f32) / (factor.sqrt_weight as f32));
    let expected_projection = [
        bits(&fixture.projection_uv_f32_bits[0]),
        bits(&fixture.projection_uv_f32_bits[1]),
    ];
    let expected_raw = [
        bits(&fixture.raw_residual_uv_f32_bits[0]),
        bits(&fixture.raw_residual_uv_f32_bits[1]),
    ];
    let expected_jl = [
        bits(&fixture.landmark_jacobian_column_major_f32_bits[0]),
        bits(&fixture.landmark_jacobian_column_major_f32_bits[1]),
        bits(&fixture.landmark_jacobian_column_major_f32_bits[2]),
        bits(&fixture.landmark_jacobian_column_major_f32_bits[3]),
        bits(&fixture.landmark_jacobian_column_major_f32_bits[4]),
        bits(&fixture.landmark_jacobian_column_major_f32_bits[5]),
    ];
    assert_f32_bits(projection.as_slice(), &expected_projection);
    assert_f32_bits(raw.as_slice(), &expected_raw);
    assert_f32_bits(jl.as_slice(), &expected_jl);
}

#[test]
fn m7et_ordinal0_jp_packet4_is_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        source: String,
        ordinal: u32,
        #[allow(dead_code)]
        camera: String,
        camera_params_f32_bits: [String; 6],
        #[allow(dead_code)]
        observation_f32_bits: [String; 2],
        direction_xy_f32_bits: [String; 2],
        inverse_distance_f32_bits: String,
        target_point_xyz_f32_bits: [String; 3],
        projection_uv_f32_bits: [String; 2],
        camera_jacobian_row_major_f32_bits: [String; 8],
        homogeneous_point_jacobian_column_major_f32_bits: [String; 12],
        raw_landmark_jacobian_column_major_f32_bits: [String; 6],
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7et_ordinal0_jp_packet4.json"
    ))
    .expect("m7et ordinal-0 packet-4 fixture must parse");
    assert_eq!(fixture.ordinal, 0);
    let bits = |value: &str| {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    };
    let f32_value = |value: &str| f32::from_bits(bits(value));
    let camera = DoubleSphereCamera::new(
        f32_value(&fixture.camera_params_f32_bits[0]) as f64,
        f32_value(&fixture.camera_params_f32_bits[1]) as f64,
        f32_value(&fixture.camera_params_f32_bits[2]) as f64,
        f32_value(&fixture.camera_params_f32_bits[3]) as f64,
        f32_value(&fixture.camera_params_f32_bits[4]) as f64,
        f32_value(&fixture.camera_params_f32_bits[5]) as f64,
        752,
        480,
    )
    .unwrap();
    let direction = StereographicDirection {
        xy: Point2::new(
            f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
            f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
        ),
    };
    // Ordinal 0 uses an identity target transform, so the active
    // homogeneous Jpp block is the stereographic unprojection Jup
    // itself.  Keep this direct assertion at the first mismatch boundary
    // instead of only checking the downstream packet product.
    let expected_bearing_jacobian = [
        bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[0]),
        bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[1]),
        bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[2]),
        bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[4]),
        bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[5]),
        bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[6]),
    ];
    assert_f32_bits(
        direction.bearing_jacobian_f32().as_slice(),
        &expected_bearing_jacobian,
    );
    let inverse_distance = f32_value(&fixture.inverse_distance_f32_bits);
    let point = sophus_homogeneous_point_normalized_rotation_f32(
        UnitQuaternion::identity(),
        Vector3::zeros(),
        direction.bearing_f32(),
        inverse_distance,
    );
    let expected_point: [u32; 3] =
        std::array::from_fn(|index| bits(&fixture.target_point_xyz_f32_bits[index]));
    assert_f32_bits(point.as_slice(), &expected_point);

    let (projection, projection_jacobian) =
        project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
    let expected_projection: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
    assert_f32_bits(projection.as_slice(), &expected_projection);

    let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
    camera_jacobian
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&projection_jacobian);
    let expected_camera: [u32; 8] =
        std::array::from_fn(|index| bits(&fixture.camera_jacobian_row_major_f32_bits[index]));
    for row in 0..2 {
        for column in 0..4 {
            let index = row * 4 + column;
            assert_eq!(
                camera_jacobian[(row, column)].to_bits(),
                expected_camera[index],
                "camera J row-major lane {index}: got {:08x}, expected {:08x}",
                camera_jacobian[(row, column)].to_bits(),
                expected_camera[index]
            );
        }
    }

    let expected_point_jacobian: [u32; 12] = std::array::from_fn(|index| {
        bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[index])
    });
    let point_wrt_landmark =
        SMatrix::<f32, 4, 3>::from_column_slice(&expected_point_jacobian.map(f32::from_bits));
    assert_f32_bits(point_wrt_landmark.as_slice(), &expected_point_jacobian);

    let raw_landmark_jacobian = eigen_landmark_jacobian_f32(camera_jacobian, point_wrt_landmark);
    let expected_raw: [u32; 6] = std::array::from_fn(|index| {
        bits(&fixture.raw_landmark_jacobian_column_major_f32_bits[index])
    });
    assert_f32_bits(raw_landmark_jacobian.as_slice(), &expected_raw);
}

#[test]
fn m7ez_ordinal3_camera_jacobian_is_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        source: String,
        ordinal: u32,
        camera_params_f32_bits: [String; 6],
        target_point_xyz_f32_bits: [String; 3],
        projection_uv_f32_bits: [String; 2],
        camera_jacobian_column_major_f32_bits: [String; 6],
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7ez_ordinal3_camera_jacobian.json"
    ))
    .expect("M7ez ordinal-3 camera-J fixture must parse");
    assert_eq!(fixture.ordinal, 3);
    let bits = |value: &str| {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    };
    let f32_value = |value: &str| f32::from_bits(bits(value));
    let camera = DoubleSphereCamera::new(
        f32_value(&fixture.camera_params_f32_bits[0]) as f64,
        f32_value(&fixture.camera_params_f32_bits[1]) as f64,
        f32_value(&fixture.camera_params_f32_bits[2]) as f64,
        f32_value(&fixture.camera_params_f32_bits[3]) as f64,
        f32_value(&fixture.camera_params_f32_bits[4]) as f64,
        f32_value(&fixture.camera_params_f32_bits[5]) as f64,
        752,
        480,
    )
    .unwrap();
    let point = Vector3::new(
        f32_value(&fixture.target_point_xyz_f32_bits[0]),
        f32_value(&fixture.target_point_xyz_f32_bits[1]),
        f32_value(&fixture.target_point_xyz_f32_bits[2]),
    );
    let (projection, jacobian) = project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
    let expected_projection: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
    let expected_jacobian: [u32; 6] =
        std::array::from_fn(|index| bits(&fixture.camera_jacobian_column_major_f32_bits[index]));
    assert_f32_bits(projection.as_slice(), &expected_projection);
    assert_f32_bits(jacobian.as_slice(), &expected_jacobian);
}

#[test]
fn m7ez_ordinal38_projection_is_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        source: String,
        ordinal: u32,
        camera_params_f32_bits: [String; 6],
        target_point_xyz_f32_bits: [String; 3],
        projection_uv_f32_bits: [String; 2],
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7ez_ordinal38_projection.json"
    ))
    .expect("M7ez ordinal-38 projection fixture must parse");
    assert_eq!(fixture.ordinal, 38);
    let bits = |value: &str| {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    };
    let f32_value = |value: &str| f32::from_bits(bits(value));
    let camera = DoubleSphereCamera::new(
        f32_value(&fixture.camera_params_f32_bits[0]) as f64,
        f32_value(&fixture.camera_params_f32_bits[1]) as f64,
        f32_value(&fixture.camera_params_f32_bits[2]) as f64,
        f32_value(&fixture.camera_params_f32_bits[3]) as f64,
        f32_value(&fixture.camera_params_f32_bits[4]) as f64,
        f32_value(&fixture.camera_params_f32_bits[5]) as f64,
        752,
        480,
    )
    .unwrap();
    let point = Vector3::new(
        f32_value(&fixture.target_point_xyz_f32_bits[0]),
        f32_value(&fixture.target_point_xyz_f32_bits[1]),
        f32_value(&fixture.target_point_xyz_f32_bits[2]),
    );
    let (projection, _) = project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
    let expected: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
    assert_f32_bits(projection.as_slice(), &expected);
}

#[test]
fn m7fi_ordinal13_homogeneous_point_and_jp_are_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        source: String,
        ordinal: u32,
        #[allow(dead_code)]
        relation: String,
        camera_params_f32_bits: [String; 6],
        direction_xy_f32_bits: [String; 2],
        inverse_distance_f32_bits: String,
        observation_f32_bits: [String; 2],
        sqrt_weight_f32_bits: String,
        transform_t_t_h_f32_bits_column_major: [String; 16],
        target_point4_f32_bits: [String; 4],
        projection_uv_f32_bits: [String; 2],
        raw_residual_uv_f32_bits: [String; 2],
        bearing_j3x2_f32_bits_column_major: [String; 6],
        camera_j2x4_f32_bits_column_major: [String; 8],
        jpp4x3_f32_bits_column_major: [String; 12],
        raw_jp2x3_f32_bits_column_major: [String; 6],
        weighted_jp2x3_f32_bits_column_major: [String; 6],
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7fi_ordinal13_jp.json"
    ))
    .expect("M7fi ordinal-13 Jp fixture must parse");
    assert_eq!(fixture.ordinal, 13);
    let bits = |value: &str| {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    };
    let f32_value = |value: &str| f32::from_bits(bits(value));
    let camera = DoubleSphereCamera::new(
        f32_value(&fixture.camera_params_f32_bits[0]) as f64,
        f32_value(&fixture.camera_params_f32_bits[1]) as f64,
        f32_value(&fixture.camera_params_f32_bits[2]) as f64,
        f32_value(&fixture.camera_params_f32_bits[3]) as f64,
        f32_value(&fixture.camera_params_f32_bits[4]) as f64,
        f32_value(&fixture.camera_params_f32_bits[5]) as f64,
        752,
        480,
    )
    .unwrap();
    let direction = StereographicDirection {
        xy: Point2::new(
            f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
            f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
        ),
    };
    let expected_transform: [u32; 16] =
        std::array::from_fn(|index| bits(&fixture.transform_t_t_h_f32_bits_column_major[index]));
    let transform =
        SMatrix::<f32, 4, 4>::from_column_slice(&expected_transform.map(f32::from_bits));
    assert_f32_bits(transform.as_slice(), &expected_transform);

    let expected_bearing_jacobian: [u32; 6] =
        std::array::from_fn(|index| bits(&fixture.bearing_j3x2_f32_bits_column_major[index]));
    assert_f32_bits(
        direction.bearing_jacobian_f32().as_slice(),
        &expected_bearing_jacobian,
    );
    let inverse_distance = f32_value(&fixture.inverse_distance_f32_bits);
    let point4 =
        eigen_homogeneous_point_product_f32(transform, direction.bearing_f32(), inverse_distance);
    let expected_point: [u32; 4] =
        std::array::from_fn(|index| bits(&fixture.target_point4_f32_bits[index]));
    assert_f32_bits(point4.as_slice(), &expected_point);
    let point = point4.fixed_rows::<3>(0).into_owned();

    let (projection, projection_jacobian) =
        project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
    let expected_projection: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
    assert_f32_bits(projection.as_slice(), &expected_projection);
    let observation = Vector2::new(
        f32_value(&fixture.observation_f32_bits[0]),
        f32_value(&fixture.observation_f32_bits[1]),
    );
    let raw = projection - observation;
    let expected_raw: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.raw_residual_uv_f32_bits[index]));
    assert_f32_bits(raw.as_slice(), &expected_raw);

    let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
    camera_jacobian
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&projection_jacobian);
    let expected_camera: [u32; 8] =
        std::array::from_fn(|index| bits(&fixture.camera_j2x4_f32_bits_column_major[index]));
    assert_f32_bits(camera_jacobian.as_slice(), &expected_camera);

    let mut transform_top_left = SMatrix::<f32, 3, 4>::zeros();
    transform_top_left.copy_from(&transform.fixed_view::<3, 4>(0, 0));
    let mut source_jup = SMatrix::<f32, 4, 2>::zeros();
    source_jup
        .fixed_view_mut::<3, 2>(0, 0)
        .copy_from(&direction.bearing_jacobian_f32());
    let mut point_wrt_landmark = SMatrix::<f32, 3, 3>::zeros();
    point_wrt_landmark.fixed_view_mut::<3, 2>(0, 0).copy_from(
        &eigen_homogeneous_landmark_direction_f32(transform_top_left, source_jup),
    );
    let translation = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);
    point_wrt_landmark.set_column(2, &translation);
    let mut homogeneous_point_wrt_landmark = SMatrix::<f32, 4, 3>::zeros();
    homogeneous_point_wrt_landmark
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&point_wrt_landmark);
    homogeneous_point_wrt_landmark[(3, 2)] = 1.0_f32;
    let expected_jpp: [u32; 12] =
        std::array::from_fn(|index| bits(&fixture.jpp4x3_f32_bits_column_major[index]));
    assert_f32_bits(homogeneous_point_wrt_landmark.as_slice(), &expected_jpp);

    let raw_landmark_jacobian =
        eigen_landmark_jacobian_f32(camera_jacobian, homogeneous_point_wrt_landmark);
    let expected_raw_jp: [u32; 6] =
        std::array::from_fn(|index| bits(&fixture.raw_jp2x3_f32_bits_column_major[index]));
    assert_f32_bits(raw_landmark_jacobian.as_slice(), &expected_raw_jp);
    let weighted_landmark_jacobian =
        raw_landmark_jacobian * f32_value(&fixture.sqrt_weight_f32_bits);
    let expected_weighted_jp: [u32; 6] =
        std::array::from_fn(|index| bits(&fixture.weighted_jp2x3_f32_bits_column_major[index]));
    assert_f32_bits(weighted_landmark_jacobian.as_slice(), &expected_weighted_jp);
}

#[test]
fn m7fi_ordinal311_homogeneous_point_and_projection_are_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        source: String,
        ordinal: u32,
        #[allow(dead_code)]
        relation: String,
        camera_params_f32_bits: [String; 6],
        direction_xy_f32_bits: [String; 2],
        inverse_distance_f32_bits: String,
        observation_f32_bits: [String; 2],
        transform_t_t_h_f32_bits_column_major: [String; 16],
        target_point4_f32_bits: [String; 4],
        projection_uv_f32_bits: [String; 2],
        raw_residual_uv_f32_bits: [String; 2],
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7fi_ordinal311_projection.json"
    ))
    .expect("M7fi ordinal-311 projection fixture must parse");
    assert_eq!(fixture.ordinal, 311);
    let bits = |value: &str| {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    };
    let f32_value = |value: &str| f32::from_bits(bits(value));
    let camera = DoubleSphereCamera::new(
        f32_value(&fixture.camera_params_f32_bits[0]) as f64,
        f32_value(&fixture.camera_params_f32_bits[1]) as f64,
        f32_value(&fixture.camera_params_f32_bits[2]) as f64,
        f32_value(&fixture.camera_params_f32_bits[3]) as f64,
        f32_value(&fixture.camera_params_f32_bits[4]) as f64,
        f32_value(&fixture.camera_params_f32_bits[5]) as f64,
        752,
        480,
    )
    .unwrap();
    let direction = StereographicDirection {
        xy: Point2::new(
            f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
            f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
        ),
    };
    let expected_transform: [u32; 16] =
        std::array::from_fn(|index| bits(&fixture.transform_t_t_h_f32_bits_column_major[index]));
    let transform =
        SMatrix::<f32, 4, 4>::from_column_slice(&expected_transform.map(f32::from_bits));
    assert_f32_bits(transform.as_slice(), &expected_transform);
    let point4 = eigen_homogeneous_point_product_f32(
        transform,
        direction.bearing_f32(),
        f32_value(&fixture.inverse_distance_f32_bits),
    );
    let expected_point: [u32; 4] =
        std::array::from_fn(|index| bits(&fixture.target_point4_f32_bits[index]));
    assert_f32_bits(point4.as_slice(), &expected_point);
    let point = point4.fixed_rows::<3>(0).into_owned();
    let (projection, _) = project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
    let expected_projection: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
    assert_f32_bits(projection.as_slice(), &expected_projection);
    let raw = projection
        - Vector2::new(
            f32_value(&fixture.observation_f32_bits[0]),
            f32_value(&fixture.observation_f32_bits[1]),
        );
    let expected_raw: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.raw_residual_uv_f32_bits[index]));
    assert_f32_bits(raw.as_slice(), &expected_raw);
}

#[test]
fn m7fa_ordinal43_transform_point_and_jp_are_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        source: String,
        ordinal: u32,
        #[allow(dead_code)]
        relation: String,
        camera_params_f32_bits: [String; 6],
        direction_xy_f32_bits: [String; 2],
        inverse_distance_f32_bits: String,
        observation_f32_bits: [String; 2],
        q_xyzw_f32_bits: [String; 4],
        t_xyz_f32_bits: [String; 3],
        transform_t_t_h_f32_bits_column_major: [String; 16],
        target_point4_f32_bits: [String; 4],
        projection_uv_f32_bits: [String; 2],
        raw_residual_uv_f32_bits: [String; 2],
        camera_jacobian_column_major_f32_bits: [String; 8],
        homogeneous_point_jacobian_column_major_f32_bits: [String; 12],
        raw_landmark_jacobian_column_major_f32_bits: [String; 6],
        weighted_landmark_jacobian_column_major_f32_bits: [String; 6],
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7fa_ordinal43_jp.json"
    ))
    .expect("M7fa ordinal-43 fixture must parse");
    assert_eq!(fixture.ordinal, 43);
    let bits = |value: &str| {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    };
    let f32_value = |value: &str| f32::from_bits(bits(value));
    let camera = DoubleSphereCamera::new(
        f32_value(&fixture.camera_params_f32_bits[0]) as f64,
        f32_value(&fixture.camera_params_f32_bits[1]) as f64,
        f32_value(&fixture.camera_params_f32_bits[2]) as f64,
        f32_value(&fixture.camera_params_f32_bits[3]) as f64,
        f32_value(&fixture.camera_params_f32_bits[4]) as f64,
        f32_value(&fixture.camera_params_f32_bits[5]) as f64,
        752,
        480,
    )
    .unwrap();
    let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
        f32_value(&fixture.q_xyzw_f32_bits[3]),
        f32_value(&fixture.q_xyzw_f32_bits[0]),
        f32_value(&fixture.q_xyzw_f32_bits[1]),
        f32_value(&fixture.q_xyzw_f32_bits[2]),
    ));
    let translation = Vector3::new(
        f32_value(&fixture.t_xyz_f32_bits[0]),
        f32_value(&fixture.t_xyz_f32_bits[1]),
        f32_value(&fixture.t_xyz_f32_bits[2]),
    );
    let rotation_matrix = eigen_quaternion_matrix_native_f32(rotation);
    let expected_transform: [u32; 16] =
        std::array::from_fn(|index| bits(&fixture.transform_t_t_h_f32_bits_column_major[index]));
    let mut transform = SMatrix::<f32, 4, 4>::identity();
    transform
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_matrix);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);
    assert_f32_bits(transform.as_slice(), &expected_transform);

    let direction = StereographicDirection {
        xy: Point2::new(
            f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
            f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
        ),
    };
    let inverse_distance = f32_value(&fixture.inverse_distance_f32_bits);
    let point = sophus_homogeneous_point_normalized_rotation_f32(
        rotation,
        translation,
        direction.bearing_f32(),
        inverse_distance,
    );
    let expected_point: [u32; 3] =
        std::array::from_fn(|index| bits(&fixture.target_point4_f32_bits[index]));
    assert_f32_bits(point.as_slice(), &expected_point);

    let (projection, projection_jacobian) =
        project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
    let expected_projection: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
    assert_f32_bits(projection.as_slice(), &expected_projection);
    let observation = Vector2::new(
        f32_value(&fixture.observation_f32_bits[0]),
        f32_value(&fixture.observation_f32_bits[1]),
    );
    let raw = projection - observation;
    let expected_raw: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.raw_residual_uv_f32_bits[index]));
    assert_f32_bits(raw.as_slice(), &expected_raw);

    let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
    camera_jacobian
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&projection_jacobian);
    let expected_camera: [u32; 8] =
        std::array::from_fn(|index| bits(&fixture.camera_jacobian_column_major_f32_bits[index]));
    assert_f32_bits(camera_jacobian.as_slice(), &expected_camera);

    let direction_jacobian = direction.bearing_jacobian_f32();
    let mut transform_top_left = SMatrix::<f32, 3, 4>::zeros();
    transform_top_left
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_matrix);
    let mut source_jup = SMatrix::<f32, 4, 2>::zeros();
    source_jup
        .fixed_view_mut::<3, 2>(0, 0)
        .copy_from(&direction_jacobian);
    let mut point_wrt_landmark = SMatrix::<f32, 3, 3>::zeros();
    point_wrt_landmark.fixed_view_mut::<3, 2>(0, 0).copy_from(
        &eigen_homogeneous_landmark_direction_f32(transform_top_left, source_jup),
    );
    point_wrt_landmark.set_column(2, &translation);
    let mut homogeneous_point_wrt_landmark = SMatrix::<f32, 4, 3>::zeros();
    homogeneous_point_wrt_landmark
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&point_wrt_landmark);
    homogeneous_point_wrt_landmark[(3, 2)] = 1.0;
    let expected_jpp: [u32; 12] = std::array::from_fn(|index| {
        bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[index])
    });
    assert_f32_bits(homogeneous_point_wrt_landmark.as_slice(), &expected_jpp);

    let raw_landmark_jacobian =
        eigen_landmark_jacobian_f32(camera_jacobian, homogeneous_point_wrt_landmark);
    let expected_raw_jp: [u32; 6] = std::array::from_fn(|index| {
        bits(&fixture.raw_landmark_jacobian_column_major_f32_bits[index])
    });
    assert_f32_bits(raw_landmark_jacobian.as_slice(), &expected_raw_jp);
    let weighted_landmark_jacobian = raw_landmark_jacobian * 2.0_f32;
    let expected_weighted_jp: [u32; 6] = std::array::from_fn(|index| {
        bits(&fixture.weighted_landmark_jacobian_column_major_f32_bits[index])
    });
    assert_f32_bits(weighted_landmark_jacobian.as_slice(), &expected_weighted_jp);
}

#[test]
fn m7fq_live_ordinal105_bearing_jacobian_and_jp_are_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        source: String,
        ordinal: u32,
        #[allow(dead_code)]
        relation: String,
        camera_params_f32_bits: [String; 6],
        direction_xy_f32_bits: [String; 2],
        inverse_distance_f32_bits: String,
        observation_f32_bits: [String; 2],
        q_xyzw_f32_bits: [String; 4],
        t_xyz_f32_bits: [String; 3],
        transform_t_t_h_f32_bits_column_major: [String; 16],
        target_point4_f32_bits: [String; 4],
        projection_uv_f32_bits: [String; 2],
        raw_residual_uv_f32_bits: [String; 2],
        bearing_jacobian_column_major_f32_bits: [String; 6],
        camera_jacobian_column_major_f32_bits: [String; 8],
        homogeneous_point_jacobian_column_major_f32_bits: [String; 12],
        raw_landmark_jacobian_column_major_f32_bits: [String; 6],
        weighted_landmark_jacobian_column_major_f32_bits: [String; 6],
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7fq_live_ordinal105.json"
    ))
    .expect("M7fq live ordinal-105 fixture must parse");
    assert_eq!(fixture.ordinal, 105);
    let bits = |value: &str| {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    };
    let f32_value = |value: &str| f32::from_bits(bits(value));
    let camera = DoubleSphereCamera::new(
        f32_value(&fixture.camera_params_f32_bits[0]) as f64,
        f32_value(&fixture.camera_params_f32_bits[1]) as f64,
        f32_value(&fixture.camera_params_f32_bits[2]) as f64,
        f32_value(&fixture.camera_params_f32_bits[3]) as f64,
        f32_value(&fixture.camera_params_f32_bits[4]) as f64,
        f32_value(&fixture.camera_params_f32_bits[5]) as f64,
        752,
        480,
    )
    .unwrap();
    let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
        f32_value(&fixture.q_xyzw_f32_bits[3]),
        f32_value(&fixture.q_xyzw_f32_bits[0]),
        f32_value(&fixture.q_xyzw_f32_bits[1]),
        f32_value(&fixture.q_xyzw_f32_bits[2]),
    ));
    let translation = Vector3::new(
        f32_value(&fixture.t_xyz_f32_bits[0]),
        f32_value(&fixture.t_xyz_f32_bits[1]),
        f32_value(&fixture.t_xyz_f32_bits[2]),
    );
    let rotation_matrix = eigen_quaternion_matrix_native_f32(rotation);
    let expected_transform: [u32; 16] =
        std::array::from_fn(|index| bits(&fixture.transform_t_t_h_f32_bits_column_major[index]));
    let mut transform = SMatrix::<f32, 4, 4>::identity();
    transform
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_matrix);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);
    assert_f32_bits(transform.as_slice(), &expected_transform);

    let direction = StereographicDirection {
        xy: Point2::new(
            f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
            f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
        ),
    };
    let direction_jacobian = direction.bearing_jacobian_f32();
    let expected_bearing_jacobian: [u32; 6] =
        std::array::from_fn(|index| bits(&fixture.bearing_jacobian_column_major_f32_bits[index]));
    // These are the actual live in-context Jpp operands' active bearing
    // lanes.  In particular, lane 0 is 3fba8d4f, not the isolated M7fn
    // replay's 3fba8d50.
    assert_f32_bits(direction_jacobian.as_slice(), &expected_bearing_jacobian);

    let inverse_distance = f32_value(&fixture.inverse_distance_f32_bits);
    let point = sophus_homogeneous_point_normalized_rotation_f32(
        rotation,
        translation,
        direction.bearing_f32(),
        inverse_distance,
    );
    let expected_point: [u32; 3] =
        std::array::from_fn(|index| bits(&fixture.target_point4_f32_bits[index]));
    assert_f32_bits(point.as_slice(), &expected_point);

    let (projection, projection_jacobian) =
        project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
    let expected_projection: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
    assert_f32_bits(projection.as_slice(), &expected_projection);
    let observation = Vector2::new(
        f32_value(&fixture.observation_f32_bits[0]),
        f32_value(&fixture.observation_f32_bits[1]),
    );
    let raw = projection - observation;
    let expected_raw: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.raw_residual_uv_f32_bits[index]));
    assert_f32_bits(raw.as_slice(), &expected_raw);

    let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
    camera_jacobian
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&projection_jacobian);
    let expected_camera: [u32; 8] =
        std::array::from_fn(|index| bits(&fixture.camera_jacobian_column_major_f32_bits[index]));
    assert_f32_bits(camera_jacobian.as_slice(), &expected_camera);

    let mut transform_top_left = SMatrix::<f32, 3, 4>::zeros();
    transform_top_left
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_matrix);
    let mut source_jup = SMatrix::<f32, 4, 2>::zeros();
    source_jup
        .fixed_view_mut::<3, 2>(0, 0)
        .copy_from(&direction_jacobian);
    let mut point_wrt_landmark = SMatrix::<f32, 3, 3>::zeros();
    point_wrt_landmark.fixed_view_mut::<3, 2>(0, 0).copy_from(
        &eigen_homogeneous_landmark_direction_f32(transform_top_left, source_jup),
    );
    point_wrt_landmark.set_column(2, &translation);
    let mut homogeneous_point_wrt_landmark = SMatrix::<f32, 4, 3>::zeros();
    homogeneous_point_wrt_landmark
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&point_wrt_landmark);
    homogeneous_point_wrt_landmark[(3, 2)] = 1.0;
    let expected_jpp: [u32; 12] = std::array::from_fn(|index| {
        bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[index])
    });
    // This is the live pre-product Jpp capture, including its one-ULP
    // lane-0 difference from the old isolated replay.
    assert_f32_bits(homogeneous_point_wrt_landmark.as_slice(), &expected_jpp);

    let raw_landmark_jacobian =
        eigen_landmark_jacobian_f32(camera_jacobian, homogeneous_point_wrt_landmark);
    let expected_raw_jp: [u32; 6] = std::array::from_fn(|index| {
        bits(&fixture.raw_landmark_jacobian_column_major_f32_bits[index])
    });
    // This is the actual live post-product Jp capture, rather than the
    // isolated M7fn product's surrogate operand/result.
    assert_f32_bits(raw_landmark_jacobian.as_slice(), &expected_raw_jp);
    let weighted_landmark_jacobian = raw_landmark_jacobian * 2.0_f32;
    let expected_weighted_jp: [u32; 6] = std::array::from_fn(|index| {
        bits(&fixture.weighted_landmark_jacobian_column_major_f32_bits[index])
    });
    assert_f32_bits(weighted_landmark_jacobian.as_slice(), &expected_weighted_jp);
}

#[test]
fn m7fu_live_ordinal109_camera_jacobian_and_jp_are_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        source: String,
        ordinal: u32,
        #[allow(dead_code)]
        relation: String,
        camera_params_f32_bits: [String; 6],
        target_point_xyz_f32_bits: [String; 3],
        camera_jacobian_column_major_f32_bits: [String; 8],
        homogeneous_point_jacobian_column_major_f32_bits: [String; 12],
        raw_landmark_jacobian_column_major_f32_bits: [String; 6],
        sqrt_weight_f32_bits: String,
        weighted_landmark_jacobian_column_major_f32_bits: [String; 6],
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7fu_live_ordinal109_camera_jp.json"
    ))
    .expect("M7fu live ordinal-109 camera-J/Jp fixture must parse");
    assert_eq!(fixture.ordinal, 109);
    let bits = |value: &str| {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    };
    let f32_value = |value: &str| f32::from_bits(bits(value));
    let camera = DoubleSphereCamera::new(
        f32_value(&fixture.camera_params_f32_bits[0]) as f64,
        f32_value(&fixture.camera_params_f32_bits[1]) as f64,
        f32_value(&fixture.camera_params_f32_bits[2]) as f64,
        f32_value(&fixture.camera_params_f32_bits[3]) as f64,
        f32_value(&fixture.camera_params_f32_bits[4]) as f64,
        f32_value(&fixture.camera_params_f32_bits[5]) as f64,
        752,
        480,
    )
    .unwrap();
    let point = Vector3::new(
        f32_value(&fixture.target_point_xyz_f32_bits[0]),
        f32_value(&fixture.target_point_xyz_f32_bits[1]),
        f32_value(&fixture.target_point_xyz_f32_bits[2]),
    );
    let (_, projection_jacobian) = project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
    let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
    camera_jacobian
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&projection_jacobian);
    let expected_camera: [u32; 8] =
        std::array::from_fn(|index| bits(&fixture.camera_jacobian_column_major_f32_bits[index]));
    assert_f32_bits(camera_jacobian.as_slice(), &expected_camera);

    let expected_jpp: [u32; 12] = std::array::from_fn(|index| {
        bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[index])
    });
    let point_wrt_landmark =
        SMatrix::<f32, 4, 3>::from_column_slice(&expected_jpp.map(f32::from_bits));
    assert_f32_bits(point_wrt_landmark.as_slice(), &expected_jpp);

    let raw_landmark_jacobian = eigen_landmark_jacobian_f32(camera_jacobian, point_wrt_landmark);
    let expected_raw: [u32; 6] = std::array::from_fn(|index| {
        bits(&fixture.raw_landmark_jacobian_column_major_f32_bits[index])
    });
    assert_f32_bits(raw_landmark_jacobian.as_slice(), &expected_raw);

    let weighted_landmark_jacobian =
        raw_landmark_jacobian * f32_value(&fixture.sqrt_weight_f32_bits);
    let expected_weighted: [u32; 6] = std::array::from_fn(|index| {
        bits(&fixture.weighted_landmark_jacobian_column_major_f32_bits[index])
    });
    assert_f32_bits(weighted_landmark_jacobian.as_slice(), &expected_weighted);
}

#[test]
fn m7fb_ordinal110_transform_point_and_projection_are_bitwise_exact() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        schema: String,
        #[allow(dead_code)]
        source: String,
        ordinal: u32,
        #[allow(dead_code)]
        relation: String,
        camera_params_f32_bits: [String; 6],
        direction_xy_f32_bits: [String; 2],
        inverse_distance_f32_bits: String,
        observation_f32_bits: [String; 2],
        q_xyzw_f32_bits: [String; 4],
        t_xyz_f32_bits: [String; 3],
        transform_t_t_h_f32_bits_column_major: [String; 16],
        target_point4_f32_bits: [String; 4],
        projection_uv_f32_bits: [String; 2],
        raw_residual_uv_f32_bits: [String; 2],
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/m7fb_ordinal110_projection.json"
    ))
    .expect("M7fb ordinal-110 fixture must parse");
    assert_eq!(fixture.ordinal, 110);
    let bits = |value: &str| {
        u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
            .expect("f32 fixture bit pattern")
    };
    let f32_value = |value: &str| f32::from_bits(bits(value));
    let camera = DoubleSphereCamera::new(
        f32_value(&fixture.camera_params_f32_bits[0]) as f64,
        f32_value(&fixture.camera_params_f32_bits[1]) as f64,
        f32_value(&fixture.camera_params_f32_bits[2]) as f64,
        f32_value(&fixture.camera_params_f32_bits[3]) as f64,
        f32_value(&fixture.camera_params_f32_bits[4]) as f64,
        f32_value(&fixture.camera_params_f32_bits[5]) as f64,
        752,
        480,
    )
    .unwrap();
    let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
        f32_value(&fixture.q_xyzw_f32_bits[3]),
        f32_value(&fixture.q_xyzw_f32_bits[0]),
        f32_value(&fixture.q_xyzw_f32_bits[1]),
        f32_value(&fixture.q_xyzw_f32_bits[2]),
    ));
    let translation = Vector3::new(
        f32_value(&fixture.t_xyz_f32_bits[0]),
        f32_value(&fixture.t_xyz_f32_bits[1]),
        f32_value(&fixture.t_xyz_f32_bits[2]),
    );
    let rotation_matrix = eigen_quaternion_matrix_native_f32(rotation);
    let mut transform = SMatrix::<f32, 4, 4>::identity();
    transform
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_matrix);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);
    let expected_transform: [u32; 16] =
        std::array::from_fn(|index| bits(&fixture.transform_t_t_h_f32_bits_column_major[index]));
    assert_f32_bits(transform.as_slice(), &expected_transform);

    let direction = StereographicDirection {
        xy: Point2::new(
            f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
            f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
        ),
    };
    let point = sophus_homogeneous_point_normalized_rotation_f32(
        rotation,
        translation,
        direction.bearing_f32(),
        f32_value(&fixture.inverse_distance_f32_bits),
    );
    let expected_point: [u32; 3] =
        std::array::from_fn(|index| bits(&fixture.target_point4_f32_bits[index]));
    assert_f32_bits(point.as_slice(), &expected_point);
    let (projection, _) = project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
    let expected_projection: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
    assert_f32_bits(projection.as_slice(), &expected_projection);
    let raw = projection
        - Vector2::new(
            f32_value(&fixture.observation_f32_bits[0]),
            f32_value(&fixture.observation_f32_bits[1]),
        );
    let expected_raw: [u32; 2] =
        std::array::from_fn(|index| bits(&fixture.raw_residual_uv_f32_bits[index]));
    assert_f32_bits(raw.as_slice(), &expected_raw);
}

#[test]
fn same_time_cam_id_has_exact_identity_and_stereo_keeps_extrinsic() {
    // This fixture deliberately gives the host and target the same frame
    // but nontrivial, otherwise unrelated state/calibration values.  A
    // same-camera observation must nevertheless be identical to an
    // all-identity chain: upstream branches on the complete TimeCamId,
    // rather than relying on the four SE(3) factors to numerically cancel.
    let camera = DoubleSphereCamera::new(300.0, 301.0, 320.0, 240.0, 0.4, 0.7, 640, 480).unwrap();
    let pose = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.03, -0.02, 0.01)),
        Vector3::new(0.2, -0.1, 0.4),
    );
    let anchor_extrinsic = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(-0.01, 0.02, 0.03)),
        Vector3::new(0.04, -0.03, 0.02),
    );
    let target_extrinsic = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.02, 0.01, -0.02)),
        Vector3::new(-0.08, 0.01, 0.03),
    );
    let landmark = InverseDistanceLandmark {
        anchor_pose: 7,
        anchor_camera_id: 0,
        direction: StereographicDirection {
            xy: Point2::new(0.08, -0.04),
        },
        inverse_distance: 0.3,
    };
    let observation = Point2::new(321.0, 239.0);
    let config = FactorConfig {
        observation_stddev: 1.0,
        huber_delta: 0.0,
        outlier_threshold: f64::INFINITY,
    };

    let same_camera = anchored_visual_reprojection_factor_f32_with_time_cam(
        &camera,
        &pose,
        &anchor_extrinsic,
        &SE3::new(pose.rotation, pose.translation),
        &target_extrinsic,
        &landmark,
        observation,
        true,
        true,
        config,
    )
    .unwrap();
    let identity_chain = anchored_visual_reprojection_factor_f32_with_time_cam(
        &camera,
        &SE3::identity(),
        &SE3::identity(),
        &SE3::identity(),
        &SE3::identity(),
        &landmark,
        observation,
        true,
        true,
        config,
    )
    .unwrap();
    assert_eq!(same_camera.projection, identity_chain.projection);
    assert_eq!(same_camera.raw_residual, identity_chain.raw_residual);
    assert_eq!(same_camera.residual, identity_chain.residual);
    assert_eq!(
        same_camera.landmark_jacobian,
        identity_chain.landmark_jacobian
    );
    assert!(same_camera
        .anchor_pose_jacobian
        .iter()
        .all(|value| *value == 0.0));
    assert!(same_camera
        .target_pose_jacobian
        .iter()
        .all(|value| *value == 0.0));

    // The other camera at the same timestamp keeps the relative
    // extrinsic transform.  Compare it to a temporal evaluation with
    // the same equal poses: only the timestamp predicate changes, so the
    // complete factor—including both pose rows—must be identical.
    let stereo = anchored_visual_reprojection_factor_f32_with_time_cam(
        &camera,
        &pose,
        &anchor_extrinsic,
        &pose,
        &target_extrinsic,
        &landmark,
        observation,
        true,
        false,
        config,
    )
    .unwrap();
    let temporal = anchored_visual_reprojection_factor_f32_with_time_cam(
        &camera,
        &pose,
        &anchor_extrinsic,
        &pose,
        &target_extrinsic,
        &landmark,
        observation,
        false,
        false,
        config,
    )
    .unwrap();
    assert_eq!(stereo.projection, temporal.projection);
    assert_eq!(stereo.raw_residual, temporal.raw_residual);
    assert_eq!(stereo.residual, temporal.residual);
    assert_eq!(stereo.landmark_jacobian, temporal.landmark_jacobian);
    assert_eq!(stereo.anchor_pose_jacobian, temporal.anchor_pose_jacobian);
    assert_eq!(stereo.target_pose_jacobian, temporal.target_pose_jacobian);
    assert!(temporal.anchor_pose_jacobian.norm() > 0.0);
    assert!(temporal.target_pose_jacobian.norm() > 0.0);
}

#[test]
fn fej_anchor_endpoint_regression_runs_in_fresh_process() {
    let executable = std::env::current_exe().expect("test executable path");
    let mut command = std::process::Command::new(executable);
    // The diagnostic environment is process-lifetime state.  Remove all
    // Basalt keys before installing the child marker and chain switch so
    // an unrelated test or parent shell cannot decide this child path.
    for (key, _) in std::env::vars_os() {
        if key
            .to_string_lossy()
            .get(.."VISLOC_BASALT_".len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("VISLOC_BASALT_"))
        {
            command.env_remove(key);
        }
    }
    let output = command
        .env("VISLOC_BASALT_FEJ_ANCHOR_ENDPOINT_CHILD", "1")
        .env("VISLOC_BASALT_VISUAL_CHAIN_TRACE", "1")
        .args(["fej_anchor_endpoint_regression_child", "--nocapture"])
        .output()
        .expect("spawn fresh FEJ endpoint regression child");
    assert!(
        output.status.success(),
        "FEJ endpoint child failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn fej_anchor_endpoint_regression_child() {
    if std::env::var_os("VISLOC_BASALT_FEJ_ANCHOR_ENDPOINT_CHILD").is_none() {
        return;
    }
    let (
        camera,
        anchor_pose,
        anchor_extrinsic,
        target_pose,
        target_extrinsic,
        landmark,
        observation,
    ) = m7_fixed_factor_inputs();
    let anchor_pose_fej = anchor_pose.clone();
    let target_pose_fej = target_pose.clone();
    let mut moved_anchor_pose = anchor_pose.clone();
    moved_anchor_pose.translation.x += 0.125;
    let config = FactorConfig {
        observation_stddev: 1.0,
        huber_delta: 0.0,
        outlier_threshold: f64::INFINITY,
    };
    let baseline = anchored_visual_reprojection_factor_f32_with_time_cam_fej(
        &camera,
        &anchor_pose,
        &anchor_pose_fej,
        &anchor_extrinsic,
        &target_pose,
        &target_pose_fej,
        &target_extrinsic,
        &landmark,
        observation,
        false,
        false,
        config,
    )
    .expect("baseline FEJ factor");
    let moved = anchored_visual_reprojection_factor_f32_with_time_cam_fej(
        &camera,
        &moved_anchor_pose,
        &anchor_pose_fej,
        &anchor_extrinsic,
        &target_pose,
        &target_pose_fej,
        &target_extrinsic,
        &landmark,
        observation,
        false,
        false,
        config,
    )
    .expect("moved current-value FEJ factor");
    let baseline_chain = baseline
        .debug_chain
        .as_ref()
        .expect("visual chain sidecar enabled");
    let moved_chain = moved
        .debug_chain
        .as_ref()
        .expect("visual chain sidecar enabled");
    let bits = |values: &[f32]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };

    // Current-value motion must still affect the value-side projection,
    // while the frozen FEJ endpoint and the production anchor matrix stay
    // fixed.  The latter assertion rejects a regression to the removed
    // value-side `for_anchor_j` reconstruction.
    assert_ne!(baseline.projection, moved.projection);
    assert_ne!(
        baseline_chain.target_camera_from_anchor_imu,
        moved_chain.target_camera_from_anchor_imu
    );
    assert_eq!(
        bits(&baseline_chain.target_camera_from_anchor_imu_fej),
        bits(&moved_chain.target_camera_from_anchor_imu_fej)
    );
    assert_eq!(
        bits(&baseline_chain.relative_wrt_anchor),
        bits(&moved_chain.relative_wrt_anchor)
    );
}

#[test]
fn frame4_mh01_huber_objective_matches_upstream_golden() {
    // Actual MH01 frame-4 landmark trace at the pinned upstream SHA:
    // target frame 1403636579963555584, camera 0, pixel
    // (649.7265625, 176.98886108398438).
    let raw = Vector2::new(-0.44049072265625_f64, 0.9515228271484375_f64);
    let huber_weight = 0.95371061563491821_f64;
    let objective = robust_objective_from_raw(
        raw.norm_squared(),
        huber_weight,
        FactorConfig::default().observation_stddev,
    );
    assert!((objective - 2.194144031284797).abs() < 1e-12);
    // The row itself is still sqrt(w)/sigma weighted; only the objective
    // uses the Huber loss correction.
    let whitened_norm_squared = raw.norm_squared() * huber_weight / 0.25;
    assert!((whitened_norm_squared - 4.19414373130865).abs() < 1e-12);
    assert!(objective < whitened_norm_squared);
}

#[test]
fn m7_frame4_visual_f32_vector2_norm_matches_eigen_packet() {
    // Track 120, frame 4 / iteration 0 / target frame 4 cam 0.  This is
    // the first clean factor whose residual crosses the Huber boundary.
    // The pinned Double-Sphere assembly computes x*x first and contracts
    // y*y + x2.  The standalone nalgebra reduction is one ulp different
    // here on this fixture.
    let x = f32::from_bits(0xbee18800);
    let y = f32::from_bits(0x3f739700);
    let squared = eigen_vector2_squared_norm_f32(x, y);
    assert_eq!(squared.to_bits(), 0x3f8cba0d);

    let huber_weight = 1.0_f32 / squared.sqrt();
    let sqrt_weight = huber_weight.sqrt() / 0.5_f32;
    assert_eq!(sqrt_weight.to_bits(), 0x3ffa0138);
    assert_eq!((x * sqrt_weight).to_bits(), 0xbf5c3fe3);
    assert_eq!((y * sqrt_weight).to_bits(), 0x3fede29f);

    // Asymmetric cross-time witness: clean native Huber capture for
    // track 73 / observation 2.  This distinguishes y.mul_add(y, x*x)
    // from the opposite FMA orientation while checking every scalar
    // robust boundary used by the production factor.
    let x = f32::from_bits(0xbe24d800);
    let y = f32::from_bits(0x3f9279c0);
    let squared = eigen_vector2_squared_norm_f32(x, y);
    assert_eq!(squared.to_bits(), 0x3faaef5d);
    let norm = squared.sqrt();
    assert_eq!(norm.to_bits(), 0x3f93eaf6);
    let huber_weight = 1.0_f32 / norm;
    assert_eq!(huber_weight.to_bits(), 0x3f5d8746);
    let sqrt_numerator = huber_weight.sqrt();
    assert_eq!(sqrt_numerator.to_bits(), 0x3f6e242c);
    assert_eq!((sqrt_numerator / 0.5_f32).to_bits(), 0x3fee242c);
}

#[test]
fn ordering_is_pose_velocity_bias_bias() {
    assert_eq!(AomBlock::Pose6.offset(), 0);
    assert_eq!(AomBlock::Velocity3.offset(), 6);
    assert_eq!(AomBlock::GyroBias3.offset(), 9);
    assert_eq!(AomBlock::AccelBias3.offset(), 12);
    assert_eq!(AomBlock::Pose6.dof(), 6);
    assert_eq!(AOM_NAV_DOF, 15);
}
#[test]
fn nullspace_reduction_matches_dense_schur_reference() {
    let f = factor(
        &[1., 0., 2., 1., 0., 1., 1., -1., 2., 1., 0., 1.],
        &[1., 0., 0., 1., 1., 1.],
        &[2., 1., 3.],
        3,
        4,
        2,
    );
    let red = reduce_landmark_factors(&[f.clone()], 4, 1e-10);
    let a = &f.state_jacobian;
    let l = &f.landmark_jacobian;
    let rr = &f.residual;
    let dense = a.transpose() * a
        - a.transpose() * l * (l.transpose() * l).try_inverse().unwrap() * l.transpose() * a;
    let db = a.transpose() * rr
        - a.transpose() * l * (l.transpose() * l).try_inverse().unwrap() * l.transpose() * rr;
    assert!((&red.h - dense).norm() < 1e-10);
    assert!((&red.b - db).norm() < 1e-10);
}

#[test]
fn abs_qr_keeps_observation_rows_after_zero_landmark_damping_rows() {
    // Basalt's ABS_QR storage appends one zero row per landmark column
    // before applying the landmark QR.  The three Q1 rows are removed,
    // leaving all original observation rows in Q2.  A factor with four
    // observations and a three-column landmark is the smallest fixture
    // that makes the row-span contract visible.
    let f = factor(
        &[0., 0., 0., 0.],
        &[1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.],
        &[1., 2., 3., 4.],
        4,
        1,
        3,
    );
    let (reduced_jacobian, reduced_residual, _) = landmark_nullspace_projection(&f, 1e-10);
    assert_eq!(reduced_jacobian.nrows(), 4);
    assert_eq!(reduced_residual.len(), 4);
}

#[test]
fn compact_landmark_backsub_full_rank_matches_legacy_bitwise() {
    let rows = 8;
    let state_cols = 11;
    let state = (0..rows * state_cols)
        .map(|index| (index as f64 - 13.0) / 17.0)
        .collect::<Vec<_>>();
    let landmark = (0..rows * 3)
        .map(|index| {
            let row = index / 3;
            let column = index % 3;
            match column {
                0 => {
                    if row == 0 {
                        1.25
                    } else {
                        0.125 * (row + 1) as f64
                    }
                }
                1 => {
                    if row == 1 {
                        -1.5
                    } else {
                        0.09375 * (row + 2) as f64
                    }
                }
                _ => {
                    if row == 2 {
                        0.75
                    } else {
                        -0.0625 * (row + 3) as f64
                    }
                }
            }
        })
        .collect::<Vec<_>>();
    let residual = (0..rows)
        .map(|row| (row as f64 - 2.0) * 0.1875)
        .collect::<Vec<_>>();
    let factor = factor(&state, &landmark, &residual, rows, state_cols, 3);
    let state_step = DVector::from_iterator(
        state_cols,
        (0..state_cols).map(|column| (column as f64 - 4.0) * 0.03125),
    );

    let legacy =
        back_substitute_landmark_upstream_f32_with_track(&factor, &state_step, 1e-10, Some(73))
            .expect("synthetic landmark block must be full rank");
    let compact =
        compact_landmark_back_substitution_f32(&factor, 5, 73, 1e-10).expect("compact payload");
    assert_eq!(compact.landmark_index, 5);
    assert_eq!(compact.track_id, 73);
    assert_eq!(compact.rank, 3);
    assert!(compact.eligible);
    assert_eq!(compact.q1_state_shape(), (3, state_cols));
    assert_eq!(compact.q1_residual_len(), 3);
    assert_eq!(compact.upper_r_shape(), (3, 3));

    let recovered = back_substitute_landmark_compact_f32(&compact, &state_step, 1e-10)
        .expect("compact landmark recovery");
    assert_eq!(recovered.len(), legacy.len());
    for (index, (&actual, &expected)) in recovered.iter().zip(legacy.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "landmark increment {index} differs"
        );
    }

    // The compact fields are extraction-only views of the same QR
    // storage consumed by the legacy path, including signed zeroes.
    let state32 = as_f32_matrix(&factor.state_jacobian);
    let landmark32 = as_f32_matrix(&factor.landmark_jacobian);
    let residual32 = as_f32_vector(&factor.residual);
    let qr = LandmarkHouseholderF32::factor(&state32, &landmark32, &residual32).unwrap();
    for row in 0..3 {
        for column in 0..state_cols {
            assert_eq!(
                compact.q1_state_value(row, column).to_bits(),
                qr.storage[qr.index(row, column)].to_bits()
            );
        }
        assert_eq!(
            compact.q1_residual_value(row).to_bits(),
            qr.storage[qr.index(row, qr.residual_offset)].to_bits()
        );
        for column in 0..3 {
            let expected = if column < row {
                0.0_f32
            } else {
                qr.storage[qr.index(row, qr.landmark_offset + column)]
            };
            assert_eq!(
                compact.upper_r_value(row, column).to_bits(),
                expected.to_bits()
            );
        }
    }
}

#[test]
fn compact_landmark_backsub_rank_deficiency_fails_closed_like_legacy() {
    let factor = factor(
        &[0.25, -0.5, 0.75, 1.0, -1.25, 1.5, 1.75, -2.0],
        &[1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 3.0, 4.0, 0.0, 4.0],
        &[1.0, -2.0, 3.0, -4.0],
        4,
        2,
        3,
    );
    let state_step = DVector::from_row_slice(&[0.125, -0.25]);
    let compact = compact_landmark_back_substitution_f32(&factor, 9, 101, 1e-10)
        .expect("rank-deficient payload remains inspectable");
    assert!(compact.rank < 3);
    assert!(!compact.eligible);
    assert!(
        back_substitute_landmark_compact_f32(&compact, &state_step, 1e-10).is_none(),
        "ineligible compact payload must not be solved"
    );
    assert!(
        back_substitute_landmark_upstream_f32_with_track(&factor, &state_step, 1e-10, None)
            .is_none(),
        "legacy rank-deficient path must also fail"
    );
}

#[test]
#[ignore = "requires pinned external frame-4 Householder capture"]
fn m7_householder_track1_fixture_is_bitwise_eigen_compatible() {
    // This is the frame-4/track-1/iteration-0 storage captured from the
    // pinned upstream ABS_QR run.  Keep this gate next to the helper so a
    // production call-site cannot silently fall back to nalgebra QR (or
    // change the packet/reduction order) without failing the exact trace.
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/m7_householder_track1_f4_i0.txt");
    let fixture = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
    let mut tokens = fixture.split_whitespace();
    let rows: usize = tokens.next().unwrap().parse().unwrap();
    let storage_cols: usize = tokens.next().unwrap().parse().unwrap();
    assert_eq!((rows, storage_cols), (15, 80));
    let input = (0..rows * storage_cols)
        .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(tokens.next(), Some("Q1_JP"));
    let expected_q1_state = (0..3 * 75)
        .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(tokens.next(), Some("Q1_JL"));
    let expected_q1_landmark = (0..9)
        .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(tokens.next(), Some("Q1_R"));
    let expected_q1_residual = (0..3)
        .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    assert!(tokens.next().is_none());

    let state = DMatrix::from_fn(12, 75, |row, column| input[row * storage_cols + column]);
    let landmark = DMatrix::from_fn(12, 3, |row, column| input[row * storage_cols + 76 + column]);
    let residual = DVector::from_iterator(12, (0..12).map(|row| input[row * storage_cols + 79]));
    let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual).unwrap();
    let fixture_factor = WhitenedFactorRowStack::new(
        state.map(f64::from),
        landmark.map(f64::from),
        residual.map(f64::from),
    )
    .unwrap();
    let compact = compact_landmark_back_substitution_f32(&fixture_factor, 0, 1, 1e-10)
        .expect("fixture compact payload");
    assert_eq!(compact.rank, 3);
    assert!(compact.eligible);
    for row in 0..3 {
        for column in 0..75 {
            assert_eq!(
                compact.q1_state_value(row, column).to_bits(),
                qr.storage[qr.index(row, column)].to_bits()
            );
        }
        assert_eq!(
            compact.q1_residual_value(row).to_bits(),
            qr.storage[qr.index(row, qr.residual_offset)].to_bits()
        );
        for column in 0..3 {
            let expected = if column < row {
                0.0_f32
            } else {
                qr.storage[qr.index(row, qr.landmark_offset + column)]
            };
            assert_eq!(
                compact.upper_r_value(row, column).to_bits(),
                expected.to_bits()
            );
        }
    }
    let fixture_step =
        DVector::from_iterator(75, (0..75).map(|column| (column as f64 - 37.0) * 0.015625));
    let legacy = back_substitute_landmark_upstream_f32_with_track(
        &fixture_factor,
        &fixture_step,
        1e-10,
        Some(1),
    )
    .expect("fixture legacy recovery");
    let recovered = back_substitute_landmark_compact_f32(&compact, &fixture_step, 1e-10)
        .expect("fixture compact recovery");
    for (index, (&actual, &expected)) in recovered.iter().zip(legacy.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "fixture landmark increment {index} differs"
        );
    }

    let mut actual_q1_state = Vec::with_capacity(3 * 75);
    let mut actual_q1_landmark = Vec::with_capacity(9);
    let mut actual_q1_residual = Vec::with_capacity(3);
    for row in 0..3 {
        for column in 0..75 {
            actual_q1_state.push(qr.storage[qr.index(row, column)]);
        }
        for column in 0..3 {
            actual_q1_landmark.push(qr.storage[qr.index(row, qr.landmark_offset + column)]);
        }
        actual_q1_residual.push(qr.storage[qr.index(row, qr.residual_offset)]);
    }
    let count_mismatches = |actual: &[f32], expected: &[f32]| {
        actual
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
            .count()
    };
    assert_eq!(
        count_mismatches(&actual_q1_state, &expected_q1_state),
        0,
        "q1 state bit mismatches"
    );
    assert_eq!(
        count_mismatches(&actual_q1_landmark, &expected_q1_landmark),
        0,
        "q1 landmark bit mismatches"
    );
    assert_eq!(
        count_mismatches(&actual_q1_residual, &expected_q1_residual),
        0,
        "q1 residual bit mismatches"
    );

    assert_f32_bits(&qr.pivots, &[0xc54060ff, 0xc54e1be3, 0x4285df56]);
    assert_f32_bits(&qr.tau, &[0x3fc25a51, 0x3fc213e7, 0x3fcbca3c]);
    assert_f32_bits(
        &[
            qr.storage[qr.index(0, qr.landmark_offset)],
            qr.storage[qr.index(1, qr.landmark_offset + 1)],
            qr.storage[qr.index(2, qr.landmark_offset + 2)],
        ],
        &[0xc54060fe, 0xc54e1be4, 0x4285df56],
    );
    for row in 0..rows {
        assert_eq!(
            qr.storage[qr.index(row, 75)].to_bits(),
            0,
            "padding column changed at row {row}"
        );
    }
    for row in 12..15 {
        for column in 0..qr.storage_cols() {
            assert_eq!(
                qr.storage[qr.index(row, column)].to_bits(),
                0,
                "damping row {row}, column {column} changed"
            );
        }
    }

    let json_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/m7al_upstream_f4_20260821T000200Z/iteration.jsonl");
    let mut expected_q2_state = None;
    let mut expected_q2_residual = None;
    for line in std::fs::read_to_string(&json_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", json_path.display()))
        .lines()
    {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        if value.get("record").and_then(serde_json::Value::as_str) == Some("landmark_qr")
            && value.get("track_id").and_then(serde_json::Value::as_u64) == Some(1)
            && value.get("iteration").and_then(serde_json::Value::as_u64) == Some(0)
        {
            let rows = value["reduced_rows"].as_array().unwrap();
            expected_q2_state = Some(
                rows.iter()
                    .flat_map(|row| {
                        row.as_array()
                            .unwrap()
                            .iter()
                            .map(|value| value.as_f64().unwrap() as f32)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            );
            expected_q2_residual = Some(
                value["reduced_rhs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>(),
            );
            break;
        }
    }
    let expected_q2_state = expected_q2_state.expect("track 1 q2 fixture");
    let expected_q2_residual = expected_q2_residual.expect("track 1 q2 rhs fixture");
    let actual_q2_state = qr.q2_state();
    let actual_q2_residual = qr.q2_residual();
    assert_eq!(actual_q2_state.shape(), (12, 75));
    assert_eq!(expected_q2_state.len(), 12 * 75);
    assert_eq!(expected_q2_residual.len(), 12);
    let actual_q2_state_ref = &actual_q2_state;
    let actual_q2_state = (0..actual_q2_state_ref.nrows())
        .flat_map(|row| {
            (0..actual_q2_state_ref.ncols()).map(move |column| actual_q2_state_ref[(row, column)])
        })
        .collect::<Vec<_>>();
    assert_eq!(
        count_mismatches(&actual_q2_state, &expected_q2_state),
        0,
        "q2 state bit mismatches"
    );
    assert_eq!(
        count_mismatches(actual_q2_residual.as_slice(), &expected_q2_residual),
        0,
        "q2 residual bit mismatches"
    );
}

#[test]
fn m7_landmark_gemv_packet_reduction_and_tails_match_golden_bits() {
    // These widths exercise the scalar-only, Packet4-plus-tail, and
    // Packet8-plus-tail paths.  The expected words come from the pinned
    // Eigen AVX2 row-major GEMV schedule: Packet8/Packet4 FMA lanes are
    // reduced once with the predux tree, followed by plain scalar tail
    // products.
    let fixture_value = |index: usize| {
        let mantissa = ((index.wrapping_mul(0x01f1_2345) + 0x2a5) & 0x007f_ffff) as u32;
        let sign: u32 = if index.is_multiple_of(3) {
            0x8000_0000
        } else {
            0
        };
        f32::from_bits(sign | 0x3e80_0000 | mantissa)
    };
    let expected = [
        (5, [0xbdeae61a, 0xbe2ec959, 0x3f571bd7]),
        (11, [0xbecfde58, 0xbee5f40c, 0x3fcd585d]),
        (14, [0xbf19e8df, 0xbf08d17c, 0x40048312]),
        (75, [0xc0669959, 0xc05b5d00, 0xc05f46cd]),
    ];
    for (columns, expected_bits) in expected {
        let matrix = DMatrix::from_fn(3, columns, |row, column| {
            fixture_value(row * columns + column)
        });
        let vector = DVector::from_iterator(
            columns,
            (0..columns).map(|column| fixture_value(1000 + column)),
        );
        let actual = eigen_row_major_gemv_f32(&matrix, &vector);
        assert_f32_bits(actual.as_slice(), &expected_bits);
    }
}

#[test]
fn m7_householder_tiny_and_signed_zero_match_eigen_semantics() {
    let state = DMatrix::<f32>::zeros(3, 0);
    let residual = DVector::<f32>::zeros(3);
    let mut landmark = DMatrix::<f32>::zeros(3, 3);
    landmark[(0, 0)] = f32::from_bits(0x8000_0000);
    landmark[(1, 0)] = f32::from_bits(0x0000_0001);
    landmark[(2, 0)] = f32::from_bits(0x0000_0001);
    let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual).unwrap();
    assert_eq!(qr.pivots[0].to_bits(), 0x8000_0000);
    assert_eq!(qr.tau[0].to_bits(), 0x0000_0000);
    assert_eq!(
        qr.storage[qr.index(0, qr.landmark_offset)].to_bits(),
        0x8000_0000
    );
    assert_eq!(
        qr.storage[qr.index(1, qr.landmark_offset)].to_bits(),
        0x0000_0001
    );
    assert_eq!(
        qr.storage[qr.index(2, qr.landmark_offset)].to_bits(),
        0x0000_0001
    );

    let mut positive = DMatrix::<f32>::zeros(3, 3);
    positive[(0, 0)] = f32::from_bits(0x0000_0001);
    let qr = LandmarkHouseholderF32::factor(&state, &positive, &residual).unwrap();
    assert_eq!(qr.pivots[0].to_bits(), 0x0000_0001);
    assert_eq!(qr.tau[0].to_bits(), 0x0000_0000);
}

#[test]
#[ignore = "requires pinned external Q2/Eigen Householder captures"]
fn m7_q2_one_factor_f32_reduction_is_bitwise_eigen_compatible() {
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/m7_track1_q2_fixture.txt");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
    let mut tokens = fixture_text.split_whitespace();
    let rows: usize = tokens.next().unwrap().parse().unwrap();
    let cols: usize = tokens.next().unwrap().parse().unwrap();
    assert_eq!((rows, cols), (12, 75));
    let input = (0..rows * cols)
        .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    let jacobian = DMatrix::from_fn(rows, cols, |row, column| input[row * cols + column]);
    assert_eq!(tokens.next(), Some("RHS"));
    let rhs = DVector::from_iterator(
        rows,
        (0..rows).map(|_| tokens.next().unwrap().parse::<f32>().unwrap()),
    );

    let oracle_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/m7_track1_q2_eigen_hb.txt");
    let oracle_text = std::fs::read_to_string(&oracle_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", oracle_path.display()));
    let mut oracle = oracle_text.split_whitespace();
    assert_eq!(oracle.next(), Some("SHAPE"));
    assert_eq!(oracle.next().unwrap().parse::<usize>().unwrap(), cols);
    assert_eq!(oracle.next().unwrap().parse::<usize>().unwrap(), cols);
    assert_eq!(oracle.next().unwrap().parse::<usize>().unwrap(), cols);
    assert_eq!(oracle.next(), Some("H_BITS"));
    let expected_h = (0..cols * cols)
        .map(|_| u32::from_str_radix(oracle.next().unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(oracle.next(), Some("B_BITS"));
    let expected_b = (0..cols)
        .map(|_| u32::from_str_radix(oracle.next().unwrap(), 16).unwrap())
        .collect::<Vec<_>>();

    let h = jacobian.transpose() * &jacobian;
    let mut b = DVector::<f32>::zeros(cols);
    accumulate_transpose_vector_f32_eigen(&mut b, &jacobian, &rhs, false);
    let mut h_mismatch = 0;
    let mut first_h = None;
    for row in 0..cols {
        for column in 0..cols {
            let index = row * cols + column;
            if h[(row, column)].to_bits() != expected_h[index] {
                if first_h.is_none() {
                    first_h = Some((row, column, h[(row, column)].to_bits(), expected_h[index]));
                }
                h_mismatch += 1;
            }
        }
    }
    let mut b_mismatch = 0;
    let mut first_b = None;
    for row in 0..cols {
        if b[row].to_bits() != expected_b[row] {
            if first_b.is_none() {
                first_b = Some((row, b[row].to_bits(), expected_b[row]));
            }
            b_mismatch += 1;
        }
    }
    assert_eq!(
        (h_mismatch, b_mismatch),
        (0, 0),
        "current f32 Q2 accumulation mismatch: first H={first_h:?}, first b={first_b:?}"
    );
}

#[test]
fn m7_q2_abs_normal_system_matches_native_events() {
    // The native logger writes the post-QR Q2 matrix and the ABS H/b
    // assignment in a compact binary record.  Keep this test local to
    // the captured parity corpus: ordinary builds without the diagnostic
    // artifacts simply have no oracle to load.
    fn read_u32(data: &[u8], offset: &mut usize) -> u32 {
        let end = *offset + 4;
        let bytes = data
            .get(*offset..end)
            .unwrap_or_else(|| panic!("truncated native Q2 fixture at {}", *offset));
        *offset = end;
        u32::from_le_bytes(bytes.try_into().unwrap())
    }
    fn read_i64(data: &[u8], offset: &mut usize) -> i64 {
        let end = *offset + 8;
        let bytes = data
            .get(*offset..end)
            .unwrap_or_else(|| panic!("truncated native Q2 fixture at {}", *offset));
        *offset = end;
        i64::from_le_bytes(bytes.try_into().unwrap())
    }
    fn read_matrix(data: &[u8], offset: &mut usize) -> (usize, usize, Vec<f32>) {
        let rows = read_u32(data, offset) as usize;
        let columns = read_u32(data, offset) as usize;
        let values = (0..rows * columns)
            .map(|_| f32::from_bits(read_u32(data, offset)))
            .collect();
        (rows, columns, values)
    }
    fn read_vector(data: &[u8], offset: &mut usize) -> Vec<f32> {
        let length = read_u32(data, offset) as usize;
        (0..length)
            .map(|_| f32::from_bits(read_u32(data, offset)))
            .collect()
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture_root = root.join("../../target/m7im15_native_abs_hb_frame6_max7_20260829");
    let first_fixture = fixture_root.join("abs.event00.bin");
    if !first_fixture.exists() {
        return;
    }

    for event in ["00", "01", "02"] {
        let path = fixture_root.join(format!("abs.event{event}.bin"));
        let data =
            std::fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert_eq!(&data[..11], b"M7IM15ABS1\0");
        let mut offset = 11;
        let _version = read_u32(&data, &mut offset);
        let _event = read_u32(&data, &mut offset);
        let q2_rows = read_u32(&data, &mut offset) as usize;
        let q2_columns = read_u32(&data, &mut offset) as usize;
        let q2_rhs_len = read_u32(&data, &mut offset) as usize;
        let order_count = read_u32(&data, &mut offset) as usize;
        for _ in 0..order_count {
            let _timestamp = read_i64(&data, &mut offset);
            let _offset = read_u32(&data, &mut offset);
            let _dof = read_u32(&data, &mut offset);
        }
        let kfs_count = read_u32(&data, &mut offset) as usize;
        for _ in 0..kfs_count {
            let _frame_id = read_i64(&data, &mut offset);
        }
        let (matrix_rows, matrix_columns, q2_values) = read_matrix(&data, &mut offset);
        let q2_rhs = read_vector(&data, &mut offset);
        let (abs_rows, abs_columns, expected_h) = read_matrix(&data, &mut offset);
        let expected_b = read_vector(&data, &mut offset);
        let _sentinel = read_u32(&data, &mut offset);
        assert_eq!(offset, data.len());
        assert_eq!((matrix_rows, matrix_columns), (q2_rows, q2_columns));
        assert_eq!(q2_rhs.len(), q2_rhs_len);
        assert_eq!((abs_rows, abs_columns), (q2_columns, q2_columns));
        assert_eq!(expected_b.len(), q2_columns);

        let jacobian = DMatrix::from_column_slice(q2_rows, q2_columns, &q2_values)
            .map(|value| f64::from(value));
        let rhs = DVector::from_iterator(q2_rhs.len(), q2_rhs.iter().copied().map(f64::from));
        let (actual_h, actual_b) = q2_f32_normal_system(&jacobian, &rhs);
        let mut h_mismatches = 0;
        let mut b_mismatches = 0;
        for index in 0..expected_h.len() {
            let row = index % q2_columns;
            let column = index / q2_columns;
            if actual_h[(row, column)].to_bits() != f64::from(expected_h[index]).to_bits() {
                h_mismatches += 1;
            }
        }
        for index in 0..expected_b.len() {
            if actual_b[index].to_bits() != f64::from(expected_b[index]).to_bits() {
                b_mismatches += 1;
            }
        }
        assert_eq!(
            (h_mismatches, b_mismatches),
            (0, 0),
            "native Q2 ABS event {event} mismatch"
        );
    }
}

#[test]
#[ignore = "requires pinned external frame-4 and frame-6 model captures"]
fn m7_reduced_model_decrease_candidate_is_not_full_qr_model() {
    // This pinned frame-4/track-1 block contains both Q1 and Q2 rows.
    // Using several deterministic trial steps makes the missing Q1
    // contribution observable without depending on a particular LM
    // damping value or on a diagnostic run being present.
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/m7_householder_track1_f4_i0.txt");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
    let mut tokens = fixture_text.split_whitespace();
    let rows: usize = tokens.next().unwrap().parse().unwrap();
    let storage_cols: usize = tokens.next().unwrap().parse().unwrap();
    assert_eq!((rows, storage_cols), (15, 80));
    let input = (0..rows * storage_cols)
        .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    let state = DMatrix::from_fn(12, 75, |row, column| {
        f64::from(input[row * storage_cols + column])
    });
    let landmark = DMatrix::from_fn(12, 3, |row, column| {
        f64::from(input[row * storage_cols + 76 + column])
    });
    let residual = DVector::from_iterator(
        12,
        (0..12).map(|row| f64::from(input[row * storage_cols + 79])),
    );
    let factor = WhitenedFactorRowStack::new(state, landmark, residual).unwrap();
    let reduced = reduce_landmark_factors_f32(std::slice::from_ref(&factor), 75, 1e-10);

    let steps = [
        DVector::from_iterator(75, (0..75).map(|column| (column as f64 - 37.0) * 0.015625)),
        DVector::from_iterator(
            75,
            (0..75).map(|column| ((column as f64 % 9.0) - 4.0) * 0.03125),
        ),
        DVector::from_iterator(
            75,
            (0..75).map(|column| {
                if column % 7 == 0 {
                    (column as f64 + 1.0) * 0.0078125
                } else {
                    0.0
                }
            }),
        ),
    ];
    let mut mismatches = 0;
    for (trial, step) in steps.iter().enumerate() {
        let full = model_cost_decrease_f32(std::slice::from_ref(&factor), step, 1e-10)
            .expect("full transformed-row model must be finite");
        let reduced_only = reduced_model_cost_decrease_f32(&reduced, step)
            .expect("reduced quadratic must be finite");
        let full_bits = (full as f32).to_bits();
        let reduced_bits = (reduced_only as f32).to_bits();
        println!(
            "m7 reduced model trial={trial} full={full:.9} full_f32={full_bits:08x} reduced={reduced_only:.9} reduced_f32={reduced_bits:08x}"
        );
        if full_bits != reduced_bits {
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, steps.len(), "Q1 rows must affect every witness");
}

#[test]
fn m7_reduced_model_candidate_differs_on_multiple_frontier_trials() {
    // The integrated M7 frame-6 detail stream records the exact reduced
    // H/b and solved f32-cast step used by the LM loop.  Its model field
    // is emitted immediately after model_cost_decrease_f32.  Replaying
    // the reduced quadratic here therefore tests the candidate at the
    // real trial boundary while retaining production call-site
    // isolation.  The fixture is optional for ordinary source builds.
    let detail_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/m7im15_accum_diag_frame6_detail_20260828.jsonl");
    if !detail_path.exists() {
        return;
    }
    let detail = std::fs::read_to_string(&detail_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", detail_path.display()));
    let scalar_fma = |reduced: &ReducedNormalSystemF32, state_step: &DVector<f64>| {
        let step = as_f32_vector(state_step);
        let mut h_step = DVector::zeros(step.len());
        for row in 0..step.len() {
            let mut value = 0.0_f32;
            for column in 0..step.len() {
                value = reduced.h[(row, column)].mul_add(step[column], value);
            }
            h_step[row] = value;
        }
        let mut linear = 0.0_f32;
        let mut quadratic = 0.0_f32;
        for index in 0..step.len() {
            linear = step[index].mul_add(reduced.b[index], linear);
            quadratic = step[index].mul_add(h_step[index], quadratic);
        }
        -(linear + 0.5_f32 * quadratic)
    };
    let ordered = |reduced: &ReducedNormalSystemF32, state_step: &DVector<f64>| {
        let step = as_f32_vector(state_step);
        let mut h_step = DVector::zeros(step.len());
        for row in 0..step.len() {
            let mut value = 0.0_f32;
            for column in 0..step.len() {
                value = add_f32_exact(value, mul_f32_exact(reduced.h[(row, column)], step[column]));
            }
            h_step[row] = value;
        }
        let mut linear = 0.0_f32;
        let mut quadratic = 0.0_f32;
        for index in 0..step.len() {
            linear = add_f32_exact(linear, mul_f32_exact(step[index], reduced.b[index]));
            quadratic = add_f32_exact(quadratic, mul_f32_exact(step[index], h_step[index]));
        }
        -add_f32_exact(linear, mul_f32_exact(0.5_f32, quadratic))
    };
    let mut checked = 0;
    for line in detail.lines() {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        if value.get("phase").and_then(serde_json::Value::as_str) != Some("trial") {
            continue;
        }
        let h_rows = value["global"]["h"].as_array().unwrap();
        let state_dof = h_rows.len();
        assert!(state_dof > 0);
        let h = DMatrix::from_fn(state_dof, state_dof, |row, column| {
            h_rows[row][column].as_f64().unwrap() as f32
        });
        let b_values = value["global"]["b"].as_array().unwrap();
        assert_eq!(b_values.len(), state_dof);
        let b = DVector::from_iterator(
            state_dof,
            b_values.iter().map(|entry| entry.as_f64().unwrap() as f32),
        );
        let step_values = value["delta"].as_array().unwrap();
        assert_eq!(step_values.len(), state_dof);
        let step = DVector::from_iterator(
            state_dof,
            step_values.iter().map(|entry| entry.as_f64().unwrap()),
        );
        let reduced = ReducedNormalSystemF32 {
            h,
            b,
            back_substitution: Vec::new(),
            compact_back_substitution: None,
            model_decrease_payload: None,
            imu_diagnostic: None,
            diagnostic_stages: None,
        };
        let reduced_decrease = reduced_model_cost_decrease_f32(&reduced, &step)
            .expect("frontier reduced quadratic must be finite")
            as f32;
        let scalar_fma_decrease = scalar_fma(&reduced, &step);
        let ordered_decrease = ordered(&reduced, &step);
        let before = value["cost"]["before"].as_f64().unwrap() as f32;
        let expected_model = value["cost"]["model"].as_f64().unwrap() as f32;
        let candidate_model = before - reduced_decrease;
        println!(
            "m7 frontier frame={} iteration={} trial={} expected_model={:08x} candidate_model={:08x} packet_decrease={:08x} scalar_fma_decrease={:08x} ordered_decrease={:08x}",
            value["frame_id"].as_u64().unwrap_or_default(),
            value["iteration"].as_u64().unwrap_or_default(),
            value["trial"].as_u64().unwrap_or_default(),
            expected_model.to_bits(),
            candidate_model.to_bits(),
            reduced_decrease.to_bits(),
            scalar_fma_decrease.to_bits(),
            ordered_decrease.to_bits(),
        );
        assert_ne!(
            expected_model.to_bits(),
            candidate_model.to_bits(),
            "reduced H/b must not replace the complete transformed-row model"
        );
        checked += 1;
        if checked == 6 {
            break;
        }
    }
    assert_eq!(checked, 6, "expected six frame-6 trial records");
}

#[test]
#[ignore = "requires pinned external frame-4 and frame-6 model captures"]
fn m7_reduced_model_q1_constant_candidate_track1_and_frontier() {
    let q1_constant_variants = |entries: &[CompactLandmarkBackSubstitutionF32]| -> [f32; 4] {
        let mut fma_per_factor = 0.0_f32;
        let mut ordered_per_factor = 0.0_f32;
        let mut fma_norm_total = 0.0_f32;
        let mut ordered_norm_total = 0.0_f32;
        for entry in entries {
            assert!(entry.eligible);
            assert_eq!(entry.rank, entry.landmark_cols);
            let mut fma_norm = 0.0_f32;
            let mut ordered_norm = 0.0_f32;
            for row in 0..entry.q1_residual_len() {
                let residual = entry.q1_residual_value(row);
                fma_norm = residual.mul_add(residual, fma_norm);
                ordered_norm = add_f32_exact(ordered_norm, mul_f32_exact(residual, residual));
            }
            fma_norm_total = add_f32_exact(fma_norm_total, fma_norm);
            ordered_norm_total = add_f32_exact(ordered_norm_total, ordered_norm);
            fma_per_factor = add_f32_exact(fma_per_factor, mul_f32_exact(0.5_f32, fma_norm));
            ordered_per_factor =
                add_f32_exact(ordered_per_factor, mul_f32_exact(0.5_f32, ordered_norm));
        }
        [
            fma_per_factor,
            ordered_per_factor,
            mul_f32_exact(0.5_f32, fma_norm_total),
            mul_f32_exact(0.5_f32, ordered_norm_total),
        ]
    };

    // First check the identity on the pinned single visual block.  The
    // Q1 term is step-independent; any remaining difference is due to
    // the global Q2 H/b association/order rather than a missing constant.
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/m7_householder_track1_f4_i0.txt");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
    let mut tokens = fixture_text.split_whitespace();
    let rows: usize = tokens.next().unwrap().parse().unwrap();
    let storage_cols: usize = tokens.next().unwrap().parse().unwrap();
    assert_eq!((rows, storage_cols), (15, 80));
    let input = (0..rows * storage_cols)
        .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    let factor = WhitenedFactorRowStack::new(
        DMatrix::from_fn(12, 75, |row, column| {
            f64::from(input[row * storage_cols + column])
        }),
        DMatrix::from_fn(12, 3, |row, column| {
            f64::from(input[row * storage_cols + 76 + column])
        }),
        DVector::from_iterator(
            12,
            (0..12).map(|row| f64::from(input[row * storage_cols + 79])),
        ),
    )
    .unwrap();
    let reduced = reduce_landmark_factors_f32(std::slice::from_ref(&factor), 75, 1e-10);
    let q1 = compact_landmark_back_substitution_f32(&factor, 0, 1, 1e-10)
        .expect("track1 compact Q1 payload");
    let track_step =
        DVector::from_iterator(75, (0..75).map(|column| (column as f64 - 37.0) * 0.015625));
    let full = model_cost_decrease_f32(std::slice::from_ref(&factor), &track_step, 1e-10)
        .expect("track1 full model decrease");
    let corrected = reduced_model_cost_decrease_with_q1_constant_f32(
        &reduced,
        std::slice::from_ref(&q1),
        &track_step,
    )
    .expect("track1 Q1-corrected model decrease");
    let track_q1_variants = q1_constant_variants(std::slice::from_ref(&q1));
    println!(
        "m7 q1-constant track1 full={:08x} corrected={:08x} q1=({:08x},{:08x},{:08x}) variants=({:08x},{:08x},{:08x},{:08x})",
        (full as f32).to_bits(),
        (corrected as f32).to_bits(),
        q1.q1_residual_value(0).to_bits(),
        q1.q1_residual_value(1).to_bits(),
        q1.q1_residual_value(2).to_bits(),
        track_q1_variants[0].to_bits(),
        track_q1_variants[1].to_bits(),
        track_q1_variants[2].to_bits(),
        track_q1_variants[3].to_bits(),
    );
    assert!(corrected.is_finite());

    // The integrated detail stream contains the original visual factor
    // rows as well as the reduced global H/b.  Rebuild Q1 residuals from
    // those rows, preserving factor order, and test the same candidate on
    // six real frame-6 trial boundaries.
    let detail_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/m7im15_accum_diag_frame6_detail_20260828.jsonl");
    if !detail_path.exists() {
        return;
    }
    let detail = std::fs::read_to_string(&detail_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", detail_path.display()));
    let matrix_f32 = |value: &serde_json::Value| {
        let rows = value.as_array().expect("matrix rows");
        let row_count = rows.len();
        let column_count = rows
            .first()
            .map(|row| row.as_array().expect("matrix row").len())
            .unwrap_or(0);
        DMatrix::from_fn(row_count, column_count, |row, column| {
            rows[row][column].as_f64().expect("matrix value") as f32
        })
    };
    let vector_f32 = |value: &serde_json::Value| {
        let values = value.as_array().expect("vector");
        DVector::from_iterator(
            values.len(),
            values
                .iter()
                .map(|entry| entry.as_f64().expect("vector value") as f32),
        )
    };
    let base_variants = |reduced: &ReducedNormalSystemF32, state_step: &DVector<f64>| {
        let packet = reduced_model_cost_decrease_f32(reduced, state_step)
            .expect("packet reduced model decrease") as f32;
        let step = as_f32_vector(state_step);
        let mut h_step = DVector::zeros(step.len());
        for row in 0..step.len() {
            let mut value = 0.0_f32;
            for column in 0..step.len() {
                value = reduced.h[(row, column)].mul_add(step[column], value);
            }
            h_step[row] = value;
        }
        let mut scalar_linear = 0.0_f32;
        let mut scalar_quadratic = 0.0_f32;
        for index in 0..step.len() {
            scalar_linear = step[index].mul_add(reduced.b[index], scalar_linear);
            scalar_quadratic = step[index].mul_add(h_step[index], scalar_quadratic);
        }
        let scalar_fma = -(scalar_linear + 0.5_f32 * scalar_quadratic);
        let mut ordered_h_step = DVector::zeros(step.len());
        for row in 0..step.len() {
            let mut value = 0.0_f32;
            for column in 0..step.len() {
                value = add_f32_exact(value, mul_f32_exact(reduced.h[(row, column)], step[column]));
            }
            ordered_h_step[row] = value;
        }
        let mut ordered_linear = 0.0_f32;
        let mut ordered_quadratic = 0.0_f32;
        for index in 0..step.len() {
            ordered_linear =
                add_f32_exact(ordered_linear, mul_f32_exact(step[index], reduced.b[index]));
            ordered_quadratic = add_f32_exact(
                ordered_quadratic,
                mul_f32_exact(step[index], ordered_h_step[index]),
            );
        }
        let ordered = -add_f32_exact(ordered_linear, mul_f32_exact(0.5_f32, ordered_quadratic));
        [packet, scalar_fma, ordered]
    };
    let mut checked = 0;
    let mut mismatches = 0;
    for line in detail.lines() {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        if value.get("phase").and_then(serde_json::Value::as_str) != Some("trial") {
            continue;
        }
        let h_rows = value["global"]["h"].as_array().unwrap();
        let state_dof = h_rows.len();
        let h = matrix_f32(&value["global"]["h"]);
        let b = vector_f32(&value["global"]["b"]);
        assert_eq!(h.nrows(), state_dof);
        let step_values = value["delta"].as_array().unwrap();
        let step = DVector::from_iterator(
            step_values.len(),
            step_values
                .iter()
                .map(|entry| entry.as_f64().expect("step value")),
        );
        let mut q1_entries = Vec::new();
        for (landmark_index, factor_value) in value["landmark_factors"]
            .as_array()
            .expect("landmark factors")
            .iter()
            .enumerate()
        {
            let state = matrix_f32(&factor_value["state_jacobian"]);
            let landmark = matrix_f32(&factor_value["landmark_jacobian"]);
            let residual = vector_f32(&factor_value["residual"]);
            let factor = WhitenedFactorRowStack::new(
                state.map(f64::from),
                landmark.map(f64::from),
                residual.map(f64::from),
            )
            .expect("detail visual factor");
            let track_id = factor_value["track_id"].as_u64().expect("track id");
            let compact =
                compact_landmark_back_substitution_f32(&factor, landmark_index, track_id, 1e-10)
                    .expect("full-rank detail visual factor");
            assert_eq!(
                compact.rank,
                factor_value["rank"].as_u64().expect("detail rank") as usize
            );
            q1_entries.push(compact);
        }
        let reduced = ReducedNormalSystemF32 {
            h,
            b,
            back_substitution: Vec::new(),
            compact_back_substitution: None,
            model_decrease_payload: None,
            imu_diagnostic: None,
            diagnostic_stages: None,
        };
        let candidate =
            reduced_model_cost_decrease_with_q1_constant_f32(&reduced, &q1_entries, &step)
                .expect("Q1-corrected frontier model decrease") as f32;
        let bases = base_variants(&reduced, &step);
        let q1_variants = q1_constant_variants(&q1_entries);
        let before = value["cost"]["before"].as_f64().unwrap() as f32;
        let expected_model = value["cost"]["model"].as_f64().unwrap() as f32;
        let candidate_model = before - candidate;
        println!(
            "m7 q1-constant frame={} iteration={} expected_model={:08x} candidate_model={:08x} decrease={:08x}",
            value["frame_id"].as_u64().unwrap_or_default(),
            value["iteration"].as_u64().unwrap_or_default(),
            expected_model.to_bits(),
            candidate_model.to_bits(),
            candidate.to_bits(),
        );
        if expected_model.to_bits() != candidate_model.to_bits() {
            for (base_index, base) in bases.iter().enumerate() {
                for (q1_index, q1_constant) in q1_variants.iter().enumerate() {
                    let decrease = add_f32_exact(*base, *q1_constant);
                    let model = before - decrease;
                    println!(
                        "m7 q1-constant mismatch-variant base={base_index} q1={q1_index} model={:08x} decrease={:08x}",
                        model.to_bits(),
                        decrease.to_bits(),
                    );
                }
            }
            mismatches += 1;
        }
        checked += 1;
        if checked == 6 {
            break;
        }
    }
    assert_eq!(checked, 6, "expected six frame-6 trial records");
    println!("m7 q1-constant frontier mismatches={mismatches}/{checked}");
}

#[test]
#[ignore = "requires pinned external frame-4 Householder capture"]
fn m10_landmark_direct_pack_matches_materialized_track1_and_all61() {
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/m7_householder_track1_f4_i0.txt");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
    let mut tokens = fixture_text.split_whitespace();
    let rows: usize = tokens.next().unwrap().parse().unwrap();
    let storage_cols: usize = tokens.next().unwrap().parse().unwrap();
    assert_eq!((rows, storage_cols), (15, 80));
    let input = (0..rows * storage_cols)
        .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    let track1_state = DMatrix::from_fn(12, 75, |row, column| input[row * storage_cols + column]);
    let track1_landmark =
        DMatrix::from_fn(12, 3, |row, column| input[row * storage_cols + 76 + column]);
    let track1_residual =
        DVector::from_iterator(12, (0..12).map(|row| input[row * storage_cols + 79]));

    // The all61 capture has 61 live visual rows and three ABS_QR damping
    // rows.  Use representable f32 source values so this exercises the
    // same f64->f32 conversion and packet-aligned 75-column layout while
    // keeping the fixture self-contained.
    let all61_state = DMatrix::from_fn(61, 75, |row, column| {
        let value = ((row * 17 + column * 5) % 97) as f32 - 48.0;
        value * 0.00390625
    });
    let all61_landmark = DMatrix::from_fn(61, 3, |row, column| {
        let diagonal = (row == column) as u8 as f32;
        match column {
            0 => 0.75 + diagonal + row as f32 * 0.001953125,
            1 => -0.5 + diagonal * 0.875 - row as f32 * 0.00146484375,
            _ => 0.25 + diagonal * 1.125 + row as f32 * 0.00244140625,
        }
    });
    let all61_residual = DVector::from_fn(61, |row, _| (row as f32 - 30.0) * 0.015625);

    let cases = [
        ("track1", track1_state, track1_landmark, track1_residual),
        ("all61", all61_state, all61_landmark, all61_residual),
    ];
    for (label, state32, landmark32, residual32) in cases {
        let factor = WhitenedFactorRowStack::new(
            state32.clone().map(f64::from),
            landmark32.clone().map(f64::from),
            residual32.clone().map(f64::from),
        )
        .unwrap()
        .with_kind(FactorKind::Visual)
        .with_landmark_metadata(17, 120);

        let materialized =
            LandmarkHouseholderF32::factor(&state32, &landmark32, &residual32).unwrap();
        let (direct, direct_norm) = LandmarkHouseholderF32::factor_from_whitened(&factor).unwrap();
        assert_landmark_qr_bitwise_equal(&direct, &materialized, label);
        let materialized_norm = landmark32.norm();
        assert_eq!(
            direct_norm.to_bits(),
            materialized_norm.to_bits(),
            "{label} landmark norm"
        );

        let metadata = Some(LandmarkFactorMetadata {
            landmark_index: 17,
            track_id: 120,
        });
        let (materialized_q2, materialized_r, materialized_rank, materialized_compact) =
            landmark_nullspace_projection_f32_with_compact(&factor, 1e-10, metadata);
        let mut arena = Vec::new();
        let (direct_q2, direct_r, direct_rank, direct_compact) =
            landmark_nullspace_projection_f32_with_compact_into(
                &factor, 1e-10, metadata, &mut arena,
            );
        assert_matrix_f32_bitwise_equal(&direct_q2, &materialized_q2, label);
        assert_vector_f32_bitwise_equal(&direct_r, &materialized_r, label);
        assert_eq!(direct_rank, materialized_rank, "{label} rank");
        match (materialized_compact, direct_compact) {
            (Some(materialized), Some(entry)) => {
                assert_compact_payload_bitwise_equal(&materialized, &arena, &entry, label);
            }
            (None, None) => {}
            _ => panic!("{label} compact presence mismatch"),
        }
    }
}

#[test]
fn m11_landmark_workspace_reuses_qr_scratch_without_changing_bits() {
    let state = DMatrix::from_fn(12, 75, |row, column| {
        ((row * 19 + column * 7) % 101) as f32 * 0.015625 - 0.75
    });
    let landmark = DMatrix::from_fn(12, 3, |row, column| {
        let diagonal = (row == column) as u8 as f32;
        (column as f32 - 1.0) * 0.25 + diagonal + row as f32 * 0.00390625
    });
    let residual = DVector::from_fn(12, |row, _| row as f32 * 0.03125 - 0.125);
    let factor = WhitenedFactorRowStack::new(
        state.map(f64::from),
        landmark.map(f64::from),
        residual.map(f64::from),
    )
    .unwrap()
    .with_kind(FactorKind::Visual)
    .with_landmark_metadata(17, 120);

    let metadata = Some(LandmarkFactorMetadata {
        landmark_index: 17,
        track_id: 120,
    });
    let mut qr_workspace = LandmarkHouseholderWorkspace::default();
    let mut arena = Vec::new();
    let first = landmark_nullspace_projection_f32_with_compact_into_with_workspace(
        &factor,
        1e-10,
        metadata,
        &mut arena,
        &mut qr_workspace,
    );
    let capacities = (
        qr_workspace.storage.capacity(),
        qr_workspace.pivots.capacity(),
        qr_workspace.tau.capacity(),
        qr_workspace.essential.capacity(),
        qr_workspace.gemv.capacity(),
    );
    assert!(capacities.0 > 0 && capacities.1 > 0 && capacities.3 > 0 && capacities.4 > 0);

    let mut second_arena = Vec::new();
    let second = landmark_nullspace_projection_f32_with_compact_into_with_workspace(
        &factor,
        1e-10,
        metadata,
        &mut second_arena,
        &mut qr_workspace,
    );
    assert_eq!(
        capacities,
        (
            qr_workspace.storage.capacity(),
            qr_workspace.pivots.capacity(),
            qr_workspace.tau.capacity(),
            qr_workspace.essential.capacity(),
            qr_workspace.gemv.capacity(),
        ),
        "same-sized visual factors must reuse QR/scratch capacities"
    );
    assert_matrix_f32_bitwise_equal(&first.0, &second.0, "workspace q2");
    assert_vector_f32_bitwise_equal(&first.1, &second.1, "workspace rhs");
    assert_eq!(first.2, second.2, "workspace rank");
    match (&first.3, &second.3) {
        (Some(first), Some(second)) => {
            assert_eq!(first.landmark_index, second.landmark_index);
            assert_eq!(first.track_id, second.track_id);
            assert_eq!(first.state_cols, second.state_cols);
            assert_eq!(first.landmark_cols, second.landmark_cols);
            assert_eq!(first.rank, second.rank);
            assert_eq!(first.eligible, second.eligible);
            let first_len = first.storage_len().expect("valid compact storage length");
            let second_len = second.storage_len().expect("valid compact storage length");
            assert_eq!(
                &arena[first.storage_offset..first.storage_offset + first_len],
                &second_arena[second.storage_offset..second.storage_offset + second_len]
            );
        }
        _ => panic!("workspace compact presence mismatch"),
    }
}

fn m11_workspace_fixture(
    observation_rows: usize,
    state_cols: usize,
    landmark_cols: usize,
    seed: f64,
    landmark_index: usize,
) -> WhitenedFactorRowStack {
    let state = DMatrix::from_fn(observation_rows, state_cols, |row, column| {
        seed + (row * 17 + column * 5) as f64 * 0.00390625
    });
    let landmark = DMatrix::from_fn(observation_rows, landmark_cols, |row, column| {
        let diagonal = (row == column) as u8 as f64;
        seed * 0.25 + (column as f64 + 1.0) * 0.125 + diagonal + row as f64 * 0.0078125
    });
    let residual = DVector::from_fn(observation_rows, |row, _| {
        seed * 0.5 + row as f64 * 0.015625 - 0.25
    });
    WhitenedFactorRowStack::new(state, landmark, residual)
        .unwrap()
        .with_kind(FactorKind::Visual)
        .with_landmark_metadata(landmark_index, 10_000 + landmark_index as u64)
}

fn m11_assert_workspace_matches_materialized(
    factor: &WhitenedFactorRowStack,
    label: &str,
    workspace: &mut LandmarkHouseholderWorkspace,
    arena: &mut Vec<f32>,
) {
    let metadata = factor.landmark_metadata;
    let expected = landmark_nullspace_projection_f32_with_compact(factor, 1e-10, metadata);
    let actual = landmark_nullspace_projection_f32_with_compact_into_with_workspace(
        factor, 1e-10, metadata, arena, workspace,
    );
    assert_matrix_f32_bitwise_equal(&actual.0, &expected.0, label);
    assert_vector_f32_bitwise_equal(&actual.1, &expected.1, label);
    assert_eq!(actual.2, expected.2, "{label} rank");
    match (&expected.3, &actual.3) {
        (Some(expected), Some(actual)) => {
            assert_compact_payload_bitwise_equal(expected, arena, actual, label);
        }
        (None, None) => {}
        _ => panic!("{label} compact presence mismatch"),
    }
}

fn m11_assert_workspace_rejects_without_mutation(
    factor: &WhitenedFactorRowStack,
    workspace: &mut LandmarkHouseholderWorkspace,
    arena: &mut Vec<f32>,
) {
    let arena_before = arena.clone();
    let capacities_before = (
        workspace.storage.capacity(),
        workspace.pivots.capacity(),
        workspace.tau.capacity(),
        workspace.essential.capacity(),
        workspace.gemv.capacity(),
    );
    let (q2, rhs, rank, compact) =
        landmark_nullspace_projection_f32_with_compact_into_with_workspace(
            factor,
            1e-10,
            factor.landmark_metadata,
            arena,
            workspace,
        );
    assert_eq!(q2.nrows(), 0, "malformed Q2 rows");
    assert_eq!(
        q2.ncols(),
        factor.state_jacobian.ncols(),
        "malformed Q2 cols"
    );
    assert_eq!(rhs.len(), 0, "malformed rhs");
    assert_eq!(rank, 0, "malformed rank");
    assert!(compact.is_none(), "malformed compact payload");
    assert_eq!(*arena, arena_before, "malformed input changed arena");
    assert_eq!(
        capacities_before,
        (
            workspace.storage.capacity(),
            workspace.pivots.capacity(),
            workspace.tau.capacity(),
            workspace.essential.capacity(),
            workspace.gemv.capacity(),
        ),
        "malformed input consumed reusable workspace"
    );
}

#[test]
fn m11_landmark_workspace_mixed_shapes_and_rejection_keep_bits() {
    let large_three = m11_workspace_fixture(24, 31, 3, 0.25, 31);
    let small_one = m11_workspace_fixture(8, 7, 1, -0.5, 11);
    let small_two = m11_workspace_fixture(10, 7, 2, 1.0, 22);
    let mut workspace = LandmarkHouseholderWorkspace::default();
    let mut arena = Vec::new();

    // Exercise 3 -> 1 -> 2 columns and a large -> small transition in a
    // single reusable workspace.  Every result is compared with the
    // materialized, independent f32 QR wrapper bit-for-bit.
    m11_assert_workspace_matches_materialized(
        &large_three,
        "mixed 3-column large",
        &mut workspace,
        &mut arena,
    );
    m11_assert_workspace_matches_materialized(
        &small_one,
        "mixed 1-column small",
        &mut workspace,
        &mut arena,
    );
    m11_assert_workspace_matches_materialized(
        &small_two,
        "mixed 2-column small",
        &mut workspace,
        &mut arena,
    );

    // Once the workspace is warm, each malformed shape must fail before
    // taking a reusable vector or appending a compact arena entry.
    let mut wrong_rows = small_two.clone();
    wrong_rows.landmark_jacobian = DMatrix::zeros(9, 2);
    m11_assert_workspace_rejects_without_mutation(&wrong_rows, &mut workspace, &mut arena);

    let mut wrong_residual = small_two.clone();
    wrong_residual.residual = DVector::zeros(9);
    m11_assert_workspace_rejects_without_mutation(&wrong_residual, &mut workspace, &mut arena);

    let mut wrong_columns = small_two.clone();
    wrong_columns.landmark_jacobian = DMatrix::zeros(10, 4);
    m11_assert_workspace_rejects_without_mutation(&wrong_columns, &mut workspace, &mut arena);

    // Reuse after all three rejection paths must still produce the same
    // Q2/RHS/rank/compact bits as the materialized reference.
    m11_assert_workspace_matches_materialized(
        &large_three,
        "post-rejection 3-column",
        &mut workspace,
        &mut arena,
    );
}

#[test]
fn m11_compact_lengths_fail_closed_without_panic() {
    assert!(checked_landmark_householder_layout(usize::MAX, 7, 3).is_none());
    assert!(checked_landmark_householder_layout(8, usize::MAX, 3).is_none());
    assert!(checked_compact_storage_len(3, usize::MAX).is_none());
    let fixture = m11_workspace_fixture(8, 7, 3, 0.5, 3);
    assert_eq!(
        checked_compact_storage_capacity(std::slice::from_ref(&fixture), 7),
        checked_compact_storage_len(3, 7)
    );
    assert!(checked_compact_storage_capacity(std::slice::from_ref(&fixture), usize::MAX).is_none());

    let entry = CompactLandmarkBackSubstitutionEntryF32 {
        landmark_index: 0,
        track_id: 0,
        storage_offset: usize::MAX,
        state_cols: usize::MAX,
        landmark_cols: 3,
        rank: 3,
        eligible: true,
    };
    assert!(entry.storage_len().is_none());
    assert!(entry.view(&[]).is_none());
    assert!(
        back_substitute_landmark_compact_entry_f32(&entry, &[], &DVector::zeros(0), 1e-10,)
            .is_none()
    );

    let payload = CompactLandmarkBackSubstitutionF32 {
        landmark_index: 0,
        track_id: 0,
        storage: Vec::new(),
        state_cols: usize::MAX,
        landmark_cols: 3,
        rank: 3,
        eligible: true,
    };
    assert!(payload.storage_len().is_none());
    assert!(back_substitute_landmark_compact_f32(&payload, &DVector::zeros(0), 1e-10,).is_none());
}

#[test]
fn m10_landmark_direct_pack_rejects_malformed_shapes_without_mutating_arena() {
    let state = DMatrix::from_fn(8, 7, |row, column| (row * 7 + column) as f32 * 0.0625);
    let landmark = DMatrix::from_fn(8, 3, |row, column| {
        if row == column {
            1.0
        } else {
            (row + column + 1) as f32 * 0.03125
        }
    });
    let residual = DVector::from_fn(8, |row, _| row as f32 * 0.125 - 0.25);
    let valid = WhitenedFactorRowStack::new(
        state.map(f64::from),
        landmark.map(f64::from),
        residual.map(f64::from),
    )
    .unwrap();
    assert!(LandmarkHouseholderF32::factor_from_whitened(&valid).is_some());

    let mut wrong_landmark_rows = valid.clone();
    wrong_landmark_rows.landmark_jacobian = DMatrix::zeros(7, 3);
    assert!(LandmarkHouseholderF32::factor_from_whitened(&wrong_landmark_rows).is_none());
    let mut arena = vec![f32::from_bits(0x3f80_0000)];
    let arena_before = arena.clone();
    let (q2, rhs, rank, compact) = landmark_nullspace_projection_f32_with_compact_into(
        &wrong_landmark_rows,
        1e-10,
        Some(LandmarkFactorMetadata {
            landmark_index: 0,
            track_id: 1,
        }),
        &mut arena,
    );
    assert_eq!(q2.nrows(), 0);
    assert_eq!(rhs.len(), 0);
    assert_eq!(rank, 0);
    assert!(compact.is_none());
    assert_eq!(arena, arena_before);

    let mut wrong_residual_len = valid;
    wrong_residual_len.residual = DVector::zeros(7);
    assert!(LandmarkHouseholderF32::factor_from_whitened(&wrong_residual_len).is_none());

    let four_landmark = WhitenedFactorRowStack::new(
        state.map(f64::from),
        DMatrix::from_fn(8, 4, |row, column| {
            if row == column {
                1.0
            } else {
                (row + column + 1) as f64 * 0.03125
            }
        }),
        residual.map(f64::from),
    )
    .unwrap();
    assert!(LandmarkHouseholderF32::factor_from_whitened(&four_landmark).is_none());
    let (q2, rhs, rank, compact) = landmark_nullspace_projection_f32_with_compact_into(
        &four_landmark,
        1e-10,
        None,
        &mut arena,
    );
    assert_eq!(q2.nrows(), 0);
    assert_eq!(rhs.len(), 0);
    assert_eq!(rank, 0);
    assert!(compact.is_none());
    assert_eq!(arena, arena_before);
}

/// The support-restricted visual gram must reproduce the dense kernel
/// plus dense `+=` bit for bit, over window widths that hit every Eigen
/// panel boundary and support layouts from sparse to nearly dense.
#[test]
fn sparse_visual_gram_accumulation_matches_dense_bits() {
    let mut state = 0x9e37_79b9_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut sparse_cases = 0;
    for case in 0..600 {
        let columns = 24 + (next() % 110) as usize;
        let rows = 3 + (next() % 40) as usize;
        // Choose 1-4 random blocks of 6 or 15 columns, like pose-only
        // keyframes and full navigation states.
        let mut active = vec![false; columns];
        for _ in 0..1 + next() % 4 {
            let width = if next() % 2 == 0 { 6 } else { 15 };
            let start = (next() as usize) % columns;
            for column in start..(start + width).min(columns) {
                active[column] = true;
            }
        }
        let jacobian = DMatrix::from_fn(rows, columns, |_, column| {
            if active[column] && next() % 8 != 0 {
                ((next() % 20_001) as f32 - 10_000.0) * 1.3e-3
            } else {
                0.0
            }
        });
        let base = DMatrix::from_fn(columns, columns, |_, _| {
            ((next() % 2001) as f32 - 1000.0) * 0.37
        });
        let mut dense = base.clone();
        dense += eigen_visual_gram_packet_tail_f32(&jacobian);
        let contribution = eigen_visual_gram_packet_tail_sparse_f32(&jacobian);
        if matches!(contribution, VisualGramContribution::Sparse { .. }) {
            sparse_cases += 1;
        }
        let mut sparse = base;
        contribution.add_to(&mut sparse);
        for (index, (d, s)) in dense.iter().zip(sparse.iter()).enumerate() {
            assert_eq!(
                d.to_bits(),
                s.to_bits(),
                "case {case} rows {rows} columns {columns} entry {index}: {d} vs {s}"
            );
        }
    }
    assert!(
        sparse_cases > 300,
        "only {sparse_cases} cases took the sparse path"
    );
}

#[test]
#[ignore = "requires pinned external frame-4 Householder capture"]
fn m7_q2_model_reuse_track1_is_bitwise_exact_without_qr() {
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/m7_householder_track1_f4_i0.txt");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
    let mut tokens = fixture_text.split_whitespace();
    let rows: usize = tokens.next().unwrap().parse().unwrap();
    let storage_cols: usize = tokens.next().unwrap().parse().unwrap();
    assert_eq!((rows, storage_cols), (15, 80));
    let input = (0..rows * storage_cols)
        .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    let state32 = DMatrix::from_fn(12, 75, |row, column| input[row * storage_cols + column]);
    let landmark32 = DMatrix::from_fn(12, 3, |row, column| input[row * storage_cols + 76 + column]);
    let residual32 = DVector::from_iterator(12, (0..12).map(|row| input[row * storage_cols + 79]));
    let factor = WhitenedFactorRowStack::new(
        state32.map(f64::from),
        landmark32.map(f64::from),
        residual32.map(f64::from),
    )
    .unwrap();

    // The only QR in this test is payload preparation.  The reuse helper
    // below receives Q1/Q2 values and never calls LandmarkHouseholderF32.
    let qr = LandmarkHouseholderF32::factor(&state32, &landmark32, &residual32).unwrap();
    let payload = QrModelReusePayloadF32 {
        q1: qr
            .compact_back_substitution(0, 1, 3, true)
            .expect("valid compact payload"),
        q2_state: qr.q2_state(),
        q2_residual: qr.q2_residual(),
    };
    let steps = [
        DVector::from_iterator(75, (0..75).map(|column| (column as f64 - 37.0) * 0.015625)),
        DVector::from_iterator(
            75,
            (0..75).map(|column| ((column as f64 % 9.0) - 4.0) * 0.03125),
        ),
        DVector::from_iterator(
            75,
            (0..75).map(|column| {
                if column % 7 == 0 {
                    (column as f64 + 1.0) * 0.0078125
                } else {
                    0.0
                }
            }),
        ),
    ];
    for (index, step) in steps.iter().enumerate() {
        let expected = model_cost_decrease_f32(std::slice::from_ref(&factor), step, 1e-10)
            .expect("track1 full model decrease") as f32;
        let actual = model_cost_decrease_from_qr_payload_f32(&payload, step, 1e-10)
            .expect("track1 QR payload model decrease") as f32;
        println!(
            "m7 q2-reuse track1 step={index} full={:08x} reuse={:08x}",
            expected.to_bits(),
            actual.to_bits(),
        );
        assert_eq!(actual.to_bits(), expected.to_bits(), "track1 step {index}");
    }
}

#[test]
fn m7_q2_model_reuse_preserves_mixed_factor_order_bitwise() {
    let state_cols = 7;
    let visual_state = DMatrix::from_fn(6, state_cols, |row, column| {
        (row as f32 - 2.0) * 0.1875 + (column as f32 - 3.0) * 0.03125
    });
    let visual_landmark = DMatrix::from_fn(6, 3, |row, column| match column {
        0 => [1.0_f32, 0.125, -0.25, 0.75, 0.5, -0.375][row],
        1 => [-0.5_f32, 1.25, 0.25, -0.625, 0.875, 0.3125][row],
        _ => [0.25_f32, -0.375, 1.5, 0.5, -0.75, 0.625][row],
    });
    let visual_residual = DVector::from_fn(6, |row, _| (row as f32 - 1.5) * 0.25);
    let visual_factor = WhitenedFactorRowStack::new(
        visual_state.map(f64::from),
        visual_landmark.map(f64::from),
        visual_residual.map(f64::from),
    )
    .unwrap()
    .with_kind(FactorKind::Visual);
    let qr =
        LandmarkHouseholderF32::factor(&visual_state, &visual_landmark, &visual_residual).unwrap();
    let visual_payload = QrModelReusePayloadF32 {
        q1: qr
            .compact_back_substitution(0, 700, 3, true)
            .expect("valid compact payload"),
        q2_state: qr.q2_state(),
        q2_residual: qr.q2_residual(),
    };

    let plain = |rows: usize, kind: FactorKind, seed: f32| {
        let state = DMatrix::from_fn(rows, state_cols, |row, column| {
            seed + (row * state_cols + column) as f32 * 0.0625
        });
        let residual = DVector::from_fn(rows, |row, _| seed * 0.25 - (row as f32 + 1.0) * 0.09375);
        let factor = WhitenedFactorRowStack::new(
            state.map(f64::from),
            DMatrix::zeros(rows, 0),
            residual.map(f64::from),
        )
        .unwrap()
        .with_kind(kind);
        (
            factor,
            ModelReusePayloadF32::Plain {
                kind,
                state,
                residual,
            },
        )
    };
    let (prior, prior_payload) = plain(4, FactorKind::Prior, 0.5);
    let (imu, imu_payload) = plain(9, FactorKind::Imu, -0.75);
    let (bias, bias_payload) = plain(6, FactorKind::Bias, 1.25);
    // Keep a deliberately mixed order.  The model accumulator is a
    // per-factor f32 subtraction, so payload order is part of the test.
    let factors = vec![prior, visual_factor, imu, bias];
    let payloads = vec![
        prior_payload,
        ModelReusePayloadF32::Visual(visual_payload),
        imu_payload,
        bias_payload,
    ];
    let steps = [
        DVector::from_iterator(
            state_cols,
            (0..state_cols).map(|column| (column as f64 - 2.0) * 0.03125),
        ),
        DVector::from_iterator(
            state_cols,
            (0..state_cols).map(|column| (column as f64 + 1.0) * -0.046875),
        ),
    ];
    for (index, step) in steps.iter().enumerate() {
        let expected = model_cost_decrease_f32(&factors, step, 1e-10)
            .expect("mixed full model decrease") as f32;
        let actual = model_cost_decrease_from_payloads_f32(&payloads, &factors, step, 1e-10)
            .expect("mixed QR payload model decrease") as f32;
        println!(
            "m7 q2-reuse mixed step={index} full={:08x} reuse={:08x}",
            expected.to_bits(),
            actual.to_bits(),
        );
        assert_eq!(actual.to_bits(), expected.to_bits(), "mixed step {index}");
    }
}

#[test]
fn m7_q2_model_reuse_reducer_rejects_malformed_shapes_without_panic() {
    let valid = WhitenedFactorRowStack::new(
        DMatrix::zeros(3, 4),
        DMatrix::zeros(3, 0),
        DVector::zeros(3),
    )
    .unwrap();
    let wrong_width = WhitenedFactorRowStack::new(
        DMatrix::zeros(3, 3),
        DMatrix::zeros(3, 0),
        DVector::zeros(3),
    )
    .unwrap();
    assert!(matches!(
        reduce_landmark_factors_f32_checked_with_compact_back_substitution(
            &[wrong_width],
            4,
            1e-10,
        ),
        Err(ImuReductionError::StateWidth {
            index: 0,
            expected: 4,
            actual: 3,
        })
    ));

    let mut wrong_rows = valid.clone();
    wrong_rows.state_jacobian = DMatrix::zeros(2, 4);
    assert!(matches!(
        reduce_landmark_factors_f32_checked_with_compact_back_substitution(&[wrong_rows], 4, 1e-10,),
        Err(ImuReductionError::InvalidFactorShape {
            index: 0,
            state_rows: 2,
            landmark_rows: 3,
            residual_len: 3,
        })
    ));

    // A malformed producer count must not be truncated by `zip`; the
    // all-or-nothing sidecar contract returns None before inspecting any
    // factor identity, allowing the caller to use the full evaluator.
    let empty_compact = CompactLandmarkBackSubstitutionBatchF32 {
        storage: Vec::new(),
        entries: Vec::new(),
    };
    assert!(move_model_decrease_payload(&[valid], Vec::new(), Some(&empty_compact),).is_none());
}

#[test]
fn m7_imu_bias_reduction_uses_one_fifteen_row_stack_and_tags_prior() {
    let state_dof = 40;
    let offsets = ImuLinkOffsets { start: 6, end: 21 };
    let local_imu_jacobian = DMatrix::from_fn(9, AOM_NAV_DOF * 2, |row, column| {
        let seed = (row * AOM_NAV_DOF * 2 + column) as f32;
        if column < AOM_NAV_DOF {
            10_000.0_f32 + seed * 0.25
        } else {
            -0.75_f32 + seed * 0.03125
        }
    });
    let mut imu_jacobian = DMatrix::<f32>::zeros(9, state_dof);
    for row in 0..9 {
        for column in 0..AOM_NAV_DOF * 2 {
            let block = column / AOM_NAV_DOF;
            let local_column = column % AOM_NAV_DOF;
            let global_offset = if block == 0 {
                offsets.start
            } else {
                offsets.end
            };
            imu_jacobian[(row, global_offset + local_column)] = local_imu_jacobian[(row, column)];
        }
    }
    let imu_residual = DVector::from_fn(9, |row, _| 0.25_f32 + row as f32 * 0.125);
    let local_bias_jacobian = DMatrix::from_fn(6, AOM_NAV_DOF * 2, |row, column| {
        let seed = (row * AOM_NAV_DOF * 2 + column) as f32;
        if column < AOM_NAV_DOF {
            0.03125_f32 + seed * 0.0078125
        } else {
            2.0_f32 + seed * 0.0625
        }
    });
    let mut bias_jacobian = DMatrix::<f32>::zeros(6, state_dof);
    for row in 0..6 {
        for column in 0..AOM_NAV_DOF * 2 {
            let block = column / AOM_NAV_DOF;
            let local_column = column % AOM_NAV_DOF;
            let global_offset = if block == 0 {
                offsets.start
            } else {
                offsets.end
            };
            bias_jacobian[(row, global_offset + local_column)] = local_bias_jacobian[(row, column)];
        }
    }
    let bias_residual = DVector::from_fn(6, |row, _| -0.5_f32 + row as f32 * 0.03125);

    let imu = WhitenedFactorRowStack::with_objective_cost_kind(
        imu_jacobian.clone().map(f64::from),
        DMatrix::zeros(9, 0),
        imu_residual.map(f64::from),
        0.0,
        FactorKind::Imu,
    )
    .unwrap()
    .with_imu_link_offsets(offsets.start, offsets.end);
    let bias = WhitenedFactorRowStack::with_objective_cost_kind(
        bias_jacobian.clone().map(f64::from),
        DMatrix::zeros(6, 0),
        bias_residual.map(f64::from),
        0.0,
        FactorKind::Bias,
    )
    .unwrap()
    .with_imu_link_offsets(offsets.start, offsets.end);

    // A nine-row prior is deliberately placed before the IMU pair. It
    // must remain a generic dynamic product despite sharing the IMU row
    // count. This also verifies that the explicit semantic tag, rather
    // than shape alone, controls the special schedule.
    let prior_jacobian = DMatrix::from_fn(9, state_dof, |row, column| {
        if (offsets.start..offsets.start + AOM_NAV_DOF).contains(&column)
            || (offsets.end..offsets.end + AOM_NAV_DOF).contains(&column)
        {
            0.5_f32 + (row * state_dof + column) as f32 * 0.09375
        } else {
            0.0
        }
    });
    let prior_residual = DVector::from_fn(9, |row, _| 1.0_f32 - row as f32 * 0.0625);
    let prior = WhitenedFactorRowStack::with_objective_cost_kind(
        prior_jacobian.map(f64::from),
        DMatrix::zeros(9, 0),
        prior_residual.map(f64::from),
        0.0,
        FactorKind::Prior,
    )
    .unwrap();

    let reduced = reduce_landmark_factors_f32(&[prior, imu, bias], state_dof, 1e-10);
    let (expected_local_h, expected_local_b) = local_imu_h_b_15x30(
        &DMatrix::from_fn(IMU_LOCAL_ROWS, IMU_LOCAL_COLS, |row, column| {
            if row < 9 {
                local_imu_jacobian[(row, column)]
            } else {
                local_bias_jacobian[(row - 9, column)]
            }
        }),
        &DVector::from_iterator(
            15,
            imu_residual
                .iter()
                .copied()
                .chain(bias_residual.iter().copied()),
        ),
    );
    let mut expected_h = DMatrix::<f32>::zeros(state_dof, state_dof);
    let mut expected_b = DVector::<f32>::zeros(state_dof);
    scatter_local_imu_h_b_15x30(
        &mut expected_h,
        &mut expected_b,
        &expected_local_h,
        &expected_local_b,
        offsets,
    )
    .unwrap();
    let prior_h = prior_jacobian.transpose() * &prior_jacobian;
    let mut prior_b = DVector::<f32>::zeros(state_dof);
    accumulate_transpose_vector_f32_eigen(&mut prior_b, &prior_jacobian, &prior_residual, false);
    expected_h += prior_h;
    expected_b += prior_b;
    assert_eq!(reduced.h, expected_h);
    assert_eq!(reduced.b, expected_b);
}

#[test]
fn m7_checked_imu_reducer_rejects_missing_bias() {
    let state_dof = 36;
    let offsets = ImuLinkOffsets { start: 3, end: 18 };
    let (imu, _) = tagged_imu_bias_pair(state_dof, Some(offsets), None, None);
    assert!(matches!(
        reduce_landmark_factors_f32_checked(&[imu], state_dof, 1e-10),
        Err(ImuReductionError::MissingBias { .. })
    ));
}

#[test]
fn m7_checked_imu_reducer_rejects_missing_offsets() {
    let state_dof = 36;
    let (imu, bias) = tagged_imu_bias_pair(state_dof, None, None, None);
    assert!(matches!(
        reduce_landmark_factors_f32_checked(&[imu, bias], state_dof, 1e-10),
        Err(ImuReductionError::MissingOffsets { .. })
    ));
}

#[test]
fn m7_checked_imu_reducer_rejects_mismatched_offsets() {
    let state_dof = 36;
    let imu_offsets = ImuLinkOffsets { start: 3, end: 18 };
    let bias_offsets = ImuLinkOffsets { start: 4, end: 19 };
    let (imu, bias) = tagged_imu_bias_pair(state_dof, Some(imu_offsets), Some(bias_offsets), None);
    assert!(matches!(
        reduce_landmark_factors_f32_checked(&[imu, bias], state_dof, 1e-10),
        Err(ImuReductionError::MismatchedOffsets { .. })
    ));
}

#[test]
fn m7_checked_imu_reducer_rejects_active_columns_outside_link_offsets() {
    let state_dof = 36;
    let offsets = ImuLinkOffsets { start: 3, end: 18 };
    let (imu, bias) = tagged_imu_bias_pair(state_dof, Some(offsets), Some(offsets), Some(35));
    assert!(matches!(
        reduce_landmark_factors_f32_checked(&[imu, bias], state_dof, 1e-10),
        Err(ImuReductionError::UnexpectedColumns { .. })
    ));
}

#[test]
fn m7_three_state_imu_links_scatter_shared_state_in_source_order() {
    let state_dof = AOM_NAV_DOF * 3;
    let link0_offsets = ImuLinkOffsets {
        start: 0,
        end: AOM_NAV_DOF,
    };
    let link1_offsets = ImuLinkOffsets {
        start: AOM_NAV_DOF,
        end: AOM_NAV_DOF * 2,
    };

    let make_pair = |offsets: ImuLinkOffsets, scale: f32| {
        let local_imu = DMatrix::from_fn(9, AOM_NAV_DOF * 2, |row, column| {
            scale + (row * AOM_NAV_DOF * 2 + column) as f32 * 0.03125
        });
        let local_bias = DMatrix::from_fn(6, AOM_NAV_DOF * 2, |row, column| {
            -scale + (row * AOM_NAV_DOF * 2 + column) as f32 * 0.0625
        });
        let mut imu_jacobian = DMatrix::<f32>::zeros(9, state_dof);
        let mut bias_jacobian = DMatrix::<f32>::zeros(6, state_dof);
        for row in 0..9 {
            for column in 0..AOM_NAV_DOF * 2 {
                let global_offset = if column < AOM_NAV_DOF {
                    offsets.start
                } else {
                    offsets.end
                };
                imu_jacobian[(row, global_offset + column % AOM_NAV_DOF)] =
                    local_imu[(row, column)];
            }
        }
        for row in 0..6 {
            for column in 0..AOM_NAV_DOF * 2 {
                let global_offset = if column < AOM_NAV_DOF {
                    offsets.start
                } else {
                    offsets.end
                };
                bias_jacobian[(row, global_offset + column % AOM_NAV_DOF)] =
                    local_bias[(row, column)];
            }
        }
        let imu_residual = DVector::from_fn(9, |row, _| scale + row as f32 * 0.125);
        let bias_residual = DVector::from_fn(6, |row, _| -scale + row as f32 * 0.03125);
        let imu = WhitenedFactorRowStack::with_objective_cost_kind(
            imu_jacobian.map(f64::from),
            DMatrix::zeros(9, 0),
            imu_residual.map(f64::from),
            0.0,
            FactorKind::Imu,
        )
        .unwrap()
        .with_imu_link_offsets(offsets.start, offsets.end);
        let bias = WhitenedFactorRowStack::with_objective_cost_kind(
            bias_jacobian.map(f64::from),
            DMatrix::zeros(6, 0),
            bias_residual.map(f64::from),
            0.0,
            FactorKind::Bias,
        )
        .unwrap()
        .with_imu_link_offsets(offsets.start, offsets.end);
        (
            imu,
            bias,
            local_imu,
            local_bias,
            imu_residual,
            bias_residual,
        )
    };

    // The landmark row seeds the visual phase. Its second row survives
    // nullspace projection and gives the shared state-1 block a baseline
    // before the two chronological IMU links are folded in.
    let mut visual_state = DMatrix::<f64>::zeros(2, state_dof);
    for column in 0..state_dof {
        visual_state[(1, column)] = 32_768.0 + column as f64 * 0.5;
    }
    let visual = WhitenedFactorRowStack::with_objective_cost_kind(
        visual_state,
        DMatrix::from_row_slice(2, 1, &[1.0, 0.0]),
        DVector::from_row_slice(&[0.0, 1.0]),
        0.0,
        FactorKind::Visual,
    )
    .unwrap();
    let visual_baseline =
        reduce_landmark_factors_f32(std::slice::from_ref(&visual), state_dof, 1e-10);

    let (imu0, bias0, local_imu0, local_bias0, imu_r0, bias_r0) = make_pair(link0_offsets, 0.75);
    let (imu1, bias1, local_imu1, local_bias1, imu_r1, bias_r1) = make_pair(link1_offsets, 1.25);
    let (local_h0, local_b0) = local_imu_h_b_15x30(
        &DMatrix::from_fn(IMU_LOCAL_ROWS, IMU_LOCAL_COLS, |row, column| {
            if row < 9 {
                local_imu0[(row, column)]
            } else {
                local_bias0[(row - 9, column)]
            }
        }),
        &DVector::from_iterator(
            IMU_LOCAL_ROWS,
            imu_r0.iter().copied().chain(bias_r0.iter().copied()),
        ),
    );
    let (local_h1, local_b1) = local_imu_h_b_15x30(
        &DMatrix::from_fn(IMU_LOCAL_ROWS, IMU_LOCAL_COLS, |row, column| {
            if row < 9 {
                local_imu1[(row, column)]
            } else {
                local_bias1[(row - 9, column)]
            }
        }),
        &DVector::from_iterator(
            IMU_LOCAL_ROWS,
            imu_r1.iter().copied().chain(bias_r1.iter().copied()),
        ),
    );

    let reduced =
        reduce_landmark_factors_f32(&[visual, imu0, bias0, imu1, bias1], state_dof, 1e-10);
    let mut imu_h = DMatrix::<f32>::zeros(state_dof, state_dof);
    let mut imu_b = DVector::<f32>::zeros(state_dof);
    scatter_local_imu_h_b_15x30(&mut imu_h, &mut imu_b, &local_h0, &local_b0, link0_offsets)
        .unwrap();
    scatter_local_imu_h_b_15x30(&mut imu_h, &mut imu_b, &local_h1, &local_b1, link1_offsets)
        .unwrap();
    let mut expected_h = visual_baseline.h;
    let mut expected_b = visual_baseline.b;
    expected_h += imu_h.clone();
    expected_b += imu_b.clone();
    assert_eq!(reduced.h, expected_h);
    assert_eq!(reduced.b, expected_b);

    let state1 = AOM_NAV_DOF;
    let mut state1_h = DMatrix::<f32>::zeros(AOM_NAV_DOF, AOM_NAV_DOF);
    let mut state1_b = DVector::<f32>::zeros(AOM_NAV_DOF);
    for row in 0..AOM_NAV_DOF {
        for column in 0..AOM_NAV_DOF {
            state1_h[(row, column)] = local_h0[(AOM_NAV_DOF + row, AOM_NAV_DOF + column)];
        }
        state1_b[row] = local_b0[AOM_NAV_DOF + row];
    }
    state1_h += local_h1.view((0, 0), (AOM_NAV_DOF, AOM_NAV_DOF));
    state1_b += local_b1.rows(0, AOM_NAV_DOF);
    assert_eq!(
        imu_h.view((state1, state1), (AOM_NAV_DOF, AOM_NAV_DOF)),
        state1_h
    );
    assert_eq!(imu_b.rows(state1, AOM_NAV_DOF), state1_b);
}

#[test]
fn model_decrease_includes_landmark_q1_rows() {
    // With no state Jacobian, the reduced camera H/b are both zero, but
    // ABS_QR still recovers a landmark increment and charges its Q1-row
    // model decrease.  This is the smallest fixture for the f4 LM
    // schedule divergence (the omitted term is not a damping tweak).
    let f = factor(&[0., 0.], &[1., 0.], &[1., 0.], 2, 1, 1);
    let reduced = reduce_landmark_factors(std::slice::from_ref(&f), 1, 1e-10);
    assert!(reduced.h[(0, 0)].abs() < 1e-12);
    assert!(reduced.b[0].abs() < 1e-12);
    let decrease = model_cost_decrease(&[f], &DVector::zeros(1), 1e-10).unwrap();
    assert!((decrease - 0.5).abs() < 1e-12);
}

#[test]
fn rank_deficient_landmark_is_reported_and_not_backsolved() {
    let f = factor(&[1., 0., 1., 0.], &[1., 1., 2., 2.], &[1., 2.], 2, 2, 2);
    let red = reduce_landmark_factors(&[f], 2, 1e-8);
    assert_eq!(red.back_substitution[0].rank, 1);
    assert!(
        back_substitute_landmark(&red.back_substitution[0], &DVector::zeros(2), 1e-8).is_none()
    );
}
#[test]
fn row_stack_preserves_block_order_and_multiple_landmarks_accumulate() {
    let a = factor(&[1., 0., 1., 0.], &[1., 0., 0., 1.], &[1., 0.], 2, 2, 2);
    let b = factor(&[0., 1., 1., 1.], &[1., 0., 0., 1.], &[0., 2.], 2, 2, 2);
    let red = reduce_landmark_factors(&[a, b], 2, 1e-10);
    assert_eq!(red.back_substitution.len(), 2);
    assert_eq!(red.h.nrows(), 2);
    assert_eq!(red.b.len(), 2);
}

#[test]
fn visual_double_sphere_factor_has_whitened_two_row_contract() {
    let cam = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
    let pose = SE3::identity();
    let point = Point3::new(0.1, -0.1, 2.0);
    let obs = cam.project(&point).unwrap() + Vector2::new(0.5, 0.0);
    let f = visual_reprojection_factor(&cam, &pose, point, obs, FactorConfig::default()).unwrap();
    assert_eq!(
        (
            f.rows(),
            f.state_jacobian.ncols(),
            f.landmark_jacobian.ncols()
        ),
        (2, 15, 3)
    );
    assert!((f.residual[0] - 1.0).abs() < 1e-10);
    assert!(f.state_jacobian[(0, 0)].is_finite());
}

fn basalt_pose_increment(pose: &SE3, column: usize, amount: f64) -> SE3 {
    let mut result = pose.clone();
    if column < 3 {
        result.translation[column] += amount;
    } else {
        let mut rotation = Vector3::zeros();
        rotation[column - 3] = amount;
        result.rotation = UnitQuaternion::from_scaled_axis(rotation) * result.rotation;
    }
    result
}

#[test]
fn anchored_stereographic_factor_matches_upstream_finite_difference_contract() {
    let camera = DoubleSphereCamera::new(458.2, 457.4, 367.1, 248.2, 0.66, 0.78, 752, 480).unwrap();
    let anchor_pose = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.04, -0.02, 0.03)),
        Vector3::new(0.3, -0.1, 0.2),
    );
    let target_pose = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(-0.03, 0.05, 0.02)),
        Vector3::new(0.55, -0.08, 0.24),
    );
    let anchor_extrinsic = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.01, 0.02, -0.01)),
        Vector3::new(0.04, -0.02, 0.01),
    );
    let target_extrinsic = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(-0.02, 0.01, 0.015)),
        Vector3::new(-0.07, 0.015, 0.005),
    );
    let landmark = InverseDistanceLandmark {
        anchor_pose: 11,
        anchor_camera_id: 0,
        direction: StereographicDirection {
            xy: Point2::new(0.08, -0.045),
        },
        inverse_distance: 0.27,
    };
    let observation = Point2::new(380.0, 235.0);
    let config = FactorConfig {
        observation_stddev: 1.0,
        huber_delta: 0.0,
        outlier_threshold: f64::INFINITY,
    };
    let linearized = anchored_visual_reprojection_factor(
        &camera,
        &anchor_pose,
        &anchor_extrinsic,
        &target_pose,
        &target_extrinsic,
        &landmark,
        observation,
        false,
        config,
    )
    .unwrap();

    let epsilon = 1e-7;
    for column in 0..6 {
        let plus_anchor = basalt_pose_increment(&anchor_pose, column, epsilon);
        let minus_anchor = basalt_pose_increment(&anchor_pose, column, -epsilon);
        let plus = anchored_visual_reprojection_factor(
            &camera,
            &plus_anchor,
            &anchor_extrinsic,
            &target_pose,
            &target_extrinsic,
            &landmark,
            observation,
            false,
            config,
        )
        .unwrap()
        .residual;
        let minus = anchored_visual_reprojection_factor(
            &camera,
            &minus_anchor,
            &anchor_extrinsic,
            &target_pose,
            &target_extrinsic,
            &landmark,
            observation,
            false,
            config,
        )
        .unwrap()
        .residual;
        let numerical = (plus - minus) / (2.0 * epsilon);
        assert!(
            (numerical - linearized.anchor_pose_jacobian.column(column)).norm() < 2e-5,
            "anchor column {column}: numerical={numerical:?}, analytic={:?}",
            linearized.anchor_pose_jacobian.column(column)
        );

        let plus_target = basalt_pose_increment(&target_pose, column, epsilon);
        let minus_target = basalt_pose_increment(&target_pose, column, -epsilon);
        let plus = anchored_visual_reprojection_factor(
            &camera,
            &anchor_pose,
            &anchor_extrinsic,
            &plus_target,
            &target_extrinsic,
            &landmark,
            observation,
            false,
            config,
        )
        .unwrap()
        .residual;
        let minus = anchored_visual_reprojection_factor(
            &camera,
            &anchor_pose,
            &anchor_extrinsic,
            &minus_target,
            &target_extrinsic,
            &landmark,
            observation,
            false,
            config,
        )
        .unwrap()
        .residual;
        let numerical = (plus - minus) / (2.0 * epsilon);
        assert!(
            (numerical - linearized.target_pose_jacobian.column(column)).norm() < 2e-5,
            "target column {column}: numerical={numerical:?}, analytic={:?}",
            linearized.target_pose_jacobian.column(column)
        );
    }

    for column in 0..3 {
        let mut plus_landmark = landmark;
        let mut minus_landmark = landmark;
        if column < 2 {
            plus_landmark.direction.xy[column] += epsilon;
            minus_landmark.direction.xy[column] -= epsilon;
        } else {
            plus_landmark.inverse_distance += epsilon;
            minus_landmark.inverse_distance -= epsilon;
        }
        let plus = anchored_visual_reprojection_factor(
            &camera,
            &anchor_pose,
            &anchor_extrinsic,
            &target_pose,
            &target_extrinsic,
            &plus_landmark,
            observation,
            false,
            config,
        )
        .unwrap()
        .residual;
        let minus = anchored_visual_reprojection_factor(
            &camera,
            &anchor_pose,
            &anchor_extrinsic,
            &target_pose,
            &target_extrinsic,
            &minus_landmark,
            observation,
            false,
            config,
        )
        .unwrap()
        .residual;
        let numerical = (plus - minus) / (2.0 * epsilon);
        assert!(
            (numerical - linearized.landmark_jacobian.column(column)).norm() < 2e-5,
            "landmark column {column}: numerical={numerical:?}, analytic={:?}",
            linearized.landmark_jacobian.column(column)
        );
    }
}

/// A landmark hosted in cam1 of a divergent rig (cam1 = cam0 rotated 75
/// degrees about x, 0.138 m baseline, as Project Aria's SLAM pair), observed
/// by cam1 and by cam0 at another pose.  The analytic host/target pose and
/// landmark Jacobians must match central differences for every
/// host/target camera combination the multi-camera VIO can produce.
#[test]
fn cam1_hosted_factor_matches_finite_difference_on_divergent_rig() {
    let camera = DoubleSphereCamera::new(241.6, 241.6, 379.0, 286.0, 0.0, 0.0, 758, 572).unwrap();
    let extrinsics = [
        SE3::identity(),
        SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(75_f64.to_radians(), 0.0, 0.0)),
            Vector3::new(0.004, -0.109, -0.085),
        ),
    ];
    let anchor_pose = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.05, -0.03, 0.2)),
        Vector3::new(0.4, -0.2, 0.1),
    );
    let target_pose = SE3::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(-0.02, 0.06, 0.25)),
        Vector3::new(0.65, -0.1, 0.18),
    );
    // In the band both cameras share: cam1 sees it low in its image.
    let point_host = Vector3::new(0.1, 1.5, 2.0);
    let landmark = InverseDistanceLandmark {
        anchor_pose: 3,
        anchor_camera_id: 1,
        direction: StereographicDirection::from_bearing(point_host.normalize()).unwrap(),
        inverse_distance: 1.0 / point_host.norm(),
    };
    let config = FactorConfig {
        observation_stddev: 1.0,
        huber_delta: 0.0,
        outlier_threshold: f64::INFINITY,
    };
    let point_world = anchor_pose
        .compose(&extrinsics[1])
        .transform_point(&Point3::from(point_host));
    for target_camera in [1_usize, 0] {
        let truth = camera
            .project(
                &target_pose
                    .compose(&extrinsics[target_camera])
                    .inverse()
                    .transform_point(&point_world),
            )
            .unwrap();
        assert!(camera.contains_pixel(&truth), "target cam{target_camera}");
        let factor = |anchor: &SE3, target: &SE3, landmark: &InverseDistanceLandmark, pixel| {
            anchored_visual_reprojection_factor(
                &camera,
                anchor,
                &extrinsics[1],
                target,
                &extrinsics[target_camera],
                landmark,
                pixel,
                false,
                config,
            )
            .unwrap()
        };
        // The exact projection gives a zero residual: the host extrinsic is
        // T_imu_cam1, not cam0's.
        assert!(
            factor(&anchor_pose, &target_pose, &landmark, truth)
                .residual
                .norm()
                < 1e-8
        );
        let observation = truth + Vector2::new(1.5, -0.8);
        let linearized = factor(&anchor_pose, &target_pose, &landmark, observation);
        let epsilon = 1e-7;
        for column in 0..6 {
            let numerical = (factor(
                &basalt_pose_increment(&anchor_pose, column, epsilon),
                &target_pose,
                &landmark,
                observation,
            )
            .residual
                - factor(
                    &basalt_pose_increment(&anchor_pose, column, -epsilon),
                    &target_pose,
                    &landmark,
                    observation,
                )
                .residual)
                / (2.0 * epsilon);
            assert!(
                (numerical - linearized.anchor_pose_jacobian.column(column)).norm() < 2e-5,
                "cam{target_camera} anchor column {column}: {numerical:?} vs {:?}",
                linearized.anchor_pose_jacobian.column(column)
            );
            let numerical = (factor(
                &anchor_pose,
                &basalt_pose_increment(&target_pose, column, epsilon),
                &landmark,
                observation,
            )
            .residual
                - factor(
                    &anchor_pose,
                    &basalt_pose_increment(&target_pose, column, -epsilon),
                    &landmark,
                    observation,
                )
                .residual)
                / (2.0 * epsilon);
            assert!(
                (numerical - linearized.target_pose_jacobian.column(column)).norm() < 2e-5,
                "cam{target_camera} target column {column}: {numerical:?} vs {:?}",
                linearized.target_pose_jacobian.column(column)
            );
        }
        for column in 0..3 {
            let mut plus = landmark;
            let mut minus = landmark;
            if column < 2 {
                plus.direction.xy[column] += epsilon;
                minus.direction.xy[column] -= epsilon;
            } else {
                plus.inverse_distance += epsilon;
                minus.inverse_distance -= epsilon;
            }
            let numerical = (factor(&anchor_pose, &target_pose, &plus, observation).residual
                - factor(&anchor_pose, &target_pose, &minus, observation).residual)
                / (2.0 * epsilon);
            assert!(
                (numerical - linearized.landmark_jacobian.column(column)).norm() < 2e-5,
                "cam{target_camera} landmark column {column}: {numerical:?} vs {:?}",
                linearized.landmark_jacobian.column(column)
            );
        }
        // The f32 production factor agrees with the f64 reference.
        let f32_factor = anchored_visual_reprojection_factor_f32(
            &camera,
            &anchor_pose,
            &extrinsics[1],
            &target_pose,
            &extrinsics[target_camera],
            &landmark,
            observation,
            false,
            config,
        )
        .unwrap();
        assert!((f32_factor.residual - linearized.residual).norm() < 1e-2);
        assert!(
            (f32_factor.anchor_pose_jacobian - linearized.anchor_pose_jacobian).norm()
                < 1e-3 * linearized.anchor_pose_jacobian.norm()
        );
        assert!(
            (f32_factor.target_pose_jacobian - linearized.target_pose_jacobian).norm()
                < 1e-3 * linearized.target_pose_jacobian.norm()
        );
        assert!(
            (f32_factor.landmark_jacobian - linearized.landmark_jacobian).norm()
                < 1e-3 * linearized.landmark_jacobian.norm()
        );
    }
}

#[test]
fn huber_and_outlier_gates_are_explicit_and_deterministic() {
    let cam = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
    let p = Point3::new(0.0, 0.0, 2.0);
    let nominal = cam.project(&p).unwrap();
    let huber = visual_reprojection_factor(
        &cam,
        &SE3::identity(),
        p,
        nominal + Vector2::new(1.0, 0.0),
        FactorConfig::default(),
    )
    .unwrap();
    assert!((huber.residual[0] - 2.0_f64.sqrt()).abs() < 1e-10);
    assert!(visual_reprojection_factor(
        &cam,
        &SE3::identity(),
        p,
        nominal + Vector2::new(10.0, 0.0),
        FactorConfig::default()
    )
    .is_none());
}

#[test]
fn stereo_and_auxiliary_factor_row_ordering_is_fixed() {
    let cam = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
    let p = Point3::new(0.1, 0.0, 2.0);
    let o = cam.project(&p).unwrap();
    let right_pose = SE3::new(
        nalgebra::UnitQuaternion::identity(),
        Vector3::new(0.2, 0.0, 0.0),
    );
    let right_o = cam
        .project(&right_pose.inverse().transform_point(&p))
        .unwrap();
    let s = stereo_reprojection_factor(
        &cam,
        &SE3::identity(),
        &cam,
        &right_pose,
        p,
        (o, right_o),
        FactorConfig::default(),
    )
    .unwrap();
    assert_eq!(
        (
            s.rows(),
            s.state_jacobian.ncols(),
            s.landmark_jacobian.ncols()
        ),
        (4, 15, 3)
    );
    let rw = bias_random_walk_factor(Vector3::new(1.0, 2.0, 3.0), Vector3::zeros(), 2.0).unwrap();
    assert_eq!(rw.rows(), 6);
}

struct Quadratic {
    target: f64,
    reject: bool,
}
impl LmProblem for Quadratic {
    fn linearize(&self, state: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
        let r = if self.reject {
            1.0
        } else {
            state[0] - self.target
        };
        let j = DMatrix::from_element(1, 1, 1.0);
        Ok(LmLinearization {
            factors: vec![WhitenedFactorRowStack::new(
                j,
                DMatrix::zeros(1, 0),
                DVector::from_element(1, r),
            )
            .unwrap()],
            cost: r * r,
        })
    }
    fn cost(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
        if self.reject {
            Ok(1.0)
        } else {
            Ok((state[0] - self.target).powi(2))
        }
    }
}

struct UpstreamF32Quadratic {
    target: f64,
    diagnostic_events: usize,
    diagnostic_patches: Cell<usize>,
}

impl LmProblem for UpstreamF32Quadratic {
    fn linearize(&self, state: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
        let residual = state[0] - self.target;
        Ok(LmLinearization {
            factors: vec![WhitenedFactorRowStack::new(
                DMatrix::from_element(1, 1, 1.0),
                DMatrix::zeros(1, 0),
                DVector::from_element(1, residual),
            )
            .unwrap()],
            cost: residual * residual,
        })
    }

    fn cost(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
        Ok((state[0] - self.target).powi(2))
    }

    fn scalar_mode(&self) -> ScalarMode {
        ScalarMode::UpstreamF32
    }

    fn diagnostic_lm_event(&mut self, _event: LmDiagnosticEvent<'_>) {
        self.diagnostic_events += 1;
    }

    fn diagnostic_patch_reduced_f32(
        &self,
        _iteration: usize,
        _h: &mut DMatrix<f32>,
        _b: &mut DVector<f32>,
    ) {
        self.diagnostic_patches
            .set(self.diagnostic_patches.get() + 1);
    }
}

#[test]
fn lm_trial_preparation_default_hook_preserves_legacy_token_result() {
    let problem = Quadratic {
        target: 2.0,
        reject: false,
    };
    let state = DVector::from_element(1, 0.5);
    let step = DVector::from_element(1, 0.25);
    let trial = problem.apply_step(&state, &step);
    let preparation = LmTrialPreparation {
        landmark_steps: vec![LmPreparedLandmarkStep {
            landmark_index: 7,
            track_id: 99,
            step: None,
        }],
        tolerance_bits: 1e-10_f64.to_bits(),
        state_fingerprint: 0,
        step_fingerprint: 0,
    };
    let mut prepared_timing = TimingBreakdown::default();
    let prepared = problem
        .trial_cost_timed_with_preparation(
            &state,
            &step,
            &trial,
            Some(preparation),
            &mut prepared_timing,
        )
        .expect("default preparation hook");
    let mut legacy_timing = TimingBreakdown::default();
    let legacy = problem
        .trial_cost_timed_with_token(&state, &step, &trial, &mut legacy_timing)
        .expect("legacy token hook");
    assert_eq!(prepared.0, legacy.0);
    assert!(prepared.1.landmark_steps.is_none());
    assert!(legacy.1.landmark_steps.is_none());
}

struct MalformedUpstreamF32;

impl LmProblem for MalformedUpstreamF32 {
    fn linearize(&self, _: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
        let factor = WhitenedFactorRowStack::with_objective_cost_kind(
            DMatrix::zeros(6, 1),
            DMatrix::zeros(6, 0),
            DVector::from_element(6, 1.0),
            1.0,
            FactorKind::Bias,
        )
        .unwrap();
        Ok(LmLinearization {
            factors: vec![factor],
            cost: 1.0,
        })
    }

    fn cost(&self, _: &DVector<f64>) -> Result<f64, LmFailure> {
        Ok(1.0)
    }

    fn scalar_mode(&self) -> ScalarMode {
        ScalarMode::UpstreamF32
    }
}

#[test]
fn lm_upstream_f32_clean_matches_retained_diagnostics_exactly() {
    let mut retained = UpstreamF32Quadratic {
        target: 3.0,
        diagnostic_events: 0,
        diagnostic_patches: Cell::new(0),
    };
    let mut clean = UpstreamF32Quadratic {
        target: 3.0,
        diagnostic_events: 0,
        diagnostic_patches: Cell::new(0),
    };
    let mut retained_timing = TimingBreakdown::default();
    let mut clean_timing = TimingBreakdown::default();
    let initial = DVector::from_element(1, 0.0);
    let config = LmConfig::default();

    let retained_result = solve_lm_with_timing(
        &mut retained,
        initial.clone(),
        config,
        true,
        true,
        &mut retained_timing,
    )
    .unwrap();
    let clean_result =
        solve_lm_with_timing(&mut clean, initial, config, true, false, &mut clean_timing).unwrap();

    assert_eq!(clean_result.state, retained_result.state);
    assert_eq!(clean_result.cost, retained_result.cost);
    assert_eq!(clean_result.lambda, retained_result.lambda);
    assert_eq!(clean_result.iterations, retained_result.iterations);
    assert_eq!(clean_result.trace, retained_result.trace);
    assert!(retained.diagnostic_events > 0);
    assert_eq!(clean.diagnostic_events, 0);
    assert!(retained.diagnostic_patches.get() > 0);
    assert_eq!(clean.diagnostic_patches.get(), 0);
    assert_eq!(clean_timing.lm_normal_system_prep.count, 0);
    assert_eq!(
        clean_timing.lm_landmark_reduction.count,
        retained_timing.lm_landmark_reduction.count
    );
}

#[test]
fn lm_f32_small_model_decrease_survives_large_cost_offset() {
    // Synthetic trial oracle isolates the decision arithmetic: the model
    // decrease is below one cost ULP, but the actual trial decreases.
    struct SmallDecrease;
    impl LmProblem for SmallDecrease {
        fn scalar_mode(&self) -> ScalarMode {
            ScalarMode::UpstreamF32
        }
        fn linearize(&self, _: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
            Ok(LmLinearization {
                factors: vec![WhitenedFactorRowStack::new(
                    DMatrix::from_element(1, 1, 1.0),
                    DMatrix::zeros(1, 0),
                    DVector::from_element(1, 0.01),
                )
                .unwrap()],
                cost: -15000.0,
            })
        }
        fn cost(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
            Ok(if state[0] == 0.0 {
                -15000.0
            } else {
                -15000.0009765625
            })
        }
    }
    for diagnostics in [false, true] {
        let result = solve_lm_with_timing(
            &mut SmallDecrease,
            DVector::zeros(1),
            LmConfig {
                max_iterations: 0,
                convergence_step: 0.0,
                ..LmConfig::default()
            },
            true,
            diagnostics,
            &mut TimingBreakdown::default(),
        )
        .unwrap();
        assert_eq!(result.trace.len(), 1);
        let entry = &result.trace[0];
        // The old cost-model reconstruction yields zero in this case.
        assert_eq!(entry.cost_before as f32, entry.model_cost as f32);
        assert_eq!(entry.decision, LmDecision::Accepted);
        assert!(result.state[0] < 0.0);
        assert!(entry.lambda_after < entry.lambda_before);
    }
}

#[test]
fn lm_f32_refreshes_before_cost_from_each_linearization() {
    struct DistinctSchedules;
    impl LmProblem for DistinctSchedules {
        fn scalar_mode(&self) -> ScalarMode {
            ScalarMode::UpstreamF32
        }
        fn linearize(&self, _: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
            Ok(LmLinearization {
                factors: vec![WhitenedFactorRowStack::new(
                    DMatrix::from_element(1, 1, 1.0),
                    DMatrix::zeros(1, 0),
                    DVector::from_element(1, 0.1),
                )
                .unwrap()],
                cost: 101.0,
            })
        }
        fn cost(&self, _: &DVector<f64>) -> Result<f64, LmFailure> {
            Ok(100.0)
        }
    }
    for diagnostics in [false, true] {
        let result = solve_lm_with_timing(
            &mut DistinctSchedules,
            DVector::zeros(1),
            LmConfig {
                max_iterations: 1,
                convergence_step: 0.0,
                ..LmConfig::default()
            },
            true,
            diagnostics,
            &mut TimingBreakdown::default(),
        )
        .unwrap();
        assert_eq!(result.trace.len(), 2);
        for entry in &result.trace {
            assert_eq!(entry.cost_before, 101.0);
            assert_eq!(entry.decision, LmDecision::Accepted);
        }
        assert_eq!(result.cost, 100.0);
        assert!(result.state[0] < -0.1);
    }
}

#[test]
fn lm_upstream_f32_clean_preserves_malformed_and_config_errors() {
    let mut retained = MalformedUpstreamF32;
    let mut clean = MalformedUpstreamF32;
    let mut retained_timing = TimingBreakdown::default();
    let mut clean_timing = TimingBreakdown::default();
    let initial = DVector::zeros(1);

    assert_eq!(
        solve_lm_with_timing(
            &mut retained,
            initial.clone(),
            LmConfig::default(),
            true,
            true,
            &mut retained_timing,
        ),
        Err(LmFailure::LinearSolve)
    );
    assert_eq!(
        solve_lm_with_timing(
            &mut clean,
            initial,
            LmConfig::default(),
            true,
            false,
            &mut clean_timing,
        ),
        Err(LmFailure::LinearSolve)
    );

    let mut retained = UpstreamF32Quadratic {
        target: 0.0,
        diagnostic_events: 0,
        diagnostic_patches: Cell::new(0),
    };
    let mut clean = UpstreamF32Quadratic {
        target: 0.0,
        diagnostic_events: 0,
        diagnostic_patches: Cell::new(0),
    };
    let invalid = LmConfig {
        lambda_initial: 0.0,
        ..LmConfig::default()
    };
    assert_eq!(
        solve_lm_with_timing(
            &mut retained,
            DVector::zeros(1),
            invalid,
            true,
            true,
            &mut TimingBreakdown::default(),
        ),
        Err(LmFailure::NonFinite)
    );
    assert_eq!(
        solve_lm_with_timing(
            &mut clean,
            DVector::zeros(1),
            invalid,
            true,
            false,
            &mut TimingBreakdown::default(),
        ),
        Err(LmFailure::NonFinite)
    );
}

#[test]
fn lm_quadratic_accepts_and_follows_lambda_trace() {
    let mut problem = Quadratic {
        target: 3.0,
        reject: false,
    };
    let result = solve_lm(
        &mut problem,
        DVector::from_element(1, 0.0),
        LmConfig::default(),
    )
    .unwrap();
    assert!((result.state[0] - 3.0).abs() < 1e-6);
    assert!(result.cost < 1e-10);
    assert_eq!(result.trace[0].lambda_before, 1e-4);
    assert!(result
        .trace
        .iter()
        .any(|x| x.decision == LmDecision::Accepted));
    assert!(result.trace.len() <= 8);
}

#[test]
fn lm_trace_suppression_preserves_solution_exactly() {
    let mut retained_problem = Quadratic {
        target: 3.0,
        reject: false,
    };
    let mut lean_problem = Quadratic {
        target: 3.0,
        reject: false,
    };
    let retained = solve_lm(
        &mut retained_problem,
        DVector::from_element(1, 0.0),
        LmConfig::default(),
    )
    .unwrap();
    let lean = solve_lm_without_trace(
        &mut lean_problem,
        DVector::from_element(1, 0.0),
        LmConfig::default(),
    )
    .unwrap();

    assert_eq!(lean.state, retained.state);
    assert_eq!(lean.cost, retained.cost);
    assert_eq!(lean.lambda, retained.lambda);
    assert_eq!(lean.iterations, retained.iterations);
    assert!(lean.trace.is_empty());
    assert!(!retained.trace.is_empty());
}

#[test]
fn lm_reject_restores_state_and_increases_lambda() {
    let mut problem = Quadratic {
        target: 3.0,
        reject: true,
    };
    let result = solve_lm(
        &mut problem,
        DVector::from_element(1, 2.0),
        LmConfig {
            max_iterations: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.state[0], 2.0);
    assert_eq!(result.cost, 1.0);
    assert_eq!(result.trace[0].decision, LmDecision::Rejected);
    assert_eq!(result.trace[0].lambda_after, 2e-4);
    assert_eq!(result.trace.len(), 3);
    assert_eq!(result.trace[1].lambda_after, 8e-4);
    assert_eq!(result.trace[2].lambda_after, 6.4e-3);
}
struct NaNProblem;
impl LmProblem for NaNProblem {
    fn linearize(&self, _: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
        Err(LmFailure::NonFinite)
    }
    fn cost(&self, _: &DVector<f64>) -> Result<f64, LmFailure> {
        Ok(f64::NAN)
    }
}
#[test]
fn lm_rejects_nan_and_invalid_lambda_configuration() {
    let mut nan_problem = NaNProblem;
    assert_eq!(
        solve_lm(&mut nan_problem, DVector::zeros(1), LmConfig::default()),
        Err(LmFailure::NonFinite)
    );
    let mut quadratic = Quadratic {
        target: 0.0,
        reject: false,
    };
    assert_eq!(
        solve_lm(
            &mut quadratic,
            DVector::zeros(1),
            LmConfig {
                lambda_initial: 0.0,
                ..Default::default()
            }
        ),
        Err(LmFailure::NonFinite)
    );
}

#[test]
fn m7_landmark_jacobian_intermediates_match_pinned_lanes() {
    let (rotation, translation, _, _) = m7_fixed_relative_point();
    let direction = StereographicDirection {
        xy: Point2::new(-0.39586612582206726, -0.1799013614654541),
    };
    let direction_jacobian = direction.bearing_jacobian_f32();
    assert_f32_bits(
        direction_jacobian.as_slice(),
        &[
            0x3f9e8bb7, 0xbe4e4fe2, 0x3f8f59cf, 0xbe4e4fe2, 0x3fcb92dc, 0x3f024aa3,
        ],
    );

    let rotated_direction_jacobian = eigen_quaternion_matrix_f32(rotation) * direction_jacobian;
    assert_f32_bits(
        rotated_direction_jacobian.as_slice(),
        &[
            0x3f9d9adf, 0xbe3b8c06, 0x3f90c8ab, 0xbe4fefe0, 0x3fccb288, 0x3ef5c0e4,
        ],
    );

    let mut point_wrt_landmark = SMatrix::<f32, 3, 3>::zeros();
    point_wrt_landmark
        .fixed_view_mut::<3, 2>(0, 0)
        .copy_from(&rotated_direction_jacobian);
    point_wrt_landmark.set_column(2, &translation);
    assert_f32_bits(
        point_wrt_landmark.as_slice(),
        &[
            0x3f9d9adf, 0xbe3b8c06, 0x3f90c8ab, 0xbe4fefe0, 0x3fccb288, 0x3ef5c0e4, 0xbde1e254,
            0xbaf1ecc0, 0xba884364,
        ],
    );

    let point = Vector3::new(
        f32::from_bits(0xbf2fc73d),
        f32::from_bits(0xbe94aaaa),
        f32::from_bits(0x3f2ec6b2),
    );
    let (_, projection_jacobian) =
        project_double_sphere_with_jacobian_f32(&m7_fixed_camera(), point).unwrap();
    let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
    camera_jacobian
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&projection_jacobian);
    let mut homogeneous_point_wrt_landmark = SMatrix::<f32, 4, 3>::zeros();
    homogeneous_point_wrt_landmark
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&point_wrt_landmark);
    let landmark_jacobian =
        eigen_landmark_jacobian_f32(camera_jacobian, homogeneous_point_wrt_landmark);
    assert_f32_bits(
        landmark_jacobian.as_slice(),
        &[
            0x44446eae, 0xc1e49388, 0xc20f4748, 0x445399f4, 0xc21720bd, 0x40df22e2,
        ],
    );
}

#[test]
fn m7_same_timestamp_stereo_homogeneous_jacobian_matches_pinned_lanes() {
    // This is a camera-agnostic check of the exact fixed-size source
    // product.  The fourth column/row is homogeneous zero, but the native
    // 3x4 * 4x2 reduction still owns the f32 association.
    let transform_top_left = SMatrix::<f32, 3, 4>::from_column_slice(&[
        f32::from_bits(0x3f7fffd3),
        f32::from_bits(0xbb0b904f),
        f32::from_bits(0x3a6ef290),
        f32::from_bits(0x3b0c6c2f),
        f32::from_bits(0x3f7ff8d7),
        f32::from_bits(0xbc6fa4f8),
        f32::from_bits(0xba66c186),
        f32::from_bits(0x3c6facff),
        f32::from_bits(0x3f7ff8f6),
        0.0,
        0.0,
        0.0,
    ]);
    let source_jup = SMatrix::<f32, 4, 2>::from_column_slice(&[
        f32::from_bits(0x3f9e8bb7),
        f32::from_bits(0xbe4e4fe2),
        f32::from_bits(0x3f8f59cf),
        0.0,
        f32::from_bits(0xbe4e4fe2),
        f32::from_bits(0x3fcb92dc),
        f32::from_bits(0x3f024aa3),
        0.0,
    ]);
    let source_jpp = eigen_homogeneous_landmark_direction_f32(transform_top_left, source_jup);
    assert_f32_bits(
        source_jpp.as_slice(),
        &[
            0x3f9e5d28, 0xbe4036e0, 0x3f8fdb6e, 0xbe4b47dd, 0x3fcc8f31, 0x3ef88cf5,
        ],
    );

    // Exercise the production factor with a same-timestamp stereo row;
    // no track or camera-id-specific branch is involved here.
    let camera = m7_fixed_camera();
    let frame_pose = SE3::new(
        UnitQuaternion::new_unchecked(Quaternion::new(
            0.5944822430610657,
            -0.052778493613004684,
            -0.8023747801780701,
            0.0,
        )),
        Vector3::zeros(),
    );
    let anchor_extrinsic = SE3::new(
        UnitQuaternion::new_normalize(Quaternion::new(
            0.7123125505904486,
            -0.007239825785317818,
            0.007541278561558601,
            0.7017845426564943,
        )),
        Vector3::new(
            -0.016774788924641534,
            -0.068938940687127,
            0.005139123188382424,
        ),
    );
    let target_extrinsic = SE3::new(
        UnitQuaternion::new_normalize(Quaternion::new(
            0.7115930283929829,
            -0.0023360576185881625,
            0.013000769689092388,
            0.7024677108343111,
        )),
        Vector3::new(
            -0.01507436282032619,
            0.0412627204046637,
            0.00316287258752953,
        ),
    );
    let landmark = InverseDistanceLandmark {
        anchor_pose: 0,
        anchor_camera_id: 0,
        direction: StereographicDirection {
            xy: Point2::new(-0.39586612582206726, -0.1799013614654541),
        },
        inverse_distance: 0.14654140174388885,
    };
    let factor = anchored_visual_reprojection_factor_f32_with_time_cam(
        &camera,
        &frame_pose,
        &anchor_extrinsic,
        &frame_pose,
        &target_extrinsic,
        &landmark,
        Point2::new(29.425615310668945, 107.3565444946289),
        true,
        false,
        FactorConfig::default(),
    )
    .unwrap();
    assert_f32_bits(
        factor.projection.map(|value| value as f32).as_slice(),
        &[0x41eb7793, 0x42d69bf8],
    );
    assert_f32_bits(
        factor
            .landmark_jacobian
            .map(|value| value as f32)
            .as_slice(),
        &[
            0x44c477bc, 0xc279df70, 0xc2840b88, 0x44d35468, 0xc2978b08, 0x4180a392,
        ],
    );
    assert!(factor.anchor_pose_jacobian.norm() > 0.0);
    assert!(factor.target_pose_jacobian.norm() > 0.0);
}

#[test]
fn sqrt_marginalization_matches_dense_factor_gram_and_gradient() {
    let j = DMatrix::from_row_slice(
        5,
        4,
        &[
            1., 0., 2., 0., 0., 1., 1., 1., 1., 1., 0., 2., 2., -1., 1., 0., 0.5, 2., 1., -1.,
        ],
    );
    let r = DVector::from_column_slice(&[1., 2., -1., 0.5, 3.]);
    let prior = sqrt_to_sqrt_marginalize(&j, &r, &[0, 2], &[1, 3], DVector::zeros(2)).unwrap();
    let h = &prior.jacobian.transpose() * &prior.jacobian;
    let b = &prior.jacobian.transpose() * &prior.rhs;
    let jk = j.select_columns(&[0, 2]);
    let jm = j.select_columns(&[1, 3]);
    let inv = (&jm.transpose() * &jm).try_inverse().unwrap();
    let expected_h = jk.transpose() * &jk - jk.transpose() * &jm * &inv * jm.transpose() * &jk;
    let expected_b = jk.transpose() * &r - jk.transpose() * &jm * &inv * jm.transpose() * &r;
    assert!((&h - expected_h).norm() < 1e-9);
    assert!((&b - expected_b).norm() < 1e-9);
}

#[test]
fn mixed_pose_only_and_nav_columns_survive_two_window_shifts() {
    let j = DMatrix::from_fn(30, 21, |row, col| {
        (((row + 1) * (col + 2)) % 17) as f64 + 0.1
    });
    let r = DVector::from_element(30, 1.0);
    let p1 = sqrt_to_sqrt_marginalize(
        &j,
        &r,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        &(15..21).collect::<Vec<_>>(),
        DVector::zeros(15),
    )
    .unwrap();
    let p2 = sqrt_to_sqrt_marginalize(
        &p1.jacobian,
        &p1.rhs,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        &(10..15).collect::<Vec<_>>(),
        DVector::zeros(10),
    )
    .unwrap();
    assert_eq!(p2.jacobian.ncols(), 10);
    assert_eq!(p2.fej_point.len(), 10);
    assert!(p2.jacobian.iter().all(|x| x.is_finite()));
}

#[test]
fn fej_re_reference_updates_rhs_without_changing_jacobian() {
    let j = DMatrix::from_row_slice(2, 2, &[2., 0., 0., 3.]);
    let r = DVector::from_column_slice(&[1., 2.]);
    let mut p = SqrtPrior {
        jacobian: j.clone(),
        rhs: r,
        fej_point: DVector::zeros(2),
    };
    p.re_reference(&DVector::from_column_slice(&[0.5, -1.0]))
        .unwrap();
    assert_eq!(p.jacobian, j);
    assert_eq!(p.rhs, DVector::from_column_slice(&[0., 5.]));
}

#[test]
fn marginalization_reports_invalid_columns_and_gauge_rank() {
    let j = DMatrix::zeros(3, 2);
    let r = DVector::zeros(3);
    assert_eq!(
        sqrt_to_sqrt_marginalize(&j, &r, &[0], &[1], DVector::zeros(1)),
        Err(MarginalizationError::RankDeficient)
    );
    assert_eq!(
        sqrt_to_sqrt_marginalize(&j, &r, &[2], &[], DVector::zeros(1)),
        Err(MarginalizationError::InvalidColumns)
    );
}

#[test]
#[ignore]
fn m7_tmp_probe_clean_visual_reduction_trees() {
    // This is a one-shot source-order probe for the frame-4 clean oracle.
    // Keep it ignored: the large capture files are workspace artifacts,
    // not checked-in fixtures.  It is intentionally in this module so it
    // can reuse the exact f32 helpers used by the production reducer.
    fn bits(value: &serde_json::Value) -> u32 {
        value
            .as_str()
            .unwrap_or_else(|| panic!("expected f32 hex string, got {value}"))
            .parse::<u32>()
            .unwrap_or_else(|_| u32::from_str_radix(value.as_str().unwrap(), 16).unwrap())
    }
    fn bit_words(value: &serde_json::Value) -> Vec<u32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| u32::from_str_radix(entry.as_str().unwrap(), 16).unwrap())
            .collect()
    }
    fn actual_f32(value: &serde_json::Value) -> f32 {
        value.as_f64().unwrap() as f32
    }
    fn add_tree(parts: &[Vec<f32>], begin: usize, end: usize, leaf: usize) -> Vec<f32> {
        if end - begin <= leaf {
            let mut result = vec![0.0_f32; parts[0].len()];
            for part in &parts[begin..end] {
                for (dst, src) in result.iter_mut().zip(part) {
                    *dst += *src;
                }
            }
            return result;
        }
        let middle = begin + (end - begin) / 2;
        let left = add_tree(parts, begin, middle, leaf);
        let right = add_tree(parts, middle, end, leaf);
        left.into_iter()
            .zip(right)
            .map(|(left, right)| left + right)
            .collect()
    }
    fn count_equal(actual: &[f32], expected: &[u32]) -> usize {
        actual
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual.to_bits() == **expected)
            .count()
    }
    fn count_equal_bits(actual: &[f32], expected: &[f32]) -> usize {
        actual
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual.to_bits() == expected.to_bits())
            .count()
    }
    fn col_major_to_row_major(words: &[u32], width: usize) -> Vec<f32> {
        (0..width)
            .flat_map(|row| {
                (0..width).map(move |column| f32::from_bits(words[column * width + row]))
            })
            .collect()
    }
    fn first_mismatch(actual: &[f32], expected: &[u32], width: usize) -> String {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            if actual.to_bits() != *expected {
                return format!(
                    "idx={} row={} col={} actual={:08x} expected={expected:08x}",
                    index,
                    index / width,
                    index % width,
                    actual.to_bits(),
                );
            }
        }
        "none".to_owned()
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let detail_text = std::fs::read_to_string(
        root.join("../../target/m7im15_rust_detail_cleancompare_frame4_20260826.jsonl"),
    )
    .unwrap();
    let detail: serde_json::Value = detail_text
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|record| record["phase"] == "iteration_start" && record["iteration"] == 0)
        .expect("iteration_start detail record");
    let frontier_text = std::fs::read_to_string(
        root.join("../../target/m7im15_rust_frontier_cleancompare_frame4_20260826.jsonl"),
    )
    .unwrap();
    let frontier: serde_json::Value = frontier_text
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|record| record["phase"] == "iteration_start" && record["iteration"] == 0)
        .expect("iteration_start frontier record");
    let imu_text = std::fs::read_to_string(
        root.join("../../target/m7im15_rust_imu_cleancompare_frame4_20260826.jsonl"),
    )
    .unwrap();
    let imu: serde_json::Value = imu_text
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|record| record["iteration"] == 0)
        .expect("iteration-0 IMU record");
    let native: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            root.join("../../target/m7im15_native_step_clean_frame4_repeat_20260826.json"),
        )
        .unwrap(),
    )
    .unwrap();

    let mut factors = detail["landmark_factors"]
        .as_array()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    factors.sort_by_key(|factor| factor["row_span"][0].as_u64().unwrap());
    assert_eq!(factors.len(), 61);
    let mut h_parts = Vec::with_capacity(factors.len());
    let mut b_parts = Vec::with_capacity(factors.len());
    for factor in factors {
        let rows = factor["reduced_rows"].as_array().unwrap();
        let rhs = factor["reduced_rhs"].as_array().unwrap();
        let rows = rows
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(actual_f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rhs = rhs.iter().map(actual_f32).collect::<Vec<_>>();
        let mut h = vec![0.0_f32; 75 * 75];
        for row in 0..75 {
            for column in 0..75 {
                let mut value = 0.0_f32;
                for depth in &rows {
                    value = depth[row].mul_add(depth[column], value);
                }
                h[row * 75 + column] = value;
            }
        }
        let mut b = vec![0.0_f32; 75];
        for column in 0..75 {
            for (row, depth) in rows.iter().enumerate() {
                b[column] = depth[column].mul_add(rhs[row], b[column]);
            }
        }
        h_parts.push(h);
        b_parts.push(b);
    }

    let native_record = &native["records"][0];
    let native_h = bit_words(&native_record["dense"]["H"]["f32_bits"]);
    let native_b = bit_words(&native_record["dense"]["b_f32_bits"]);
    assert_eq!(native_h.len(), 75 * 75);
    assert_eq!(native_b.len(), 75);

    let old_visual_h_words =
        bit_words(&frontier["normal_system_stages"]["visual_accumulator"]["h"]["bits"]);
    let old_visual_b_words =
        bit_words(&frontier["normal_system_stages"]["visual_accumulator"]["b"]);
    let flat_h = add_tree(&h_parts, 0, h_parts.len(), h_parts.len());
    let flat_b = add_tree(&b_parts, 0, b_parts.len(), b_parts.len());
    let old_visual_h = col_major_to_row_major(&old_visual_h_words, 75);
    assert_eq!(count_equal_bits(&flat_h, &old_visual_h), 75 * 75);
    assert_eq!(count_equal(&flat_b, &old_visual_b_words), 75);

    let imu_h_words = bit_words(&imu["imu_accumulator"]["h"]["bits"]);
    let imu_b_words = bit_words(&imu["imu_accumulator"]["b"]["bits"]);
    let prior_h_words = bit_words(&frontier["normal_system_stages"]["marginal_prior"]["h"]["bits"]);
    let prior_b_words = bit_words(&frontier["normal_system_stages"]["marginal_prior"]["b"]);
    let imu_h = col_major_to_row_major(&imu_h_words, 75);
    let prior_h = col_major_to_row_major(&prior_h_words, 75);
    let capture_root = root.join("../../target/m7im15_gdb_clean_landmark_capture_20260826");
    let read_capture = |ordinal: usize, field: &str| {
        std::fs::read(capture_root.join(format!("{ordinal:03}.{field}_after.bin")))
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let mut cumulative_h = vec![0.0_f32; 75 * 75];
    let mut cumulative_b = vec![0.0_f32; 75];
    for ordinal in 0..h_parts.len() {
        for index in 0..cumulative_h.len() {
            cumulative_h[index] += h_parts[ordinal][index];
        }
        for index in 0..cumulative_b.len() {
            cumulative_b[index] += b_parts[ordinal][index];
        }
        let native_h_col_major = read_capture(ordinal, "h");
        let native_h = (0..75)
            .flat_map(|row| {
                let value = &native_h_col_major;
                (0..75).map(move |column| value[column * 75 + row])
            })
            .collect::<Vec<_>>();
        let native_b = read_capture(ordinal, "b");
        println!(
            "ordinal={ordinal} h={} b={} first_h={} first_b={}",
            count_equal(&cumulative_h, &native_h),
            count_equal(&cumulative_b, &native_b),
            first_mismatch(&cumulative_h, &native_h, 75),
            first_mismatch(&cumulative_b, &native_b, 75),
        );
        if ordinal == 0 {
            let row_counts = (0..75)
                .map(|row| {
                    (0..75)
                        .filter(|column| {
                            cumulative_h[row * 75 + column].to_bits() != native_h[row * 75 + column]
                        })
                        .count()
                })
                .collect::<Vec<_>>();
            let col_counts = (0..75)
                .map(|column| {
                    (0..75)
                        .filter(|row| {
                            cumulative_h[row * 75 + column].to_bits() != native_h[row * 75 + column]
                        })
                        .count()
                })
                .collect::<Vec<_>>();
            println!("ordinal=0 row_mismatches={row_counts:?}");
            println!("ordinal=0 col_mismatches={col_counts:?}");
        }
    }
    let leaves = [1_usize, 2, 4, 8, 16, 32, 64, 61];
    for &leaf in &leaves {
        let visual_h = add_tree(&h_parts, 0, h_parts.len(), leaf);
        let visual_b = add_tree(&b_parts, 0, b_parts.len(), leaf);
        let mut h = visual_h.clone();
        for index in 0..h.len() {
            h[index] += imu_h[index];
            h[index] += prior_h[index];
        }
        let mut b = visual_b.clone();
        for index in 0..b.len() {
            b[index] += f32::from_bits(imu_b_words[index]);
            b[index] += f32::from_bits(prior_b_words[index]);
        }
        println!(
            "leaf={leaf} visual_h={} visual_b={} final_h={} final_b={} first_h={} first_b={}",
            count_equal_bits(&visual_h, &old_visual_h),
            count_equal(&visual_b, &old_visual_b_words),
            count_equal(&h, &native_h),
            count_equal(&b, &native_b),
            first_mismatch(&h, &native_h, 75),
            first_mismatch(&b, &native_b, 75),
        );
    }
    let _ = bits; // Keep the parser helper available while probing captures.
}

#[test]
fn m7_probe_relpose_candidate_vs_generic_all_iter0_pairs() {
    fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
        let q = std::hint::black_box(q);
        let t = std::hint::black_box(t);
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        }
    }
    fn lanes(pose: F32Pose) -> [u32; 7] {
        [
            pose.rotation.i.to_bits(),
            pose.rotation.j.to_bits(),
            pose.rotation.k.to_bits(),
            pose.rotation.w.to_bits(),
            pose.translation.x.to_bits(),
            pose.translation.y.to_bits(),
            pose.translation.z.to_bits(),
        ]
    }
    let states = std::hint::black_box([
        pose(
            [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
            [0x00000000, 0x00000000, 0x00000000],
        ),
        pose(
            [0xbd5cdea6, 0xbf4d341f, 0xbb2aa78b, 0x3f186f67],
            [0x39c8bb60, 0xb8cebae8, 0xbb0527fc],
        ),
        pose(
            [0xbd5cf5f2, 0xbf4d97c0, 0xbbac4997, 0x3f17e7a6],
            [0x3af3e90a, 0xb9fd97e2, 0xbbfc3924],
        ),
        pose(
            [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
            [0x3b8ebfa0, 0xba9428f4, 0xbc857642],
        ),
        pose(
            [0xbd5f79e6, 0xbf4f45a2, 0xbbb53f68, 0x3f159719],
            [0x3c02c75a, 0xbb0c3bda, 0xbcdc2b7f],
        ),
    ]);
    let extrinsics = std::hint::black_box([
        pose(
            [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
            [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
        ),
        pose(
            [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
            [0xbc76fa76, 0x3d290319, 0x3b4f4832],
        ),
    ]);
    let host = std::hint::black_box(states[0]);
    for target_frame in 0..5 {
        for target_cam in 0..2 {
            if target_frame == 0 && target_cam == 0 {
                continue;
            }
            let tmp2 = std::hint::black_box(extrinsics[target_cam].inverse());
            let target = std::hint::black_box(states[target_frame]);
            // OLD production FEJ chain, spelled locally as it was before
            // the out-of-line computeRelPose candidate was introduced.
            let target_inverse_rotation = sophus_so3_inverse(target.rotation);
            let relative_rotation = sophus_quat_product_f32(target_inverse_rotation, host.rotation);
            let relative_translation = sophus_rotate_difference_f32(
                target_inverse_rotation,
                host.translation,
                target.translation,
            );
            let generic = std::hint::black_box(F32Pose {
                rotation: sophus_quat_product_f32(tmp2.rotation, relative_rotation),
                translation: sophus_rotate_f32(tmp2.rotation, relative_translation)
                    + tmp2.translation,
            });
            let candidate = std::hint::black_box(sophus_compute_relpose_tmp_out_of_line_f32(
                tmp2, target, host,
            ));
            let candidate_lanes = lanes(candidate);
            let generic_lanes = lanes(generic);
            if (target_frame == 1 && target_cam == 1)
                || (target_frame == 4 && (target_cam == 0 || target_cam == 1))
            {
                let candidate_raw = std::hint::black_box(m7_relpose_packet_product_raw_for_test(
                    tmp2.rotation,
                    relative_rotation,
                ));
                let generic_raw = std::hint::black_box(m7_generic_packet_product_raw_for_test(
                    tmp2.rotation,
                    relative_rotation,
                ));
                println!(
                    concat!(
                        "pair={}/{} raw_candidate=",
                        "{:08x},{:08x},{:08x},{:08x} raw_generic=",
                        "{:08x},{:08x},{:08x},{:08x} generic_q=",
                        "{:08x},{:08x},{:08x},{:08x}"
                    ),
                    target_frame,
                    target_cam,
                    candidate_raw[0].to_bits(),
                    candidate_raw[1].to_bits(),
                    candidate_raw[2].to_bits(),
                    candidate_raw[3].to_bits(),
                    generic_raw[0].to_bits(),
                    generic_raw[1].to_bits(),
                    generic_raw[2].to_bits(),
                    generic_raw[3].to_bits(),
                    generic.rotation.i.to_bits(),
                    generic.rotation.j.to_bits(),
                    generic.rotation.k.to_bits(),
                    generic.rotation.w.to_bits(),
                );
            }
            let diffs = candidate_lanes
                .iter()
                .zip(generic_lanes.iter())
                .enumerate()
                .filter_map(|(index, (actual, expected))| {
                    (actual != expected).then_some(format!("{index}:{actual:08x}/{expected:08x}"))
                })
                .collect::<Vec<_>>();
            let d_candidate = eigen_adjoint_times_rotation_blocks_f32(
                candidate,
                eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
                1.0,
            );
            let d_generic = eigen_adjoint_times_rotation_blocks_f32(
                generic,
                eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
                1.0,
            );
            let d_diffs = d_candidate
                .as_slice()
                .iter()
                .zip(d_generic.as_slice().iter())
                .enumerate()
                .filter_map(|(index, (actual, expected))| {
                    (actual.to_bits() != expected.to_bits()).then_some(format!(
                        "{index}:{:08x}/{:08x}",
                        actual.to_bits(),
                        expected.to_bits()
                    ))
                })
                .collect::<Vec<_>>();
            let drel_diff_count = |actual: F32Pose| {
                let d_actual = eigen_adjoint_times_rotation_blocks_f32(
                    actual,
                    eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
                    1.0,
                );
                d_actual
                    .as_slice()
                    .iter()
                    .zip(d_generic.as_slice().iter())
                    .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
                    .count()
            };
            let candidate_rotation_only = F32Pose {
                rotation: candidate.rotation,
                translation: generic.translation,
            };
            let candidate_translation_only = F32Pose {
                rotation: generic.rotation,
                translation: candidate.translation,
            };
            if !d_diffs.is_empty() {
                println!(
                    concat!(
                        "pair={}/{} intermediate ",
                        "cand_rot={:08x},{:08x},{:08x},{:08x} ",
                        "gen_rot={:08x},{:08x},{:08x},{:08x} ",
                        "cand_t={:08x},{:08x},{:08x} ",
                        "gen_t={:08x},{:08x},{:08x} ",
                        "drel_total={} drel_rot_only={} drel_trans_only={}"
                    ),
                    target_frame,
                    target_cam,
                    candidate.rotation.i.to_bits(),
                    candidate.rotation.j.to_bits(),
                    candidate.rotation.k.to_bits(),
                    candidate.rotation.w.to_bits(),
                    generic.rotation.i.to_bits(),
                    generic.rotation.j.to_bits(),
                    generic.rotation.k.to_bits(),
                    generic.rotation.w.to_bits(),
                    candidate.translation.x.to_bits(),
                    candidate.translation.y.to_bits(),
                    candidate.translation.z.to_bits(),
                    generic.translation.x.to_bits(),
                    generic.translation.y.to_bits(),
                    generic.translation.z.to_bits(),
                    d_diffs.len(),
                    drel_diff_count(candidate_rotation_only),
                    drel_diff_count(candidate_translation_only),
                );
            }
            if !diffs.is_empty() || !d_diffs.is_empty() {
                println!(
                    "pair={target_frame}/{target_cam} pose_diffs=[{}] drel_diffs=[{}]",
                    diffs.join(","),
                    d_diffs.join(",")
                );
            }
        }
    }
}

#[test]
fn m7im15_relpose_dual_pass_native_drel_and_intermediates() {
    // Direct native capture for the shared frame0/cam0 -> frame3/cam1
    // call at optimization passes 0 and 1.  Keep both passes in one
    // fixture because pass 0 is the guard against a schedule change that
    // merely fixes the later six cross lanes.
    fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        }
    }
    fn lanes(pose: F32Pose) -> [u32; 7] {
        [
            pose.rotation.i.to_bits(),
            pose.rotation.j.to_bits(),
            pose.rotation.k.to_bits(),
            pose.rotation.w.to_bits(),
            pose.translation.x.to_bits(),
            pose.translation.y.to_bits(),
            pose.translation.z.to_bits(),
        ]
    }
    fn format_lanes(values: [u32; 7]) -> String {
        values
            .iter()
            .map(|value| format!("{value:08x}"))
            .collect::<Vec<_>>()
            .join(",")
    }
    fn drel(pose: F32Pose, host: F32Pose) -> SMatrix<f32, 6, 6> {
        eigen_adjoint_times_rotation_blocks_f32(
            pose,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
            1.0,
        )
    }
    fn exact(actual: &SMatrix<f32, 6, 6>, expected: &[u32; 36]) -> usize {
        actual
            .as_slice()
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual.to_bits() == **expected)
            .count()
    }
    let host = pose(
        [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
        [0x00000000, 0x00000000, 0x00000000],
    );
    let target_ext = pose(
        [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
        [0xbc76fa76, 0x3d290319, 0x3b4f4832],
    );
    let cases = [
        (
            pose(
                [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
                [0x3b8ebfa0, 0xba9428f4, 0xbc857642],
            ),
            [
                0xba954ed4, 0xbbce8a7c, 0x3ad34473, 0x3f7ffe93, 0xbde1e309, 0xbc8b6eac, 0xbacb5e9a,
            ],
            [
                0x3c31fe46, 0xbc520562, 0xbf33585a, 0x3f36a0a8, 0xbd27a81a, 0xbd05555f, 0xbb8ddf30,
            ],
            [
                0x3de42768, 0x3e92d17a, 0xbf7395cf, 0x00000000, 0x00000000, 0x00000000, 0x3f7e3e31,
                0xbd881872, 0x3dc51f14, 0x00000000, 0x00000000, 0x00000000, 0xbd11f0b3, 0xbf74a889,
                0xbe9599d5, 0x00000000, 0x00000000, 0x00000000, 0x3d03f3e6, 0xbd21806f, 0xbc04e3d7,
                0x3de42768, 0x3e92d17a, 0xbf7395cf, 0xbb6030cb, 0xb9bcd31b, 0x3d0f8f42, 0x3f7e3e31,
                0xbd881872, 0x3dc51f14, 0x3bb0151d, 0xbc416c25, 0x3d1b7a6e, 0xbd11f0b3, 0xbf74a889,
                0xbe9599d5,
            ],
        ),
        (
            pose(
                [0xbd5e3af5, 0xbf4e66c7, 0xbbc31a02, 0x3f16cb91],
                [0x3b2b6284, 0xbb51bffc, 0xbd4c7b94],
            ),
            [
                0xba7446f7, 0xbbcdd358, 0x3adb39c1, 0x3f7ffe96, 0xbddfb9b6, 0xbd47e71b, 0xbc527981,
            ],
            [
                0x3c342b3c, 0xbc4f602d, 0xbf3355a4, 0x3f36a35f, 0xbd235394, 0xbd83bd75, 0xbc801294,
            ],
            [
                0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0x00000000, 0x00000000, 0x00000000, 0x3f7e3f01,
                0xbd87ecbc, 0x3dc4f985, 0x00000000, 0x00000000, 0x00000000, 0xbd118207, 0xbf74a0e5,
                0xbe95cd73, 0x00000000, 0x00000000, 0x00000000, 0x3d8687db, 0xbd228425, 0xbb8c8d8c,
                0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0xbbecbb07, 0xbc3f8e51, 0x3d8841e7, 0x3f7e3f01,
                0xbd87ecbc, 0x3dc4f985, 0x3b7e5e58, 0xbc360bff, 0x3d12b627, 0xbd118207, 0xbf74a0e5,
                0xbe95cd73,
            ],
        ),
    ];
    for (pass, (target, native_relative, native_tmp, native_drel)) in cases.into_iter().enumerate()
    {
        let target_camera_from_imu = target_ext.inverse();
        let relative = sophus_relative_imu_f32(target, host);
        let generic = F32Pose {
            rotation: sophus_quat_product_f32(target_camera_from_imu.rotation, relative.rotation),
            translation: sophus_rotate_f32(target_camera_from_imu.rotation, relative.translation)
                + target_camera_from_imu.translation,
        };
        let candidate =
            sophus_compute_relpose_tmp_out_of_line_f32(target_camera_from_imu, target, host);
        let generic_rotated_translation =
            sophus_rotate_f32(target_camera_from_imu.rotation, relative.translation);
        let candidate_rotated_translation = sophus_rotate_relpose_out_of_line_f32(
            target_camera_from_imu.rotation,
            relative.translation,
        );
        let packet_rotated_translation =
            sophus_rotate_step_packet_f32(target_camera_from_imu.rotation, relative.translation);
        let packet = F32Pose {
            rotation: sophus_quat_product_f32(target_camera_from_imu.rotation, relative.rotation),
            translation: packet_rotated_translation + target_camera_from_imu.translation,
        };
        let generic_drel = drel(generic, host);
        let candidate_drel = drel(candidate, host);
        let packet_drel = drel(packet, host);
        let relative_expected = pose(
            [
                native_relative[0],
                native_relative[1],
                native_relative[2],
                native_relative[3],
            ],
            [native_relative[4], native_relative[5], native_relative[6]],
        );
        let tmp_expected = pose(
            [native_tmp[0], native_tmp[1], native_tmp[2], native_tmp[3]],
            [native_tmp[4], native_tmp[5], native_tmp[6]],
        );
        let generic_exact = exact(&generic_drel, &native_drel);
        let candidate_exact = exact(&candidate_drel, &native_drel);
        let packet_exact = exact(&packet_drel, &native_drel);
        println!(
            "dual_pass={pass} relative={} relative_exact={} tmp2={} relative_t={} generic_rot={} candidate_rot={} packet_rot={} generic_tmp={} generic_tmp_exact={} candidate_tmp={} candidate_tmp_exact={} packet_tmp={} packet_tmp_exact={} generic_drel_exact={generic_exact}/36 candidate_drel_exact={candidate_exact}/36 packet_drel_exact={packet_exact}/36",
            format_lanes(lanes(relative)),
            lanes(relative) == lanes(relative_expected),
            format_lanes(lanes(target_camera_from_imu)),
            format_lanes([
                relative.translation.x.to_bits(),
                relative.translation.y.to_bits(),
                relative.translation.z.to_bits(),
                0,
                0,
                0,
                0,
            ]),
            format_lanes(lanes(F32Pose { rotation: UnitQuaternion::identity(), translation: generic_rotated_translation })),
            format_lanes(lanes(F32Pose { rotation: UnitQuaternion::identity(), translation: candidate_rotated_translation })),
            format_lanes(lanes(F32Pose { rotation: UnitQuaternion::identity(), translation: packet_rotated_translation })),
            format_lanes(lanes(generic)),
            lanes(generic) == lanes(tmp_expected),
            format_lanes(lanes(candidate)),
            lanes(candidate) == lanes(tmp_expected),
            format_lanes(lanes(packet)),
            lanes(packet) == lanes(tmp_expected),
        );
        // The direct GDB `T_t_h_sophus_qt_u32` capture is retained in the
        // printout above, but this helper's packet schedule is not the
        // final `tmp` boundary by itself.  Do not turn that intermediate
        // observation into a production assertion: the composed generic
        // path is the contract being compared below.
        assert_eq!(
            generic_exact,
            if pass == 0 { 36 } else { 30 },
            "current generic d_rel_h exact count at pass {pass}"
        );
        assert_eq!(
            candidate_exact,
            if pass == 0 { 23 } else { 19 },
            "out-of-line candidate d_rel_h exact count at pass {pass}"
        );
        assert_eq!(
            lanes(packet),
            lanes(tmp_expected),
            "step-only packet SO3 action must reproduce native tmp at pass {pass}"
        );
        assert_eq!(
            packet_exact, 36,
            "step-only packet SO3 action must reproduce native d_rel_h at pass {pass}"
        );
    }
}

#[test]
fn m7im15_current_relative_packet_pass0_3_exact() {
    // The stage-only native capture stops immediately before the current
    // camera suffix and records the normalized target inverse in xmm0,
    // the current host quaternion in rdi, and the normalized relative
    // result at 0x3055f6.  Feed those exact words to the isolated packet
    // helper so this test covers the quaternion product/normalization
    // boundary independently of translation and camera extrinsics.
    fn quaternion(words: [u32; 4]) -> UnitQuaternion<f32> {
        UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(words[3]),
            f32::from_bits(words[0]),
            f32::from_bits(words[1]),
            f32::from_bits(words[2]),
        ))
    }

    let target_inverse = [
        [0x3d5e760f, 0x3f4e5e85, 0x3bc2d95f, 0x3f16d68a],
        [0x3d5e3af6, 0x3f4e66c8, 0x3bc31a03, 0x3f16cb92],
        [0x3d5e2068, 0x3f4e76f4, 0x3bc519c6, 0x3f16b589],
        [0x3d5cd678, 0x3f4e9bf7, 0x3bd4932e, 0x3f168457],
    ];
    let host_current = [
        [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
        [0xbd587ff0, 0xbf4d69a0, 0x38d8e054, 0x3f182def],
        [0xbd589afe, 0xbf4d86a4, 0x38cfb660, 0x3f180696],
        [0xbd5782db, 0xbf4db536, 0xb9aa33e3, 0x3f17c91a],
    ];
    let expected = [
        [0x3bc353bb, 0x3bc9740e, 0x3b240734, 0x3f7ffd65],
        [0x3bc3e5d4, 0x3bceeae2, 0x3b2fc253, 0x3f7ffd49],
        [0x3bc3ffb8, 0x3bc40dbc, 0x3b33b475, 0x3f7ffd69],
        [0x3bc409b5, 0x3bbc4711, 0x3b3739d6, 0x3f7ffd7d],
    ];

    for pass in 0..4 {
        for _target_cam in 0..2 {
            let result = sophus_quat_product_current_packet_f32(
                quaternion(target_inverse[pass]),
                quaternion(host_current[pass]),
            );
            let q = result.quaternion();
            assert_f32_bits(&[q.i, q.j, q.k, q.w], &expected[pass]);
        }
    }
}

#[test]
#[ignore = "requires pinned external target-frame3 relpose capture"]
fn m7im15_current_relpose_targetframe3_cam0_cam1_schedule_probe() {
    // This is a diagnostic sidecar for the first iter3 visual factor.  It
    // uses the exact current/FEJ endpoint words from the native fixture,
    // then prints the value-chain schedule variants.  No production
    // branch is selected by this test; it exists to make the cam0 path
    // (which was not present in the original cam1-only fixture) directly
    // comparable before any call-site change.
    fn pose(words: &[u32]) -> F32Pose {
        assert_eq!(words.len(), 7);
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(words[3]),
                f32::from_bits(words[0]),
                f32::from_bits(words[1]),
                f32::from_bits(words[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(words[4]),
                f32::from_bits(words[5]),
                f32::from_bits(words[6]),
            ),
        }
    }
    fn words(value: &serde_json::Value, key: &str) -> Vec<u32> {
        value[key]
            .as_array()
            .unwrap_or_else(|| panic!("missing array {key}"))
            .iter()
            .map(|word| {
                let word = word.as_str().expect("fixture bit must be a string");
                u32::from_str_radix(word, 16).expect("fixture bit must be hexadecimal")
            })
            .collect()
    }
    fn lanes(value: F32Pose) -> [u32; 7] {
        [
            value.rotation.i.to_bits(),
            value.rotation.j.to_bits(),
            value.rotation.k.to_bits(),
            value.rotation.w.to_bits(),
            value.translation.x.to_bits(),
            value.translation.y.to_bits(),
            value.translation.z.to_bits(),
        ]
    }
    fn exact_pose(actual: F32Pose, expected: &[u32]) -> usize {
        lanes(actual)
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual == expected)
            .count()
    }
    fn exact_matrix(actual: &SMatrix<f32, 6, 6>, expected: &[u32]) -> usize {
        actual
            .as_slice()
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual.to_bits() == **expected)
            .count()
    }
    fn fmt7(value: F32Pose) -> String {
        lanes(value)
            .iter()
            .map(|word| format!("{word:08x}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let native_path = root.join("../../target/m7im15_relpose_targetframe3_pass0_3_20260827.json");
    let cam1_path = root.join("../../target/m7im15_step_relpose_inputs_gdb_20260827.json");
    let native: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&native_path).expect("target-frame3 native capture is required"),
    )
    .expect("target-frame3 native capture must be valid JSON");
    let cam1: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&cam1_path).expect("cam1 native fixture is required"),
    )
    .expect("cam1 native fixture must be valid JSON");
    let native_records = native["records"].as_array().expect("native records array");
    let cam1_records = cam1["records"].as_array().expect("cam1 records array");

    let host_ext = pose(&[
        0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc, 0x3ba86617,
    ]);
    let target_exts = [
        host_ext,
        pose(&[
            0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6, 0xbc76fa76, 0x3d290319, 0x3b4f4832,
        ]),
    ];
    let mut total = [[0usize; 5]; 2];
    let mut total_drel_h = [[0usize; 5]; 2];
    let mut total_drel_t = [[0usize; 5]; 2];
    for pass in 0..4 {
        let host_record = cam1_records
            .iter()
            .find(|record| record["optimization_linearization_pass_index"] == pass)
            .expect("cam1 fixture pass");
        let host_current = pose(&words(host_record, "host_pose_current_qt_u32"));
        let host_fej = pose(&words(host_record, "host_pose_lin_qt_u32"));
        let native_pass = native_records
            .iter()
            .filter(|record| record["pass"] == pass)
            .collect::<Vec<_>>();
        assert_eq!(native_pass.len(), 2);
        for native_record in native_pass {
            let target_cam = native_record["target_cam"].as_u64().unwrap() as usize;
            let target_current = pose(&words(native_record, "target_current_qt_u32"));
            let target_fej = pose(&words(native_record, "target_lin_qt_u32"));
            let expected = words(native_record, "T_t_h_sophus_qt_u32");
            let expected_h = words(native_record, "d_rel_d_h_column_major_u32");
            let expected_t = words(native_record, "d_rel_d_t_column_major_u32");
            let target_camera_from_imu = target_exts[target_cam].inverse();
            let target_inverse_rotation = sophus_so3_inverse(target_current.rotation);

            // Production current value chain: scalar relative action,
            // camera-prefix/suffix quaternion packets, generic SO3 action
            // at both camera composition points.
            let target_imu_from_host = sophus_relative_imu_f32(target_current, host_current);
            let tmp_generic = F32Pose {
                rotation: sophus_quat_product_camera_prefix_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_host.rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_host.translation,
                ) + target_camera_from_imu.translation,
            };
            let result_generic = F32Pose {
                rotation: sophus_quat_product_camera_suffix_f32(
                    tmp_generic.rotation,
                    host_ext.rotation,
                ),
                translation: sophus_rotate_f32(tmp_generic.rotation, host_ext.translation)
                    + tmp_generic.translation,
            };

            // Isolate the current relative quaternion packet while
            // retaining the already-proven translation/camera schedule.
            let target_imu_from_host_packet = F32Pose {
                rotation: sophus_quat_product_current_packet_f32(
                    target_inverse_rotation,
                    host_current.rotation,
                ),
                translation: sophus_rotate_difference_visual_f32(
                    target_inverse_rotation,
                    host_current.translation,
                    target_current.translation,
                ),
            };
            let tmp_current_packet = F32Pose {
                rotation: sophus_quat_product_camera_prefix_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_host_packet.rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_host_packet.translation,
                ) + target_camera_from_imu.translation,
            };
            let result_current_packet = F32Pose {
                rotation: sophus_quat_product_camera_suffix_f32(
                    tmp_current_packet.rotation,
                    host_ext.rotation,
                ),
                translation: sophus_rotate_f32(tmp_current_packet.rotation, host_ext.translation)
                    + tmp_current_packet.translation,
            };

            // The two candidate step-only actions are kept independent so
            // a match identifies the exact SO3*Vector call-site boundary.
            let mut variants = [result_generic; 5];
            for (variant, (tmp_action, suffix_action)) in
                [(false, false), (true, false), (false, true), (true, true)]
                    .into_iter()
                    .enumerate()
            {
                let tmp_translation = if tmp_action {
                    sophus_rotate_step_packet_f32(
                        target_camera_from_imu.rotation,
                        target_imu_from_host.translation,
                    )
                } else {
                    sophus_rotate_f32(
                        target_camera_from_imu.rotation,
                        target_imu_from_host.translation,
                    )
                } + target_camera_from_imu.translation;
                let tmp = F32Pose {
                    rotation: tmp_generic.rotation,
                    translation: tmp_translation,
                };
                let suffix_translation = if suffix_action {
                    sophus_rotate_step_packet_f32(tmp.rotation, host_ext.translation)
                } else {
                    sophus_rotate_f32(tmp.rotation, host_ext.translation)
                };
                variants[variant + 1] = F32Pose {
                    rotation: tmp.rotation,
                    translation: suffix_translation + tmp.translation,
                };
            }

            // A packetized relative translation is also reported as a
            // separate family; it is not selected by production here.
            let relative_packet = F32Pose {
                rotation: target_imu_from_host.rotation,
                translation: sophus_rotate_difference_f32(
                    sophus_so3_inverse(target_current.rotation),
                    host_current.translation,
                    target_current.translation,
                ),
            };
            let packet_tmp = F32Pose {
                rotation: sophus_quat_product_camera_prefix_f32(
                    target_camera_from_imu.rotation,
                    relative_packet.rotation,
                ),
                translation: sophus_rotate_step_packet_f32(
                    target_camera_from_imu.rotation,
                    relative_packet.translation,
                ) + target_camera_from_imu.translation,
            };
            variants[4] = F32Pose {
                rotation: sophus_quat_product_camera_suffix_f32(
                    packet_tmp.rotation,
                    host_ext.rotation,
                ),
                translation: sophus_rotate_step_packet_f32(
                    packet_tmp.rotation,
                    host_ext.translation,
                ) + packet_tmp.translation,
            };

            let target_inverse_fej = sophus_so3_inverse(target_fej.rotation);
            let relative_fej = sophus_relative_imu_f32(target_fej, host_fej);
            let tmp_fej = F32Pose {
                rotation: sophus_quat_product_f32(
                    target_camera_from_imu.rotation,
                    relative_fej.rotation,
                ),
                translation: sophus_rotate_step_packet_f32(
                    target_camera_from_imu.rotation,
                    relative_fej.translation,
                ) + target_camera_from_imu.translation,
            };
            let drel_h = eigen_adjoint_times_rotation_blocks_f32(
                tmp_fej,
                eigen_quaternion_matrix_f32(sophus_so3_inverse(host_fej.rotation)),
                1.0,
            );
            let drel_t = eigen_adjoint_times_rotation_blocks_f32(
                target_camera_from_imu,
                eigen_quaternion_matrix_f32(target_inverse_fej),
                -1.0,
            );
            println!(
                "m7im15_current_relpose pass={pass} cam={target_cam} native={} generic={} packet={} exact={}/7 packet_exact={}/7 variants={:?} native_drel={}/{} rust_drel_h={}/36 rust_drel_t={}/36",
                expected.iter().map(|word| format!("{word:08x}")).collect::<Vec<_>>().join(" "),
                fmt7(result_generic),
                fmt7(result_current_packet),
                exact_pose(result_generic, &expected),
                exact_pose(result_current_packet, &expected),
                variants.iter().map(|value| exact_pose(*value, &expected)).collect::<Vec<_>>(),
                exact_matrix(&drel_h, &expected_h),
                exact_matrix(&drel_t, &expected_t),
                exact_matrix(&drel_h, &expected_h),
                exact_matrix(&drel_t, &expected_t),
            );
            for (index, value) in variants.iter().enumerate() {
                total[target_cam][index] += exact_pose(*value, &expected);
            }
            total_drel_h[target_cam][0] += exact_matrix(&drel_h, &expected_h);
            total_drel_t[target_cam][0] += exact_matrix(&drel_t, &expected_t);
        }
    }
    println!("m7im15_current_relpose totals cam0={total:?} drel_h={total_drel_h:?} drel_t={total_drel_t:?}");
}

#[test]
#[ignore = "requires pinned external target-frame3 transform captures"]
fn m7im15_current_packet_transform_matrix_pass0_3_probe() {
    // The target-frame3 relpose fixture records the exact current endpoint
    // words, while the current-transform fixture records the independent
    // inlined linearizePoint matrix.  Compare the packet-relative-
    // quaternion variant against that matrix, keeping the existing
    // translation and camera prefix/suffix schedules unchanged.
    fn pose(words: &[u32]) -> F32Pose {
        assert_eq!(words.len(), 7);
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(words[3]),
                f32::from_bits(words[0]),
                f32::from_bits(words[1]),
                f32::from_bits(words[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(words[4]),
                f32::from_bits(words[5]),
                f32::from_bits(words[6]),
            ),
        }
    }
    fn words(value: &serde_json::Value, key: &str) -> Vec<u32> {
        value[key]
            .as_array()
            .unwrap_or_else(|| panic!("missing array {key}"))
            .iter()
            .map(|word| {
                u32::from_str_radix(word.as_str().expect("fixture bit must be a string"), 16)
                    .expect("fixture bit must be hexadecimal")
            })
            .collect()
    }
    fn matrix_bits(value: F32Pose) -> [u32; 16] {
        let rotation = eigen_quaternion_matrix_f32(value.rotation);
        [
            rotation[(0, 0)].to_bits(),
            rotation[(1, 0)].to_bits(),
            rotation[(2, 0)].to_bits(),
            0,
            rotation[(0, 1)].to_bits(),
            rotation[(1, 1)].to_bits(),
            rotation[(2, 1)].to_bits(),
            0,
            rotation[(0, 2)].to_bits(),
            rotation[(1, 2)].to_bits(),
            rotation[(2, 2)].to_bits(),
            0,
            value.translation.x.to_bits(),
            value.translation.y.to_bits(),
            value.translation.z.to_bits(),
            0x3f800000,
        ]
    }
    fn exact(actual: &[u32], expected: &[u32]) -> usize {
        actual
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual == expected)
            .count()
    }
    fn pose_q_bits(value: F32Pose) -> String {
        [
            value.rotation.i.to_bits(),
            value.rotation.j.to_bits(),
            value.rotation.k.to_bits(),
            value.rotation.w.to_bits(),
        ]
        .iter()
        .map(|word| format!("{word:08x}"))
        .collect::<Vec<_>>()
        .join(" ")
    }
    fn first_mismatch(actual: &[u32], expected: &[u32]) -> String {
        actual
            .iter()
            .zip(expected)
            .enumerate()
            .find_map(|(index, (actual, expected))| {
                (actual != expected)
                    .then(|| format!("idx={index} native={expected:08x} rust={actual:08x}"))
            })
            .unwrap_or_else(|| "none".to_string())
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let native_path = root.join("../../target/m7im15_relpose_targetframe3_pass0_3_20260827.json");
    let matrix_path =
        root.join("../../target/m7im15_current_transform_factor_fixture_20260827.json");
    let native: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&native_path).expect("target-frame3 native capture is required"),
    )
    .expect("target-frame3 native capture must be valid JSON");
    let matrix_fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&matrix_path).expect("current-transform fixture is required"),
    )
    .expect("current-transform fixture must be valid JSON");
    let native_records = native["records"].as_array().expect("native records array");

    let host_ext = pose(&[
        0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc, 0x3ba86617,
    ]);
    let target_exts = [
        host_ext,
        pose(&[
            0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6, 0xbc76fa76, 0x3d290319, 0x3b4f4832,
        ]),
    ];

    for pass in 0..4 {
        let fixture_case = &matrix_fixture["current_transform_cases"][pass];
        for target_cam in 0..2 {
            let native_record = native_records
                .iter()
                .find(|record| record["pass"] == pass && record["target_cam"] == target_cam)
                .expect("target-frame3 native pass/camera record");
            let fixture_observation = fixture_case["observations"]
                .as_array()
                .expect("fixture observations array")
                .iter()
                .find(|observation| observation["target_cam"] == target_cam)
                .expect("current-transform camera observation");
            let expected_matrix = words(fixture_observation, "native_T_t_h_bits_column_major");
            assert_eq!(expected_matrix.len(), 16);
            // The native inlined current visual branch receives the
            // effective endpoint returned by PoseStateWithLin::getPose.
            // For this fixture target frame 3 is non-linearized, so that
            // endpoint is its raw pose_linearized payload.  The host
            // endpoint remains current; target translation is likewise
            // the value-side endpoint selected by getPose().
            let target_current = pose(&words(native_record, "target_lin_qt_u32"));
            let host_current = pose(&words(native_record, "host_current_qt_u32"));
            let target_camera_from_imu = target_exts[target_cam].inverse();
            let target_inverse_rotation =
                sophus_quat_inverse_current_packet_f32(target_current.rotation);
            let relative_packet = F32Pose {
                rotation: sophus_quat_product_current_packet_f32(
                    target_inverse_rotation,
                    host_current.rotation,
                ),
                translation: sophus_rotate_difference_visual_f32(
                    target_inverse_rotation,
                    host_current.translation,
                    target_current.translation,
                ),
            };
            let tmp_packet = F32Pose {
                rotation: sophus_quat_product_camera_prefix_f32(
                    target_camera_from_imu.rotation,
                    relative_packet.rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_imu.rotation,
                    relative_packet.translation,
                ) + target_camera_from_imu.translation,
            };
            let result_packet = F32Pose {
                rotation: sophus_quat_product_camera_suffix_f32(
                    tmp_packet.rotation,
                    host_ext.rotation,
                ),
                translation: sophus_rotate_f32(tmp_packet.rotation, host_ext.translation)
                    + tmp_packet.translation,
            };
            let actual_packet = matrix_bits(result_packet);
            println!(
                "m7im15_current_packet_matrix pass={pass} cam={target_cam} exact={}/16 first={} q={} t={} target_ext_inv={} target_pose_inv={} relative={} prefix={}",
                exact(&actual_packet, &expected_matrix),
                first_mismatch(&actual_packet, &expected_matrix),
                pose_q_bits(result_packet),
                [
                    result_packet.translation.x.to_bits(),
                    result_packet.translation.y.to_bits(),
                    result_packet.translation.z.to_bits(),
                ]
                .iter()
                .map(|word| format!("{word:08x}"))
                .collect::<Vec<_>>()
                .join(" "),
                pose_q_bits(target_camera_from_imu),
                pose_q_bits(F32Pose {
                    rotation: target_inverse_rotation,
                    translation: Vector3::zeros(),
                }),
                pose_q_bits(relative_packet),
                pose_q_bits(tmp_packet),
            );
        }
    }
}

#[test]
fn m7_probe_candidate_weighted_operands_track120_frame1_cam1() {
    fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
        let q = std::hint::black_box(q);
        let t = std::hint::black_box(t);
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        }
    }
    fn print_matrix<const R: usize, const C: usize>(name: &str, matrix: &SMatrix<f32, R, C>) {
        let bits = matrix
            .as_slice()
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>();
        println!("{name}_column_major={}", bits.join(","));
        let rows = (0..R)
            .flat_map(|row| {
                (0..C).map(move |column| format!("{:08x}", matrix[(row, column)].to_bits()))
            })
            .collect::<Vec<_>>();
        println!("{name}_row_major={}", rows.join(","));
    }

    let host = pose(
        [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
        [0x00000000, 0x00000000, 0x00000000],
    );
    let target = pose(
        [0xbd5cdea6, 0xbf4d341f, 0xbb2aa78b, 0x3f186f67],
        [0x39c8bb60, 0xb8cebae8, 0xbb0527fc],
    );
    let host_ext = pose(
        [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
        [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
    );
    let target_ext = pose(
        [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
        [0xbc76fa76, 0x3d290319, 0x3b4f4832],
    );
    let landmark = InverseDistanceLandmark {
        anchor_pose: 0,
        anchor_camera_id: 0,
        direction: StereographicDirection {
            xy: Point2::new(
                f32::from_bits(0x3ea596dd) as f64,
                f32::from_bits(0xbd8fc60e) as f64,
            ),
        },
        inverse_distance: f32::from_bits(0x3d804cdd) as f64,
    };
    let target_camera_from_imu = target_ext.inverse();
    let target_inverse_rotation = sophus_so3_inverse(target.rotation);
    let target_imu_from_anchor_imu = F32Pose {
        rotation: sophus_quat_product_f32(target_inverse_rotation, host.rotation),
        translation: sophus_rotate_difference_visual_f32(
            target_inverse_rotation,
            host.translation,
            target.translation,
        ),
    };
    let target_camera_from_anchor_imu = F32Pose {
        rotation: sophus_quat_product_camera_prefix_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.translation,
        ) + target_camera_from_imu.translation,
    };
    let target_camera_from_anchor_camera = F32Pose {
        rotation: sophus_quat_product_camera_suffix_f32(
            target_camera_from_anchor_imu.rotation,
            host_ext.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_anchor_imu.rotation,
            host_ext.translation,
        ) + target_camera_from_anchor_imu.translation,
    };
    let bearing = landmark.direction.bearing_f32();
    let point4_target = eigen_homogeneous_point_gemv_f32(
        target_camera_from_anchor_camera.rotation,
        target_camera_from_anchor_camera.translation,
        bearing,
        landmark.inverse_distance as f32,
    );
    let point_target = point4_target.fixed_rows::<3>(0).into_owned();
    let (_, projection_jacobian) =
        project_double_sphere_with_jacobian_f32(&m7_fixed_camera(), point_target).unwrap();
    let mut point_wrt_relative_pose = SMatrix::<f32, 3, 6>::zeros();
    point_wrt_relative_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(SMatrix::<f32, 3, 3>::identity() * landmark.inverse_distance as f32));
    point_wrt_relative_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew3_f32(point_target)));
    let residual_wrt_relative_pose =
        eigen_relative_pose_jacobian_f32(projection_jacobian, point_wrt_relative_pose);
    let mut weighted_relative = residual_wrt_relative_pose;
    for value in weighted_relative.as_mut_slice() {
        *value *= f32::from_bits(0x40000000);
    }
    // Keep the packet-generation witness separate from the current-value
    // chain above.  Native `computeRelPose` materializes the generic
    // Sophus product at this boundary; it is this pose (not the camera
    // prefix packet spelling used for the value path) that feeds drel.
    let generic_target_imu_from_anchor_imu = sophus_relative_imu_f32(target, host);
    let generic_target_camera_from_anchor_imu = F32Pose {
        rotation: sophus_quat_product_f32(
            target_camera_from_imu.rotation,
            generic_target_imu_from_anchor_imu.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_imu.rotation,
            generic_target_imu_from_anchor_imu.translation,
        ) + target_camera_from_imu.translation,
    };
    let candidate_tmp =
        sophus_compute_relpose_tmp_out_of_line_f32(target_camera_from_imu, target, host);
    let baseline_drel = eigen_adjoint_times_rotation_blocks_f32(
        generic_target_camera_from_anchor_imu,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
        1.0,
    );
    let drel = eigen_adjoint_times_rotation_blocks_f32(
        candidate_tmp,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
        1.0,
    );
    let drel_target = eigen_adjoint_times_rotation_blocks_f32(
        target_camera_from_imu,
        eigen_quaternion_matrix_f32(sophus_so3_inverse(target.rotation)),
        -1.0,
    );
    let anchor = eigen_weighted_pose_jacobian_f32(
        residual_wrt_relative_pose,
        drel,
        f32::from_bits(0x40000000),
    );
    // The generic visual-chain pose is the faithful diagnostic packet
    // input for this tied native call.  Keep the out-of-line candidate
    // above visible as a negative witness: it differs in 15/36 lanes.
    let expected_drel = [
        0x3dda79ad, 0x3e8b3905, 0xbf74d5e7, 0x00000000, 0x00000000, 0x00000000, 0x3f7e5131,
        0xbd8df5f8, 0x3dba92eb, 0x00000000, 0x00000000, 0x00000000, 0xbd2a12ca, 0xbf75b6c8,
        0xbe8e17f1, 0x00000000, 0x00000000, 0x00000000, 0x3c93c295, 0xbd226d6d, 0xbc17c30c,
        0x3dda79ad, 0x3e8b3905, 0xbf74d5e7, 0xbaf80701, 0xb9828894, 0x3ca77d72, 0x3f7e5131,
        0xbd8df5f8, 0x3dba92eb, 0x3a8bd267, 0xbc37c50e, 0x3d1e3cc6, 0xbd2a12ca, 0xbf75b6c8,
        0xbe8e17f1,
    ];
    assert_f32_bits(baseline_drel.as_slice(), &expected_drel);
    assert_eq!(
        drel.as_slice()
            .iter()
            .zip(expected_drel)
            .filter(|(actual, expected)| actual.to_bits() != *expected)
            .count(),
        15,
        "out-of-line diagnostic drel should remain the 15-lane negative witness"
    );
    print_matrix("weighted_relative", &weighted_relative);
    print_matrix("baseline_drel", &baseline_drel);
    print_matrix("drel", &drel);
    print_matrix("drel_target", &drel_target);
    print_matrix("anchor", &anchor);
    println!(
        "point_target={:08x},{:08x},{:08x} projection_jacobian={}",
        point_target.x.to_bits(),
        point_target.y.to_bits(),
        point_target.z.to_bits(),
        projection_jacobian
            .as_slice()
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join(",")
    );
}

/// Keep the native `RowMajor` destination boundary separate from the
/// production FEJ path.  The upstream expression is a fixed `2x6 * 6x6`
/// product followed by `block +=`; this test-only wrapper makes that
/// destination update explicit while retaining the scalar Eigen packet
/// schedule in [`eigen_matrix_product_2x6_f32`].
fn diagnostic_row_major_pose_block_add_assign(
    destination: &mut [f32; 12],
    left: SMatrix<f32, 2, 6>,
    right: SMatrix<f32, 6, 6>,
) {
    let product = eigen_matrix_product_2x6_f32(left, right);
    for row in 0..2 {
        for column in 0..6 {
            destination[row * 6 + column] += product[(row, column)];
        }
    }
}

#[test]
fn m7_actual_eigen_site_track120_obs3_anchor_block_is_bit_exact() {
    // Captured from the pinned native Eigen site for frame timestamp
    // 1403636579963555584, iteration 0, track 120, observation 3
    // (target frame 1/cam 1).  The operands are already whitened and are
    // intentionally supplied in Eigen's column-major memory order.
    let left = SMatrix::<f32, 2, 6>::from_column_slice(&[
        f32::from_bits(0x423ee1b5),
        f32::from_bits(0x4032bc80),
        f32::from_bits(0x40334645),
        f32::from_bits(0x42746653),
        f32::from_bits(0xc2053c7b),
        f32::from_bits(0x40d2d4c3),
        f32::from_bits(0x41c095b6),
        f32::from_bits(0xc4480ee4),
        f32::from_bits(0x4465cd32),
        f32::from_bits(0xc1c001b4),
        f32::from_bits(0x42df9490),
        f32::from_bits(0x440c7238),
    ]);
    let right = SMatrix::<f32, 6, 6>::from_column_slice(&[
        f32::from_bits(0x3dda79ad),
        f32::from_bits(0x3e8b3905),
        f32::from_bits(0xbf74d5e7),
        f32::from_bits(0x00000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3f7e5131),
        f32::from_bits(0xbd8df5f8),
        f32::from_bits(0x3dba92eb),
        f32::from_bits(0x00000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0xbd2a12ca),
        f32::from_bits(0xbf75b6c8),
        f32::from_bits(0xbe8e17f1),
        f32::from_bits(0x00000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3c93c295),
        f32::from_bits(0xbd226d6d),
        f32::from_bits(0xbc17c30c),
        f32::from_bits(0x3dda79ad),
        f32::from_bits(0x3e8b3905),
        f32::from_bits(0xbf74d5e7),
        f32::from_bits(0xbaf80701),
        f32::from_bits(0xb9828894),
        f32::from_bits(0x3ca77d72),
        f32::from_bits(0x3f7e5131),
        f32::from_bits(0xbd8df5f8),
        f32::from_bits(0x3dba92eb),
        f32::from_bits(0x3a8bd267),
        f32::from_bits(0xbc37c50e),
        f32::from_bits(0x3d1e3cc6),
        f32::from_bits(0xbd2a12ca),
        f32::from_bits(0xbf75b6c8),
        f32::from_bits(0xbe8e17f1),
    ]);
    let mut destination = [0.0_f32; 12];
    let destination_before = destination;
    diagnostic_row_major_pose_block_add_assign(&mut destination, left, right);

    assert_f32_bits(&destination_before, &[0x00000000; 12]);
    assert_f32_bits(
        &destination,
        &[
            0x4216d5cf, 0x4230b65b, 0x40925ef6, 0x4312a950, 0xc1f31d9c, 0xc464e41e, 0x4129c6d0,
            0xbf5c5301, 0xc2725b87, 0xc41de71f, 0xc43980fe, 0xc2c8260a,
        ],
    );
}

#[test]
fn visual_prefix_filter_defaults_to_all_events_and_selects_exact_context() {
    let all = visual_prefix_trace_filter_from_cached(None, None, None).unwrap();
    assert!(all.matches(None, None, 0));
    assert!(all.matches(Some(12), Some(0), 0));

    let selected =
        visual_prefix_trace_filter_from_cached(Some(Ok(12)), Some(Ok(0)), Some(Ok(0))).unwrap();
    assert!(selected.matches(Some(12), Some(0), 0));
    assert!(!selected.matches(Some(11), Some(0), 0));
    assert!(!selected.matches(Some(12), Some(1), 0));
    assert!(!selected.matches(Some(12), Some(0), 1));
    assert!(!selected.matches(None, Some(0), 0));
}

#[test]
fn visual_prefix_filter_rejects_malformed_values() {
    for (frame_id, iteration, trial) in [
        (Some(Err(())), None, None),
        (Some(Err(())), None, None),
        (None, Some(Err(())), None),
        (None, None, Some(Err(()))),
    ] {
        let result = visual_prefix_trace_filter_from_cached(frame_id, iteration, trial);
        assert!(result.is_err(), "malformed selector must fail closed");
    }
}

#[test]
fn visual_prefix_filter_non_target_skips_open_and_materialization() {
    let selected =
        visual_prefix_trace_filter_from_cached(Some(Ok(12)), Some(Ok(0)), Some(Ok(0))).unwrap();
    let factor = WhitenedFactorRowStack::new(
        DMatrix::from_row_slice(3, 2, &[1.0, 0.25, -0.5, 2.0, 0.75, -1.25]),
        DMatrix::from_row_slice(3, 1, &[0.5, -1.0, 2.0]),
        DVector::from_column_slice(&[0.25, -0.75, 1.5]),
    )
    .expect("visual sidecar fixture")
    .with_kind(FactorKind::Visual)
    .with_landmark_metadata(7, 701)
    .with_visual_observation_ids(vec![(0, 0), (2, 1)]);

    // An empty path would fail if the writer were opened.  The selector
    // must short-circuit before path validation, factor metadata copying,
    // or any sidecar materialization for a non-target event.
    let path = visual_prefix_test_path("non_target");
    let canonical = canonical_visual_prefix_trace_key(&path).expect("canonical test path");
    let lock_path = visual_prefix_trace_lock_path(&canonical).expect("lock test path");
    let result = visual_prefix_trace_writer_selected(
        selected,
        Some(11),
        Some(0),
        0,
        Some(&path),
        std::slice::from_ref(&factor),
        2,
    )
    .expect("non-target events are successful no-ops");
    assert!(result.is_none());
    assert!(!path.exists(), "non-target event must not create target");
    assert!(!lock_path.exists(), "non-target event must not create lock");
}

#[test]
fn visual_prefix_trace_records_exact_prefix_and_prior_boundaries() {
    let factor = WhitenedFactorRowStack::new(
        DMatrix::from_row_slice(3, 2, &[1.0, 0.25, -0.5, 2.0, 0.75, -1.25]),
        DMatrix::from_row_slice(3, 1, &[0.5, -1.0, 2.0]),
        DVector::from_column_slice(&[0.25, -0.75, 1.5]),
    )
    .expect("visual sidecar fixture")
    .with_kind(FactorKind::Visual)
    .with_landmark_metadata(7, 701)
    .with_visual_observation_ids(vec![(0, 0), (2, 1)]);
    let factors = vec![factor];
    let path = std::env::temp_dir().join(format!(
        "visloc_visual_prefix_trace_{}_{}.jsonl",
        std::process::id(),
        VISUAL_PREFIX_TRACE_EVENT.fetch_add(1, Ordering::Relaxed)
    ));
    set_active_diagnostic_lm_frame(Some(42));
    set_active_diagnostic_lm_iteration(Some(3));
    let mut writer = VisualPrefixTraceWriter::open(path.clone(), &factors, 2)
        .expect("sidecar writer")
        .expect("sidecar enabled fixture");
    let (projected, projected_residual, rank) =
        landmark_nullspace_projection_f32(&factors[0], 1e-10);
    let mut visual_h = DMatrix::<f32>::zeros(2, 2);
    let mut visual_b = DVector::<f32>::zeros(2);
    let pending = writer
        .begin_visual_prefix(
            0,
            &factors[0],
            &projected,
            &projected_residual,
            rank,
            &visual_h,
            &visual_b,
        )
        .expect("visual prefix metadata");
    visual_h += projected.transpose() * &projected;
    accumulate_transpose_vector_f32_eigen(&mut visual_b, &projected, &projected_residual, false);
    writer
        .finish_visual_prefix(pending, &visual_h, &visual_b)
        .expect("visual prefix record");
    writer
        .write_stage("visual_total", &visual_h, &visual_b)
        .expect("visual total record");
    let prior_h = DMatrix::<f32>::zeros(2, 2);
    let prior_b = DVector::<f32>::zeros(2);
    writer
        .write_prior_before(&visual_h, &visual_b, &prior_h, &prior_b)
        .expect("prior boundary record");
    writer
        .write_stage("prior_after", &visual_h, &visual_b)
        .expect("prior after record");
    writer
        .write_stage("final", &visual_h, &visual_b)
        .expect("final record");
    drop(writer);
    set_active_diagnostic_lm_frame(None);
    set_active_diagnostic_lm_iteration(None);

    let lines = std::fs::read_to_string(&path)
        .expect("sidecar output")
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("JSONL record"))
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 6);
    assert_eq!(lines[0]["record"], "header");
    assert_eq!(lines[1]["record"], "visual_prefix");
    assert_eq!(lines[1]["frame_id"], 42);
    assert_eq!(lines[1]["iteration"], 3);
    assert_eq!(lines[1]["visual_ordinal"], 0);
    assert_eq!(lines[1]["factor_index"], 0);
    assert_eq!(lines[1]["track_id"], 701);
    assert_eq!(lines[1]["observations"].as_array().unwrap().len(), 2);
    assert_eq!(lines[2]["stage"], "visual_total");
    assert_eq!(lines[3]["stage"], "prior_before");
    assert_eq!(lines[4]["stage"], "prior_after");
    assert_eq!(lines[5]["stage"], "final");
    assert_eq!(lines[1]["global_h_after"], diagnostic_f32_matrix(&visual_h));
    assert_eq!(lines[1]["global_b_after"], diagnostic_f32_vector(&visual_b));
    std::fs::remove_file(path).expect("remove sidecar fixture");
}

#[test]
fn visual_prefix_trace_rejects_missing_observation_identity() {
    let factor = WhitenedFactorRowStack::new(
        DMatrix::from_row_slice(3, 1, &[1.0, 2.0, 3.0]),
        DMatrix::from_row_slice(3, 1, &[1.0, 0.5, -0.25]),
        DVector::from_column_slice(&[0.25, -0.75, 1.5]),
    )
    .expect("visual sidecar fixture")
    .with_kind(FactorKind::Visual)
    .with_landmark_metadata(3, 300);
    let path = std::env::temp_dir().join(format!(
        "visloc_visual_prefix_trace_invalid_{}_{}.jsonl",
        std::process::id(),
        VISUAL_PREFIX_TRACE_EVENT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut writer = VisualPrefixTraceWriter::open(path.clone(), &[factor.clone()], 1)
        .expect("sidecar writer")
        .expect("sidecar enabled fixture");
    let (projected, residual, rank) = landmark_nullspace_projection_f32(&factor, 1e-10);
    let h = DMatrix::<f32>::zeros(1, 1);
    let b = DVector::<f32>::zeros(1);
    assert!(matches!(
        writer.begin_visual_prefix(0, &factor, &projected, &residual, rank, &h, &b),
        Err(ImuReductionError::VisualPrefixTraceInvalid { index: 0 })
    ));
    drop(writer);
    std::fs::remove_file(path).expect("remove invalid sidecar fixture");
}

#[test]
fn visual_prefix_trace_rejects_empty_output_path() {
    let factor = WhitenedFactorRowStack::new(
        DMatrix::zeros(0, 0),
        DMatrix::zeros(0, 0),
        DVector::zeros(0),
    )
    .expect("empty fixture");
    assert!(matches!(
        VisualPrefixTraceWriter::open(std::path::PathBuf::new(), &[factor], 0),
        Err(ImuReductionError::VisualPrefixTraceIo)
    ));
}

fn visual_prefix_test_path(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "visloc_visual_prefix_{label}_{}_{}.jsonl",
        std::process::id(),
        VISUAL_PREFIX_TRACE_EVENT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn visual_prefix_disabled_reducer_child() {
    if std::env::var_os("VISLOC_VISUAL_PREFIX_DISABLED_CHILD").is_none() {
        return;
    }
    let factor = WhitenedFactorRowStack::new(
        DMatrix::from_row_slice(3, 1, &[1.0, -2.0, 3.0]),
        DMatrix::from_row_slice(3, 1, &[0.5, 1.0, -0.25]),
        DVector::from_column_slice(&[0.25, -0.75, 1.5]),
    )
    .expect("lean visual fixture")
    .with_kind(FactorKind::Visual)
    .with_landmark_metadata(1, 101)
    .with_visual_observation_ids(vec![(0, 0)]);
    let before = VISUAL_PREFIX_TRACE_OPEN_ATTEMPTS.load(Ordering::Relaxed);
    let reduced =
        reduce_landmark_factors_f32_checked_without_back_substitution(&[factor], 1, 1e-10)
            .expect("lean reducer");
    assert!(!crate::vio::window::diagnostic_env_active());
    assert_eq!(
        VISUAL_PREFIX_TRACE_OPEN_ATTEMPTS.load(Ordering::Relaxed),
        before,
        "disabled reducer must not open a visual-prefix writer"
    );
    assert!(reduced.diagnostic_stages.is_none());
    assert!(reduced.imu_diagnostic.is_none());
}

#[test]
fn visual_prefix_disabled_reducer_has_no_sidecar_work() {
    let executable = std::env::current_exe().expect("test executable path");
    let mut command = std::process::Command::new(executable);
    for (key, _) in std::env::vars_os() {
        if key
            .to_string_lossy()
            .get(0.."VISLOC_BASALT_".len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("VISLOC_BASALT_"))
        {
            command.env_remove(key);
        }
    }
    let output = command
        .env("VISLOC_VISUAL_PREFIX_DISABLED_CHILD", "1")
        .args(["visual_prefix_disabled_reducer_child", "--nocapture"])
        .output()
        .expect("spawn clean reducer child");
    assert!(
        output.status.success(),
        "clean reducer child failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn visual_prefix_writer_cross_process_lock_child() {
    let Some(role) = std::env::var_os("VISLOC_VISUAL_PREFIX_LOCK_CHILD") else {
        return;
    };
    let target = std::path::PathBuf::from(
        std::env::var_os("VISLOC_VISUAL_PREFIX_LOCK_TARGET").expect("lock target"),
    );
    let ready = std::path::PathBuf::from(
        std::env::var_os("VISLOC_VISUAL_PREFIX_LOCK_READY").expect("lock ready"),
    );
    let release = std::path::PathBuf::from(
        std::env::var_os("VISLOC_VISUAL_PREFIX_LOCK_RELEASE").expect("lock release"),
    );
    match role.to_string_lossy().as_ref() {
        "holder" => {
            let writer = VisualPrefixTraceWriter::open(&target, &[], 0)
                .expect("holder open")
                .expect("holder owns cross-process lock");
            std::fs::write(&ready, b"holder-ready").expect("holder ready");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !release.exists() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "holder release timeout"
                );
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            drop(writer);
        }
        "contender" => {
            let result = VisualPrefixTraceWriter::open(&target, &[], 0);
            assert!(
                matches!(result, Err(ImuReductionError::VisualPrefixTraceIo)),
                "existing cross-process lock must reject contender"
            );
            std::fs::write(&ready, b"contender-rejected").expect("contender ready");
        }
        _ => panic!("unknown lock child role"),
    }
}

#[test]
fn visual_prefix_writer_cross_process_lock_and_stale_lock_rejection() {
    fn spawn_child(
        role: &str,
        target: &std::path::Path,
        ready: &std::path::Path,
        release: &std::path::Path,
    ) -> std::process::Child {
        let executable = std::env::current_exe().expect("test executable path");
        let mut command = std::process::Command::new(executable);
        for (key, _) in std::env::vars_os() {
            let key = key.to_string_lossy();
            if key
                .get(0.."VISLOC_BASALT_".len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("VISLOC_BASALT_"))
            {
                command.env_remove(key.as_ref());
            }
        }
        command
            .env("VISLOC_VISUAL_PREFIX_LOCK_CHILD", role)
            .env("VISLOC_VISUAL_PREFIX_LOCK_TARGET", target)
            .env("VISLOC_VISUAL_PREFIX_LOCK_READY", ready)
            .env("VISLOC_VISUAL_PREFIX_LOCK_RELEASE", release)
            .args([
                "visual_prefix_writer_cross_process_lock_child",
                "--nocapture",
            ])
            .spawn()
            .expect("spawn lock child")
    }

    fn wait_for_file(path: &std::path::Path) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !path.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        path.exists()
    }

    let target = visual_prefix_test_path("cross_process");
    let holder_ready = target.with_extension("holder.ready");
    let contender_ready = target.with_extension("contender.ready");
    let release = target.with_extension("release");
    let canonical = canonical_visual_prefix_trace_key(&target).expect("canonical target");
    let lock_path = visual_prefix_trace_lock_path(&canonical).expect("lock path");
    let mut holder = spawn_child("holder", &target, &holder_ready, &release);
    if !wait_for_file(&holder_ready) {
        let _ = holder.kill();
        let _ = holder.wait();
        panic!("holder did not acquire lock");
    }
    let payload = std::fs::read_to_string(&lock_path).expect("lock payload");
    let payload: serde_json::Value = serde_json::from_str(&payload).expect("lock JSON");
    assert_eq!(payload["schema"], "basalt.m11.visual_prefix_trace_lock.v1");
    assert_eq!(payload["run_id"].as_str().map(str::len), Some(32));
    assert!(payload["pid"].as_u64().is_some());
    assert_eq!(
        payload["target_path_hash"],
        format!("{:016x}", visual_prefix_path_hash(&canonical))
    );

    let mut contender = spawn_child("contender", &target, &contender_ready, &release);
    if !wait_for_file(&contender_ready) {
        let _ = contender.kill();
        let _ = contender.wait();
        let _ = std::fs::write(&release, b"release");
        let _ = holder.wait();
        panic!("contender did not report");
    }
    assert!(contender.wait().expect("wait contender").success());
    std::fs::write(&release, b"release").expect("release holder");
    assert!(holder.wait().expect("wait holder").success());
    assert!(!lock_path.exists(), "live lease must remove its lock");

    let writer = VisualPrefixTraceWriter::open(&target, &[], 0)
        .expect("reacquire after cleanup")
        .expect("reacquire writer");
    drop(writer);
    assert!(!lock_path.exists(), "reacquired lease must clean lock");

    std::fs::write(&lock_path, br#"{"schema":"stale"}"#).expect("stale lock");
    let rejected = VisualPrefixTraceWriter::open(&target, &[], 0);
    assert!(matches!(
        rejected,
        Err(ImuReductionError::VisualPrefixTraceIo)
    ));
    assert!(lock_path.exists(), "stale lock must not be auto-removed");

    for path in [target, holder_ready, contender_ready, release, lock_path] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn visual_prefix_writer_maps_write_failures_to_fail_closed_error() {
    let path = visual_prefix_test_path("write_failure");
    let mut writer = VisualPrefixTraceWriter::open(&path, &[], 0)
        .expect("writer open")
        .expect("writer enabled");
    let read_only = std::fs::OpenOptions::new()
        .read(true)
        .open(&path)
        .expect("read-only replacement handle");
    let writable = std::mem::replace(&mut writer.file, read_only);
    drop(writable);
    assert!(matches!(
        writer.write_record(&json!({"record": "write_failure"})),
        Err(ImuReductionError::VisualPrefixTraceIo)
    ));
    drop(writer);
    std::fs::remove_file(path).expect("remove writer failure fixture");
}

#[test]
fn visual_prefix_writer_rejects_same_canonical_path_concurrently() {
    use std::sync::{mpsc, Arc, Barrier};

    let path = visual_prefix_test_path("same_path");
    let start = Arc::new(Barrier::new(3));
    let release = Arc::new(Barrier::new(3));
    let (sender, receiver) = mpsc::channel();
    let mut workers = Vec::new();
    for _ in 0..2 {
        let path = path.clone();
        let start = Arc::clone(&start);
        let release = Arc::clone(&release);
        let sender = sender.clone();
        workers.push(std::thread::spawn(move || {
            start.wait();
            let result = VisualPrefixTraceWriter::open(path, &[], 0);
            sender
                .send(result.as_ref().is_ok_and(Option::is_some))
                .expect("send writer result");
            release.wait();
            drop(result);
        }));
    }
    start.wait();
    let mut opened = [
        receiver.recv().expect("first writer result"),
        receiver.recv().expect("second writer result"),
    ];
    opened.sort_unstable();
    assert_eq!(opened, [false, true]);
    release.wait();
    for worker in workers {
        worker.join().expect("join same-path writer");
    }
    drop(sender);
    std::fs::remove_file(path).expect("remove same-path fixture");
}

#[test]
fn visual_prefix_writer_allows_distinct_canonical_paths_concurrently() {
    use std::sync::{mpsc, Arc, Barrier};

    let paths = [
        visual_prefix_test_path("distinct_a"),
        visual_prefix_test_path("distinct_b"),
    ];
    let start = Arc::new(Barrier::new(3));
    let release = Arc::new(Barrier::new(3));
    let (sender, receiver) = mpsc::channel();
    let mut workers = Vec::new();
    for path in paths.iter().cloned() {
        let start = Arc::clone(&start);
        let release = Arc::clone(&release);
        let sender = sender.clone();
        workers.push(std::thread::spawn(move || {
            start.wait();
            let result = VisualPrefixTraceWriter::open(path, &[], 0);
            sender
                .send(result.as_ref().is_ok_and(Option::is_some))
                .expect("send writer result");
            release.wait();
            drop(result);
        }));
    }
    start.wait();
    assert!(receiver.recv().expect("first distinct writer result"));
    assert!(receiver.recv().expect("second distinct writer result"));
    release.wait();
    for worker in workers {
        worker.join().expect("join distinct-path writer");
    }
    drop(sender);
    for path in paths {
        std::fs::remove_file(path).expect("remove distinct-path fixture");
    }
}

#[test]
fn visual_prefix_writer_binds_run_context_and_is_thread_local() {
    set_active_diagnostic_lm_frame(None);
    set_active_diagnostic_lm_iteration(None);
    let guard = begin_diagnostic_lm_run();
    let run_id = active_diagnostic_lm_run_id().expect("run id");
    set_active_diagnostic_lm_frame(Some(17));
    set_active_diagnostic_lm_iteration(Some(2));
    assert_eq!(active_diagnostic_lm_run_id(), Some(run_id));
    assert_eq!(active_diagnostic_lm_frame(), Some(17));
    assert_eq!(active_diagnostic_lm_iteration(), Some(2));
    std::thread::spawn(|| {
        assert_eq!(active_diagnostic_lm_run_id(), None);
        assert_eq!(active_diagnostic_lm_frame(), None);
        assert_eq!(active_diagnostic_lm_iteration(), None);
    })
    .join()
    .expect("join thread-local context check");
    drop(guard);
    assert_eq!(active_diagnostic_lm_run_id(), None);
    assert_eq!(active_diagnostic_lm_frame(), None);
    assert_eq!(active_diagnostic_lm_iteration(), None);
}

/// Builds the 70-factor (1 prior + 61 visual landmark + 4x(IMU+bias))
/// frame-4 fixture shared by [`m11_full70_frame4_ordered_f32_legacy_compact_model_recovery_parity`]
/// and the parallel-landmark-reduction bit-identity test below. Kept as
/// its own helper so both tests build the exact same factor set from
/// one source of truth instead of two independently hand-maintained
/// copies drifting apart.
fn build_full70_frame4_factors() -> (Vec<WhitenedFactorRowStack>, usize) {
    let state_dof = 75;
    let prior_jacobian = DMatrix::from_fn(15, state_dof, |row, column| {
        if column == row {
            1.0 + row as f64 * 0.015625
        } else if column < 15 {
            (row + column + 1) as f64 * 0.00390625
        } else {
            0.0
        }
    });
    let prior_residual = DVector::from_fn(15, |row, _| (row as f64 - 7.0) * 0.03125);
    let prior = WhitenedFactorRowStack::with_objective_cost_kind(
        prior_jacobian,
        DMatrix::zeros(15, 0),
        prior_residual.clone(),
        0.5 * prior_residual.norm_squared(),
        FactorKind::Prior,
    )
    .unwrap()
    .with_prior_state_columns((0..state_dof).collect());
    let mut factors = vec![prior];
    for landmark_index in 0..61 {
        let rows = if landmark_index < 9 { 20 } else { 19 };
        let state_jacobian = DMatrix::from_fn(rows, state_dof, |row, column| {
            if column < 15 {
                ((row + 1) * (column + 3) % 17) as f64 * 0.015625
            } else {
                0.0
            }
        });
        let landmark_jacobian = DMatrix::from_fn(rows, 3, |row, column| {
            if row % 3 == column {
                1.0 + (row + column) as f64 * 0.0078125
            } else {
                ((row + column + 1) % 7) as f64 * 0.03125
            }
        });
        let residual = DVector::from_fn(rows, |row, _| {
            ((row + landmark_index + 1) % 13) as f64 * 0.03125 - 0.125
        });
        let factor = WhitenedFactorRowStack::with_objective_cost_kind(
            state_jacobian,
            landmark_jacobian,
            residual.clone(),
            0.5 * residual.norm_squared(),
            FactorKind::Visual,
        )
        .unwrap()
        .with_landmark_metadata(landmark_index, 10_000 + landmark_index as u64)
        .with_visual_observation_ids(vec![(0, 0), (1, 1), (2, 0)]);
        factors.push(factor);
    }
    for link in 0..4 {
        let offsets = ImuLinkOffsets {
            start: link * AOM_NAV_DOF,
            end: (link + 1) * AOM_NAV_DOF,
        };
        let mut imu_jacobian = DMatrix::<f64>::zeros(9, state_dof);
        for row in 0..9 {
            for local_column in 0..AOM_NAV_DOF * 2 {
                let offset = if local_column < AOM_NAV_DOF {
                    offsets.start
                } else {
                    offsets.end
                };
                imu_jacobian[(row, offset + local_column % AOM_NAV_DOF)] =
                    (1 + row + local_column + link) as f64 * 0.0078125;
            }
        }
        let imu_residual = DVector::from_fn(9, |row, _| (row + link + 1) as f64 * 0.03125);
        let imu = WhitenedFactorRowStack::with_objective_cost_kind(
            imu_jacobian,
            DMatrix::zeros(9, 0),
            imu_residual.clone(),
            0.5 * imu_residual.norm_squared(),
            FactorKind::Imu,
        )
        .unwrap()
        .with_imu_link_offsets(offsets.start, offsets.end);

        let mut bias_jacobian = DMatrix::<f64>::zeros(6, state_dof);
        for row in 0..6 {
            for local_column in 0..AOM_NAV_DOF * 2 {
                let offset = if local_column < AOM_NAV_DOF {
                    offsets.start
                } else {
                    offsets.end
                };
                bias_jacobian[(row, offset + local_column % AOM_NAV_DOF)] =
                    (2 + row + local_column + link) as f64 * 0.00390625;
            }
        }
        let bias_residual = DVector::from_fn(6, |row, _| -0.0625 + (row + link) as f64 * 0.015625);
        let bias = WhitenedFactorRowStack::with_objective_cost_kind(
            bias_jacobian,
            DMatrix::zeros(6, 0),
            bias_residual.clone(),
            0.5 * bias_residual.norm_squared(),
            FactorKind::Bias,
        )
        .unwrap()
        .with_imu_link_offsets(offsets.start, offsets.end);
        factors.push(imu);
        factors.push(bias);
    }

    validate_full70_frame4_factors(&factors, state_dof).expect("strict frame-4 factor order");
    (factors, state_dof)
}

#[test]
fn m11_full70_frame4_ordered_f32_legacy_compact_model_recovery_parity() {
    let (factors, state_dof) = build_full70_frame4_factors();
    let legacy = reduce_landmark_factors_f32_checked(&factors, state_dof, 1e-10)
        .expect("legacy f32 frame-4 reduction");
    let compact = reduce_landmark_factors_f32_checked_with_compact_back_substitution(
        &factors, state_dof, 1e-10,
    )
    .expect("compact f32 frame-4 reduction");
    assert!(full70_matrix_bits_match(&legacy.h, &compact.h));
    assert!(full70_vector_bits_match(&legacy.b, &compact.b));
    assert_eq!(legacy.back_substitution.len(), 70);
    let compact_batch = compact
        .compact_back_substitution
        .as_ref()
        .expect("all 61 visual factors have compact entries");
    assert_eq!(compact_batch.entries.len(), 61);
    assert_eq!(compact_batch.entries[0].landmark_index, 0);
    assert_eq!(compact_batch.entries[60].landmark_index, 60);

    let state_step = DVector::from_fn(state_dof, |column, _| (column as f64 - 23.0) * 0.0078125);
    for (entry_index, entry) in compact_batch.entries.iter().enumerate() {
        let factor_index = entry_index + 1;
        let legacy_step = back_substitute_landmark_f32_with_track(
            &legacy.back_substitution[factor_index],
            &state_step,
            1e-10,
            Some(entry.track_id),
        )
        .expect("legacy visual recovery");
        let compact_step = back_substitute_landmark_compact_entry_f32(
            entry,
            &compact_batch.storage,
            &state_step,
            1e-10,
        )
        .expect("compact visual recovery");
        assert_eq!(legacy_step.len(), compact_step.len());
        for (lane, (legacy, compact)) in legacy_step.iter().zip(compact_step.iter()).enumerate() {
            assert_eq!(
                (*legacy as f32).to_bits(),
                (*compact as f32).to_bits(),
                "frame-4 recovery factor {entry_index} lane {lane}"
            );
        }
    }

    // The model evaluator is exposed as a complete transformed-row path,
    // while the compact reducer exposes Q1/R recovery.  Build the exact
    // Q1/Q2 payloads from the same f32 factor boundary and verify the
    // already-computed compact payload's model contract for all 70 source
    // factors, preserving source factor order (including IMU/Bias pairs).
    let model_payloads = factors
        .iter()
        .map(|factor| {
            if factor.landmark_jacobian.ncols() == 0 {
                ModelReusePayloadF32::Plain {
                    kind: factor.kind,
                    state: as_f32_matrix(&factor.state_jacobian),
                    residual: as_f32_vector(&factor.residual),
                }
            } else {
                let state = as_f32_matrix(&factor.state_jacobian);
                let landmark = as_f32_matrix(&factor.landmark_jacobian);
                let residual = as_f32_vector(&factor.residual);
                let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual)
                    .expect("model payload QR");
                let rank = (0..landmark.ncols())
                    .filter(|&index| qr.pivots[index].abs() > 1e-10_f32)
                    .count();
                let metadata = factor.landmark_metadata.expect("visual metadata");
                ModelReusePayloadF32::Visual(QrModelReusePayloadF32 {
                    q1: qr
                        .compact_back_substitution(
                            metadata.landmark_index,
                            metadata.track_id,
                            rank,
                            rank == landmark.ncols(),
                        )
                        .expect("model payload compact Q1"),
                    q2_state: qr.q2_state(),
                    q2_residual: qr.q2_residual(),
                })
            }
        })
        .collect::<Vec<_>>();
    let full_model = model_cost_decrease_f32(&factors, &state_step, 1e-10)
        .expect("complete transformed-row model");
    let payload_model =
        model_cost_decrease_from_payloads_f32(&model_payloads, &factors, &state_step, 1e-10)
            .expect("compact Q1/Q2 model payload");
    assert_eq!(
        (full_model as f32).to_bits(),
        (payload_model as f32).to_bits()
    );

    let linearization = LmLinearization { factors, cost: 0.0 };
    let state = DVector::zeros(state_dof);
    let trial_state = &state + &state_step;
    let damping = DVector::zeros(state_dof);
    let reduced = legacy.as_f64();
    let event = LmDiagnosticEvent {
        iteration: 0,
        trial: 0,
        phase: "iteration_start",
        lambda: 1e-4,
        lambda_after: 1e-4,
        cost_before: 0.0,
        model_cost: Some(full_model),
        model_decrease: None,
        actual_cost: Some(full_model),
        step_norm: Some(state_step.norm()),
        decision: "pending",
        base_state: &state,
        state: &state,
        trial_state: Some(&trial_state),
        step: Some(&state_step),
        damping_diag: &damping,
        damped_h: None,
        linearization: &linearization,
        reduced: &reduced,
    };
    let payload = full70_oracle_payload(
        4,
        &event,
        &json!({ "state_blocks": [], "prior_source": { "source": "test" } }),
        &json!({ "status": "test_bound" }),
    )
    .expect("full frame-4 oracle payload");
    assert_eq!(
        payload["schema"].as_str(),
        Some("basalt.m11.full70_factor_oracle.v1")
    );
    assert_eq!(payload["factor_count"].as_u64(), Some(70));
    assert_eq!(payload["row_count"].as_u64(), Some(1243));
    assert_eq!(payload["visual_factor_count"].as_u64(), Some(61));
    assert_eq!(payload["reducer"]["h_bitwise_equal"].as_bool(), Some(true));
    assert_eq!(payload["reducer"]["b_bitwise_equal"].as_bool(), Some(true));
    assert_eq!(payload["factors"][0]["kind"].as_str(), Some("Prior"));
    assert_eq!(payload["factors"][1]["kind"].as_str(), Some("Visual"));
    assert_eq!(payload["factors"][62]["kind"].as_str(), Some("Imu"));
    assert_eq!(payload["factors"][63]["kind"].as_str(), Some("Bias"));
    assert_eq!(
        payload["factors"][62]["imu_local_15x30"]["shape"],
        json!([15, 30])
    );
    assert_eq!(
        payload["recovery"]["status"].as_str(),
        Some("captured_from_same_ordered_rows")
    );
    assert!(payload["recovery"]["landmark_steps"]
        .as_array()
        .is_some_and(|steps| steps.iter().all(|step| step["bitwise_equal"] == true)));
}

/// Proves the per-landmark parallel H/b contribution pre-pass in
/// `reduce_landmark_factors_f32_checked_with_options` is bit-identical
/// regardless of the rayon thread-pool size it runs under. Reuses the
/// real 70-factor (1 prior + 61 visual landmark + 4x(IMU+bias)) frame-4
/// fixture from [`m11_full70_frame4_ordered_f32_legacy_compact_model_recovery_parity`]
/// as the dense H/b oracle: the reduction is run inside three separately
/// scoped rayon thread pools (1, 4, and 8 threads -- 1 thread forces the
/// same effectively-serial execution order the pre-parallelization code
/// always used), and the resulting `h`/`b` (plus the derived compact
/// back-substitution landmark recovery and the model cost decrease used
/// by the LM trial step) must match exactly, to the bit, across all
/// three. Landmark contributions are pure functions of their own
/// already-projected `(jacobian, residual)`, folded into `visual_h` /
/// `visual_b` by a `+=` sequence that is unconditionally serial and in
/// original factor-index order (see the comment above that fold in
/// `reduce_landmark_factors_f32_checked_with_options`), so thread count
/// must not be observable in the result.
#[test]
fn m11_full70_frame4_parallel_landmark_reduction_is_thread_count_invariant() {
    let (factors, state_dof) = build_full70_frame4_factors();
    let state_step = DVector::from_fn(state_dof, |column, _| (column as f64 - 23.0) * 0.0078125);

    let mut previous: Option<(ReducedNormalSystemF32, f64, Vec<Vec<f64>>)> = None;
    for threads in [1usize, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("scoped rayon pool");
        let reduced = pool
            .install(|| {
                reduce_landmark_factors_f32_checked_with_compact_back_substitution(
                    &factors, state_dof, 1e-10,
                )
            })
            .expect("f32 frame-4 reduction");

        // The model cost decrease consumes the same per-factor Q1/Q2
        // payloads the parallel pre-pass feeds from; check it is also
        // unaffected, since an LM trial step depends on it every
        // iteration.
        let model_decrease = pool
            .install(|| model_cost_decrease_f32(&factors, &state_step, 1e-10))
            .expect("model cost decrease");
        assert!(
            model_decrease.is_finite(),
            "model decrease not finite at threads={threads}"
        );

        let compact_batch = reduced
            .compact_back_substitution
            .as_ref()
            .expect("all 61 visual factors have compact entries");
        assert_eq!(compact_batch.entries.len(), 61);
        let recovered_steps: Vec<Vec<f64>> = compact_batch
            .entries
            .iter()
            .map(|entry| {
                let step = pool
                    .install(|| {
                        back_substitute_landmark_compact_entry_f32(
                            entry,
                            &compact_batch.storage,
                            &state_step,
                            1e-10,
                        )
                    })
                    .expect("compact visual recovery");
                assert!(
                    step.iter().all(|value| value.is_finite()),
                    "landmark recovery not finite at threads={threads}"
                );
                step.iter().copied().collect::<Vec<f64>>()
            })
            .collect();

        if let Some((previous_reduced, previous_decrease, previous_steps)) = previous.as_ref() {
            assert!(
                full70_matrix_bits_match(&previous_reduced.h, &reduced.h),
                "H differs at threads={threads}"
            );
            assert!(
                full70_vector_bits_match(&previous_reduced.b, &reduced.b),
                "b differs at threads={threads}"
            );
            assert_eq!(
                previous_decrease.to_bits(),
                model_decrease.to_bits(),
                "model cost decrease differs at threads={threads}"
            );
            assert_eq!(previous_steps.len(), recovered_steps.len());
            for (entry_index, (previous_step, step)) in
                previous_steps.iter().zip(&recovered_steps).enumerate()
            {
                assert_eq!(previous_step.len(), step.len());
                for (lane, (previous_value, value)) in previous_step.iter().zip(step).enumerate() {
                    assert_eq!(
                        previous_value.to_bits(),
                        value.to_bits(),
                        "landmark recovery factor {entry_index} lane {lane} differs at threads={threads}"
                    );
                }
            }
        }

        previous = Some((reduced, model_decrease, recovered_steps));
    }
}

#[test]
fn m11_full70_oracle_rejects_count_and_nonfinite_inputs() {
    assert!(matches!(
        validate_full70_frame4_factors(&[], 75),
        Err(ImuReductionError::Full70OracleInvalid { .. })
    ));
    assert!(matches!(
        full70_checked_f32_bits(f64::NAN, 3),
        Err(ImuReductionError::Full70OracleInvalid { index: 3 })
    ));
    assert!(!full70_hash_like("not-a-hash"));
    assert!(full70_hash_like(&"a".repeat(64)));
}

#[test]
fn m11_retry11_native_operands_replay_current_chain_stagewise() {
    struct NativeRelPoseFixture {
        name: &'static str,
        state_h: [u32; 7],
        state_t: [u32; 7],
        extrinsic_h: [u32; 7],
        extrinsic_t: [u32; 7],
        after_inverse: [u32; 7],
        relative_q: [u32; 4],
        relative_t: [u32; 3],
        before_adj: [u32; 7],
        after_adj: [u32; 36],
        after_adj2: [u32; 36],
        returned: [u32; 7],
    }

    fn pose(words: [u32; 7]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(words[3]),
                f32::from_bits(words[0]),
                f32::from_bits(words[1]),
                f32::from_bits(words[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(words[4]),
                f32::from_bits(words[5]),
                f32::from_bits(words[6]),
            ),
        }
    }

    fn pose_bits(value: F32Pose) -> [u32; 7] {
        let q = value.rotation.quaternion();
        [
            q.i.to_bits(),
            q.j.to_bits(),
            q.k.to_bits(),
            q.w.to_bits(),
            value.translation.x.to_bits(),
            value.translation.y.to_bits(),
            value.translation.z.to_bits(),
        ]
    }

    fn adjoint_bits(value: F32Pose) -> Vec<u32> {
        // This is the same fixed Eigen block construction used by the
        // production d_rel helper, without the right-hand rotation
        // product. The native retry11 dump is the standalone Adj() result.
        let rotation = eigen_quaternion_matrix_f32(value.rotation);
        let cross = eigen_matrix_product_3x3_f32(skew3_f32(value.translation), rotation);
        let mut adjoint = SMatrix::<f32, 6, 6>::zeros();
        adjoint.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
        adjoint.fixed_view_mut::<3, 3>(0, 3).copy_from(&cross);
        adjoint.fixed_view_mut::<3, 3>(3, 3).copy_from(&rotation);
        adjoint
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    fn record_stage(
        fixture: &str,
        stage: &str,
        actual: &[u32],
        expected: &[u32],
        failures: &mut Vec<String>,
    ) {
        assert_eq!(actual.len(), expected.len(), "{fixture} {stage} length");
        let mismatches = actual
            .iter()
            .zip(expected)
            .enumerate()
            .filter(|(_, (actual, expected))| actual != expected)
            .collect::<Vec<_>>();
        println!(
            "M11_V7_STAGE fixture={fixture} stage={stage} exact={}/{}",
            actual.len() - mismatches.len(),
            actual.len()
        );
        if let Some((index, (actual, expected))) = mismatches.first() {
            failures.push(format!(
                "{fixture} {stage} first lane {index}: got {actual:08x}, expected {expected:08x} ({} mismatches)",
                mismatches.len()
            ));
        }
    }

    let fixtures = [
        NativeRelPoseFixture {
            name: "ordinal8_frame4_cam0_obs8",
            state_h: [
                0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd, 0x00000000, 0x00000000, 0x00000000,
            ],
            state_t: [
                0xbd5f79e6, 0xbf4f45a2, 0xbbb53f68, 0x3f159719, 0x3c02c75a, 0xbb0c3bda, 0xbcdc2b7f,
            ],
            extrinsic_h: [
                0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc, 0x3ba86617,
            ],
            extrinsic_t: [
                0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc, 0x3ba86617,
            ],
            after_inverse: [
                0x3bed3c0f, 0xbbf71cd4, 0xbf33a827, 0x3f365a1e, 0x3d8ddf2e, 0xbc810144, 0xbb71aa03,
            ],
            relative_q: [0x3bc5abb3, 0x3c4782fd, 0x3b130617, 0x3f7ff9c9],
            // retry12 moved the action breakpoint after the native
            // SO3 store.  The old rich3 value was captured one
            // instruction before that store and is intentionally not
            // used as the endpoint oracle.
            relative_t: [0x3ce63e7b, 0xb8d7afbc, 0xba563900],
            before_adj: [
                0x3ca45f54, 0xbb4c3a84, 0xbf33324e, 0x3f36c007, 0x3d8e8d8c, 0xbd339e72, 0xbb932378,
            ],
            after_adj: [
                0x3ca3fe80, 0xbf7fe08e, 0xbcc1ab36, 0x00000000, 0x00000000, 0x00000000, 0x3f7fd02a,
                0x3c9d8ea0, 0x3d0735b3, 0x00000000, 0x00000000, 0x00000000, 0xbd05484e, 0xbcc6f0e0,
                0x3f7fc9f5, 0x00000000, 0x00000000, 0x00000000, 0xbb623180, 0x3acbe7de, 0xbd8cafc7,
                0x3ca3fe80, 0xbf7fe08e, 0xbcc1ab36, 0xbab26aa0, 0xbbde5285, 0x3d38f8a5, 0x3f7fd02a,
                0x3c9d8ea0, 0x3d0735b3, 0xbd33eadf, 0xbd8e22d9, 0xbb4c4ba8, 0xbd05484e, 0xbcc6f0e0,
                0x3f7fc9f5,
            ],
            after_adj2: [
                0x3c73d880, 0xbf7ff8bc, 0x3a188a9d, 0x00000000, 0x00000000, 0x00000000, 0x3f7fea6c,
                0x3c73fdc0, 0x3cab33d7, 0x00000000, 0x00000000, 0x00000000, 0xbcab4127, 0x398de86d,
                0x3f7ff1ad, 0x00000000, 0x00000000, 0x00000000, 0xbb723ce4, 0xb8c7a1b5, 0xbd8d6046,
                0x3c73d880, 0xbf7ff8bc, 0x3a188a9d, 0xb98fc173, 0xbba83b39, 0x3c8969dc, 0x3f7fea6c,
                0x3c73fdc0, 0x3cab33d7, 0xbc80f7f4, 0xbd8daed3, 0xb9a2c4c3, 0xbcab4127, 0x398de86d,
                0x3f7ff1ad,
            ],
            returned: [
                0x3c48260a, 0xbbbfafc5, 0x3b23e6e5, 0x3f7ff9ca, 0x3960ac00, 0xbce9c4da, 0xbaa1d03e,
            ],
        },
        NativeRelPoseFixture {
            name: "ordinal9_frame4_cam1_obs9",
            state_h: [
                0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd, 0x00000000, 0x00000000, 0x00000000,
            ],
            state_t: [
                0xbd5f79e6, 0xbf4f45a2, 0xbbb53f68, 0x3f159719, 0x3c02c75a, 0xbb0c3bda, 0xbcdc2b7f,
            ],
            extrinsic_h: [
                0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc, 0x3ba86617,
            ],
            extrinsic_t: [
                0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6, 0xbc76fa76, 0x3d290319, 0x3b4f4832,
            ],
            after_inverse: [
                0x3b19188b, 0xbc55012e, 0xbf33d4ed, 0x3f362af6, 0xbd27e3b1, 0xbc8044de, 0xbb7a8e6e,
            ],
            relative_q: [0x3bc5abb3, 0x3c4782fd, 0x3b130617, 0x3f7ff9c9],
            relative_t: [0x3ce63e7b, 0xb8d7afbc, 0xba563900],
            before_adj: [
                0x3c784607, 0xbc0c871e, 0xbf3360ef, 0x3f369746, 0xbd26c55e, 0xbd334a15, 0xbb8a1a0a,
            ],
            after_adj: [
                0x3c929ea0, 0xbf7ff2db, 0xbc1377bf, 0x00000000, 0x00000000, 0x00000000, 0x3f7fd0c9,
                0x3c901020, 0x3d09c617, 0x00000000, 0x00000000, 0x00000000, 0xbd091909, 0xbc1d399c,
                0x3f7fd843, 0x00000000, 0x00000000, 0x00000000, 0xbb7a540c, 0xb9e7aee4, 0x3d29f249,
                0x3c929ea0, 0xbf7ff2db, 0xbc1377bf, 0xbab743d6, 0xbb3a4078, 0x3d303a38, 0x3f7fd0c9,
                0x3c901020, 0x3d09c617, 0xbd3358a9, 0x3d273f66, 0xba8cd213, 0xbd091909, 0xbc1d399c,
                0x3f7fd843,
            ],
            after_adj2: [
                0x3c50bc00, 0xbf7ff318, 0x3c795f6c, 0x00000000, 0x00000000, 0x00000000, 0x3f7feb21,
                0x3c561800, 0x3cb0dd46, 0x00000000, 0x00000000, 0x00000000, 0xbcb27575, 0x3c74c968,
                0x3f7fe922, 0x00000000, 0x00000000, 0x00000000, 0xbb851013, 0x3a16c64f, 0x3d28ac66,
                0x3c50bc00, 0xbf7ff318, 0x3c795f6c, 0xb9970b1d, 0xbb407b2d, 0x3c77ae51, 0x3f7feb21,
                0x3c561800, 0x3cb0dd46, 0xbc7f833d, 0x3d282c07, 0xba79f3d7, 0xbcb27575, 0x3c74c968,
                0x3f7fe922,
            ],
            returned: [
                0x3ba066a6, 0xbbce2fcd, 0x3ac222e9, 0x3f7ffdda, 0xbde17018, 0xbce785da, 0xbaa35d96,
            ],
        },
    ];

    let mut failures = Vec::new();
    let mut candidate_failures = Vec::new();
    for fixture in fixtures {
        let host = pose(fixture.state_h);
        let target = pose(fixture.state_t);
        let anchor_camera = pose(fixture.extrinsic_h);
        let target_camera = pose(fixture.extrinsic_t);

        let target_camera_from_imu = target_camera.inverse();
        let target_inverse_rotation = sophus_so3_inverse(target.rotation);
        let target_imu_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_current_packet_f32(
                target_inverse_rotation,
                host.rotation,
            ),
            translation: sophus_rotate_difference_visual_f32(
                target_inverse_rotation,
                host.translation,
                target.translation,
            ),
        };
        let target_camera_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.translation,
            ) + target_camera_from_imu.translation,
        };
        let returned = F32Pose {
            rotation: sophus_quat_product_camera_suffix_f32(
                target_camera_from_anchor_imu.rotation,
                anchor_camera.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_anchor_imu.rotation,
                anchor_camera.translation,
            ) + target_camera_from_anchor_imu.translation,
        };

        record_stage(
            fixture.name,
            "after_inverse",
            &pose_bits(target_camera_from_imu),
            &fixture.after_inverse,
            &mut failures,
        );
        record_stage(
            fixture.name,
            "after_relative_world_norm_q",
            &[
                target_imu_from_anchor_imu.rotation.i.to_bits(),
                target_imu_from_anchor_imu.rotation.j.to_bits(),
                target_imu_from_anchor_imu.rotation.k.to_bits(),
                target_imu_from_anchor_imu.rotation.w.to_bits(),
            ],
            &fixture.relative_q,
            &mut failures,
        );
        record_stage(
            fixture.name,
            "after_so3_action_xyz",
            &[
                target_imu_from_anchor_imu.translation.x.to_bits(),
                target_imu_from_anchor_imu.translation.y.to_bits(),
                target_imu_from_anchor_imu.translation.z.to_bits(),
            ],
            &fixture.relative_t,
            &mut failures,
        );
        record_stage(
            fixture.name,
            "before_adj_tmp",
            &pose_bits(target_camera_from_anchor_imu),
            &fixture.before_adj,
            &mut failures,
        );
        let adjoint = adjoint_bits(target_camera_from_anchor_imu);
        record_stage(
            fixture.name,
            "after_adj",
            &adjoint,
            &fixture.after_adj,
            &mut failures,
        );
        let adjoint2 = adjoint_bits(target_camera_from_imu);
        record_stage(
            fixture.name,
            "after_adj2",
            &adjoint2,
            &fixture.after_adj2,
            &mut failures,
        );
        record_stage(
            fixture.name,
            "returned",
            &pose_bits(returned),
            &fixture.returned,
            &mut failures,
        );

        // The clean native computeRelPose body uses its out-of-line
        // scalar-lane SO3 action after the packet relative quaternion.
        // Keep this route test-only until it has been checked against
        // both camera suffixes and the existing track fixtures.  The
        // current production value chain above intentionally remains the
        // diagnostic control.
        let candidate_relative_t = sophus_rotate_difference_f32(
            target_inverse_rotation,
            host.translation,
            target.translation,
        );
        let candidate_target_imu_from_anchor_imu = F32Pose {
            rotation: target_imu_from_anchor_imu.rotation,
            translation: candidate_relative_t,
        };
        let candidate_target_camera_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                candidate_target_imu_from_anchor_imu.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_imu.rotation,
                candidate_target_imu_from_anchor_imu.translation,
            ) + target_camera_from_imu.translation,
        };
        let candidate_returned = F32Pose {
            rotation: sophus_quat_product_camera_suffix_f32(
                candidate_target_camera_from_anchor_imu.rotation,
                anchor_camera.rotation,
            ),
            translation: sophus_rotate_f32(
                candidate_target_camera_from_anchor_imu.rotation,
                anchor_camera.translation,
            ) + candidate_target_camera_from_anchor_imu.translation,
        };
        record_stage(
            fixture.name,
            "candidate_after_so3_action_post_store",
            &[
                candidate_target_imu_from_anchor_imu.translation.x.to_bits(),
                candidate_target_imu_from_anchor_imu.translation.y.to_bits(),
                candidate_target_imu_from_anchor_imu.translation.z.to_bits(),
            ],
            &fixture.relative_t,
            &mut candidate_failures,
        );
        record_stage(
            fixture.name,
            "candidate_before_adj_tmp",
            &pose_bits(candidate_target_camera_from_anchor_imu),
            &fixture.before_adj,
            &mut candidate_failures,
        );
        record_stage(
            fixture.name,
            "candidate_after_adj",
            &adjoint_bits(candidate_target_camera_from_anchor_imu),
            &fixture.after_adj,
            &mut candidate_failures,
        );
        record_stage(
            fixture.name,
            "candidate_after_adj2",
            &adjoint2,
            &fixture.after_adj2,
            &mut candidate_failures,
        );
        record_stage(
            fixture.name,
            "candidate_returned",
            &pose_bits(candidate_returned),
            &fixture.returned,
            &mut candidate_failures,
        );
    }

    assert_eq!(
        failures.len(),
        8,
        "retry12 current-chain mismatch boundary changed:\n{}",
        failures.join("\n")
    );
    assert!(
        candidate_failures.is_empty(),
        "retry12 out-of-line candidate mismatches:\n{}",
        candidate_failures.join("\n")
    );
}

/// Replay the real ordered frame-4 visual row stack captured by the
/// current full-path detail oracle.  The non-visual rows are intentionally
/// not reconstructed here (the detail contract exposes their aggregate),
/// but the visual reducer/recovery boundary is the only part that differs
/// between the retained and clean LM paths.  This keeps the oracle tied to
/// a real 61-factor frame rather than a synthetic one-landmark fixture.
#[test]
fn m11_frame4_visual_stack_compact_matches_legacy_each_iteration() {
    let Some(path) = std::env::var_os("VISLOC_M11_FRAME4_FACTOR_FIXTURE") else {
        eprintln!(
            "skipping current frame-4 compact/legacy oracle: \
                 VISLOC_M11_FRAME4_FACTOR_FIXTURE is not set"
        );
        return;
    };
    let path = std::path::PathBuf::from(path);
    let fixture_text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let fixture: serde_json::Value = serde_json::from_str(&fixture_text)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
    assert_eq!(
        fixture["schema"].as_str(),
        Some("m11.frame4.visual-factor-stack.v2")
    );
    assert_eq!(fixture["frame_id"].as_u64(), Some(4));
    let state_dof = fixture["state_dof"].as_u64().expect("frame-4 state_dof") as usize;
    assert_eq!(state_dof, 75);

    let matrix = |value: &serde_json::Value, label: &str| -> DMatrix<f64> {
        let rows = value
            .as_array()
            .unwrap_or_else(|| panic!("{label} must be a row array"));
        let row_count = rows.len();
        let column_count = rows
            .first()
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        assert!(
            row_count != 0 && column_count != 0,
            "{label} must be non-empty"
        );
        let mut values = Vec::with_capacity(row_count * column_count);
        for (row, value) in rows.iter().enumerate() {
            let row_values = value
                .as_array()
                .unwrap_or_else(|| panic!("{label}[{row}] must be an array"));
            assert_eq!(row_values.len(), column_count, "{label} row {row} width");
            values.extend(row_values.iter().map(|value| {
                value
                    .as_f64()
                    .unwrap_or_else(|| panic!("{label}[{row}] contains non-number"))
            }));
        }
        DMatrix::from_row_slice(row_count, column_count, &values)
    };
    let vector = |value: &serde_json::Value, label: &str| -> DVector<f64> {
        let values = value
            .as_array()
            .unwrap_or_else(|| panic!("{label} must be an array"))
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .unwrap_or_else(|| panic!("{label} contains non-number"))
            })
            .collect::<Vec<_>>();
        DVector::from_vec(values)
    };

    let iterations = fixture["iterations"]
        .as_array()
        .expect("frame-4 iterations array");
    assert!(!iterations.is_empty());
    for iteration in iterations {
        let iteration_id = iteration["iteration"].as_u64().expect("iteration id") as usize;
        let factor_values = iteration["factors"].as_array().expect("iteration factors");
        assert_eq!(
            factor_values.len(),
            iteration["visual_factor_count"]
                .as_u64()
                .expect("visual factor count") as usize
        );
        assert_eq!(factor_values.len(), 61, "frame-4 visual factor count");
        let factors = factor_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let track_id = value["track_id"].as_u64().expect("track id");
                let state_jacobian = matrix(
                    &value["state_jacobian"],
                    &format!("iteration {iteration_id} factor {index} state Jacobian"),
                );
                let landmark_jacobian = matrix(
                    &value["landmark_jacobian"],
                    &format!("iteration {iteration_id} factor {index} landmark Jacobian"),
                );
                let residual = vector(
                    &value["residual"],
                    &format!("iteration {iteration_id} factor {index} residual"),
                );
                assert_eq!(state_jacobian.ncols(), state_dof);
                assert_eq!(landmark_jacobian.ncols(), 3);
                WhitenedFactorRowStack::with_objective_cost_kind(
                    state_jacobian,
                    landmark_jacobian,
                    residual,
                    0.0,
                    FactorKind::Visual,
                )
                .expect("valid frame-4 visual factor")
                .with_landmark_metadata(index, track_id)
            })
            .collect::<Vec<_>>();
        let state = DVector::zeros(state_dof);
        let state_step = vector(
            &iteration["trial_delta"],
            &format!("iteration {iteration_id} trial delta"),
        );
        assert_eq!(state_step.len(), state_dof);

        let legacy = reduce_landmark_factors_f32_checked(&factors, state_dof, 1e-10)
            .unwrap_or_else(|error| {
                panic!("legacy reduction at iteration {iteration_id}: {error:?}")
            });
        let compact = reduce_landmark_factors_f32_checked_with_compact_back_substitution(
            &factors, state_dof, 1e-10,
        )
        .unwrap_or_else(|error| panic!("compact reduction at iteration {iteration_id}: {error:?}"));
        assert_matrix_f32_bitwise_equal(
            &compact.h,
            &legacy.h,
            &format!("iteration {iteration_id} reduced H"),
        );
        assert_vector_f32_bitwise_equal(
            &compact.b,
            &legacy.b,
            &format!("iteration {iteration_id} reduced b"),
        );

        let compact_batch = compact
            .compact_back_substitution
            .as_ref()
            .expect("compact frame-4 payload");
        assert_eq!(compact_batch.entries.len(), factors.len());
        for (index, (factor, value)) in factors.iter().zip(factor_values).enumerate() {
            let entry = &compact_batch.entries[index];
            assert_eq!(entry.landmark_index, index);
            assert_eq!(entry.track_id, factor.landmark_metadata.unwrap().track_id);
            let compact_step = back_substitute_landmark_compact_entry_f32(
                entry,
                &compact_batch.storage,
                &state_step,
                1e-10,
            )
            .unwrap_or_else(|| {
                panic!("compact recovery at iteration {iteration_id} factor {index}")
            });
            let legacy_data = LandmarkBackSubstitution {
                state_jacobian: factor.state_jacobian.clone(),
                landmark_jacobian: factor.landmark_jacobian.clone(),
                residual: factor.residual.clone(),
                rank: value["rank"].as_u64().unwrap() as usize,
            };
            let legacy_step = back_substitute_landmark_f32_with_track(
                &legacy_data,
                &state_step,
                1e-10,
                Some(value["track_id"].as_u64().unwrap()),
            )
            .unwrap_or_else(|| {
                panic!("legacy recovery at iteration {iteration_id} factor {index}")
            });
            let public_legacy_step = back_substitute_landmark_upstream_f32_with_track(
                factor,
                &state_step,
                1e-10,
                Some(value["track_id"].as_u64().unwrap()),
            )
            .unwrap_or_else(|| {
                panic!("public legacy recovery at iteration {iteration_id} factor {index}")
            });
            assert_eq!(compact_step.len(), legacy_step.len());
            for (lane, (&compact, &legacy)) in
                compact_step.iter().zip(legacy_step.iter()).enumerate()
            {
                assert_eq!(
                    compact.to_bits(),
                    legacy.to_bits(),
                    "iteration {iteration_id} factor {index} landmark step lane {lane}"
                );
            }
            for (lane, (&public_legacy, &legacy)) in public_legacy_step
                .iter()
                .zip(legacy_step.iter())
                .enumerate()
            {
                assert_eq!(
                    public_legacy.to_bits(),
                    legacy.to_bits(),
                    "iteration {iteration_id} factor {index} public legacy step lane {lane}"
                );
            }
        }

        let preparation = compact
            .into_trial_preparation(&state, &state_step, 1e-10)
            .expect("compact trial preparation");
        let (_, state_fingerprint, step_fingerprint, prepared) = preparation.take_landmark_steps();
        assert_ne!(state_fingerprint, 0);
        assert_ne!(step_fingerprint, 0);
        assert_eq!(prepared.len(), factors.len());
        for (index, (_, track_id, step)) in prepared.into_iter().enumerate() {
            assert_eq!(track_id, factors[index].landmark_metadata.unwrap().track_id);
            assert!(step.is_some(), "prepared step at factor {index}");
        }
    }
}
