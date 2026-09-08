//! Bounded landmark QR kernel, staged before native linearization integration.
//! Input is the weighted three-column landmark Jacobian, including LM rows.
//! No camera-column matrix, full Q, projector, or pose-pair graph is stored.
//! Inspired by nullspace elimination in Demmel et al., CVPR 2021; this is
//! an independent compact-Householder implementation, not copied RootBA code.

#[derive(Debug)]
struct LandmarkQr {
    reflectors: [Vec<f64>; 3],
    upper: [[f64; 3]; 3],
    rows: usize,
}

impl LandmarkQr {
    fn factor(mut a: Vec<[f64; 3]>) -> Result<Self, &'static str> {
        let rows = a.len();
        if rows < 3 || !a.iter().flatten().all(|x| x.is_finite()) {
            return Err("invalid landmark Jacobian");
        }
        let mut reflectors: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        for k in 0..3 {
            let norm = a[k..].iter().fold(0.0_f64, |n, row| n.hypot(row[k]));
            if norm == 0.0 || !norm.is_finite() {
                return Err("rank deficient or nonfinite landmark column");
            }
            // Normalize first, avoiding overflow in x[0] +/- norm.
            let mut v: Vec<_> = a[k..].iter().map(|row| row[k] / norm).collect();
            v[0] += if v[0] >= 0.0 { 1.0 } else { -1.0 };
            let vnorm = v.iter().fold(0.0_f64, |n, x| n.hypot(*x));
            for x in &mut v {
                *x /= vnorm;
            }
            for col in k..3 {
                let dot: f64 = v.iter().zip(&a[k..]).map(|(x, row)| x * row[col]).sum();
                for (row, x) in a[k..].iter_mut().zip(&v) {
                    row[col] -= 2.0 * dot * x;
                }
            }
            reflectors[k] = v;
        }
        if !a.iter().flatten().all(|x| x.is_finite()) {
            return Err("nonfinite QR factor");
        }
        let upper =
            std::array::from_fn(|r| std::array::from_fn(|c| if c >= r { a[r][c] } else { 0.0 }));
        Ok(Self {
            reflectors,
            upper,
            rows,
        })
    }

    fn transform(&self, x: &mut [f64], transpose: bool) -> Result<(), &'static str> {
        if x.len() != self.rows || !x.iter().all(|v| v.is_finite()) {
            return Err("invalid QR vector");
        }
        // Q^T = H2 H1 H0: H0 is applied first. Q uses reverse order.
        for step in 0..3 {
            let k = if transpose { step } else { 2 - step };
            let v = &self.reflectors[k];
            let dot: f64 = v.iter().zip(&x[k..]).map(|(a, b)| a * b).sum();
            for (value, reflector) in x[k..].iter_mut().zip(v) {
                *value -= 2.0 * dot * reflector;
            }
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err("nonfinite QR action");
        }
        Ok(())
    }

    fn solve_landmark(&self, transformed_rhs: &[f64]) -> Result<[f64; 3], &'static str> {
        if transformed_rhs.len() != self.rows || !transformed_rhs.iter().all(|v| v.is_finite()) {
            return Err("invalid back substitution vector");
        }
        let mut x = [0.0; 3];
        for i in (0..3).rev() {
            if self.upper[i][i] == 0.0 {
                return Err("singular landmark factor");
            }
            let sum: f64 = ((i + 1)..3).map(|j| self.upper[i][j] * x[j]).sum();
            x[i] = (transformed_rhs[i] - sum) / self.upper[i][i];
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err("nonfinite landmark step");
        }
        Ok(x)
    }
}

#[derive(Clone, Debug)]
struct WeightedRow {
    pose: Option<usize>,
    pose_jacobian: [f64; 6],
    landmark_jacobian: [f64; 3],
    residual: f64,
}

#[derive(Debug)]
struct PoseRow {
    pose: Option<usize>,
    jacobian: [f64; 6],
    residual: f64,
}

/// One landmark's reduced operator. Inputs are already robust-weighted;
/// repeated sensors and fixed-pose rows remain distinct and in input order.
#[derive(Debug)]
struct ReducedLandmark {
    rows: Vec<PoseRow>,
    qr: Option<LandmarkQr>,
    pose_dimension: usize,
}

impl ReducedLandmark {
    fn new(
        rows: Vec<WeightedRow>,
        poses: usize,
        variable: bool,
        lambda: f64,
    ) -> Result<Self, &'static str> {
        let pose_dimension = poses.checked_mul(6).ok_or("pose dimension overflow")?;
        if rows.is_empty() || !lambda.is_finite() || lambda <= 0.0 {
            return Err("positive damping and observation rows required");
        }
        for row in &rows {
            if row.pose.is_some_and(|p| p >= poses)
                || !row
                    .pose_jacobian
                    .iter()
                    .chain(&row.landmark_jacobian)
                    .chain(std::iter::once(&row.residual))
                    .all(|v| v.is_finite())
            {
                return Err("invalid weighted observation row");
            }
        }
        let qr = if variable {
            let mut jl: Vec<_> = rows.iter().map(|r| r.landmark_jacobian).collect();
            for i in 0..3 {
                let mut row = [0.0; 3];
                row[i] = lambda.sqrt();
                jl.push(row);
            }
            Some(LandmarkQr::factor(jl)?)
        } else {
            None
        };
        Ok(Self {
            rows: rows
                .into_iter()
                .map(|r| PoseRow {
                    pose: r.pose,
                    jacobian: r.pose_jacobian,
                    residual: r.residual,
                })
                .collect(),
            qr,
            pose_dimension,
        })
    }

    fn prepare(&self, scratch: &mut Vec<f64>) {
        scratch.resize(self.rows.len() + if self.qr.is_some() { 3 } else { 0 }, 0.0);
        scratch.fill(0.0);
    }

    fn pose_action(&self, x: &[f64], scratch: &mut Vec<f64>) -> Result<(), &'static str> {
        if x.len() != self.pose_dimension || !x.iter().all(|v| v.is_finite()) {
            return Err("invalid pose vector");
        }
        self.prepare(scratch);
        for (value, row) in scratch.iter_mut().zip(&self.rows) {
            if let Some(p) = row.pose {
                *value = row
                    .jacobian
                    .iter()
                    .zip(&x[p * 6..p * 6 + 6])
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
        if !scratch.iter().all(|v| v.is_finite()) {
            return Err("nonfinite pose action");
        }
        Ok(())
    }

    fn apply<'a>(&self, x: &[f64], scratch: &'a mut Vec<f64>) -> Result<&'a [f64], &'static str> {
        self.pose_action(x, scratch)?;
        if let Some(qr) = &self.qr {
            qr.transform(scratch, true)?;
            Ok(&scratch[3..])
        } else {
            Ok(scratch)
        }
    }

    fn scatter(&self, values: &[f64], out: &mut [f64], sign: f64) -> Result<(), &'static str> {
        if out.len() != self.pose_dimension || !out.iter().all(|v| v.is_finite()) {
            return Err("invalid adjoint output");
        }
        for (value, row) in values.iter().zip(&self.rows) {
            if let Some(p) = row.pose {
                for (dst, j) in out[p * 6..p * 6 + 6].iter_mut().zip(row.jacobian) {
                    *dst += sign * value * j;
                }
            }
        }
        if !out.iter().all(|v| v.is_finite()) {
            return Err("nonfinite adjoint");
        }
        Ok(())
    }

    fn adjoint_add(
        &self,
        y: &[f64],
        out: &mut [f64],
        scratch: &mut Vec<f64>,
    ) -> Result<(), &'static str> {
        if y.len() != self.rows.len() || !y.iter().all(|v| v.is_finite()) {
            return Err("invalid reduced vector");
        }
        self.prepare(scratch);
        if let Some(qr) = &self.qr {
            scratch[3..].copy_from_slice(y);
            qr.transform(scratch, false)?;
        } else {
            scratch.copy_from_slice(y);
        }
        self.scatter(scratch, out, 1.0)
    }

    /// Add A^T A x using one reusable longest-track buffer. Global pose LM
    /// damping is added once by the caller, not once per landmark.
    fn normal_add(
        &self,
        x: &[f64],
        out: &mut [f64],
        scratch: &mut Vec<f64>,
    ) -> Result<(), &'static str> {
        self.pose_action(x, scratch)?;
        if let Some(qr) = &self.qr {
            qr.transform(scratch, true)?;
            scratch[..3].fill(0.0);
            qr.transform(scratch, false)?;
        }
        self.scatter(scratch, out, 1.0)
    }

    fn rhs_add(&self, out: &mut [f64], scratch: &mut Vec<f64>) -> Result<(), &'static str> {
        self.prepare(scratch);
        for (value, row) in scratch.iter_mut().zip(&self.rows) {
            *value = row.residual;
        }
        if let Some(qr) = &self.qr {
            qr.transform(scratch, true)?;
            scratch[..3].fill(0.0);
            qr.transform(scratch, false)?;
        }
        self.scatter(scratch, out, -1.0)
    }

    fn back_substitute(
        &self,
        x: &[f64],
        scratch: &mut Vec<f64>,
    ) -> Result<Option<[f64; 3]>, &'static str> {
        self.pose_action(x, scratch)?;
        let Some(qr) = &self.qr else {
            return Ok(None);
        };
        for (value, row) in scratch.iter_mut().zip(&self.rows) {
            *value = -(*value + row.residual);
        }
        qr.transform(scratch, true)?;
        qr.solve_landmark(scratch).map(Some)
    }
}

#[test]
fn landmark_qr_eliminates_columns_and_preserves_adjoint() {
    let a = vec![
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0],
        [2.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ];
    let qr = LandmarkQr::factor(a.clone()).unwrap();
    for col in 0..3 {
        let mut column: Vec<_> = a.iter().map(|r| r[col]).collect();
        qr.transform(&mut column, true).unwrap();
        assert!(column[3..].iter().all(|v| v.abs() < 1e-13));
        qr.transform(&mut column, false).unwrap();
        for (value, row) in column.iter().zip(&a) {
            assert!((value - row[col]).abs() < 1e-13);
        }
    }
    let x = [0.2, -0.3, 0.4];
    let mut rhs: Vec<_> = a
        .iter()
        .map(|r| (0..3).map(|i| r[i] * x[i]).sum())
        .collect();
    qr.transform(&mut rhs, true).unwrap();
    let recovered = qr.solve_landmark(&rhs).unwrap();
    for i in 0..3 {
        assert!((recovered[i] - x[i]).abs() < 1e-13);
    }
    let u = vec![0.0, 0.0, 0.0, 0.7, -0.2, 0.3];
    let v = vec![0.1, 0.2, -0.1, 0.5, 0.9, 0.4];
    let mut qu = u.clone();
    qr.transform(&mut qu, false).unwrap();
    let mut qtv = v.clone();
    qr.transform(&mut qtv, true).unwrap();
    let left: f64 = qu.iter().zip(&v).map(|(a, b)| a * b).sum();
    let right: f64 = u.iter().zip(&qtv).map(|(a, b)| a * b).sum();
    assert!((left - right).abs() < 1e-13);
}

#[test]
fn landmark_qr_projected_cost_matches_schur_for_damped_fixture() {
    use nalgebra::{DMatrix, DVector};
    for lambda in [1e-4_f64, 0.5, 100.0] {
        let mut rows = vec![
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 3.0],
            [2.0, 0.0, 1.0],
            [1.0, -1.0, 0.5],
        ];
        for i in 0..3 {
            let mut r = [0.0; 3];
            r[i] = lambda.sqrt();
            rows.push(r);
        }
        let qr = LandmarkQr::factor(rows.clone()).unwrap();
        let jl = DMatrix::from_fn(7, 3, |r, c| rows[r][c]);
        let residual = DVector::from_vec(vec![0.2, -0.1, 0.5, 0.3, 0.0, 0.0, 0.0]);
        let cross = jl.transpose() * &residual;
        let step = (jl.transpose() * &jl).cholesky().unwrap().solve(&cross);
        let expected = (&residual - &jl * step).norm_squared();
        let mut transformed = residual.as_slice().to_vec();
        qr.transform(&mut transformed, true).unwrap();
        let actual: f64 = transformed[3..].iter().map(|v| v * v).sum();
        assert!((actual - expected).abs() < 1e-13);
    }
}

#[test]
fn landmark_qr_long_track_storage_is_linear_and_invalid_inputs_fail() {
    let n = 20003;
    let mut a = vec![[0.1, 0.2, 0.3]; n];
    for i in 0..3 {
        a[n - 3 + i] = [0.0; 3];
        a[n - 3 + i][i] = 1.0;
    }
    let qr = LandmarkQr::factor(a).unwrap();
    assert_eq!(qr.reflectors.iter().map(Vec::len).sum::<usize>(), 3 * n - 3);
    assert!(LandmarkQr::factor(vec![[0.0; 3]; 4]).is_err());
    assert!(LandmarkQr::factor(vec![[f64::NAN; 3]; 4]).is_err());
    assert!(qr.transform(&mut [0.0; 2], true).is_err());
}

#[test]
fn sparse_qr_action_rhs_and_full_step_match_damped_normal_system() {
    use nalgebra::{DMatrix, DVector};
    let rows: Vec<_> = (0..12)
        .map(|i| {
            let weight = if i % 3 == 0 { 0.3 } else { 1.0 };
            WeightedRow {
                pose: if i % 4 == 0 { None } else { Some(i % 2) },
                pose_jacobian: std::array::from_fn(|j| weight * ((i * 7 + j * 3 + 1) as f64).sin()),
                landmark_jacobian: std::array::from_fn(|j| weight * ((i * 3 + j + 2) as f64).cos()),
                residual: weight * (i as f64 * 0.2 - 0.5),
            }
        })
        .collect();
    for variable in [false, true] {
        for lambda in [1e-4_f64, 0.5, 100.0] {
            let block = ReducedLandmark::new(rows.clone(), 2, variable, lambda).unwrap();
            let jp = DMatrix::from_fn(12, 12, |i, j| {
                if rows[i].pose == Some(j / 6) {
                    rows[i].pose_jacobian[j % 6]
                } else {
                    0.0
                }
            });
            let jl = DMatrix::from_fn(12, 3, |i, j| rows[i].landmark_jacobian[j]);
            let r = DVector::from_iterator(12, rows.iter().map(|r| r.residual));
            let hpp = jp.transpose() * &jp + DMatrix::identity(12, 12) * lambda;
            let hll = jl.transpose() * &jl + DMatrix::identity(3, 3) * lambda;
            let cross = jp.transpose() * &jl;
            let inv = hll.clone().cholesky().unwrap().inverse();
            let expected_s = if variable {
                &hpp - &cross * &inv * cross.transpose()
            } else {
                hpp.clone()
            };
            let expected_b = if variable {
                -jp.transpose() * &r + &cross * &inv * jl.transpose() * &r
            } else {
                -jp.transpose() * &r
            };
            let mut scratch = Vec::new();
            let mut a = DMatrix::zeros(12, 12);
            for col in 0..12 {
                let mut x = vec![0.0; 12];
                x[col] = 1.0;
                a.column_mut(col).copy_from(&DVector::from_column_slice(
                    block.apply(&x, &mut scratch).unwrap(),
                ));
            }
            let actual_s = a.transpose() * &a + DMatrix::identity(12, 12) * lambda;
            assert!((&actual_s - &expected_s).norm() < 1e-11);
            let probe = DVector::from_fn(12, |i, _| (i as f64 * 0.3).sin());
            let mut normal = vec![0.0; 12];
            block
                .normal_add(probe.as_slice(), &mut normal, &mut scratch)
                .unwrap();
            assert!(
                (DVector::from_vec(normal) + &probe * lambda - &expected_s * probe).norm() < 1e-11
            );
            let y = DVector::from_fn(12, |i, _| i as f64 / 13.0);
            let mut adjoint = vec![0.0; 12];
            block
                .adjoint_add(y.as_slice(), &mut adjoint, &mut scratch)
                .unwrap();
            assert!((DVector::from_vec(adjoint) - a.transpose() * y).norm() < 1e-12);
            let mut rhs = vec![0.0; 12];
            block.rhs_add(&mut rhs, &mut scratch).unwrap();
            assert!((DVector::from_column_slice(&rhs) - &expected_b).norm() < 1e-12);
            let dx = actual_s.cholesky().unwrap().solve(&DVector::from_vec(rhs));
            let point = block.back_substitute(dx.as_slice(), &mut scratch).unwrap();
            if variable {
                let mut full = DMatrix::zeros(15, 15);
                full.view_mut((0, 0), (12, 12)).copy_from(&hpp);
                full.view_mut((0, 12), (12, 3)).copy_from(&cross);
                full.view_mut((12, 0), (3, 12))
                    .copy_from(&cross.transpose());
                full.view_mut((12, 12), (3, 3)).copy_from(&hll);
                let mut b = DVector::zeros(15);
                b.rows_mut(0, 12).copy_from(&(-jp.transpose() * &r));
                b.rows_mut(12, 3).copy_from(&(-jl.transpose() * &r));
                let direct = full.cholesky().unwrap().solve(&b);
                assert!((&dx - direct.rows(0, 12)).norm() < 1e-8);
                assert!(
                    (DVector::from_column_slice(&point.unwrap()) - direct.rows(12, 3)).norm()
                        < 1e-8
                );
            } else {
                assert!(point.is_none());
            }
        }
    }
}

#[test]
fn sparse_qr_long_track_has_no_pose_pair_storage_and_validates_inputs() {
    let row = WeightedRow {
        pose: Some(0),
        pose_jacobian: [0.2; 6],
        landmark_jacobian: [0.3; 3],
        residual: 0.1,
    };
    let n = 10003;
    let block = ReducedLandmark::new(vec![row.clone(); n], 10000, true, 0.5).unwrap();
    assert_eq!(block.rows.len(), n);
    assert_eq!(
        block
            .qr
            .as_ref()
            .unwrap()
            .reflectors
            .iter()
            .map(Vec::len)
            .sum::<usize>(),
        3 * (n + 3) - 3
    );
    let mut scratch = Vec::new();
    assert_eq!(
        block.apply(&vec![0.0; 60000], &mut scratch).unwrap().len(),
        n
    );
    assert_eq!(scratch.len(), n + 3);
    assert!(block.apply(&[0.0; 6], &mut scratch).is_err());
    assert!(ReducedLandmark::new(vec![row.clone()], 0, true, 0.5).is_err());
    assert!(ReducedLandmark::new(vec![row], 1, true, 0.0).is_err());
}
