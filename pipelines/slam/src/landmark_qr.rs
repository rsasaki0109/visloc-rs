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
