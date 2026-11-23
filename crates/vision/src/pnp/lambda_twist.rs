//! Lambda Twist P3P (Persson & Nordberg, ECCV 2018, "Lambda Twist: An
//! Accurate Fast Robust Perspective Three Point (P3P) Solver").
//!
//! Grunert's formulation ([`super::p3p`]) eliminates two of the three point
//! depths into a quartic in one depth *ratio*. The elimination divides by
//! quantities that vanish in ordinary geometry: the quartic's leading
//! coefficient `(p−1)² − 4c²/b²·cos²α` and the back-substitution denominator
//! `cos γ − v·cos α`. Both pass through zero for symmetric triangles seen
//! head-on — the usual view of a wall, where the camera faces the plane the
//! three points lie on — and nearby the roots lose most of their digits.
//!
//! Lambda Twist keeps the three depth constraints
//! `λᵢ² + λⱼ² − 2·cos θᵢⱼ·λᵢλⱼ = ‖Xᵢ − Xⱼ‖²` as quadratic forms. Two
//! scale-free combinations of them span a pencil of conics; the pencil holds
//! a degenerate conic (one real root of a cubic) which factors into two
//! planes through the origin, and on each plane the depths follow from a
//! quadratic. A few Gauss-Newton steps on the three original constraints then
//! polish each solution, which is where most of the method's accuracy comes
//! from. No step divides by a geometry-dependent quantity that vanishes for a
//! valid configuration other than a collinear triple.
//!
//! This follows the paper's construction but is written from the quadratic
//! forms directly (the cubic's coefficients come from the determinant of the
//! pencil) rather than from the paper's expanded closed-form coefficients.
//! Everything else is closed form, as in the paper: the cubic by Cardano, the
//! plane split from the known zero eigenvalue, and the pose from the two
//! congruent triangles without an SVD. About 0.9 us per solve, against about
//! 6 us for Grunert (whose SVD-based absolute orientation is most of its
//! cost).

use nalgebra::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector2, Vector3};
use visloc_core::geometry::Pose;

/// Solve P3P for the up-to-four world→camera poses that map the three world
/// `points` onto the three unit bearings `rays` (camera frame).
pub(crate) fn solve_lambda_twist(
    points: &[Point3<f64>; 3],
    rays: &[Vector3<f64>; 3],
) -> Option<Vec<Pose>> {
    // Both scale-free combinations below share the pair (2, 3); when that
    // side is short they are nearly proportional and the pencil carries no
    // information. Order the points so that it is the longest side (as the
    // paper does).
    let side = |i: usize, j: usize| (points[i] - points[j]).norm_squared();
    let order = if side(1, 2) >= side(0, 1) && side(1, 2) >= side(0, 2) {
        [0, 1, 2]
    } else if side(0, 2) >= side(0, 1) {
        [1, 0, 2]
    } else {
        [2, 0, 1]
    };
    let points = &order.map(|i| points[i]);
    let y = [
        unit(&rays[order[0]])?,
        unit(&rays[order[1]])?,
        unit(&rays[order[2]])?,
    ];
    let a12 = (points[0] - points[1]).norm_squared();
    let a13 = (points[0] - points[2]).norm_squared();
    let a23 = (points[1] - points[2]).norm_squared();
    let scale = a12.max(a13).max(a23);
    if !(scale.is_finite() && scale > f64::EPSILON) {
        return None;
    }
    // Work with unit-scale squared distances so the tolerances below are
    // geometry independent; depths scale back by √scale at the end.
    let a = [a12 / scale, a13 / scale, a23 / scale];
    if a.iter().any(|&v| v <= 1.0e-12) {
        return None;
    }
    let b12 = -2.0 * y[0].dot(&y[1]);
    let b13 = -2.0 * y[0].dot(&y[2]);
    let b23 = -2.0 * y[1].dot(&y[2]);

    // λᵀ·Mᵢⱼ·λ = aᵢⱼ for each pair.
    let m12 = Matrix3::new(1.0, 0.5 * b12, 0.0, 0.5 * b12, 1.0, 0.0, 0.0, 0.0, 0.0);
    let m13 = Matrix3::new(1.0, 0.0, 0.5 * b13, 0.0, 0.0, 0.0, 0.5 * b13, 0.0, 1.0);
    let m23 = Matrix3::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.5 * b23, 0.0, 0.5 * b23, 1.0);
    // Scale-free (homogeneous) combinations: every solution has λᵀ·Dₖ·λ = 0.
    let d1 = m12 * a[2] - m23 * a[0];
    let d2 = m13 * a[2] - m23 * a[1];

    let constraints = Constraints {
        a,
        b: [b12, b13, b23],
    };

    // det(D1 + γ·D2) is a cubic in γ. Its real roots make the conic
    // degenerate: a pair of planes through the origin. Every real degenerate
    // member of the pencil contains every real solution, so one suffices;
    // take the best-conditioned split and fall back to the others only if it
    // yields nothing.
    let mut pencils: Vec<(f64, Split)> = cubic_pencil_roots(&d1, &d2)
        .into_iter()
        .filter_map(|gamma| Some((gamma, Split::new(&(d1 + d2 * gamma))?)))
        .collect();
    pencils.sort_by(|l, r| r.1.conditioning.total_cmp(&l.1.conditioning));

    let mut depths: Vec<Vector3<f64>> = Vec::with_capacity(4);
    for (gamma, split) in &pencils {
        // On the planes D0 vanishes, so D1 = −γ·D2 there: impose whichever of
        // the two is not (nearly) the zero form on the plane.
        let homogeneous = if gamma.abs() > 1.0 { d1 } else { d2 };
        for normal in split.normals() {
            for lambda in depths_on_plane(&normal, &homogeneous, &m12, a[0]) {
                let Some(lambda) = constraints.refine(lambda) else {
                    continue;
                };
                if depths
                    .iter()
                    .all(|known| (known - lambda).norm() > 1.0e-9 * (1.0 + lambda.norm()))
                {
                    depths.push(lambda);
                }
            }
        }
        if !depths.is_empty() {
            break;
        }
    }

    let unscale = scale.sqrt();
    let poses: Vec<Pose> = depths
        .into_iter()
        .filter_map(|lambda| {
            let camera_points = [
                Point3::from(y[0] * (lambda.x * unscale)),
                Point3::from(y[1] * (lambda.y * unscale)),
                Point3::from(y[2] * (lambda.z * unscale)),
            ];
            pose_from_three(points, &camera_points)
        })
        .collect();
    if poses.is_empty() {
        None
    } else {
        Some(poses)
    }
}

fn unit(v: &Vector3<f64>) -> Option<Vector3<f64>> {
    let norm = v.norm();
    (norm.is_finite() && norm > f64::EPSILON).then(|| v / norm)
}

/// The three depth constraints `λᵢ² + λⱼ² + bᵢⱼ·λᵢλⱼ = aᵢⱼ` (pairs 12, 13, 23).
struct Constraints {
    a: [f64; 3],
    b: [f64; 3],
}

impl Constraints {
    fn residual(&self, l: &Vector3<f64>) -> Vector3<f64> {
        let [b12, b13, b23] = self.b;
        Vector3::new(
            l.x * l.x + l.y * l.y + b12 * l.x * l.y - self.a[0],
            l.x * l.x + l.z * l.z + b13 * l.x * l.z - self.a[1],
            l.y * l.y + l.z * l.z + b23 * l.y * l.z - self.a[2],
        )
    }

    /// Gauss-Newton polish of a root; `None` unless it ends as a valid
    /// solution (positive depths that satisfy all three constraints).
    fn refine(&self, mut l: Vector3<f64>) -> Option<Vector3<f64>> {
        let [b12, b13, b23] = self.b;
        let mut r = self.residual(&l);
        for _ in 0..5 {
            let error = r.norm_squared();
            if error < 1.0e-30 {
                break;
            }
            let jacobian = Matrix3::new(
                2.0 * l.x + b12 * l.y,
                2.0 * l.y + b12 * l.x,
                0.0,
                2.0 * l.x + b13 * l.z,
                0.0,
                2.0 * l.z + b13 * l.x,
                0.0,
                2.0 * l.y + b23 * l.z,
                2.0 * l.z + b23 * l.y,
            );
            let Some(step) = jacobian.lu().solve(&r) else {
                break;
            };
            // Halve the step until it helps: near a double root (two
            // solutions merging) the Jacobian is close to singular and the
            // full step overshoots.
            let mut improved = None;
            let mut scale = 1.0;
            for _ in 0..8 {
                let next = l - step * scale;
                let next_r = self.residual(&next);
                if next_r.norm_squared() < error {
                    improved = Some((next, next_r));
                    break;
                }
                scale *= 0.5;
            }
            let Some((next, next_r)) = improved else {
                break;
            };
            l = next;
            r = next_r;
        }
        // The constraints are in units of the longest squared side (1), so
        // this only drops roots that went nowhere; among the rest the caller's
        // reprojection check picks the pose.
        let valid = l.iter().all(|v| v.is_finite() && *v > 0.0) && r.amax() < 1.0e-4;
        valid.then_some(l)
    }
}

/// Real roots of `det(D1 + γ·D2) = 0`.
fn cubic_pencil_roots(d1: &Matrix3<f64>, d2: &Matrix3<f64>) -> Vec<f64> {
    // det(A + γB) = c0 + c1·γ + c2·γ² + c3·γ³; c0 and c3 exactly, c1 and c2
    // from the values at γ = ±1.
    let c0 = d1.determinant();
    let c3 = d2.determinant();
    let f_plus = (d1 + d2).determinant();
    let f_minus = (d1 - d2).determinant();
    let c2 = 0.5 * (f_plus + f_minus) - c0;
    let c1 = 0.5 * (f_plus - f_minus) - c3;
    let mut roots = real_polynomial_roots(&[c0, c1, c2, c3]);
    // Newton polish against the cubic itself.
    for root in &mut roots {
        for _ in 0..3 {
            let g = *root;
            let f = ((c3 * g + c2) * g + c1) * g + c0;
            let df = (3.0 * c3 * g + 2.0 * c2) * g + c1;
            if df.abs() <= f64::EPSILON * (1.0 + f.abs()) {
                break;
            }
            let next = g - f / df;
            if !next.is_finite() {
                break;
            }
            *root = next;
        }
    }
    roots
}

/// Real roots of `c[0] + c[1]x + … + c[n]x^n` for n ≤ 3, dropping leading
/// coefficients that are negligible against the rest.
fn real_polynomial_roots(c: &[f64; 4]) -> Vec<f64> {
    let largest = c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if !(largest.is_finite() && largest > 0.0) {
        return Vec::new();
    }
    let c: Vec<f64> = c.iter().map(|v| v / largest).collect();
    let tiny = 1.0e-12;
    if c[3].abs() > tiny {
        return monic_cubic_roots(c[2] / c[3], c[1] / c[3], c[0] / c[3]);
    }
    if c[2].abs() > tiny {
        let disc = c[1] * c[1] - 4.0 * c[2] * c[0];
        if disc < 0.0 {
            return Vec::new();
        }
        let q = -0.5 * (c[1] + c[1].signum() * disc.sqrt());
        let mut roots = vec![q / c[2]];
        if q.abs() > f64::EPSILON {
            roots.push(c[0] / q);
        }
        return roots;
    }
    if c[1].abs() > tiny {
        return vec![-c[0] / c[1]];
    }
    Vec::new()
}

/// Real roots of `x³ + p·x² + q·x + r` in closed form (trigonometric for
/// three real roots, Cardano for one); the caller polishes them with Newton.
fn monic_cubic_roots(p: f64, q: f64, r: f64) -> Vec<f64> {
    if ![p, q, r].iter().all(|v| v.is_finite()) {
        return Vec::new();
    }
    // Depressed cubic t³ + a·t + b with x = t − p/3.
    let shift = p / 3.0;
    let a = q - p * shift;
    let b = (2.0 * p * p * p / 27.0) - (p * q / 3.0) + r;
    let disc = (b * b / 4.0) + (a * a * a / 27.0);
    if disc < 0.0 {
        // Three real roots (a < 0 here).
        let m = 2.0 * (-a / 3.0).sqrt();
        let theta = ((3.0 * b) / (a * m)).clamp(-1.0, 1.0).acos() / 3.0;
        (0..3)
            .map(|k| m * (theta - std::f64::consts::TAU * f64::from(k) / 3.0).cos() - shift)
            .collect()
    } else {
        let sq = disc.sqrt();
        let u = (-b / 2.0 + sq).cbrt();
        let v = (-b / 2.0 - sq).cbrt();
        vec![u + v - shift]
    }
}

/// The split of a degenerate, indefinite conic `D0` (one zero eigenvalue)
/// into two planes: `e₊(u₊·λ)² + e₋(u₋·λ)² = 0` ⇔ `(√e₊·u₊ ± √−e₋·u₋)·λ = 0`.
///
/// With the zero eigenvalue known, the other two are the roots of
/// `e² − tr(D0)·e + m = 0` (`m` the sum of the principal 2×2 minors) and the
/// eigenvectors come from cross products, as in the paper, instead of an
/// iterative eigensolver.
struct Split {
    e_pos: f64,
    u_pos: Vector3<f64>,
    e_neg: f64,
    u_neg: Vector3<f64>,
    /// The smaller non-zero eigenvalue magnitude over the larger.
    conditioning: f64,
}

impl Split {
    fn new(d0: &Matrix3<f64>) -> Option<Self> {
        if !d0.iter().all(|v| v.is_finite()) {
            return None;
        }
        let trace = d0.trace();
        let minors = d0[(0, 0)] * d0[(1, 1)] - d0[(0, 1)] * d0[(1, 0)] + d0[(0, 0)] * d0[(2, 2)]
            - d0[(0, 2)] * d0[(2, 0)]
            + d0[(1, 1)] * d0[(2, 2)]
            - d0[(1, 2)] * d0[(2, 1)];
        let disc = trace * trace - 4.0 * minors;
        if disc <= 0.0 {
            return None;
        }
        // Numerically stable pair of roots of e² − trace·e + minors.
        let q = 0.5 * (trace + trace.signum() * disc.sqrt());
        let (r1, r2) = if q == 0.0 {
            (0.5 * disc.sqrt(), -0.5 * disc.sqrt())
        } else {
            (q, minors / q)
        };
        let (e_pos, e_neg) = (r1.max(r2), r1.min(r2));
        if e_pos <= 0.0 || e_neg >= 0.0 {
            return None;
        }
        let largest = e_pos.max(-e_neg);
        // The null vector, and how null it is: a conic that is not
        // degenerate enough is no plane pair.
        let null = largest_cross(d0)?;
        if (d0 * null).norm() > 1.0e-3 * largest {
            return None;
        }
        // Eigenvector of the larger-magnitude eigenvalue from D0 − e·I (best
        // separated from the rest), the other one orthogonal to both.
        let e_big = if e_pos >= -e_neg { e_pos } else { e_neg };
        let u_big = largest_cross(&(d0 - Matrix3::identity() * e_big))?;
        let u_small = unit(&null.cross(&u_big))?;
        let (u_pos, u_neg) = if e_big > 0.0 {
            (u_big, u_small)
        } else {
            (u_small, u_big)
        };
        Some(Self {
            e_pos,
            u_pos,
            e_neg,
            u_neg,
            conditioning: e_pos.min(-e_neg) / largest,
        })
    }

    fn normals(&self) -> [Vector3<f64>; 2] {
        let (p, q) = (
            self.u_pos * self.e_pos.sqrt(),
            self.u_neg * (-self.e_neg).sqrt(),
        );
        [p + q, p - q]
    }
}

/// Unit null direction of a rank-2 symmetric matrix: the longest cross
/// product of two of its rows.
fn largest_cross(m: &Matrix3<f64>) -> Option<Vector3<f64>> {
    let rows = [
        m.row(0).transpose(),
        m.row(1).transpose(),
        m.row(2).transpose(),
    ];
    let candidates = [
        rows[0].cross(&rows[1]),
        rows[0].cross(&rows[2]),
        rows[1].cross(&rows[2]),
    ];
    let best = candidates
        .iter()
        .max_by(|l, r| l.norm_squared().total_cmp(&r.norm_squared()))?;
    unit(best)
}

/// World→camera pose mapping three world points onto their camera-frame
/// reconstructions. With exact depths the two triangles are congruent, so the
/// rotation follows from two edges and their normal (`R = Y·X⁻¹`, as in the
/// paper) without an SVD; it is then re-orthonormalized against rounding.
fn pose_from_three(world: &[Point3<f64>; 3], camera: &[Point3<f64>; 3]) -> Option<Pose> {
    let frame = |p: &[Point3<f64>; 3]| -> Option<Matrix3<f64>> {
        let e1 = unit(&(p[1] - p[0]))?;
        let n = unit(&e1.cross(&(p[2] - p[0])))?;
        Some(Matrix3::from_columns(&[e1, n, e1.cross(&n)]))
    };
    // Orthonormal frames of both triangles: R maps one onto the other.
    let x = frame(world)?;
    let y = frame(camera)?;
    let rotation = Rotation3::from_matrix_unchecked(y * x.transpose());
    let translation = camera[0].coords - rotation * world[0].coords;
    if !translation.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(Pose::from_world_to_camera(
        UnitQuaternion::from_rotation_matrix(&rotation),
        translation,
    ))
}

/// Depths on the plane `normal·λ = 0` satisfying the homogeneous constraint
/// `λᵀ·H·λ = 0`, scaled so that `λᵀ·M·λ = a`.
fn depths_on_plane(
    normal: &Vector3<f64>,
    homogeneous: &Matrix3<f64>,
    m: &Matrix3<f64>,
    a: f64,
) -> Vec<Vector3<f64>> {
    let Some(n) = unit(normal) else {
        return Vec::new();
    };
    // Orthonormal basis (p, q) of the plane.
    let helper = if n.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let p = n.cross(&helper).normalize();
    let q = n.cross(&p);
    // H restricted to the plane: c00·s² + 2·c01·s·t + c11·t² = 0.
    let c00 = p.dot(&(homogeneous * p));
    let c01 = p.dot(&(homogeneous * q));
    let c11 = q.dot(&(homogeneous * q));
    let disc = c01 * c01 - c00 * c11;
    // A slightly negative discriminant is a double root (two solutions
    // merging) lost to rounding; keep it; the polish and the caller's
    // reprojection check reject it if it is not a solution.
    if disc < -1.0e-6 * (c00.abs() + c01.abs() + c11.abs()).powi(2) {
        return Vec::new();
    }
    let root = disc.max(0.0).sqrt();
    let directions: Vec<Vector2<f64>> = if c00.abs() >= c11.abs() {
        if c00.abs() <= f64::EPSILON {
            return Vec::new();
        }
        [-c01 + root, -c01 - root]
            .into_iter()
            .map(|s| Vector2::new(s / c00, 1.0))
            .collect()
    } else {
        [-c01 + root, -c01 - root]
            .into_iter()
            .map(|t| Vector2::new(1.0, t / c11))
            .collect()
    };
    directions
        .into_iter()
        .filter_map(|st| {
            let d = p * st.x + q * st.y;
            let along = d.dot(&(m * d));
            if along.is_nan() || along <= f64::EPSILON {
                return None;
            }
            let mut lambda = d * (a / along).sqrt();
            if lambda.sum() < 0.0 {
                lambda = -lambda;
            }
            // Allow a little slack before the Gauss-Newton polish.
            (lambda.min() > -1.0e-6).then_some(lambda)
        })
        .collect()
}
