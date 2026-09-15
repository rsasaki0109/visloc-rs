//! Faithful port of COLMAP's GP3P minimal solver: `GP3PEstimator::Estimate`
//! (`estimators/solvers/generalized_absolute_pose.{h,cc}`), which itself
//! delegates entirely to PoseLib's `gp3p` solver
//! (`PoseLib/solvers/gp3p.{h,cc}`, "Re-implementation of the gP3P solver from
//! Kukelova et al., Efficient Intersection of Three Quadrics and Applications
//! in Computer Vision, CVPR 2016"), which in turn is built on PoseLib's
//! `re3q3`/`re3q3_rotation` "3 quadrics in 3 unknowns" solver
//! (`PoseLib/misc/re3q3.{h,cc}`). COLMAP vendors PoseLib as an external
//! dependency rather than reimplementing it, so there is no COLMAP-pinned
//! PoseLib commit to match; this module was ported directly from PoseLib's
//! `master`-branch sources (fetched during this port), matching the published
//! algorithm rather than a specific pinned revision.
//!
//! ## Pipeline (ported 1:1 unless noted below)
//! 1. [`gp3p_solve`] builds the 6x13 linear system `A * [t; vec(R); 1] = 0`
//!    from the 3 correspondences' rig-frame ray origins/bearings and world
//!    points (`gp3p.cc:39-49`), eliminates `t` using 3 of its 6 rows
//!    (`gp3p.cc:51`), leaving a 3x10 system linear in `vec(R)` (`AR`,
//!    `gp3p.cc:53`).
//! 2. [`re3q3_rotation`] draws an auxiliary unit-quaternion pre-rotation `q0`
//!    (avoids a Cayley-parameterization singularity at 180 degrees; see
//!    deviation below), substitutes `R = R0*R'`, converts the resulting
//!    linear-in-`R'` system to a quadratic-in-Cayley-parameter system via
//!    [`rotation_to_3q3`] (`re3q3.cc:60-73`), and calls [`re3q3`].
//! 3. [`re3q3`] eliminates one Cayley variable algebraically (picking the
//!    best-conditioned of x/y/z by a 3x3 determinant proxy, `re3q3.cc:126-144`,
//!    with a random-affine-change-of-variables fallback when even the best is
//!    near-singular, `re3q3.cc:146-185`), forming the resultant polynomial
//!    `det(M(x))` (degree <=8, `re3q3.cc:187-368`) whose real roots are the
//!    eliminated variable's solutions; the other two Cayley variables are then
//!    read off via a closed-form 2x2 solve per root (`re3q3.cc:374-389`), and
//!    every solution gets 5 Newton-polish iterations against the *original*
//!    (pre-elimination) quadratics ([`refine_3q3`], `re3q3.cc:82-117`).
//! 4. Each Cayley solution maps to a quaternion `(1,x,y,z)` normalized, then
//!    composed with `q0` to undo the pre-rotation (`re3q3.cc:415-419`); `gp3p`
//!    recovers each candidate's `t` from the eliminated linear relation
//!    (`gp3p.cc:61`).
//!
//! ## Deviations from PoseLib
//! - **Real-root finding.** Upstream isolates the resultant polynomial's real
//!   roots via a Sturm-sequence bisection (`PoseLib/misc/sturm.h`, not
//!   ported). This module instead builds the polynomial's companion matrix
//!   and takes `nalgebra`'s `complex_eigenvalues()`, keeping eigenvalues with
//!   negligible imaginary part ([`real_roots_of_degree_le_8`]) — a companion-
//!   matrix eigendecomposition is a standard way to find polynomial roots and
//!   is already this crate's convention for exactly this kind of minimal-
//!   solver resultant (see `p3p.rs::real_quartic_roots`,
//!   `gr6p.rs`'s `complex_eigenvalues()` use), and is explicitly permitted by
//!   the task brief ("companion/eigen or Sturm approach ... use nalgebra for
//!   the eigen decomposition"). Sturm bisection is faster and can be more
//!   robust for closely-spaced roots at the very high call rates a full
//!   Ceres-based SfM pipeline drives; RANSAC here already discards a bad
//!   hypothesis by reprojection scoring, so this is a performance trade, not
//!   a change to which roots exist in the generic (non-repeated-root) case.
//! - **Auxiliary randomness.** Both the mandatory `re3q3_rotation`
//!   pre-rotation (drawn on *every* call, not just degenerate ones) and the
//!   rare `try_random_var_change` degeneracy fallback draw from
//!   `Eigen::Quaternion::UnitRandom()`/`Vector3d::setRandom()` upstream, i.e.
//!   process-global, unseeded randomness — upstream GP3P is therefore not
//!   itself reproducible run-to-run. This port threads an explicit `u64`
//!   `seed` through to a [`rand::rngs::SmallRng`] instead, so the same seed
//!   always reproduces the same auxiliary rotation and the same roots (needed
//!   for this port's determinism requirement). The auxiliary rotation is
//!   drawn via three random Euler angles rather than `Eigen`'s exact
//!   uniform-on-SO(3) sampler ([`random_rotation_matrix`]) — sufficient for
//!   its only purpose here (breaking the Cayley singularity), not a
//!   statistically uniform sample.
//! - **Panoramic-rig fallback.** COLMAP/PoseLib special-case coincident ray
//!   origins (`generalized_absolute_pose.cc:64-70`) by calling PoseLib's
//!   *separate* `p3p.cc` solver (a different quartic derivation, not
//!   `re3q3`-based). This module instead reuses this crate's existing,
//!   already-tested Grunert P3P solver ([`super::p3p::solve_grunert`],
//!   visibility widened to `pub(crate)` for this call site) for the same
//!   geometric problem (central absolute pose from 3 bearings) rather than
//!   porting a second, algebraically unrelated quartic solver.

use nalgebra::{DMatrix, Matrix3, Point3, SMatrix, UnitQuaternion, Vector3, Vector4};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use visloc_core::geometry::Pose;

use super::p3p::solve_grunert;

/// Up to 8 `rig_from_world` candidate poses from exactly 3 generalized
/// (possibly multi-origin) 2D-3D correspondences. `origins`/`bearings` are
/// ray origins/directions in the *rig* frame (bearings need not be
/// unit-length — normalized internally, matching
/// `GP3PEstimator::Estimate`'s `.normalized()` call on `rays_in_rig`).
/// `seed` drives the auxiliary randomness the algorithm needs internally (see
/// module doc "Auxiliary randomness" deviation) — the same seed always
/// reproduces the same candidate set for the same input.
///
/// Port of `GP3PEstimator::Estimate` (`generalized_absolute_pose.cc:43-79`).
pub fn gp3p_solve(
    origins: &[Point3<f64>; 3],
    bearings: &[Vector3<f64>; 3],
    points_world: &[Point3<f64>; 3],
    seed: u64,
) -> Vec<Pose> {
    let mut x = [Vector3::zeros(); 3];
    for i in 0..3 {
        let Some(n) = bearings[i].try_normalize(1.0e-15) else {
            return Vec::new();
        };
        x[i] = n;
    }

    // Panoramic-rig fallback: all three ray origins coincide
    // (`generalized_absolute_pose.cc:64-70`). See module doc deviation.
    let o0 = origins[0].coords;
    let panoramic = (1..3).all(|i| {
        let oi = origins[i].coords;
        (oi - o0).norm() <= 1.0e-6 * o0.norm().max(oi.norm()).max(1.0)
    });
    if panoramic {
        let Some(candidates) = solve_grunert(points_world, &x) else {
            return Vec::new();
        };
        return candidates
            .into_iter()
            .map(|pose| {
                Pose::from_world_to_camera(
                    pose.world_to_camera.rotation,
                    pose.world_to_camera.translation + o0,
                )
            })
            .collect();
    }

    let mut rng = SmallRng::seed_from_u64(seed);
    gp3p_core(origins, &x, points_world, &mut rng)
}

/// Non-panoramic core: builds `A`, eliminates `t`, solves the resulting
/// linear-in-rotation system via [`re3q3_rotation`], and recovers each
/// candidate's translation. Port of `gp3p()` (`gp3p.cc:36-66`).
fn gp3p_core(
    p: &[Point3<f64>; 3],
    x: &[Vector3<f64>; 3],
    big_x: &[Point3<f64>; 3],
    rng: &mut SmallRng,
) -> Vec<Pose> {
    let mut a = SMatrix::<f64, 6, 13>::zeros();
    for i in 0..3 {
        let xi = x[i];
        let xw = big_x[i].coords;
        let pi = p[i].coords;
        let r0 = 2 * i;
        let r1 = 2 * i + 1;
        a[(r0, 0)] = xi.z;
        a[(r0, 2)] = -xi.x;
        a[(r1, 1)] = xi.z;
        a[(r1, 2)] = -xi.y;
        for j in 0..3 {
            a[(r0, 3 + 3 * j)] = xw[j] * xi.z;
            a[(r0, 3 + 3 * j + 2)] = -xw[j] * xi.x;
            a[(r1, 3 + 3 * j + 1)] = xw[j] * xi.z;
            a[(r1, 3 + 3 * j + 2)] = -xw[j] * xi.y;
        }
        a[(r0, 12)] = -pi.x * xi.z + pi.z * xi.x;
        a[(r1, 12)] = -pi.y * xi.z + pi.z * xi.y;
    }

    let a00: Matrix3<f64> = a.fixed_view::<3, 3>(0, 0).into_owned();
    let Some(b) = a00.try_inverse() else {
        return Vec::new();
    };
    let a30: Matrix3<f64> = a.fixed_view::<3, 3>(3, 0).into_owned();
    let a03_10: SMatrix<f64, 3, 10> = a.fixed_view::<3, 10>(0, 3).into_owned();
    let a33_10: SMatrix<f64, 3, 10> = a.fixed_view::<3, 10>(3, 3).into_owned();
    let rcoeffs = a33_10 - a30 * b * a03_10;

    let quats = re3q3_rotation(&rcoeffs, rng, true);

    let a03_9: SMatrix<f64, 3, 9> = a.fixed_view::<3, 9>(0, 3).into_owned();
    let a_const = Vector3::new(a[(0, 12)], a[(1, 12)], a[(2, 12)]);

    let mut out = Vec::with_capacity(quats.len());
    for q in quats {
        let uq = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3]));
        let r = uq.to_rotation_matrix().into_inner();
        // Column-major flatten of `R`, matching `vec(R)` above
        // (`nalgebra::Matrix` is column-major, same as Eigen's default).
        let vec_r = nalgebra::SVector::<f64, 9>::from_column_slice(r.as_slice());
        let t = -(b * (a03_9 * vec_r + a_const));
        if t.iter().all(|v| v.is_finite()) {
            out.push(Pose::from_world_to_camera(uq, t));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// re3q3 / re3q3_rotation
// ---------------------------------------------------------------------------

/// `(qw, qx, qy, qz)` order throughout this module — matches PoseLib's
/// `quaternion.h` convention ("we dont use Eigen::Quaterniond here since we
/// want qw,qx,qy,qz ordering"). Only converted to `nalgebra::UnitQuaternion`
/// at [`gp3p_core`]'s very end.
type QuatWxyz = Vector4<f64>;

fn quat_to_rotmat(q: &QuatWxyz) -> Matrix3<f64> {
    UnitQuaternion::new_normalize(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3]))
        .to_rotation_matrix()
        .into_inner()
}

/// Port of `quat_multiply` (`PoseLib/misc/quaternion.h:52-59`).
fn quat_multiply(qa: &QuatWxyz, qb: &QuatWxyz) -> QuatWxyz {
    let (qa1, qa2, qa3, qa4) = (qa[0], qa[1], qa[2], qa[3]);
    let (qb1, qb2, qb3, qb4) = (qb[0], qb[1], qb[2], qb[3]);
    Vector4::new(
        qa1 * qb1 - qa2 * qb2 - qa3 * qb3 - qa4 * qb4,
        qa1 * qb2 + qa2 * qb1 + qa3 * qb4 - qa4 * qb3,
        qa1 * qb3 + qa3 * qb1 - qa2 * qb4 + qa4 * qb2,
        qa1 * qb4 + qa2 * qb3 - qa3 * qb2 + qa4 * qb1,
    )
}

/// Symmetry-breaking auxiliary rotation. See module doc "Auxiliary
/// randomness" deviation for why this isn't `Eigen::Quaternion::UnitRandom()`.
fn random_unit_quat_wxyz(rng: &mut SmallRng) -> QuatWxyz {
    let tau = std::f64::consts::TAU;
    let a: f64 = rng.gen_range(0.0..tau);
    let b: f64 = rng.gen_range(0.0..tau);
    let c: f64 = rng.gen_range(0.0..tau);
    let uq = UnitQuaternion::from_euler_angles(a, b, c);
    let xyzw = uq.into_inner().coords; // nalgebra internal order: (x, y, z, w)
    Vector4::new(xyzw[3], xyzw[0], xyzw[1], xyzw[2])
}

fn random_rotation_matrix(rng: &mut SmallRng) -> Matrix3<f64> {
    let tau = std::f64::consts::TAU;
    let a: f64 = rng.gen_range(0.0..tau);
    let b: f64 = rng.gen_range(0.0..tau);
    let c: f64 = rng.gen_range(0.0..tau);
    UnitQuaternion::from_euler_angles(a, b, c)
        .to_rotation_matrix()
        .into_inner()
}

fn random_unit_vector3(rng: &mut SmallRng) -> Vector3<f64> {
    loop {
        let v = Vector3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        if let Some(n) = v.try_normalize(1.0e-9) {
            return n;
        }
    }
}

/// Port of the inhomogeneous `rotation_to_3q3` overload
/// (`re3q3.cc:60-73`): `Rcoeffs*[R(:);1]=0` (3x10, linear in `vec(R)` plus a
/// constant column) converted into the quadratic-in-Cayley-parameter system
/// `coeffs` (3x10, monomial order `x^2,xy,xz,y^2,yz,z^2,x,y,z,1`) used by
/// [`re3q3`].
fn rotation_to_3q3(rcoeffs: &SMatrix<f64, 3, 10>) -> SMatrix<f64, 3, 10> {
    let mut coeffs = SMatrix::<f64, 3, 10>::zeros();
    for k in 0..3 {
        let r = |c: usize| rcoeffs[(k, c)];
        coeffs[(k, 0)] = r(0) - r(4) - r(8) + r(9);
        coeffs[(k, 1)] = 2.0 * r(1) + 2.0 * r(3);
        coeffs[(k, 2)] = 2.0 * r(2) + 2.0 * r(6);
        coeffs[(k, 3)] = r(4) - r(0) - r(8) + r(9);
        coeffs[(k, 4)] = 2.0 * r(5) + 2.0 * r(7);
        coeffs[(k, 5)] = r(8) - r(4) - r(0) + r(9);
        coeffs[(k, 6)] = 2.0 * r(5) - 2.0 * r(7);
        coeffs[(k, 7)] = 2.0 * r(6) - 2.0 * r(2);
        coeffs[(k, 8)] = 2.0 * r(1) - 2.0 * r(3);
        coeffs[(k, 9)] = r(0) + r(4) + r(8) + r(9);
    }
    coeffs
}

/// Port of `refine_3q3` (`re3q3.cc:82-117`): 5 Newton iterations per solution
/// against the original (pre-elimination) quadratics, stopping early once the
/// residual max-norm is below `1e-8`.
fn refine_3q3(coeffs: &SMatrix<f64, 3, 10>, solutions: &mut [(f64, f64, f64)]) {
    for sol in solutions.iter_mut() {
        let (mut x, mut y, mut z) = *sol;
        for _ in 0..5 {
            let r = coeffs.column(0) * (x * x)
                + coeffs.column(1) * (x * y)
                + coeffs.column(2) * (x * z)
                + coeffs.column(3) * (y * y)
                + coeffs.column(4) * (y * z)
                + coeffs.column(5) * (z * z)
                + coeffs.column(6) * x
                + coeffs.column(7) * y
                + coeffs.column(8) * z
                + coeffs.column(9);
            if r.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1.0e-8 {
                break;
            }
            let j = Matrix3::from_columns(&[
                (coeffs.column(0) * (2.0 * x)
                    + coeffs.column(1) * y
                    + coeffs.column(2) * z
                    + coeffs.column(6))
                .into_owned(),
                (coeffs.column(1) * x
                    + coeffs.column(3) * (2.0 * y)
                    + coeffs.column(4) * z
                    + coeffs.column(7))
                .into_owned(),
                (coeffs.column(2) * x
                    + coeffs.column(4) * y
                    + coeffs.column(5) * (2.0 * z)
                    + coeffs.column(8))
                .into_owned(),
            ]);
            let Some(j_inv) = j.try_inverse() else {
                break;
            };
            let dx = j_inv * r;
            x -= dx[0];
            y -= dx[1];
            z -= dx[2];
        }
        *sol = (x, y, z);
    }
}

/// Port of the 10x10 monomial-transform matrix built inline in `re3q3`'s
/// degenerate branch (`re3q3.cc:151-170`) for a random affine change of
/// variables `[x;y;z] = A3*[x';y';z'] + a4` (`a4` = the 3x4 matrix `[A3|a3]`
/// passed in here).
fn monomial_transform_10x10(a4: &SMatrix<f64, 3, 4>) -> SMatrix<f64, 10, 10> {
    let a = |r: usize, c: usize| a4[(r, c)];
    let rows: [[f64; 10]; 10] = [
        [
            a(0, 0) * a(0, 0),
            2.0 * a(0, 0) * a(0, 1),
            2.0 * a(0, 0) * a(0, 2),
            a(0, 1) * a(0, 1),
            2.0 * a(0, 1) * a(0, 2),
            a(0, 2) * a(0, 2),
            2.0 * a(0, 0) * a(0, 3),
            2.0 * a(0, 1) * a(0, 3),
            2.0 * a(0, 2) * a(0, 3),
            a(0, 3) * a(0, 3),
        ],
        [
            a(0, 0) * a(1, 0),
            a(0, 0) * a(1, 1) + a(0, 1) * a(1, 0),
            a(0, 0) * a(1, 2) + a(0, 2) * a(1, 0),
            a(0, 1) * a(1, 1),
            a(0, 1) * a(1, 2) + a(0, 2) * a(1, 1),
            a(0, 2) * a(1, 2),
            a(0, 0) * a(1, 3) + a(0, 3) * a(1, 0),
            a(0, 1) * a(1, 3) + a(0, 3) * a(1, 1),
            a(0, 2) * a(1, 3) + a(0, 3) * a(1, 2),
            a(0, 3) * a(1, 3),
        ],
        [
            a(0, 0) * a(2, 0),
            a(0, 0) * a(2, 1) + a(0, 1) * a(2, 0),
            a(0, 0) * a(2, 2) + a(0, 2) * a(2, 0),
            a(0, 1) * a(2, 1),
            a(0, 1) * a(2, 2) + a(0, 2) * a(2, 1),
            a(0, 2) * a(2, 2),
            a(0, 0) * a(2, 3) + a(0, 3) * a(2, 0),
            a(0, 1) * a(2, 3) + a(0, 3) * a(2, 1),
            a(0, 2) * a(2, 3) + a(0, 3) * a(2, 2),
            a(0, 3) * a(2, 3),
        ],
        [
            a(1, 0) * a(1, 0),
            2.0 * a(1, 0) * a(1, 1),
            2.0 * a(1, 0) * a(1, 2),
            a(1, 1) * a(1, 1),
            2.0 * a(1, 1) * a(1, 2),
            a(1, 2) * a(1, 2),
            2.0 * a(1, 0) * a(1, 3),
            2.0 * a(1, 1) * a(1, 3),
            2.0 * a(1, 2) * a(1, 3),
            a(1, 3) * a(1, 3),
        ],
        [
            a(1, 0) * a(2, 0),
            a(1, 0) * a(2, 1) + a(1, 1) * a(2, 0),
            a(1, 0) * a(2, 2) + a(1, 2) * a(2, 0),
            a(1, 1) * a(2, 1),
            a(1, 1) * a(2, 2) + a(1, 2) * a(2, 1),
            a(1, 2) * a(2, 2),
            a(1, 0) * a(2, 3) + a(1, 3) * a(2, 0),
            a(1, 1) * a(2, 3) + a(1, 3) * a(2, 1),
            a(1, 2) * a(2, 3) + a(1, 3) * a(2, 2),
            a(1, 3) * a(2, 3),
        ],
        [
            a(2, 0) * a(2, 0),
            2.0 * a(2, 0) * a(2, 1),
            2.0 * a(2, 0) * a(2, 2),
            a(2, 1) * a(2, 1),
            2.0 * a(2, 1) * a(2, 2),
            a(2, 2) * a(2, 2),
            2.0 * a(2, 0) * a(2, 3),
            2.0 * a(2, 1) * a(2, 3),
            2.0 * a(2, 2) * a(2, 3),
            a(2, 3) * a(2, 3),
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            a(0, 0),
            a(0, 1),
            a(0, 2),
            a(0, 3),
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            a(1, 0),
            a(1, 1),
            a(1, 2),
            a(1, 3),
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            a(2, 0),
            a(2, 1),
            a(2, 2),
            a(2, 3),
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let mut b = SMatrix::<f64, 10, 10>::zeros();
    for (r, row) in rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            b[(r, c)] = v;
        }
    }
    b
}

/// Real roots of `c[8]*x^8 + c[7]*x^7 + ... + c[0]` via the companion-matrix
/// eigenvalues of the monic polynomial after trimming negligible leading
/// coefficients (handles resultants of degree < 8, common for special
/// correspondence configurations). See module doc "Real-root finding"
/// deviation for why this replaces PoseLib's Sturm-sequence bisection
/// (`sturm::bisect_sturm<8>`, `re3q3.cc:372`).
fn real_roots_of_degree_le_8(c: &[f64; 9]) -> Vec<f64> {
    let scale = c.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if scale <= 0.0 {
        return Vec::new();
    }
    let mut deg = 8usize;
    while deg > 0 && c[deg].abs() <= 1.0e-12 * scale {
        deg -= 1;
    }
    if deg == 0 {
        return Vec::new();
    }
    let lead = c[deg];
    let mut companion = DMatrix::<f64>::zeros(deg, deg);
    for i in 0..deg {
        companion[(0, i)] = -c[deg - 1 - i] / lead;
    }
    for i in 1..deg {
        companion[(i, i - 1)] = 1.0;
    }
    let eig = companion.complex_eigenvalues();
    let mut roots: Vec<f64> = eig
        .iter()
        .filter(|e| e.im.abs() < 1.0e-7 * (1.0 + e.re.abs()))
        .map(|e| e.re)
        .collect();
    roots.sort_by(|a, b| a.total_cmp(b));
    roots
}

/// Port of `re3q3` (`re3q3.cc:123-399`): solves 3 quadrics in 3 unknowns
/// `(x,y,z)` (`coeffs`, monomial order `x^2,xy,xz,y^2,yz,z^2,x,y,z,1`),
/// returning up to 8 real solutions.
fn re3q3(
    coeffs: &SMatrix<f64, 3, 10>,
    try_random_var_change: bool,
    rng: &mut SmallRng,
) -> Vec<(f64, f64, f64)> {
    let ax = Matrix3::from_columns(&[
        coeffs.column(3).into_owned(),
        coeffs.column(5).into_owned(),
        coeffs.column(4).into_owned(),
    ]);
    let ay = Matrix3::from_columns(&[
        coeffs.column(0).into_owned(),
        coeffs.column(5).into_owned(),
        coeffs.column(2).into_owned(),
    ]);
    let az = Matrix3::from_columns(&[
        coeffs.column(3).into_owned(),
        coeffs.column(0).into_owned(),
        coeffs.column(1).into_owned(),
    ]);

    let detx = ax.determinant().abs();
    let dety = ay.determinant().abs();
    let detz = az.determinant().abs();
    let mut elim_var = 0usize;
    let mut det = detx;
    if det < dety {
        det = dety;
        elim_var = 1;
    }
    if det < detz {
        det = detz;
        elim_var = 2;
    }

    if try_random_var_change && det < 1.0e-10 {
        let r0 = random_rotation_matrix(rng);
        let a3 = random_unit_vector3(rng);
        let mut a4 = SMatrix::<f64, 3, 4>::zeros();
        a4.fixed_view_mut::<3, 3>(0, 0).copy_from(&r0);
        a4.fixed_view_mut::<3, 1>(0, 3).copy_from(&a3);
        let m = monomial_transform_10x10(&a4);
        let coeffs_b = coeffs * m;
        let sols_b = re3q3(&coeffs_b, false, rng);
        let mut sols: Vec<(f64, f64, f64)> = sols_b
            .into_iter()
            .map(|(x, y, z)| {
                let v = r0 * Vector3::new(x, y, z) + a3;
                (v.x, v.y, v.z)
            })
            .collect();
        refine_3q3(coeffs, &mut sols);
        return sols;
    }

    let ax_inv = ax.try_inverse().unwrap_or_else(Matrix3::zeros);
    let ay_inv = ay.try_inverse().unwrap_or_else(Matrix3::zeros);
    let az_inv = az.try_inverse().unwrap_or_else(Matrix3::zeros);

    let p: SMatrix<f64, 3, 7> = match elim_var {
        0 => {
            let mut m = SMatrix::<f64, 3, 7>::zeros();
            for (dst, &src) in [0usize, 1, 2, 6, 7, 8, 9].iter().enumerate() {
                m.set_column(dst, &coeffs.column(src));
            }
            -(ax_inv * m)
        }
        1 => {
            let mut m = SMatrix::<f64, 3, 7>::zeros();
            for (dst, &src) in [3usize, 1, 4, 7, 6, 8, 9].iter().enumerate() {
                m.set_column(dst, &coeffs.column(src));
            }
            -(ay_inv * m)
        }
        _ => {
            let mut m = SMatrix::<f64, 3, 7>::zeros();
            for (dst, &src) in [5usize, 4, 2, 8, 7, 6, 9].iter().enumerate() {
                m.set_column(dst, &coeffs.column(src));
            }
            -(az_inv * m)
        }
    };
    let p = |r: usize, c: usize| p[(r, c)];

    let a11: f64 =
        p(0, 1) * p(2, 1) + p(0, 2) * p(1, 1) - p(2, 1) * p(0, 1) - p(2, 2) * p(2, 1) - p(2, 0);
    let a12: f64 = p(0, 1) * p(2, 4) + p(0, 4) * p(2, 1) + p(0, 2) * p(1, 4) + p(0, 5) * p(1, 1)
        - p(2, 1) * p(0, 4)
        - p(2, 4) * p(0, 1)
        - p(2, 2) * p(2, 4)
        - p(2, 5) * p(2, 1)
        - p(2, 3);
    let a13: f64 =
        p(0, 4) * p(2, 4) + p(0, 5) * p(1, 4) - p(2, 4) * p(0, 4) - p(2, 5) * p(2, 4) - p(2, 6);
    let a14: f64 =
        p(0, 1) * p(2, 2) + p(0, 2) * p(1, 2) - p(2, 1) * p(0, 2) - p(2, 2) * p(2, 2) + p(0, 0);
    let a15: f64 = p(0, 1) * p(2, 5) + p(0, 4) * p(2, 2) + p(0, 2) * p(1, 5) + p(0, 5) * p(1, 2)
        - p(2, 1) * p(0, 5)
        - p(2, 4) * p(0, 2)
        - p(2, 2) * p(2, 5)
        - p(2, 5) * p(2, 2)
        + p(0, 3);
    let a16: f64 =
        p(0, 4) * p(2, 5) + p(0, 5) * p(1, 5) - p(2, 4) * p(0, 5) - p(2, 5) * p(2, 5) + p(0, 6);
    let a17: f64 = p(0, 1) * p(2, 0) + p(0, 2) * p(1, 0) - p(2, 1) * p(0, 0) - p(2, 2) * p(2, 0);
    let a18: f64 = p(0, 1) * p(2, 3) + p(0, 4) * p(2, 0) + p(0, 2) * p(1, 3) + p(0, 5) * p(1, 0)
        - p(2, 1) * p(0, 3)
        - p(2, 4) * p(0, 0)
        - p(2, 2) * p(2, 3)
        - p(2, 5) * p(2, 0);
    let a19: f64 = p(0, 1) * p(2, 6) + p(0, 4) * p(2, 3) + p(0, 2) * p(1, 6) + p(0, 5) * p(1, 3)
        - p(2, 1) * p(0, 6)
        - p(2, 4) * p(0, 3)
        - p(2, 2) * p(2, 6)
        - p(2, 5) * p(2, 3);
    let a110: f64 = p(0, 4) * p(2, 6) + p(0, 5) * p(1, 6) - p(2, 4) * p(0, 6) - p(2, 5) * p(2, 6);

    let a21: f64 =
        p(2, 1) * p(2, 1) + p(2, 2) * p(1, 1) - p(1, 1) * p(0, 1) - p(1, 2) * p(2, 1) - p(1, 0);
    let a22: f64 = p(2, 1) * p(2, 4) + p(2, 4) * p(2, 1) + p(2, 2) * p(1, 4) + p(2, 5) * p(1, 1)
        - p(1, 1) * p(0, 4)
        - p(1, 4) * p(0, 1)
        - p(1, 2) * p(2, 4)
        - p(1, 5) * p(2, 1)
        - p(1, 3);
    let a23: f64 =
        p(2, 4) * p(2, 4) + p(2, 5) * p(1, 4) - p(1, 4) * p(0, 4) - p(1, 5) * p(2, 4) - p(1, 6);
    let a24: f64 =
        p(2, 1) * p(2, 2) + p(2, 2) * p(1, 2) - p(1, 1) * p(0, 2) - p(1, 2) * p(2, 2) + p(2, 0);
    let a25: f64 = p(2, 1) * p(2, 5) + p(2, 4) * p(2, 2) + p(2, 2) * p(1, 5) + p(2, 5) * p(1, 2)
        - p(1, 1) * p(0, 5)
        - p(1, 4) * p(0, 2)
        - p(1, 2) * p(2, 5)
        - p(1, 5) * p(2, 2)
        + p(2, 3);
    let a26: f64 =
        p(2, 4) * p(2, 5) + p(2, 5) * p(1, 5) - p(1, 4) * p(0, 5) - p(1, 5) * p(2, 5) + p(2, 6);
    let a27: f64 = p(2, 1) * p(2, 0) + p(2, 2) * p(1, 0) - p(1, 1) * p(0, 0) - p(1, 2) * p(2, 0);
    let a28: f64 = p(2, 1) * p(2, 3) + p(2, 4) * p(2, 0) + p(2, 2) * p(1, 3) + p(2, 5) * p(1, 0)
        - p(1, 1) * p(0, 3)
        - p(1, 4) * p(0, 0)
        - p(1, 2) * p(2, 3)
        - p(1, 5) * p(2, 0);
    let a29: f64 = p(2, 1) * p(2, 6) + p(2, 4) * p(2, 3) + p(2, 2) * p(1, 6) + p(2, 5) * p(1, 3)
        - p(1, 1) * p(0, 6)
        - p(1, 4) * p(0, 3)
        - p(1, 2) * p(2, 6)
        - p(1, 5) * p(2, 3);
    let a210: f64 = p(2, 4) * p(2, 6) + p(2, 5) * p(1, 6) - p(1, 4) * p(0, 6) - p(1, 5) * p(2, 6);

    let t2: f64 = p(2, 1) * p(2, 1);
    let t3: f64 = p(2, 2) * p(2, 2);
    let t4: f64 = p(0, 1) * p(1, 4);
    let t5: f64 = p(0, 4) * p(1, 1);
    let t6: f64 = t4 + t5;
    let t7: f64 = p(0, 2) * p(1, 5);
    let t8: f64 = p(0, 5) * p(1, 2);
    let t9: f64 = t7 + t8;
    let t10: f64 = p(0, 1) * p(1, 5);
    let t11: f64 = p(0, 4) * p(1, 2);
    let t12: f64 = t10 + t11;
    let t13: f64 = p(0, 2) * p(1, 4);
    let t14: f64 = p(0, 5) * p(1, 1);
    let t15: f64 = t13 + t14;
    let t16: f64 = p(2, 1) * p(2, 5);
    let t17: f64 = p(2, 2) * p(2, 4);
    let t18: f64 = t16 + t17;
    let t19: f64 = p(2, 4) * p(2, 4);
    let t20: f64 = p(2, 5) * p(2, 5);
    let a31: f64 = p(0, 0) * p(1, 1) + p(0, 1) * p(1, 0)
        - p(2, 0) * p(2, 1) * 2.0
        - p(0, 1) * t2
        - p(1, 1) * t3
        - p(2, 2) * t2 * 2.0
        + (p(0, 1) * p(0, 1)) * p(1, 1)
        + p(0, 2) * p(1, 1) * p(1, 2)
        + p(0, 1) * p(1, 2) * p(2, 1)
        + p(0, 2) * p(1, 1) * p(2, 1);
    let a32: f64 = p(0, 0) * p(1, 4) + p(0, 1) * p(1, 3) + p(0, 3) * p(1, 1) + p(0, 4) * p(1, 0)
        - p(2, 0) * p(2, 4) * 2.0
        - p(2, 1) * p(2, 3) * 2.0
        - p(0, 4) * t2
        + p(0, 1) * t6
        - p(1, 4) * t3
        + p(1, 1) * t9
        + p(2, 1) * t12
        + p(2, 1) * t15
        - p(2, 1) * t18 * 2.0
        + p(0, 1) * p(0, 4) * p(1, 1)
        + p(0, 2) * p(1, 2) * p(1, 4)
        + p(0, 1) * p(1, 2) * p(2, 4)
        + p(0, 2) * p(1, 1) * p(2, 4)
        - p(0, 1) * p(2, 1) * p(2, 4) * 2.0
        - p(1, 1) * p(2, 2) * p(2, 5) * 2.0
        - p(2, 1) * p(2, 2) * p(2, 4) * 2.0;
    let a33: f64 = p(0, 1) * p(1, 6) + p(0, 3) * p(1, 4) + p(0, 4) * p(1, 3) + p(0, 6) * p(1, 1)
        - p(2, 1) * p(2, 6) * 2.0
        - p(2, 3) * p(2, 4) * 2.0
        + p(0, 4) * t6
        - p(0, 1) * t19
        + p(1, 4) * t9
        - p(1, 1) * t20
        + p(2, 4) * t12
        + p(2, 4) * t15
        - p(2, 4) * t18 * 2.0
        + p(0, 1) * p(0, 4) * p(1, 4)
        + p(0, 5) * p(1, 1) * p(1, 5)
        + p(0, 4) * p(1, 5) * p(2, 1)
        + p(0, 5) * p(1, 4) * p(2, 1)
        - p(0, 4) * p(2, 1) * p(2, 4) * 2.0
        - p(1, 4) * p(2, 2) * p(2, 5) * 2.0
        - p(2, 1) * p(2, 4) * p(2, 5) * 2.0;
    let a34: f64 = p(0, 4) * p(1, 6) + p(0, 6) * p(1, 4)
        - p(2, 4) * p(2, 6) * 2.0
        - p(0, 4) * t19
        - p(1, 4) * t20
        - p(2, 5) * t19 * 2.0
        + (p(0, 4) * p(0, 4)) * p(1, 4)
        + p(0, 5) * p(1, 4) * p(1, 5)
        + p(0, 4) * p(1, 5) * p(2, 4)
        + p(0, 5) * p(1, 4) * p(2, 4);
    let a35: f64 = p(0, 0) * p(1, 2) + p(0, 2) * p(1, 0)
        - p(2, 0) * p(2, 2) * 2.0
        - p(0, 2) * t2
        - p(1, 2) * t3
        - p(2, 1) * t3 * 2.0
        + p(0, 2) * (p(1, 2) * p(1, 2))
        + p(0, 1) * p(0, 2) * p(1, 1)
        + p(0, 1) * p(1, 2) * p(2, 2)
        + p(0, 2) * p(1, 1) * p(2, 2);
    let a36: f64 = p(0, 0) * p(1, 5) + p(0, 2) * p(1, 3) + p(0, 3) * p(1, 2) + p(0, 5) * p(1, 0)
        - p(2, 0) * p(2, 5) * 2.0
        - p(2, 2) * p(2, 3) * 2.0
        - p(0, 5) * t2
        + p(0, 2) * t6
        - p(1, 5) * t3
        + p(1, 2) * t9
        + p(2, 2) * t12
        + p(2, 2) * t15
        - p(2, 2) * t18 * 2.0
        + p(0, 1) * p(0, 5) * p(1, 1)
        + p(0, 2) * p(1, 2) * p(1, 5)
        + p(0, 1) * p(1, 2) * p(2, 5)
        + p(0, 2) * p(1, 1) * p(2, 5)
        - p(0, 2) * p(2, 1) * p(2, 4) * 2.0
        - p(1, 2) * p(2, 2) * p(2, 5) * 2.0
        - p(2, 1) * p(2, 2) * p(2, 5) * 2.0;
    let a37: f64 = p(0, 2) * p(1, 6) + p(0, 3) * p(1, 5) + p(0, 5) * p(1, 3) + p(0, 6) * p(1, 2)
        - p(2, 2) * p(2, 6) * 2.0
        - p(2, 3) * p(2, 5) * 2.0
        + p(0, 5) * t6
        - p(0, 2) * t19
        + p(1, 5) * t9
        - p(1, 2) * t20
        + p(2, 5) * t12
        + p(2, 5) * t15
        - p(2, 5) * t18 * 2.0
        + p(0, 2) * p(0, 4) * p(1, 4)
        + p(0, 5) * p(1, 2) * p(1, 5)
        + p(0, 4) * p(1, 5) * p(2, 2)
        + p(0, 5) * p(1, 4) * p(2, 2)
        - p(0, 5) * p(2, 1) * p(2, 4) * 2.0
        - p(1, 5) * p(2, 2) * p(2, 5) * 2.0
        - p(2, 2) * p(2, 4) * p(2, 5) * 2.0;
    let a38: f64 = p(0, 5) * p(1, 6) + p(0, 6) * p(1, 5)
        - p(2, 5) * p(2, 6) * 2.0
        - p(0, 5) * t19
        - p(1, 5) * t20
        - p(2, 4) * t20 * 2.0
        + p(0, 5) * (p(1, 5) * p(1, 5))
        + p(0, 4) * p(0, 5) * p(1, 4)
        + p(0, 4) * p(1, 5) * p(2, 5)
        + p(0, 5) * p(1, 4) * p(2, 5);
    let a39: f64 = p(0, 0) * p(1, 0) - p(0, 0) * t2 - p(1, 0) * t3 - p(2, 0) * p(2, 0)
        + p(0, 0) * p(0, 1) * p(1, 1)
        + p(0, 2) * p(1, 0) * p(1, 2)
        + p(0, 1) * p(1, 2) * p(2, 0)
        + p(0, 2) * p(1, 1) * p(2, 0)
        - p(2, 0) * p(2, 1) * p(2, 2) * 2.0;
    let a310: f64 = p(0, 0) * p(1, 3) + p(0, 3) * p(1, 0) - p(2, 0) * p(2, 3) * 2.0 - p(0, 3) * t2
        + p(0, 0) * t6
        - p(1, 3) * t3
        + p(1, 0) * t9
        + p(2, 0) * t12
        + p(2, 0) * t15
        - p(2, 0) * t18 * 2.0
        + p(0, 1) * p(0, 3) * p(1, 1)
        + p(0, 2) * p(1, 2) * p(1, 3)
        + p(0, 1) * p(1, 2) * p(2, 3)
        + p(0, 2) * p(1, 1) * p(2, 3)
        - p(0, 0) * p(2, 1) * p(2, 4) * 2.0
        - p(1, 0) * p(2, 2) * p(2, 5) * 2.0
        - p(2, 1) * p(2, 2) * p(2, 3) * 2.0;
    let a311: f64 = p(0, 0) * p(1, 6) + p(0, 3) * p(1, 3) + p(0, 6) * p(1, 0)
        - p(2, 0) * p(2, 6) * 2.0
        - p(0, 6) * t2
        + p(0, 3) * t6
        - p(0, 0) * t19
        - p(1, 6) * t3
        + p(1, 3) * t9
        - p(1, 0) * t20
        + p(2, 3) * t12
        + p(2, 3) * t15
        - p(2, 3) * t18 * 2.0
        - p(2, 3) * p(2, 3)
        + p(0, 0) * p(0, 4) * p(1, 4)
        + p(0, 1) * p(0, 6) * p(1, 1)
        + p(0, 2) * p(1, 2) * p(1, 6)
        + p(0, 5) * p(1, 0) * p(1, 5)
        + p(0, 1) * p(1, 2) * p(2, 6)
        + p(0, 2) * p(1, 1) * p(2, 6)
        + p(0, 4) * p(1, 5) * p(2, 0)
        + p(0, 5) * p(1, 4) * p(2, 0)
        - p(0, 3) * p(2, 1) * p(2, 4) * 2.0
        - p(1, 3) * p(2, 2) * p(2, 5) * 2.0
        - p(2, 0) * p(2, 4) * p(2, 5) * 2.0
        - p(2, 1) * p(2, 2) * p(2, 6) * 2.0;
    let a312: f64 = p(0, 3) * p(1, 6) + p(0, 6) * p(1, 3) - p(2, 3) * p(2, 6) * 2.0 + p(0, 6) * t6
        - p(0, 3) * t19
        + p(1, 6) * t9
        - p(1, 3) * t20
        + p(2, 6) * t12
        + p(2, 6) * t15
        - p(2, 6) * t18 * 2.0
        + p(0, 3) * p(0, 4) * p(1, 4)
        + p(0, 5) * p(1, 3) * p(1, 5)
        + p(0, 4) * p(1, 5) * p(2, 3)
        + p(0, 5) * p(1, 4) * p(2, 3)
        - p(0, 6) * p(2, 1) * p(2, 4) * 2.0
        - p(1, 6) * p(2, 2) * p(2, 5) * 2.0
        - p(2, 3) * p(2, 4) * p(2, 5) * 2.0;
    let a313: f64 = p(0, 6) * p(1, 6) - p(0, 6) * t19 - p(1, 6) * t20 - p(2, 6) * p(2, 6)
        + p(0, 4) * p(0, 6) * p(1, 4)
        + p(0, 5) * p(1, 5) * p(1, 6)
        + p(0, 4) * p(1, 5) * p(2, 6)
        + p(0, 5) * p(1, 4) * p(2, 6)
        - p(2, 4) * p(2, 5) * p(2, 6) * 2.0;

    // det(M(x)); `c[k]` is the coefficient of `x^k`.
    let mut c = [0.0f64; 9];
    c[8] = a14 * a27 * a31 - a17 * a24 * a31 - a11 * a27 * a35 + a17 * a21 * a35 + a11 * a24 * a39
        - a14 * a21 * a39;
    c[7] = a14 * a27 * a32 + a14 * a28 * a31 + a15 * a27 * a31
        - a17 * a24 * a32
        - a17 * a25 * a31
        - a18 * a24 * a31
        - a11 * a27 * a36
        - a11 * a28 * a35
        - a12 * a27 * a35
        + a17 * a21 * a36
        + a17 * a22 * a35
        + a18 * a21 * a35
        + a11 * a25 * a39
        + a12 * a24 * a39
        - a14 * a22 * a39
        - a15 * a21 * a39
        + a11 * a24 * a310
        - a14 * a21 * a310;
    c[6] = a14 * a27 * a33
        + a14 * a28 * a32
        + a14 * a29 * a31
        + a15 * a27 * a32
        + a15 * a28 * a31
        + a16 * a27 * a31
        - a17 * a24 * a33
        - a17 * a25 * a32
        - a17 * a26 * a31
        - a18 * a24 * a32
        - a18 * a25 * a31
        - a19 * a24 * a31
        - a11 * a27 * a37
        - a11 * a28 * a36
        - a11 * a29 * a35
        - a12 * a27 * a36
        - a12 * a28 * a35
        - a13 * a27 * a35
        + a17 * a21 * a37
        + a17 * a22 * a36
        + a17 * a23 * a35
        + a18 * a21 * a36
        + a18 * a22 * a35
        + a19 * a21 * a35
        + a11 * a26 * a39
        + a12 * a25 * a39
        + a13 * a24 * a39
        - a14 * a23 * a39
        - a15 * a22 * a39
        - a16 * a21 * a39
        + a11 * a24 * a311
        + a11 * a25 * a310
        + a12 * a24 * a310
        - a14 * a21 * a311
        - a14 * a22 * a310
        - a15 * a21 * a310;
    c[5] = a14 * a27 * a34
        + a14 * a28 * a33
        + a14 * a29 * a32
        + a15 * a27 * a33
        + a15 * a28 * a32
        + a15 * a29 * a31
        + a16 * a27 * a32
        + a16 * a28 * a31
        - a17 * a24 * a34
        - a17 * a25 * a33
        - a17 * a26 * a32
        - a18 * a24 * a33
        - a18 * a25 * a32
        - a18 * a26 * a31
        - a19 * a24 * a32
        - a19 * a25 * a31
        - a11 * a27 * a38
        - a11 * a28 * a37
        - a11 * a29 * a36
        - a12 * a27 * a37
        - a12 * a28 * a36
        - a12 * a29 * a35
        - a13 * a27 * a36
        - a13 * a28 * a35
        + a17 * a21 * a38
        + a17 * a22 * a37
        + a17 * a23 * a36
        + a18 * a21 * a37
        + a18 * a22 * a36
        + a18 * a23 * a35
        + a19 * a21 * a36
        + a19 * a22 * a35
        + a12 * a26 * a39
        + a13 * a25 * a39
        - a15 * a23 * a39
        - a16 * a22 * a39
        - a24 * a31 * a110
        + a21 * a35 * a110
        + a14 * a31 * a210
        - a11 * a35 * a210
        + a11 * a24 * a312
        + a11 * a25 * a311
        + a11 * a26 * a310
        + a12 * a24 * a311
        + a12 * a25 * a310
        + a13 * a24 * a310
        - a14 * a21 * a312
        - a14 * a22 * a311
        - a14 * a23 * a310
        - a15 * a21 * a311
        - a15 * a22 * a310
        - a16 * a21 * a310;
    c[4] = a14 * a28 * a34
        + a14 * a29 * a33
        + a15 * a27 * a34
        + a15 * a28 * a33
        + a15 * a29 * a32
        + a16 * a27 * a33
        + a16 * a28 * a32
        + a16 * a29 * a31
        - a17 * a25 * a34
        - a17 * a26 * a33
        - a18 * a24 * a34
        - a18 * a25 * a33
        - a18 * a26 * a32
        - a19 * a24 * a33
        - a19 * a25 * a32
        - a19 * a26 * a31
        - a11 * a28 * a38
        - a11 * a29 * a37
        - a12 * a27 * a38
        - a12 * a28 * a37
        - a12 * a29 * a36
        - a13 * a27 * a37
        - a13 * a28 * a36
        - a13 * a29 * a35
        + a17 * a22 * a38
        + a17 * a23 * a37
        + a18 * a21 * a38
        + a18 * a22 * a37
        + a18 * a23 * a36
        + a19 * a21 * a37
        + a19 * a22 * a36
        + a19 * a23 * a35
        + a13 * a26 * a39
        - a16 * a23 * a39
        - a24 * a32 * a110
        - a25 * a31 * a110
        + a21 * a36 * a110
        + a22 * a35 * a110
        + a14 * a32 * a210
        + a15 * a31 * a210
        - a11 * a36 * a210
        - a12 * a35 * a210
        + a11 * a24 * a313
        + a11 * a25 * a312
        + a11 * a26 * a311
        + a12 * a24 * a312
        + a12 * a25 * a311
        + a12 * a26 * a310
        + a13 * a24 * a311
        + a13 * a25 * a310
        - a14 * a21 * a313
        - a14 * a22 * a312
        - a14 * a23 * a311
        - a15 * a21 * a312
        - a15 * a22 * a311
        - a15 * a23 * a310
        - a16 * a21 * a311
        - a16 * a22 * a310;
    c[3] = a14 * a29 * a34
        + a15 * a28 * a34
        + a15 * a29 * a33
        + a16 * a27 * a34
        + a16 * a28 * a33
        + a16 * a29 * a32
        - a17 * a26 * a34
        - a18 * a25 * a34
        - a18 * a26 * a33
        - a19 * a24 * a34
        - a19 * a25 * a33
        - a19 * a26 * a32
        - a11 * a29 * a38
        - a12 * a28 * a38
        - a12 * a29 * a37
        - a13 * a27 * a38
        - a13 * a28 * a37
        - a13 * a29 * a36
        + a17 * a23 * a38
        + a18 * a22 * a38
        + a18 * a23 * a37
        + a19 * a21 * a38
        + a19 * a22 * a37
        + a19 * a23 * a36
        - a24 * a33 * a110
        - a25 * a32 * a110
        - a26 * a31 * a110
        + a21 * a37 * a110
        + a22 * a36 * a110
        + a23 * a35 * a110
        + a14 * a33 * a210
        + a15 * a32 * a210
        + a16 * a31 * a210
        - a11 * a37 * a210
        - a12 * a36 * a210
        - a13 * a35 * a210
        + a11 * a25 * a313
        + a11 * a26 * a312
        + a12 * a24 * a313
        + a12 * a25 * a312
        + a12 * a26 * a311
        + a13 * a24 * a312
        + a13 * a25 * a311
        + a13 * a26 * a310
        - a14 * a22 * a313
        - a14 * a23 * a312
        - a15 * a21 * a313
        - a15 * a22 * a312
        - a15 * a23 * a311
        - a16 * a21 * a312
        - a16 * a22 * a311
        - a16 * a23 * a310;
    c[2] = a15 * a29 * a34 + a16 * a28 * a34 + a16 * a29 * a33
        - a18 * a26 * a34
        - a19 * a25 * a34
        - a19 * a26 * a33
        - a12 * a29 * a38
        - a13 * a28 * a38
        - a13 * a29 * a37
        + a18 * a23 * a38
        + a19 * a22 * a38
        + a19 * a23 * a37
        - a24 * a34 * a110
        - a25 * a33 * a110
        - a26 * a32 * a110
        + a21 * a38 * a110
        + a22 * a37 * a110
        + a23 * a36 * a110
        + a14 * a34 * a210
        + a15 * a33 * a210
        + a16 * a32 * a210
        - a11 * a38 * a210
        - a12 * a37 * a210
        - a13 * a36 * a210
        + a11 * a26 * a313
        + a12 * a25 * a313
        + a12 * a26 * a312
        + a13 * a24 * a313
        + a13 * a25 * a312
        + a13 * a26 * a311
        - a14 * a23 * a313
        - a15 * a22 * a313
        - a15 * a23 * a312
        - a16 * a21 * a313
        - a16 * a22 * a312
        - a16 * a23 * a311;
    c[1] = a16 * a29 * a34 - a19 * a26 * a34 - a13 * a29 * a38 + a19 * a23 * a38
        - a25 * a34 * a110
        - a26 * a33 * a110
        + a22 * a38 * a110
        + a23 * a37 * a110
        + a15 * a34 * a210
        + a16 * a33 * a210
        - a12 * a38 * a210
        - a13 * a37 * a210
        + a12 * a26 * a313
        + a13 * a25 * a313
        + a13 * a26 * a312
        - a15 * a23 * a313
        - a16 * a22 * a313
        - a16 * a23 * a312;
    c[0] = -a26 * a34 * a110 + a23 * a38 * a110 + a16 * a34 * a210 - a13 * a38 * a210
        + a13 * a26 * a313
        - a16 * a23 * a313;

    let roots = real_roots_of_degree_le_8(&c);

    let mut solutions: Vec<(f64, f64, f64)> = Vec::with_capacity(roots.len());
    for xs1 in roots {
        let xs2 = xs1 * xs1;
        let xs3 = xs1 * xs2;
        let xs4 = xs1 * xs3;

        let mut a_mat = Matrix3::<f64>::zeros();
        a_mat[(0, 0)] = a11 * xs2 + a12 * xs1 + a13;
        a_mat[(0, 1)] = a14 * xs2 + a15 * xs1 + a16;
        a_mat[(0, 2)] = a17 * xs3 + a18 * xs2 + a19 * xs1 + a110;
        a_mat[(1, 0)] = a21 * xs2 + a22 * xs1 + a23;
        a_mat[(1, 1)] = a24 * xs2 + a25 * xs1 + a26;
        a_mat[(1, 2)] = a27 * xs3 + a28 * xs2 + a29 * xs1 + a210;
        a_mat[(2, 0)] = a31 * xs3 + a32 * xs2 + a33 * xs1 + a34;
        a_mat[(2, 1)] = a35 * xs3 + a36 * xs2 + a37 * xs1 + a38;
        a_mat[(2, 2)] = a39 * xs4 + a310 * xs3 + a311 * xs2 + a312 * xs1 + a313;

        let y_num = a_mat[(1, 2)] * a_mat[(0, 1)] - a_mat[(0, 2)] * a_mat[(1, 1)];
        let y_den = a_mat[(0, 0)] * a_mat[(1, 1)] - a_mat[(1, 0)] * a_mat[(0, 1)];
        let z_num = a_mat[(1, 2)] * a_mat[(0, 0)] - a_mat[(0, 2)] * a_mat[(1, 0)];
        let z_den = a_mat[(0, 1)] * a_mat[(1, 0)] - a_mat[(1, 1)] * a_mat[(0, 0)];
        let y = y_num / y_den;
        let z = z_num / z_den;
        if y.is_finite() && z.is_finite() {
            solutions.push((xs1, y, z));
        }
    }

    match elim_var {
        1 => {
            for s in solutions.iter_mut() {
                *s = (s.1, s.0, s.2);
            }
        }
        2 => {
            for s in solutions.iter_mut() {
                *s = (s.2, s.1, s.0);
            }
        }
        _ => {}
    }

    refine_3q3(coeffs, &mut solutions);
    solutions
}

/// Port of `re3q3_rotation` (`re3q3.cc:401-421`, inhomogeneous overload):
/// applies the mandatory pre-rotation, converts to the Cayley-quadratic
/// system, solves via [`re3q3`], and composes each solution's quaternion
/// back with the pre-rotation.
fn re3q3_rotation(
    rcoeffs: &SMatrix<f64, 3, 10>,
    rng: &mut SmallRng,
    try_random_var_change: bool,
) -> Vec<QuatWxyz> {
    let q0 = random_unit_quat_wxyz(rng);
    let r0 = quat_to_rotmat(&q0);

    let mut rc = *rcoeffs;
    for g in 0..3 {
        let block: Matrix3<f64> = rc.fixed_view::<3, 3>(0, 3 * g).into_owned();
        let new_block = block * r0;
        rc.fixed_view_mut::<3, 3>(0, 3 * g).copy_from(&new_block);
    }

    let coeffs = rotation_to_3q3(&rc);
    let sols = re3q3(&coeffs, try_random_var_change, rng);

    sols.into_iter()
        .filter_map(|(x, y, z)| {
            let qv = Vector4::new(1.0, x, y, z);
            let n = qv.norm();
            if n <= 1.0e-15 || !n.is_finite() {
                return None;
            }
            let q = qv / n;
            Some(quat_multiply(&q0, &q))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use visloc_core::geometry::SE3;

    /// Synthetic 2-camera rig with a known `rig_from_world` pose, matching
    /// the fixtures in `generalized.rs`'s own tests.
    fn test_rig() -> (SE3, SE3) {
        // (sensor1_from_rig, sensor2_from_rig)
        (
            SE3::identity(),
            SE3::new(
                nalgebra::UnitQuaternion::from_euler_angles(0.002, -0.004, 0.003),
                Vector3::new(-0.20, 0.001, -0.002),
            ),
        )
    }

    fn truth_pose() -> SE3 {
        SE3::new(
            nalgebra::UnitQuaternion::from_euler_angles(0.06, -0.12, 0.04),
            Vector3::new(0.35, -0.18, 0.42),
        )
    }

    /// `(origins, bearings, world_points)`, all rig-frame except
    /// `world_points`.
    type Triple = ([Point3<f64>; 3], [Vector3<f64>; 3], [Point3<f64>; 3]);

    /// Three correspondences spanning both sensors (non-panoramic): world
    /// points + which sensor observed them, from which we derive rig-frame
    /// origins/bearings under the ground-truth pose.
    fn synthetic_triple(rig_from_world: &SE3) -> Triple {
        let (s1, s2) = test_rig();
        let world_points = [
            Point3::new(0.3, -0.4, 4.2),
            Point3::new(-0.6, 0.5, 3.6),
            Point3::new(0.15, 0.35, 5.1),
        ];
        let sensors = [&s1, &s2, &s1];
        let mut origins = [Point3::origin(); 3];
        let mut bearings = [Vector3::zeros(); 3];
        for i in 0..3 {
            let sensor_from_rig = sensors[i];
            let rig_from_sensor = sensor_from_rig.inverse();
            origins[i] = Point3::from(rig_from_sensor.translation);
            let point_rig = rig_from_world.transform_point(&world_points[i]);
            let point_sensor = sensor_from_rig.transform_point(&point_rig);
            let bearing_sensor = point_sensor.coords.normalize();
            bearings[i] = sensor_from_rig
                .rotation
                .inverse_transform_vector(&bearing_sensor);
        }
        (origins, bearings, world_points)
    }

    /// C3 task item: "GP3P recovers the pose from 3 noise-free
    /// correspondences (all real roots checked)". `re3q3` can return up to 8
    /// real roots; at least one candidate must match ground truth, and every
    /// *returned* candidate must actually satisfy the 3 input correspondences
    /// (small residual) since `re3q3` doesn't itself disambiguate — that's
    /// RANSAC's job.
    #[test]
    fn gp3p_recovers_pose_from_three_noise_free_correspondences() {
        let truth = truth_pose();
        let (origins, bearings, world_points) = synthetic_triple(&truth);
        let candidates = gp3p_solve(&origins, &bearings, &world_points, 42);
        assert!(!candidates.is_empty(), "gp3p returned no candidates");

        let mut best_rot = f64::INFINITY;
        let mut best_trans = f64::INFINITY;
        for pose in &candidates {
            let rot_err = (pose.world_to_camera.rotation * truth.rotation.inverse()).angle();
            let trans_err = (pose.world_to_camera.translation - truth.translation).norm();
            best_rot = best_rot.min(rot_err);
            best_trans = best_trans.min(trans_err);

            // Every returned candidate must be an (algebraic) solution of
            // the 3 input equations `cross(x_i, R*X_i + t - p_i) = 0`, i.e.
            // the transformed point must be *collinear* with the ray through
            // `p_i` along `x_i` — but not necessarily on the `lambda_i > 0`
            // half (the cross-product constraint is sign-blind, the classic
            // minimal-solver cheirality ambiguity; COLMAP's own
            // `GP3PEstimator::Residuals` explicitly re-checks
            // `point3D_in_cam.z() > 0` downstream,
            // `generalized_absolute_pose.cc:97`, rather than the solver
            // itself only ever returning forward-facing solutions). So this
            // checks collinearity (either direction), not same-direction.
            for i in 0..3 {
                let point_rig = pose.world_to_camera.transform_point(&world_points[i]);
                let dir = (point_rig - origins[i]).normalize();
                let cos = dir.dot(&bearings[i]).abs().clamp(-1.0, 1.0);
                assert!(
                    cos.acos() < 1.0e-4,
                    "candidate does not lie on input correspondence {i}'s line: angle {}",
                    cos.acos()
                );
            }
        }
        assert!(best_rot < 1.0e-6, "best rotation error {best_rot}");
        assert!(best_trans < 1.0e-6, "best translation error {best_trans}");
    }

    /// C3 task item: determinism with a fixed seed.
    #[test]
    fn gp3p_is_deterministic_for_a_fixed_seed() {
        let truth = truth_pose();
        let (origins, bearings, world_points) = synthetic_triple(&truth);
        let a = gp3p_solve(&origins, &bearings, &world_points, 1234);
        let b = gp3p_solve(&origins, &bearings, &world_points, 1234);
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.world_to_camera.rotation, pb.world_to_camera.rotation);
            assert_eq!(
                pa.world_to_camera.translation,
                pb.world_to_camera.translation
            );
        }
    }

    /// Different seeds still recover the correct pose (the auxiliary
    /// pre-rotation must not change *which* geometric solutions are found,
    /// only bookkeeping/numerics).
    #[test]
    fn gp3p_recovers_pose_across_multiple_seeds() {
        let truth = truth_pose();
        let (origins, bearings, world_points) = synthetic_triple(&truth);
        for seed in [0u64, 1, 7, 99, 123456] {
            let candidates = gp3p_solve(&origins, &bearings, &world_points, seed);
            let found = candidates.iter().any(|pose| {
                let rot_err = (pose.world_to_camera.rotation * truth.rotation.inverse()).angle();
                let trans_err = (pose.world_to_camera.translation - truth.translation).norm();
                rot_err < 1.0e-5 && trans_err < 1.0e-5
            });
            assert!(found, "seed {seed}: ground truth not recovered");
        }
    }
}
