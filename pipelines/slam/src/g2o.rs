//! `.g2o` pose-graph exchange format I/O for [`PoseGraph`].
//!
//! [g2o](https://github.com/RainerKuemmerle/g2o) is the de-facto text format
//! for SE(3) pose-graph benchmarks (sphere2500, parking-garage, etc.), so
//! supporting it lets visloc-rs's [`PoseGraph::optimize_se3_iterative`] run
//! head-to-head on the same canonical datasets the SLAM community uses. Only
//! the 3D pose-graph subset is handled:
//!
//! ```text
//! VERTEX_SE3:QUAT  i  x y z  qx qy qz qw
//! EDGE_SE3:QUAT    i j  x y z  qx qy qz qw  <21 upper-triangular info entries>
//! FIX              i
//! ```
//!
//! The 21 information-matrix entries are the row-major upper triangle of the
//! 6×6 matrix in g2o's `[translation; rotation]` ordering, which matches this
//! crate's [`SE3::log`] tangent layout `[ρ; ω]`, so no axis permutation is
//! needed — but a measurement-adjoint congruence IS (see Convention).
//!
//! # Convention
//!
//! g2o's `EDGE_SE3:QUAT` measurement is the *body-frame* relative pose
//! `Z = T_i⁻¹ · T_j` (the pose of `j` seen from `i`), whereas this crate's
//! solver compares against the *world-frame* relative `T_to · T_from⁻¹`. The
//! two are reconciled by storing the inverse of every transform: a vertex
//! `T` becomes `world_to_camera = T⁻¹` and an edge measurement `Z` becomes
//! `Z⁻¹`. With that mapping a consistent g2o graph yields exactly zero
//! residual (verified in the unit tests), and [`write_g2o`] inverts back so a
//! load/save round-trip is the identity.
//!
//! Inverting the measurement also rotates the solver's residual by the
//! measurement adjoint — `r_visloc = −Ad(Z)·e_g2o` — so to keep the *weighted*
//! cost `rᵀΩr` equal to g2o's `e_g2oᵀΩe_g2o` the information is carried by the
//! matching congruence `Ω → Ad(Z⁻¹)ᵀ Ω Ad(Z⁻¹)` on read (undone on write). This
//! is a no-op for isotropic `Ω` (hence consistent / unit-info graphs were always
//! correct) but is essential for the anisotropic information of datasets like
//! sphere2500 / cubicle, where omitting it makes visloc minimize a subtly
//! different (adjoint-twisted) cost and converge to a worse optimum than the
//! g2o / GTSAM reference.

use std::path::Path;

use nalgebra::{Matrix6, Quaternion, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, SE3};

use crate::{PoseGraph, PoseGraphEdgeKind};

/// Error returned while reading a `.g2o` file.
#[derive(Debug)]
pub enum G2oError {
    /// The file could not be read.
    Io(std::io::Error),
    /// A line was syntactically malformed (unknown tag, missing column, bad
    /// number). `line` is 1-based.
    Syntax { line: usize, reason: String },
}

impl std::fmt::Display for G2oError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            G2oError::Io(e) => write!(f, "g2o I/O error: {e}"),
            G2oError::Syntax { line, reason } => {
                write!(f, "g2o syntax error on line {line}: {reason}")
            }
        }
    }
}

impl std::error::Error for G2oError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            G2oError::Io(e) => Some(e),
            G2oError::Syntax { .. } => None,
        }
    }
}

/// Parse a `.g2o` file into a [`PoseGraph`].
///
/// The anchor (fixed gauge) is the vertex named by a `FIX` line if present,
/// otherwise the smallest vertex id — the conventional "fix vertex 0" choice.
/// Unknown line tags (2D vertices/edges, parameters, etc.) are ignored so a
/// mixed file still loads its SE(3) subset.
pub fn read_g2o(path: impl AsRef<Path>) -> Result<PoseGraph, G2oError> {
    let text = std::fs::read_to_string(path).map_err(G2oError::Io)?;
    let mut graph = PoseGraph::new();
    let mut fixed: Option<u64> = None;
    let mut min_vertex: Option<u64> = None;

    for (idx, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let lineno = idx + 1;
        let mut tok = line.split_ascii_whitespace();
        let tag = tok.next().unwrap_or("");
        match tag {
            "VERTEX_SE3:QUAT" => {
                let id = next_u64(&mut tok, lineno, "vertex id")?;
                let transform = read_se3(&mut tok, lineno)?;
                // Stored inverted (see module convention).
                graph.add_pose(
                    id,
                    Pose {
                        world_to_camera: transform.inverse(),
                    },
                );
                min_vertex = Some(min_vertex.map_or(id, |m| m.min(id)));
            }
            "EDGE_SE3:QUAT" => {
                let from = next_u64(&mut tok, lineno, "edge from")?;
                let to = next_u64(&mut tok, lineno, "edge to")?;
                let measurement = read_se3(&mut tok, lineno)?;
                let information = read_information(&mut tok, lineno)?;
                let kind = if to == from + 1 {
                    PoseGraphEdgeKind::Sequential
                } else {
                    PoseGraphEdgeKind::LoopClosure
                };
                // Inverting the measurement (Z → Z⁻¹, module convention) rotates
                // the solver's residual by the measurement adjoint:
                // r_visloc = −Ad(Z)·e_g2o (e_g2o = log(Z⁻¹ T_i⁻¹ T_j)). For the
                // weighted cost rᵀΩr to equal g2o's e_g2oᵀΩe_g2o, the information
                // must be carried by the same congruence: Ω → Ad(Z⁻¹)ᵀ Ω Ad(Z⁻¹)
                // (= Ad(Z)⁻ᵀ Ω Ad(Z)⁻¹). For isotropic Ω this is a no-op (Ad is
                // orthogonal up to the translation shear, and Ω·I commutes), which
                // is why consistent/zero-residual graphs were unaffected; for the
                // anisotropic info of sphere2500 / cubicle it is essential.
                let m_inv = measurement.inverse();
                let ad = m_inv.adjoint();
                let information = ad.transpose() * information * ad;
                graph.add_edge_with_information(from, to, m_inv, kind, information);
            }
            "FIX" => {
                fixed = Some(next_u64(&mut tok, lineno, "fix id")?);
            }
            // 2D types, parameter blocks, etc. are not part of the SE(3)
            // pose-graph subset; skip them rather than failing the whole load.
            _ => continue,
        }
    }

    if let Some(anchor) = fixed.or(min_vertex) {
        graph.anchor(anchor);
    }
    Ok(graph)
}

/// Write a [`PoseGraph`] to a `.g2o` file. Inverts the storage convention so
/// the output round-trips through [`read_g2o`]; edges without an explicit
/// information matrix are emitted as `weight · I₆`.
pub fn write_g2o(graph: &PoseGraph, path: impl AsRef<Path>) -> std::io::Result<()> {
    let mut text = String::new();
    for (id, pose) in graph.poses.iter() {
        let t = pose.world_to_camera.inverse();
        let q = t.rotation.into_inner();
        text.push_str(&format!(
            "VERTEX_SE3:QUAT {id} {x} {y} {z} {qx} {qy} {qz} {qw}\n",
            x = t.translation.x,
            y = t.translation.y,
            z = t.translation.z,
            qx = q.i,
            qy = q.j,
            qz = q.k,
            qw = q.w,
        ));
    }
    if let Some(anchor) = graph.anchor {
        text.push_str(&format!("FIX {anchor}\n"));
    }
    for edge in &graph.edges {
        let z = edge.measurement.inverse();
        let q = z.rotation.into_inner();
        // Undo the adjoint congruence applied on read so the file round-trips:
        // the stored Ω is Ad(Z⁻¹)ᵀ Ω_g2o Ad(Z⁻¹), so Ω_g2o = Ad(Z)ᵀ Ω Ad(Z)
        // with Z = edge.measurement.inverse() (= the g2o measurement).
        let ad = z.adjoint();
        let omega = ad.transpose()
            * edge
                .information
                .unwrap_or_else(|| Matrix6::identity() * edge.weight)
            * ad;
        let mut line = format!(
            "EDGE_SE3:QUAT {from} {to} {x} {y} {z} {qx} {qy} {qz} {qw}",
            from = edge.from,
            to = edge.to,
            x = z.translation.x,
            y = z.translation.y,
            z = z.translation.z,
            qx = q.i,
            qy = q.j,
            qz = q.k,
            qw = q.w,
        );
        for row in 0..6 {
            for col in row..6 {
                line.push_str(&format!(" {}", omega[(row, col)]));
            }
        }
        line.push('\n');
        text.push_str(&line);
    }
    std::fs::write(path, text)
}

fn read_se3<'a, I: Iterator<Item = &'a str>>(tok: &mut I, lineno: usize) -> Result<SE3, G2oError> {
    let x = next_f64(tok, lineno, "x")?;
    let y = next_f64(tok, lineno, "y")?;
    let z = next_f64(tok, lineno, "z")?;
    let qx = next_f64(tok, lineno, "qx")?;
    let qy = next_f64(tok, lineno, "qy")?;
    let qz = next_f64(tok, lineno, "qz")?;
    let qw = next_f64(tok, lineno, "qw")?;
    let rotation = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));
    Ok(SE3::new(rotation, Vector3::new(x, y, z)))
}

/// Read the 21 upper-triangular entries of the 6×6 information matrix, mirror
/// them into a symmetric matrix, and project the result onto the
/// positive-semidefinite cone (see `project_to_psd`).
fn read_information<'a, I: Iterator<Item = &'a str>>(
    tok: &mut I,
    lineno: usize,
) -> Result<Matrix6<f64>, G2oError> {
    let mut omega = Matrix6::zeros();
    for row in 0..6 {
        for col in row..6 {
            let value = next_f64(tok, lineno, "information entry")?;
            omega[(row, col)] = value;
            omega[(col, row)] = value;
        }
    }
    Ok(project_to_psd(omega))
}

/// Project a symmetric matrix onto the positive-semidefinite cone by clamping
/// its eigenvalues at zero (`V·max(Λ, 0)·Vᵀ`).
///
/// An information matrix (inverse covariance) must be PSD, but real `.g2o`
/// datasets derived from scan matching — notably `cubicle` and `rim` — ship
/// edges whose information matrices are *not* PSD (e.g. a rotation sub-block
/// with off-diagonal entries far larger than its diagonal). Fed straight into
/// the Gauss-Newton normal equations such an `Ω` makes the assembled `H`
/// indefinite, so the Cholesky factorization fails outright (no amount of
/// Levenberg damping rescues a matrix with a large negative eigenvalue).
///
/// Clamping the negative eigenvalues to zero discards only the spurious
/// negative-curvature directions while preserving the well-posed part of the
/// constraint, and is exactly the identity on a genuine information matrix.
fn project_to_psd(omega: Matrix6<f64>) -> Matrix6<f64> {
    // Symmetrize defensively before decomposing (guards against round-off; the
    // parser already mirrors the entries).
    let symmetric = (omega + omega.transpose()) * 0.5;
    let eigen = symmetric.symmetric_eigen();
    if eigen.eigenvalues.iter().all(|&lambda| lambda >= 0.0) {
        // Already PSD: return the symmetrized matrix untouched so valid data
        // round-trips bit-for-bit (no reconstruction round-off).
        return symmetric;
    }
    let clamped = eigen.eigenvalues.map(|lambda| lambda.max(0.0));
    let v = eigen.eigenvectors;
    v * Matrix6::from_diagonal(&clamped) * v.transpose()
}

fn next_u64<'a, I: Iterator<Item = &'a str>>(
    tok: &mut I,
    lineno: usize,
    field: &str,
) -> Result<u64, G2oError> {
    let raw = tok.next().ok_or_else(|| G2oError::Syntax {
        line: lineno,
        reason: format!("missing {field}"),
    })?;
    raw.parse::<u64>().map_err(|e| G2oError::Syntax {
        line: lineno,
        reason: format!("bad {field} '{raw}': {e}"),
    })
}

fn next_f64<'a, I: Iterator<Item = &'a str>>(
    tok: &mut I,
    lineno: usize,
    field: &str,
) -> Result<f64, G2oError> {
    let raw = tok.next().ok_or_else(|| G2oError::Syntax {
        line: lineno,
        reason: format!("missing {field}"),
    })?;
    raw.parse::<f64>().map_err(|e| G2oError::Syntax {
        line: lineno,
        reason: format!("bad {field} '{raw}': {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PoseGraphSe3Config;

    fn pose(tx: f64, ty: f64, tz: f64, yaw: f64) -> Pose {
        let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, yaw);
        Pose {
            world_to_camera: SE3::new(rotation, Vector3::new(tx, ty, tz)),
        }
    }

    /// Write a graph, read it back, and confirm vertices, edges, anchor and the
    /// information matrices all survive the round-trip.
    #[test]
    fn g2o_round_trips_through_disk() {
        let dir = std::env::temp_dir().join(format!("visloc_g2o_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("graph.g2o");

        let mut graph = PoseGraph::new();
        graph.add_pose(0, pose(0.0, 0.0, 0.0, 0.0));
        graph.add_pose(1, pose(1.0, 0.2, 0.0, 0.1));
        graph.add_pose(2, pose(2.0, -0.1, 0.3, -0.2));
        graph.anchor(0);
        let m01 = SE3::new(UnitQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0));
        let mut info = Matrix6::identity();
        info[(0, 0)] = 25.0;
        info[(3, 3)] = 4.0;
        graph.add_edge_with_information(0, 1, m01.clone(), PoseGraphEdgeKind::Sequential, info);
        graph.add_edge_with_information(
            1,
            2,
            m01,
            PoseGraphEdgeKind::Sequential,
            Matrix6::identity(),
        );

        write_g2o(&graph, &path).unwrap();
        let loaded = read_g2o(&path).unwrap();

        assert_eq!(loaded.poses.len(), 3);
        assert_eq!(loaded.edges.len(), 2);
        assert_eq!(loaded.anchor, Some(0));
        for id in [0u64, 1, 2] {
            let a = &graph.poses[&id].world_to_camera;
            let b = &loaded.poses[&id].world_to_camera;
            assert!((a.translation - b.translation).norm() < 1e-9);
            assert!(a.rotation.angle_to(&b.rotation) < 1e-9);
        }
        let info_back = loaded.edges[0].information.expect("info preserved");
        assert!((info_back[(0, 0)] - 25.0).abs() < 1e-9);
        assert!((info_back[(3, 3)] - 4.0).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A geometrically consistent g2o graph (edges equal to the true relative
    /// poses) must load with ~zero cost, proving the inverse-mapping
    /// convention is correct end-to-end.
    #[test]
    fn consistent_g2o_graph_loads_with_zero_cost() {
        let dir =
            std::env::temp_dir().join(format!("visloc_g2o_consistent_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("consistent.g2o");

        // Three ground-truth world poses (as raw g2o transforms T).
        let t0 = SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let t1 = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.3),
            Vector3::new(1.0, 0.0, 0.0),
        );
        let t2 = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, -0.2),
            Vector3::new(2.0, 0.5, 0.0),
        );
        // g2o body-frame relative measurements Z = T_i⁻¹ · T_j.
        let z01 = t0.inverse().compose(&t1);
        let z12 = t1.inverse().compose(&t2);
        let z02 = t0.inverse().compose(&t2); // a "loop" edge

        let fmt_se3 = |t: &SE3| {
            let q = t.rotation.into_inner();
            format!(
                "{} {} {} {} {} {} {}",
                t.translation.x, t.translation.y, t.translation.z, q.i, q.j, q.k, q.w
            )
        };
        let identity_info: String = {
            let mut s = String::new();
            for row in 0..6 {
                for col in row..6 {
                    s.push_str(if row == col { " 1" } else { " 0" });
                }
            }
            s
        };
        let mut text = String::new();
        text.push_str(&format!("VERTEX_SE3:QUAT 0 {}\n", fmt_se3(&t0)));
        text.push_str(&format!("VERTEX_SE3:QUAT 1 {}\n", fmt_se3(&t1)));
        text.push_str(&format!("VERTEX_SE3:QUAT 2 {}\n", fmt_se3(&t2)));
        text.push_str("FIX 0\n");
        text.push_str(&format!(
            "EDGE_SE3:QUAT 0 1 {}{}\n",
            fmt_se3(&z01),
            identity_info
        ));
        text.push_str(&format!(
            "EDGE_SE3:QUAT 1 2 {}{}\n",
            fmt_se3(&z12),
            identity_info
        ));
        text.push_str(&format!(
            "EDGE_SE3:QUAT 0 2 {}{}\n",
            fmt_se3(&z02),
            identity_info
        ));
        std::fs::write(&path, text).unwrap();

        let graph = read_g2o(&path).unwrap();
        assert!(
            graph.se3_cost() < 1e-12,
            "consistent graph must have ~zero cost, got {}",
            graph.se3_cost()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// With an ANISOTROPIC information matrix, the loaded cost must equal g2o's
    /// own weighted residual `e_g2oᵀ Ω e_g2o` (`e_g2o = log(Z⁻¹ T_i⁻¹ T_j)`).
    ///
    /// This guards the adjoint-congruence the reader applies to `Ω`: inverting
    /// the measurement (the storage convention) rotates the solver's residual by
    /// `Ad(Z)`, so the information must be carried by the matching congruence
    /// `Ω → Ad(Z⁻¹)ᵀ Ω Ad(Z⁻¹)` or the weighted cost — and hence the optimum —
    /// silently differs from the reference (g2o / GTSAM) for any non-isotropic
    /// `Ω`. A consistent graph has zero residual and so cannot catch this; a
    /// non-zero residual with distinct translation/rotation weights does.
    #[test]
    fn anisotropic_g2o_cost_matches_the_g2o_convention() {
        use nalgebra::Vector6;

        // Two raw g2o world poses with a deliberate (non-zero) edge residual.
        let t0 = SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let t1 = SE3::new(
            UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
            Vector3::new(1.0, 0.5, -0.3),
        );
        // A measurement that does NOT match T_0⁻¹ T_1 (so e_g2o ≠ 0), with both a
        // rotation and a translation so Ad(Z) genuinely mixes the two blocks.
        let z = SE3::new(
            UnitQuaternion::from_euler_angles(-0.05, 0.15, 0.2),
            Vector3::new(0.8, 0.2, 0.1),
        );
        // Strongly anisotropic info (sphere2500-like): translation ≪ rotation.
        let omega =
            Matrix6::<f64>::from_diagonal(&Vector6::new(10.0, 10.0, 10.0, 400.0, 400.0, 99.0));

        // g2o-convention residual and weighted cost (the reference value).
        let e_g2o = z.inverse().compose(&t0.inverse()).compose(&t1).log();
        let expected = (e_g2o.transpose() * omega * e_g2o)[(0, 0)];

        let fmt_se3 = |t: &SE3| {
            let q = t.rotation.into_inner();
            format!(
                "{} {} {} {} {} {} {}",
                t.translation.x, t.translation.y, t.translation.z, q.i, q.j, q.k, q.w
            )
        };
        let info_str: String = {
            let mut s = String::new();
            for row in 0..6 {
                for col in row..6 {
                    s.push_str(&format!(" {}", omega[(row, col)]));
                }
            }
            s
        };
        let text = format!(
            "VERTEX_SE3:QUAT 0 {}\nVERTEX_SE3:QUAT 1 {}\nFIX 0\nEDGE_SE3:QUAT 0 1 {}{}\n",
            fmt_se3(&t0),
            fmt_se3(&t1),
            fmt_se3(&z),
            info_str,
        );
        let dir = std::env::temp_dir().join(format!("visloc_g2o_aniso_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("aniso.g2o");
        std::fs::write(&path, text).unwrap();

        let graph = read_g2o(&path).unwrap();
        let cost = graph.se3_cost();
        assert!(
            (cost - expected).abs() < 1e-9,
            "loaded anisotropic cost {cost} must match g2o convention {expected}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Perturb one vertex of an otherwise-consistent graph; the SE(3) solver
    /// must drive the cost back down toward zero.
    #[test]
    fn perturbed_g2o_graph_is_recovered_by_optimization() {
        let dir = std::env::temp_dir().join(format!("visloc_g2o_recover_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("perturbed.g2o");

        let t0 = SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let t1 = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.3),
            Vector3::new(1.0, 0.0, 0.0),
        );
        let t2 = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, -0.2),
            Vector3::new(2.0, 0.5, 0.0),
        );
        let z01 = t0.inverse().compose(&t1);
        let z12 = t1.inverse().compose(&t2);
        let z02 = t0.inverse().compose(&t2);
        // Drag vertex 2 away from its true pose to inject drift.
        let t2_bad = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.4),
            Vector3::new(1.3, -0.4, 0.2),
        );

        let fmt_se3 = |t: &SE3| {
            let q = t.rotation.into_inner();
            format!(
                "{} {} {} {} {} {} {}",
                t.translation.x, t.translation.y, t.translation.z, q.i, q.j, q.k, q.w
            )
        };
        let identity_info: String = {
            let mut s = String::new();
            for row in 0..6 {
                for col in row..6 {
                    s.push_str(if row == col { " 1" } else { " 0" });
                }
            }
            s
        };
        let mut text = String::new();
        text.push_str(&format!("VERTEX_SE3:QUAT 0 {}\n", fmt_se3(&t0)));
        text.push_str(&format!("VERTEX_SE3:QUAT 1 {}\n", fmt_se3(&t1)));
        text.push_str(&format!("VERTEX_SE3:QUAT 2 {}\n", fmt_se3(&t2_bad)));
        text.push_str("FIX 0\n");
        text.push_str(&format!(
            "EDGE_SE3:QUAT 0 1 {}{}\n",
            fmt_se3(&z01),
            identity_info
        ));
        text.push_str(&format!(
            "EDGE_SE3:QUAT 1 2 {}{}\n",
            fmt_se3(&z12),
            identity_info
        ));
        text.push_str(&format!(
            "EDGE_SE3:QUAT 0 2 {}{}\n",
            fmt_se3(&z02),
            identity_info
        ));
        std::fs::write(&path, text).unwrap();

        let mut graph = read_g2o(&path).unwrap();
        let cost_before = graph.se3_cost();
        assert!(
            cost_before > 1e-3,
            "expected injected drift, got {cost_before}"
        );
        let result = graph
            .optimize_se3_iterative(&PoseGraphSe3Config::default())
            .expect("solve");
        assert!(result.converged);
        assert!(
            result.final_cost < 1e-9,
            "solver should recover the consistent graph, final cost {}",
            result.final_cost
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Marquardt diagonal damping (`H + λ·diag(H)`) is a correct LM variant: on a
    /// solvable graph it must reach the SAME optimum as the default identity
    /// damping (`H + λI`). It differs only in the convergence path / robustness on
    /// ill-conditioned graphs, never in the answer on a well-posed one.
    #[test]
    fn diagonal_damping_reaches_the_same_optimum_as_identity() {
        use crate::DampingMode;

        let t0 = SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let t1 = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.3),
            Vector3::new(1.0, 0.0, 0.0),
        );
        let t2 = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, -0.2),
            Vector3::new(2.0, 0.5, 0.0),
        );
        let z01 = t0.inverse().compose(&t1);
        let z12 = t1.inverse().compose(&t2);
        let z02 = t0.inverse().compose(&t2);
        let t2_bad = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.4),
            Vector3::new(1.3, -0.4, 0.2),
        );
        let fmt_se3 = |t: &SE3| {
            let q = t.rotation.into_inner();
            format!(
                "{} {} {} {} {} {} {}",
                t.translation.x, t.translation.y, t.translation.z, q.i, q.j, q.k, q.w
            )
        };
        // Anisotropic info so identity vs diagonal damping are genuinely different.
        let info: String = {
            let vals = [10.0, 10.0, 10.0, 400.0, 400.0, 99.0];
            let mut s = String::new();
            for row in 0..6 {
                for col in row..6 {
                    s.push_str(&format!(" {}", if row == col { vals[row] } else { 0.0 }));
                }
            }
            s
        };
        let text = format!(
            "VERTEX_SE3:QUAT 0 {}\nVERTEX_SE3:QUAT 1 {}\nVERTEX_SE3:QUAT 2 {}\nFIX 0\n\
             EDGE_SE3:QUAT 0 1 {}{}\nEDGE_SE3:QUAT 1 2 {}{}\nEDGE_SE3:QUAT 0 2 {}{}\n",
            fmt_se3(&t0),
            fmt_se3(&t1),
            fmt_se3(&t2_bad),
            fmt_se3(&z01),
            info,
            fmt_se3(&z12),
            info,
            fmt_se3(&z02),
            info,
        );
        let dir = std::env::temp_dir().join(format!("visloc_g2o_diag_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("diag.g2o");
        std::fs::write(&path, text).unwrap();

        let lm = |damping| PoseGraphSe3Config {
            initial_lambda: Some(1e-3),
            damping,
            ..PoseGraphSe3Config::default()
        };
        let mut g_id = read_g2o(&path).unwrap();
        let r_id = g_id
            .optimize_se3_iterative(&lm(DampingMode::Identity))
            .unwrap();
        let mut g_diag = read_g2o(&path).unwrap();
        let r_diag = g_diag
            .optimize_se3_iterative(&lm(DampingMode::Diagonal))
            .unwrap();

        assert!(r_id.converged && r_diag.converged);
        assert!(
            r_diag.final_cost < 1e-9,
            "diagonal-damped solve should recover the consistent graph, got {}",
            r_diag.final_cost
        );
        assert!(
            (r_id.final_cost - r_diag.final_cost).abs() < 1e-6,
            "identity {} vs diagonal {} should reach the same optimum",
            r_id.final_cost,
            r_diag.final_cost
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn project_to_psd_is_identity_on_a_valid_information_matrix() {
        // A diagonal PD matrix and a non-trivially-correlated PD matrix must
        // both pass through untouched.
        let mut omega = Matrix6::identity();
        omega[(0, 0)] = 25.0;
        omega[(3, 3)] = 4.0;
        let projected = project_to_psd(omega);
        assert!((projected - omega).norm() < 1e-12);

        // Symmetric PD with off-diagonal coupling (small enough to stay PD).
        let mut coupled = Matrix6::identity() * 10.0;
        coupled[(0, 1)] = 1.0;
        coupled[(1, 0)] = 1.0;
        let projected = project_to_psd(coupled);
        assert!((projected - coupled).norm() < 1e-12);
    }

    #[test]
    fn project_to_psd_clamps_an_indefinite_information_matrix() {
        // A rotation-style sub-block with off-diagonal mass dwarfing the
        // diagonal, mirroring the pathological `cubicle` edges.
        let mut omega = Matrix6::identity() * 10.0;
        omega[(2, 3)] = 84_022.3;
        omega[(3, 2)] = 84_022.3;
        omega[(2, 4)] = 132_748.0;
        omega[(4, 2)] = 132_748.0;
        assert!(
            omega.symmetric_eigen().eigenvalues.min() < 0.0,
            "test fixture must start indefinite"
        );

        let projected = project_to_psd(omega);
        let min_eig = projected.symmetric_eigen().eigenvalues.min();
        assert!(
            min_eig >= -1e-9,
            "projected matrix must be PSD, min eig {min_eig}"
        );
        // The projection is symmetric.
        assert!((projected - projected.transpose()).norm() < 1e-9);
    }
}
