//! Fusing several odometries with map fixes in a sliding-window pose graph.
//!
//! [`OdomCorrection`](crate::OdomCorrection) keeps one rigid transform from
//! one odometry's frame to the map, the mean over the last few seconds of
//! fixes. [`MapFusion`] instead estimates the body's map pose at a chain of
//! nodes (one at each fix, and one every [`MapFusionConfig::node_interval_s`]
//! between fixes), constrained by
//!
//! - the relative motion of each odometry source between consecutive nodes
//!   (a VIO, a robot's wheel odometry, a legged robot's kinematic odometry,
//!   ...), each with its own drift model, and
//! - the map fixes, with a Huber loss.
//!
//! Like `OdomCorrection`, the graph is four degrees of freedom: position and
//! yaw. Every source is assumed gravity-aligned (z up), and roll and pitch of
//! the output are taken from the first source that has a pose. Positions are
//! three-dimensional, so a source that climbs (stairs, ramps) is handled; a
//! wheel odometry, which stays in its plane, can be given a loose vertical
//! sigma instead ([`OdometryNoise::sigma_z_per_m`]).
//!
//! Where one source is missing (a VIO that lost tracking, a wheel odometry
//! message gap), the others carry the chain. Where all are, the pose has
//! nothing to follow: the fusion forgets its chain and starts again at the
//! next fix (only a fix's own node can be joined to the chain by the loose
//! "standing still" gap edge).
//!
//! Nodes older than [`MapFusionConfig::window_s`] are marginalized: the chain
//! is block-tridiagonal, so dropping the oldest node is one Schur complement
//! into a linear prior on the next one.
//!
//! With [`MapFusionConfig::source_check_window_s`] set, each source is also
//! checked against the fixes: at every fix, the motion a source reports since
//! the oldest fix of that window is compared with the motion between the two
//! fixes. A source that disagrees (a VIO dragged off by people walking in
//! front of its camera, while the wheels and the fixes agree) has its edges
//! over that stretch left out of the graph, and the pose is carried by the
//! sources that agree; it comes back where it agrees again. Off by default.
//!
//! The fusion does not gate fixes. Callers feed it the fixes that an
//! `OdomCorrection` running beside it accepted (or initialized), so the two
//! see the same fixes and differ only in how they combine them.

use std::collections::VecDeque;

use nalgebra::{Matrix4, UnitQuaternion, Vector3, Vector4};
use visloc_core::geometry::SE3;

use crate::odom_correction::{wrap_angle, yaw_of};

/// Drift model of one odometry source, per edge between two nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdometryNoise {
    /// Position sigma (m) of every edge, however short.
    pub sigma_m: f64,
    /// Horizontal position sigma per metre travelled along the edge.
    pub sigma_per_m: f64,
    /// Vertical position sigma per metre travelled along the edge.
    pub sigma_z_per_m: f64,
    /// Yaw sigma (deg) of every edge.
    pub sigma_yaw_deg: f64,
    /// Yaw sigma per radian turned along the edge (as a fraction).
    pub sigma_yaw_per_rad: f64,
    /// Yaw sigma (deg) per metre travelled along the edge.
    pub sigma_yaw_deg_per_m: f64,
}

impl OdometryNoise {
    /// A visual-inertial odometry: about 1-2% of distance travelled.
    pub const VIO: Self = Self {
        sigma_m: 0.002,
        sigma_per_m: 0.02,
        sigma_z_per_m: 0.02,
        sigma_yaw_deg: 0.05,
        sigma_yaw_per_rad: 0.01,
        sigma_yaw_deg_per_m: 0.3,
    };

    /// A calibrated wheel odometry: good distance, a heading that slips in
    /// turns, nothing to say about height.
    pub const WHEEL: Self = Self {
        sigma_m: 0.002,
        sigma_per_m: 0.02,
        sigma_z_per_m: 0.2,
        sigma_yaw_deg: 0.05,
        sigma_yaw_per_rad: 0.05,
        sigma_yaw_deg_per_m: 0.5,
    };

    /// A legged robot's kinematic odometry (leg kinematics + contact +
    /// IMU attitude): a few percent of distance, including height.
    pub const LEGGED: Self = Self {
        sigma_m: 0.005,
        sigma_per_m: 0.05,
        sigma_z_per_m: 0.05,
        sigma_yaw_deg: 0.1,
        sigma_yaw_per_rad: 0.02,
        sigma_yaw_deg_per_m: 0.5,
    };

    fn information(&self, delta: &Vector3<f64>, delta_yaw: f64) -> Vector4<f64> {
        let distance = delta.xy().norm().max(delta.z.abs());
        let horizontal = self.sigma_m.hypot(self.sigma_per_m * distance);
        let vertical = self.sigma_m.hypot(self.sigma_z_per_m * distance);
        let yaw = (self.sigma_yaw_deg.to_radians())
            .hypot(self.sigma_yaw_per_rad * delta_yaw.abs())
            .hypot(self.sigma_yaw_deg_per_m.to_radians() * distance);
        Vector4::new(
            horizontal.powi(-2),
            horizontal.powi(-2),
            vertical.powi(-2),
            yaw.powi(-2),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapFusionConfig {
    /// One drift model per odometry source, in the order of the poses passed
    /// to [`MapFusion::observe`].
    pub sources: Vec<OdometryNoise>,
    /// A node is added at least this often (s) between fixes.
    pub node_interval_s: f64,
    /// Nodes older than this (s) relative to the newest are marginalized.
    pub window_s: f64,
    /// Position sigma (m) of a fix.
    pub fix_sigma_m: f64,
    /// Yaw sigma (deg) of a fix.
    pub fix_sigma_yaw_deg: f64,
    /// Huber threshold on a fix's whitened residual norm.
    pub fix_huber: f64,
    /// Sigma of the edge added where no source spans two nodes: position
    /// (m) and yaw (deg).
    pub gap_sigma_m: f64,
    pub gap_sigma_yaw_deg: f64,
    /// Gauss-Newton iterations per update.
    pub iterations: usize,
    /// When positive, the source check described in the module
    /// documentation looks back this far (s) at every fix. 0 turns it off.
    pub source_check_window_s: f64,
    /// A source disagrees with the fixes past this position error (m) ...
    pub source_check_tolerance_m: f64,
    /// ... plus this share of the distance between the two fixes ...
    pub source_check_tolerance_per_m: f64,
    /// ... or past this yaw error (deg).
    pub source_check_tolerance_yaw_deg: f64,
}

impl Default for MapFusionConfig {
    fn default() -> Self {
        Self {
            sources: vec![OdometryNoise::VIO],
            node_interval_s: 0.25,
            window_s: 10.0,
            fix_sigma_m: 0.03,
            fix_sigma_yaw_deg: 1.0,
            fix_huber: 2.0,
            gap_sigma_m: 1.0,
            gap_sigma_yaw_deg: 30.0,
            iterations: 3,
            source_check_window_s: 0.0,
            source_check_tolerance_m: 0.05,
            source_check_tolerance_per_m: 0.1,
            source_check_tolerance_yaw_deg: 3.0,
        }
    }
}

/// Position and yaw of the body in the map.
type State = Vector4<f64>;

#[derive(Debug, Clone)]
struct Node {
    timestamp_ns: i64,
    state: State,
    /// Each source's pose of the body at this node, where it had one.
    odometry: Vec<Option<SE3>>,
    /// The fix at this node: map position and yaw.
    fix: Option<State>,
    /// Per source: its edge from the previous node to this one is left out
    /// (the source disagreed with the fixes over it, or at the last check
    /// before this node was added).
    excluded: Vec<bool>,
}

/// Linear prior left on the oldest node by marginalization:
/// `0.5 * dx^T h dx + g^T dx`, `dx = x - linearization`.
#[derive(Debug, Clone)]
struct Prior {
    linearization: State,
    h: Matrix4<f64>,
    g: State,
}

fn state_difference(a: &State, b: &State) -> State {
    let mut d = a - b;
    d[3] = wrap_angle(d[3]);
    d
}

fn rz(yaw: f64) -> nalgebra::Matrix3<f64> {
    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw)
        .to_rotation_matrix()
        .into_inner()
}

/// A relative-motion edge between consecutive nodes `i` and `i + 1`:
/// the motion in node `i`'s yaw frame and the yaw change, with diagonal
/// information.
#[derive(Debug, Clone, Copy)]
struct Edge {
    delta: Vector3<f64>,
    delta_yaw: f64,
    information: Vector4<f64>,
}

/// Odometry fused with map fixes; see the module documentation.
#[derive(Debug, Clone)]
pub struct MapFusion {
    config: MapFusionConfig,
    nodes: VecDeque<Node>,
    prior: Option<Prior>,
    /// Per source: whether the newest check found it disagreeing.
    suspect: Vec<bool>,
    /// Per source: edges left out so far.
    excluded_edges: Vec<u64>,
}

impl MapFusion {
    pub fn new(config: MapFusionConfig) -> Self {
        let sources = config.sources.len();
        Self {
            config,
            nodes: VecDeque::new(),
            prior: None,
            suspect: vec![false; sources],
            excluded_edges: vec![0; sources],
        }
    }

    /// Per source: the number of edges the source check has left out.
    pub fn excluded_edges(&self) -> &[u64] {
        &self.excluded_edges
    }

    /// Per source: whether the newest check found it disagreeing with the
    /// fixes (its motion is then not used for the current pose).
    pub fn suspect_sources(&self) -> &[bool] {
        &self.suspect
    }

    pub const fn config(&self) -> &MapFusionConfig {
        &self.config
    }

    /// Whether the fusion has a map pose, i.e. has seen a fix.
    pub fn is_set(&self) -> bool {
        !self.nodes.is_empty()
    }

    /// Number of nodes in the window.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// One frame's poses, one per source (`None` where a source has no pose
    /// at this stamp). Adds a node once `node_interval_s` has passed since
    /// the newest.
    pub fn observe(&mut self, timestamp_ns: i64, odometry: &[Option<SE3>]) {
        let Some(last) = self.nodes.back() else {
            return;
        };
        if (timestamp_ns - last.timestamp_ns) as f64 * 1e-9 >= self.config.node_interval_s {
            self.push(timestamp_ns, odometry, None);
        }
    }

    /// A map fix of the body at `timestamp_ns`, with each source's pose at
    /// the same instant.
    pub fn add_fix(&mut self, timestamp_ns: i64, map_from_body: &SE3, odometry: &[Option<SE3>]) {
        let fix = Vector4::new(
            map_from_body.translation.x,
            map_from_body.translation.y,
            map_from_body.translation.z,
            yaw_of(&map_from_body.rotation),
        );
        if self
            .nodes
            .back()
            .is_some_and(|node| node.timestamp_ns == timestamp_ns)
        {
            let node = self.nodes.back_mut().expect("checked");
            node.fix = Some(fix);
            for (stored, pose) in node.odometry.iter_mut().zip(odometry) {
                if stored.is_none() {
                    stored.clone_from(pose);
                }
            }
            self.check_sources();
            self.optimize();
            return;
        }
        self.push(timestamp_ns, odometry, Some(fix));
    }

    /// The source check (see the module documentation), at the newest node,
    /// which holds a fix.
    fn check_sources(&mut self) {
        if self.config.source_check_window_s <= 0.0 || self.nodes.len() < 2 {
            return;
        }
        let newest = self.nodes.len() - 1;
        let Some(fix_b) = self.nodes[newest].fix else {
            return;
        };
        let window_ns = (self.config.source_check_window_s * 1e9) as i64;
        let start_ns = self.nodes[newest].timestamp_ns - window_ns;
        let sources = self.config.sources.len();
        let mut disagree = vec![None::<usize>; sources];
        for (source, verdict) in disagree.iter_mut().enumerate() {
            let Some(pose_b) = self.nodes[newest].odometry[source].clone() else {
                continue;
            };
            // The oldest node of the window with a fix and this source's pose.
            let Some(a) = (0..newest).find(|&i| {
                self.nodes[i].timestamp_ns >= start_ns
                    && self.nodes[i].fix.is_some()
                    && self.nodes[i].odometry[source].is_some()
            }) else {
                continue;
            };
            let fix_a = self.nodes[a].fix.expect("checked");
            let pose_a = self.nodes[a].odometry[source].as_ref().expect("checked");
            let fix_delta = rz(fix_a[3]).transpose() * (fix_b.xyz() - fix_a.xyz());
            let fix_turn = wrap_angle(fix_b[3] - fix_a[3]);
            let yaw_a = yaw_of(&pose_a.rotation);
            let source_delta = rz(yaw_a).transpose() * (pose_b.translation - pose_a.translation);
            let source_turn = wrap_angle(yaw_of(&pose_b.rotation) - yaw_a);
            let tolerance = self.config.source_check_tolerance_m
                + self.config.source_check_tolerance_per_m * fix_delta.norm();
            let off = (source_delta - fix_delta).norm() > tolerance
                || wrap_angle(source_turn - fix_turn).abs().to_degrees()
                    > self.config.source_check_tolerance_yaw_deg;
            if off {
                *verdict = Some(a);
            }
        }
        // Every source disagreeing says more about the fixes than about the
        // sources: leave them all in.
        let checked = (0..sources)
            .filter(|&s| self.nodes[newest].odometry[s].is_some())
            .count();
        let disagreeing = disagree.iter().filter(|d| d.is_some()).count();
        let drop = disagreeing > 0 && disagreeing < checked;
        for (source, from) in disagree.iter().enumerate() {
            self.suspect[source] = drop && from.is_some();
            if !drop {
                continue;
            }
            if let Some(a) = *from {
                for i in a + 1..=newest {
                    if !self.nodes[i].excluded[source] {
                        self.nodes[i].excluded[source] = true;
                        self.excluded_edges[source] += 1;
                    }
                }
            }
        }
    }

    /// Forgets everything (e.g. after the caller's fix gate reset).
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.prior = None;
        self.suspect.iter_mut().for_each(|s| *s = false);
    }

    /// The body's map pose now, from the newest node and the motion of the
    /// first source that has a pose both there and in `odometry`, skipping
    /// sources the source check found disagreeing unless none other has one.
    pub fn map_from_body(&self, odometry: &[Option<SE3>]) -> Option<SE3> {
        let node = self.nodes.back()?;
        let pairs =
            || {
                node.odometry.iter().zip(odometry).enumerate().filter_map(
                    |(source, (then, now))| Some((source, then.as_ref()?, now.as_ref()?)),
                )
            };
        let (_, then, now) = pairs()
            .find(|(source, _, _)| !self.suspect.get(*source).copied().unwrap_or(false))
            .or_else(|| pairs().next())?;
        // map <- odom of this source, as the node's state puts it.
        let yaw = node.state[3] - yaw_of(&then.rotation);
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
        let translation = node.state.xyz() - rotation * then.translation;
        Some(SE3::new(rotation, translation).compose(now))
    }

    fn push(&mut self, timestamp_ns: i64, odometry: &[Option<SE3>], fix: Option<State>) {
        let mut odometry = odometry.to_vec();
        odometry.resize(self.config.sources.len(), None);
        let state = match (self.nodes.back(), fix) {
            (None, Some(fix)) => fix,
            (None, None) => return,
            (Some(_), _) => {
                // Predict from the chain so Gauss-Newton starts close.
                let predicted = self.map_from_body(&odometry);
                match (predicted, fix) {
                    (Some(pose), _) => Vector4::new(
                        pose.translation.x,
                        pose.translation.y,
                        pose.translation.z,
                        yaw_of(&pose.rotation),
                    ),
                    (None, Some(fix)) => fix,
                    // No source spans the newest node and now: nothing ties
                    // the pose to the chain any more. Start over at the
                    // next fix rather than guess.
                    (None, None) => {
                        self.reset();
                        return;
                    }
                }
            }
        };
        // A source found disagreeing at the last check stays out until a
        // check finds it agreeing again.
        let excluded = if self.nodes.is_empty() {
            vec![false; self.config.sources.len()]
        } else {
            self.suspect.clone()
        };
        for (count, &out) in self.excluded_edges.iter_mut().zip(&excluded) {
            *count += u64::from(out);
        }
        self.nodes.push_back(Node {
            timestamp_ns,
            state,
            odometry,
            fix,
            excluded,
        });
        if fix.is_some() {
            self.check_sources();
        }
        while self.nodes.len() > 2
            && (timestamp_ns - self.nodes[0].timestamp_ns) as f64 * 1e-9 > self.config.window_s
        {
            self.marginalize_oldest();
        }
        self.optimize();
    }

    /// The edges between consecutive nodes `i` and `i + 1`: one per source
    /// with a pose at both, or the loose gap edge where none has.
    fn edges(&self, i: usize) -> Vec<Edge> {
        let (a, b) = (&self.nodes[i], &self.nodes[i + 1]);
        let mut edges = Vec::new();
        for (source, noise) in self.config.sources.iter().enumerate() {
            let (Some(pa), Some(pb)) = (&a.odometry[source], &b.odometry[source]) else {
                continue;
            };
            if b.excluded[source] {
                continue;
            }
            let yaw_a = yaw_of(&pa.rotation);
            let delta = rz(yaw_a).transpose() * (pb.translation - pa.translation);
            let delta_yaw = wrap_angle(yaw_of(&pb.rotation) - yaw_a);
            edges.push(Edge {
                delta,
                delta_yaw,
                information: noise.information(&delta, delta_yaw),
            });
        }
        if edges.is_empty() {
            let p = self.config.gap_sigma_m.powi(-2);
            edges.push(Edge {
                delta: Vector3::zeros(),
                delta_yaw: 0.0,
                information: Vector4::new(
                    p,
                    p,
                    p,
                    self.config.gap_sigma_yaw_deg.to_radians().powi(-2),
                ),
            });
        }
        edges
    }

    /// Residual of `edge` and its Jacobians with respect to nodes `i` and
    /// `i + 1`.
    fn edge_linearization(
        xa: &State,
        xb: &State,
        edge: &Edge,
    ) -> (State, Matrix4<f64>, Matrix4<f64>) {
        let rt = rz(xa[3]).transpose();
        let dp = xb.xyz() - xa.xyz();
        let u = rt * dp;
        let r_p = u - edge.delta;
        let r_yaw = wrap_angle(xb[3] - xa[3] - edge.delta_yaw);
        let residual = Vector4::new(r_p.x, r_p.y, r_p.z, r_yaw);
        let mut ja = Matrix4::zeros();
        let mut jb = Matrix4::zeros();
        ja.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-rt));
        jb.fixed_view_mut::<3, 3>(0, 0).copy_from(&rt);
        // d(Rz(yaw)^T dp)/d yaw = (u.y, -u.x, 0).
        ja[(0, 3)] = u.y;
        ja[(1, 3)] = -u.x;
        ja[(3, 3)] = -1.0;
        jb[(3, 3)] = 1.0;
        (residual, ja, jb)
    }

    fn fix_information(&self, node: &Node) -> Option<(State, State)> {
        let fix = node.fix?;
        let residual = state_difference(&node.state, &fix);
        let information = Vector4::new(
            self.config.fix_sigma_m.powi(-2),
            self.config.fix_sigma_m.powi(-2),
            self.config.fix_sigma_m.powi(-2),
            self.config.fix_sigma_yaw_deg.to_radians().powi(-2),
        );
        let whitened = residual.component_mul(&information.map(f64::sqrt)).norm();
        let weight = if whitened > self.config.fix_huber {
            self.config.fix_huber / whitened
        } else {
            1.0
        };
        Some((residual, information * weight))
    }

    /// Normal equations of the chain: diagonal blocks, the blocks between
    /// `i` and `i + 1`, and the gradient.
    fn normal_equations(&self) -> (Vec<Matrix4<f64>>, Vec<Matrix4<f64>>, Vec<State>) {
        let n = self.nodes.len();
        let mut diagonal = vec![Matrix4::zeros(); n];
        let mut off = vec![Matrix4::zeros(); n.saturating_sub(1)];
        let mut gradient = vec![State::zeros(); n];
        if let Some(prior) = &self.prior {
            let dx = state_difference(&self.nodes[0].state, &prior.linearization);
            diagonal[0] += prior.h;
            gradient[0] += prior.h * dx + prior.g;
        }
        for (i, node) in self.nodes.iter().enumerate() {
            if let Some((residual, information)) = self.fix_information(node) {
                diagonal[i] += Matrix4::from_diagonal(&information);
                gradient[i] += residual.component_mul(&information);
            }
        }
        for i in 0..n.saturating_sub(1) {
            let (xa, xb) = (&self.nodes[i].state, &self.nodes[i + 1].state);
            for edge in self.edges(i) {
                let (residual, ja, jb) = Self::edge_linearization(xa, xb, &edge);
                let w = Matrix4::from_diagonal(&edge.information);
                diagonal[i] += ja.transpose() * w * ja;
                diagonal[i + 1] += jb.transpose() * w * jb;
                off[i] += ja.transpose() * w * jb;
                gradient[i] += ja.transpose() * w * residual;
                gradient[i + 1] += jb.transpose() * w * residual;
            }
        }
        (diagonal, off, gradient)
    }

    fn optimize(&mut self) {
        for _ in 0..self.config.iterations {
            let (diagonal, off, gradient) = self.normal_equations();
            let Some(step) = solve_block_tridiagonal(&diagonal, &off, &gradient) else {
                return;
            };
            let mut largest: f64 = 0.0;
            for (node, step) in self.nodes.iter_mut().zip(&step) {
                node.state -= step;
                node.state[3] = wrap_angle(node.state[3]);
                largest = largest.max(step.amax());
            }
            if largest < 1e-6 {
                break;
            }
        }
    }

    /// Schur-complements the oldest node into a prior on the next one, at
    /// the current estimate.
    fn marginalize_oldest(&mut self) {
        let (diagonal, off, gradient) = self.normal_equations();
        let Some(inverse) = diagonal[0].try_inverse() else {
            self.nodes.pop_front();
            self.prior = None;
            return;
        };
        // Only the terms that involve node 0 move to the prior: node 1's own
        // blocks from its other factors stay where they are. Rebuild node 1's
        // share of the blocks from node 0's factors only.
        let mut h01 = off[0];
        let mut d1 = Matrix4::zeros();
        let mut g1 = State::zeros();
        let (xa, xb) = (&self.nodes[0].state, &self.nodes[1].state);
        h01.fill(0.0);
        for edge in self.edges(0) {
            let (residual, ja, jb) = Self::edge_linearization(xa, xb, &edge);
            let w = Matrix4::from_diagonal(&edge.information);
            d1 += jb.transpose() * w * jb;
            h01 += ja.transpose() * w * jb;
            g1 += jb.transpose() * w * residual;
        }
        let h = d1 - h01.transpose() * inverse * h01;
        let g = g1 - h01.transpose() * inverse * gradient[0];
        let linearization = self.nodes[1].state;
        self.nodes.pop_front();
        self.prior = Some(Prior {
            linearization,
            h: 0.5 * (h + h.transpose()),
            g,
        });
    }
}

/// Solves `H x = g` for block-tridiagonal symmetric `H` (diagonal blocks
/// `diagonal`, blocks `off[i]` between `i` and `i + 1`).
fn solve_block_tridiagonal(
    diagonal: &[Matrix4<f64>],
    off: &[Matrix4<f64>],
    gradient: &[State],
) -> Option<Vec<State>> {
    let n = diagonal.len();
    if n == 0 {
        return Some(Vec::new());
    }
    // Forward elimination: D'_i = D_i - B_{i-1}^T D'_{i-1}^{-1} B_{i-1}.
    let mut d = diagonal.to_vec();
    let mut g = gradient.to_vec();
    let mut inverses = Vec::with_capacity(n);
    for i in 0..n {
        if i > 0 {
            let inv_prev: &Matrix4<f64> = &inverses[i - 1];
            let b = off[i - 1];
            d[i] -= b.transpose() * inv_prev * b;
            let correction = b.transpose() * inv_prev * g[i - 1];
            g[i] -= correction;
        }
        inverses.push(d[i].try_inverse()?);
    }
    let mut x = vec![State::zeros(); n];
    x[n - 1] = inverses[n - 1] * g[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = inverses[i] * (g[i] - off[i] * x[i + 1]);
    }
    Some(x)
}

/// Whether fix `b` lies where fix `a` and one odometry's motion between
/// them put it, within `tolerance_m` plus `tolerance_per_m` of that motion's
/// length, and `tolerance_deg`. `a_sources` and `b_sources` are the
/// odometries' poses of the body at the two fixes, as [`MapFusion`] takes
/// them; a source missing at either is skipped.
pub fn fixes_agree_under_a_source(
    a: &SE3,
    a_sources: &[Option<SE3>],
    b: &SE3,
    b_sources: &[Option<SE3>],
    tolerance_m: f64,
    tolerance_per_m: f64,
    tolerance_deg: f64,
) -> bool {
    a_sources.iter().zip(b_sources).any(|(odom_a, odom_b)| {
        let (Some(odom_a), Some(odom_b)) = (odom_a, odom_b) else {
            return false;
        };
        let motion = odom_a.inverse().compose(odom_b);
        let predicted = a.compose(&motion);
        (predicted.translation - b.translation).norm()
            <= tolerance_m + tolerance_per_m * motion.translation.norm()
            && predicted.rotation.angle_to(&b.rotation) <= tolerance_deg.to_radians()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(x: f64, y: f64, z: f64, yaw: f64) -> SE3 {
        SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw),
            Vector3::new(x, y, z),
        )
    }

    fn map_from_odom() -> SE3 {
        pose(1.0, -2.0, 0.5, 0.7)
    }

    /// A body driving a circle of radius 2 m at 1 rad/s, sampled at 20 Hz.
    fn truth(index: usize) -> SE3 {
        let t = index as f64 * 0.05;
        pose(
            2.0 * t.cos(),
            2.0 * t.sin(),
            0.1 * t,
            t + std::f64::consts::FRAC_PI_2,
        )
    }

    fn assert_close(a: &SE3, b: &SE3, tolerance: f64) {
        assert!(
            (a.translation - b.translation).norm() < tolerance,
            "{a:?} vs {b:?}"
        );
        assert!(
            a.rotation.angle_to(&b.rotation) < tolerance,
            "{a:?} vs {b:?}"
        );
    }

    #[test]
    fn exact_odometry_and_fixes_follow_the_truth() {
        let mut fusion = MapFusion::new(MapFusionConfig::default());
        let odom = map_from_odom().inverse();
        for index in 0..400 {
            let poses = [Some(odom.compose(&truth(index)))];
            fusion.observe(index as i64 * 50_000_000, &poses);
            if index % 10 == 0 {
                fusion.add_fix(index as i64 * 50_000_000, &truth(index), &poses);
            }
            if fusion.is_set() {
                let estimate = fusion.map_from_body(&poses).expect("set");
                assert_close(&estimate, &truth(index), 1e-6);
            }
        }
        // Marginalization keeps the window bounded.
        assert!(fusion.node_count() < 120, "{}", fusion.node_count());
    }

    #[test]
    fn a_second_source_carries_the_chain_where_the_first_is_missing() {
        let mut fusion = MapFusion::new(MapFusionConfig {
            sources: vec![OdometryNoise::VIO, OdometryNoise::WHEEL],
            ..MapFusionConfig::default()
        });
        let vio_frame = map_from_odom().inverse();
        let wheel_frame = pose(-3.0, 4.0, 0.0, -1.2);
        for index in 0..200 {
            let vio = (!(60..120).contains(&index)).then(|| vio_frame.compose(&truth(index)));
            // The VIO restarts in a new frame after its gap.
            let vio = vio.map(|pose| {
                if index >= 120 {
                    self::pose(0.3, 0.2, 0.0, 0.4).compose(&pose)
                } else {
                    pose
                }
            });
            let poses = [vio, Some(wheel_frame.compose(&truth(index)))];
            fusion.observe(index as i64 * 50_000_000, &poses);
            if index < 20 && index % 10 == 0 {
                fusion.add_fix(index as i64 * 50_000_000, &truth(index), &poses);
            }
            if fusion.is_set() {
                let estimate = fusion.map_from_body(&poses).expect("set");
                assert_close(&estimate, &truth(index), 1e-3);
            }
        }
    }

    #[test]
    fn fixes_pull_a_drifting_odometry_back() {
        let mut fusion = MapFusion::new(MapFusionConfig::default());
        // 2% scale error and a yaw drift of 0.6 deg/s.
        let mut odom = SE3::identity();
        let mut previous = truth(0);
        let mut worst: f64 = 0.0;
        for index in 0..400 {
            let current = truth(index);
            let mut step = previous.inverse().compose(&current);
            step.translation *= 1.02;
            step.rotation =
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0005) * step.rotation;
            odom = odom.compose(&step);
            previous = current.clone();
            let poses = [Some(odom.clone())];
            fusion.observe(index as i64 * 50_000_000, &poses);
            if index % 10 == 0 {
                fusion.add_fix(index as i64 * 50_000_000, &current, &poses);
            }
            if index > 20 {
                let estimate = fusion.map_from_body(&poses).expect("set");
                worst = worst.max((estimate.translation - current.translation).norm());
            }
        }
        // Between fixes 0.5 s apart the odometry drifts about 2 cm.
        assert!(worst < 0.05, "worst {worst}");
    }

    /// A VIO that runs off (from 5 s on it reports three times the true
    /// motion, as when people walking along in front of its camera drag it),
    /// fused with exact wheels and exact fixes every 0.5 s. Returns the worst
    /// map pose error from the first fix after the VIO runs off (the check
    /// runs at fixes, so until then nothing tells the two sources apart),
    /// and the fusion.
    fn runaway_vio(check_window_s: f64) -> (f64, MapFusion) {
        let mut fusion = MapFusion::new(MapFusionConfig {
            sources: vec![OdometryNoise::VIO, OdometryNoise::WHEEL],
            source_check_window_s: check_window_s,
            ..MapFusionConfig::default()
        });
        let wheel_frame = pose(-3.0, 4.0, 0.0, -1.2);
        let mut vio = map_from_odom().inverse().compose(&truth(0));
        let mut worst: f64 = 0.0;
        for index in 0..300 {
            if index > 0 {
                let mut step = truth(index - 1).inverse().compose(&truth(index));
                if index >= 100 {
                    step.translation *= 3.0;
                }
                vio = vio.compose(&step);
            }
            let poses = [Some(vio.clone()), Some(wheel_frame.compose(&truth(index)))];
            fusion.observe(index as i64 * 50_000_000, &poses);
            if index % 10 == 0 {
                fusion.add_fix(index as i64 * 50_000_000, &truth(index), &poses);
            }
            if index >= 110 {
                let estimate = fusion.map_from_body(&poses).expect("set");
                worst = worst.max((estimate.translation - truth(index).translation).norm());
            }
        }
        (worst, fusion)
    }

    #[test]
    fn the_source_check_drops_a_runaway_vio_that_the_wheels_and_fixes_contradict() {
        let (without, plain) = runaway_vio(0.0);
        let (with, checked) = runaway_vio(1.0);
        assert!(without > 0.1, "without the check: {without}");
        assert!(with < 0.05, "with the check: {with}");
        assert_eq!(plain.excluded_edges(), &[0, 0]);
        assert!(checked.excluded_edges()[0] > 0);
        assert_eq!(checked.excluded_edges()[1], 0);
        assert_eq!(checked.suspect_sources(), &[true, false]);
    }

    #[test]
    fn the_source_check_keeps_sound_sources_and_a_lone_one() {
        // Exact sources: nothing is left out.
        let mut fusion = MapFusion::new(MapFusionConfig {
            sources: vec![OdometryNoise::VIO, OdometryNoise::WHEEL],
            source_check_window_s: 1.0,
            ..MapFusionConfig::default()
        });
        let wheel_frame = pose(-3.0, 4.0, 0.0, -1.2);
        let odom = map_from_odom().inverse();
        for index in 0..200 {
            let poses = [
                Some(odom.compose(&truth(index))),
                Some(wheel_frame.compose(&truth(index))),
            ];
            fusion.observe(index as i64 * 50_000_000, &poses);
            if index % 10 == 0 {
                fusion.add_fix(index as i64 * 50_000_000, &truth(index), &poses);
            }
        }
        assert_eq!(fusion.excluded_edges(), &[0, 0]);
        // One source that disagrees is still the only one: kept.
        let mut lone = MapFusion::new(MapFusionConfig {
            source_check_window_s: 1.0,
            ..MapFusionConfig::default()
        });
        for index in 0..60 {
            let mut odom_pose = odom.compose(&truth(index));
            odom_pose.translation *= 3.0;
            let poses = [Some(odom_pose)];
            lone.observe(index as i64 * 50_000_000, &poses);
            if index % 10 == 0 {
                lone.add_fix(index as i64 * 50_000_000, &truth(index), &poses);
            }
        }
        assert_eq!(lone.excluded_edges(), &[0]);
    }

    #[test]
    fn block_tridiagonal_solve_matches_dense() {
        let n = 5;
        let mut diagonal = Vec::new();
        let mut off = Vec::new();
        let mut gradient = Vec::new();
        for i in 0..n {
            let mut d = Matrix4::identity() * (4.0 + i as f64);
            d[(0, 1)] = 0.3;
            d[(1, 0)] = 0.3;
            diagonal.push(d);
            gradient.push(Vector4::new(i as f64, 1.0, -2.0, 0.5));
            if i + 1 < n {
                off.push(Matrix4::from_fn(|r, c| 0.1 * (r as f64 - c as f64 + 1.0)));
            }
        }
        let x = solve_block_tridiagonal(&diagonal, &off, &gradient).expect("solves");
        let mut dense = nalgebra::DMatrix::zeros(4 * n, 4 * n);
        let mut rhs = nalgebra::DVector::zeros(4 * n);
        for i in 0..n {
            dense
                .view_mut((4 * i, 4 * i), (4, 4))
                .copy_from(&diagonal[i]);
            rhs.rows_mut(4 * i, 4).copy_from(&gradient[i]);
            if i + 1 < n {
                dense
                    .view_mut((4 * i, 4 * i + 4), (4, 4))
                    .copy_from(&off[i]);
                dense
                    .view_mut((4 * i + 4, 4 * i), (4, 4))
                    .copy_from(&off[i].transpose());
            }
        }
        let expected = dense.lu().solve(&rhs).expect("solves");
        for (i, xi) in x.iter().enumerate() {
            assert!((xi - expected.rows(4 * i, 4)).norm() < 1e-9);
        }
    }

    #[test]
    fn fixes_agree_when_one_source_moved_as_they_did() {
        let a = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.4),
            Vector3::new(1.0, 2.0, 0.0),
        );
        let step = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.1),
            Vector3::new(0.5, 0.0, 0.0),
        );
        let b = a.compose(&step);
        let good = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, -1.0),
            Vector3::new(5.0, 0.0, 0.0),
        );
        let runaway = good.compose(&SE3::new(step.rotation, step.translation * 3.0));
        let a_sources = [Some(good.clone()), Some(good.clone())];
        // The second source moved as the fixes did; the first ran off.
        let b_sources = [Some(runaway.clone()), Some(good.compose(&step))];
        assert!(fixes_agree_under_a_source(
            &a, &a_sources, &b, &b_sources, 0.05, 0.1, 3.0
        ));
        let b_sources = [Some(runaway), None];
        assert!(!fixes_agree_under_a_source(
            &a, &a_sources, &b, &b_sources, 0.05, 0.1, 3.0
        ));
    }
}
