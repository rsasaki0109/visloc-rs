//! GNSS-aided pose graph: visual-odometry relative-pose factors fused with
//! GNSS position factors and a jointly-estimated GNSS-ENU-to-map alignment.
//!
//! This is the optimization back end behind [`crate::gnss_fusion`]. It
//! is a *loosely-coupled, position-domain* estimator: each GNSS input is an
//! already-solved antenna position (with a 3×3 covariance) in a local
//! East-North-Up (ENU) frame, not raw pseudoranges or carrier phase.
//!
//! # State
//!
//! * One node per camera frame, expressed in the visual-odometry **map** frame:
//!   a camera-to-map rotation `Rᵢ`, camera centre `pᵢ`, and (only in
//!   [`GnssScaleMode::ScaleDrift`]) a local log-scale `σᵢ` that lets a
//!   monocular trajectory bend its scale (a Sim(3)-style node).
//! * One global [`GnssAlignment`] `x_enu = s·R·x_map + t` mapping the map frame
//!   into ENU. `R` is either a pure yaw about the ENU up axis
//!   ([`AlignmentRotationDof::YawOnly`], for gravity-aligned VIO maps whose `+z`
//!   is up) or a full 3-D rotation ([`AlignmentRotationDof::Full`], for
//!   camera-only VO whose map frame is an arbitrary first-camera frame). `s` is
//!   fixed at `1` in [`GnssScaleMode::Metric`] and free otherwise.
//!
//! The gauge is fixed by holding at least one node constant (the oldest frame
//! of the window): the map frame is *defined* by the VO, and the alignment is a
//! genuine unknown estimated jointly with the trajectory.
//!
//! # Factors
//!
//! * [`VoRelativeFactor`] — VO relative motion `Zᵢⱼ = Tᵢ⁻¹ Tⱼ`
//!   (camera-to-map convention). Residual `[e^{-σᵢ} Rᵢᵀ(pⱼ − pᵢ) − t_Z;
//!   Log(R_Zᵀ Rᵢᵀ Rⱼ); σⱼ − σᵢ]`, whitened by the 6×6 information (and a
//!   scalar log-scale information in scale-drift mode).
//! * [`GnssPositionFactor`] — antenna position `s·R·pᵢ + t + R·Rᵢ·l − z`
//!   with the camera-frame lever arm `l` (metres), whitened by the per-fix
//!   3×3 information. A robust kernel or Graduated Non-Convexity (GNC,
//!   [`crate::gnc`]) down-weights multipath jumps.
//! * [`GnssAlignmentPrior`] — optional Gaussian prior on the alignment, used by
//!   a sliding window to remember what earlier (dropped) frames said about it.
//!
//! Every Jacobian is analytic; the unit tests check each against central
//! finite differences.
//!
//! # Solver
//!
//! Levenberg-Marquardt with Marquardt (`diag(H)`) damping on the normal
//! equations, solved by the crate's block-sparse Cholesky
//! (block size 7). Nodes are ordered by id with the alignment block last, so a
//! chain of sequential VO factors plus GNSS factors forms an "arrow" matrix
//! that factors without fill.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DVector, Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, SE3};

use crate::block_cholesky::{self, BlockSymbolic};
use crate::gnc::{self, GncConfig, GncKernel, GncState};
use crate::RobustKernel;

/// Size of every variable block (`[δp; δθ; δσ]` for nodes and `[δt; δφ; δσ]`
/// for the alignment).
const BLOCK: usize = 7;

type Vector7 = SVector<f64, 7>;
type Matrix7 = SMatrix<f64, 7, 7>;
type Matrix3x7 = SMatrix<f64, 3, 7>;
type Matrix6 = SMatrix<f64, 6, 6>;

/// χ² (3 DoF) 99.9 % quantile: the default outlier gate on a GNSS factor's
/// whitened squared residual.
pub const GNSS_CHI2_3DOF_999: f64 = 16.266;

/// Which rotational degrees of freedom the GNSS-ENU-to-map alignment has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlignmentRotationDof {
    /// Yaw about the ENU up axis only. Requires a gravity-aligned map frame
    /// with `+z` up (VIO, or VO with a known gravity direction).
    #[default]
    YawOnly,
    /// Full 3-D rotation. Use for camera-only VO whose map frame is the first
    /// camera frame (gravity direction unknown). Roll about a straight-line
    /// trajectory is unobservable from positions alone.
    Full,
}

/// How metric scale is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GnssScaleMode {
    /// The VO is metric (stereo / VIO): alignment scale fixed to `1`, no
    /// per-node scale.
    #[default]
    Metric,
    /// Monocular VO with a single unknown global scale: the alignment scale is
    /// estimated, nodes keep the VO's internal scale.
    GlobalScale,
    /// Monocular VO with scale drift: the alignment scale *and* a per-node
    /// log-scale (Sim(3)-style nodes, random-walk constrained by the relative
    /// factors) are estimated.
    ScaleDrift,
}

/// Similarity mapping the VO map frame into the local ENU frame:
/// `x_enu = scale · rotation · x_map + translation`.
#[derive(Debug, Clone, PartialEq)]
pub struct GnssAlignment {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
    pub scale: f64,
}

impl Default for GnssAlignment {
    fn default() -> Self {
        Self::identity()
    }
}

impl GnssAlignment {
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
            scale: 1.0,
        }
    }

    /// Yaw-only alignment (`rotation = Rz(yaw)`).
    pub fn from_yaw(yaw: f64, translation: Vector3<f64>, scale: f64) -> Self {
        Self {
            rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw),
            translation,
            scale,
        }
    }

    /// Map a map-frame point into ENU.
    pub fn transform_point(&self, map_point: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * (map_point * self.scale) + self.translation
    }

    /// Map an ENU point back into the map frame.
    pub fn inverse_transform_point(&self, enu_point: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.inverse() * (enu_point - self.translation) / self.scale
    }

    /// Camera-to-ENU transform of a camera-to-map pose: rotation `R·Rᵢ`,
    /// centre `s·R·pᵢ + t` (scale only affects positions).
    pub fn transform_camera_to_map(&self, camera_to_map: &SE3) -> SE3 {
        SE3::new(
            self.rotation * camera_to_map.rotation,
            self.transform_point(&camera_to_map.translation),
        )
    }

    /// Heading of the alignment's map `+x` axis in ENU (radians, from East
    /// toward North). Equals the yaw for a yaw-only alignment.
    pub fn yaw(&self) -> f64 {
        let x = self.rotation * Vector3::x();
        x.y.atan2(x.x)
    }
}

/// One frame of the graph, in the VO map frame.
#[derive(Debug, Clone, PartialEq)]
pub struct GnssGraphNode {
    /// Camera-to-map pose `(Rᵢ, pᵢ)`.
    pub camera_to_map: SE3,
    /// Local log-scale `σᵢ` (only optimized in [`GnssScaleMode::ScaleDrift`]).
    pub log_scale: f64,
}

impl GnssGraphNode {
    pub fn new(camera_to_map: SE3) -> Self {
        Self {
            camera_to_map,
            log_scale: 0.0,
        }
    }

    /// Node from a world-to-camera [`Pose`] (the crate's pose convention).
    pub fn from_pose(pose: &Pose) -> Self {
        Self::new(pose.camera_to_world())
    }
}

/// Visual-odometry relative motion between two nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct VoRelativeFactor {
    pub from: u64,
    pub to: u64,
    /// Measured `T_fromᵀ T_to` in the camera-to-map convention: rotation of
    /// `to` in `from`'s camera frame and `to`'s centre in `from`'s frame (VO
    /// units).
    pub relative: SE3,
    /// 6×6 information over `[translation; rotation]` residuals.
    pub information: Matrix6,
    /// Information of the log-scale random walk `σ_to − σ_from` (used only in
    /// [`GnssScaleMode::ScaleDrift`]).
    pub log_scale_information: f64,
}

impl VoRelativeFactor {
    /// Factor whose measurement is read off two VO poses (world-to-camera).
    pub fn from_vo_poses(
        from: u64,
        to: u64,
        vo_from: &Pose,
        vo_to: &Pose,
        information: Matrix6,
        log_scale_information: f64,
    ) -> Self {
        Self {
            from,
            to,
            relative: vo_from
                .camera_to_world()
                .inverse()
                .compose(&vo_to.camera_to_world()),
            information,
            log_scale_information,
        }
    }
}

/// GNSS antenna position measured in ENU for one node.
#[derive(Debug, Clone, PartialEq)]
pub struct GnssPositionFactor {
    pub node: u64,
    pub position_enu: Vector3<f64>,
    /// 3×3 information (inverse covariance) of `position_enu`.
    pub information: Matrix3<f64>,
}

/// Gaussian prior on the alignment over the tangent `[δt; δφ; δσ]`
/// (translation, left rotation, log-scale).
#[derive(Debug, Clone, PartialEq)]
pub struct GnssAlignmentPrior {
    pub mean: GnssAlignment,
    pub information: Matrix7,
}

impl GnssAlignmentPrior {
    /// Diagonal prior from standard deviations (metres, radians, log-scale).
    pub fn from_sigmas(
        mean: GnssAlignment,
        translation_sigma: f64,
        rotation_sigma: f64,
        log_scale_sigma: f64,
    ) -> Self {
        let mut information = Matrix7::zeros();
        for k in 0..3 {
            information[(k, k)] = 1.0 / (translation_sigma * translation_sigma);
            information[(k + 3, k + 3)] = 1.0 / (rotation_sigma * rotation_sigma);
        }
        information[(6, 6)] = 1.0 / (log_scale_sigma * log_scale_sigma);
        Self { mean, information }
    }
}

/// Robust treatment of the GNSS factors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GnssRobustMode {
    /// Plain least squares (outliers pull the trajectory).
    None,
    /// IRLS with an M-estimator on the whitened squared residual.
    Kernel(RobustKernel),
    /// Graduated Non-Convexity starting from the least-squares solution.
    Gnc(GncConfig),
}

impl Default for GnssRobustMode {
    /// GNC with the truncated-least-squares surrogate and an inlier scale at
    /// the χ²(3) 99.9 % gate (`c ≈ 4.03` whitened sigmas).
    fn default() -> Self {
        Self::Gnc(GncConfig {
            kernel: GncKernel::TruncatedLeastSquares,
            c: GNSS_CHI2_3DOF_999.sqrt(),
            anneal_factor: 2.0,
            max_outer: 40,
            inner_iterations: 3,
            auto_scale: None,
            auto_scale_readapt: false,
        })
    }
}

/// Configuration for [`GnssPoseGraph::optimize`].
#[derive(Debug, Clone, PartialEq)]
pub struct GnssPoseGraphConfig {
    pub rotation_dof: AlignmentRotationDof,
    pub scale_mode: GnssScaleMode,
    pub robust: GnssRobustMode,
    /// Estimate the alignment jointly (`true`) or hold it fixed.
    pub optimize_alignment: bool,
    /// LM iteration cap for each plain solve (and for the final refinement).
    pub max_iterations: usize,
    pub initial_lambda: f64,
    pub lambda_increase_factor: f64,
    pub lambda_decrease_factor: f64,
    pub min_lambda: f64,
    pub max_lambda: f64,
    /// Convergence threshold on the largest block update.
    pub step_tolerance: f64,
    /// Convergence threshold on the relative cost decrease.
    pub relative_cost_tolerance: f64,
    /// Whitened squared residual above which a GNSS factor is classified as
    /// an outlier after the robust solve.
    pub outlier_chi2_threshold: f64,
    /// After classification, re-solve with outliers removed and inliers at
    /// full weight (a hard-rejection refinement).
    pub refine_without_outliers: bool,
}

impl Default for GnssPoseGraphConfig {
    fn default() -> Self {
        Self {
            rotation_dof: AlignmentRotationDof::YawOnly,
            scale_mode: GnssScaleMode::Metric,
            robust: GnssRobustMode::default(),
            optimize_alignment: true,
            max_iterations: 30,
            initial_lambda: 1e-4,
            lambda_increase_factor: 10.0,
            lambda_decrease_factor: 0.3,
            min_lambda: 1e-9,
            max_lambda: 1e10,
            step_tolerance: 1e-7,
            relative_cost_tolerance: 1e-6,
            outlier_chi2_threshold: GNSS_CHI2_3DOF_999,
            refine_without_outliers: true,
        }
    }
}

/// Errors returned by [`GnssPoseGraph::optimize`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GnssPoseGraphError {
    /// No node is held fixed, so the map-frame gauge is free.
    NoFixedNode,
    /// A factor or fixed id references a node that is not in the graph.
    MissingNode(u64),
    /// Every node is fixed and the alignment is not optimized.
    NoVariables,
    /// A factor's information matrix is not symmetric positive definite.
    InvalidInformation { kind: &'static str, index: usize },
    /// The damped normal equations could not be factored.
    SingularSystem,
}

impl std::fmt::Display for GnssPoseGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFixedNode => write!(f, "GNSS pose graph has no fixed (gauge) node"),
            Self::MissingNode(id) => write!(f, "GNSS pose graph is missing node {id}"),
            Self::NoVariables => write!(f, "GNSS pose graph has nothing to optimize"),
            Self::InvalidInformation { kind, index } => {
                write!(f, "{kind} factor {index} has a non-SPD information matrix")
            }
            Self::SingularSystem => write!(f, "GNSS pose graph normal equations are singular"),
        }
    }
}

impl std::error::Error for GnssPoseGraphError {}

/// Outcome of [`GnssPoseGraph::optimize`].
#[derive(Debug, Clone, PartialEq)]
pub struct GnssPoseGraphResult {
    pub variable_node_count: usize,
    pub relative_factor_count: usize,
    pub gnss_factor_count: usize,
    /// Plain (non-robust) least-squares cost at the starting point.
    pub initial_cost: f64,
    /// Plain least-squares cost over the relative factors, the prior, and the
    /// GNSS factors classified as inliers, at the final estimate.
    pub final_inlier_cost: f64,
    /// Total LM iterations (accepted and rejected) across every stage.
    pub iterations: usize,
    /// GNC outer levels executed (0 when GNC is not used).
    pub gnc_levels: usize,
    /// Whether the last LM stage met a convergence tolerance.
    pub converged: bool,
    /// Final robust weight in `[0, 1]` of each GNSS factor (factor order).
    pub gnss_weights: Vec<f64>,
    /// Final whitened squared residual of each GNSS factor.
    pub gnss_squared_residuals: Vec<f64>,
    /// Final inlier classification of each GNSS factor.
    pub gnss_inliers: Vec<bool>,
}

impl GnssPoseGraphResult {
    pub fn outlier_count(&self) -> usize {
        self.gnss_inliers.iter().filter(|inlier| !**inlier).count()
    }
}

/// Pose graph combining VO relative factors with GNSS position factors and a
/// jointly-estimated [`GnssAlignment`]. See the module docs for the model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GnssPoseGraph {
    pub nodes: BTreeMap<u64, GnssGraphNode>,
    pub relative_factors: Vec<VoRelativeFactor>,
    pub gnss_factors: Vec<GnssPositionFactor>,
    pub alignment: GnssAlignment,
    pub alignment_prior: Option<GnssAlignmentPrior>,
    /// GNSS antenna position in the camera frame (metres).
    pub lever_arm: Vector3<f64>,
    /// Nodes held constant (at least one is required to fix the gauge).
    pub fixed_nodes: BTreeSet<u64>,
}

/// Per-factor GNSS weighting used inside one LM stage.
#[derive(Debug, Clone, Copy)]
enum Weighting<'a> {
    /// IRLS with the given kernel (weights recomputed from residuals).
    Kernel(RobustKernel),
    /// Fixed multiplicative weights (GNC level or hard inlier mask).
    Fixed(&'a [f64]),
}

struct Layout {
    node_index: BTreeMap<u64, usize>,
    alignment_block: usize,
    dim: usize,
    /// Per scalar variable: `true` when the variable is held at zero step.
    frozen: Vec<bool>,
}

struct Whitening {
    relative: Vec<Matrix7>,
    gnss: Vec<Matrix3<f64>>,
    prior: Option<Matrix7>,
}

impl GnssPoseGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, id: u64, node: GnssGraphNode) {
        self.nodes.insert(id, node);
    }

    pub fn fix_node(&mut self, id: u64) {
        self.fixed_nodes.insert(id);
    }

    pub fn add_relative_factor(&mut self, factor: VoRelativeFactor) {
        self.relative_factors.push(factor);
    }

    pub fn add_gnss_factor(&mut self, factor: GnssPositionFactor) {
        self.gnss_factors.push(factor);
    }

    /// Camera-to-ENU pose of a node under the current alignment.
    pub fn node_camera_to_enu(&self, id: u64) -> Option<SE3> {
        self.nodes
            .get(&id)
            .map(|node| self.alignment.transform_camera_to_map(&node.camera_to_map))
    }

    /// Predicted ENU antenna position of a node.
    pub fn predicted_antenna_enu(&self, id: u64) -> Option<Vector3<f64>> {
        self.nodes
            .get(&id)
            .map(|node| predicted_antenna(&self.alignment, node, &self.lever_arm))
    }

    /// Whitened squared residual `rᵀΩr` of every GNSS factor at the current
    /// estimate (factor order).
    pub fn gnss_squared_residuals(&self) -> Vec<f64> {
        self.gnss_factors
            .iter()
            .map(|factor| {
                let Some(node) = self.nodes.get(&factor.node) else {
                    return 0.0;
                };
                let r =
                    predicted_antenna(&self.alignment, node, &self.lever_arm) - factor.position_enu;
                (r.transpose() * factor.information * r)[(0, 0)]
            })
            .collect()
    }

    /// Jointly optimize the free nodes and the alignment.
    pub fn optimize(
        &mut self,
        config: &GnssPoseGraphConfig,
    ) -> Result<GnssPoseGraphResult, GnssPoseGraphError> {
        let layout = self.layout(config)?;
        let whitening = self.whitening(config)?;
        let gnss_count = self.gnss_factors.len();
        let ones = vec![1.0; gnss_count];
        let initial_cost = self.cost(&whitening, Weighting::Fixed(&ones));

        let mut symbolic: Option<BlockSymbolic> = None;
        let mut total_iterations = 0usize;
        let mut gnc_levels = 0usize;
        let mut weights = ones.clone();

        let mut converged = match config.robust {
            GnssRobustMode::None => {
                let stage = self.run_lm(
                    &layout,
                    &whitening,
                    config,
                    Weighting::Fixed(&ones),
                    config.max_iterations,
                    &mut symbolic,
                )?;
                total_iterations += stage.iterations;
                stage.converged
            }
            GnssRobustMode::Kernel(kernel) => {
                let stage = self.run_lm(
                    &layout,
                    &whitening,
                    config,
                    Weighting::Kernel(kernel),
                    config.max_iterations,
                    &mut symbolic,
                )?;
                total_iterations += stage.iterations;
                for (weight, s) in weights.iter_mut().zip(self.gnss_squared_residuals()) {
                    *weight = kernel.weight(s);
                }
                stage.converged
            }
            GnssRobustMode::Gnc(gnc_config) => {
                // Least-squares warm start: GNC's first surrogate is convex, so
                // seeding it from the all-inlier solution is consistent.
                let stage = self.run_lm(
                    &layout,
                    &whitening,
                    config,
                    Weighting::Fixed(&ones),
                    config.max_iterations,
                    &mut symbolic,
                )?;
                total_iterations += stage.iterations;
                let mut converged = stage.converged;
                if gnss_count > 0 {
                    let residuals = self.gnss_squared_residuals();
                    let s_max = residuals.iter().copied().fold(0.0_f64, f64::max);
                    let effective = match gnc_config.auto_scale {
                        Some(k) => GncConfig {
                            c: gnc::estimate_scale_mad(&residuals, k)
                                .map_or(gnc_config.c, |c| c.max(gnc_config.c)),
                            ..gnc_config
                        },
                        None => gnc_config,
                    };
                    let mut state = GncState::new(&effective, s_max);
                    for _ in 0..gnc_config.max_outer.max(1) {
                        gnc_levels += 1;
                        let terminal = state.is_terminal();
                        let residuals = self.gnss_squared_residuals();
                        if gnc_config.auto_scale_readapt {
                            if let Some(k) = gnc_config.auto_scale {
                                if let Some(c) = gnc::estimate_scale_mad(&residuals, k) {
                                    state.set_inlier_scale(c.max(gnc_config.c));
                                }
                            }
                        }
                        for (weight, s) in weights.iter_mut().zip(&residuals) {
                            *weight = state.weight(*s);
                        }
                        let stage = self.run_lm(
                            &layout,
                            &whitening,
                            config,
                            Weighting::Fixed(&weights),
                            gnc_config.inner_iterations.max(1),
                            &mut symbolic,
                        )?;
                        total_iterations += stage.iterations;
                        converged = stage.converged;
                        if terminal {
                            break;
                        }
                        state.anneal();
                    }
                    for (weight, s) in weights.iter_mut().zip(self.gnss_squared_residuals()) {
                        *weight = state.weight(s);
                    }
                }
                converged
            }
        };

        let classify = |weights: &[f64], residuals: &[f64]| -> Vec<bool> {
            weights
                .iter()
                .zip(residuals)
                .map(|(&w, &s)| w >= 0.5 && s <= config.outlier_chi2_threshold)
                .collect::<Vec<_>>()
        };
        let mut residuals = self.gnss_squared_residuals();
        let mut inliers = classify(&weights, &residuals);

        if config.refine_without_outliers && gnss_count > 0 {
            let mask: Vec<f64> = inliers
                .iter()
                .map(|&inlier| if inlier { 1.0 } else { 0.0 })
                .collect();
            let stage = self.run_lm(
                &layout,
                &whitening,
                config,
                Weighting::Fixed(&mask),
                config.max_iterations,
                &mut symbolic,
            )?;
            total_iterations += stage.iterations;
            converged = stage.converged;
            residuals = self.gnss_squared_residuals();
            // Hard rejection keeps the earlier verdicts; a previously-accepted
            // fix that the refined trajectory now contradicts is demoted.
            for ((inlier, weight), &s) in inliers.iter_mut().zip(&mut weights).zip(&residuals) {
                if *inlier && s > config.outlier_chi2_threshold {
                    *inlier = false;
                }
                *weight = if *inlier { 1.0 } else { 0.0 };
            }
        }

        let mask: Vec<f64> = inliers
            .iter()
            .map(|&inlier| if inlier { 1.0 } else { 0.0 })
            .collect();
        let final_inlier_cost = self.cost(&whitening, Weighting::Fixed(&mask));

        Ok(GnssPoseGraphResult {
            variable_node_count: layout.node_index.len(),
            relative_factor_count: self.relative_factors.len(),
            gnss_factor_count: gnss_count,
            initial_cost,
            final_inlier_cost,
            iterations: total_iterations,
            gnc_levels,
            converged,
            gnss_weights: weights,
            gnss_squared_residuals: residuals,
            gnss_inliers: inliers,
        })
    }

    fn layout(&self, config: &GnssPoseGraphConfig) -> Result<Layout, GnssPoseGraphError> {
        if self.fixed_nodes.is_empty() {
            return Err(GnssPoseGraphError::NoFixedNode);
        }
        for id in &self.fixed_nodes {
            if !self.nodes.contains_key(id) {
                return Err(GnssPoseGraphError::MissingNode(*id));
            }
        }
        for factor in &self.relative_factors {
            for id in [factor.from, factor.to] {
                if !self.nodes.contains_key(&id) {
                    return Err(GnssPoseGraphError::MissingNode(id));
                }
            }
        }
        for factor in &self.gnss_factors {
            if !self.nodes.contains_key(&factor.node) {
                return Err(GnssPoseGraphError::MissingNode(factor.node));
            }
        }

        let mut node_index = BTreeMap::new();
        for &id in self.nodes.keys() {
            if !self.fixed_nodes.contains(&id) {
                let next = node_index.len();
                node_index.insert(id, next);
            }
        }
        let alignment_free = config.optimize_alignment
            && (!self.gnss_factors.is_empty() || self.alignment_prior.is_some());
        if node_index.is_empty() && !alignment_free {
            return Err(GnssPoseGraphError::NoVariables);
        }
        let alignment_block = node_index.len();
        let dim = (alignment_block + 1) * BLOCK;
        let mut frozen = vec![false; dim];
        if config.scale_mode != GnssScaleMode::ScaleDrift {
            for block in 0..alignment_block {
                frozen[block * BLOCK + 6] = true;
            }
        }
        let a = alignment_block * BLOCK;
        if alignment_free {
            if config.rotation_dof == AlignmentRotationDof::YawOnly {
                frozen[a + 3] = true;
                frozen[a + 4] = true;
            }
            if config.scale_mode == GnssScaleMode::Metric {
                frozen[a + 6] = true;
            }
        } else {
            for flag in &mut frozen[a..a + BLOCK] {
                *flag = true;
            }
        }
        Ok(Layout {
            node_index,
            alignment_block,
            dim,
            frozen,
        })
    }

    fn whitening(&self, config: &GnssPoseGraphConfig) -> Result<Whitening, GnssPoseGraphError> {
        let scale_drift = config.scale_mode == GnssScaleMode::ScaleDrift;
        let mut relative = Vec::with_capacity(self.relative_factors.len());
        for (index, factor) in self.relative_factors.iter().enumerate() {
            let invalid = GnssPoseGraphError::InvalidInformation {
                kind: "relative",
                index,
            };
            let w6 = sqrt_information(&factor.information).ok_or(invalid.clone())?;
            let mut w = Matrix7::zeros();
            w.fixed_view_mut::<6, 6>(0, 0).copy_from(&w6);
            if scale_drift {
                if !(factor.log_scale_information.is_finite() && factor.log_scale_information > 0.0)
                {
                    return Err(invalid);
                }
                w[(6, 6)] = factor.log_scale_information.sqrt();
            }
            relative.push(w);
        }
        let mut gnss = Vec::with_capacity(self.gnss_factors.len());
        for (index, factor) in self.gnss_factors.iter().enumerate() {
            gnss.push(sqrt_information(&factor.information).ok_or(
                GnssPoseGraphError::InvalidInformation {
                    kind: "gnss",
                    index,
                },
            )?);
        }
        let prior = match &self.alignment_prior {
            Some(prior) => Some(sqrt_information(&prior.information).ok_or(
                GnssPoseGraphError::InvalidInformation {
                    kind: "alignment prior",
                    index: 0,
                },
            )?),
            None => None,
        };
        Ok(Whitening {
            relative,
            gnss,
            prior,
        })
    }

    /// Total cost `Σ‖W r‖²` with GNSS factors weighted per `weighting`.
    fn cost(&self, whitening: &Whitening, weighting: Weighting<'_>) -> f64 {
        let mut total = 0.0;
        for (factor, w) in self.relative_factors.iter().zip(&whitening.relative) {
            let (r, _, _) = relative_residual(
                &self.nodes[&factor.from],
                &self.nodes[&factor.to],
                &factor.relative,
            );
            total += (w * r).norm_squared();
        }
        for (index, (factor, w)) in self.gnss_factors.iter().zip(&whitening.gnss).enumerate() {
            let r = predicted_antenna(&self.alignment, &self.nodes[&factor.node], &self.lever_arm)
                - factor.position_enu;
            let s = (w * r).norm_squared();
            total += match weighting {
                Weighting::Kernel(kernel) => kernel.cost(s),
                Weighting::Fixed(weights) => weights[index] * s,
            };
        }
        if let (Some(prior), Some(w)) = (&self.alignment_prior, &whitening.prior) {
            let (r, _) = alignment_prior_residual(&self.alignment, &prior.mean);
            total += (w * r).norm_squared();
        }
        total
    }

    fn run_lm(
        &mut self,
        layout: &Layout,
        whitening: &Whitening,
        config: &GnssPoseGraphConfig,
        weighting: Weighting<'_>,
        max_iterations: usize,
        symbolic: &mut Option<BlockSymbolic>,
    ) -> Result<LmStage, GnssPoseGraphError> {
        let mut lambda = config.initial_lambda.max(config.min_lambda);
        let mut cost = self.cost(whitening, weighting);
        let mut iterations = 0usize;
        let mut converged = false;
        // The normal equations only change after an accepted step, so a
        // rejected LM trial re-solves the cached system with a larger λ.
        let mut system = self.assemble(layout, whitening, weighting);
        while iterations < max_iterations {
            iterations += 1;
            let delta = solve_damped(layout, &system.0, &system.1, lambda, symbolic)?;
            let saved_nodes = self.nodes.clone();
            let saved_alignment = self.alignment.clone();
            let max_step = self.apply_step(layout, &delta);
            let new_cost = self.cost(whitening, weighting);
            if new_cost.is_finite() && new_cost <= cost {
                let decrease = cost - new_cost;
                cost = new_cost;
                lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
                if max_step < config.step_tolerance
                    || decrease <= config.relative_cost_tolerance * cost.max(1e-12)
                {
                    converged = true;
                    break;
                }
                system = self.assemble(layout, whitening, weighting);
            } else {
                self.nodes = saved_nodes;
                self.alignment = saved_alignment;
                lambda *= config.lambda_increase_factor;
                if lambda > config.max_lambda {
                    break;
                }
            }
        }
        Ok(LmStage {
            iterations,
            converged,
        })
    }

    /// Assemble `H = Σ JᵀJ` as lower block columns (`columns[j][i] = H_ij`,
    /// `i ≥ j`) and `g = Σ Jᵀr` over whitened residuals.
    fn assemble(
        &self,
        layout: &Layout,
        whitening: &Whitening,
        weighting: Weighting<'_>,
    ) -> (BlockColumns, DVector<f64>) {
        let block_count = layout.alignment_block + 1;
        // Every diagonal block exists so frozen variables can be pinned.
        let mut columns: BlockColumns = (0..block_count)
            .map(|block| BTreeMap::from([(block, Matrix7::zeros())]))
            .collect();
        let mut gradient = DVector::<f64>::zeros(layout.dim);

        for (factor, w) in self.relative_factors.iter().zip(&whitening.relative) {
            let (r, j_from, j_to) = relative_residual(
                &self.nodes[&factor.from],
                &self.nodes[&factor.to],
                &factor.relative,
            );
            accumulate(
                &mut columns,
                &mut gradient,
                &[
                    (layout.node_index.get(&factor.from).copied(), w * j_from),
                    (layout.node_index.get(&factor.to).copied(), w * j_to),
                ],
                &(w * r),
                1.0,
            );
        }

        for (index, (factor, w)) in self.gnss_factors.iter().zip(&whitening.gnss).enumerate() {
            let node = &self.nodes[&factor.node];
            let (r, j_node, j_alignment) =
                gnss_residual(&self.alignment, node, &self.lever_arm, &factor.position_enu);
            let rw = w * r;
            let weight = match weighting {
                Weighting::Kernel(kernel) => kernel.weight(rw.norm_squared()),
                Weighting::Fixed(weights) => weights[index],
            };
            if weight <= 0.0 {
                continue;
            }
            accumulate(
                &mut columns,
                &mut gradient,
                &[
                    (layout.node_index.get(&factor.node).copied(), w * j_node),
                    (Some(layout.alignment_block), w * j_alignment),
                ],
                &rw,
                weight,
            );
        }

        if let (Some(prior), Some(w)) = (&self.alignment_prior, &whitening.prior) {
            let (r, j) = alignment_prior_residual(&self.alignment, &prior.mean);
            accumulate(
                &mut columns,
                &mut gradient,
                &[(Some(layout.alignment_block), w * j)],
                &(w * r),
                1.0,
            );
        }

        (columns, gradient)
    }

    fn apply_step(&mut self, layout: &Layout, delta: &DVector<f64>) -> f64 {
        let mut max_step: f64 = 0.0;
        for (id, &block) in &layout.node_index {
            let d = Vector7::from_fn(|k, _| delta[block * BLOCK + k]);
            max_step = max_step.max(d.norm());
            let node = self.nodes.get_mut(id).expect("layout node exists");
            let dp = Vector3::new(d[0], d[1], d[2]);
            let dtheta = Vector3::new(d[3], d[4], d[5]);
            node.camera_to_map = SE3::new(
                node.camera_to_map.rotation * UnitQuaternion::from_scaled_axis(dtheta),
                node.camera_to_map.translation + dp,
            );
            node.log_scale += d[6];
        }
        let a = layout.alignment_block * BLOCK;
        let d = Vector7::from_fn(|k, _| delta[a + k]);
        max_step = max_step.max(d.norm());
        let dphi = Vector3::new(d[3], d[4], d[5]);
        self.alignment.translation += Vector3::new(d[0], d[1], d[2]);
        self.alignment.rotation = UnitQuaternion::from_scaled_axis(dphi) * self.alignment.rotation;
        self.alignment.scale *= d[6].exp();
        max_step
    }
}

struct LmStage {
    iterations: usize,
    converged: bool,
}

fn solve_damped(
    layout: &Layout,
    columns: &BlockColumns,
    gradient: &DVector<f64>,
    lambda: f64,
    symbolic: &mut Option<BlockSymbolic>,
) -> Result<DVector<f64>, GnssPoseGraphError> {
    // Scale-aware floor so variables with (near-)zero curvature stay solvable.
    let max_diag = columns
        .iter()
        .enumerate()
        .filter_map(|(j, column)| column.get(&j))
        .flat_map(|block| (0..BLOCK).map(move |k| block[(k, k)]))
        .fold(0.0_f64, f64::max);
    let floor = (max_diag * 1e-12).max(1e-12);

    let mut damped = columns.clone();
    for (j, column) in damped.iter_mut().enumerate() {
        for (&i, block) in column.iter_mut() {
            for r in 0..BLOCK {
                let row = i * BLOCK + r;
                for c in 0..BLOCK {
                    let col = j * BLOCK + c;
                    if layout.frozen[row] || layout.frozen[col] {
                        block[(r, c)] = if row == col { 1.0 } else { 0.0 };
                    } else if row == col {
                        let value = block[(r, c)];
                        block[(r, c)] = value + lambda * value.max(floor) + floor;
                    }
                }
            }
        }
    }
    let rhs = DVector::from_fn(layout.dim, |row, _| {
        if layout.frozen[row] {
            0.0
        } else {
            -gradient[row]
        }
    });
    let delta = block_cholesky::solve_spd_blocks_cached::<BLOCK>(symbolic, damped, &rhs)
        .map_err(|()| GnssPoseGraphError::SingularSystem)?;
    if delta.iter().all(|v| v.is_finite()) {
        Ok(delta)
    } else {
        Err(GnssPoseGraphError::SingularSystem)
    }
}

/// Lower block columns of the normal matrix: `columns[j][i] = H_ij`, `i ≥ j`.
type BlockColumns = Vec<BTreeMap<usize, Matrix7>>;

/// Add one factor's `JᵀJ` / `Jᵀr` to the normal equations. `terms` pairs each
/// involved variable block (`None` = fixed) with its whitened `R×7` Jacobian.
fn accumulate<const R: usize>(
    columns: &mut BlockColumns,
    gradient: &mut DVector<f64>,
    terms: &[(Option<usize>, SMatrix<f64, R, 7>)],
    residual: &SVector<f64, R>,
    weight: f64,
) {
    for (bi, ji) in terms {
        let Some(bi) = *bi else { continue };
        let gi = ji.transpose() * residual * weight;
        for k in 0..BLOCK {
            gradient[bi * BLOCK + k] += gi[k];
        }
        for (bj, jj) in terms {
            let Some(bj) = *bj else { continue };
            if bi < bj {
                continue;
            }
            // Block (row bi, column bj) of the lower triangle.
            let h = ji.transpose() * jj * weight;
            *columns[bj].entry(bi).or_insert_with(Matrix7::zeros) += h;
        }
    }
}

/// Upper-triangular `W` with `WᵀW = Ω` (so `‖W r‖² = rᵀΩr`).
fn sqrt_information<const N: usize>(information: &SMatrix<f64, N, N>) -> Option<SMatrix<f64, N, N>>
where
    nalgebra::Const<N>: nalgebra::DimMin<nalgebra::Const<N>, Output = nalgebra::Const<N>>,
{
    if !information.iter().all(|v| v.is_finite()) {
        return None;
    }
    let symmetric = (information + information.transpose()) * 0.5;
    let cholesky = symmetric.cholesky()?;
    Some(cholesky.l().transpose())
}

pub(crate) fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Inverse right Jacobian of SO(3): `Log(Exp(φ) Exp(δ)) ≈ φ + J_r⁻¹(φ) δ`.
fn so3_right_jacobian_inverse(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();
    let k = skew(phi);
    if theta < 1e-6 {
        return Matrix3::identity() + 0.5 * k + (1.0 / 12.0) * k * k;
    }
    let coefficient = 1.0 / (theta * theta) - (1.0 + theta.cos()) / (2.0 * theta * theta.sin());
    Matrix3::identity() + 0.5 * k + coefficient * k * k
}

/// Inverse left Jacobian of SO(3): `Log(Exp(δ) Exp(φ)) ≈ φ + J_l⁻¹(φ) δ`.
fn so3_left_jacobian_inverse(phi: &Vector3<f64>) -> Matrix3<f64> {
    so3_right_jacobian_inverse(&(-phi))
}

/// Predicted ENU antenna position `s·R·p + t + R·Rᵢ·l`.
fn predicted_antenna(
    alignment: &GnssAlignment,
    node: &GnssGraphNode,
    lever_arm: &Vector3<f64>,
) -> Vector3<f64> {
    alignment.transform_point(&node.camera_to_map.translation)
        + alignment.rotation * (node.camera_to_map.rotation * lever_arm)
}

/// GNSS residual and Jacobians w.r.t. the node `[δp; δθ; δσ]` and the
/// alignment `[δt; δφ; δσ_a]` (unwhitened).
pub(crate) fn gnss_residual(
    alignment: &GnssAlignment,
    node: &GnssGraphNode,
    lever_arm: &Vector3<f64>,
    measured_enu: &Vector3<f64>,
) -> (Vector3<f64>, Matrix3x7, Matrix3x7) {
    let ra = alignment.rotation.to_rotation_matrix().into_inner();
    let ri = node
        .camera_to_map
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let p = node.camera_to_map.translation;
    let scaled = ra * p * alignment.scale;
    let lever_enu = ra * ri * lever_arm;
    let r = scaled + alignment.translation + lever_enu - measured_enu;

    let mut j_node = Matrix3x7::zeros();
    j_node
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(ra * alignment.scale));
    j_node
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-(ra * ri) * skew(lever_arm)));

    let mut j_alignment = Matrix3x7::zeros();
    j_alignment
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&Matrix3::identity());
    j_alignment
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew(&(scaled + lever_enu))));
    j_alignment.fixed_view_mut::<3, 1>(0, 6).copy_from(&scaled);
    (r, j_node, j_alignment)
}

/// Relative-pose residual `[e^{-σᵢ}Rᵢᵀ(pⱼ−pᵢ) − t_Z; Log(R_Zᵀ Rᵢᵀ Rⱼ); σⱼ − σᵢ]`
/// and its Jacobians w.r.t. `[δp; δθ; δσ]` of `from` and `to` (unwhitened).
pub(crate) fn relative_residual(
    from: &GnssGraphNode,
    to: &GnssGraphNode,
    measurement: &SE3,
) -> (Vector7, Matrix7, Matrix7) {
    let ri = from
        .camera_to_map
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let rj = to.camera_to_map.rotation.to_rotation_matrix().into_inner();
    let inv_scale = (-from.log_scale).exp();
    let v = ri.transpose() * (to.camera_to_map.translation - from.camera_to_map.translation);
    let r_t = v * inv_scale - measurement.translation;
    let rotation_error = measurement.rotation.inverse()
        * from.camera_to_map.rotation.inverse()
        * to.camera_to_map.rotation;
    let r_r = rotation_error.scaled_axis();
    let r_s = to.log_scale - from.log_scale;

    let mut r = Vector7::zeros();
    r.fixed_view_mut::<3, 1>(0, 0).copy_from(&r_t);
    r.fixed_view_mut::<3, 1>(3, 0).copy_from(&r_r);
    r[6] = r_s;

    let jr_inv = so3_right_jacobian_inverse(&r_r);
    let mut j_from = Matrix7::zeros();
    j_from
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-ri.transpose() * inv_scale));
    j_from
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(skew(&v) * inv_scale));
    j_from
        .fixed_view_mut::<3, 1>(0, 6)
        .copy_from(&(-v * inv_scale));
    j_from
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(-jr_inv * rj.transpose() * ri));
    j_from[(6, 6)] = -1.0;

    let mut j_to = Matrix7::zeros();
    j_to.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(ri.transpose() * inv_scale));
    j_to.fixed_view_mut::<3, 3>(3, 3).copy_from(&jr_inv);
    j_to[(6, 6)] = 1.0;
    (r, j_from, j_to)
}

/// Alignment prior residual `[t − t₀; Log(R R₀ᵀ); ln s − ln s₀]` and its
/// Jacobian w.r.t. `[δt; δφ; δσ]`.
pub(crate) fn alignment_prior_residual(
    alignment: &GnssAlignment,
    mean: &GnssAlignment,
) -> (Vector7, Matrix7) {
    let e_r = (alignment.rotation * mean.rotation.inverse()).scaled_axis();
    let mut r = Vector7::zeros();
    r.fixed_view_mut::<3, 1>(0, 0)
        .copy_from(&(alignment.translation - mean.translation));
    r.fixed_view_mut::<3, 1>(3, 0).copy_from(&e_r);
    r[6] = alignment.scale.ln() - mean.scale.ln();
    let mut j = Matrix7::identity();
    j.fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&so3_left_jacobian_inverse(&e_r));
    (r, j)
}

// ---------------------------------------------------------------------------
// Robust closed-form bootstrap of the alignment.
// ---------------------------------------------------------------------------

/// One map-frame camera pose paired with a GNSS antenna position for the
/// alignment bootstrap.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentCorrespondence {
    /// Camera-to-map pose of the frame (VO estimate).
    pub camera_to_map: SE3,
    /// GNSS antenna position in ENU.
    pub position_enu: Vector3<f64>,
    pub horizontal_sigma: f64,
    pub vertical_sigma: f64,
}

/// Configuration for [`bootstrap_gnss_alignment`].
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentBootstrapConfig {
    pub rotation_dof: AlignmentRotationDof,
    pub scale_mode: GnssScaleMode,
    /// Antenna position in the camera frame (metres).
    pub lever_arm: Vector3<f64>,
    /// RANSAC hypotheses (minimal samples) to draw.
    pub ransac_iterations: usize,
    /// Inlier gate: Mahalanobis distance (in sigmas) of a correspondence's
    /// residual under a hypothesis.
    pub inlier_threshold_sigmas: f64,
    /// Additive slack (metres) on the gate, absorbing VO drift inside the
    /// bootstrap window.
    pub inlier_threshold_floor: f64,
    /// Minimum inliers for the bootstrap to be accepted.
    pub min_inliers: usize,
    /// Minimum horizontal extent (metres, in ENU) of the inlier fixes; below
    /// this the yaw (and scale) are not observable.
    pub min_horizontal_extent: f64,
    /// Seed of the deterministic sampler.
    pub seed: u64,
}

impl Default for AlignmentBootstrapConfig {
    fn default() -> Self {
        Self {
            rotation_dof: AlignmentRotationDof::YawOnly,
            scale_mode: GnssScaleMode::Metric,
            lever_arm: Vector3::zeros(),
            ransac_iterations: 300,
            inlier_threshold_sigmas: 3.0,
            inlier_threshold_floor: 1.0,
            min_inliers: 6,
            min_horizontal_extent: 20.0,
            seed: 0x5eed_9a55,
        }
    }
}

/// Result of [`bootstrap_gnss_alignment`].
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentBootstrap {
    pub alignment: GnssAlignment,
    pub inliers: Vec<bool>,
    pub inlier_count: usize,
    /// RMS ENU residual (metres) over the inliers.
    pub inlier_rms: f64,
}

/// Weighted closed-form fit of the alignment to correspondences
/// (yaw-only 2-D Procrustes or 3-D Umeyama, with optional scale). The lever
/// arm depends on the unknown rotation, so the fit is iterated a few times.
/// Returns `None` for a degenerate configuration (fewer than 2 points, zero
/// spread).
pub fn fit_gnss_alignment(
    correspondences: &[AlignmentCorrespondence],
    rotation_dof: AlignmentRotationDof,
    scale_mode: GnssScaleMode,
    lever_arm: &Vector3<f64>,
) -> Option<GnssAlignment> {
    if correspondences.len() < 2 {
        return None;
    }
    let mut alignment = GnssAlignment::identity();
    let iterations = if lever_arm.norm() > 0.0 { 4 } else { 1 };
    for _ in 0..iterations {
        let targets: Vec<Vector3<f64>> = correspondences
            .iter()
            .map(|c| c.position_enu - alignment.rotation * (c.camera_to_map.rotation * lever_arm))
            .collect();
        alignment = fit_points(correspondences, &targets, rotation_dof, scale_mode)?;
    }
    Some(alignment)
}

fn fit_points(
    correspondences: &[AlignmentCorrespondence],
    targets: &[Vector3<f64>],
    rotation_dof: AlignmentRotationDof,
    scale_mode: GnssScaleMode,
) -> Option<GnssAlignment> {
    let weights: Vec<f64> = correspondences
        .iter()
        .map(|c| 1.0 / c.horizontal_sigma.max(1e-6).powi(2))
        .collect();
    let total: f64 = weights.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let mut source_mean = Vector3::zeros();
    let mut target_mean = Vector3::zeros();
    for ((c, t), w) in correspondences.iter().zip(targets).zip(&weights) {
        source_mean += c.camera_to_map.translation * *w;
        target_mean += t * *w;
    }
    source_mean /= total;
    target_mean /= total;
    let estimate_scale = scale_mode != GnssScaleMode::Metric;

    let (rotation, scale) = match rotation_dof {
        AlignmentRotationDof::YawOnly => {
            let (mut sin_sum, mut cos_sum, mut source_var) = (0.0, 0.0, 0.0);
            for ((c, t), w) in correspondences.iter().zip(targets).zip(&weights) {
                let x = c.camera_to_map.translation - source_mean;
                let y = t - target_mean;
                sin_sum += w * (x.x * y.y - x.y * y.x);
                cos_sum += w * (x.x * y.x + x.y * y.y);
                source_var += w * (x.x * x.x + x.y * x.y);
            }
            if source_var <= 1e-12 * total || (sin_sum == 0.0 && cos_sum == 0.0) {
                return None;
            }
            let yaw = sin_sum.atan2(cos_sum);
            let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
            let scale = if estimate_scale {
                // Horizontal-only scale: vertical GNSS is noisier and a
                // gravity-aligned map shares the ENU up axis.
                (sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / source_var
            } else {
                1.0
            };
            (rotation, scale)
        }
        AlignmentRotationDof::Full => {
            let mut sigma = Matrix3::zeros();
            let mut source_var = 0.0;
            for ((c, t), w) in correspondences.iter().zip(targets).zip(&weights) {
                let x = c.camera_to_map.translation - source_mean;
                let y = t - target_mean;
                sigma += y * x.transpose() * *w;
                source_var += w * x.norm_squared();
            }
            if source_var <= 1e-12 * total {
                return None;
            }
            let svd = sigma.try_svd(true, true, 1e-15, 1000)?;
            let (u, v_t) = (svd.u?, svd.v_t?);
            let mut d = Matrix3::identity();
            if (u * v_t).determinant() < 0.0 {
                d[(2, 2)] = -1.0;
            }
            let r = u * d * v_t;
            let scale = if estimate_scale {
                (svd.singular_values.component_mul(&d.diagonal())).sum() / source_var
            } else {
                1.0
            };
            (
                UnitQuaternion::from_matrix_eps(&r, 1e-12, 100, UnitQuaternion::identity()),
                scale,
            )
        }
    };
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let translation = target_mean - rotation * (source_mean * scale);
    Some(GnssAlignment {
        rotation,
        translation,
        scale,
    })
}

fn bootstrap_residual(
    alignment: &GnssAlignment,
    correspondence: &AlignmentCorrespondence,
    lever_arm: &Vector3<f64>,
) -> Vector3<f64> {
    alignment.transform_point(&correspondence.camera_to_map.translation)
        + alignment.rotation * (correspondence.camera_to_map.rotation * lever_arm)
        - correspondence.position_enu
}

fn is_bootstrap_inlier(
    alignment: &GnssAlignment,
    correspondence: &AlignmentCorrespondence,
    config: &AlignmentBootstrapConfig,
) -> bool {
    let r = bootstrap_residual(alignment, correspondence, &config.lever_arm);
    let k = config.inlier_threshold_sigmas;
    let floor = config.inlier_threshold_floor;
    let h_gate = k * correspondence.horizontal_sigma + floor;
    let v_gate = k * correspondence.vertical_sigma + floor;
    (r.x * r.x + r.y * r.y) / (h_gate * h_gate) + r.z * r.z / (v_gate * v_gate) <= 1.0
}

/// Robust (RANSAC + weighted least-squares refit) estimate of the
/// GNSS-ENU-to-map alignment from map-frame camera poses and GNSS fixes.
///
/// Minimal samples are 2 correspondences for yaw-only alignments and 3 for
/// full rotations. Returns `None` when there are too few correspondences, the
/// inlier fixes span less than [`AlignmentBootstrapConfig::min_horizontal_extent`]
/// metres (yaw/scale unobservable), or no hypothesis gathers
/// [`AlignmentBootstrapConfig::min_inliers`] inliers.
pub fn bootstrap_gnss_alignment(
    correspondences: &[AlignmentCorrespondence],
    config: &AlignmentBootstrapConfig,
) -> Option<AlignmentBootstrap> {
    let n = correspondences.len();
    let minimal = match config.rotation_dof {
        AlignmentRotationDof::YawOnly => 2,
        AlignmentRotationDof::Full => 3,
    };
    if n < minimal.max(config.min_inliers) {
        return None;
    }
    let mut rng = config.seed | 1;
    let mut next = move || {
        // xorshift64*: deterministic and dependency-free.
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let mut best: Option<(usize, f64, Vec<bool>)> = None;
    for _ in 0..config.ransac_iterations.max(1) {
        let mut sample: Vec<usize> = Vec::with_capacity(minimal);
        let mut guard = 0;
        while sample.len() < minimal && guard < 64 {
            guard += 1;
            let index = (next() % n as u64) as usize;
            if !sample.contains(&index) {
                sample.push(index);
            }
        }
        if sample.len() < minimal {
            continue;
        }
        let subset: Vec<AlignmentCorrespondence> =
            sample.iter().map(|&i| correspondences[i].clone()).collect();
        // Reject near-degenerate samples (points too close in the map).
        let spread = subset
            .iter()
            .map(|c| (c.camera_to_map.translation - subset[0].camera_to_map.translation).norm())
            .fold(0.0_f64, f64::max);
        if spread < 1e-3 {
            continue;
        }
        let Some(hypothesis) = fit_gnss_alignment(
            &subset,
            config.rotation_dof,
            config.scale_mode,
            &config.lever_arm,
        ) else {
            continue;
        };
        let inliers: Vec<bool> = correspondences
            .iter()
            .map(|c| is_bootstrap_inlier(&hypothesis, c, config))
            .collect();
        let count = inliers.iter().filter(|x| **x).count();
        let score: f64 = correspondences
            .iter()
            .zip(&inliers)
            .filter(|(_, inlier)| **inlier)
            .map(|(c, _)| bootstrap_residual(&hypothesis, c, &config.lever_arm).norm_squared())
            .sum();
        let better = match &best {
            None => true,
            Some((best_count, best_score, _)) => {
                count > *best_count || (count == *best_count && score < *best_score)
            }
        };
        if better {
            best = Some((count, score, inliers));
        }
    }
    let (_, _, mut inliers) = best?;
    // Refit on the consensus set, then re-gate once against the refit.
    let mut alignment = None;
    for _ in 0..2 {
        let subset: Vec<AlignmentCorrespondence> = correspondences
            .iter()
            .zip(&inliers)
            .filter(|(_, inlier)| **inlier)
            .map(|(c, _)| c.clone())
            .collect();
        let refit = fit_gnss_alignment(
            &subset,
            config.rotation_dof,
            config.scale_mode,
            &config.lever_arm,
        )?;
        inliers = correspondences
            .iter()
            .map(|c| is_bootstrap_inlier(&refit, c, config))
            .collect();
        alignment = Some(refit);
    }
    let alignment = alignment?;
    let inlier_points: Vec<&AlignmentCorrespondence> = correspondences
        .iter()
        .zip(&inliers)
        .filter(|(_, inlier)| **inlier)
        .map(|(c, _)| c)
        .collect();
    let inlier_count = inlier_points.len();
    if inlier_count < config.min_inliers.max(minimal) {
        return None;
    }
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for c in &inlier_points {
        min_x = min_x.min(c.position_enu.x);
        max_x = max_x.max(c.position_enu.x);
        min_y = min_y.min(c.position_enu.y);
        max_y = max_y.max(c.position_enu.y);
    }
    let extent = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();
    if extent < config.min_horizontal_extent {
        return None;
    }
    let inlier_rms = (inlier_points
        .iter()
        .map(|c| bootstrap_residual(&alignment, c, &config.lever_arm).norm_squared())
        .sum::<f64>()
        / inlier_count as f64)
        .sqrt();
    Some(AlignmentBootstrap {
        alignment,
        inliers,
        inlier_count,
        inlier_rms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(rotation: Vector3<f64>, position: Vector3<f64>, log_scale: f64) -> GnssGraphNode {
        GnssGraphNode {
            camera_to_map: SE3::new(UnitQuaternion::from_scaled_axis(rotation), position),
            log_scale,
        }
    }

    fn perturb_node(node: &GnssGraphNode, d: &Vector7) -> GnssGraphNode {
        GnssGraphNode {
            camera_to_map: SE3::new(
                node.camera_to_map.rotation
                    * UnitQuaternion::from_scaled_axis(Vector3::new(d[3], d[4], d[5])),
                node.camera_to_map.translation + Vector3::new(d[0], d[1], d[2]),
            ),
            log_scale: node.log_scale + d[6],
        }
    }

    fn perturb_alignment(alignment: &GnssAlignment, d: &Vector7) -> GnssAlignment {
        GnssAlignment {
            rotation: UnitQuaternion::from_scaled_axis(Vector3::new(d[3], d[4], d[5]))
                * alignment.rotation,
            translation: alignment.translation + Vector3::new(d[0], d[1], d[2]),
            scale: alignment.scale * d[6].exp(),
        }
    }

    fn numeric<const R: usize>(f: impl Fn(&Vector7) -> SVector<f64, R>) -> SMatrix<f64, R, 7> {
        let h = 1e-6;
        let mut j = SMatrix::<f64, R, 7>::zeros();
        for k in 0..7 {
            let mut plus = Vector7::zeros();
            plus[k] = h;
            let mut minus = Vector7::zeros();
            minus[k] = -h;
            j.set_column(k, &((f(&plus) - f(&minus)) / (2.0 * h)));
        }
        j
    }

    fn assert_close<const R: usize>(a: &SMatrix<f64, R, 7>, b: &SMatrix<f64, R, 7>) {
        let err = (a - b).abs().max();
        assert!(
            err < 1e-6,
            "jacobian mismatch {err}\nanalytic {a}\nnumeric {b}"
        );
    }

    #[test]
    fn gnss_factor_jacobians_match_finite_differences() {
        let alignment = GnssAlignment {
            rotation: UnitQuaternion::from_scaled_axis(Vector3::new(0.1, -0.2, 0.7)),
            translation: Vector3::new(3.0, -2.0, 1.0),
            scale: 1.7,
        };
        let n = node(
            Vector3::new(0.3, 0.2, -0.4),
            Vector3::new(4.0, 1.0, -2.0),
            0.2,
        );
        let lever = Vector3::new(0.3, -1.2, -0.5);
        let z = Vector3::new(10.0, 2.0, 0.5);
        let (_, j_node, j_alignment) = gnss_residual(&alignment, &n, &lever, &z);
        let num_node = numeric(|d| gnss_residual(&alignment, &perturb_node(&n, d), &lever, &z).0);
        let num_alignment =
            numeric(|d| gnss_residual(&perturb_alignment(&alignment, d), &n, &lever, &z).0);
        assert_close(&j_node, &num_node);
        assert_close(&j_alignment, &num_alignment);
    }

    #[test]
    fn relative_factor_jacobians_match_finite_differences() {
        let from = node(
            Vector3::new(0.1, -0.3, 0.2),
            Vector3::new(1.0, 2.0, 3.0),
            0.15,
        );
        let to = node(
            Vector3::new(0.2, -0.1, 0.5),
            Vector3::new(2.5, 1.0, 3.5),
            -0.1,
        );
        let measurement = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.05, 0.2, 0.1)),
            Vector3::new(0.8, -0.4, 1.1),
        );
        let (_, j_from, j_to) = relative_residual(&from, &to, &measurement);
        let num_from = numeric(|d| relative_residual(&perturb_node(&from, d), &to, &measurement).0);
        let num_to = numeric(|d| relative_residual(&from, &perturb_node(&to, d), &measurement).0);
        assert_close(&j_from, &num_from);
        assert_close(&j_to, &num_to);
    }

    #[test]
    fn alignment_prior_jacobian_matches_finite_differences() {
        let mean = GnssAlignment {
            rotation: UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.1, 1.2)),
            translation: Vector3::new(1.0, 2.0, 3.0),
            scale: 0.8,
        };
        let current = GnssAlignment {
            rotation: UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -0.1, 1.5)),
            translation: Vector3::new(1.5, 1.0, 2.0),
            scale: 1.1,
        };
        let (_, j) = alignment_prior_residual(&current, &mean);
        let num = numeric(|d| alignment_prior_residual(&perturb_alignment(&current, d), &mean).0);
        assert_close(&j, &num);
    }

    fn straight_and_turn_path(count: usize) -> Vec<SE3> {
        (0..count)
            .map(|i| {
                let t = i as f64;
                let heading = 0.02 * t;
                SE3::new(
                    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), heading),
                    Vector3::new(10.0 * (0.05 * t).sin() + t, 0.02 * t * t, 0.05 * t),
                )
            })
            .collect()
    }

    #[test]
    fn bootstrap_recovers_yaw_scale_translation_with_outliers() {
        let truth = GnssAlignment::from_yaw(1.1, Vector3::new(100.0, -50.0, 20.0), 2.5);
        let lever = Vector3::new(0.0, -1.0, -0.4);
        let path = straight_and_turn_path(40);
        let mut correspondences: Vec<AlignmentCorrespondence> = path
            .iter()
            .map(|pose| AlignmentCorrespondence {
                camera_to_map: pose.clone(),
                position_enu: truth.transform_point(&pose.translation)
                    + truth.rotation * (pose.rotation * lever),
                horizontal_sigma: 0.5,
                vertical_sigma: 1.0,
            })
            .collect();
        for index in [5, 17, 30] {
            correspondences[index].position_enu += Vector3::new(40.0, -25.0, 3.0);
        }
        let config = AlignmentBootstrapConfig {
            scale_mode: GnssScaleMode::GlobalScale,
            lever_arm: lever,
            ..AlignmentBootstrapConfig::default()
        };
        let result = bootstrap_gnss_alignment(&correspondences, &config).expect("bootstrap");
        assert_eq!(result.inlier_count, 37);
        assert!(!result.inliers[5] && !result.inliers[17] && !result.inliers[30]);
        assert!((result.alignment.yaw() - 1.1).abs() < 1e-6);
        assert!((result.alignment.scale - 2.5).abs() < 1e-6);
        assert!((result.alignment.translation - truth.translation).norm() < 1e-5);
    }

    #[test]
    fn bootstrap_recovers_full_rotation() {
        let truth = GnssAlignment {
            rotation: UnitQuaternion::from_scaled_axis(Vector3::new(1.2, -0.4, 0.3)),
            translation: Vector3::new(-5.0, 7.0, 2.0),
            scale: 1.0,
        };
        let path = straight_and_turn_path(30);
        let correspondences: Vec<AlignmentCorrespondence> = path
            .iter()
            .map(|pose| AlignmentCorrespondence {
                camera_to_map: pose.clone(),
                position_enu: truth.transform_point(&pose.translation),
                horizontal_sigma: 0.5,
                vertical_sigma: 1.0,
            })
            .collect();
        let config = AlignmentBootstrapConfig {
            rotation_dof: AlignmentRotationDof::Full,
            ..AlignmentBootstrapConfig::default()
        };
        let result = bootstrap_gnss_alignment(&correspondences, &config).expect("bootstrap");
        assert_eq!(result.inlier_count, 30);
        assert!(result.alignment.rotation.angle_to(&truth.rotation) < 1e-6);
        assert!((result.alignment.translation - truth.translation).norm() < 1e-5);
    }

    #[test]
    fn bootstrap_rejects_short_baseline() {
        let path: Vec<SE3> = (0..20)
            .map(|i| {
                SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(0.1 * i as f64, 0.0, 0.0),
                )
            })
            .collect();
        let correspondences: Vec<AlignmentCorrespondence> = path
            .iter()
            .map(|pose| AlignmentCorrespondence {
                camera_to_map: pose.clone(),
                position_enu: pose.translation,
                horizontal_sigma: 0.5,
                vertical_sigma: 1.0,
            })
            .collect();
        assert!(
            bootstrap_gnss_alignment(&correspondences, &AlignmentBootstrapConfig::default())
                .is_none()
        );
    }

    /// Chain of metric VO nodes, true alignment, and noiseless GNSS: the
    /// optimizer must recover the alignment from a perturbed start.
    #[test]
    fn optimizer_recovers_alignment_and_rejects_outlier() {
        let truth = GnssAlignment::from_yaw(-0.6, Vector3::new(20.0, 30.0, -4.0), 1.0);
        let path = straight_and_turn_path(30);
        let mut graph = GnssPoseGraph::new();
        for (i, pose) in path.iter().enumerate() {
            graph.add_node(i as u64, GnssGraphNode::new(pose.clone()));
        }
        graph.fix_node(0);
        for i in 1..path.len() {
            graph.add_relative_factor(VoRelativeFactor {
                from: i as u64 - 1,
                to: i as u64,
                relative: path[i - 1].inverse().compose(&path[i]),
                information: Matrix6::identity() * 100.0,
                log_scale_information: 1.0,
            });
        }
        for (i, pose) in path.iter().enumerate() {
            let mut position_enu = truth.transform_point(&pose.translation);
            if i == 12 {
                position_enu += Vector3::new(25.0, 0.0, 0.0);
            }
            graph.add_gnss_factor(GnssPositionFactor {
                node: i as u64,
                position_enu,
                information: Matrix3::identity(),
            });
        }
        graph.alignment = GnssAlignment::from_yaw(-0.4, Vector3::new(18.0, 33.0, -3.0), 1.0);
        let result = graph
            .optimize(&GnssPoseGraphConfig::default())
            .expect("optimize");
        assert_eq!(result.outlier_count(), 1);
        assert!(!result.gnss_inliers[12]);
        assert!(
            (graph.alignment.yaw() + 0.6).abs() < 1e-4,
            "{}",
            graph.alignment.yaw()
        );
        assert!((graph.alignment.translation - truth.translation).norm() < 1e-3);
        for (i, pose) in path.iter().enumerate() {
            let fused = &graph.nodes[&(i as u64)].camera_to_map.translation;
            assert!((fused - pose.translation).norm() < 1e-3);
        }
    }

    #[test]
    fn optimizer_requires_fixed_node() {
        let mut graph = GnssPoseGraph::new();
        graph.add_node(0, GnssGraphNode::new(SE3::identity()));
        assert_eq!(
            graph.optimize(&GnssPoseGraphConfig::default()),
            Err(GnssPoseGraphError::NoFixedNode)
        );
    }
}
