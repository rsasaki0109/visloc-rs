//! Rust port of the two pieces of DPVO's `Update.forward` that resisted
//! ONNX export: `fastba.neighbors` (edge-neighbour bookkeeping) and
//! `SoftAgg` (grouped softmax-aggregation). Ported from
//! `scripts/export_dpvo_onnx.py`'s `neighbors_cpu` / `SoftAggReference`
//! (themselves faithful reimplementations of upstream's CUDA
//! `fastba::neighbors` and `torch_scatter.scatter_softmax`/`scatter_sum`,
//! per that script's own module doc — see `docs/dpvo_droid_port_plan.md`'s
//! M1 results, "What stayed host-side, and why", for the full rationale).
//!
//! Implemented with `ndarray` (already a workspace dependency behind this
//! crate's `onnx-inference` feature, and the natural type for the
//! `(num_edges, DIM)` tensors this module trades with the ONNX session
//! wrapper) rather than a from-scratch `nalgebra`-only version — `nalgebra`
//! has no ergonomic equivalent for the ragged, dynamically-shaped
//! `(num_groups, DIM)` scatter targets this needs, and introducing a second
//! tensor type here would only add conversions at every call site.

use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;
use std::fmt;

use super::npz::{NpzArchive, NpzError};

/// One `Linear(in_dim, out_dim)` layer's weights, PyTorch convention:
/// `weight` is `(out_dim, in_dim)` and the forward pass is
/// `y = x @ weight.T + bias`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearWeights {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
}

impl LinearWeights {
    /// `x`: `(n, in_dim)` → `(n, out_dim)`.
    pub fn apply(&self, x: ArrayView2<'_, f32>) -> Array2<f32> {
        let mut y = x.dot(&self.weight.t());
        y += &self.bias;
        y
    }

    fn in_out_dim(&self) -> (usize, usize) {
        (self.weight.shape()[1], self.weight.shape()[0])
    }
}

/// Errors constructing/loading a [`SoftAgg`].
#[derive(Debug, Clone, PartialEq)]
pub enum SoftAggError {
    Npz(NpzError),
    ShapeMismatch {
        field: &'static str,
        message: String,
    },
}

impl fmt::Display for SoftAggError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Npz(inner) => write!(f, "SoftAgg weight load failed: {inner}"),
            Self::ShapeMismatch { field, message } => {
                write!(f, "SoftAgg weight shape mismatch in {field}: {message}")
            }
        }
    }
}

impl std::error::Error for SoftAggError {}

impl From<NpzError> for SoftAggError {
    fn from(value: NpzError) -> Self {
        Self::Npz(value)
    }
}

/// Rust port of `dpvo.blocks.SoftAgg` (upstream: `torch_scatter.scatter_softmax`
/// and `scatter_sum`) via `export_dpvo_onnx.py`'s `SoftAggReference` — see
/// that class's own doc comment for the "verified equivalent by
/// construction, not cross-checked against `torch_scatter` itself" caveat,
/// which applies here too.
///
/// `Update.forward` always calls `SoftAgg` with its default `expand=True`
/// (see `net.py`: `self.agg_kk(net, kk)`, `self.agg_ij(net, ii*12345+jj)`),
/// so only that path is ported — the `expand=False` (per-group, not
/// per-edge, output) branch is not needed and not implemented.
///
/// # Why group-*label* order does not matter here
///
/// The Python reference assigns group ids via
/// `torch.unique(index, return_inverse=True)`, which happens to sort unique
/// values. This implementation instead assigns group ids in first-seen
/// order via a `HashMap`. That is safe *only* because `expand=True` reads
/// every edge's result back from its own group (`h(y)[:, jx]`) — the
/// softmax/sum math is invariant to which arbitrary integer label a group
/// is assigned, as long as the same edge→group partition is used
/// consistently for the max, the sum, and the read-back. This lets the
/// implementation skip reproducing `torch.unique`'s specific sort order,
/// which would otherwise require an unnecessary dependency on a stable
/// total order over the (possibly negative, possibly huge —
/// `ii*12345+jj`) index values.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftAgg {
    pub f: LinearWeights,
    pub g: LinearWeights,
    pub h: LinearWeights,
}

impl SoftAgg {
    /// Construct directly from already-loaded weights, validating that all
    /// three layers share one `dim` (`Linear(dim, dim)`), matching
    /// `SoftAggReference.__init__(self, dim=DIM, expand=True)`.
    pub fn new(f: LinearWeights, g: LinearWeights, h: LinearWeights) -> Result<Self, SoftAggError> {
        let dims = [f.in_out_dim(), g.in_out_dim(), h.in_out_dim()];
        let (in0, out0) = dims[0];
        if in0 != out0 || dims.iter().any(|&(i, o)| i != in0 || o != out0) {
            return Err(SoftAggError::ShapeMismatch {
                field: "f/g/h",
                message: format!(
                    "expected all three Linear(dim, dim) layers to share one square dim, got {dims:?}"
                ),
            });
        }
        Ok(Self { f, g, h })
    }

    /// Load `f`/`g`/`h` weights from an `.npz` archive whose entries are
    /// named `{prefix}f_weight`, `{prefix}f_bias`, `{prefix}g_weight`, ...
    /// — the convention `scripts/export_dpvo_onnx.py`'s
    /// `dump_softagg_weights_fixture` writes (e.g. `prefix = "agg_kk_"` or
    /// `"agg_ij_"` against `fixtures/softagg_weights_fixture.npz`; see the
    /// `dpvo` module doc for why that fixture had to be added in M2).
    pub fn load_from_npz(archive: &NpzArchive, prefix: &str) -> Result<Self, SoftAggError> {
        let load_linear = |name: &str| -> Result<LinearWeights, SoftAggError> {
            let (w_shape, w_data) = archive.read_f32(&format!("{prefix}{name}_weight"))?;
            let (b_shape, b_data) = archive.read_f32(&format!("{prefix}{name}_bias"))?;
            if w_shape.len() != 2 {
                return Err(SoftAggError::ShapeMismatch {
                    field: "weight",
                    message: format!("expected a 2-D weight, got shape {w_shape:?}"),
                });
            }
            if b_shape.len() != 1 || b_shape[0] != w_shape[0] {
                return Err(SoftAggError::ShapeMismatch {
                    field: "bias",
                    message: format!(
                        "expected a 1-D bias matching weight's out_dim {}, got shape {b_shape:?}",
                        w_shape[0]
                    ),
                });
            }
            let weight = Array2::from_shape_vec((w_shape[0], w_shape[1]), w_data).map_err(|e| {
                SoftAggError::ShapeMismatch {
                    field: "weight",
                    message: e.to_string(),
                }
            })?;
            let bias = Array1::from_vec(b_data);
            Ok(LinearWeights { weight, bias })
        };
        Self::new(load_linear("f")?, load_linear("g")?, load_linear("h")?)
    }

    /// The GRU update cell's `agg_kk`/`agg_ij` calls, e.g.
    /// `net = net + self.agg_kk(net, kk)` — this method returns just the
    /// SoftAgg term itself (`self.agg_kk(net, kk)`), not the residual add;
    /// callers add it in, matching `net.py`'s own `net = net + ...` lines.
    ///
    /// * `x`: `(num_edges, dim)` per-edge features (`net_pre_agg`, squeezed
    ///   out of its batch dimension by the caller).
    /// * `group_key`: `(num_edges,)` integer group id per edge. `agg_kk`
    ///   passes patch ids (`kk`); `agg_ij` passes `ii*12345 + jj` (see
    ///   `net.py`) — computing that composite key is the caller's job, not
    ///   this method's, since it is a one-liner the caller already has both
    ///   operands for.
    ///
    /// Returns `(num_edges, dim)`, one aggregated-then-broadcast-back value
    /// per edge (the `expand=True` behaviour described above).
    pub fn forward(&self, x: ArrayView2<'_, f32>, group_key: &[i64]) -> Array2<f32> {
        let num_edges = x.nrows();
        let dim = x.ncols();
        debug_assert_eq!(group_key.len(), num_edges);

        if num_edges == 0 {
            return Array2::zeros((0, dim));
        }

        // First-seen-order group assignment — see the struct doc for why
        // the specific label order is immaterial to the result.
        let mut group_of: HashMap<i64, usize> = HashMap::new();
        let group_ids: Vec<usize> = group_key
            .iter()
            .map(|&key| {
                let next_id = group_of.len();
                *group_of.entry(key).or_insert(next_id)
            })
            .collect();
        let num_groups = group_of.len();

        let g_x = self.g.apply(x);
        let f_x = self.f.apply(x);

        // Numerically-stable segment-softmax: per-group, per-channel max,
        // matching `scatter_reduce_(..., reduce="amax", include_self=True)`
        // against a `-inf`-initialised accumulator (every referenced group
        // has ≥ 1 member by construction, so `-inf` never survives into the
        // final `group_max`).
        let mut group_max = Array2::<f32>::from_elem((num_groups, dim), f32::NEG_INFINITY);
        for (edge, &gid) in group_ids.iter().enumerate() {
            for c in 0..dim {
                let v = g_x[(edge, c)];
                if v > group_max[(gid, c)] {
                    group_max[(gid, c)] = v;
                }
            }
        }

        let mut exp_x = Array2::<f32>::zeros((num_edges, dim));
        for (edge, &gid) in group_ids.iter().enumerate() {
            for c in 0..dim {
                exp_x[(edge, c)] = (g_x[(edge, c)] - group_max[(gid, c)]).exp();
            }
        }

        let mut group_sum = Array2::<f32>::zeros((num_groups, dim));
        for (edge, &gid) in group_ids.iter().enumerate() {
            for c in 0..dim {
                group_sum[(gid, c)] += exp_x[(edge, c)];
            }
        }

        // Weighted sum of `f(x)` within each group: `y = scatter_sum(f(x) * softmax_weight, group)`.
        let mut y = Array2::<f32>::zeros((num_groups, dim));
        for (edge, &gid) in group_ids.iter().enumerate() {
            for c in 0..dim {
                let weight = exp_x[(edge, c)] / group_sum[(gid, c)];
                y[(gid, c)] += f_x[(edge, c)] * weight;
            }
        }

        let y_h = self.h.apply(y.view());

        // `expand=True`: broadcast each group's aggregate back to every
        // edge in that group (`h(y)[:, jx]` in the Python reference).
        let mut out = Array2::<f32>::zeros((num_edges, dim));
        for (edge, &gid) in group_ids.iter().enumerate() {
            out.row_mut(edge).assign(&y_h.row(gid));
        }
        out
    }
}

/// Rust port of `fastba/ba.cpp`'s `neighbors()`, via `export_dpvo_onnx.py`'s
/// `neighbors_cpu` (a faithful pure-Python/CPU reimplementation of that CUDA
/// kernel's integer bookkeeping — see the plan doc's M1 results). Groups
/// edge indices by `kk` (patch id), stable-sorts each group by `jj` (target
/// frame), and returns each edge's previous/next sibling index within its
/// own group, or `-1` if it is first/last.
///
/// `net.py`'s `Update.forward` calls this as `ix, jx = fastba.neighbors(kk,
/// jj)`; `ix`/`jx` then feed `dpvo_update_pre_agg.onnx` as ordinary integer
/// input tensors (see the plan doc's exported-graph contract) — this
/// function's whole job is to produce those two tensors, nothing else.
///
/// Returns `(ix, jx)`, each length `kk.len()`.
pub fn neighbors_cpu(kk: &[i64], jj: &[i64]) -> (Vec<i64>, Vec<i64>) {
    debug_assert_eq!(kk.len(), jj.len());
    let n = kk.len();

    let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
    for (edge, &key) in kk.iter().enumerate() {
        groups.entry(key).or_default().push(edge);
    }

    let mut ix = vec![-1i64; n];
    let mut jx = vec![-1i64; n];
    for members in groups.values_mut() {
        // `Vec::sort_by_key` is a stable sort, matching `list.sort` in the
        // Python reference (itself documented there as matching
        // `std::stable_sort`).
        members.sort_by_key(|&edge| jj[edge]);
        for (position, &edge) in members.iter().enumerate() {
            ix[edge] = if position > 0 {
                members[position - 1] as i64
            } else {
                -1
            };
            jx[edge] = if position + 1 < members.len() {
                members[position + 1] as i64
            } else {
                -1
            };
        }
    }
    (ix, jx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Hand-worked example matching `export_dpvo_onnx.py`'s own module doc:
    /// patch ids `kk = [0, 0, 1, 1]`, per-edge target frames
    /// `jj = [2, 1, 2, 1]` (deliberately out of order within each group, so
    /// the by-`jj` sort inside each `kk` group actually does something).
    ///
    /// Group `kk=0` has edges `{0, 1}` with `jj = {2, 1}`; sorted by `jj`
    /// gives order `[edge 1 (jj=1), edge 0 (jj=2)]`. So edge 1 is first in
    /// its group (`ix=-1`, `jx=0`) and edge 0 is last (`ix=1`, `jx=-1`).
    /// Symmetric reasoning for group `kk=1` (edges `{2, 3}`, `jj = {2,1}`).
    #[test]
    fn neighbors_cpu_matches_hand_worked_example() {
        let kk = vec![0, 0, 1, 1];
        let jj = vec![2, 1, 2, 1];
        let (ix, jx) = neighbors_cpu(&kk, &jj);
        assert_eq!(ix, vec![1, -1, 3, -1]);
        assert_eq!(jx, vec![-1, 0, -1, 2]);
    }

    #[test]
    fn neighbors_cpu_singleton_groups_have_no_neighbours() {
        let kk = vec![0, 1, 2];
        let jj = vec![5, 5, 5];
        let (ix, jx) = neighbors_cpu(&kk, &jj);
        assert_eq!(ix, vec![-1, -1, -1]);
        assert_eq!(jx, vec![-1, -1, -1]);
    }

    #[test]
    fn neighbors_cpu_handles_empty_input() {
        let (ix, jx) = neighbors_cpu(&[], &[]);
        assert!(ix.is_empty() && jx.is_empty());
    }

    /// Ties within a group (equal `jj`) must break by original edge order
    /// (stable sort), matching Python's guaranteed-stable `list.sort`.
    #[test]
    fn neighbors_cpu_breaks_jj_ties_by_original_order() {
        let kk = vec![0, 0, 0];
        let jj = vec![5, 5, 5]; // all tied
        let (ix, jx) = neighbors_cpu(&kk, &jj);
        // Stable sort on equal keys preserves original relative order:
        // edge 0, then 1, then 2.
        assert_eq!(ix, vec![-1, 0, 1]);
        assert_eq!(jx, vec![1, 2, -1]);
    }

    fn identity_linear(dim: usize) -> LinearWeights {
        LinearWeights {
            weight: Array2::eye(dim),
            bias: Array1::zeros(dim),
        }
    }

    /// With `f = g = identity, h = identity`, `SoftAgg` reduces to a plain
    /// per-channel grouped softmax-weighted average of `x` itself — a case
    /// simple enough to check by hand. Two groups of sizes {2, 1}, 1 channel.
    #[test]
    fn softagg_forward_reduces_to_hand_computable_softmax_average_with_identity_weights() {
        let dim = 1;
        let agg = SoftAgg::new(
            identity_linear(dim),
            identity_linear(dim),
            identity_linear(dim),
        )
        .unwrap();
        // x = [[0.0], [1.0], [5.0]], groups = [0, 0, 1].
        let x = array![[0.0_f32], [1.0], [5.0]];
        let group_key = vec![0_i64, 0, 1];
        let out = agg.forward(x.view(), &group_key);

        // Group 0: edges 0,1 with g(x) = x = [0, 1]. Softmax over [0,1]:
        // shift by max=1 -> exp([-1,0]) = [0.3678794, 1.0], sum=1.3678794,
        // weights = [0.2689414, 0.7310586]. f(x) = x, so
        // y_group0 = 0.2689414*0 + 0.7310586*1 = 0.7310586.
        let expected_group0 = 0.7310586_f32;
        assert!(
            (out[(0, 0)] - expected_group0).abs() < 1e-5,
            "out[0]={}",
            out[(0, 0)]
        );
        assert!(
            (out[(1, 0)] - expected_group0).abs() < 1e-5,
            "out[1]={}",
            out[(1, 0)]
        );

        // Group 1: a single-member group; softmax over one element is
        // always 1.0, so the aggregate is exactly that element's own
        // value: y_group1 = 5.0.
        assert!((out[(2, 0)] - 5.0).abs() < 1e-5, "out[2]={}", out[(2, 0)]);
    }

    #[test]
    fn softagg_forward_handles_empty_input() {
        let dim = 4;
        let agg = SoftAgg::new(
            identity_linear(dim),
            identity_linear(dim),
            identity_linear(dim),
        )
        .unwrap();
        let x = Array2::<f32>::zeros((0, dim));
        let out = agg.forward(x.view(), &[]);
        assert_eq!(out.shape(), &[0, dim]);
    }

    #[test]
    fn softagg_new_rejects_mismatched_layer_dims() {
        let dim4 = identity_linear(4);
        let dim8 = identity_linear(8);
        let result = SoftAgg::new(dim4.clone(), dim8, dim4);
        assert!(matches!(result, Err(SoftAggError::ShapeMismatch { .. })));
    }
}
