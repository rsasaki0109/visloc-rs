//! `ort`-backed session wrapper around the four legacy ONNX graphs
//! `scripts/export_dpvo_onnx.py` produces (`fnet.onnx`, `inet.onnx`,
//! `dpvo_update_pre_agg.onnx`, `dpvo_update_post_agg.onnx` — see
//! `docs/dpvo_droid_port_plan.md`'s M1 results for the exact contract each
//! graph was exported with), plus [`DpvoOnnxSession::update_iteration`],
//! which stitches the pre-agg graph, the host-side [`SoftAgg`] step, and
//! the post-agg graph together into one full GRU update-cell call — the
//! "two-graph update split" the plan doc's M1 section describes. New model
//! bundles additionally contain `dpvo_update_full.onnx`, which keeps both
//! SoftAgg reductions inside one graph and is selected automatically.
//!
//! Mirrors [`crate::features::superpoint_onnx`]'s session-loading pattern
//! (builder + execution-provider selection + `Arc<Mutex<Session>>` sharing)
//! rather than reinventing one; see that module's doc comment for the
//! `load-dynamic`/`ORT_DYLIB_PATH` runtime-library setup this crate already
//! documents (unchanged here — same `ort` dependency, same feature gate).

use ndarray::{Array2, Array3, Array4, ArrayView3, ArrayView4, Axis};
use ort::session::Session;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::features::superpoint_onnx::OnnxBackend;

use super::softagg::{neighbors_cpu, SoftAgg};

/// Errors from the DPVO ONNX session wrapper.
#[derive(Debug, Clone, PartialEq)]
pub enum DpvoOnnxError {
    /// A model output had an unexpected shape, or a named output was
    /// missing entirely.
    OutputShapeMismatch {
        expected: &'static str,
        actual: String,
    },
    /// An input tensor's shape did not match what a graph requires (e.g.
    /// `net`/`inp`/`corr` disagreeing on the edge-count axis).
    InputShapeMismatch { message: String },
    /// Underlying ONNX Runtime error (stored as `String` to stay cheaply
    /// comparable, matching this crate's other ONNX wrappers).
    OnnxRuntime(String),
}

impl fmt::Display for DpvoOnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "DPVO ONNX output shape: expected {expected}, got {actual}"
                )
            }
            Self::InputShapeMismatch { message } => {
                write!(f, "DPVO ONNX input shape mismatch: {message}")
            }
            Self::OnnxRuntime(message) => write!(f, "DPVO ONNX Runtime error: {message}"),
        }
    }
}

impl std::error::Error for DpvoOnnxError {}

impl DpvoOnnxError {
    fn from_ort<E: std::fmt::Display>(error: E) -> Self {
        Self::OnnxRuntime(error.to_string())
    }
}

/// The GRU update cell's three outputs: `(net_out, delta, weight)`, shapes
/// `((1, E, 384), (1, E, 2), (1, E, 2))`. Named here (rather than left as
/// an inline tuple type) purely to give `run_update_post_agg` and
/// `update_iteration`'s signatures a readable return type.
pub type UpdateCellOutput = (Array3<f32>, Array3<f32>, Array3<f32>);

/// The four ONNX graphs `scripts/export_dpvo_onnx.py` writes into one
/// output directory, wrapped as four independent `ort` sessions (each
/// guarded by its own mutex, mirroring
/// [`crate::features::superpoint_onnx::SuperPointOnnxExtractor`]'s
/// single-session pattern applied per-graph). Cheaply `Clone`-able for the
/// same reason that type is: sharing one loaded set of sessions across,
/// e.g., stereo cam0/cam1 callers.
#[derive(Clone)]
pub struct DpvoOnnxSession {
    backend: OnnxBackend,
    fnet: Arc<Mutex<Session>>,
    inet: Arc<Mutex<Session>>,
    update_pre_agg: Arc<Mutex<Session>>,
    update_post_agg: Arc<Mutex<Session>>,
    update_full: Option<Arc<Mutex<Session>>>,
    correlation: Option<Arc<Mutex<Session>>>,
}

impl fmt::Debug for DpvoOnnxSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DpvoOnnxSession").finish_non_exhaustive()
    }
}

impl DpvoOnnxSession {
    /// Load all four graphs, preferring CUDA with a CPU fallback (see
    /// [`OnnxBackend::default`]).
    pub fn load_from_paths(
        fnet_path: impl AsRef<Path>,
        inet_path: impl AsRef<Path>,
        update_pre_agg_path: impl AsRef<Path>,
        update_post_agg_path: impl AsRef<Path>,
    ) -> Result<Self, DpvoOnnxError> {
        Self::load_from_paths_with_backend(
            fnet_path,
            inet_path,
            update_pre_agg_path,
            update_post_agg_path,
            OnnxBackend::default(),
        )
    }

    /// Like [`load_from_paths`](Self::load_from_paths) with an explicit
    /// execution-provider choice.
    pub fn load_from_paths_with_backend(
        fnet_path: impl AsRef<Path>,
        inet_path: impl AsRef<Path>,
        update_pre_agg_path: impl AsRef<Path>,
        update_post_agg_path: impl AsRef<Path>,
        backend: OnnxBackend,
    ) -> Result<Self, DpvoOnnxError> {
        let update_pre_agg_path = update_pre_agg_path.as_ref();
        let full_path = update_pre_agg_path.with_file_name("dpvo_update_full.onnx");
        let correlation_path = update_pre_agg_path.with_file_name("dpvo_corr_pyramid.onnx");
        Ok(Self {
            backend,
            fnet: Arc::new(Mutex::new(build_session(fnet_path.as_ref(), backend)?)),
            inet: Arc::new(Mutex::new(build_session(inet_path.as_ref(), backend)?)),
            update_pre_agg: Arc::new(Mutex::new(build_session(update_pre_agg_path, backend)?)),
            update_post_agg: Arc::new(Mutex::new(build_session(
                update_post_agg_path.as_ref(),
                backend,
            )?)),
            update_full: full_path
                .is_file()
                .then(|| build_session(&full_path, backend))
                .transpose()?
                .map(|session| Arc::new(Mutex::new(session))),
            correlation: correlation_path
                .is_file()
                .then(|| build_session(&correlation_path, backend))
                .transpose()?
                .map(|session| Arc::new(Mutex::new(session))),
        })
    }

    /// Whether this model bundle supplies the fused update graph.
    pub fn full_update_enabled(&self) -> bool {
        self.update_full.is_some()
    }

    pub fn correlation_graph_enabled(&self) -> bool {
        self.correlation.is_some()
    }

    pub fn run_correlation_pyramid(
        &self,
        anchor: ArrayView4<'_, f32>,
        target_level0: ArrayView4<'_, f32>,
        target_level1: ArrayView4<'_, f32>,
        coords_level0: ArrayView4<'_, f32>,
    ) -> Result<Array2<f32>, DpvoOnnxError> {
        let session = self.correlation.as_ref().ok_or_else(|| {
            DpvoOnnxError::OnnxRuntime("model bundle has no dpvo_corr_pyramid.onnx".into())
        })?;
        let tensor = |array: ArrayView4<'_, f32>| {
            ort::value::Tensor::from_array(array.to_owned()).map_err(DpvoOnnxError::from_ort)
        };
        let mut session = session.lock().map_err(lock_poisoned)?;
        let mut outputs = session
            .run(ort::inputs![
                tensor(anchor)?,
                tensor(target_level0)?,
                tensor(target_level1)?,
                tensor(coords_level0)?
            ])
            .map_err(DpvoOnnxError::from_ort)?;
        extract_array2(&mut outputs, "corr")
    }

    /// Run `fnet`: raw `[0, 255]`-range pixels in `(1, 3, H, W)`, matching
    /// feature map out `(1, 128, H/4, W/4)` (the graph already bakes in
    /// DPVO's own pre/post scaling — see the plan doc's M1 results, "What
    /// exported cleanly" — so no extra normalization belongs at this call
    /// site).
    pub fn run_fnet(&self, image: ArrayView4<'_, f32>) -> Result<Array4<f32>, DpvoOnnxError> {
        run_encoder(&self.fnet, image, "fmap")
    }

    /// Run `inet`: same input contract as [`run_fnet`](Self::run_fnet),
    /// `(1, 384, H/4, W/4)` context map out.
    pub fn run_inet(&self, image: ArrayView4<'_, f32>) -> Result<Array4<f32>, DpvoOnnxError> {
        run_encoder(&self.inet, image, "imap")
    }

    /// Run the independent feature and context encoders concurrently.
    ///
    /// The graphs consume the same immutable input and own separate ORT
    /// sessions, so there is no dependency that requires serial execution.
    /// Concurrency is limited to strict CUDA: CPU and fallback-capable
    /// sessions retain their legacy serial behavior because thread dispatch
    /// can cost more than it saves for small CPU inputs.
    pub fn run_encoders(
        &self,
        image: ArrayView4<'_, f32>,
    ) -> Result<(Array4<f32>, Array4<f32>), DpvoOnnxError> {
        if self.backend == OnnxBackend::Cuda {
            let (fmap, imap) = rayon::join(
                || self.run_fnet(image.view()),
                || self.run_inet(image.view()),
            );
            Ok((fmap?, imap?))
        } else {
            Ok((self.run_fnet(image.view())?, self.run_inet(image)?))
        }
    }

    /// Run `dpvo_update_pre_agg.onnx`: `Update.forward` up to (but
    /// excluding) the two `SoftAgg` calls.
    ///
    /// * `net`, `inp`: `(1, E, 384)`.
    /// * `corr`: `(1, E, 882)`.
    /// * `ix`, `jx`: `(E,)`, from [`neighbors_cpu`] (or supplied directly by
    ///   a caller that already has them, e.g. an integration test replaying
    ///   a fixture).
    ///
    /// Returns `net_pre_agg`, `(1, E, 384)`.
    pub fn run_update_pre_agg(
        &self,
        net: ArrayView3<'_, f32>,
        inp: ArrayView3<'_, f32>,
        corr: ArrayView3<'_, f32>,
        ix: &[i64],
        jx: &[i64],
    ) -> Result<Array3<f32>, DpvoOnnxError> {
        let num_edges = net.shape()[1];
        if inp.shape()[1] != num_edges || corr.shape()[1] != num_edges {
            return Err(DpvoOnnxError::InputShapeMismatch {
                message: format!(
                    "net/inp/corr disagree on num_edges: net={}, inp={}, corr={}",
                    num_edges,
                    inp.shape()[1],
                    corr.shape()[1]
                ),
            });
        }
        if ix.len() != num_edges || jx.len() != num_edges {
            return Err(DpvoOnnxError::InputShapeMismatch {
                message: format!(
                    "ix/jx length must match num_edges={num_edges}, got ix={}, jx={}",
                    ix.len(),
                    jx.len()
                ),
            });
        }

        let net_tensor =
            ort::value::Tensor::from_array(net.to_owned()).map_err(DpvoOnnxError::from_ort)?;
        let inp_tensor =
            ort::value::Tensor::from_array(inp.to_owned()).map_err(DpvoOnnxError::from_ort)?;
        let corr_tensor =
            ort::value::Tensor::from_array(corr.to_owned()).map_err(DpvoOnnxError::from_ort)?;
        let ix_tensor = ort::value::Tensor::from_array(ndarray::Array1::from_vec(ix.to_vec()))
            .map_err(DpvoOnnxError::from_ort)?;
        let jx_tensor = ort::value::Tensor::from_array(ndarray::Array1::from_vec(jx.to_vec()))
            .map_err(DpvoOnnxError::from_ort)?;

        let mut session = self.update_pre_agg.lock().map_err(lock_poisoned)?;
        // Positional binding follows the export's declared input order:
        // net, inp, corr, ix, jx (see this module doc's contract).
        let mut outputs = session
            .run(ort::inputs![
                net_tensor,
                inp_tensor,
                corr_tensor,
                ix_tensor,
                jx_tensor
            ])
            .map_err(DpvoOnnxError::from_ort)?;
        extract_array3(&mut outputs, "net_pre_agg")
    }

    /// Run `dpvo_update_post_agg.onnx`: the GRU block + `d`/`w` output
    /// heads. `net_post_agg`: `(1, E, 384)` (the host-side [`SoftAgg`]
    /// sum already folded in by the caller —
    /// see [`Self::update_iteration`] for the full assembly).
    ///
    /// Returns `(net_out, delta, weight)` = `((1,E,384), (1,E,2), (1,E,2))`.
    pub fn run_update_post_agg(
        &self,
        net_post_agg: ArrayView3<'_, f32>,
    ) -> Result<UpdateCellOutput, DpvoOnnxError> {
        let tensor = ort::value::Tensor::from_array(net_post_agg.to_owned())
            .map_err(DpvoOnnxError::from_ort)?;
        let mut session = self.update_post_agg.lock().map_err(lock_poisoned)?;
        let mut outputs = session
            .run(ort::inputs![tensor])
            .map_err(DpvoOnnxError::from_ort)?;
        let net_out = extract_array3(&mut outputs, "net_out")?;
        let delta = extract_array3(&mut outputs, "delta")?;
        let weight = extract_array3(&mut outputs, "weight")?;
        Ok((net_out, delta, weight))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_update_full(
        &self,
        net: ArrayView3<'_, f32>,
        inp: ArrayView3<'_, f32>,
        corr: ArrayView3<'_, f32>,
        ix: &[i64],
        jx: &[i64],
        group_kk: &[i64],
        group_ij: &[i64],
    ) -> Result<UpdateCellOutput, DpvoOnnxError> {
        let session = self
            .update_full
            .as_ref()
            .expect("run_update_full is only called when the fused graph exists");
        let net_tensor =
            ort::value::Tensor::from_array(net.to_owned()).map_err(DpvoOnnxError::from_ort)?;
        let inp_tensor =
            ort::value::Tensor::from_array(inp.to_owned()).map_err(DpvoOnnxError::from_ort)?;
        let corr_tensor =
            ort::value::Tensor::from_array(corr.to_owned()).map_err(DpvoOnnxError::from_ort)?;
        let vector = |values: &[i64]| {
            ort::value::Tensor::from_array(ndarray::Array1::from_vec(values.to_vec()))
                .map_err(DpvoOnnxError::from_ort)
        };
        let mut session = session.lock().map_err(lock_poisoned)?;
        let mut outputs = session
            .run(ort::inputs![
                net_tensor,
                inp_tensor,
                corr_tensor,
                vector(ix)?,
                vector(jx)?,
                vector(group_kk)?,
                vector(group_ij)?
            ])
            .map_err(DpvoOnnxError::from_ort)?;
        let net_out = extract_array3(&mut outputs, "net_out")?;
        let delta = extract_array3(&mut outputs, "delta")?;
        let weight = extract_array3(&mut outputs, "weight")?;
        Ok((net_out, delta, weight))
    }

    /// One full GRU update-cell iteration: `dpvo_update_pre_agg.onnx` →
    /// host-side `SoftAgg` (both `agg_kk` and `agg_ij`) →
    /// `dpvo_update_post_agg.onnx`. This is the "two-graph update split"
    /// the plan doc's M1 results describe — see that section for why the
    /// SoftAgg step cannot itself be an ONNX graph.
    ///
    /// * `net`, `inp`: `(1, E, 384)`; `corr`: `(1, E, 882)`.
    /// * `kk`, `ii`, `jj`: `(E,)` each — patch id / source frame / target
    ///   frame per edge, i.e. exactly DPVO's own edge-graph triple.
    /// * `agg_kk`, `agg_ij`: the two trained `SoftAgg` instances (see
    ///   [`SoftAgg::load_from_npz`] for how to obtain their weights until a
    ///   production (non-fixture) weight-export path exists — see the
    ///   `dpvo` module doc for that gap).
    ///
    /// Returns `(net_out, delta, weight)`; `net_out` is the new hidden
    /// state to feed back in as `net` on the next iteration (steady state:
    /// one iteration per incoming frame; the ~8-frame initialization burst
    /// calls this 12× — see the plan doc §1.2 — both are ordinary Rust
    /// `for` loops over this one method, per the plan doc §3's "one call =
    /// one static graph" finding).
    #[allow(clippy::too_many_arguments)]
    pub fn update_iteration(
        &self,
        net: ArrayView3<'_, f32>,
        inp: ArrayView3<'_, f32>,
        corr: ArrayView3<'_, f32>,
        kk: &[i64],
        ii: &[i64],
        jj: &[i64],
        agg_kk: &SoftAgg,
        agg_ij: &SoftAgg,
    ) -> Result<UpdateCellOutput, DpvoOnnxError> {
        let (ix, jx) = neighbors_cpu(kk, jj);
        if self.update_full.is_some() {
            let group_kk = compact_group_ids(kk);
            let pair_key: Vec<i64> = ii
                .iter()
                .zip(jj.iter())
                .map(|(&i, &j)| i * 12345 + j)
                .collect();
            let group_ij = compact_group_ids(&pair_key);
            return self.run_update_full(net, inp, corr, &ix, &jx, &group_kk, &group_ij);
        }
        let net_pre_agg = self.run_update_pre_agg(net, inp, corr, &ix, &jx)?;

        // Squeeze the batch dim (always 1) to feed `SoftAgg`'s 2-D
        // `(num_edges, dim)` contract, then restore it afterwards — the
        // ONNX graphs and `SoftAgg` disagree on this by convention (the
        // exported graphs keep PyTorch's batch axis; `SoftAgg` here works
        // in plain `(edges, dim)` since DPVO never batches beyond 1 frame
        // pair at a time).
        let net_pre_agg_2d = net_pre_agg.index_axis(Axis(0), 0).to_owned();

        let agg_kk_out = agg_kk.forward(net_pre_agg_2d.view(), kk);
        let pair_key: Vec<i64> = ii
            .iter()
            .zip(jj.iter())
            .map(|(&i, &j)| i * 12345 + j)
            .collect();
        let agg_ij_out = agg_ij.forward(net_pre_agg_2d.view(), &pair_key);

        let net_post_agg_2d = &net_pre_agg_2d + &agg_kk_out + &agg_ij_out;
        let net_post_agg = net_post_agg_2d.insert_axis(Axis(0));

        self.run_update_post_agg(net_post_agg.view())
    }
}

fn compact_group_ids(keys: &[i64]) -> Vec<i64> {
    let mut groups = HashMap::new();
    let mut next = 0_i64;
    keys.iter()
        .map(|&key| {
            *groups.entry(key).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            })
        })
        .collect()
}

fn build_session(path: &Path, backend: OnnxBackend) -> Result<Session, DpvoOnnxError> {
    use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};

    let providers = match backend {
        OnnxBackend::CudaThenCpu => vec![
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ],
        OnnxBackend::Cuda => vec![CUDAExecutionProvider::default().build().error_on_failure()],
        OnnxBackend::Cpu => vec![CPUExecutionProvider::default().build()],
    };

    Session::builder()
        .map_err(DpvoOnnxError::from_ort)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(DpvoOnnxError::from_ort)?
        .with_execution_providers(providers)
        .map_err(DpvoOnnxError::from_ort)?
        .commit_from_file(path)
        .map_err(DpvoOnnxError::from_ort)
}

fn run_encoder(
    session: &Mutex<Session>,
    image: ArrayView4<'_, f32>,
    output_name: &str,
) -> Result<Array4<f32>, DpvoOnnxError> {
    let tensor =
        ort::value::Tensor::from_array(image.to_owned()).map_err(DpvoOnnxError::from_ort)?;
    let mut session = session.lock().map_err(lock_poisoned)?;
    let mut outputs = session
        .run(ort::inputs![tensor])
        .map_err(DpvoOnnxError::from_ort)?;
    extract_array4(&mut outputs, output_name)
}

fn extract_array3(
    outputs: &mut ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Array3<f32>, DpvoOnnxError> {
    let value = outputs
        .remove(name)
        .ok_or_else(|| DpvoOnnxError::OutputShapeMismatch {
            expected: "an output present in the session's declared outputs",
            actual: format!("missing output named `{name}`"),
        })?;
    let array = value
        .try_extract_array::<f32>()
        .map_err(DpvoOnnxError::from_ort)?;
    array
        .into_owned()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| DpvoOnnxError::OutputShapeMismatch {
            expected: "a 3-D output",
            actual: e.to_string(),
        })
}

fn extract_array2(
    outputs: &mut ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Array2<f32>, DpvoOnnxError> {
    let value = outputs
        .remove(name)
        .ok_or_else(|| DpvoOnnxError::OutputShapeMismatch {
            expected: "an output present in the session's declared outputs",
            actual: format!("missing output named `{name}`"),
        })?;
    let array = value
        .try_extract_array::<f32>()
        .map_err(DpvoOnnxError::from_ort)?;
    array
        .into_owned()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| DpvoOnnxError::OutputShapeMismatch {
            expected: "a 2-D output",
            actual: e.to_string(),
        })
}

fn extract_array4(
    outputs: &mut ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Array4<f32>, DpvoOnnxError> {
    let value = outputs
        .remove(name)
        .ok_or_else(|| DpvoOnnxError::OutputShapeMismatch {
            expected: "an output present in the session's declared outputs",
            actual: format!("missing output named `{name}`"),
        })?;
    let array = value
        .try_extract_array::<f32>()
        .map_err(DpvoOnnxError::from_ort)?;
    array
        .into_owned()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| DpvoOnnxError::OutputShapeMismatch {
            expected: "a 4-D output",
            actual: e.to_string(),
        })
}

fn lock_poisoned<T>(error: std::sync::PoisonError<T>) -> DpvoOnnxError {
    DpvoOnnxError::OnnxRuntime(format!("session mutex poisoned: {error}"))
}

#[cfg(test)]
mod tests {
    use super::compact_group_ids;

    #[test]
    fn compact_groups_preserve_equality_and_first_occurrence_order() {
        assert_eq!(compact_group_ids(&[9, 3, 9, 7, 3]), [0, 1, 0, 2, 1]);
        assert!(compact_group_ids(&[]).is_empty());
    }
}
