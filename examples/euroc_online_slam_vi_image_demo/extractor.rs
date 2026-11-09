//! The runtime-dispatched feature extractor (`DemoExtractor`), including the
//! offline SuperPoint descriptor replay.

use super::*;

/// Runtime-dispatched feature extractor. The demo's default
/// `CornerFeatureExtractor` produces a small raw-pixel patch descriptor;
/// the alternative `HogLikeFeatureExtractor` produces a unit-norm
/// 128-D HOG/SIFT-style descriptor with optionally oriented bins,
/// raising the descriptor signal-to-noise at the cost of ~4× CPU per
/// keypoint. `SuperPointOffline` replays pre-exported SuperPoint
/// 256-D descriptors from disk (produced by
/// `scripts/export_superpoint_lightglue.py --mono-dir …`), so the
/// demo can A/B against a true deep descriptor without dragging an
/// ONNX runtime into the workspace. The error types differ across
/// the three backing extractors, so the enum unifies them through
/// `String`.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone)]
pub(super) enum DemoExtractor {
    Corner(CornerFeatureExtractor),
    Hog(HogLikeFeatureExtractor),
    SuperPointOffline(SuperPointOfflineExtractor),
    /// Phase-27: in-Rust SuperPoint ONNX inference. The wrapped
    /// extractor's runtime behaviour depends on whether
    /// `visloc-vision` was built with `--features onnx-inference`:
    /// without the feature `extract()` returns
    /// `FeatureDisabled` and the demo aborts at the first frame
    /// with a clear error message pointing the operator at the
    /// feature flag and the model path.
    SuperPointOnnx(visloc_vision::features::superpoint_onnx::SuperPointOnnxExtractor),
    /// Basalt-faithful frame-to-frame KLT; track ids become descriptors.
    OpticalFlow(visloc_rs::vision::optical_flow::OpticalFlowFeatureExtractor),
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SuperPointOnnxBackendArg {
    CudaThenCpu,
    Cuda,
    Cpu,
}

#[cfg(feature = "image-io")]
impl SuperPointOnnxBackendArg {
    pub(super) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cuda-then-cpu" => Ok(Self::CudaThenCpu),
            "cuda" => Ok(Self::Cuda),
            "cpu" => Ok(Self::Cpu),
            other => Err(format!(
                "--superpoint-onnx-backend: expected 'cuda-then-cpu', 'cuda', or 'cpu', got {other:?}"
            )),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::CudaThenCpu => "cuda-then-cpu",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

#[cfg(feature = "image-io")]
impl FeatureExtractor for DemoExtractor {
    type Image = GrayscaleImage;
    type Error = String;

    fn extract(&self, image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        use visloc_vision::features::deep::DeepFeatureExtractor;
        match self {
            DemoExtractor::Corner(e) => e.extract(image).map_err(|err| err.to_string()),
            DemoExtractor::Hog(e) => e.extract(image).map_err(|err| err.to_string()),
            DemoExtractor::SuperPointOffline(e) => e.extract(image),
            DemoExtractor::SuperPointOnnx(e) => e
                .extract_deep(image)
                .map(|deep| deep.into_feature_set())
                .map_err(|err| err.to_string()),
            DemoExtractor::OpticalFlow(e) => e.extract(image),
        }
    }
}

#[cfg(feature = "image-io")]
impl DemoExtractor {
    /// Set the index of the next-to-replay frame for the offline
    /// extractor. A no-op for `Corner` / `Hog` (which derive features
    /// from the supplied image on the fly). Call this before every
    /// `extract()` when running the offline path so the replay stays
    /// in sync with the EuRoC frame stream — the offline extractor
    /// returns the preloaded features at the configured index instead
    /// of running an extractor on the image.
    pub(super) fn set_frame_idx(&self, frame_idx: usize) {
        if let DemoExtractor::SuperPointOffline(e) = self {
            e.set_frame_idx(frame_idx);
        }
    }

    /// Set the active camera for the offline extractor. A no-op for
    /// `Corner` / `Hog`. Used by the stereo-bootstrap path to switch
    /// from cam0 (loop default) to cam1 for the one-shot cam1
    /// `extract()` call.
    pub(super) fn set_camera(&self, camera: SuperPointCamera) {
        if let DemoExtractor::SuperPointOffline(e) = self {
            e.set_camera(camera);
        }
    }
}

/// Camera selector for [`SuperPointOfflineExtractor`]. The demo loop
/// runs the cam0 path; the stereo bootstrap path makes a single
/// cam1 `extract()` call at the seed frame.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SuperPointCamera {
    Cam0,
    Cam1,
}

/// Offline SuperPoint descriptor replay. Pre-loads `frame_NNNNNN_features.txt`
/// files emitted by the Python `scripts/export_superpoint_lightglue.py
/// --mono-dir <dir>` helper for one or both cameras into in-memory
/// `Vec<FeatureSet>` tables at construction, then returns the indexed
/// entry on every `extract()` call. The replay index and active camera
/// are set explicitly by the demo via [`DemoExtractor::set_frame_idx`]
/// and [`DemoExtractor::set_camera`] before each `extract()` — auto-
/// increment would race against the stereo-bootstrap path (which makes
/// a single cam1 call at the seed frame), so the explicit setters are
/// mandatory.
///
/// When cam1 features are loaded, `--stereo-bootstrap` is permitted:
/// the seed-frame cam0 and cam1 SuperPoint feature sets are fed into
/// the existing stereo-bootstrap path so triangulated metric-depth
/// landmarks replace the fixed `bootstrap_depth` back-projection. This
/// is the Phase-17 lever that addresses the Phase-15 finding that
/// fixed 4 m depth is the bench's binding constraint for SuperPoint.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone)]
pub(super) struct SuperPointOfflineExtractor {
    cam0_frames: std::sync::Arc<Vec<FeatureSet>>,
    cam1_frames: Option<std::sync::Arc<Vec<FeatureSet>>>,
    current_idx: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// `0` = Cam0, `1` = Cam1. Atomics over enums require a small
    /// integer round-trip; `set_camera` / extract decode this.
    current_camera: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(feature = "image-io")]
fn load_frame_features(dir: &std::path::Path) -> Result<Vec<FeatureSet>, String> {
    let mut entries: Vec<(usize, std::path::PathBuf)> = Vec::new();
    let read_dir = std::fs::read_dir(dir).map_err(|e| {
        format!(
            "SuperPointOfflineExtractor: read_dir {}: {e}",
            dir.display()
        )
    })?;
    for entry in read_dir {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        const PREFIX: &str = "frame_";
        const SUFFIX: &str = "_features.txt";
        if !name.starts_with(PREFIX) || !name.ends_with(SUFFIX) {
            continue;
        }
        let middle = &name[PREFIX.len()..name.len() - SUFFIX.len()];
        let Ok(idx) = middle.parse::<usize>() else {
            continue;
        };
        entries.push((idx, path));
    }
    entries.sort_by_key(|(idx, _)| *idx);
    if entries.is_empty() {
        return Err(format!(
            "SuperPointOfflineExtractor: no frame_*_features.txt files found in {}",
            dir.display(),
        ));
    }
    for (i, (idx, path)) in entries.iter().enumerate() {
        if *idx != i {
            return Err(format!(
                "SuperPointOfflineExtractor: feature files must be contiguous starting at frame_000000; gap at slot {i}, file {}",
                path.display(),
            ));
        }
    }
    let mut frames = Vec::with_capacity(entries.len());
    for (_idx, path) in &entries {
        let ext_set = read_external_deep_features_txt(path)
            .map_err(|e| format!("SuperPointOfflineExtractor: read {}: {e}", path.display()))?;
        let fs = ext_set.to_feature_set().map_err(|e| {
            format!(
                "SuperPointOfflineExtractor: to_feature_set {}: {e}",
                path.display()
            )
        })?;
        frames.push(fs);
    }
    Ok(frames)
}

#[cfg(feature = "image-io")]
impl SuperPointOfflineExtractor {
    pub(super) fn load_from_dir(cam0_dir: &std::path::Path) -> Result<Self, String> {
        let cam0_frames = load_frame_features(cam0_dir)?;
        Ok(Self {
            cam0_frames: std::sync::Arc::new(cam0_frames),
            cam1_frames: None,
            current_idx: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            current_camera: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    pub(super) fn load_with_cam1(
        cam0_dir: &std::path::Path,
        cam1_dir: &std::path::Path,
    ) -> Result<Self, String> {
        let mut cam0_frames = load_frame_features(cam0_dir)?;
        let mut cam1_frames = load_frame_features(cam1_dir)?;
        // Allow partially-exported cam1 directories (e.g. during incremental export):
        // truncate both to the shorter length so the caller can use max_frames to pick a
        // window within the available frames, rather than requiring exact equality.
        if cam0_frames.len() != cam1_frames.len() {
            let min_len = cam0_frames.len().min(cam1_frames.len());
            eprintln!(
                "SuperPointOfflineExtractor: cam0 ({}) and cam1 ({}) feature counts differ; \
                 truncating both to {} (use --max-frames ≤ {} to run within available frames)",
                cam0_frames.len(),
                cam1_frames.len(),
                min_len,
                min_len,
            );
            cam0_frames.truncate(min_len);
            cam1_frames.truncate(min_len);
        }
        Ok(Self {
            cam0_frames: std::sync::Arc::new(cam0_frames),
            cam1_frames: Some(std::sync::Arc::new(cam1_frames)),
            current_idx: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            current_camera: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    fn set_frame_idx(&self, idx: usize) {
        self.current_idx
            .store(idx, std::sync::atomic::Ordering::SeqCst);
    }

    fn set_camera(&self, camera: SuperPointCamera) {
        let value = match camera {
            SuperPointCamera::Cam0 => 0,
            SuperPointCamera::Cam1 => 1,
        };
        self.current_camera
            .store(value, std::sync::atomic::Ordering::SeqCst);
    }

    pub(super) fn has_cam1(&self) -> bool {
        self.cam1_frames.is_some()
    }

    pub(super) fn len(&self) -> usize {
        self.cam0_frames.len()
    }
}

#[cfg(feature = "image-io")]
impl FeatureExtractor for SuperPointOfflineExtractor {
    type Image = GrayscaleImage;
    type Error = String;

    fn extract(&self, _image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        let idx = self.current_idx.load(std::sync::atomic::Ordering::SeqCst);
        let cam = self
            .current_camera
            .load(std::sync::atomic::Ordering::SeqCst);
        let frames = match cam {
            0 => &self.cam0_frames,
            1 => self.cam1_frames.as_ref().ok_or_else(|| {
                "SuperPointOfflineExtractor: cam1 features not loaded; pass --superpoint-cam1-features-dir to enable stereo bootstrap".to_string()
            })?,
            other => return Err(format!("SuperPointOfflineExtractor: invalid camera id {other}")),
        };
        if idx >= frames.len() {
            return Err(format!(
                "SuperPointOfflineExtractor: requested frame_idx {idx} for cam{cam} but only {} frames preloaded; \
                 re-export with --frames covering the demo's --max-frames",
                frames.len()
            ));
        }
        Ok(frames[idx].clone())
    }
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FeatureExtractorKind {
    Corner,
    Hog,
    SuperPointOffline,
    /// Phase-27: in-Rust SuperPoint ONNX inference. Requires
    /// `visloc-vision` built with `--features onnx-inference` and
    /// `--superpoint-onnx-model <path>` pointing at a SuperPoint
    /// ONNX model with the LightGlue-ONNX-style I/O contract (see
    /// `docs/superpoint_onnx_runtime_plan.md`).
    SuperPointOnnx,
    /// Basalt-style frame-to-frame optical flow (Pattern51 KLT). Selected
    /// automatically by `--basalt-euroc-profile` / `--basalt-config`.
    OpticalFlow,
}
