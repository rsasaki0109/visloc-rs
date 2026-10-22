//! wgpu device/queue creation for the headless renderer.

use thiserror::Error;

/// Errors from GPU setup.
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no compatible GPU adapter was found")]
    NoAdapter,
    #[error("failed to create the GPU device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("the GPU device lacks a required feature: {0}")]
    MissingFeature(&'static str),
}

/// A ready-to-use wgpu device and queue.
pub struct GpuContext {
    // Keep the instance alive for the lifetime of the device: some backends
    // release resources the device depends on when the instance is dropped.
    _instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    /// Adapter limits actually requested for the device (used to size buffers).
    pub limits: wgpu::Limits,
    /// Optional features enabled on the device (`SUBGROUP` when supported).
    pub features: wgpu::Features,
}

impl GpuContext {
    /// Create a high-performance adapter (a discrete GPU if present).
    ///
    /// On Windows the default backend set is restricted to DX12 there, because
    /// third-party Vulkan layers (Steam / Epic / Rockstar overlays) and hybrid
    /// Intel+NVIDIA setups are known to crash Vulkan/GL instance creation. Set
    /// `WGPU_BACKEND` or use [`GpuContext::with_backends`] to override.
    pub fn new() -> Result<Self, GpuError> {
        if let Ok(env) = std::env::var("WGPU_BACKEND") {
            let backends = match env.to_ascii_lowercase().as_str() {
                "dx12" => wgpu::Backends::DX12,
                "vulkan" => wgpu::Backends::VULKAN,
                "gl" => wgpu::Backends::GL,
                "metal" => wgpu::Backends::METAL,
                _ => wgpu::Backends::all(),
            };
            return Self::with_backends(backends);
        }
        #[cfg(target_os = "windows")]
        {
            Self::with_backends(wgpu::Backends::DX12)
        }
        #[cfg(not(target_os = "windows"))]
        {
            Self::with_backends(wgpu::Backends::PRIMARY)
        }
    }

    /// Create a context restricted to specific backends (useful in tests).
    pub fn with_backends(backends: wgpu::Backends) -> Result<Self, GpuError> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        desc.backend_options.dx12.shader_compiler = dx12_shader_compiler();
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|_| GpuError::NoAdapter)?;
        let info = adapter.get_info();
        // Request the adapter's real limits so large scenes are not capped by
        // the conservative WebGPU defaults (e.g. 256 MiB buffers).
        let limits = adapter.limits();
        // Subgroup ops are used by the backward pass (training) when present.
        let features =
            adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("visloc-gsplat-render"),
                required_features: features,
                required_limits: limits.clone(),
                ..Default::default()
            }))?;
        Ok(Self {
            _instance: instance,
            device,
            queue,
            adapter_info: info,
            limits,
            features,
        })
    }
}

/// DX12 shader compiler choice.
///
/// wgpu's `Auto` falls back to FXC when `dxcompiler.dll` is not on the DLL
/// search path, and FXC takes ~10 minutes to compile this renderer's compute
/// shaders (DXC: seconds). So unless `WGPU_DX12_COMPILER` says otherwise, look
/// for `dxcompiler.dll` on `PATH` and then in the newest installed Windows SDK,
/// and load it explicitly. An older DXC only lowers the maximum shader model,
/// which these baseline-WebGPU shaders do not need.
fn dx12_shader_compiler() -> wgpu::Dx12Compiler {
    if let Some(from_env) = wgpu::Dx12Compiler::from_env() {
        return from_env;
    }
    #[cfg(target_os = "windows")]
    if let Some(path) = find_dxcompiler() {
        return wgpu::Dx12Compiler::DynamicDxc {
            dxc_path: path.to_string_lossy().into_owned(),
        };
    }
    wgpu::Dx12Compiler::Auto
}

#[cfg(target_os = "windows")]
fn find_dxcompiler() -> Option<std::path::PathBuf> {
    const DLL: &str = "dxcompiler.dll";
    if let Some(paths) = std::env::var_os("PATH") {
        if let Some(p) = std::env::split_paths(&paths)
            .map(|d| d.join(DLL))
            .find(|p| p.is_file())
        {
            return Some(p);
        }
    }
    let arch = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x64"
    };
    let kits = std::path::PathBuf::from(
        std::env::var_os("ProgramFiles(x86)").unwrap_or_else(|| r"C:\Program Files (x86)".into()),
    )
    .join("Windows Kits")
    .join("10")
    .join("bin");
    // SDK dirs are named like `10.0.26100.0`; take the highest version that
    // ships the DLL.
    let version =
        |name: &str| -> Vec<u32> { name.split('.').map(|c| c.parse().unwrap_or(0)).collect() };
    std::fs::read_dir(kits)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let dll = e.path().join(arch).join(DLL);
            dll.is_file()
                .then(|| (version(&e.file_name().to_string_lossy()), dll))
        })
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, dll)| dll)
}

/// Try to create a GPU context, returning `None` when no adapter is available.
///
/// Tests use this so they skip gracefully on headless machines instead of
/// failing.
pub fn try_context() -> Option<GpuContext> {
    GpuContext::new().ok()
}
