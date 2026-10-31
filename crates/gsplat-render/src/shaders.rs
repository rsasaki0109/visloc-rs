//! WGSL shader assembly.
//!
//! WGSL has no `#include`, so each compute pipeline gets its own module built
//! by concatenating the shared `common.wgsl` prelude with the kernel body.

/// Source of a compute kernel: the shared prelude plus one kernel file.
pub struct Kernel {
    pub name: &'static str,
    pub source: String,
}

const COMMON: &str = include_str!("shaders/common.wgsl");

fn build(name: &'static str, body: &str) -> Kernel {
    Kernel {
        name,
        source: format!("{COMMON}\n{body}"),
    }
}

/// Keep or drop the lines between `// #if GEO` and `// #else` / `// #endif`
/// markers (the geometry variant renders normals and depth as extra
/// channels; the plain variant is compiled without any trace of them).
fn select_geo(src: &str, geo: bool) -> String {
    let mut out = String::with_capacity(src.len());
    // None: outside a block; Some(keep): inside, keeping the current branch.
    let mut keep: Option<bool> = None;
    for line in src.lines() {
        match line.trim() {
            "// #if GEO" => keep = Some(geo),
            "// #else" => keep = keep.map(|k| !k),
            "// #endif" => keep = None,
            _ => {
                if keep.unwrap_or(true) {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
    }
    out
}

fn build_geo(name: &'static str, body: &str, geo: bool) -> Kernel {
    build(name, &select_geo(body, geo))
}

/// Uses subgroup ops: naga (wgpu 30) accepts them without an `enable`
/// directive when the device has `Features::SUBGROUP`.
pub fn rasterize_backward() -> Kernel {
    rasterize_backward_variant(false)
}

/// `geo`: also back-propagate the normal / depth channels (see
/// `rasterize_variant`).
pub fn rasterize_backward_variant(geo: bool) -> Kernel {
    build_geo(
        "rasterize_backward",
        include_str!("shaders/rasterize_backward.wgsl"),
        geo,
    )
}

pub fn project_backward() -> Kernel {
    project_backward_variant(false)
}

pub fn project_backward_variant(geo: bool) -> Kernel {
    build_geo(
        "project_backward",
        include_str!("shaders/project_backward.wgsl"),
        geo,
    )
}

/// Per visible gaussian: camera-space normal (shortest axis, facing the
/// camera) and depth, for the geometry channels.
pub fn project_geo() -> Kernel {
    build("project_geo", include_str!("shaders/project_geo.wgsl"))
}

pub fn project_forward() -> Kernel {
    build(
        "project_forward",
        include_str!("shaders/project_forward.wgsl"),
    )
}

pub fn project_visible() -> Kernel {
    build(
        "project_visible",
        include_str!("shaders/project_visible.wgsl"),
    )
}

pub fn map_gaussians() -> Kernel {
    build("map_gaussians", include_str!("shaders/map_gaussians.wgsl"))
}

pub fn gather_compact() -> Kernel {
    build(
        "gather_compact",
        include_str!("shaders/gather_compact.wgsl"),
    )
}

pub fn tile_offsets() -> Kernel {
    build("tile_offsets", include_str!("shaders/tile_offsets.wgsl"))
}

pub fn rasterize() -> Kernel {
    rasterize_variant(false)
}

/// `geo`: also composite each splat's camera-space normal and depth into a
/// 4-float-per-pixel buffer (alpha-weighted sums, no background).
pub fn rasterize_variant(geo: bool) -> Kernel {
    build_geo("rasterize", include_str!("shaders/rasterize.wgsl"), geo)
}

pub fn rasterize_depth() -> Kernel {
    build(
        "rasterize_depth",
        include_str!("shaders/rasterize_depth.wgsl"),
    )
}

pub fn radix() -> Kernel {
    build("radix", include_str!("shaders/radix.wgsl"))
}

pub fn scan() -> Kernel {
    build("scan", include_str!("shaders/scan.wgsl"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_geo_keeps_one_branch() {
        let src = "a\n// #if GEO\ng\n// #else\nplain\n// #endif\nb\n";
        assert_eq!(super::select_geo(src, true), "a\ng\nb\n");
        assert_eq!(super::select_geo(src, false), "a\nplain\nb\n");
    }

    /// Every kernel (both variants of the geometry ones) parses and
    /// validates with naga, as wgpu does before building a pipeline, so a
    /// shader edit is checked without a GPU.
    #[test]
    fn every_kernel_validates() {
        use naga::valid::{
            Capabilities, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator,
        };
        let kernels = [
            rasterize_backward_variant(false),
            rasterize_backward_variant(true),
            project_backward_variant(false),
            project_backward_variant(true),
            project_geo(),
            project_forward(),
            project_visible(),
            map_gaussians(),
            gather_compact(),
            tile_offsets(),
            rasterize_variant(false),
            rasterize_variant(true),
            rasterize_depth(),
            radix(),
            scan(),
        ];
        for kernel in kernels {
            let module = naga::front::wgsl::parse_str(&kernel.source).unwrap_or_else(|e| {
                panic!("{}: {}", kernel.name, e.emit_to_string(&kernel.source))
            });
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .subgroup_stages(ShaderStages::all())
                .subgroup_operations(SubgroupOperationSet::all())
                .validate(&module)
                .unwrap_or_else(|e| {
                    panic!("{}: {}", kernel.name, e.emit_to_string(&kernel.source))
                });
        }
    }
}
