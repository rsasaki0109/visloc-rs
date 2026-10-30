//! The inputs of one PatchMatch pass on a device, shared by the wgpu shaders
//! (`crate::gpu`) and the CUDA kernels (`crate::cuda`): the per-dispatch
//! parameters, the source cameras and grey levels, and the starting planes
//! of a reference view, as [`crate::patch_match`]'s pass reads them.

use crate::deformable::Deformable;
use crate::patch_match::{DepthMap, DetailRestorer, PatchMatchOptions, ViewSelection, MAX_SOURCES};
use crate::planar_prior::{PlanarPrior, PlanarPriorOptions};
use crate::workspace::{luma, Workspace};

/// Parameters of one dispatch (`Params` in the shader and the kernels).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "gpu", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub(crate) struct Params {
    pub width: u32,
    pub height: u32,
    pub num_sources: u32,
    pub color: u32,
    pub iteration: u32,
    pub seed: u32,
    pub view_id: u32,
    pub geometric: u32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub depth_min: f32,
    pub depth_max: f32,
    pub tau: f32,
    pub tau1: f32,
    pub beta: f32,
    pub n1: u32,
    pub n2: u32,
    pub init_top_k: u32,
    pub geom_lambda: f32,
    pub geom_delta: f32,
    pub depth_perturbation: f32,
    pub normal_perturbation: f32,
    pub window_radius: i32,
    pub window_step: i32,
    pub sigma_spatial: f32,
    pub sigma_color: f32,
    /// 0: [`ViewSelection::Confidence`], 1: [`ViewSelection::Sampled`].
    pub view_selection: u32,
    pub inv_texture_width: f32,
    pub inv_texture_height: f32,
    /// 1 in the planar prior pass, with its posterior's terms and its
    /// bandwidths.
    pub planar_prior: u32,
    pub prior_inv_alpha: f32,
    pub prior_gamma: f32,
    pub prior_depth_weight: f32,
    pub prior_angle_weight: f32,
    pub prior_depth_sigma: f32,
    pub prior_angle_sigma: f32,
    /// The detail restorer: 0 [`DetailRestorer::Off`], 1
    /// [`DetailRestorer::Start`], 2 [`DetailRestorer::Infinity`]; its
    /// margin.
    pub restore: u32,
    pub restore_margin: f32,
    /// 1 when the pass has deformable patches, with their settings (the
    /// RANSAC threshold in depth units) and the mean baseline to the
    /// sources.
    pub deformable: u32,
    pub sweep: i32,
    pub peak_radius: i32,
    pub max_peak_cost: f32,
    pub strong_cost: f32,
    pub peak_spread: f32,
    pub sectors: u32,
    pub ransac_threshold: f32,
    pub min_inliers: u32,
    pub center_weight: f32,
    pub baseline: f32,
}

/// A source view as the shader and the kernels read it (`Source`): the
/// reference-to-source rotation's rows with the translation in `w`, the
/// intrinsics, and (width, height, unused, offset of its earlier depth map).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "gpu", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub(crate) struct SourceCam {
    pub r0: [f32; 4],
    pub r1: [f32; 4],
    pub r2: [f32; 4],
    pub k: [f32; 4],
    pub dims: [u32; 4],
}

/// Everything a device needs for one pass of view `view`.
pub(crate) struct PassSetup {
    pub w: usize,
    pub h: usize,
    /// The sources used (at most [`MAX_SOURCES`]).
    pub source_ids: Vec<usize>,
    pub sources: Vec<SourceCam>,
    /// The source images' layer size: the largest source.
    pub texture_width: usize,
    pub texture_height: usize,
    /// One layer of grey levels per source (at least one layer), each
    /// `texture_width x texture_height` with the source in its top-left.
    pub layers: Vec<f32>,
    /// Floats of the sources' earlier depth maps (geometric mode).
    pub prior_len: usize,
    /// The starting planes as (normal, depth), depth 0 for a random one.
    pub init_planes: Vec<[f32; 4]>,
    /// The parameters of `init` and `find_anchors` (with the deformable
    /// settings when the pass has deformable patches).
    pub base: Params,
    /// One per (iteration, colour), in dispatch order.
    pub propagate: Vec<Params>,
    /// The deformable patches of the pass, when it starts from a map.
    pub deformable: Option<Deformable>,
    /// The settings to sort the result's pixels with, and `classify`'s
    /// parameters for them.
    pub sort: Option<(Deformable, Params)>,
}

impl PassSetup {
    /// One pass (`crate::gpu`'s `run`): starting from `start`'s planes
    /// (random ones where it has no depth, or everywhere without it), with
    /// geometric consistency against `prior`, or as the planar prior pass
    /// with `planar`'s prior (where `start`'s cost is not credible, from the
    /// prior's plane), and `restorer` deciding which pixels may leave their
    /// start plane.
    pub fn new(
        ws: &Workspace,
        view: usize,
        options: &PatchMatchOptions,
        prior: Option<&[DepthMap]>,
        start: Option<&DepthMap>,
        planar: Option<(&PlanarPriorOptions, &PlanarPrior)>,
        restorer: DetailRestorer,
    ) -> Self {
        let reference = &ws.views[view];
        let (w, h) = (reference.width, reference.height);
        let source_ids: Vec<usize> = reference
            .sources
            .iter()
            .take(MAX_SOURCES)
            .copied()
            .collect();

        // The sources' grey levels (one layer each) and their earlier depth
        // maps' offsets.
        let texture_width = source_ids
            .iter()
            .map(|&s| ws.views[s].width)
            .max()
            .unwrap_or(1);
        let texture_height = source_ids
            .iter()
            .map(|&s| ws.views[s].height)
            .max()
            .unwrap_or(1);
        let mut layers: Vec<f32> =
            Vec::with_capacity(texture_width * texture_height * source_ids.len().max(1));
        let mut prior_len = 0usize;
        let mut sources = Vec::with_capacity(source_ids.len());
        for &s in &source_ids {
            let source = &ws.views[s];
            let r = source.rotation * reference.rotation.transpose();
            let t = source.translation - r * reference.translation;
            let prior_offset = prior_len as u32;
            if let Some(maps) = prior {
                assert_eq!(
                    maps[s].depth.len(),
                    source.width * source.height,
                    "source view {s}'s earlier depth map"
                );
                prior_len += maps[s].depth.len();
            }
            sources.push(SourceCam {
                r0: [
                    r[(0, 0)] as f32,
                    r[(0, 1)] as f32,
                    r[(0, 2)] as f32,
                    t.x as f32,
                ],
                r1: [
                    r[(1, 0)] as f32,
                    r[(1, 1)] as f32,
                    r[(1, 2)] as f32,
                    t.y as f32,
                ],
                r2: [
                    r[(2, 0)] as f32,
                    r[(2, 1)] as f32,
                    r[(2, 2)] as f32,
                    t.z as f32,
                ],
                k: [
                    source.fx as f32,
                    source.fy as f32,
                    source.cx as f32,
                    source.cy as f32,
                ],
                dims: [source.width as u32, source.height as u32, 0, prior_offset],
            });
            let start = layers.len();
            layers.resize(start + texture_width * texture_height, 0.0);
            let layer = &mut layers[start..];
            for y in 0..source.height {
                let texels = &mut layer[y * texture_width..y * texture_width + source.width];
                let pixels = y * source.width..(y + 1) * source.width;
                if source.gray.is_empty() {
                    // The colours' luma, as View::gray_levels computes it.
                    for (texel, &c) in texels.iter_mut().zip(&source.rgb[pixels]) {
                        *texel = luma(c);
                    }
                } else {
                    texels.copy_from_slice(&source.gray[pixels]);
                }
            }
        }
        if source_ids.is_empty() {
            layers.resize(texture_width * texture_height, 0.0);
        }
        let mut init_planes: Vec<[f32; 4]> = match start {
            Some(map) => {
                assert_eq!((map.width, map.height), (w, h), "starting depth map size");
                map.depth
                    .iter()
                    .zip(&map.normal)
                    .map(|(&d, n)| [n[0], n[1], n[2], d.max(0.0)])
                    .collect()
            }
            None => vec![[0.0; 4]; w * h],
        };
        if let (Some((prior_options, planar)), Some(map)) = (planar, start) {
            assert_eq!((planar.width, planar.height), (w, h), "prior size");
            assert_eq!(map.cost.len(), w * h, "the start needs its costs");
            let (fx, fy) = (reference.fx as f32, reference.fy as f32);
            let (cx, cy) = (reference.cx as f32, reference.cy as f32);
            for (i, init) in init_planes.iter_mut().enumerate() {
                let Some([nx, ny, nz, rho]) = planar.plane(i) else {
                    continue;
                };
                if map.depth[i] > 0.0 && map.cost[i] < prior_options.credible_cost {
                    continue;
                }
                let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                let denom = nx * (x - cx) / fx + ny * (y - cy) / fy + nz;
                let depth = rho / denom;
                if denom < -1e-6 && depth > 0.0 && depth.is_finite() {
                    *init = [nx, ny, nz, depth];
                }
            }
        }

        // Deformable patches in a pass that starts from a map, and the
        // result's pixels sorted for the next pass.
        let deformable = options
            .deformable
            .as_ref()
            .filter(|_| start.is_some() && options.iterations > 0);
        let center = reference.center();
        let baseline = if source_ids.is_empty() {
            0.0
        } else {
            source_ids
                .iter()
                .map(|&s| (ws.views[s].center() - center).norm() as f32)
                .sum::<f32>()
                / source_ids.len() as f32
        };
        let base = Params {
            width: w as u32,
            height: h as u32,
            num_sources: sources.len() as u32,
            color: 0,
            iteration: 0,
            seed: options.seed,
            view_id: view as u32,
            geometric: u32::from(prior.is_some()),
            fx: reference.fx as f32,
            fy: reference.fy as f32,
            cx: reference.cx as f32,
            cy: reference.cy as f32,
            depth_min: reference.depth_range.0,
            depth_max: reference.depth_range.1,
            tau: options.tau0,
            tau1: options.tau1,
            beta: options.beta,
            n1: options.n1 as u32,
            n2: options.n2 as u32,
            init_top_k: options.init_top_k as u32,
            geom_lambda: options.geom_lambda,
            geom_delta: options.geom_delta,
            depth_perturbation: options.depth_perturbation,
            normal_perturbation: options.normal_perturbation,
            window_radius: options.window_radius,
            window_step: options.window_step,
            sigma_spatial: options.sigma_spatial,
            sigma_color: options.sigma_color,
            view_selection: u32::from(options.view_selection == ViewSelection::Sampled),
            inv_texture_width: 1.0 / texture_width as f32,
            inv_texture_height: 1.0 / texture_height as f32,
            planar_prior: u32::from(planar.is_some()),
            prior_inv_alpha: planar.map_or(0.0, |(o, _)| 1.0 / o.alpha),
            prior_gamma: planar.map_or(0.0, |(o, _)| o.gamma),
            prior_depth_weight: planar.map_or(0.0, |(o, _)| {
                let sigma = o.depth_bandwidth * (reference.depth_range.1 - reference.depth_range.0);
                1.0 / (2.0 * sigma * sigma)
            }),
            prior_angle_weight: planar.map_or(0.0, |(o, _)| {
                let sigma = o.angle_bandwidth_deg.to_radians();
                1.0 / (2.0 * sigma * sigma)
            }),
            prior_depth_sigma: planar.map_or(0.0, |(o, _)| {
                o.depth_bandwidth * (reference.depth_range.1 - reference.depth_range.0)
            }),
            prior_angle_sigma: planar.map_or(0.0, |(o, _)| o.angle_bandwidth_deg.to_radians()),
            restore: match restorer {
                DetailRestorer::Off => 0,
                DetailRestorer::Start(_) => 1,
                DetailRestorer::Infinity(_) => 2,
            },
            restore_margin: match restorer {
                DetailRestorer::Off => 0.0,
                DetailRestorer::Start(margin) | DetailRestorer::Infinity(margin) => margin,
            },
            deformable: 0,
            sweep: 0,
            peak_radius: 0,
            max_peak_cost: 0.0,
            strong_cost: 0.0,
            peak_spread: 0.0,
            sectors: 0,
            ransac_threshold: 0.0,
            min_inliers: 0,
            center_weight: 0.0,
            baseline,
        };
        let with_settings = |params: Params, d: &Deformable| Params {
            sweep: d.sweep.min(30),
            peak_radius: d.peak_radius,
            max_peak_cost: d.max_peak_cost,
            strong_cost: d.strong_cost,
            peak_spread: d.peak_spread,
            sectors: d.sectors as u32,
            ransac_threshold: d.ransac_threshold
                * (reference.depth_range.1 - reference.depth_range.0),
            min_inliers: d.min_inliers as u32,
            center_weight: d.center_weight,
            ..params
        };
        let base = match deformable {
            Some(d) => Params {
                deformable: 1,
                ..with_settings(base, d)
            },
            None => base,
        };
        let sort = options
            .sort_pixels
            .as_ref()
            .map(|d| (d.clone(), with_settings(base, d)));
        let mut propagate = Vec::with_capacity(2 * options.iterations);
        for iteration in 0..options.iterations {
            let t = iteration as f32;
            for color in 0..2 {
                propagate.push(Params {
                    color: color as u32,
                    iteration: iteration as u32,
                    tau: options.tau0 * (-t * t / options.alpha).exp(),
                    ..base
                });
            }
        }
        Self {
            w,
            h,
            source_ids,
            sources,
            texture_width,
            texture_height,
            layers,
            prior_len,
            init_planes,
            base,
            propagate,
            deformable: deformable.cloned(),
            sort,
        }
    }
}

/// The device tests shared by `crate::gpu` and `crate::cuda`: each device
/// against the CPU path's tests and results.
#[cfg(test)]
pub(crate) mod checks {
    use crate::deformable::{Class, Deformable};
    use crate::patch_match::{
        estimate, photometric_score, plane_scene, plane_scene_truth, DepthMap, DetailRestorer,
        PatchMatchOptions, ViewSelection, MAX_COST,
    };
    use crate::planar_prior::{
        mean_relative_error, planar_prior, textureless_scene, PlanarPrior, PlanarPriorOptions,
    };
    use crate::workspace::Workspace;

    /// A device's passes (its inherent methods, through [`impl_device`]).
    pub(crate) trait Device {
        fn estimate(
            &self,
            ws: &Workspace,
            view: usize,
            options: &PatchMatchOptions,
            prior: Option<&[DepthMap]>,
        ) -> DepthMap;
        fn estimate_from(
            &self,
            ws: &Workspace,
            view: usize,
            options: &PatchMatchOptions,
            start: &DepthMap,
            restorer: DetailRestorer,
        ) -> DepthMap;
        fn estimate_with_prior(
            &self,
            ws: &Workspace,
            view: usize,
            options: &PatchMatchOptions,
            prior_options: &PlanarPriorOptions,
            start: &DepthMap,
            prior: &PlanarPrior,
        ) -> DepthMap;
        fn photometric_score(
            &self,
            ws: &Workspace,
            view: usize,
            options: &PatchMatchOptions,
            map: &DepthMap,
        ) -> Vec<f32>;
    }

    /// [`Device`] for a type with the four passes as inherent methods.
    macro_rules! impl_device {
        ($device:ty) => {
            impl $crate::pass_setup::checks::Device for $device {
                fn estimate(
                    &self,
                    ws: &$crate::workspace::Workspace,
                    view: usize,
                    options: &$crate::patch_match::PatchMatchOptions,
                    prior: Option<&[$crate::patch_match::DepthMap]>,
                ) -> $crate::patch_match::DepthMap {
                    <$device>::estimate(self, ws, view, options, prior)
                }
                fn estimate_from(
                    &self,
                    ws: &$crate::workspace::Workspace,
                    view: usize,
                    options: &$crate::patch_match::PatchMatchOptions,
                    start: &$crate::patch_match::DepthMap,
                    restorer: $crate::patch_match::DetailRestorer,
                ) -> $crate::patch_match::DepthMap {
                    <$device>::estimate_from(self, ws, view, options, start, restorer)
                }
                fn estimate_with_prior(
                    &self,
                    ws: &$crate::workspace::Workspace,
                    view: usize,
                    options: &$crate::patch_match::PatchMatchOptions,
                    prior_options: &$crate::planar_prior::PlanarPriorOptions,
                    start: &$crate::patch_match::DepthMap,
                    prior: &$crate::planar_prior::PlanarPrior,
                ) -> $crate::patch_match::DepthMap {
                    <$device>::estimate_with_prior(
                        self,
                        ws,
                        view,
                        options,
                        prior_options,
                        start,
                        prior,
                    )
                }
                fn photometric_score(
                    &self,
                    ws: &$crate::workspace::Workspace,
                    view: usize,
                    options: &$crate::patch_match::PatchMatchOptions,
                    map: &$crate::patch_match::DepthMap,
                ) -> Vec<f32> {
                    <$device>::photometric_score(self, ws, view, options, map)
                }
            }
        };
    }
    pub(crate) use impl_device;

    /// The planar prior pass fills a textureless region on the device as on
    /// the CPU (`planar_prior`'s test).
    pub(crate) fn prior_fills_a_textureless_region(device: &impl Device) {
        let (ws, inner) = textureless_scene();
        let options = PatchMatchOptions::default();
        let photometric = device.estimate(&ws, 0, &options, None);
        let prior_options = PlanarPriorOptions::default();
        let prior = planar_prior(&ws.views[0], &photometric, &prior_options);
        let with_prior =
            device.estimate_with_prior(&ws, 0, &options, &prior_options, &photometric, &prior);
        let truth = plane_scene_truth(&ws, 0);
        let before = mean_relative_error(&photometric, &truth, &inner);
        let after = mean_relative_error(&with_prior, &truth, &inner);
        assert!(after < 0.01, "with the prior: {after}");
        assert!(
            before > 5.0 * after,
            "photometric {before} vs prior {after}"
        );
    }

    /// The device recovers the synthetic plane like the CPU path, with and
    /// without geometric consistency, under either view selection.
    pub(crate) fn recovers_a_plane(device: &impl Device) {
        let ws = plane_scene();
        let within = |map: &DepthMap| {
            let mut good = 0;
            for y in 8..40 {
                for x in 12..52 {
                    if (map.depth[y * 64 + x] - 4.0).abs() < 0.04 {
                        good += 1;
                    }
                }
            }
            good
        };
        let inside = 32 * 40;
        for view_selection in [ViewSelection::Confidence, ViewSelection::Sampled] {
            let options = PatchMatchOptions {
                iterations: 4,
                view_selection,
                ..PatchMatchOptions::default()
            };
            let map = device.estimate(&ws, 0, &options, None);
            let cpu = estimate(&ws, 0, &options, None);
            assert!(
                within(&map) * 10 >= inside * 9,
                "device {view_selection:?}: {} of {inside}",
                within(&map)
            );
            assert!(
                within(&cpu) * 10 >= inside * 9,
                "CPU {view_selection:?}: {} of {inside}",
                within(&cpu)
            );
            let maps: Vec<DepthMap> = (0..3)
                .map(|v| device.estimate(&ws, v, &options, None))
                .collect();
            let refined = device.estimate(&ws, 0, &options, Some(&maps));
            assert!(
                within(&refined) * 10 >= inside * 9,
                "device geometric {view_selection:?}: {}",
                within(&refined)
            );
        }
    }

    /// The detail restorer on the device as on the CPU (`patch_match`'s
    /// test): from a start 2% too deep, the restorer against the start's
    /// cost keeps it, the one against the plane at infinity re-estimates.
    pub(crate) fn restorer_matches_the_cpu(device: &impl Device) {
        let ws = plane_scene();
        let truth = plane_scene_truth(&ws, 0);
        let mut start = truth.clone();
        start.depth.iter_mut().for_each(|d| *d *= 1.02);
        let options = PatchMatchOptions::default();
        let inside: Vec<usize> = (8..40)
            .flat_map(|y| (12..52).map(move |x| y * 64 + x))
            .collect();
        let error = |restorer| {
            let map = device.estimate_from(&ws, 0, &options, &start, restorer);
            mean_relative_error(&map, &truth, &inside)
        };
        let kept = error(DetailRestorer::Start(0.1));
        assert!(kept > 0.015, "against the start: {kept}");
        let error = error(DetailRestorer::Infinity(0.1));
        assert!(error < 0.002, "against infinity: {error}");
    }

    /// Deformable patches fill a textureless region on the device as on the
    /// CPU (`patch_match`'s test).
    pub(crate) fn deformable_patches_fill_a_textureless_region(device: &impl Device) {
        let (ws, inner) = textureless_scene();
        let options = PatchMatchOptions::default();
        let photometric = device.estimate(&ws, 0, &options, None);
        let deformable = PatchMatchOptions {
            deformable: Some(Deformable::default()),
            ..options.clone()
        };
        let truth = plane_scene_truth(&ws, 0);
        let plain = device.estimate_from(&ws, 0, &options, &photometric, DetailRestorer::Off);
        let with = device.estimate_from(&ws, 0, &deformable, &photometric, DetailRestorer::Off);
        let before = mean_relative_error(&plain, &truth, &inner);
        let after = mean_relative_error(&with, &truth, &inner);
        assert!(after < 0.015, "deformable: {after}");
        assert!(
            before > 10.0 * after,
            "plain {before} vs deformable {after}"
        );
    }

    /// On the device the strong-only median filters with the pass's
    /// classes: a pass whose pixels are all weak keeps every depth as
    /// estimated, and one whose pixels are all strong filters every depth.
    pub(crate) fn strong_median_follows_the_pass_classes(device: &impl Device) {
        let (ws, _) = textureless_scene();
        let options = PatchMatchOptions::default();
        let start = device.estimate(&ws, 0, &options, None);
        let with_classes = |class: Class| DepthMap {
            classes: vec![class; start.depth.len()],
            ..start.clone()
        };
        let deformable = |strong_median: bool, median_filter: bool| PatchMatchOptions {
            deformable: Some(Deformable {
                strong_median,
                ..Deformable::default()
            }),
            median_filter,
            ..options.clone()
        };
        let run = |o: &PatchMatchOptions, class: Class| {
            device.estimate_from(&ws, 0, o, &with_classes(class), DetailRestorer::Off)
        };
        let weak_median = run(&deformable(true, true), Class::Weak);
        let weak_unfiltered = run(&deformable(false, false), Class::Weak);
        assert_eq!(weak_median.depth, weak_unfiltered.depth);
        let strong_median = run(&deformable(true, true), Class::Strong);
        let strong_unfiltered = run(&deformable(false, false), Class::Strong);
        assert_ne!(strong_median.depth, strong_unfiltered.depth);
    }

    /// `deformable_passes_take_the_start_classes` on the device.
    pub(crate) fn deformable_passes_take_the_start_classes(device: &impl Device) {
        let (ws, inner) = textureless_scene();
        let options = PatchMatchOptions::default();
        let sorting = PatchMatchOptions {
            sort_pixels: Some(Deformable::default()),
            ..options.clone()
        };
        let photometric = device.estimate(&ws, 0, &sorting, None);
        assert_eq!(photometric.classes.len(), photometric.depth.len());
        assert!(inner.iter().any(|&i| photometric.classes[i] == Class::Weak));
        let deformable = PatchMatchOptions {
            deformable: Some(Deformable::default()),
            ..options.clone()
        };
        let truth = plane_scene_truth(&ws, 0);
        let with = device.estimate_from(&ws, 0, &deformable, &photometric, DetailRestorer::Off);
        let after = mean_relative_error(&with, &truth, &inner);
        assert!(after < 0.015, "deformable: {after}");
        assert!(with.classes.is_empty());
        let all_strong = DepthMap {
            classes: vec![Class::Strong; photometric.depth.len()],
            ..photometric.clone()
        };
        let plain = device.estimate_from(&ws, 0, &options, &photometric, DetailRestorer::Off);
        let strong = device.estimate_from(&ws, 0, &deformable, &all_strong, DetailRestorer::Off);
        assert_eq!(strong.depth, plain.depth);
    }

    /// The device's photometric score of a map matches the CPU's: the
    /// median and 99th percentile differences below `median` and `p99`.
    pub(crate) fn photometric_score_matches_the_cpu(device: &impl Device, median: f32, p99: f32) {
        let ws = plane_scene();
        let options = PatchMatchOptions::default();
        let mut map = plane_scene_truth(&ws, 0);
        // A pixel without depth scores the maximum on both.
        map.depth[0] = 0.0;
        let cpu = photometric_score(&ws, 0, &options, &map);
        let on_device = device.photometric_score(&ws, 0, &options, &map);
        assert_eq!(on_device[0], MAX_COST);
        let mut differences: Vec<f32> = cpu
            .iter()
            .zip(&on_device)
            .map(|(a, b)| (a - b).abs())
            .collect();
        differences.sort_by(f32::total_cmp);
        let at = |q: f64| differences[((differences.len() - 1) as f64 * q) as usize];
        assert!(at(0.5) < median, "median difference {}", at(0.5));
        assert!(at(0.99) < p99, "99th percentile difference {}", at(0.99));
    }
}
