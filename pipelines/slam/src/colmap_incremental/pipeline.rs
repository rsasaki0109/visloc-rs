//! Faithful (C2-scoped) port of `controllers/incremental_pipeline.{h,cc}`'s
//! `IncrementalPipeline::Run`/`Reconstruct`/`ReconstructSubModel`/
//! `InitializeReconstruction`/`CheckRunGlobalRefinement`.
//!
//! Ported from (commit `64805cb870b574a569dccc34918d95a2db2b2fee`, pinned by
//! `docs/colmap_rig_mapper_port_plan.md` §0):
//! `src/colmap/controllers/incremental_pipeline.h/.cc` — `Options`
//! (`.h:47-215`, control defaults per plan §0.1/§1.2:
//! `multiple_models=true`, `max_num_models=50`, `min_model_size=10`),
//! `LocalBundleAdjustment`/`GlobalBundleAdjustment` factories
//! (`.cc:192-282`, see `bundle_adjustment.rs` for the iteration-count-only
//! subset threaded through), `Run` (`.cc:381-448`), `InitializeReconstruction`
//! (`.cc:450-528`), `CheckRunGlobalRefinement` (`.cc:530-542`),
//! `ReconstructSubModel` (`.cc:544-711`, structure-less registration
//! omitted — see deviation below), `Reconstruct` (`.cc:713-820`).
//!
//! ## Deviations
//!
//! - **`structure_less_registration_fallback` (COLMAP default `true`,
//!   `incremental_pipeline.h:83`) is not ported.** `RegisterNextStructureLessImage`
//!   (`incremental_mapper.cc:671-949`, epipolar/Sampson-based 2D-2D
//!   resectioning with its own minimal solver) is a materially large
//!   additional estimator that the C2 task brief did not list among
//!   `mapper.rs`'s required routines (`RegisterNextImage`,
//!   `RegisterNextGeneralFrame`, `FindNextImages`, BA, filtering,
//!   retriangulation). Its omission means this port will fail to register
//!   any frame that COLMAP only recovers via structure-less fallback (no
//!   2D-3D correspondences, only 2D-2D) — flagged as a real, expected
//!   source of registered-frame shortfall alongside the GP3P gap (see
//!   `mapper.rs`'s module doc), not silently absorbed.
//! - **`AlignReconstructionToPriorsOrRigScale`/`Normalize()`** (`.cc:96-160`)
//!   are both omitted — see `mapper.rs`'s `iterative_global_refinement` doc
//!   comment: with `ba_refine_sensor_from_rig=0` this control's rescale
//!   always recovers scale factor exactly `1.0`, making the
//!   `Normalize()`-then-rescale round trip a no-op for the final model.
//! - **No snapshotting, color extraction, or `max_runtime_seconds`/stop
//!   callback support** — none are relevant to a benchmark harness reading
//!   pre-extracted features with no source images.
//! - **The two-stage init-threshold relaxation loop (`Run`, `.cc:418-445`)
//!   is reproduced but its `ShouldStop` early-exit only checks
//!   `num_total_reg_images == num_images`** (no interrupt/max-runtime
//!   support to check, per the point above).

use super::bundle_adjustment::BundleAdjustmentOptions;
use super::database_cache::DatabaseCache;
use super::incremental_triangulator::Options as TriangulatorOptions;
use super::mapper::{IncrementalMapper, Options as MapperOptions};
use super::reconstruction::Reconstruction;
use super::types::ImageT;

/// Port of `IncrementalPipelineOptions`' control-relevant subset (`.h:47-215`).
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineOptions {
    pub multiple_models: bool,
    pub max_num_models: usize,
    pub min_model_size: usize,
    pub max_model_overlap: usize,
    pub init_num_trials: usize,
    pub ba_local_max_refinements: usize,
    pub ba_local_max_refinement_change: f64,
    pub ba_global_max_refinements: usize,
    pub ba_global_max_refinement_change: f64,
    pub ba_global_frames_ratio: f64,
    pub ba_global_points_ratio: f64,
    pub ba_global_frames_freq: usize,
    pub ba_global_points_freq: usize,
    pub mapper: MapperOptions,
    pub triangulation: TriangulatorOptions,
    pub local_ba: BundleAdjustmentOptions,
    pub global_ba: BundleAdjustmentOptions,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            multiple_models: true,
            max_num_models: 50,
            min_model_size: 10,
            max_model_overlap: 20,
            init_num_trials: 200,
            ba_local_max_refinements: 2,
            ba_local_max_refinement_change: 0.001,
            ba_global_max_refinements: 5,
            ba_global_max_refinement_change: 0.0005,
            ba_global_frames_ratio: 1.1,
            ba_global_points_ratio: 1.1,
            ba_global_frames_freq: 500,
            ba_global_points_freq: 250_000,
            mapper: MapperOptions::default(),
            triangulation: TriangulatorOptions::default(),
            local_ba: BundleAdjustmentOptions::local(),
            global_ba: BundleAdjustmentOptions::global(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Success,
    BadInitialPair,
    NoInitialPair,
}

pub(crate) fn reconstruction_from_cache(db: &DatabaseCache) -> Reconstruction {
    let mut recon = Reconstruction::new();
    for rig in db.rigs().values() {
        recon.add_rig(rig.clone());
    }
    for camera in db.cameras().values() {
        recon.add_camera(camera.clone());
    }
    for frame in db.frames().values() {
        recon.add_frame(frame.clone());
    }
    for image in db.images().values() {
        recon.add_image(image.clone());
    }
    recon
}

/// Port of `InitializeReconstruction` (`.cc:450-528`).
fn initialize_reconstruction(
    options: &PipelineOptions,
    mapper_options: &MapperOptions,
    mapper: &mut IncrementalMapper,
    db: &DatabaseCache,
    recon: &mut Reconstruction,
) -> Status {
    let Some((image_id1, image_id2, cam2_from_cam1)) =
        mapper.find_initial_image_pair(mapper_options, recon, db.correspondence_graph())
    else {
        return Status::NoInitialPair;
    };

    mapper.register_initial_image_pair(
        recon,
        db.correspondence_graph(),
        image_id1,
        image_id2,
        cam2_from_cam1,
    );

    let mut tri_options = options.triangulation.clone();
    tri_options.min_angle = mapper_options.init_min_tri_angle_deg;
    for image_id in [image_id1, image_id2] {
        let frame_id = recon.image(image_id).frame_id;
        let frame_images: Vec<ImageT> = recon.frame(frame_id).image_ids().collect();
        for iid in frame_images {
            mapper.triangulate_image(&tri_options, recon, db.correspondence_graph(), iid);
        }
    }

    if recon.num_points3d() == 0 {
        return Status::BadInitialPair;
    }

    mapper.adjust_global_bundle(&options.global_ba, recon, db.correspondence_graph());
    mapper.filter_points(mapper_options, recon, db.correspondence_graph());
    mapper.filter_frames(mapper_options, recon, db.correspondence_graph());

    if recon.num_reg_frames() == 0 || recon.num_points3d() == 0 {
        return Status::BadInitialPair;
    }
    if recon.num_points3d() < mapper_options.abs_pose_min_num_inliers {
        return Status::BadInitialPair;
    }

    Status::Success
}

/// Port of `CheckRunGlobalRefinement` (`.cc:530-542`).
fn check_run_global_refinement(
    options: &PipelineOptions,
    recon: &Reconstruction,
    prev_frames: usize,
    prev_points: usize,
) -> bool {
    (recon.num_reg_frames() as f64) >= options.ba_global_frames_ratio * (prev_frames as f64)
        || recon.num_reg_frames() >= options.ba_global_frames_freq + prev_frames
        || (recon.num_points3d() as f64) >= options.ba_global_points_ratio * (prev_points as f64)
        || recon.num_points3d() >= options.ba_global_points_freq + prev_points
}

/// Port of the free function `IterativeGlobalRefinement`
/// (`incremental_pipeline.cc:84-94`).
fn iterative_global_refinement(
    options: &PipelineOptions,
    mapper_options: &MapperOptions,
    mapper: &mut IncrementalMapper,
    db: &DatabaseCache,
    recon: &mut Reconstruction,
) {
    let t0 = std::time::Instant::now();
    mapper.iterative_global_refinement(
        options.ba_global_max_refinements,
        options.ba_global_max_refinement_change,
        mapper_options,
        &options.global_ba,
        &options.triangulation,
        recon,
        db.correspondence_graph(),
    );
    eprintln!(
        "TIMING iterative_global_refinement elapsed_ms={} num_reg_frames={} points3d={}",
        t0.elapsed().as_millis(),
        recon.num_reg_frames(),
        recon.num_points3d()
    );
    mapper.filter_frames(mapper_options, recon, db.correspondence_graph());
}

/// Port of `ReconstructSubModel` (`.cc:544-711`), structure-less fallback
/// omitted (module doc).
fn reconstruct_sub_model(
    options: &PipelineOptions,
    mapper_options: &MapperOptions,
    mapper: &mut IncrementalMapper,
    db: &DatabaseCache,
    recon: &mut Reconstruction,
) -> Status {
    mapper.begin_reconstruction(recon, db.correspondence_graph());

    if recon.num_reg_frames() == 0 {
        let init_status = initialize_reconstruction(options, mapper_options, mapper, db, recon);
        if init_status != Status::Success {
            return init_status;
        }
    }

    let mut ba_prev_num_reg_frames = recon.num_reg_frames();
    let mut ba_prev_num_points = recon.num_points3d();

    let mut reg_next_success = true;
    #[allow(unused_assignments)]
    let mut prev_reg_next_success = true;
    loop {
        prev_reg_next_success = reg_next_success;
        reg_next_success = false;
        let mut registered_image_id: Option<ImageT> = None;

        let next_images = mapper.find_next_images(mapper_options, recon);
        for &candidate in &next_images {
            if mapper.register_next_image(
                mapper_options,
                recon,
                db.correspondence_graph(),
                candidate,
            ) {
                reg_next_success = true;
                registered_image_id = Some(candidate);
                break;
            }
        }

        if reg_next_success {
            let image_id = registered_image_id.unwrap();
            let frame_id = recon.image(image_id).frame_id;
            let frame_images: Vec<ImageT> = recon.frame(frame_id).image_ids().collect();
            let t_tri = std::time::Instant::now();
            for iid in frame_images {
                mapper.triangulate_image(
                    &options.triangulation,
                    recon,
                    db.correspondence_graph(),
                    iid,
                );
            }
            eprintln!(
                "TIMING triangulate_image frame={} elapsed_ms={} points3d={}",
                frame_id,
                t_tri.elapsed().as_millis(),
                recon.num_points3d()
            );
            let t_local = std::time::Instant::now();
            mapper.iterative_local_refinement(
                options.ba_local_max_refinements,
                options.ba_local_max_refinement_change,
                mapper_options,
                &options.local_ba,
                &options.triangulation,
                recon,
                db.correspondence_graph(),
                image_id,
            );
            eprintln!(
                "TIMING iterative_local_refinement frame={} elapsed_ms={} points3d={}",
                frame_id,
                t_local.elapsed().as_millis(),
                recon.num_points3d()
            );

            if check_run_global_refinement(
                options,
                recon,
                ba_prev_num_reg_frames,
                ba_prev_num_points,
            ) {
                iterative_global_refinement(options, mapper_options, mapper, db, recon);
                ba_prev_num_points = recon.num_points3d();
                ba_prev_num_reg_frames = recon.num_reg_frames();
            }
        }

        if mapper.num_shared_reg_images() >= options.max_model_overlap {
            break;
        }

        if !reg_next_success && prev_reg_next_success {
            iterative_global_refinement(options, mapper_options, mapper, db, recon);
        }

        if !(reg_next_success || prev_reg_next_success) {
            break;
        }
    }

    if recon.num_reg_frames() > 0
        && recon.num_reg_frames() != ba_prev_num_reg_frames
        && recon.num_points3d() != ba_prev_num_points
    {
        iterative_global_refinement(options, mapper_options, mapper, db, recon);
    }

    Status::Success
}

/// Result of one kept sub-model.
pub struct ModelResult {
    pub reconstruction: Reconstruction,
}

/// Everything `run` produces: every kept sub-model plus the full
/// chronological run log (`IncrementalMapper::log`) accumulated by the one
/// `IncrementalMapper` instance used across every trial/relaxation stage —
/// per-registration lines (frame id, path A/B, inliers, `num_reg_frames`)
/// and BA events, consumed by `examples/colmap_incremental_mapper.rs`.
pub struct RunResult {
    pub models: Vec<ModelResult>,
    pub log: Vec<String>,
}

/// Port of `IncrementalPipeline::Run`/`Reconstruct` (`.cc:381-448,713-820`).
/// Returns every kept sub-model, in the order they were finalized, plus the
/// mapper's run log.
pub fn run(options: &PipelineOptions, db: &DatabaseCache) -> RunResult {
    let mut mapper = IncrementalMapper::new();
    let mut kept: Vec<ModelResult> = Vec::new();
    let num_images = db.num_images();

    let relax0 = options.mapper.clone();
    let mut relax1 = relax0.clone();
    relax1.init_min_num_inliers = (relax1.init_min_num_inliers / 2).max(1);
    let mut relax2 = relax1.clone();
    relax2.init_min_tri_angle_deg /= 2.0;
    // `Run()`'s outer loop (`.cc:418-445`): original options, then up to
    // `kNumInitRelaxations=2` progressively relaxed passes.
    let variants = [relax0, relax1, relax2];

    'relax: for mapper_options in &variants {
        if mapper.num_total_reg_images() >= num_images {
            break 'relax;
        }
        mapper.reset_initialization_stats();

        for _trial in 0..options.init_num_trials {
            let mut recon = reconstruction_from_cache(db);
            let status =
                reconstruct_sub_model(options, mapper_options, &mut mapper, db, &mut recon);
            match status {
                Status::Success => {
                    let num_reg_images = recon.num_reg_images();
                    let discard = (options.multiple_models
                        && !kept.is_empty()
                        && num_reg_images < options.min_model_size)
                        || num_reg_images == 0;
                    if discard {
                        mapper.end_reconstruction(&mut recon, db.correspondence_graph(), true);
                    } else {
                        mapper.end_reconstruction(&mut recon, db.correspondence_graph(), false);
                        kept.push(ModelResult {
                            reconstruction: recon,
                        });
                    }
                    if !options.multiple_models
                        || kept.len() >= options.max_num_models
                        || mapper.num_total_reg_images() + 1 >= num_images
                    {
                        break 'relax;
                    }
                    // else: keep trying to reconstruct another sub-model in
                    // the next trial.
                }
                Status::BadInitialPair => {
                    mapper.end_reconstruction(&mut recon, db.correspondence_graph(), true);
                }
                Status::NoInitialPair => {
                    mapper.end_reconstruction(&mut recon, db.correspondence_graph(), true);
                    break;
                }
            }
        }
    }

    let summary = format!(
        "SUMMARY path_a_attempts={} path_a_registered={} path_b_attempts={} path_b_registered={} path_b_rejected_lt6_corrs={}",
        mapper.path_a_attempts,
        mapper.path_a_registered,
        mapper.path_b_attempts,
        mapper.path_b_registered,
        mapper.path_b_rejected_lt6_corrs
    );
    eprintln!("{summary}");
    let mut log = mapper.log;
    log.push(summary);

    RunResult { models: kept, log }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;
    use visloc_tracking::umeyama_similarity_transform;

    use crate::colmap_incremental::test_support::build_synthetic_rig_scene;

    /// C2 task item 7: "mapper end-to-end on a small synthetic rig sequence
    /// (e.g. 20 frames, 2 cameras, known points) recovers metric scale
    /// within 1% and all frames." Runs the full `pipeline::run` — initial
    /// pair search, incremental registration (Path B, since every synthetic
    /// frame is a 2-camera rig with perfectly known intrinsics — see
    /// `mapper.rs`'s module doc), triangulation, local/global BA — against a
    /// from-scratch synthetic `DatabaseCache`, then Umeyama-aligns (with
    /// scale) the recovered frame-center trajectory against ground truth.
    #[test]
    fn pipeline_recovers_all_frames_and_metric_scale() {
        let scene = build_synthetic_rig_scene(20, 4);
        let mut options = PipelineOptions::default();
        // COLMAP's `init_min_num_inliers=100` default assumes real feature
        // counts; this synthetic scene has only 45 points total. Scale the
        // init/registration inlier floors down proportionally (control's
        // `abs_pose_min_num_inliers=8` is already well under 45 and is left
        // untouched) — the property under test is metric-scale recovery and
        // full registration, not COLMAP's real-world inlier-count tuning.
        options.mapper.init_min_num_inliers = 20;
        // Likewise `init_min_tri_angle=16deg` assumes real-scene parallax;
        // this synthetic sequence's frame-to-frame motion tops out around
        // 4-5deg by construction (see `test_support.rs`). Lowered, not
        // eliminated, so the gate still exercises a real (if smaller)
        // threshold check.
        options.mapper.init_min_tri_angle_deg = 1.0;

        let result = run(&options, &scene.db);
        assert!(
            !result.models.is_empty(),
            "expected at least one kept model"
        );
        assert!(!result.log.is_empty(), "expected non-empty run log");
        let model = &result.models[0].reconstruction;
        assert_eq!(
            model.num_reg_frames(),
            scene.images_per_frame.len(),
            "expected every synthetic frame registered"
        );

        let mut source = Vec::new();
        let mut target = Vec::new();
        for (&frame_id, gt) in &scene.ground_truth_rig_from_world {
            let got = model.frame(frame_id).rig_from_world();
            source.push(Point3::from(got.inverse().translation));
            target.push(Point3::from(gt.inverse().translation));
        }
        let transform = umeyama_similarity_transform(&source, &target, true)
            .expect("Umeyama alignment must succeed on a full trajectory");
        assert!(
            (transform.scale - 1.0).abs() < 0.01,
            "recovered scale {} not within 1% of ground truth 1.0",
            transform.scale
        );

        let mut max_err = 0.0f64;
        for (s, t) in source.iter().zip(target.iter()) {
            max_err = max_err.max((transform.apply(s) - t).norm());
        }
        assert!(
            max_err < 0.02,
            "max Umeyama-aligned frame-center error {max_err} too large"
        );
    }
}
