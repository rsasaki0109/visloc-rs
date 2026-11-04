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
//! `ReconstructSubModel` (`.cc:544-711`), `Reconstruct` (`.cc:713-820`).
//!
//! ## Deviations
//!
//! - **`structure_less_registration_fallback` (COLMAP 4 default `true`)
//!   is off by default** (`MapperOptions::structure_less_registration_fallback`):
//!   the port's frozen controls predate it. When on, a round in which no
//!   candidate registers from 2D-3D correspondences retries the candidates
//!   from 2D-2D ones (`IncrementalMapper::register_next_structure_less_image`,
//!   which lists its own deviations).
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
use super::mapper::{CandidateRound, IncrementalMapper, Options as MapperOptions};
use super::reconstruction::Reconstruction;
use super::types::{FrameT, ImageT};

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
    /// Not in COLMAP: end each model with one more global refinement whose
    /// BA runs to convergence (absolute gradient stop, up to
    /// [`FINAL_CONVERGE_MAX_ITERATIONS`] LM steps) instead of the
    /// relative `gradient_tolerance=1.0` that stops COLMAP's global BAs
    /// after one or two steps. Each of those starts from Ceres' initial
    /// trust radius, whose damping holds back the smooth, low-curvature
    /// modes of a long capture (scale drifting along a street), so no
    /// global BA of the incremental loop ever corrects them. Runs on
    /// models of at least this many registered frames
    /// (`ba_global_final_converge_min_frames`, default
    /// [`FINAL_CONVERGE_MIN_FRAMES`]); `0` runs it on every model, a number
    /// above any model's size turns it off.
    pub ba_global_final_converge_min_frames: usize,
    /// Not in COLMAP: once every model is built, extend the largest one
    /// with the images it left out ([`complete_model`]). The structure-less
    /// candidates there are the images with at least
    /// `2 * abs_pose_min_num_inliers` 2D-2D correspondences to registered
    /// images ([`CandidateRound::StructureLessByCorrespondences`]), the
    /// count the structure-less registration itself needs. COLMAP's filter
    /// (`abs_pose_min_num_inliers` visible 3D points) keeps out exactly the
    /// images that see too few 3D points. Running after the models are
    /// final, it does not change which models are built; a model it does
    /// not extend is left as it was. Off by default
    /// (`complete_largest_model`).
    pub complete_largest_model: bool,
    pub ba_global_frames_ratio: f64,
    pub ba_global_points_ratio: f64,
    pub ba_global_frames_freq: usize,
    pub ba_global_points_freq: usize,
    pub mapper: MapperOptions,
    pub triangulation: TriangulatorOptions,
    pub local_ba: BundleAdjustmentOptions,
    pub global_ba: BundleAdjustmentOptions,
    /// Diagnostic override: if set, `initialize_reconstruction` uses this
    /// exact `(image_id1, image_id2)` pair and **disables** the auto search
    /// for every initialization stage. A rejected pair returns
    /// `Status::NoInitialPair` for the current stage instead of falling back,
    /// so `run`'s existing relaxation loop (`Run`, `.cc:418-445`) retries the
    /// same pair with the relaxed `init_min_num_inliers`/`init_min_tri_angle`
    /// — e.g. COLMAP's own successful 5k pair has only 59 direct two-view
    /// inliers and is accepted only after the first relaxation halves
    /// `init_min_num_inliers` to 50. `None` (the default) leaves COLMAP's
    /// auto search untouched; this exists to A/B whether a downstream
    /// geometry difference is caused by the initial-pair choice (see
    /// `docs/colmap_rig_mapper_port_plan.md` §4.2 item 1).
    pub forced_init_pair: Option<(ImageT, ImageT)>,
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
            ba_global_final_converge_min_frames: FINAL_CONVERGE_MIN_FRAMES,
            complete_largest_model: false,
            ba_global_frames_ratio: 1.1,
            ba_global_points_ratio: 1.1,
            ba_global_frames_freq: 500,
            ba_global_points_freq: 250_000,
            mapper: MapperOptions::default(),
            triangulation: TriangulatorOptions::default(),
            local_ba: BundleAdjustmentOptions::local(),
            global_ba: BundleAdjustmentOptions::global(),
            forced_init_pair: None,
        }
    }
}

impl PipelineOptions {
    /// Apply one `key=value` override: COLMAP's `ba_refine_focal_length` /
    /// `ba_refine_principal_point` / `ba_refine_extra_params` /
    /// `ba_refine_sensor_from_rig` (`0`/`1`, on
    /// both the local and the global BA options), this port's
    /// `ba_global_final_converge_min_frames` (a frame count) and
    /// `complete_largest_model` (`0`/`1`), else
    /// [`MapperOptions::apply_override`].
    pub fn apply_override(&mut self, kv: &str) -> Result<(), String> {
        let flag = |value: &str, key: &str| {
            value
                .parse::<f64>()
                .map(|v| v != 0.0)
                .map_err(|e| format!("{key}: {e}"))
        };
        match kv.split_once('=') {
            Some((key @ "ba_refine_focal_length", value)) => {
                let on = flag(value, key)?;
                self.local_ba.refine_focal_length = on;
                self.global_ba.refine_focal_length = on;
                Ok(())
            }
            Some((key @ "ba_refine_principal_point", value)) => {
                let on = flag(value, key)?;
                self.local_ba.refine_principal_point = on;
                self.global_ba.refine_principal_point = on;
                Ok(())
            }
            Some((key @ "ba_refine_extra_params", value)) => {
                let on = flag(value, key)?;
                self.local_ba.refine_extra_params = on;
                self.global_ba.refine_extra_params = on;
                Ok(())
            }
            Some((key @ "ba_refine_sensor_from_rig", value)) => {
                let on = flag(value, key)?;
                self.local_ba.refine_sensor_from_rig = on;
                self.global_ba.refine_sensor_from_rig = on;
                Ok(())
            }
            Some((key @ "ba_global_final_converge_min_frames", value)) => {
                self.ba_global_final_converge_min_frames =
                    value.trim().parse().map_err(|e| format!("{key}: {e}"))?;
                Ok(())
            }
            Some((key @ "complete_largest_model", value)) => {
                self.complete_largest_model = flag(value, key)?;
                Ok(())
            }
            _ => self.mapper.apply_override(kv),
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
    let found = match options.forced_init_pair {
        Some((image_id1, image_id2)) => mapper
            .estimate_initial_two_view_geometry(
                mapper_options,
                recon,
                db.correspondence_graph(),
                image_id1,
                image_id2,
            )
            .map(|cam2_from_cam1| (image_id1, image_id2, cam2_from_cam1)),
        None => mapper.find_initial_image_pair(mapper_options, recon, db.correspondence_graph()),
    };
    let Some((image_id1, image_id2, cam2_from_cam1)) = found else {
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

/// Default [`PipelineOptions::ba_global_final_converge_min_frames`]: the
/// size at which global BAs switch to the iterative solver, below which the
/// incremental loop's direct solves leave little drift behind (ETH3D's
/// scenes, at most 76 frames, change only within noise with it on), while
/// SmallCity's 5,816 frames go from 0.611 m to 0.178 m median centre error.
pub const FINAL_CONVERGE_MIN_FRAMES: usize = 1000;

/// LM step cap of the [`PipelineOptions::ba_global_final_converge_min_frames`] BA.
/// A smooth drift needs the trust radius to grow by about 1e4 (three
/// accepted steps per decade) before it moves; the solve usually stops
/// earlier, on the function tolerance.
pub const FINAL_CONVERGE_MAX_ITERATIONS: usize = 50;

/// Port of the free function `IterativeGlobalRefinement`
/// (`incremental_pipeline.cc:84-94`).
fn iterative_global_refinement(
    options: &PipelineOptions,
    mapper_options: &MapperOptions,
    mapper: &mut IncrementalMapper,
    db: &DatabaseCache,
    recon: &mut Reconstruction,
) {
    let t0 = crate::time_compat::Instant::now();
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

        // Structure-based registration first; when every candidate fails,
        // the structure-less fallback (COLMAP 4) if enabled.
        let rounds: &[CandidateRound] = if mapper_options.structure_less_registration_fallback {
            &[CandidateRound::Structure, CandidateRound::StructureLess]
        } else {
            &[CandidateRound::Structure]
        };
        let t_reg = crate::time_compat::Instant::now();
        let mut attempts = 0usize;
        let mut find_ms = 0;
        let mut num_candidates = 0;
        for &round in rounds {
            let t_find = crate::time_compat::Instant::now();
            let next_images = mapper.find_next_images(mapper_options, recon, round);
            find_ms += t_find.elapsed().as_millis();
            num_candidates += next_images.len();
            for &candidate in &next_images {
                attempts += 1;
                let registered = if round != CandidateRound::Structure {
                    mapper.register_next_structure_less_image(
                        mapper_options,
                        recon,
                        db.correspondence_graph(),
                        candidate,
                    )
                } else {
                    mapper.register_next_image(
                        mapper_options,
                        recon,
                        db.correspondence_graph(),
                        candidate,
                    )
                };
                if registered {
                    reg_next_success = true;
                    registered_image_id = Some(candidate);
                    break;
                }
            }
            if reg_next_success {
                break;
            }
        }
        eprintln!("TIMING find_next_images elapsed_ms={find_ms} candidates={num_candidates}");
        eprintln!(
            "TIMING register_next_image elapsed_ms={} attempts={attempts} success={reg_next_success}",
            t_reg.elapsed().as_millis()
        );

        if reg_next_success {
            let image_id = registered_image_id.unwrap();
            let frame_id = recon.image(image_id).frame_id;
            let frame_images: Vec<ImageT> = recon.frame(frame_id).image_ids().collect();
            let t_tri = crate::time_compat::Instant::now();
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
            let t_local = crate::time_compat::Instant::now();
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

            // Diagnostic hook (10k boundary investigation): writing the
            // reconstruction immediately after a chosen frame is registered
            // lets us diff the frontier model state against COLMAP's control
            // model without rerunning COLMAP.
            if let (Some(target), Some(dir)) = (
                std::env::var("VISLOC_SNAPSHOT_AFTER_FRAME")
                    .ok()
                    .and_then(|spec| spec.parse::<FrameT>().ok()),
                std::env::var_os("VISLOC_SNAPSHOT_DIR"),
            ) {
                let image_id = registered_image_id.unwrap();
                if recon.image(image_id).frame_id == target {
                    let dir = std::path::PathBuf::from(dir);
                    let _ = std::fs::create_dir_all(&dir);
                    let _ = recon.export_colmap_text(&dir);
                    eprintln!(
                        "SNAPSHOT frame={} n_reg_frames={} points3d={} dir={}",
                        target,
                        recon.num_reg_frames(),
                        recon.num_points3d(),
                        dir.display()
                    );
                    std::process::exit(0);
                }
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

    if recon.num_reg_frames() >= options.ba_global_final_converge_min_frames.max(2) {
        // Diagnostic: the model as the incremental loop left it, to compare
        // with the converged one from the same run.
        if let Some(dir) = std::env::var_os("VISLOC_SNAPSHOT_BEFORE_FINAL") {
            let dir = std::path::PathBuf::from(dir).join(recon.num_reg_frames().to_string());
            let _ = std::fs::create_dir_all(&dir);
            let _ = recon.export_colmap_text(&dir);
        }
        let mut converge = options.clone();
        converge.ba_global_max_refinements = 1;
        converge.global_ba.gradient_tolerance_rel = None;
        converge.global_ba.max_num_iterations = FINAL_CONVERGE_MAX_ITERATIONS;
        iterative_global_refinement(&converge, mapper_options, mapper, db, recon);
    }

    Status::Success
}

/// [`PipelineOptions::complete_largest_model`]: registers what it can of
/// the images `recon` left out, each round structure-based first and then
/// structure-less over every image with enough 2D-2D correspondences. Each
/// registration is followed by triangulation and the local refinement as
/// in [`reconstruct_sub_model`]. An image that then sees fewer 3D points
/// than `abs_pose_min_num_inliers` is undone (the model goes back to its
/// state before the registration) and not tried again: on ETH3D's electro
/// the one image placed metres off kept 24 points, the ones placed within
/// centimetres 49 or more. Once no candidate registers, a global refinement
/// and one more round. Returns the number of frames registered; with none,
/// `recon` is unchanged.
fn complete_model(
    options: &PipelineOptions,
    mapper_options: &MapperOptions,
    mapper: &mut IncrementalMapper,
    db: &DatabaseCache,
    recon: &mut Reconstruction,
) -> usize {
    let graph = db.correspondence_graph();
    mapper.begin_reconstruction(recon, graph);
    let mut added = 0;
    let mut refined = false;
    let mut rejected = std::collections::BTreeSet::new();
    loop {
        let before = recon.clone();
        let mut registered = None;
        'rounds: for round in [
            CandidateRound::Structure,
            CandidateRound::StructureLessByCorrespondences,
        ] {
            for candidate in mapper.find_next_images(mapper_options, recon, round) {
                if rejected.contains(&candidate) {
                    continue;
                }
                let ok = if round == CandidateRound::Structure {
                    mapper.register_next_image(mapper_options, recon, graph, candidate)
                } else {
                    mapper.register_next_structure_less_image(
                        mapper_options,
                        recon,
                        graph,
                        candidate,
                    )
                };
                if ok {
                    registered = Some(candidate);
                    break 'rounds;
                }
            }
        }
        match registered {
            Some(image_id) => {
                let frame_id = recon.image(image_id).frame_id;
                let frame_images: Vec<ImageT> = recon.frame(frame_id).image_ids().collect();
                for iid in frame_images {
                    mapper.triangulate_image(&options.triangulation, recon, graph, iid);
                }
                mapper.iterative_local_refinement(
                    options.ba_local_max_refinements,
                    options.ba_local_max_refinement_change,
                    mapper_options,
                    &options.local_ba,
                    &options.triangulation,
                    recon,
                    graph,
                    image_id,
                );
                let seen = recon.image(image_id).num_points3d();
                if seen < mapper_options.abs_pose_min_num_inliers {
                    mapper.log.push(format!(
                        "COMPLETE undo frame={frame_id} image={image_id} points3d={seen}"
                    ));
                    *recon = before;
                    mapper.begin_reconstruction(recon, graph);
                    rejected.insert(image_id);
                    continue;
                }
                added += 1;
                refined = false;
            }
            None if added > 0 && !refined => {
                iterative_global_refinement(options, mapper_options, mapper, db, recon);
                refined = true;
            }
            None => break,
        }
    }
    if added > 0 && recon.num_reg_frames() >= options.ba_global_final_converge_min_frames.max(2) {
        let mut converge = options.clone();
        converge.ba_global_max_refinements = 1;
        converge.global_ba.gradient_tolerance_rel = None;
        converge.global_ba.max_num_iterations = FINAL_CONVERGE_MAX_ITERATIONS;
        iterative_global_refinement(&converge, mapper_options, mapper, db, recon);
    }
    mapper.end_reconstruction(recon, graph, false);
    added
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
    let mut relax3 = relax2.clone();
    relax3.init_min_num_inliers = (relax3.init_min_num_inliers / 2).max(1);
    let mut relax4 = relax3.clone();
    relax4.init_min_tri_angle_deg /= 2.0;
    // `Run()`'s outer loop (`.cc:418-445`): the original options, then
    // `kNumInitRelaxations=2` passes that each halve `init_min_num_inliers`
    // and then `init_min_tri_angle` — five stages in total:
    // (100,16), (50,16), (50,8), (25,8), (25,4).
    let variants = [relax0, relax1, relax2, relax3, relax4];

    'relax: for mapper_options in &variants {
        if std::env::var_os("VISLOC_DEBUG_INIT").is_some() {
            eprintln!(
                "run stage: init_min_num_inliers={} init_min_tri_angle_deg={}",
                mapper_options.init_min_num_inliers, mapper_options.init_min_tri_angle_deg
            );
        }
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

    if options.complete_largest_model {
        // The first of the largest models, the one a caller keeping the
        // largest would pick.
        let largest = (0..kept.len())
            .rev()
            .max_by_key(|&k| kept[k].reconstruction.num_reg_images());
        if let Some(k) = largest {
            let added = complete_model(
                options,
                &options.mapper,
                &mut mapper,
                db,
                &mut kept[k].reconstruction,
            );
            mapper
                .log
                .push(format!("COMPLETE model={k} registered_frames={added}"));
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
    use std::collections::BTreeMap;
    use visloc_core::geometry::SE3;
    use visloc_tracking::umeyama_similarity_transform;

    use crate::colmap_incremental::test_support::{
        build_synthetic_central_rig_scene, build_synthetic_lens_scene, build_synthetic_rig_scene,
        build_synthetic_spherical_scene, build_three_view_scene,
    };

    /// `complete_model` registers an image its model left out from 2D-2D
    /// correspondences alone (it sees no 3D point, so neither COLMAP's
    /// structure-based nor its structure-less round would try it), and
    /// leaves a model with nothing to add as it was.
    #[test]
    fn complete_model_registers_a_left_out_image() {
        let (db, poses, _) = build_three_view_scene();
        let mut recon = reconstruction_from_cache(&db);
        for (k, pose) in poses.iter().take(2).enumerate() {
            recon.frame_mut(k as u64).set_rig_from_world(pose.clone());
            recon.register_frame(k as u64);
        }
        let mut options = PipelineOptions::default();
        options.mapper.abs_pose_min_num_inliers = 20;
        let mut mapper = IncrementalMapper::new();

        let added = complete_model(&options, &options.mapper, &mut mapper, &db, &mut recon);
        assert_eq!(added, 1);
        assert!(recon.is_image_registered(2));
        let got = recon.frame(2).rig_from_world();
        let centre_error = (got.inverse().translation - poses[2].inverse().translation).norm();
        assert!(centre_error < 1e-3, "centre off by {centre_error}");

        let before: Vec<SE3> = (0..3)
            .map(|k| recon.frame(k).rig_from_world().clone())
            .collect();
        let points_before = recon.num_points3d();
        let added = complete_model(&options, &options.mapper, &mut mapper, &db, &mut recon);
        assert_eq!(added, 0);
        let after: Vec<SE3> = (0..3)
            .map(|k| recon.frame(k).rig_from_world().clone())
            .collect();
        assert_eq!(after, before);
        assert_eq!(recon.num_points3d(), points_before);
    }

    /// C2 task item 7: "mapper end-to-end on a small synthetic rig sequence
    /// (e.g. 20 frames, 2 cameras, known points) recovers metric scale
    /// within 1% and all frames." Runs the full `pipeline::run` — initial
    /// pair search, incremental registration (Path B, since every synthetic
    /// frame is a 2-camera rig with perfectly known intrinsics — see
    /// `mapper.rs`'s module doc), triangulation, local/global BA — against a
    /// from-scratch synthetic `DatabaseCache`, then Umeyama-aligns (with
    /// scale) the recovered frame-center trajectory against ground truth.
    #[test]
    fn final_converge_threshold_override() {
        let mut options = PipelineOptions::default();
        assert_eq!(
            options.ba_global_final_converge_min_frames,
            FINAL_CONVERGE_MIN_FRAMES
        );
        options
            .apply_override("ba_global_final_converge_min_frames=0")
            .unwrap();
        assert_eq!(options.ba_global_final_converge_min_frames, 0);
        assert!(options
            .apply_override("ba_global_final_converge_min_frames=on")
            .is_err());
    }

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

    /// A central rig (both cameras at one centre, e.g. two cube faces of a
    /// panorama) initializes from its ordinary two-view geometry, where
    /// GR6P is degenerate, and registers every frame with the cameras'
    /// relative rotation kept exact.
    #[test]
    fn pipeline_reconstructs_a_central_rig() {
        let scene = build_synthetic_central_rig_scene(12, 4);
        let mut options = PipelineOptions::default();
        // As in `pipeline_recovers_all_frames_and_metric_scale`.
        options.mapper.init_min_num_inliers = 20;
        options.mapper.init_min_tri_angle_deg = 1.0;

        let result = run(&options, &scene.db);
        assert!(!result.models.is_empty(), "expected a kept model");
        let model = &result.models[0].reconstruction;
        assert_eq!(model.num_reg_frames(), scene.images_per_frame.len());

        // Up to scale (a central rig has none of its own): the frame
        // centres after a similarity alignment, and the frames' rotations.
        let mut source = Vec::new();
        let mut target = Vec::new();
        let mut worst_rotation = 0.0f64;
        let first = |m: &BTreeMap<u64, SE3>| m.values().next().unwrap().rotation;
        let got_poses: BTreeMap<u64, SE3> = scene
            .ground_truth_rig_from_world
            .keys()
            .map(|&f| (f, model.frame(f).rig_from_world().clone()))
            .collect();
        for (&frame_id, gt) in &scene.ground_truth_rig_from_world {
            let got = &got_poses[&frame_id];
            source.push(Point3::from(got.inverse().translation));
            target.push(Point3::from(gt.inverse().translation));
            // Rotation relative to frame 0, free of the gauge.
            let rel_got = got.rotation * first(&got_poses).inverse();
            let rel_gt = gt.rotation * first(&scene.ground_truth_rig_from_world).inverse();
            worst_rotation = worst_rotation.max(rel_got.angle_to(&rel_gt));
        }
        let transform = umeyama_similarity_transform(&source, &target, true)
            .expect("Umeyama alignment must succeed on a full trajectory");
        let max_err = source
            .iter()
            .zip(&target)
            .map(|(s, t)| (transform.apply(s) - t).norm())
            .fold(0.0, f64::max);
        assert!(max_err < 0.02, "frame-centre error {max_err}");
        assert!(worst_rotation < 1e-3, "rotation error {worst_rotation} rad");
    }

    /// An equirectangular camera among points all around it: every frame
    /// registers, and the poses match the truth up to a similarity.
    #[test]
    fn pipeline_reconstructs_a_spherical_camera() {
        let (db, truth) = build_synthetic_spherical_scene(10, 4);
        let mut options = PipelineOptions::default();
        // 120 points per image, as in the other synthetic scenes.
        options.mapper.init_min_num_inliers = 40;
        options.mapper.init_min_tri_angle_deg = 2.0;
        options.mapper.abs_pose_min_num_inliers = 20;

        let result = run(&options, &db);
        assert!(!result.models.is_empty(), "expected a kept model");
        let model = &result.models[0].reconstruction;
        assert_eq!(model.num_reg_frames(), truth.len());
        let (mut source, mut target) = (Vec::new(), Vec::new());
        let mut worst_rotation = 0.0f64;
        let first_got = model.frame(0).rig_from_world().rotation;
        let first_gt = truth[&0].rotation;
        for (&frame_id, gt) in &truth {
            let got = model.frame(frame_id).rig_from_world();
            source.push(Point3::from(got.inverse().translation));
            target.push(Point3::from(gt.inverse().translation));
            let rel_got = got.rotation * first_got.inverse();
            let rel_gt = gt.rotation * first_gt.inverse();
            worst_rotation = worst_rotation.max(rel_got.angle_to(&rel_gt));
        }
        let transform = umeyama_similarity_transform(&source, &target, true)
            .expect("Umeyama alignment must succeed on a full trajectory");
        let max_err = source
            .iter()
            .zip(&target)
            .map(|(s, t)| (transform.apply(s) - t).norm())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-3, "frame-centre error {max_err} m");
        assert!(worst_rotation < 1e-4, "rotation error {worst_rotation} rad");
        // Points behind the camera were triangulated and kept.
        assert!(
            model.num_points3d() > 100,
            "{} points",
            model.num_points3d()
        );
    }

    /// Fisheye lenses seeing points up to 60° off their axis: a
    /// Kannala-Brandt one and ETH3D's thin-prism model. The mapper
    /// reconstructs the trajectory exactly; before the bundle adjustment
    /// followed the lens, it solved these cameras as pinholes.
    #[test]
    fn pipeline_reconstructs_through_fisheye_lenses() {
        use visloc_core::types::{Camera, CameraModel};
        let lenses = [
            Camera::opencv_fisheye(
                1,
                752,
                480,
                220.0,
                221.0,
                380.0,
                236.0,
                [0.03, -0.01, 0.002, -0.0005],
            ),
            Camera {
                id: 1,
                model: CameraModel::ThinPrismFisheye,
                width: 752,
                height: 480,
                params: vec![
                    220.0, 219.5, 372.0, 244.0, -0.0654, -0.0781, -0.0013, 0.0017, 0.1155, -0.0711,
                    -0.0103, 0.0070,
                ],
            },
        ];
        for camera in lenses {
            let (db, truth) = build_synthetic_lens_scene(camera.clone(), 60.0, 10, 4);
            let mut options = PipelineOptions::default();
            options.mapper.init_min_num_inliers = 40;
            options.mapper.init_min_tri_angle_deg = 2.0;
            options.mapper.abs_pose_min_num_inliers = 20;

            let result = run(&options, &db);
            assert!(!result.models.is_empty(), "{:?}: no model", camera.model);
            let model = &result.models[0].reconstruction;
            assert_eq!(model.num_reg_frames(), truth.len(), "{:?}", camera.model);
            let (mut source, mut target) = (Vec::new(), Vec::new());
            let mut worst_rotation = 0.0f64;
            let first_got = model.frame(0).rig_from_world().rotation;
            let first_gt = truth[&0].rotation;
            for (&frame_id, gt) in &truth {
                let got = model.frame(frame_id).rig_from_world();
                source.push(Point3::from(got.inverse().translation));
                target.push(Point3::from(gt.inverse().translation));
                let rel_got = got.rotation * first_got.inverse();
                let rel_gt = gt.rotation * first_gt.inverse();
                worst_rotation = worst_rotation.max(rel_got.angle_to(&rel_gt));
            }
            let transform = umeyama_similarity_transform(&source, &target, true)
                .expect("Umeyama alignment must succeed on a full trajectory");
            let max_err = source
                .iter()
                .zip(&target)
                .map(|(s, t)| (transform.apply(s) - t).norm())
                .fold(0.0, f64::max);
            assert!(
                max_err < 1e-3,
                "{:?}: frame-centre error {max_err} m",
                camera.model
            );
            assert!(
                worst_rotation < 1e-4,
                "{:?}: rotation error {worst_rotation} rad",
                camera.model
            );
            assert!(
                model.num_points3d() > 100,
                "{} points",
                model.num_points3d()
            );
            // The lens is left as calibrated.
            assert_eq!(model.camera(1).params, camera.params);
        }
    }

    /// A fisheye wider than 180° (Kannala-Brandt, 640 px square at f = 150:
    /// ±122° across its sides) seeing points up to 100° off its axis, about
    /// a fifth of them behind its image plane. The mapper takes it on unit
    /// rays, as it does a panorama (`Camera::sees_behind`): the trajectory
    /// comes out exact, and the points behind the image plane are
    /// triangulated too.
    #[test]
    fn pipeline_reconstructs_through_a_fisheye_past_180_degrees() {
        use visloc_core::types::Camera;
        let camera = Camera::opencv_fisheye(
            1,
            640,
            640,
            150.0,
            150.5,
            322.0,
            318.0,
            [0.02, -0.004, 0.0005, -0.0001],
        );
        assert!(camera.sees_behind());
        let (db, truth) = build_synthetic_lens_scene(camera, 100.0, 10, 4);
        let mut options = PipelineOptions::default();
        options.mapper.init_min_num_inliers = 40;
        options.mapper.init_min_tri_angle_deg = 2.0;
        options.mapper.abs_pose_min_num_inliers = 20;

        let result = run(&options, &db);
        assert!(!result.models.is_empty(), "no model");
        let model = &result.models[0].reconstruction;
        assert_eq!(model.num_reg_frames(), truth.len());
        let (mut source, mut target) = (Vec::new(), Vec::new());
        let mut worst_rotation = 0.0f64;
        let first_got = model.frame(0).rig_from_world().rotation;
        let first_gt = truth[&0].rotation;
        for (&frame_id, gt) in &truth {
            let got = model.frame(frame_id).rig_from_world();
            source.push(Point3::from(got.inverse().translation));
            target.push(Point3::from(gt.inverse().translation));
            let rel_got = got.rotation * first_got.inverse();
            let rel_gt = gt.rotation * first_gt.inverse();
            worst_rotation = worst_rotation.max(rel_got.angle_to(&rel_gt));
        }
        let transform = umeyama_similarity_transform(&source, &target, true)
            .expect("Umeyama alignment must succeed on a full trajectory");
        let max_err = source
            .iter()
            .zip(&target)
            .map(|(s, t)| (transform.apply(s) - t).norm())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-3, "frame-centre error {max_err} m");
        assert!(worst_rotation < 1e-4, "rotation error {worst_rotation} rad");
        let first = model.frame(0).rig_from_world();
        let behind = model
            .point3d_ids()
            .into_iter()
            .filter(|&id| first.transform_point(&model.point3d(id).xyz).z < 0.0)
            .count();
        assert!(
            model.num_points3d() > 100 && behind > 15,
            "{} points, {behind} behind frame 0's image plane",
            model.num_points3d()
        );
    }

    /// `forced_init_pair` (plan §4.2 item 1 diagnostic) must seed the
    /// reconstruction with exactly the requested image pair, and fall back to
    /// the auto search only when the forced pair is rejected.
    #[test]
    fn forced_init_pair_overrides_the_auto_search() {
        let scene = build_synthetic_rig_scene(12, 4);
        let mut options = PipelineOptions::default();
        options.mapper.init_min_num_inliers = 20;
        options.mapper.init_min_tri_angle_deg = 1.0;

        // Auto search first, to learn which pair it would pick.
        let auto = run(&options, &scene.db);
        let auto_init = auto
            .log
            .iter()
            .find(|line| line.starts_with("INIT_PAIR"))
            .expect("auto run logs an INIT_PAIR")
            .clone();

        // Force a cross-frame cam1/cam1 pair (frames 3 and 6, within the
        // scene's 4-frame correspondence window) that the auto search is free
        // to pick or not.
        let (forced1, _) = scene.images_per_frame[3];
        let (forced2, _) = scene.images_per_frame[6];
        options.forced_init_pair = Some((forced1, forced2));
        let result = run(&options, &scene.db);
        assert!(!result.models.is_empty(), "forced run kept no model");
        let forced_init = result
            .log
            .iter()
            .find(|line| line.starts_with("INIT_PAIR"))
            .expect("forced run logs an INIT_PAIR");
        assert!(
            forced_init.contains(&format!("image1={forced1} image2={forced2}")),
            "forced pair {forced1}/{forced2} not honored; got {forced_init:?} (auto was {auto_init:?})"
        );
    }

    /// A forced pair that the initialization gates reject in every relaxation
    /// stage must produce no model rather than silently falling back to the
    /// auto search (`forced_init_pair` is "force only", see its doc).
    #[test]
    fn forced_init_pair_does_not_fall_back_to_auto() {
        let scene = build_synthetic_rig_scene(12, 4);
        let mut options = PipelineOptions::default();
        options.mapper.init_min_num_inliers = 20;
        options.mapper.init_min_tri_angle_deg = 1.0;
        let last = scene.images_per_frame.len() - 1;
        // Frames 0 and 11 are farther apart than the scene's 4-frame
        // correspondence window, so this pair has no matches at all.
        options.forced_init_pair =
            Some((scene.images_per_frame[0].0, scene.images_per_frame[last].0));

        let mut sanity_options = options.clone();
        sanity_options.forced_init_pair = None;
        let auto = run(&sanity_options, &scene.db);
        assert!(
            !auto.models.is_empty(),
            "sanity: the auto search finds a model"
        );

        let result = run(&options, &scene.db);
        assert!(
            result.models.is_empty(),
            "invalid forced pair must not fall back to the auto search"
        );
        assert!(
            !result.log.iter().any(|line| line.starts_with("INIT_PAIR")),
            "no pair should have been registered"
        );
    }
}
