//! Real-data deep-frontend query-side localization demo.
//!
//! Loads the COLMAP South Building public dataset's sparse model
//! (`<root>/sparse/{cameras,images,points3D}.txt`) plus its 128 source
//! photos, picks one image as the *map source* and another as the
//! *query*, runs feature extraction + matching + PnP RANSAC for both
//! the classical (`CornerFeatureExtractor` + `BruteForceMatcher`) and
//! deep (`HogLikeFeatureExtractor` + `MutualSoftmaxMatcher`) and
//! multi-scale deep (`MultiScaleDeepExtractor<HogLikeFeatureExtractor>`
//! with `MutualSoftmaxMatcher`) frontends, and reports translation /
//! rotation / inlier-ratio errors against COLMAP's reference pose.
//!
//! Fetch the dataset once with:
//!     mkdir -p ~/datasets/south-building && cd ~/datasets/south-building && \
//!     curl -L -o south-building.zip \
//!       https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip && \
//!     unzip south-building.zip
//!
//! Run a single pair:
//!     cargo run --release --features image-io --example deep_localization_demo -- \
//!         --root ~/datasets/south-building/south-building \
//!         --map-image P1180141.JPG --query-image P1180155.JPG
//!
//! Or sweep a hard-coded grid of (map, query) pairs at multiple
//! viewpoint distances:
//!     cargo run --release --features image-io --example deep_localization_demo -- \
//!         --root ~/datasets/south-building/south-building --sweep \
//!         --out-dir target/deep_localization_sweep
//!
//! Optional `--out-dir <dir>` writes a summary.txt (single-pair) or
//! pairs.csv + summary_by_gap.txt (sweep) for downstream comparison.

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!("deep_localization_demo requires --features image-io");
    std::process::exit(2);
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    inner::run()
}

#[cfg(feature = "image-io")]
mod inner {
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};

    use nalgebra::Point2;
    use visloc_rs::core::geometry::Pose;
    use visloc_rs::core::types::QueryImage;
    use visloc_rs::core::types::{Camera, Keyframe, Landmark, VisualMap};
    use visloc_rs::io::colmap::read_colmap_text_model;
    use visloc_rs::io::images::read_common_image;
    use visloc_rs::vision::features::{
        CornerFeatureConfig, CornerFeatureExtractor, GrayscaleImage, HogLikeFeatureConfig,
        HogLikeFeatureExtractor, MultiScaleDeepConfig, MultiScaleDeepExtractor,
    };
    use visloc_rs::vision::matching::{
        BruteForceMatcher, MutualSoftmaxConfig, MutualSoftmaxMatcher,
    };
    use visloc_rs::{AllLandmarksSelector, LocalizationPipeline, LocalizationResult, PnPRansac};

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = parse_args()?;
        let sparse_dir = args.root.join("sparse");
        let images_dir = args.root.join("images");
        let map = read_colmap_text_model(&sparse_dir)?;
        println!(
            "loaded COLMAP model: cameras={} keyframes={} landmarks={}",
            map.cameras.len(),
            map.keyframes.len(),
            map.landmarks.len()
        );

        let camera_id = *map
            .cameras
            .keys()
            .next()
            .ok_or("no camera in COLMAP model")?;
        let camera = map.cameras[&camera_id].clone();
        println!(
            "camera: {}x{} model={:?} f≈{:.1}",
            camera.width,
            camera.height,
            camera.model,
            camera.intrinsics().map(|(f, _, _, _)| f).unwrap_or(0.0)
        );
        let images_txt = sparse_dir.join("images.txt");

        if args.sweep {
            run_sweep(&args, &images_dir, &images_txt, &map, &camera)?;
            return Ok(());
        }

        run_single_pair(
            &args.map_image,
            &args.query_image,
            &images_dir,
            &images_txt,
            &map,
            &camera,
            args.out_dir.as_deref(),
        )
    }

    fn run_single_pair(
        map_image_name: &str,
        query_image_name: &str,
        images_dir: &Path,
        images_txt: &Path,
        map: &VisualMap,
        camera: &Camera,
        out_dir: Option<&Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pair = load_pair(
            map_image_name,
            query_image_name,
            images_dir,
            images_txt,
            map,
        )?;

        println!(
            "map image    : {} ({} COLMAP observations)",
            map_image_name,
            pair.map_keyframe.observations.len()
        );
        println!(
            "query image  : {} ({} COLMAP observations)",
            query_image_name,
            pair.query_keyframe.observations.len()
        );
        if pair.map_image.width() != camera.width as usize
            || pair.map_image.height() != camera.height as usize
        {
            eprintln!(
                "# warning: image {}x{} differs from camera {}x{} — proceeding anyway",
                pair.map_image.width(),
                pair.map_image.height(),
                camera.width,
                camera.height,
            );
        }

        let mut results_with_keypoints: Vec<(
            ExtractorChoice,
            LocalizationResult,
            Vec<Point2<f64>>,
        )> = Vec::with_capacity(FRONTENDS.len());
        for frontend in FRONTENDS.iter() {
            let mut query_keypoints: Vec<Point2<f64>> = Vec::new();
            let result = run_pipeline(
                frontend.label(),
                camera,
                map,
                &pair.map_keyframe,
                &pair.query_keyframe,
                &pair.map_image,
                &pair.query_image,
                *frontend,
                Some(&mut query_keypoints),
            );
            results_with_keypoints.push((*frontend, result, query_keypoints));
        }
        let results: Vec<(ExtractorChoice, LocalizationResult)> = results_with_keypoints
            .iter()
            .map(|(frontend, result, _)| (*frontend, result.clone()))
            .collect();

        println!();
        println!("== Summary (truth from COLMAP `images.txt`) ==");
        for (frontend, result) in &results {
            print_diagnostics(frontend.label(), result, &pair.truth_pose);
        }

        if let Some(dir) = out_dir {
            write_summary(
                dir,
                map_image_name,
                query_image_name,
                &pair.truth_pose,
                &results,
            )?;
            println!("wrote {}/summary.txt", dir.display());
            write_correspondences_json(
                dir,
                map_image_name,
                query_image_name,
                &pair.map_keyframe,
                &results_with_keypoints,
            )?;
            println!("wrote {}/correspondences.json", dir.display());
        }
        Ok(())
    }

    /// Write a JSON file capturing per-frontend inlier matches so a
    /// downstream renderer (see `scripts/render_deep_localization_matches.py`)
    /// can draw real match lines between the map image and the query
    /// image. The map-side 2D position of each inlier landmark is the
    /// COLMAP observation's `xy` on the map keyframe (the same anchor
    /// the pipeline used to attach descriptors).
    fn write_correspondences_json(
        dir: &Path,
        map_image_name: &str,
        query_image_name: &str,
        map_keyframe: &Keyframe,
        results: &[(ExtractorChoice, LocalizationResult, Vec<Point2<f64>>)],
    ) -> std::io::Result<()> {
        // Build a lookup from landmark_id -> (map_x, map_y) for fast
        // per-inlier resolution. We pick the *first* observation seen
        // for each id; in COLMAP a landmark has exactly one observation
        // per source image, so the map_keyframe contributes one entry
        // per landmark in any case.
        let mut map_xy_by_id: std::collections::HashMap<u64, (f64, f64)> =
            std::collections::HashMap::new();
        for observation in &map_keyframe.observations {
            map_xy_by_id
                .entry(observation.landmark_id)
                .or_insert((observation.xy.x, observation.xy.y));
        }

        let mut body = String::new();
        body.push_str("{\n");
        body.push_str(&format!("  \"map_image\": {:?},\n", map_image_name));
        body.push_str(&format!("  \"query_image\": {:?},\n", query_image_name));
        body.push_str("  \"frontends\": [\n");
        for (idx, (frontend, result, query_keypoints)) in results.iter().enumerate() {
            body.push_str("    {\n");
            body.push_str(&format!("      \"id\": {:?},\n", frontend.id()));
            body.push_str(&format!("      \"label\": {:?},\n", frontend.label()));
            body.push_str(&format!("      \"match_count\": {},\n", result.match_count));
            body.push_str(&format!(
                "      \"inlier_count\": {},\n",
                result.inlier_count
            ));
            // For DeepMultiScale the synthetic landmark ids do not
            // appear in `map_xy_by_id` (they are derived from
            // `next_landmark_id`, not from `observation.landmark_id`),
            // so we skip those inliers — the renderer just shows
            // Classical vs Deep, which is the README A/B anyway.
            let mut inlier_pairs: Vec<((f64, f64), (f64, f64))> = Vec::new();
            for (query_idx, landmark_id) in result
                .inlier_query_indices
                .iter()
                .zip(result.inlier_landmark_ids.iter())
            {
                let Some(query_xy) = query_keypoints.get(*query_idx) else {
                    continue;
                };
                let Some(map_xy) = map_xy_by_id.get(landmark_id) else {
                    continue;
                };
                inlier_pairs.push(((query_xy.x, query_xy.y), *map_xy));
            }
            body.push_str(&format!(
                "      \"inlier_pairs_rendered\": {},\n",
                inlier_pairs.len()
            ));
            body.push_str("      \"inlier_pairs\": [\n");
            for (j, (qxy, mxy)) in inlier_pairs.iter().enumerate() {
                body.push_str(&format!(
                    "        {{\"query_xy\": [{:.3}, {:.3}], \"map_xy\": [{:.3}, {:.3}]}}",
                    qxy.0, qxy.1, mxy.0, mxy.1
                ));
                if j + 1 < inlier_pairs.len() {
                    body.push(',');
                }
                body.push('\n');
            }
            body.push_str("      ]\n");
            body.push_str("    }");
            if idx + 1 < results.len() {
                body.push(',');
            }
            body.push('\n');
        }
        body.push_str("  ]\n");
        body.push_str("}\n");

        fs::create_dir_all(dir)?;
        fs::write(dir.join("correspondences.json"), body)
    }

    struct LoadedPair {
        map_keyframe: Keyframe,
        query_keyframe: Keyframe,
        map_image: GrayscaleImage,
        query_image: GrayscaleImage,
        truth_pose: Pose,
    }

    fn load_pair(
        map_image_name: &str,
        query_image_name: &str,
        images_dir: &Path,
        images_txt: &Path,
        map: &VisualMap,
    ) -> Result<LoadedPair, Box<dyn std::error::Error>> {
        let map_frame_id =
            frame_id_for_image_name(images_txt, map_image_name).ok_or_else(|| {
                format!(
                    "map image '{}' not found in {:?}",
                    map_image_name, images_txt
                )
            })?;
        let query_frame_id =
            frame_id_for_image_name(images_txt, query_image_name).ok_or_else(|| {
                format!(
                    "query image '{}' not found in {:?}",
                    query_image_name, images_txt
                )
            })?;
        let map_keyframe = map
            .keyframes
            .get(&map_frame_id)
            .ok_or_else(|| format!("frame id {} missing from VisualMap", map_frame_id))?
            .clone();
        let query_keyframe = map
            .keyframes
            .get(&query_frame_id)
            .ok_or_else(|| format!("frame id {} missing from VisualMap", query_frame_id))?
            .clone();
        let truth_pose = query_keyframe
            .frame
            .pose
            .clone()
            .ok_or("query keyframe has no pose")?;
        let map_image = read_common_image(images_dir.join(map_image_name))?;
        let query_image = read_common_image(images_dir.join(query_image_name))?;
        Ok(LoadedPair {
            map_keyframe,
            query_keyframe,
            map_image,
            query_image,
            truth_pose,
        })
    }

    /// Hard-coded sweep grid: 5 map images × 5 viewpoint gaps = 25 pairs
    /// per frontend (50 pipeline runs). Pairs that drop below COLMAP's
    /// detection set or that cannot find the named image are silently
    /// skipped — those are dataset-level gaps, not pipeline failures, and
    /// would distort the per-gap summary if they appeared as 0-inlier
    /// rows.
    fn sweep_pairs() -> Vec<(String, String, u32)> {
        let map_indices: [u32; 5] = [141, 142, 143, 144, 145];
        let gaps: [u32; 5] = [1, 2, 3, 4, 5];
        let mut out = Vec::with_capacity(map_indices.len() * gaps.len());
        for &m in &map_indices {
            for &g in &gaps {
                out.push((
                    format!("P1180{:03}.JPG", m),
                    format!("P1180{:03}.JPG", m + g),
                    g,
                ));
            }
        }
        out
    }

    #[derive(Debug, Clone)]
    struct SweepRow {
        map_image: String,
        query_image: String,
        gap: u32,
        frontend: &'static str,
        success: bool,
        match_count: usize,
        inlier_count: usize,
        inlier_ratio: f64,
        translation_error_m: Option<f64>,
        rotation_error_rad: Option<f64>,
        reprojection_error_px: Option<f64>,
    }

    fn run_sweep(
        args: &CliArgs,
        images_dir: &Path,
        images_txt: &Path,
        map: &VisualMap,
        camera: &Camera,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pairs = sweep_pairs();
        println!();
        println!(
            "== Sweep mode: {} pairs × {} frontends = {} pipeline runs ==",
            pairs.len(),
            FRONTENDS.len(),
            pairs.len() * FRONTENDS.len()
        );
        let mut rows: Vec<SweepRow> = Vec::with_capacity(pairs.len() * FRONTENDS.len());
        for (map_name, query_name, gap) in &pairs {
            let pair = match load_pair(map_name, query_name, images_dir, images_txt, map) {
                Ok(p) => p,
                Err(err) => {
                    eprintln!("# skipping {} -> {}: {}", map_name, query_name, err);
                    continue;
                }
            };
            println!();
            println!(
                "[gap={}] map={} query={} ({} obs, {} kp on query)",
                gap,
                map_name,
                query_name,
                pair.map_keyframe.observations.len(),
                pair.query_keyframe.frame.keypoints.len()
            );
            for choice in FRONTENDS {
                let result = run_pipeline(
                    choice.label(),
                    camera,
                    map,
                    &pair.map_keyframe,
                    &pair.query_keyframe,
                    &pair.map_image,
                    &pair.query_image,
                    choice,
                    None,
                );
                let (translation_error_m, rotation_error_rad) =
                    match (result.success, result.pose.as_ref()) {
                        (true, Some(pose)) => (
                            Some(
                                (pose.camera_center_world()
                                    - pair.truth_pose.camera_center_world())
                                .norm(),
                            ),
                            Some(
                                pose.world_to_camera
                                    .rotation
                                    .rotation_to(&pair.truth_pose.world_to_camera.rotation)
                                    .angle(),
                            ),
                        ),
                        _ => (None, None),
                    };
                rows.push(SweepRow {
                    map_image: map_name.clone(),
                    query_image: query_name.clone(),
                    gap: *gap,
                    frontend: choice.id(),
                    success: result.success,
                    match_count: result.match_count,
                    inlier_count: result.inlier_count,
                    inlier_ratio: result.inlier_ratio,
                    translation_error_m,
                    rotation_error_rad,
                    reprojection_error_px: result.reprojection_error,
                });
            }
        }

        println!();
        print_sweep_table(&rows);

        if let Some(dir) = args.out_dir.as_ref() {
            write_sweep_csv(dir, &rows)?;
            write_sweep_summary(dir, &rows)?;
            println!(
                "wrote {0}/pairs.csv and {0}/summary_by_gap.txt",
                dir.display()
            );
        }
        Ok(())
    }

    fn print_sweep_table(rows: &[SweepRow]) {
        println!("== Per-pair results ==");
        println!(
            "{:<14} {:<14} {:>3} {:<10} {:>6} {:>4} {:>5} {:>9} {:>10} {:>10}",
            "map",
            "query",
            "gap",
            "frontend",
            "match",
            "in",
            "ratio",
            "transl_m",
            "rot_deg",
            "rep_px"
        );
        for r in rows {
            let transl = r
                .translation_error_m
                .map(|v| format!("{:>9.4}", v))
                .unwrap_or_else(|| format!("{:>9}", "-"));
            let rot_deg = r
                .rotation_error_rad
                .map(|v| format!("{:>10.4}", v.to_degrees()))
                .unwrap_or_else(|| format!("{:>10}", "-"));
            let rep = r
                .reprojection_error_px
                .map(|v| format!("{:>10.3}", v))
                .unwrap_or_else(|| format!("{:>10}", "-"));
            println!(
                "{:<14} {:<14} {:>3} {:<10} {:>6} {:>4} {:>5.2} {} {} {}",
                r.map_image,
                r.query_image,
                r.gap,
                r.frontend,
                r.match_count,
                r.inlier_count,
                r.inlier_ratio,
                transl,
                rot_deg,
                rep,
            );
        }

        println!();
        println!("== Aggregated by gap ==");
        println!(
            "{:>3} {:<10} {:>5} {:>9} {:>9} {:>11} {:>11}",
            "gap", "frontend", "n", "succ_rate", "mean_in", "mean_transl", "mean_rot_d"
        );
        for gap in 1..=5u32 {
            for frontend in FRONTENDS.map(|choice| choice.id()) {
                let group: Vec<&SweepRow> = rows
                    .iter()
                    .filter(|r| r.gap == gap && r.frontend == frontend)
                    .collect();
                if group.is_empty() {
                    continue;
                }
                let n = group.len();
                let succ = group.iter().filter(|r| r.success).count();
                let mean_inliers =
                    group.iter().map(|r| r.inlier_count as f64).sum::<f64>() / n as f64;
                let succ_group: Vec<&&SweepRow> = group.iter().filter(|r| r.success).collect();
                let mean_transl = if succ_group.is_empty() {
                    None
                } else {
                    Some(
                        succ_group
                            .iter()
                            .filter_map(|r| r.translation_error_m)
                            .sum::<f64>()
                            / succ_group.len() as f64,
                    )
                };
                let mean_rot_deg = if succ_group.is_empty() {
                    None
                } else {
                    Some(
                        succ_group
                            .iter()
                            .filter_map(|r| r.rotation_error_rad)
                            .sum::<f64>()
                            / succ_group.len() as f64
                            * 180.0
                            / std::f64::consts::PI,
                    )
                };
                let transl_str = mean_transl
                    .map(|v| format!("{:>11.4}", v))
                    .unwrap_or_else(|| format!("{:>11}", "-"));
                let rot_str = mean_rot_deg
                    .map(|v| format!("{:>11.4}", v))
                    .unwrap_or_else(|| format!("{:>11}", "-"));
                println!(
                    "{:>3} {:<10} {:>5} {:>8.2} {:>9.1} {} {}",
                    gap,
                    frontend,
                    n,
                    succ as f64 / n as f64,
                    mean_inliers,
                    transl_str,
                    rot_str,
                );
            }
        }
    }

    fn write_sweep_csv(dir: &Path, rows: &[SweepRow]) -> std::io::Result<()> {
        fs::create_dir_all(dir)?;
        let mut body = String::from(
            "map_image,query_image,gap,frontend,success,match_count,inlier_count,inlier_ratio,translation_error_m,rotation_error_rad,reprojection_error_px\n",
        );
        for r in rows {
            body.push_str(&format!(
                "{},{},{},{},{},{},{},{:.6},{},{},{}\n",
                r.map_image,
                r.query_image,
                r.gap,
                r.frontend,
                r.success,
                r.match_count,
                r.inlier_count,
                r.inlier_ratio,
                r.translation_error_m
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_default(),
                r.rotation_error_rad
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_default(),
                r.reprojection_error_px
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_default(),
            ));
        }
        fs::write(dir.join("pairs.csv"), body)
    }

    fn write_sweep_summary(dir: &Path, rows: &[SweepRow]) -> std::io::Result<()> {
        let mut body = String::new();
        body.push_str("gap,frontend,n,success_rate,mean_inliers,mean_translation_error_m,mean_rotation_error_deg\n");
        for gap in 1..=5u32 {
            for frontend in FRONTENDS.map(|choice| choice.id()) {
                let group: Vec<&SweepRow> = rows
                    .iter()
                    .filter(|r| r.gap == gap && r.frontend == frontend)
                    .collect();
                if group.is_empty() {
                    continue;
                }
                let n = group.len();
                let succ = group.iter().filter(|r| r.success).count();
                let mean_inliers =
                    group.iter().map(|r| r.inlier_count as f64).sum::<f64>() / n as f64;
                let succ_group: Vec<&&SweepRow> = group.iter().filter(|r| r.success).collect();
                let mean_transl = if succ_group.is_empty() {
                    "".to_string()
                } else {
                    format!(
                        "{:.6}",
                        succ_group
                            .iter()
                            .filter_map(|r| r.translation_error_m)
                            .sum::<f64>()
                            / succ_group.len() as f64
                    )
                };
                let mean_rot_deg = if succ_group.is_empty() {
                    "".to_string()
                } else {
                    format!(
                        "{:.6}",
                        succ_group
                            .iter()
                            .filter_map(|r| r.rotation_error_rad)
                            .sum::<f64>()
                            / succ_group.len() as f64
                            * 180.0
                            / std::f64::consts::PI
                    )
                };
                body.push_str(&format!(
                    "{},{},{},{:.4},{:.2},{},{}\n",
                    gap,
                    frontend,
                    n,
                    succ as f64 / n as f64,
                    mean_inliers,
                    mean_transl,
                    mean_rot_deg,
                ));
            }
        }
        fs::write(dir.join("summary_by_gap.txt"), body)
    }

    #[derive(Debug)]
    struct CliArgs {
        root: PathBuf,
        map_image: String,
        query_image: String,
        out_dir: Option<PathBuf>,
        sweep: bool,
    }

    fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
        let mut root: Option<PathBuf> = None;
        let mut map_image: Option<String> = None;
        let mut query_image: Option<String> = None;
        let mut out_dir: Option<PathBuf> = None;
        let mut sweep = false;
        let mut iter = env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--root" => root = iter.next().map(PathBuf::from),
                "--map-image" => map_image = iter.next(),
                "--query-image" => query_image = iter.next(),
                "--out-dir" => out_dir = iter.next().map(PathBuf::from),
                "--sweep" => sweep = true,
                other => return Err(format!("unrecognised flag {other}").into()),
            }
        }
        Ok(CliArgs {
            root: root.ok_or("--root <south-building dir> is required")?,
            map_image: map_image.unwrap_or_else(|| "P1180141.JPG".to_string()),
            query_image: query_image.unwrap_or_else(|| "P1180155.JPG".to_string()),
            out_dir,
            sweep,
        })
    }

    /// Map COLMAP image filename → frame id by re-scanning the raw
    /// `sparse/images.txt`. The visloc COLMAP parser drops the filename
    /// (it only keeps `frame_id`, pose, and 2D observations), so we
    /// recover the name → id mapping by parsing the same file
    /// independently here. Returns `None` when the file can't be read or
    /// the name doesn't appear.
    fn frame_id_for_image_name(images_txt: &Path, name: &str) -> Option<u64> {
        let contents = fs::read_to_string(images_txt).ok()?;
        for header in contents
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        {
            let tokens: Vec<&str> = header.split_whitespace().collect();
            if tokens.len() < 10 {
                continue;
            }
            // Header lines have token[0] as `image_id` (u64) and token[9]
            // as the `.JPG` filename; 2D points lines that follow each
            // header have float-prefixed tokens, so a failed integer
            // parse here means we should skip the line, not abort the
            // search.
            let Ok(parsed_id) = tokens[0].parse::<u64>() else {
                continue;
            };
            if tokens[9] == name {
                return Some(parsed_id);
            }
        }
        None
    }

    const FRONTENDS: [ExtractorChoice; 3] = [
        ExtractorChoice::Classical,
        ExtractorChoice::Deep,
        ExtractorChoice::DeepMultiScale,
    ];

    #[derive(Debug, Clone, Copy)]
    enum ExtractorChoice {
        Classical,
        Deep,
        DeepMultiScale,
    }

    impl ExtractorChoice {
        fn id(self) -> &'static str {
            match self {
                ExtractorChoice::Classical => "classical",
                ExtractorChoice::Deep => "deep",
                ExtractorChoice::DeepMultiScale => "deep-ms",
            }
        }

        fn label(self) -> &'static str {
            match self {
                ExtractorChoice::Classical => "Classical (Corner + BF)",
                ExtractorChoice::Deep => "Deep (HogLike + MutualSoftmax)",
                ExtractorChoice::DeepMultiScale => "Deep-MS (MultiScale HogLike + MutualSoftmax)",
            }
        }

        fn expands_landmarks(self) -> bool {
            matches!(self, ExtractorChoice::DeepMultiScale)
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// `query_keypoints_out`, when supplied, is populated with the
    /// query keypoints actually fed into the pipeline (one entry per
    /// described query feature, in the same order the pipeline sees
    /// them). The caller uses it together with the
    /// `LocalizationResult::inlier_query_indices` to look up the 2D
    /// pixel position of each inlier match for the
    /// correspondence-export rendering path.
    fn run_pipeline(
        label: &str,
        camera: &Camera,
        map: &VisualMap,
        map_keyframe: &Keyframe,
        query_keyframe: &Keyframe,
        map_image: &GrayscaleImage,
        query_image: &GrayscaleImage,
        choice: ExtractorChoice,
        query_keypoints_out: Option<&mut Vec<Point2<f64>>>,
    ) -> LocalizationResult {
        println!();
        println!("-- {} --", label);

        // Anchor each landmark's descriptor at the COLMAP-detected
        // keypoint location for that landmark on the map image, by
        // calling our extractor's `describe_at(image, cx, cy)`. This
        // sidesteps the SimpleRadial → pinhole projection mismatch (the
        // map camera has `k = -0.020` distortion which displaces edge
        // pixels by ~30 px from a pinhole-only projection, but the
        // COLMAP-supplied 2D location is already in the *distorted*
        // image frame). We use COLMAP locations only for the descriptor
        // anchor; the matcher still consumes our extractor's output on
        // the query side.
        let mut anchored_map = (*map).clone();
        anchored_map.landmarks.clear();
        let mut anchored = 0_usize;
        let mut next_landmark_id = map
            .landmarks
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        for observation in &map_keyframe.observations {
            let cx = observation.xy.x.round() as i64;
            let cy = observation.xy.y.round() as i64;
            if cx < 0 || cy < 0 {
                continue;
            }
            let cx = cx as usize;
            let cy = cy as usize;
            let descriptors = match choice {
                ExtractorChoice::Classical => CornerFeatureExtractor::new(corner_config())
                    .describe_at(map_image, cx, cy)
                    .into_iter()
                    .collect::<Vec<_>>(),
                ExtractorChoice::Deep => HogLikeFeatureExtractor::new(hog_config())
                    .describe_at(map_image, cx, cy)
                    .into_iter()
                    .collect::<Vec<_>>(),
                ExtractorChoice::DeepMultiScale => {
                    multiscale_hog_extractor().describe_at(map_image, cx, cy)
                }
            };
            if descriptors.is_empty() {
                continue;
            }
            let Some(source_landmark) = map.landmarks.get(&observation.landmark_id) else {
                continue;
            };
            for descriptor in descriptors {
                let landmark_id = if choice.expands_landmarks() {
                    let synthetic_id = next_landmark_id;
                    next_landmark_id = next_landmark_id.saturating_add(1);
                    synthetic_id
                } else {
                    observation.landmark_id
                };
                let mut landmark = Landmark::new(landmark_id, source_landmark.position);
                landmark.descriptor = Some(descriptor);
                anchored_map.landmarks.insert(landmark_id, landmark);
                anchored += 1;
            }
        }
        println!(
            "  landmarks with anchored descriptor: {} / {} observations",
            anchored,
            map_keyframe.observations.len()
        );

        // Real-data PnP RANSAC needs more iterations + a more permissive
        // reprojection threshold than the synthetic defaults: the
        // matcher's correspondence set has a much higher outlier ratio
        // on real images (descriptors are similar but not identical
        // across viewpoint change), and 4-pixel threshold is too tight
        // when the camera has 2560 px focal and 3072 px width.
        let pnp = PnPRansac {
            iterations: 65536,
            reprojection_threshold: 12.0,
            ..PnPRansac::default()
        };
        // Build the query image's (keypoints, descriptors) at COLMAP's
        // SIFT-detected keypoint locations on the query image. We anchor
        // descriptors on both sides at the same keypoint detector's
        // output (COLMAP's), so the only moving variable in the
        // classical-vs-deep comparison is the descriptor + matcher
        // pair, not which detector found the keypoints. This is the
        // right design for "what does the deep descriptor add over
        // classical at the localization layer" rather than "does
        // HogLike find different corners than Corner".
        let mut query_keypoints: Vec<Point2<f64>> = Vec::new();
        let mut query_descriptors: Vec<Vec<f32>> = Vec::new();
        // Use *all* COLMAP-detected keypoints (frame.keypoints) so the
        // matcher has the entire SIFT detection pool to match against,
        // not just landmark-anchored ones (which would lose query
        // candidates whose anchor descriptor we couldn't compute).
        for kp in &query_keyframe.frame.keypoints {
            let cx = kp.x.round() as i64;
            let cy = kp.y.round() as i64;
            if cx < 0 || cy < 0 {
                continue;
            }
            let cx = cx as usize;
            let cy = cy as usize;
            let descriptors = match choice {
                ExtractorChoice::Classical => CornerFeatureExtractor::new(corner_config())
                    .describe_at(query_image, cx, cy)
                    .into_iter()
                    .collect::<Vec<_>>(),
                ExtractorChoice::Deep => HogLikeFeatureExtractor::new(hog_config())
                    .describe_at(query_image, cx, cy)
                    .into_iter()
                    .collect::<Vec<_>>(),
                ExtractorChoice::DeepMultiScale => {
                    multiscale_hog_extractor().describe_at(query_image, cx, cy)
                }
            };
            for descriptor in descriptors {
                query_keypoints.push(*kp);
                query_descriptors.push(descriptor);
            }
        }
        println!(
            "  query keypoints described: {} / {}",
            query_keypoints.len(),
            query_keyframe.frame.keypoints.len()
        );
        if let Some(out) = query_keypoints_out {
            *out = query_keypoints.clone();
        }
        let query = QueryImage {
            camera: camera.clone(),
            keypoints: query_keypoints,
            descriptors: query_descriptors,
        };

        let result = match choice {
            ExtractorChoice::Classical => {
                let pipeline =
                    LocalizationPipeline::<BruteForceMatcher, AllLandmarksSelector, PnPRansac> {
                        matcher: BruteForceMatcher { ratio: Some(0.8) },
                        candidate_selector: AllLandmarksSelector,
                        pose_estimator: pnp,
                        config: visloc_rs::LocalizationConfig::default(),
                    };
                pipeline.localize(&query, &anchored_map)
            }
            ExtractorChoice::Deep | ExtractorChoice::DeepMultiScale => {
                let pipeline =
                    LocalizationPipeline::<MutualSoftmaxMatcher, AllLandmarksSelector, PnPRansac> {
                        matcher: MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
                            temperature: 25.0,
                            min_confidence: 0.15,
                            emit_ratio_metadata: false,
                        }),
                        candidate_selector: AllLandmarksSelector,
                        pose_estimator: pnp,
                        config: visloc_rs::LocalizationConfig::default(),
                    };
                pipeline.localize(&query, &anchored_map)
            }
        };

        println!(
            "  matches: {}  inliers: {}  reprojection: {:?}",
            result.match_count, result.inlier_count, result.reprojection_error
        );
        result
    }

    fn corner_config() -> CornerFeatureConfig {
        CornerFeatureConfig {
            max_features: 4000,
            min_score: 0.02,
            descriptor_radius: 9,
        }
    }

    fn hog_config() -> HogLikeFeatureConfig {
        HogLikeFeatureConfig {
            max_features: 4000,
            min_corner_score: 0.02,
            descriptor_clip: 0.2,
            orient: false,
        }
    }

    fn multiscale_hog_extractor() -> MultiScaleDeepExtractor<HogLikeFeatureExtractor> {
        MultiScaleDeepExtractor::new(
            HogLikeFeatureExtractor::new(hog_config()),
            MultiScaleDeepConfig {
                octaves: 3,
                area_weighted_octave_cap: false,
            },
        )
    }

    fn print_diagnostics(label: &str, result: &LocalizationResult, truth: &Pose) {
        println!("-- {} --", label);
        if let (true, Some(pose)) = (result.success, result.pose.as_ref()) {
            let estimated_translation = pose.camera_center_world();
            let truth_translation = truth.camera_center_world();
            let translation_error = (estimated_translation - truth_translation).norm();
            let rotation_error = pose
                .world_to_camera
                .rotation
                .rotation_to(&truth.world_to_camera.rotation)
                .angle();
            println!(
                "  matches : {}  inliers : {} ({:.2})",
                result.match_count,
                result.inlier_count,
                result.inlier_count as f64 / result.match_count.max(1) as f64
            );
            println!("  translation error : {:.4} m", translation_error);
            println!(
                "  rotation error    : {:.4} rad ({:.2} deg)",
                rotation_error,
                rotation_error.to_degrees()
            );
            if let Some(rep_error) = result.reprojection_error {
                println!("  mean reprojection : {:.4} px", rep_error);
            }
        } else {
            println!("  FAILED — no pose returned");
            println!("  matches : {}", result.match_count);
        }
    }

    fn write_summary(
        dir: &Path,
        map_image: &str,
        query_image: &str,
        truth: &Pose,
        results: &[(ExtractorChoice, LocalizationResult)],
    ) -> std::io::Result<()> {
        fs::create_dir_all(dir)?;
        let mut body = String::new();
        body.push_str(&format!(
            "map_image={} query_image={}\n",
            map_image, query_image
        ));
        let truth_translation = truth.camera_center_world();
        body.push_str(&format!(
            "truth_translation x={:.4} y={:.4} z={:.4}\n",
            truth_translation.x, truth_translation.y, truth_translation.z,
        ));
        for (frontend, result) in results {
            body.push_str(&format!(
                "[{}]\n  matches={} inliers={}\n",
                frontend.id(),
                result.match_count,
                result.inlier_count
            ));
            if let (true, Some(pose)) = (result.success, result.pose.as_ref()) {
                let est = pose.camera_center_world();
                let translation_error = (est - truth_translation).norm();
                let rotation_error = pose
                    .world_to_camera
                    .rotation
                    .rotation_to(&truth.world_to_camera.rotation)
                    .angle();
                body.push_str(&format!(
                    "  estimated_translation x={:.4} y={:.4} z={:.4}\n  translation_error_m={:.4} rotation_error_rad={:.4}\n",
                    est.x, est.y, est.z, translation_error, rotation_error
                ));
            } else {
                body.push_str("  FAILED\n");
            }
        }
        fs::write(dir.join("summary.txt"), body)
    }

    // Suppress unused-import warnings on `Point2` / `Landmark`; they are
    // referenced indirectly through the COLMAP map structure.
    fn _imports() {
        let _ = Point2::<f64>::new(0.0, 0.0);
        let _ = Landmark::new(0, nalgebra::Point3::new(0.0, 0.0, 0.0));
    }
}
