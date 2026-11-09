//! Command-line parsing and output-path validation.

use super::*;

pub(super) fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = args.into_iter();
    let _program = values.next();
    let mut rig_manifest = None;
    let mut nodes_tsv = None;
    let mut atlas_dir = None;
    let mut out_dir = None;
    let mut pre_ba_out_dir = None;
    let mut post_pass1_out_dir = None;
    let mut recover_zero_support_frames = false;
    let mut joint_rig_ba = false;
    let mut joint_rig_ba_filter_observations = false;
    let mut joint_rig_ba_preserve_optimized_points = false;
    let mut joint_rig_ba_filter_sweeps = 1u8;
    let mut filter_sweeps_explicit = false;
    let mut diagnose_cross_boundary_pnp = false;
    let mut diagnostic_left_max_frame = None;
    let mut repair_cross_boundary = false;
    let mut repair_left_max_frame = None;
    while let Some(flag) = values.next() {
        if flag == "-h" || flag == "--help" {
            return Err(USAGE.to_owned());
        }
        if flag == "--recover-zero-support-frames" {
            if recover_zero_support_frames {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            recover_zero_support_frames = true;
            continue;
        }
        if flag == "--joint-rig-ba" {
            if joint_rig_ba {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            joint_rig_ba = true;
            continue;
        }
        if flag == "--joint-rig-ba-filter-observations" {
            if joint_rig_ba_filter_observations {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            joint_rig_ba_filter_observations = true;
            continue;
        }
        if flag == "--joint-rig-ba-preserve-optimized-points" {
            if joint_rig_ba_preserve_optimized_points {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            joint_rig_ba_preserve_optimized_points = true;
            continue;
        }
        if flag == "--pre-ba-out-dir" {
            if pre_ba_out_dir.is_some() {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires PATH\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!("{flag} requires PATH, got {value:?}\n{USAGE}"));
            }
            pre_ba_out_dir = Some(PathBuf::from(value));
            continue;
        }
        if flag == "--post-pass1-out-dir" {
            if post_pass1_out_dir.is_some() {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires PATH\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!("{flag} requires PATH, got {value:?}\n{USAGE}"));
            }
            post_pass1_out_dir = Some(PathBuf::from(value));
            continue;
        }
        if flag == "--joint-rig-ba-filter-sweeps" {
            if filter_sweeps_explicit {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires 1 or 2\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!("{flag} requires 1 or 2, got {value:?}\n{USAGE}"));
            }
            joint_rig_ba_filter_sweeps = value
                .parse::<u8>()
                .map_err(|error| format!("{flag} requires 1 or 2: {error}\n{USAGE}"))?;
            if !matches!(joint_rig_ba_filter_sweeps, 1 | 2) {
                return Err(format!("{flag} accepts only 1 or 2\n{USAGE}"));
            }
            filter_sweeps_explicit = true;
            continue;
        }
        if flag == "--diagnose-cross-boundary-pnp" {
            if diagnose_cross_boundary_pnp {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            diagnose_cross_boundary_pnp = true;
            continue;
        }
        if flag == "--repair-cross-boundary" {
            if repair_cross_boundary {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            repair_cross_boundary = true;
            continue;
        }
        if flag == "--diagnostic-left-max-frame" {
            if diagnostic_left_max_frame.is_some() {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires FRAME\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!("{flag} requires non-negative FRAME, got {value:?}"));
            }
            let frame = value
                .parse::<u64>()
                .map_err(|error| format!("{flag} requires an integer FRAME: {error}\n{USAGE}"))?;
            diagnostic_left_max_frame = Some(frame);
            continue;
        }
        if flag == "--repair-left-max-frame" {
            if repair_left_max_frame.is_some() {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires FRAME\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!("{flag} requires non-negative FRAME, got {value:?}"));
            }
            let frame = value
                .parse::<u64>()
                .map_err(|error| format!("{flag} requires an integer FRAME: {error}\n{USAGE}"))?;
            repair_left_max_frame = Some(frame);
            continue;
        }
        let slot = match flag.as_str() {
            "--rig-manifest" => &mut rig_manifest,
            "--nodes-tsv" => &mut nodes_tsv,
            "--atlas-dir" => &mut atlas_dir,
            "--out-dir" => &mut out_dir,
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        };
        if slot.is_some() {
            return Err(format!("duplicate argument {flag}\n{USAGE}"));
        }
        let value = values
            .next()
            .ok_or_else(|| format!("{flag} requires PATH\n{USAGE}"))?;
        if value.starts_with('-') {
            return Err(format!("{flag} requires PATH, got {value:?}\n{USAGE}"));
        }
        *slot = Some(PathBuf::from(value));
    }
    if diagnose_cross_boundary_pnp && diagnostic_left_max_frame.is_none() {
        return Err(format!(
            "--diagnose-cross-boundary-pnp requires --diagnostic-left-max-frame FRAME\n{USAGE}"
        ));
    }
    if !diagnose_cross_boundary_pnp && diagnostic_left_max_frame.is_some() {
        return Err(format!(
            "--diagnostic-left-max-frame requires --diagnose-cross-boundary-pnp\n{USAGE}"
        ));
    }
    if repair_cross_boundary && repair_left_max_frame.is_none() {
        return Err(format!(
            "--repair-cross-boundary requires --repair-left-max-frame FRAME\n{USAGE}"
        ));
    }
    if !repair_cross_boundary && repair_left_max_frame.is_some() {
        return Err(format!(
            "--repair-left-max-frame requires --repair-cross-boundary\n{USAGE}"
        ));
    }
    if diagnose_cross_boundary_pnp && (recover_zero_support_frames || joint_rig_ba) {
        return Err(format!(
            "--diagnose-cross-boundary-pnp is diagnostic-only and cannot be combined with recovery or BA\n{USAGE}"
        ));
    }
    if diagnose_cross_boundary_pnp && repair_cross_boundary {
        return Err(format!(
            "--diagnose-cross-boundary-pnp and --repair-cross-boundary are mutually exclusive\n{USAGE}"
        ));
    }
    if pre_ba_out_dir.is_some() && !joint_rig_ba {
        return Err(format!("--pre-ba-out-dir requires --joint-rig-ba\n{USAGE}"));
    }
    if post_pass1_out_dir.is_some() && joint_rig_ba_filter_sweeps != 2 {
        return Err(format!(
            "--post-pass1-out-dir requires --joint-rig-ba-filter-sweeps 2\n{USAGE}"
        ));
    }
    if filter_sweeps_explicit && !joint_rig_ba_filter_observations {
        return Err(format!(
            "--joint-rig-ba-filter-sweeps requires --joint-rig-ba-filter-observations\n{USAGE}"
        ));
    }
    if joint_rig_ba_filter_observations && !joint_rig_ba {
        return Err(format!(
            "--joint-rig-ba-filter-observations requires --joint-rig-ba\n{USAGE}"
        ));
    }
    if joint_rig_ba_preserve_optimized_points && !(joint_rig_ba && joint_rig_ba_filter_observations)
    {
        return Err(format!(
            "--joint-rig-ba-preserve-optimized-points requires --joint-rig-ba and --joint-rig-ba-filter-observations\n{USAGE}"
        ));
    }
    if joint_rig_ba_filter_sweeps == 2 && post_pass1_out_dir.is_none() {
        return Err(format!(
            "--joint-rig-ba-filter-sweeps 2 requires --post-pass1-out-dir PATH\n{USAGE}"
        ));
    }
    if joint_rig_ba_filter_sweeps == 2 && !joint_rig_ba_filter_observations {
        return Err(format!(
            "--joint-rig-ba-filter-sweeps 2 requires --joint-rig-ba-filter-observations\n{USAGE}"
        ));
    }
    if joint_rig_ba_filter_sweeps == 2 && joint_rig_ba_preserve_optimized_points {
        return Err(format!(
            "--joint-rig-ba-filter-sweeps 2 cannot be combined with --joint-rig-ba-preserve-optimized-points\n{USAGE}"
        ));
    }
    Ok(Args {
        rig_manifest: rig_manifest.ok_or_else(|| format!("--rig-manifest is required\n{USAGE}"))?,
        nodes_tsv: nodes_tsv.ok_or_else(|| format!("--nodes-tsv is required\n{USAGE}"))?,
        atlas_dir: atlas_dir.ok_or_else(|| format!("--atlas-dir is required\n{USAGE}"))?,
        out_dir: out_dir.ok_or_else(|| format!("--out-dir is required\n{USAGE}"))?,
        pre_ba_out_dir,
        post_pass1_out_dir,
        recover_zero_support_frames,
        joint_rig_ba,
        joint_rig_ba_filter_observations,
        joint_rig_ba_preserve_optimized_points,
        joint_rig_ba_filter_sweeps,
        diagnose_cross_boundary_pnp,
        diagnostic_left_max_frame,
        repair_cross_boundary,
        repair_left_max_frame,
    })
}

fn resolved_comparison_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve current directory: {error}"))?
            .join(path)
    };
    let mut probe = absolute;
    let mut missing = Vec::new();
    let base = loop {
        match fs::canonicalize(&probe) {
            Ok(path) => break path,
            Err(error) => {
                let Some(name) = probe.file_name() else {
                    return Err(format!("cannot resolve path {:?}: {error}", path));
                };
                missing.push(name.to_os_string());
                if !probe.pop() {
                    return Err(format!("cannot resolve path {:?}: {error}", path));
                }
            }
        }
    };
    let mut resolved = base;
    for component in missing.iter().rev() {
        if component == "." {
            continue;
        }
        if component == ".." {
            resolved.pop();
        } else {
            resolved.push(component);
        }
    }
    Ok(resolved)
}

pub(super) fn paths_overlap(first: &Path, second: &Path) -> Result<bool, String> {
    let first = resolved_comparison_path(first)?;
    let second = resolved_comparison_path(second)?;
    Ok(first == second || first.starts_with(&second) || second.starts_with(&first))
}

pub(super) fn validate_write_destinations(args: &Args, nodes: &[NodeSpec]) -> Result<(), String> {
    let mut destinations = vec![("--out-dir", args.out_dir.as_path())];
    if let Some(path) = args.pre_ba_out_dir.as_deref() {
        destinations.push(("--pre-ba-out-dir", path));
    }
    if let Some(path) = args.post_pass1_out_dir.as_deref() {
        destinations.push(("--post-pass1-out-dir", path));
    }
    let mut inputs = vec![
        ("--rig-manifest", args.rig_manifest.as_path()),
        ("--nodes-tsv", args.nodes_tsv.as_path()),
        ("--atlas-dir", args.atlas_dir.as_path()),
    ];
    inputs.extend(
        nodes
            .iter()
            .map(|node| ("node images.txt", node.images_txt.as_path())),
    );
    for (index, (destination_name, destination)) in destinations.iter().enumerate() {
        for (other_name, other) in destinations.iter().skip(index + 1) {
            if paths_overlap(destination, other)? {
                return Err(format!(
                    "{destination_name} {:?} overlaps {other_name} {:?}; refusing ambiguous output destinations",
                    destination, other
                ));
            }
        }
        for (input_name, input) in &inputs {
            if paths_overlap(destination, input)? {
                return Err(format!(
                    "{destination_name} {:?} overlaps input {input_name} {:?}; refusing to write input data",
                    destination, input
                ));
            }
        }
    }
    if let Some(path) = args.pre_ba_out_dir.as_deref() {
        if path.exists() {
            if !path.is_dir() {
                return Err(format!(
                    "--pre-ba-out-dir {:?} exists but is not a directory",
                    path
                ));
            }
            let mut entries = fs::read_dir(path)
                .map_err(|error| format!("read --pre-ba-out-dir {}: {error}", path.display()))?;
            if entries
                .next()
                .transpose()
                .map_err(|error| format!("inspect --pre-ba-out-dir {}: {error}", path.display()))?
                .is_some()
            {
                return Err(format!(
                    "--pre-ba-out-dir {:?} must be new or empty; refusing to overwrite an existing checkpoint",
                    path
                ));
            }
        }
    }
    if let Some(path) = args.post_pass1_out_dir.as_deref() {
        if path.exists() {
            if !path.is_dir() {
                return Err(format!(
                    "--post-pass1-out-dir {:?} exists but is not a directory",
                    path
                ));
            }
            let mut entries = fs::read_dir(path).map_err(|error| {
                format!("read --post-pass1-out-dir {}: {error}", path.display())
            })?;
            if entries
                .next()
                .transpose()
                .map_err(|error| {
                    format!("inspect --post-pass1-out-dir {}: {error}", path.display())
                })?
                .is_some()
            {
                return Err(format!(
                    "--post-pass1-out-dir {:?} must be new or empty; refusing to overwrite an existing checkpoint",
                    path
                ));
            }
        }
    }
    Ok(())
}
