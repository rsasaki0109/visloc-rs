//! Bind selected learned-retrieval rig pairs to a frozen verified snapshot.
//!
//! Every selected rig pair must be reciprocal and supported by at least two
//! adjacent query frames. It expands to synchronized cross-sensor image pairs,
//! removes pairs already present in the structure snapshot, and atomically
//! publishes a matcher candidate manifest plus an attribution ledger.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use visloc_rs::verified_pair_snapshot::read as read_snapshot;

struct Args {
    retrieval: PathBuf,
    rig_manifest: PathBuf,
    base_snapshot: PathBuf,
    output_directory: PathBuf,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut retrieval = None;
    let mut rig_manifest = None;
    let mut base_snapshot = None;
    let mut output_directory = None;
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        let mut next = || {
            values
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--retrieval" => retrieval = Some(PathBuf::from(next()?)),
            "--rig-manifest" => rig_manifest = Some(PathBuf::from(next()?)),
            "--base-snapshot" => base_snapshot = Some(PathBuf::from(next()?)),
            "--output-directory" => output_directory = Some(PathBuf::from(next()?)),
            "-h" | "--help" => {
                println!(
                    "materialize_learned_rig_candidates --retrieval PAIRS.tsv \
                     --rig-manifest RIG.txt --base-snapshot BASE.vps \
                     --output-directory NEW_DIR"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}").into()),
        }
    }
    Ok(Args {
        retrieval: retrieval.ok_or("--retrieval is required")?,
        rig_manifest: rig_manifest.ok_or("--rig-manifest is required")?,
        base_snapshot: base_snapshot.ok_or("--base-snapshot is required")?,
        output_directory: output_directory.ok_or("--output-directory is required")?,
    })
}

#[derive(Debug, Clone)]
struct RetrievedPair {
    query: usize,
    candidate: usize,
    score: f32,
    sequence_support: usize,
}

struct Retrieval {
    metadata: BTreeMap<String, String>,
    selected: Vec<RetrievedPair>,
    policy: AdmissionPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdmissionPolicy {
    ReciprocalSequenceV1,
    RankMarginPathV2,
    RankPathCycleV3,
    ComponentBridgeV4,
    MultiScaleComponentBridgeV5,
}

impl AdmissionPolicy {
    fn name(self) -> &'static str {
        match self {
            Self::ReciprocalSequenceV1 => "reciprocal-sequence-v1",
            Self::RankMarginPathV2 => "rank-margin-path-v2",
            Self::RankPathCycleV3 => "rank-path-cycle-v3",
            Self::ComponentBridgeV4 => "component-bridge-v4",
            Self::MultiScaleComponentBridgeV5 => "multi-scale-component-bridge-v5",
        }
    }
}

fn parse_retrieval(path: &Path) -> Result<Retrieval, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read retrieval {}: {error}", path.display()))?;
    let mut metadata = BTreeMap::new();
    let mut selected = Vec::new();
    let mut policy = None;
    for (zero_line, raw) in text.lines().enumerate() {
        let line_number = zero_line + 1;
        let line = raw.trim();
        if line == "# visloc-learned-retrieval-v1" {
            policy = Some(AdmissionPolicy::ReciprocalSequenceV1);
            continue;
        }
        if line == "# visloc-learned-retrieval-v2" {
            policy = Some(AdmissionPolicy::RankMarginPathV2);
            continue;
        }
        if line == "# visloc-learned-retrieval-v3" {
            policy = Some(AdmissionPolicy::RankPathCycleV3);
            continue;
        }
        if line == "# visloc-learned-retrieval-v4" {
            policy = Some(AdmissionPolicy::ComponentBridgeV4);
            continue;
        }
        if line == "# visloc-learned-retrieval-v5" {
            policy = Some(AdmissionPolicy::MultiScaleComponentBridgeV5);
            continue;
        }
        if let Some(comment) = line.strip_prefix("# ") {
            if comment.starts_with("pair ") {
                continue;
            }
            if let Some((key, value)) = comment.split_once(' ') {
                if metadata.insert(key.to_owned(), value.to_owned()).is_some() {
                    return Err(format!("retrieval repeats metadata {key:?}"));
                }
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() < 4 || fields.first() != Some(&"pair") {
            return Err(format!("retrieval line {line_number} is malformed"));
        }
        let query: usize = fields[1]
            .parse()
            .map_err(|error| format!("retrieval line {line_number} query: {error}"))?;
        let candidate: usize = fields[2]
            .parse()
            .map_err(|error| format!("retrieval line {line_number} candidate: {error}"))?;
        let score: f32 = fields[3]
            .parse()
            .map_err(|error| format!("retrieval line {line_number} score: {error}"))?;
        if query >= candidate || !score.is_finite() {
            return Err(format!(
                "retrieval line {line_number} has invalid pair/score"
            ));
        }
        let (sequence_support, is_selected, expected_selected) = match policy {
            Some(AdmissionPolicy::ReciprocalSequenceV1) if fields.len() == 7 => {
                let mutual: bool = fields[4]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} mutual: {error}"))?;
                let support: usize = fields[5]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} support: {error}"))?;
                let selected: bool = fields[6]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} selected: {error}"))?;
                (support, selected, mutual && support >= 2)
            }
            Some(AdmissionPolicy::RankMarginPathV2) if fields.len() == 10 => {
                let query_rank: isize = fields[4]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} query rank: {error}"))?;
                let candidate_rank: isize = fields[5].parse().map_err(|error| {
                    format!("retrieval line {line_number} candidate rank: {error}")
                })?;
                let query_ratio: f32 = fields[6].parse().map_err(|error| {
                    format!("retrieval line {line_number} query ratio: {error}")
                })?;
                let candidate_ratio: f32 = fields[7].parse().map_err(|error| {
                    format!("retrieval line {line_number} candidate ratio: {error}")
                })?;
                let path: bool = fields[8]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} path: {error}"))?;
                let selected: bool = fields[9]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} selected: {error}"))?;
                let expected = (0..2).contains(&query_rank)
                    && (0..2).contains(&candidate_rank)
                    && query_ratio.is_finite()
                    && candidate_ratio.is_finite()
                    && query_ratio <= 0.8
                    && candidate_ratio <= 0.8
                    && path;
                (3, selected, expected)
            }
            Some(AdmissionPolicy::RankPathCycleV3) if fields.len() == 10 => {
                let query_rank: isize = fields[4]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} query rank: {error}"))?;
                let candidate_rank: isize = fields[5].parse().map_err(|error| {
                    format!("retrieval line {line_number} candidate rank: {error}")
                })?;
                let _: f32 = fields[6].parse().map_err(|error| {
                    format!("retrieval line {line_number} query ratio diagnostic: {error}")
                })?;
                let _: f32 = fields[7].parse().map_err(|error| {
                    format!("retrieval line {line_number} candidate ratio diagnostic: {error}")
                })?;
                let path: bool = fields[8]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} path: {error}"))?;
                let selected: bool = fields[9]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} selected: {error}"))?;
                let expected =
                    (0..2).contains(&query_rank) && (0..2).contains(&candidate_rank) && path;
                (3, selected, expected)
            }
            Some(AdmissionPolicy::ComponentBridgeV4) if fields.len() == 10 => {
                let query_rank: isize = fields[4]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} query rank: {error}"))?;
                let candidate_rank: isize = fields[5].parse().map_err(|error| {
                    format!("retrieval line {line_number} candidate rank: {error}")
                })?;
                let _: f32 = fields[6].parse().map_err(|error| {
                    format!("retrieval line {line_number} query ratio diagnostic: {error}")
                })?;
                let _: f32 = fields[7].parse().map_err(|error| {
                    format!("retrieval line {line_number} candidate ratio diagnostic: {error}")
                })?;
                let path: bool = fields[8]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} path: {error}"))?;
                let selected: bool = fields[9]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} selected: {error}"))?;
                let expected =
                    (0..8).contains(&query_rank) && (0..8).contains(&candidate_rank) && path;
                (3, selected, expected)
            }
            Some(AdmissionPolicy::MultiScaleComponentBridgeV5) if fields.len() == 10 => {
                let query_rank: isize = fields[4]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} query rank: {error}"))?;
                let candidate_rank: isize = fields[5].parse().map_err(|error| {
                    format!("retrieval line {line_number} candidate rank: {error}")
                })?;
                let _: f32 = fields[6].parse().map_err(|error| {
                    format!("retrieval line {line_number} query ratio diagnostic: {error}")
                })?;
                let _: f32 = fields[7].parse().map_err(|error| {
                    format!("retrieval line {line_number} candidate ratio diagnostic: {error}")
                })?;
                let path: bool = fields[8]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} path: {error}"))?;
                let selected: bool = fields[9]
                    .parse()
                    .map_err(|error| format!("retrieval line {line_number} selected: {error}"))?;
                let expected =
                    (0..8).contains(&query_rank) && (0..8).contains(&candidate_rank) && path;
                (3, selected, expected)
            }
            _ => {
                return Err(format!(
                    "retrieval line {line_number} is malformed for its schema"
                ))
            }
        };
        if is_selected != expected_selected {
            return Err(format!(
                "retrieval line {line_number} selected flag violates its declared admission policy"
            ));
        }
        if is_selected {
            selected.push(RetrievedPair {
                query,
                candidate,
                score,
                sequence_support,
            });
        }
    }
    let policy = policy.ok_or_else(|| "retrieval header is missing".to_owned())?;
    let declared: usize = required(&metadata, "selected_pairs")?
        .parse()
        .map_err(|error| format!("selected_pairs: {error}"))?;
    if declared != selected.len() {
        return Err(format!(
            "retrieval selected count {} differs from declared {declared}",
            selected.len()
        ));
    }
    if required(&metadata, "topk")? != "32" {
        return Err("retrieval topk must be frozen at 32".to_owned());
    }
    if matches!(
        policy,
        AdmissionPolicy::RankMarginPathV2 | AdmissionPolicy::RankPathCycleV3
    ) {
        let mut keys = vec![
            "admission_rank",
            "competitor_exclusion_radius",
            "sequence_path_radius",
            "per_frame_addition_budget",
        ];
        if policy == AdmissionPolicy::RankMarginPathV2 {
            keys.push("distance_ratio_max");
        }
        for key in keys {
            let expected = match key {
                "admission_rank" | "competitor_exclusion_radius" | "per_frame_addition_budget" => {
                    "2"
                }
                "distance_ratio_max" => "0.8",
                "sequence_path_radius" => "1",
                _ => unreachable!(),
            };
            if required(&metadata, key)? != expected {
                return Err(format!(
                    "retrieval metadata {key:?} is not frozen at {expected}"
                ));
            }
        }
        if policy == AdmissionPolicy::RankPathCycleV3
            && (required(&metadata, "post_verification_gate")? != "rig-rotation-cycle-v1"
                || required(&metadata, "distance_ratio_diagnostic_only")? != "true")
        {
            return Err("v3 retrieval is not bound to its frozen post-verification gate".into());
        }
        let mut degree = BTreeMap::<usize, usize>::new();
        for row in &selected {
            *degree.entry(row.query).or_default() += 1;
            *degree.entry(row.candidate).or_default() += 1;
        }
        if degree.values().any(|value| *value > 2) {
            return Err("strict retrieval exceeds the per-frame addition budget".to_owned());
        }
    }
    if matches!(
        policy,
        AdmissionPolicy::ComponentBridgeV4 | AdmissionPolicy::MultiScaleComponentBridgeV5
    ) {
        for (key, expected) in [
            ("admission_rank", "8"),
            ("sequence_path_radius", "1"),
            ("per_frame_addition_budget", "2"),
            ("post_verification_gate", "rig-rotation-cycle-v1"),
            ("cross_component_ann_only", "true"),
            ("requires_unregistered_endpoint", "true"),
        ] {
            if required(&metadata, key)? != expected {
                return Err(format!(
                    "retrieval metadata {key:?} is not frozen at {expected}"
                ));
            }
        }
        for key in [
            "rig_manifest_sha256",
            "retrieval_component_manifest_sha256",
            "registered_rows",
            "unregistered_rows",
            "registered_components",
            "inferred_unregistered_runs",
            "component_rank_path_survival",
        ] {
            let _ = required(&metadata, key)?;
        }
        if policy == AdmissionPolicy::MultiScaleComponentBridgeV5 {
            for (key, expected) in [
                ("sequence_offsets", "-32,-16,-8,0,8,16,32"),
                ("sequence_direction", "forward-or-reverse"),
                ("sequence_pre_rerank_k", "128"),
            ] {
                if required(&metadata, key)? != expected {
                    return Err(format!(
                        "retrieval metadata {key:?} is not frozen at {expected}"
                    ));
                }
            }
        }
        let mut degree = BTreeMap::<usize, usize>::new();
        for row in &selected {
            *degree.entry(row.query).or_default() += 1;
            *degree.entry(row.candidate).or_default() += 1;
        }
        if degree.values().any(|value| *value > 2) {
            return Err("component retrieval exceeds the per-frame addition budget".to_owned());
        }
    }
    Ok(Retrieval {
        metadata,
        selected,
        policy,
    })
}

fn required<'a>(metadata: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, String> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("retrieval metadata {key:?} is missing"))
}

fn parse_rig(path: &Path, image_names: &[String]) -> Result<(Vec<Vec<usize>>, String), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read rig manifest {}: {error}", path.display()))?;
    let image_indices: BTreeMap<_, _> = image_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();
    let mut rows = BTreeMap::<usize, BTreeMap<usize, (String, usize)>>::new();
    for (zero_line, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || !line.starts_with("F ") {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 4 {
            return Err(format!("rig line {} has malformed F row", zero_line + 1));
        }
        let frame: usize = fields[1]
            .parse()
            .map_err(|error| format!("rig line {} frame: {error}", zero_line + 1))?;
        let sensor: usize = fields[3]
            .parse()
            .map_err(|error| format!("rig line {} sensor: {error}", zero_line + 1))?;
        let image = *image_indices
            .get(fields[2])
            .ok_or_else(|| format!("rig names unknown snapshot image {:?}", fields[2]))?;
        if rows
            .entry(frame)
            .or_default()
            .insert(sensor, (fields[2].to_owned(), image))
            .is_some()
        {
            return Err(format!("rig frame {frame} assigns sensor {sensor} twice"));
        }
    }
    if rows.is_empty() {
        return Err("rig manifest contains no frame rows".to_owned());
    }
    let expected_sensors: Vec<_> = rows
        .first_key_value()
        .expect("non-empty")
        .1
        .keys()
        .copied()
        .collect();
    let mut frames = Vec::with_capacity(rows.len());
    let mut digest = Sha256::new();
    let mut assigned = BTreeSet::new();
    for (expected_frame, (frame, sensors)) in rows.into_iter().enumerate() {
        if frame != expected_frame {
            return Err(format!("rig frames are not contiguous at {expected_frame}"));
        }
        if sensors.keys().copied().ne(expected_sensors.iter().copied()) {
            return Err(format!("rig frame {frame} has an inconsistent sensor set"));
        }
        let canonical = format!(
            "frame {frame} {}",
            sensors
                .iter()
                .map(|(sensor, (name, _))| format!("{sensor}:{name}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        digest.update(canonical.as_bytes());
        digest.update(b"\n");
        let mut images = Vec::with_capacity(sensors.len());
        for (_, image) in sensors.into_values() {
            if !assigned.insert(image) {
                return Err(format!("snapshot image index {image} is assigned twice"));
            }
            images.push(image);
        }
        frames.push(images);
    }
    if assigned.len() != image_names.len() {
        return Err(format!(
            "rig assigns {} images but snapshot contains {}",
            assigned.len(),
            image_names.len()
        ));
    }
    Ok((frames, format!("{:x}", digest.finalize())))
}

#[derive(Debug)]
struct Addition {
    rig_query: usize,
    rig_candidate: usize,
    score: f32,
    sequence_support: usize,
}

fn expand_additions(
    retrieval: &[RetrievedPair],
    frames: &[Vec<usize>],
    existing: &BTreeSet<(usize, usize)>,
) -> Result<BTreeMap<(usize, usize), Addition>, String> {
    let mut additions = BTreeMap::new();
    for row in retrieval {
        let query = frames
            .get(row.query)
            .ok_or_else(|| format!("retrieval query frame {} is out of range", row.query))?;
        let candidate = frames.get(row.candidate).ok_or_else(|| {
            format!(
                "retrieval candidate frame {} is out of range",
                row.candidate
            )
        })?;
        for &left in query {
            for &right in candidate {
                let pair = (left.min(right), left.max(right));
                if existing.contains(&pair) {
                    continue;
                }
                let addition = Addition {
                    rig_query: row.query,
                    rig_candidate: row.candidate,
                    score: row.score,
                    sequence_support: row.sequence_support,
                };
                if additions.insert(pair, addition).is_some() {
                    return Err(format!("expanded image pair {pair:?} has multiple owners"));
                }
            }
        }
    }
    Ok(additions)
}

fn hash_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_synced(path: &Path, contents: &str) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(contents.as_bytes())?;
    file.sync_all()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.output_directory.exists() {
        return Err(format!(
            "refusing to replace output directory {}",
            args.output_directory.display()
        )
        .into());
    }
    let snapshot = read_snapshot(&args.base_snapshot).map_err(std::io::Error::other)?;
    let retrieval = parse_retrieval(&args.retrieval).map_err(std::io::Error::other)?;
    let (frames, rig_order_sha256) =
        parse_rig(&args.rig_manifest, &snapshot.image_names).map_err(std::io::Error::other)?;
    if required(&retrieval.metadata, "manifest_sha256")? != rig_order_sha256 {
        return Err("retrieval rig-order SHA-256 differs from the frozen rig manifest".into());
    }
    let declared_rows: usize = required(&retrieval.metadata, "rows")?.parse()?;
    if declared_rows != frames.len() {
        return Err(format!(
            "retrieval rows {declared_rows} != rig frames {}",
            frames.len()
        )
        .into());
    }
    let existing: BTreeSet<_> = snapshot
        .pairs
        .iter()
        .map(|pair| {
            let left = pair.image_i as usize;
            let right = pair.image_j as usize;
            (left.min(right), left.max(right))
        })
        .collect();
    let additions =
        expand_additions(&retrieval.selected, &frames, &existing).map_err(std::io::Error::other)?;
    let retrieval_sha256 = hash_file(&args.retrieval)?;
    let snapshot_sha256 = hash_file(&args.base_snapshot)?;
    let rig_sha256 = hash_file(&args.rig_manifest)?;

    let name = args
        .output_directory
        .file_name()
        .ok_or("output directory has no filename")?;
    let mut temporary_name = name.to_os_string();
    temporary_name.push(format!(".tmp-{}", std::process::id()));
    let temporary = args.output_directory.with_file_name(temporary_name);
    std::fs::create_dir(&temporary)?;

    let mut candidates = format!(
        "visloc_candidate_manifest_v1\nimages {}\n",
        snapshot.image_names.len()
    );
    for (index, name) in snapshot.image_names.iter().enumerate() {
        writeln!(candidates, "image {index} {name}")?;
    }
    writeln!(
        candidates,
        "metadata base_snapshot_sha256 {snapshot_sha256}"
    )?;
    writeln!(
        candidates,
        "metadata pair_source learned-rig-{}",
        retrieval.policy.name()
    )?;
    writeln!(candidates, "metadata retrieval_sha256 {retrieval_sha256}")?;
    writeln!(candidates, "metadata rig_manifest_sha256 {rig_sha256}")?;
    writeln!(candidates, "pairs {}", additions.len())?;
    for &(left, right) in additions.keys() {
        writeln!(candidates, "pair {left} {right}")?;
    }

    let mut ledger = format!(
        "# visloc-learned-rig-additions-v1\n# admission_policy {}\n# retrieval_sha256 {retrieval_sha256}\n# base_snapshot_sha256 {snapshot_sha256}\n# rig_manifest_sha256 {rig_sha256}\n# rig_order_sha256 {rig_order_sha256}\n# selected_rig_pairs {}\n# existing_image_pairs {}\n# added_image_pairs {}\n# image_pair image_i image_j rig_query rig_candidate cosine sequence_support\n",
        retrieval.policy.name(),
        retrieval.selected.len(),
        existing.len(),
        additions.len()
    );
    for (&(left, right), row) in &additions {
        writeln!(
            ledger,
            "image_pair {left} {right} {} {} {:.9} {}",
            row.rig_query, row.rig_candidate, row.score, row.sequence_support
        )?;
    }
    write_synced(&temporary.join("candidates.txt"), &candidates)?;
    write_synced(&temporary.join("addition-ledger.tsv"), &ledger)?;
    File::open(&temporary)?.sync_all()?;
    std::fs::rename(&temporary, &args.output_directory)?;
    if let Some(parent) = args.output_directory.parent() {
        File::open(parent)?.sync_all()?;
    }
    println!(
        "materialized selected_rig_pairs={} existing_image_pairs={} added_image_pairs={} -> {}",
        retrieval.selected.len(),
        existing.len(),
        additions.len(),
        args.output_directory.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_removes_existing_pairs_and_preserves_owner() {
        let retrieval = vec![RetrievedPair {
            query: 0,
            candidate: 2,
            score: 0.9,
            sequence_support: 3,
        }];
        let frames = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
        let existing = BTreeSet::from([(0, 4)]);
        let additions = expand_additions(&retrieval, &frames, &existing).unwrap();
        assert_eq!(
            additions.keys().copied().collect::<Vec<_>>(),
            vec![(0, 5), (1, 4), (1, 5)]
        );
        assert!(additions
            .values()
            .all(|row| row.rig_query == 0 && row.rig_candidate == 2));
    }

    #[test]
    fn parser_rejects_a_selected_nonreciprocal_pair() {
        let path =
            std::env::temp_dir().join(format!("visloc-bad-retrieval-{}", std::process::id()));
        std::fs::write(
            &path,
            "# visloc-learned-retrieval-v1\n# topk 32\n# selected_pairs 1\npair 0 2 0.9 false 3 true\n",
        )
        .unwrap();
        assert!(parse_retrieval(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn parser_accepts_a_bounded_component_bridge_policy() {
        let path =
            std::env::temp_dir().join(format!("visloc-component-retrieval-{}", std::process::id()));
        std::fs::write(
            &path,
            "# visloc-learned-retrieval-v4\n\
             # topk 32\n# selected_pairs 1\n# admission_rank 8\n\
             # sequence_path_radius 1\n# per_frame_addition_budget 2\n\
             # post_verification_gate rig-rotation-cycle-v1\n\
             # cross_component_ann_only true\n# requires_unregistered_endpoint true\n\
             # rig_manifest_sha256 rig\n# retrieval_component_manifest_sha256 components\n\
             # registered_rows 3\n# unregistered_rows 3\n# registered_components 1\n\
             # inferred_unregistered_runs 1\n# component_rank_path_survival 2:0,8:1\n\
             pair 1 4 0.9 0 1 1.0 1.0 true true\n",
        )
        .unwrap();
        let retrieval = parse_retrieval(&path).unwrap();
        assert_eq!(retrieval.policy, AdmissionPolicy::ComponentBridgeV4);
        assert_eq!(retrieval.selected.len(), 1);
        std::fs::remove_file(path).unwrap();
    }
}
