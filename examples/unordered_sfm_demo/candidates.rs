//! Candidate-pair generation (rig / temporal / VLAD / LSH / vocabulary tree) and candidate manifests.

use super::*;

/// Parse the trailing decimal run from an image stem.  A pair-window run is
/// intentionally strict: silently falling back to lexical order would make a
/// supposedly sequence-local reconstruction depend on an unrelated filename
/// convention.
fn trailing_numeric_stem(name: &str) -> Result<u64, String> {
    let stem = image_stem(name);
    let bytes = stem.as_bytes();
    let mut start = bytes.len();
    while start > 0 && bytes[start - 1].is_ascii_digit() {
        start -= 1;
    }
    if start == bytes.len() {
        return Err(format!(
            "image {name:?} has no trailing numeric stem (expected e.g. DSC_0001)"
        ));
    }
    stem[start..].parse::<u64>().map_err(|error| {
        format!(
            "image {name:?} has an invalid trailing numeric stem {:?}: {error}",
            &stem[start..]
        )
    })
}

/// Validate and return the numeric stem for every loaded image.  Duplicate
/// numeric suffixes are rejected because they would make the window relation
/// ambiguous even when the lexical filenames differ.
pub(super) fn numeric_stem_values(image_names: &[String]) -> Result<Vec<u64>, String> {
    let mut seen: HashMap<u64, (usize, String)> = HashMap::new();
    let mut values = Vec::with_capacity(image_names.len());
    for (index, name) in image_names.iter().enumerate() {
        let value = trailing_numeric_stem(name)?;
        if let Some((other_index, other_name)) = seen.insert(value, (index, name.clone())) {
            return Err(format!(
                "duplicate numeric image stem {value} in images {other_index} ({other_name:?}) and {index} ({name:?})"
            ));
        }
        values.push(value);
    }
    Ok(values)
}

/// Return the camera-prefix and trailing timestamp from a rig image name.
///
/// Rig-aware local grouping is deliberately opt-in because a flat image set
/// does not otherwise have a reliable camera-name convention.  The accepted
/// form is `<prefix>_<decimal-timestamp>` (with the normal image extension
/// still present in `name`); the prefix is retained verbatim and is compared
/// case-sensitively.  A duplicate `(prefix, timestamp)` is rejected, while
/// the same timestamp across distinct prefixes is the expected stereo/rig
/// case.
fn rig_camera_timestamp(name: &str) -> Result<(String, u64), String> {
    let stem = image_stem(name);
    let Some(separator) = stem.rfind('_') else {
        return Err(format!(
            "rig-local grouping requires image {name:?} to use <camera-prefix>_<numeric-timestamp>"
        ));
    };
    let prefix = &stem[..separator];
    let timestamp = &stem[separator + 1..];
    if prefix.is_empty() || timestamp.is_empty() {
        return Err(format!(
            "rig-local grouping requires image {name:?} to use <camera-prefix>_<numeric-timestamp>"
        ));
    }
    let timestamp = timestamp.parse::<u64>().map_err(|error| {
        format!(
            "rig-local grouping image {name:?} has invalid numeric timestamp {timestamp:?}: {error}"
        )
    })?;
    Ok((prefix.to_owned(), timestamp))
}

/// Build deterministic camera-aware local edges for a multi-camera rig.
///
/// Within each camera prefix, timestamps within `window` are connected.  At
/// each timestamp, every pair of distinct camera prefixes is connected once;
/// this is bounded by the number of camera pairs at that instant and does not
/// create the quadratic cross-product between neighbouring timestamps.  The
/// returned pairs use canonical image indices and are sorted by pair key.
pub(super) fn rig_local_pairs(
    image_names: &[String],
    window: u64,
) -> Result<Vec<(usize, usize)>, String> {
    if window == 0 {
        return Err("--local-stem-window must be at least 1".into());
    }
    let mut by_camera = BTreeMap::<String, Vec<(u64, usize)>>::new();
    let mut by_timestamp = BTreeMap::<u64, Vec<(String, usize)>>::new();
    let mut seen = HashSet::<(String, u64)>::new();
    for (index, name) in image_names.iter().enumerate() {
        let (camera, timestamp) = rig_camera_timestamp(name)?;
        if !seen.insert((camera.clone(), timestamp)) {
            return Err(format!(
                "rig-local grouping repeats timestamp {timestamp} for camera prefix {camera:?}"
            ));
        }
        by_camera
            .entry(camera.clone())
            .or_default()
            .push((timestamp, index));
        by_timestamp
            .entry(timestamp)
            .or_default()
            .push((camera, index));
    }

    let mut pairs = HashSet::<(usize, usize)>::new();
    for entries in by_camera.values_mut() {
        entries.sort_unstable_by_key(|&(timestamp, index)| (timestamp, index));
        let mut first = 0usize;
        for right in 0..entries.len() {
            let right_timestamp = entries[right].0;
            while first < right && right_timestamp.abs_diff(entries[first].0) > window {
                first += 1;
            }
            for left in first..right {
                let pair = (
                    entries[left].1.min(entries[right].1),
                    entries[left].1.max(entries[right].1),
                );
                pairs.insert(pair);
            }
        }
    }
    for entries in by_timestamp.values_mut() {
        entries.sort_unstable();
        for left in 0..entries.len() {
            for right in left + 1..entries.len() {
                // Duplicate `(camera,timestamp)` values were rejected above,
                // so every pair here is a distinct-camera rig edge.
                let pair = (
                    entries[left].1.min(entries[right].1),
                    entries[left].1.max(entries[right].1),
                );
                pairs.insert(pair);
            }
        }
    }
    let mut pairs: Vec<_> = pairs.into_iter().collect();
    pairs.sort_unstable();
    Ok(pairs)
}

/// Build the two deterministic rig-aware components of the temporal-pyramid
/// schedule.  Temporal offsets are positions in a camera's timestamp-sorted
/// sequence, not differences between the nanosecond timestamp values.  This
/// distinction matters for ETH3D, where captures are not numbered by a dense
/// integer frame counter.  Same-timestamp cross-camera pairs are returned
/// separately so the caller can give them a stable priority after temporal
/// edges and before VLAD fill edges.
fn generalized_rig_frame_groups(
    path: &Path,
    image_names: &[String],
) -> Result<Vec<(String, u64)>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read --rig-frame-manifest {path:?}: {error}"))?;
    let image_indices = image_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut groups = vec![None; image_names.len()];
    let mut magic = false;
    for (line_number, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line == "# generalized-rig-manifest-v1" {
            magic = true;
            continue;
        }
        if line.is_empty() || line.starts_with('#') || line.starts_with("S ") {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.first() != Some(&"F") || fields.len() != 4 {
            return Err(format!(
                "invalid --rig-frame-manifest row {} (expected F frame_id image_name sensor_index)",
                line_number + 1
            ));
        }
        let frame = fields[1].parse::<u64>().map_err(|error| {
            format!(
                "invalid frame id {:?} at --rig-frame-manifest row {}: {error}",
                fields[1],
                line_number + 1
            )
        })?;
        let sensor = fields[3].parse::<u64>().map_err(|error| {
            format!(
                "invalid sensor index {:?} at --rig-frame-manifest row {}: {error}",
                fields[3],
                line_number + 1
            )
        })?;
        let Some(&image) = image_indices.get(fields[2]) else {
            return Err(format!(
                "--rig-frame-manifest row {} names unknown image {:?}",
                line_number + 1,
                fields[2]
            ));
        };
        if groups[image]
            .replace((format!("sensor-{sensor}"), frame))
            .is_some()
        {
            return Err(format!(
                "--rig-frame-manifest repeats image {:?}",
                fields[2]
            ));
        }
    }
    if !magic {
        return Err("--rig-frame-manifest is not generalized-rig-manifest-v1".into());
    }
    if let Some((index, _)) = groups.iter().enumerate().find(|(_, group)| group.is_none()) {
        return Err(format!(
            "--rig-frame-manifest is missing loaded image {:?}",
            image_names[index]
        ));
    }
    Ok(groups.into_iter().map(Option::unwrap).collect())
}

fn rig_temporal_pyramid_pairs_from_groups(
    groups: &[(String, u64)],
    max_offset: u64,
) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>), String> {
    if max_offset == 0 {
        return Err("--temporal-pyramid-max-offset must be at least 1".into());
    }
    let mut by_camera = BTreeMap::<String, Vec<(u64, usize)>>::new();
    let mut by_timestamp = BTreeMap::<u64, Vec<(String, usize)>>::new();
    let mut seen = HashSet::<(String, u64)>::new();
    for (index, (camera, timestamp)) in groups.iter().enumerate() {
        let (camera, timestamp) = (camera.clone(), *timestamp);
        if !seen.insert((camera.clone(), timestamp)) {
            return Err(format!(
                "temporal-pyramid grouping repeats timestamp {timestamp} for camera prefix {camera:?}"
            ));
        }
        by_camera
            .entry(camera.clone())
            .or_default()
            .push((timestamp, index));
        by_timestamp
            .entry(timestamp)
            .or_default()
            .push((camera, index));
    }

    // Generate adjacent edges first, then progressively longer pyramid
    // levels.  That gives a bounded budget the most local support while
    // retaining every level when no budget is requested.
    let mut offsets = Vec::new();
    let mut offset = 1u64;
    loop {
        offsets.push(offset as usize);
        if offset > max_offset / 2 {
            break;
        }
        offset *= 2;
    }
    let mut temporal = Vec::new();
    let mut long_temporal = Vec::new();
    let mut temporal_seen = HashSet::<(usize, usize)>::new();
    for &offset in &offsets {
        let stride = if offset <= 32 { 1 } else { offset / 16 };
        for entries in by_camera.values() {
            for left in (0..entries.len().saturating_sub(offset)).step_by(stride) {
                let right = left + offset;
                let pair = (
                    entries[left].1.min(entries[right].1),
                    entries[left].1.max(entries[right].1),
                );
                if temporal_seen.insert(pair) {
                    if offset <= 32 {
                        temporal.push(pair);
                    } else {
                        long_temporal.push(pair);
                    }
                }
            }
        }
    }

    let mut cross_camera = Vec::new();
    let mut cross_seen = HashSet::<(usize, usize)>::new();
    for entries in by_timestamp.values_mut() {
        entries.sort_unstable();
        for left in 0..entries.len() {
            for right in left + 1..entries.len() {
                // A duplicate `(camera,timestamp)` was rejected above, so
                // every edge here is between distinct camera prefixes.
                let pair = (
                    entries[left].1.min(entries[right].1),
                    entries[left].1.max(entries[right].1),
                );
                if cross_seen.insert(pair) {
                    cross_camera.push(pair);
                }
            }
        }
    }
    // Metric same-frame edges must not be displaced by sparse long-baseline
    // levels under a bounded budget. The caller consumes this priority tail
    // after dense temporal edges, so append long levels after rig edges.
    cross_camera.extend(long_temporal);
    Ok((temporal, cross_camera))
}

pub(super) fn rig_temporal_pyramid_pairs(
    image_names: &[String],
    max_offset: u64,
) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>), String> {
    let groups = image_names
        .iter()
        .map(|name| rig_camera_timestamp(name))
        .collect::<Result<Vec<_>, _>>()?;
    rig_temporal_pyramid_pairs_from_groups(&groups, max_offset)
}

pub(super) fn rig_temporal_pyramid_pairs_with_manifest(
    image_names: &[String],
    max_offset: u64,
    rig_frame_manifest: Option<&Path>,
) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>), String> {
    if let Some(path) = rig_frame_manifest {
        let groups = generalized_rig_frame_groups(path, image_names)?;
        rig_temporal_pyramid_pairs_from_groups(&groups, max_offset)
    } else {
        rig_temporal_pyramid_pairs(image_names, max_offset)
    }
}

fn rig_frame_ids(
    image_names: &[String],
    rig_frame_manifest: Option<&Path>,
) -> Result<Vec<u64>, String> {
    if let Some(path) = rig_frame_manifest {
        return Ok(generalized_rig_frame_groups(path, image_names)?
            .into_iter()
            .map(|(_, frame)| frame)
            .collect());
    }
    image_names
        .iter()
        .map(|name| rig_camera_timestamp(name).map(|(_, timestamp)| timestamp))
        .collect()
}

pub(super) fn temporal_pyramid_offsets_string(max_offset: u64) -> String {
    let mut values = Vec::new();
    let mut offset = 1u64;
    loop {
        values.push(offset.to_string());
        if offset > max_offset / 2 {
            break;
        }
        offset *= 2;
    }
    values.join(",")
}

fn retrieval_component_labels(
    path: &Path,
    image_names: &[String],
) -> Result<Vec<Option<u64>>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read --retrieval-component-manifest {path:?}: {error}"))?;
    let image_indices = image_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut labels = vec![None; image_names.len()];
    let mut magic = false;
    let mut components = HashSet::new();
    for (line_number, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line == "# retrieval-component-manifest-v1" {
            magic = true;
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.first() != Some(&"C") || fields.len() != 3 {
            return Err(format!(
                "invalid --retrieval-component-manifest row {} (expected C component_id image_name)",
                line_number + 1
            ));
        }
        let component = fields[1].parse::<u64>().map_err(|error| {
            format!(
                "invalid component id {:?} at --retrieval-component-manifest row {}: {error}",
                fields[1],
                line_number + 1
            )
        })?;
        let Some(&image) = image_indices.get(fields[2]) else {
            return Err(format!(
                "--retrieval-component-manifest row {} names unknown image {:?}",
                line_number + 1,
                fields[2]
            ));
        };
        if labels[image].replace(component).is_some() {
            return Err(format!(
                "--retrieval-component-manifest repeats image {:?}",
                fields[2]
            ));
        }
        components.insert(component);
    }
    if !magic {
        return Err("--retrieval-component-manifest is not retrieval-component-manifest-v1".into());
    }
    if components.len() < 2 {
        return Err(
            "--retrieval-component-manifest must contain at least two component ids".into(),
        );
    }
    Ok(labels)
}

fn append_component_balanced_retrieval(
    selected: &mut Vec<(usize, usize)>,
    seen: &mut HashSet<(usize, usize)>,
    retrieval: &[((usize, usize), f32)],
    labels: &[Option<u64>],
    budget: usize,
) {
    let mut by_component_pair = BTreeMap::<(u64, u64), Vec<(usize, usize)>>::new();
    for &(pair, _) in retrieval {
        let (Some(left), Some(right)) = (labels[pair.0], labels[pair.1]) else {
            continue;
        };
        if left == right {
            continue;
        }
        let key = (left.min(right), left.max(right));
        by_component_pair.entry(key).or_default().push(pair);
    }
    let mut cursors = BTreeMap::<(u64, u64), usize>::new();
    loop {
        let mut progress = false;
        for (&component_pair, pairs) in &by_component_pair {
            let cursor = cursors.entry(component_pair).or_default();
            while *cursor < pairs.len() && seen.contains(&pairs[*cursor]) {
                *cursor += 1;
            }
            if *cursor == pairs.len() {
                continue;
            }
            let pair = pairs[*cursor];
            *cursor += 1;
            seen.insert(pair);
            selected.push(pair);
            progress = true;
            if selected.len() == budget {
                return;
            }
        }
        if !progress {
            return;
        }
    }
}

/// Candidate pairs from a rig-aware temporal pyramid plus a deterministic
/// VLAD fill.  Temporal edges are selected first, same-timestamp rig edges
/// second, and retrieval-only edges by descending VLAD score last.  The
/// final stream is sorted by canonical pair key after this priority selection
/// so the archived manifest is stable and independent of map iteration.
fn candidate_pairs_temporal_pyramid(
    features: &[FeatureSet],
    image_names: &[String],
    vocab_size: usize,
    topk: usize,
    max_offset: u64,
    budget: Option<usize>,
    rig_frame_manifest: Option<&Path>,
    retrieval_component_manifest: Option<&Path>,
    retrieval_min_frame_gap: Option<u64>,
) -> Result<Vec<(usize, usize)>, String> {
    let retrieval = candidate_pairs_vlad_scored(features, vocab_size, topk, false, false);
    candidate_pairs_temporal_pyramid_from_retrieval(
        image_names,
        retrieval,
        max_offset,
        budget,
        rig_frame_manifest,
        retrieval_component_manifest,
        retrieval_min_frame_gap,
    )
}

pub(super) fn candidate_pairs_temporal_pyramid_from_retrieval(
    image_names: &[String],
    mut retrieval: Vec<((usize, usize), f32)>,
    max_offset: u64,
    budget: Option<usize>,
    rig_frame_manifest: Option<&Path>,
    retrieval_component_manifest: Option<&Path>,
    retrieval_min_frame_gap: Option<u64>,
) -> Result<Vec<(usize, usize)>, String> {
    let (temporal, cross_camera) =
        rig_temporal_pyramid_pairs_with_manifest(image_names, max_offset, rig_frame_manifest)?;
    let mut selected = Vec::new();
    let mut seen = HashSet::<(usize, usize)>::new();
    for pair in temporal.into_iter().chain(cross_camera) {
        if seen.insert(pair) {
            selected.push(pair);
        }
    }
    if let Some(min_gap) = retrieval_min_frame_gap {
        let frames = rig_frame_ids(image_names, rig_frame_manifest)?;
        retrieval.retain(|&((left, right), _)| frames[left].abs_diff(frames[right]) >= min_gap);
    }
    // Retrieval producers retain pairs in canonical-key order so their output
    // is deterministic.  A bounded temporal-pyramid fill, however, promises
    // the strongest appearance edges first.  Re-rank here at the policy
    // boundary instead of relying on an incidental producer traversal order.
    retrieval.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
    if let Some(budget) = budget {
        selected.truncate(budget);
        if selected.len() < budget {
            if let Some(path) = retrieval_component_manifest {
                let labels = retrieval_component_labels(path, image_names)?;
                append_component_balanced_retrieval(
                    &mut selected,
                    &mut seen,
                    &retrieval,
                    &labels,
                    budget,
                );
            }
            for (pair, _) in retrieval {
                if selected.len() == budget {
                    break;
                }
                if seen.insert(pair) {
                    selected.push(pair);
                }
            }
        }
    } else {
        for (pair, _) in retrieval {
            if seen.insert(pair) {
                selected.push(pair);
            }
        }
    }
    selected.sort_unstable();
    Ok(selected)
}

pub(super) fn pair_within_stem_window(
    pair: (usize, usize),
    stem_values: &[u64],
    window: u64,
) -> Result<bool, String> {
    let (i, j) = pair;
    let (&left, &right) = (
        stem_values.get(i).ok_or_else(|| {
            format!(
                "pair index {i} is outside the loaded image range 0..{}",
                stem_values.len()
            )
        })?,
        stem_values.get(j).ok_or_else(|| {
            format!(
                "pair index {j} is outside the loaded image range 0..{}",
                stem_values.len()
            )
        })?,
    );
    let difference = left.abs_diff(right);
    Ok(difference <= window)
}

/// Apply the validated numeric-stem window while preserving the input order.
/// Candidate generators already provide deterministic order (and imported
/// records have a deterministic file order), so filtering must not introduce
/// a second traversal policy.
pub(super) fn filter_pairs_by_stem_window(
    pairs: Vec<(usize, usize)>,
    image_names: &[String],
    window: Option<u64>,
) -> Result<Vec<(usize, usize)>, String> {
    let Some(window) = window else {
        return Ok(pairs);
    };
    if window == 0 {
        return Err("--pair-stem-window must be at least 1".into());
    }
    let stem_values = numeric_stem_values(image_names)?;
    pairs
        .into_iter()
        .filter_map(
            |pair| match pair_within_stem_window(pair, &stem_values, window) {
                Ok(true) => Some(Ok(pair)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            },
        )
        .collect()
}

pub(super) const CANDIDATE_MANIFEST_MAGIC: &str = "visloc_candidate_manifest_v1";
pub(super) const CANDIDATE_SHARD_MAGIC_V2: &str = "visloc_candidate_shard_v2";

/// Parse a small, image-name-bound candidate-pair manifest.  The format is
/// intentionally line-oriented so it can be inspected, hashed, and generated
/// without a JSON dependency in the Rust example:
///
/// visloc_candidate_manifest_v1
/// images 2
/// image 0 first.JPG
/// image 1 second.JPG
/// pairs 1
/// pair 0 1
///
/// Pair order is preserved, while duplicate/reversed pairs are rejected.  A
/// manifest is a candidate schedule only; it contains no raw matches or
/// verification outcomes.
pub(super) fn parse_candidate_manifest(
    path: &Path,
    image_names: &[String],
) -> Result<Vec<(usize, usize)>, String> {
    parse_candidate_manifest_with_metadata(path, image_names).map(|(pairs, _)| pairs)
}

/// Parse a candidate manifest and retain its optional deterministic policy
/// metadata.  Metadata is deliberately a tiny `metadata KEY VALUE` block so
/// older readers can still reject unsupported extensions rather than silently
/// changing the candidate schedule.
pub(super) fn parse_candidate_manifest_with_metadata(
    path: &Path,
    image_names: &[String],
) -> Result<(Vec<(usize, usize)>, BTreeMap<String, String>), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read candidate manifest {path:?}: {error}"))?;
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let mut cursor = 0usize;
    let next = |cursor: &mut usize, label: &str| -> Result<&str, String> {
        let line = lines.get(*cursor).copied().ok_or_else(|| {
            format!("candidate manifest {path:?} is truncated while reading {label}")
        })?;
        *cursor += 1;
        Ok(line)
    };
    if next(&mut cursor, "header")? != CANDIDATE_MANIFEST_MAGIC {
        return Err(format!(
            "candidate manifest {path:?} has unsupported header (expected {CANDIDATE_MANIFEST_MAGIC})"
        ));
    }
    let image_header = next(&mut cursor, "image count")?;
    let image_fields: Vec<&str> = image_header.split_whitespace().collect();
    if image_fields.len() != 2 || image_fields[0] != "images" {
        return Err(format!(
            "candidate manifest {path:?} image count must be images N"
        ));
    }
    let image_count: usize = image_fields[1].parse().map_err(|error| {
        format!("candidate manifest {path:?} image count is not numeric: {error}")
    })?;
    if image_count != image_names.len() {
        return Err(format!(
            "candidate manifest {path:?} image count {} differs from loaded {}",
            image_count,
            image_names.len()
        ));
    }
    for expected_index in 0..image_count {
        let line = next(&mut cursor, "image entry")?;
        let mut fields = line.splitn(3, char::is_whitespace);
        let kind = fields.next().unwrap_or_default();
        let index = fields.next().unwrap_or_default();
        let name = fields.next().unwrap_or_default().trim();
        if kind != "image" || name.is_empty() {
            return Err(format!(
                "candidate manifest {path:?} image entry must be image INDEX NAME"
            ));
        }
        let index: usize = index.parse().map_err(|error| {
            format!("candidate manifest {path:?} image index is not numeric: {error}")
        })?;
        if index != expected_index || name != image_names[index] {
            return Err(format!(
                "candidate manifest {path:?} image entry {expected_index} does not match loaded image order"
            ));
        }
    }
    let mut metadata = BTreeMap::new();
    while let Some(line) = lines.get(cursor).copied() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.first().copied() != Some("metadata") {
            break;
        }
        if fields.len() != 3 || fields[1].is_empty() || fields[2].is_empty() {
            return Err(format!(
                "candidate manifest {path:?} metadata must be metadata KEY VALUE"
            ));
        }
        if metadata
            .insert(fields[1].to_owned(), fields[2].to_owned())
            .is_some()
        {
            return Err(format!(
                "candidate manifest {path:?} repeats metadata key {:?}",
                fields[1]
            ));
        }
        cursor += 1;
    }
    let pair_header = next(&mut cursor, "pair count")?;
    let pair_fields: Vec<&str> = pair_header.split_whitespace().collect();
    if pair_fields.len() != 2 || pair_fields[0] != "pairs" {
        return Err(format!(
            "candidate manifest {path:?} pair count must be pairs N"
        ));
    }
    let pair_count: usize = pair_fields[1].parse().map_err(|error| {
        format!("candidate manifest {path:?} pair count is not numeric: {error}")
    })?;
    let mut pairs = Vec::with_capacity(pair_count);
    let mut seen = HashSet::with_capacity(pair_count);
    for pair_number in 0..pair_count {
        let line = next(&mut cursor, "pair entry")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 3 || fields[0] != "pair" {
            return Err(format!(
                "candidate manifest {path:?} pair {pair_number} must be pair I J"
            ));
        }
        let i: usize = fields[1].parse().map_err(|error| {
            format!("candidate manifest {path:?} pair {pair_number} first index is not numeric: {error}")
        })?;
        let j: usize = fields[2].parse().map_err(|error| {
            format!("candidate manifest {path:?} pair {pair_number} second index is not numeric: {error}")
        })?;
        if i >= image_names.len() || j >= image_names.len() || i >= j {
            return Err(format!(
                "candidate manifest {path:?} pair {pair_number} must satisfy 0 <= I < J < {}",
                image_names.len()
            ));
        }
        if !seen.insert((i, j)) {
            return Err(format!(
                "candidate manifest {path:?} repeats pair ({i},{j})"
            ));
        }
        pairs.push((i, j));
    }
    if cursor != lines.len() {
        return Err(format!(
            "candidate manifest {path:?} has unexpected trailing data"
        ));
    }
    Ok((pairs, metadata))
}

/// Compute the SHA-256 binding for the canonical image order carried by a
/// persistent match plan.  This is intentionally independent of the source
/// manifest's pair schedule: compact shards contain only pair indices and
/// repeat this digest in their small envelope.
pub(super) fn candidate_image_manifest_sha256(image_names: &[String]) -> String {
    let mut digest = Sha256::new();
    for (index, name) in image_names.iter().enumerate() {
        digest.update(format!("image {index} {name}\n").as_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

/// Hash a candidate shard without retaining its text.  The plan already
/// carries this digest because the Python preparation stage validates it; the
/// worker repeats the check so a changed/corrupt shard cannot pass merely on
/// syntactically valid pair indices.
pub(super) fn candidate_file_sha256(path: &Path) -> Result<String, String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("cannot open candidate shard {path:?}: {error}"))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash candidate shard {path:?}: {error}"))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>())
}

/// Parse a compact v2 candidate shard and bind it to the canonical plan.
/// Unlike the legacy v1 parser, this never materializes an image-name list
/// from the shard.  The plan's image names are the sole authority for index
/// bounds and order, while the source/image SHA-256 envelope prevents a shard
/// from being replayed with a different source schedule or image order.
#[cfg(test)]
pub(super) fn parse_candidate_shard_v2(
    path: &Path,
    image_names: &[String],
    expected_source_manifest_sha256: Option<&str>,
    expected_image_manifest_sha256: Option<&str>,
) -> Result<Vec<(usize, usize)>, String> {
    let canonical_image_manifest_sha256 = candidate_image_manifest_sha256(image_names);
    parse_candidate_shard_v2_bound(
        path,
        image_names,
        expected_source_manifest_sha256,
        expected_image_manifest_sha256,
        &canonical_image_manifest_sha256,
    )
}

/// Parse a compact shard after the plan parser has already checked the
/// canonical image-order digest.  Keeping that digest outside this helper
/// avoids hashing the O(N) plan image list once per shard during the worker's
/// validation and matching passes.
pub(super) fn parse_candidate_shard_v2_bound(
    path: &Path,
    image_names: &[String],
    expected_source_manifest_sha256: Option<&str>,
    expected_image_manifest_sha256: Option<&str>,
    canonical_image_manifest_sha256: &str,
) -> Result<Vec<(usize, usize)>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read candidate shard {path:?}: {error}"))?;
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let mut cursor = 0usize;
    let next = |cursor: &mut usize, label: &str| -> Result<&str, String> {
        let line = lines.get(*cursor).copied().ok_or_else(|| {
            format!("candidate shard {path:?} is truncated while reading {label}")
        })?;
        *cursor += 1;
        Ok(line)
    };
    if next(&mut cursor, "header")? != CANDIDATE_SHARD_MAGIC_V2 {
        return Err(format!(
            "candidate shard {path:?} has unsupported v2 header (expected {CANDIDATE_SHARD_MAGIC_V2})"
        ));
    }
    let parse_hash = |line: &str, kind: &str| -> Result<String, String> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let value = fields.get(1).copied().unwrap_or_default();
        if fields.len() != 2
            || fields[0] != kind
            || value.len() != 64
            || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(format!("candidate shard {path:?} requires `{kind} SHA256`"));
        }
        Ok(value.to_ascii_lowercase())
    };
    let parse_count = |line: &str, kind: &str| -> Result<usize, String> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 2 || fields[0] != kind {
            return Err(format!("candidate shard {path:?} requires `{kind} N`"));
        }
        fields[1].parse::<usize>().map_err(|error| {
            format!("candidate shard {path:?} {kind} count is not numeric: {error}")
        })
    };
    let source_manifest_sha256 = parse_hash(
        next(&mut cursor, "source manifest hash")?,
        "source_manifest_sha256",
    )?;
    let expected_source = expected_source_manifest_sha256.ok_or_else(|| {
        format!("candidate shard {path:?} requires a v2 persistent plan source hash")
    })?;
    if source_manifest_sha256 != expected_source.to_ascii_lowercase() {
        return Err(format!(
            "candidate shard {path:?} source manifest hash differs from persistent plan"
        ));
    }
    let image_manifest_sha256 = parse_hash(
        next(&mut cursor, "image manifest hash")?,
        "image_manifest_sha256",
    )?;
    if image_manifest_sha256 != canonical_image_manifest_sha256 {
        return Err(format!(
            "candidate shard {path:?} image manifest hash does not match plan image order"
        ));
    }
    let expected_image = expected_image_manifest_sha256.ok_or_else(|| {
        format!("candidate shard {path:?} requires a v2 persistent plan image hash")
    })?;
    if image_manifest_sha256 != expected_image.to_ascii_lowercase() {
        return Err(format!(
            "candidate shard {path:?} image manifest hash differs from persistent plan"
        ));
    }
    let image_count = parse_count(next(&mut cursor, "image count")?, "images")?;
    if image_count != image_names.len() {
        return Err(format!(
            "candidate shard {path:?} image count {} differs from loaded {}",
            image_count,
            image_names.len()
        ));
    }
    let pair_count = parse_count(next(&mut cursor, "pair count")?, "pairs")?;
    let mut pairs = Vec::with_capacity(pair_count);
    let mut seen = HashSet::with_capacity(pair_count);
    for pair_number in 0..pair_count {
        let line = next(&mut cursor, "pair entry")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 3 || fields[0] != "pair" {
            return Err(format!(
                "candidate shard {path:?} pair {pair_number} must be pair I J"
            ));
        }
        let i = fields[1].parse::<usize>().map_err(|error| {
            format!(
                "candidate shard {path:?} pair {pair_number} first index is not numeric: {error}"
            )
        })?;
        let j = fields[2].parse::<usize>().map_err(|error| {
            format!(
                "candidate shard {path:?} pair {pair_number} second index is not numeric: {error}"
            )
        })?;
        if i >= image_names.len() || j >= image_names.len() || i >= j {
            return Err(format!(
                "candidate shard {path:?} pair {pair_number} must satisfy 0 <= I < J < {}",
                image_names.len()
            ));
        }
        if !seen.insert((i, j)) {
            return Err(format!("candidate shard {path:?} repeats pair ({i},{j})"));
        }
        pairs.push((i, j));
    }
    if cursor != lines.len() {
        return Err(format!(
            "candidate shard {path:?} has unexpected trailing data"
        ));
    }
    Ok(pairs)
}

/// Write a candidate manifest through a same-directory temporary file and
/// rename.  This keeps an interrupted cheap retrieval pass from leaving a
/// file that a later benchmark could mistake for a complete schedule.
#[cfg(test)]
pub(super) fn write_candidate_manifest(
    path: &Path,
    image_names: &[String],
    pairs: &[(usize, usize)],
) -> Result<(), String> {
    write_candidate_manifest_with_metadata(path, image_names, pairs, &BTreeMap::new())
}

/// Write a candidate manifest with a canonical metadata block.  Keys are
/// sorted by `BTreeMap`, making the bytes stable across runs and suitable for
/// the hash-bound shard index.
pub(super) fn write_candidate_manifest_with_metadata(
    path: &Path,
    image_names: &[String],
    pairs: &[(usize, usize)],
    metadata: &BTreeMap<String, String>,
) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|error| {
        format!("cannot create candidate manifest directory {parent:?}: {error}")
    })?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("candidate manifest path has no valid filename: {path:?}"))?;
    let temporary = parent.join(format!(".{file_name}.tmp"));
    let mut text = String::new();
    text.push_str(CANDIDATE_MANIFEST_MAGIC);
    text.push('\n');
    text.push_str(&format!("images {}\n", image_names.len()));
    for (index, name) in image_names.iter().enumerate() {
        if name.chars().any(char::is_whitespace) {
            return Err(format!(
                "candidate manifest cannot encode whitespace in image name {name:?}"
            ));
        }
        text.push_str(&format!("image {index} {name}\n"));
    }
    for (key, value) in metadata {
        if key.is_empty()
            || value.is_empty()
            || key.chars().any(char::is_whitespace)
            || value.chars().any(char::is_whitespace)
        {
            return Err(format!(
                "candidate manifest metadata must use non-empty whitespace-free KEY VALUE (got {key:?}={value:?})"
            ));
        }
        text.push_str(&format!("metadata {key} {value}\n"));
    }
    text.push_str(&format!("pairs {}\n", pairs.len()));
    let mut seen = HashSet::with_capacity(pairs.len());
    for &(i, j) in pairs {
        if i >= image_names.len() || j >= image_names.len() || i >= j {
            return Err(format!(
                "candidate pair ({i},{j}) is outside canonical image order"
            ));
        }
        if !seen.insert((i, j)) {
            return Err(format!("candidate pair ({i},{j}) is duplicated"));
        }
        text.push_str(&format!("pair {i} {j}\n"));
    }
    std::fs::write(&temporary, text).map_err(|error| {
        format!("cannot write temporary candidate manifest {temporary:?}: {error}")
    })?;
    std::fs::rename(&temporary, path).map_err(|error| {
        format!("cannot atomically install candidate manifest {path:?}: {error}")
    })?;
    Ok(())
}

/// Add the numeric-consecutive edges required by the opt-in sequence
/// relative-pose fallback.  Retrieval remains the normal source for every
/// other edge; only missing consecutive pairs are appended, so the flag does
/// not silently turn a bounded retrieval run into an exhaustive matcher.
pub(super) fn append_consecutive_stem_candidates(
    pairs: &mut Vec<(usize, usize)>,
    image_names: &[String],
) -> Result<usize, String> {
    let stem_values = numeric_stem_values(image_names)?;
    let mut by_stem: Vec<(u64, usize)> = stem_values
        .into_iter()
        .enumerate()
        .map(|(image, stem)| (stem, image))
        .collect();
    by_stem.sort_unstable();

    let existing: HashSet<(usize, usize)> = pairs.iter().copied().collect();
    let mut added = 0usize;
    for window in by_stem.windows(2) {
        let [(left_stem, left_image), (right_stem, right_image)] = window else {
            unreachable!("windows(2) always has two entries");
        };
        if right_stem.saturating_sub(*left_stem) != 1 {
            continue;
        }
        let pair = (*left_image, *right_image);
        if !existing.contains(&pair) {
            pairs.push(pair);
            added += 1;
        }
    }
    Ok(added)
}

pub(super) fn filter_imported_verified_pairs_by_stem_window(
    imported: Vec<ImportedVerifiedPair>,
    image_names: &[String],
    window: Option<u64>,
) -> Result<Vec<ImportedVerifiedPair>, String> {
    let Some(window) = window else {
        return Ok(imported);
    };
    let stem_values = numeric_stem_values(image_names)?;
    imported
        .into_iter()
        .filter_map(|pair| {
            match pair_within_stem_window((pair.image_i, pair.image_j), &stem_values, window) {
                Ok(true) => Some(Ok(pair)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect()
}

/// Bounded, deterministic descriptor sample for training a retrieval
/// vocabulary — k-means over *every* descriptor (262 k for 128×2048-kpt
/// images) is the pipeline's bottleneck and unnecessary for either VLAD or
/// the vocab-tree: both only need a representative sample. Strides the full
/// descriptor list down to ~`VOCAB_SAMPLE`. Shared by
/// [`candidate_pairs_vlad`] and [`candidate_pairs_vocab_tree`] (M3).
pub(super) const VLAD_VOCAB_SAMPLE: usize = 40_000;

pub(super) fn sampled_training_descriptors(features: &[FeatureSet]) -> Vec<&[f32]> {
    let all_desc: Vec<&[f32]> = features
        .iter()
        .flat_map(|f| f.descriptors.iter().map(|d| d.as_slice()))
        .collect();
    let stride = (all_desc.len() / VLAD_VOCAB_SAMPLE).max(1);
    all_desc.iter().step_by(stride).copied().collect()
}

/// All `(i, j)` pairs with `i < j` — the exhaustive fallback shared by both
/// pair sources.
pub(super) fn all_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    pairs
}

/// Return the exact same `(image, cosine)` top-K as a full descending sort,
/// while retaining at most K scored rows for one query. The deterministic
/// image-index tie break is part of the candidate-manifest contract.
pub(super) fn exact_topk_similar_images(
    query: usize,
    globals: &[Vec<f32>],
    topk: usize,
) -> Vec<(usize, f32)> {
    if topk == 0 {
        return Vec::new();
    }
    let mut best = Vec::<(usize, f32)>::with_capacity(topk.min(globals.len().saturating_sub(1)));
    for candidate in 0..globals.len() {
        if candidate == query {
            continue;
        }
        let row = (
            candidate,
            cosine_similarity(&globals[query], &globals[candidate]),
        );
        insert_exact_topk_row(&mut best, row, topk);
    }
    best
}

fn insert_exact_topk_row(best: &mut Vec<(usize, f32)>, row: (usize, f32), topk: usize) {
    let position = best.partition_point(|existing| {
        existing
            .1
            .total_cmp(&row.1)
            .reverse()
            .then_with(|| existing.0.cmp(&row.0))
            .is_lt()
    });
    if position < topk {
        best.insert(position, row);
        if best.len() > topk {
            best.pop();
        }
    }
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn lsh_signature(global: &[f32], table: usize, bits: usize) -> (u64, Vec<usize>) {
    let mut projections = vec![0.0f32; bits];
    let table_seed = (table as u64).wrapping_mul(0xd6e8feb86659fd93);
    for (dimension, &value) in global.iter().enumerate() {
        let hash = splitmix64((dimension as u64) ^ table_seed);
        let bit = (hash as usize) % bits;
        let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
        projections[bit] += sign * value;
    }
    let mut signature = 0u64;
    for (bit, &projection) in projections.iter().enumerate() {
        if projection >= 0.0 {
            signature |= 1u64 << bit;
        }
    }
    let mut probe_bits = (0..bits).collect::<Vec<_>>();
    probe_bits.sort_by(|&lhs, &rhs| {
        projections[lhs]
            .abs()
            .total_cmp(&projections[rhs].abs())
            .then_with(|| lhs.cmp(&rhs))
    });
    (signature, probe_bits)
}

/// Deterministic multi-table feature-hash LSH. Buckets propose a bounded pool;
/// original VLAD cosine re-ranks that pool, so approximation affects only
/// neighbour recall rather than the score ordering of retrieved images.
pub(super) fn candidate_pairs_vlad_lsh_scored(
    globals: &[Vec<f32>],
    topk: usize,
    tables: usize,
    bits: usize,
    probes: usize,
) -> Vec<((usize, usize), f32)> {
    let n = globals.len();
    if n <= topk + 1 {
        return all_pairs(n).into_iter().map(|pair| (pair, 0.0)).collect();
    }
    let mut signatures = vec![vec![0u64; n]; tables];
    let mut probe_orders = vec![vec![Vec::<usize>::new(); n]; tables];
    let mut buckets = (0..tables)
        .map(|_| BTreeMap::<u64, Vec<usize>>::new())
        .collect::<Vec<_>>();
    for table in 0..tables {
        for (image, global) in globals.iter().enumerate() {
            let (signature, order) = lsh_signature(global, table, bits);
            signatures[table][image] = signature;
            probe_orders[table][image] = order;
            buckets[table].entry(signature).or_default().push(image);
        }
    }

    let mut scores = BTreeMap::<(usize, usize), f32>::new();
    let mut fallback_queries = 0usize;
    let mut pool_sum = 0usize;
    let mut pool_max = 0usize;
    for query in 0..n {
        let mut pool = HashSet::<usize>::new();
        for table in 0..tables {
            let signature = signatures[table][query];
            if let Some(images) = buckets[table].get(&signature) {
                pool.extend(images.iter().copied().filter(|&image| image != query));
            }
            for &bit in probe_orders[table][query].iter().take(probes) {
                if let Some(images) = buckets[table].get(&(signature ^ (1u64 << bit))) {
                    pool.extend(images.iter().copied().filter(|&image| image != query));
                }
            }
        }
        if pool.len() < topk {
            fallback_queries += 1;
            pool.extend((0..n).filter(|&image| image != query));
        }
        pool_sum += pool.len();
        pool_max = pool_max.max(pool.len());
        let mut best = Vec::<(usize, f32)>::with_capacity(topk);
        for candidate in pool {
            let row = (
                candidate,
                cosine_similarity(&globals[query], &globals[candidate]),
            );
            insert_exact_topk_row(&mut best, row, topk);
        }
        for (candidate, score) in best {
            let pair = (query.min(candidate), query.max(candidate));
            scores
                .entry(pair)
                .and_modify(|best| *best = best.max(score))
                .or_insert(score);
        }
    }
    println!(
        "VLAD LSH: tables={tables} bits={bits} probes={probes} mean_pool={:.1} max_pool={pool_max} exact_fallback_queries={fallback_queries}",
        pool_sum as f64 / n.max(1) as f64,
    );
    scores.into_iter().collect()
}

/// Candidate image pairs `(i, j)` with `i < j` from flat-VLAD top-K cosine
/// retrieval (or all pairs when `exhaustive`) — the pre-M3 pair source,
/// unchanged.
pub(super) fn candidate_pairs_vlad_scored(
    features: &[FeatureSet],
    vocab_size: usize,
    topk: usize,
    exhaustive: bool,
    mutual: bool,
) -> Vec<((usize, usize), f32)> {
    let n = features.len();
    if exhaustive || n <= topk + 1 {
        return all_pairs(n).into_iter().map(|pair| (pair, 0.0)).collect();
    }

    let sample = sampled_training_descriptors(features);
    let Some(vocab) = Vocabulary::build(&sample, vocab_size, 10, 0) else {
        // Fall back to exhaustive if the vocabulary cannot be built.
        return all_pairs(n).into_iter().map(|pair| (pair, 0.0)).collect();
    };
    // VLAD aggregation is a pure per-image function; the indexed parallel
    // map preserves image order, so the resulting globals are identical.
    let globals: Vec<Vec<f32>> = features
        .par_iter()
        .map(|f| vlad(&f.descriptors, &vocab))
        .collect();

    candidate_pairs_vlad_scored_from_globals(&globals, topk, mutual)
}

pub(super) fn candidate_pairs_vlad_scored_from_globals(
    globals: &[Vec<f32>],
    topk: usize,
    mutual: bool,
) -> Vec<((usize, usize), f32)> {
    let n = globals.len();

    // Each image's exact top-k is independent of every other image's, and
    // `exact_topk_similar_images` is deterministic.  Computing the per-image
    // neighbourhoods in parallel therefore leaves the admitted pair set and
    // its max scores byte-identical while removing the single-core O(N^2 * D)
    // similarity scan from the critical path.
    let per_query: Vec<Vec<(usize, f32)>> = (0..n)
        .into_par_iter()
        .map(|i| exact_topk_similar_images(i, globals, topk))
        .collect();

    let mut scores = std::collections::BTreeMap::<(usize, usize), f32>::new();
    if mutual {
        let neighbors: Vec<HashSet<usize>> = per_query
            .iter()
            .map(|top| top.iter().map(|&(j, _)| j).collect())
            .collect();
        // Admission is exactly symmetric and independent of image traversal
        // order: both directions of the pair must appear in the top-k.
        for (i, top) in per_query.iter().enumerate() {
            for &(j, score) in top {
                if neighbors[i].contains(&j) && neighbors[j].contains(&i) {
                    let pair = (i.min(j), i.max(j));
                    scores
                        .entry(pair)
                        .and_modify(|best| *best = best.max(score))
                        .or_insert(score);
                }
            }
        }
    } else {
        for (i, top) in per_query.iter().enumerate() {
            for &(j, score) in top {
                let pair = (i.min(j), i.max(j));
                scores
                    .entry(pair)
                    .and_modify(|best| *best = best.max(score))
                    .or_insert(score);
            }
        }
    }
    scores.into_iter().collect()
}

pub(super) fn candidate_pairs_vlad(
    features: &[FeatureSet],
    vocab_size: usize,
    topk: usize,
    exhaustive: bool,
) -> Vec<(usize, usize)> {
    candidate_pairs_vlad_scored(features, vocab_size, topk, exhaustive, false)
        .into_iter()
        .map(|(pair, _)| pair)
        .collect()
}

fn candidate_pairs_vlad_mutual(
    features: &[FeatureSet],
    vocab_size: usize,
    topk: usize,
    exhaustive: bool,
) -> Vec<(usize, usize)> {
    candidate_pairs_vlad_scored(features, vocab_size, topk, exhaustive, true)
        .into_iter()
        .map(|(pair, _)| pair)
        .collect()
}

/// Candidate pairs from a bounded union of local numeric-stem overlap and
/// VLAD retrieval. Local edges are selected before retrieval edges under a
/// budget, then retrieval-only edges are ranked by pre-match VLAD score.
#[cfg(test)]
pub(super) fn candidate_pairs_vlad_union(
    features: &[FeatureSet],
    image_names: &[String],
    vocab_size: usize,
    topk: usize,
    local_window: u64,
    budget: Option<usize>,
) -> Result<Vec<(usize, usize)>, String> {
    candidate_pairs_vlad_union_with_grouping(
        features,
        image_names,
        vocab_size,
        topk,
        local_window,
        budget,
        false,
    )
}

fn candidate_pairs_vlad_union_with_grouping(
    features: &[FeatureSet],
    image_names: &[String],
    vocab_size: usize,
    topk: usize,
    local_window: u64,
    budget: Option<usize>,
    rig_local_grouping: bool,
) -> Result<Vec<(usize, usize)>, String> {
    let local = if rig_local_grouping {
        rig_local_pairs(image_names, local_window)?
    } else {
        filter_pairs_by_stem_window(all_pairs(features.len()), image_names, Some(local_window))?
    };
    let retrieval = candidate_pairs_vlad_scored(features, vocab_size, topk, false, false);
    let local_set: HashSet<(usize, usize)> = local.iter().copied().collect();
    let mut ranked: Vec<((usize, usize), bool, f32)> = retrieval
        .into_iter()
        .map(|(pair, score)| (pair, local_set.contains(&pair), score))
        .collect();
    let mut seen: HashSet<(usize, usize)> = ranked.iter().map(|(pair, _, _)| *pair).collect();
    for pair in local {
        if seen.insert(pair) {
            ranked.push((pair, true, f32::NEG_INFINITY));
        }
    }
    ranked.sort_by(|lhs, rhs| {
        rhs.1
            .cmp(&lhs.1)
            .then_with(|| rhs.2.total_cmp(&lhs.2))
            .then_with(|| lhs.0.cmp(&rhs.0))
    });
    if let Some(budget) = budget {
        ranked.truncate(budget);
    }
    ranked.sort_unstable_by_key(|(pair, _, _)| *pair);
    Ok(ranked.into_iter().map(|(pair, _, _)| pair).collect())
}

/// Candidate image pairs `(i, j)` with `i < j` from the M3 hierarchical
/// vocab-tree (`visloc_rs::vision::vocab_tree`, COLMAP's
/// `VocabTreePairGenerator`-equivalent, `docs/colmap_port_plan.md`'s M3
/// milestone), or all pairs when `exhaustive`.
///
/// Trains the hierarchical vocabulary on the same bounded descriptor sample
/// [`candidate_pairs_vlad`] uses, indexes every image's *full* descriptor
/// set (unsampled — retrieval quality for images the tree has never seen
/// depends on it having every one of their features, unlike the shared
/// training sample which only needs to be representative), then queries each
/// image against the finalized tree with its own descriptors, keeping the
/// top `vocab_tree_num_images` other images per query
/// ([`generate_pairs`]/[`VocabTreePairGeneratorOptions`]).
pub(super) fn candidate_pairs_vocab_tree(
    features: &[FeatureSet],
    branching_factor: usize,
    depth: usize,
    num_images: usize,
    exhaustive: bool,
) -> Vec<(usize, usize)> {
    let n = features.len();
    if exhaustive {
        return all_pairs(n);
    }

    let sample = sampled_training_descriptors(features);
    let hkm_options = HkmBuildOptions {
        branching_factor,
        depth,
        ..HkmBuildOptions::default()
    };
    let vocab_tree_options = VocabTreeOptions::default();
    let Some(mut tree) = VocabTree::build(&sample, &hkm_options, &vocab_tree_options) else {
        // Fall back to exhaustive if the vocabulary cannot be built (mirrors
        // candidate_pairs_vlad's own degenerate-input fallback).
        return all_pairs(n);
    };
    for (i, f) in features.iter().enumerate() {
        tree.add_image(i, &f.descriptors);
    }
    tree.finalize();
    println!(
        "vocab-tree: {} leaf words (requested {}^{}={}), {} images indexed",
        tree.num_words(),
        branching_factor,
        depth,
        branching_factor.pow(depth as u32),
        tree.num_images(),
    );

    let image_descriptors: Vec<Vec<Vec<f32>>> =
        features.iter().map(|f| f.descriptors.clone()).collect();
    generate_pairs(
        &tree,
        &image_descriptors,
        &VocabTreePairGeneratorOptions { num_images },
    )
}

/// How many transitive-expansion rounds
/// ([`PairSource::Transitive`], COLMAP's `TransitivePairGenerator`) run
/// after the vocab-tree base pass. Two rounds cover the common
/// "bridge image chains a-b-c and b-d-e" real-scene topology; each round
/// only proposes pairs not proposed before, so cost is bounded by the
/// verified-graph neighbourhood size.
pub(super) const TRANSITIVE_ROUNDS: usize = 2;

/// Candidate image pairs `(i, j)` with `i < j` — dispatches on
/// [`PairSource`] (`docs/colmap_port_plan.md`'s M3 A/B switch); `exhaustive`
/// overrides either source, matching pre-M3 behaviour.
/// [`PairSource::Transitive`] returns its *base* pass here (vocab-tree);
/// the transitive expansion happens in [`expand_transitive`] after those
/// base pairs are verified, mirroring COLMAP's generator running against
/// an existing match table.
pub(super) fn candidate_pairs(
    features: &[FeatureSet],
    image_names: &[String],
    args: &Args,
) -> Result<Vec<(usize, usize)>, String> {
    match args.pair_source {
        PairSource::Vlad => Ok(candidate_pairs_vlad(
            features,
            args.vocab_size,
            args.retrieval_topk,
            args.exhaustive,
        )),
        PairSource::VladMutual => Ok(candidate_pairs_vlad_mutual(
            features,
            args.vocab_size,
            args.retrieval_topk,
            args.exhaustive,
        )),
        PairSource::VladUnion => candidate_pairs_vlad_union_with_grouping(
            features,
            image_names,
            args.vocab_size,
            args.retrieval_topk,
            args.local_stem_window
                .expect("validated vlad-union local window"),
            args.candidate_budget,
            args.rig_local_grouping,
        ),
        PairSource::TemporalPyramid => candidate_pairs_temporal_pyramid(
            features,
            image_names,
            args.vocab_size,
            args.retrieval_topk,
            args.temporal_pyramid_max_offset,
            args.candidate_budget,
            args.rig_frame_manifest.as_deref(),
            args.retrieval_component_manifest.as_deref(),
            args.retrieval_min_frame_gap,
        ),
        PairSource::VocabTree | PairSource::Transitive => Ok(candidate_pairs_vocab_tree(
            features,
            args.vocab_tree_branching,
            args.vocab_tree_depth,
            args.vocab_tree_num_images,
            args.exhaustive,
        )),
    }
}

/// Metadata written beside generated candidate pairs.  It is intentionally
/// descriptive rather than used to reconstruct pairs: the image-name-bound
/// pair list remains the authority, while this block makes an archived
/// manifest auditable and lets sharding tools preserve the exact schedule.
pub(super) fn effective_ann_bits(requested: usize, image_count: usize) -> usize {
    if requested != 0 {
        return requested;
    }
    let mut bits = 6usize;
    let mut scale = (image_count / 1_000).max(1);
    while scale >= 2 && bits < 63 {
        bits += 1;
        scale /= 2;
    }
    bits
}

pub(super) fn candidate_manifest_metadata(
    args: &Args,
    image_count: usize,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    let (policy, pair_source) = match args.pair_source {
        PairSource::Vlad => ("vlad-topk-v1", "vlad"),
        PairSource::VladMutual => ("vlad-mutual-v1", "vlad-mutual"),
        PairSource::VladUnion => ("vlad-union-v1", "vlad-union"),
        PairSource::TemporalPyramid => ("temporal-pyramid-v1", "temporal-pyramid"),
        PairSource::VocabTree => ("vocab-tree-v1", "vocab-tree"),
        PairSource::Transitive => ("transitive-v1", "transitive"),
    };
    metadata.insert("candidate_policy".to_owned(), policy.to_owned());
    metadata.insert("pair_source".to_owned(), pair_source.to_owned());
    metadata.insert("retrieval_topk".to_owned(), args.retrieval_topk.to_string());
    if args.retrieval_backend == RetrievalBackend::Lsh {
        metadata.insert("retrieval_backend".to_owned(), "vlad-lsh-v1".to_owned());
        metadata.insert("ann_tables".to_owned(), args.ann_tables.to_string());
        metadata.insert(
            "ann_bits".to_owned(),
            effective_ann_bits(args.ann_bits, image_count).to_string(),
        );
        if args.ann_bits == 0 {
            metadata.insert("ann_bits_mode".to_owned(), "auto-v1".to_owned());
        }
        metadata.insert("ann_probes".to_owned(), args.ann_probes.to_string());
    }
    if args.pair_source == PairSource::VladUnion {
        metadata.insert(
            "local_grouping".to_owned(),
            if args.rig_local_grouping {
                "rig-prefix-timestamp-v1"
            } else {
                "unique-numeric-stem-v1"
            }
            .to_owned(),
        );
        metadata.insert(
            "cross_camera_rule".to_owned(),
            if args.rig_local_grouping {
                "same-timestamp"
            } else {
                "none"
            }
            .to_owned(),
        );
        metadata.insert(
            "local_stem_window".to_owned(),
            args.local_stem_window
                .expect("validated vlad-union local window")
                .to_string(),
        );
        metadata.insert(
            "candidate_budget".to_owned(),
            args.candidate_budget
                .map_or_else(|| "none".to_owned(), |budget| budget.to_string()),
        );
    }
    if args.pair_source == PairSource::TemporalPyramid {
        metadata.insert(
            "local_grouping".to_owned(),
            if args.rig_frame_manifest.is_some() {
                "generalized-rig-manifest-v1"
            } else {
                "rig-prefix-timestamp-v1"
            }
            .to_owned(),
        );
        if let Some(path) = &args.rig_frame_manifest {
            metadata.insert(
                "rig_frame_manifest".to_owned(),
                path.to_string_lossy().into_owned(),
            );
        }
        if let Some(path) = &args.retrieval_component_manifest {
            metadata.insert(
                "retrieval_component_manifest".to_owned(),
                path.to_string_lossy().into_owned(),
            );
            metadata.insert(
                "retrieval_fill_policy".to_owned(),
                "component-pair-round-robin-then-global-score-v1".to_owned(),
            );
        }
        if let Some(gap) = args.retrieval_min_frame_gap {
            metadata.insert("retrieval_min_frame_gap".to_owned(), gap.to_string());
            metadata.insert(
                "retrieval_temporal_exclusion".to_owned(),
                "rig-frame-absolute-gap-v1".to_owned(),
            );
        }
        metadata.insert("cross_camera_rule".to_owned(), "same-timestamp".to_owned());
        metadata.insert(
            "temporal_offsets".to_owned(),
            temporal_pyramid_offsets_string(args.temporal_pyramid_max_offset),
        );
        metadata.insert(
            "temporal_pyramid_max_offset".to_owned(),
            args.temporal_pyramid_max_offset.to_string(),
        );
        metadata.insert(
            "temporal_long_level_sampling".to_owned(),
            "dense-through-32;stride=offset/16".to_owned(),
        );
        if args.retrieval_component_manifest.is_none() {
            metadata.insert(
                "retrieval_fill_policy".to_owned(),
                "global-score-desc-v1".to_owned(),
            );
        }
        metadata.insert(
            "candidate_budget".to_owned(),
            args.candidate_budget
                .map_or_else(|| "none".to_owned(), |budget| budget.to_string()),
        );
    }
    metadata
}

/// One round of COLMAP's `TransitivePairGenerator` (`src/colmap/pairing.cc`):
/// from the verified-match adjacency, propose every `(i, k)` with `i < k`
/// that shares a common matched partner `j` but has no direct pair yet.
pub(super) fn expand_transitive(
    pairwise: &[PairwiseMatches],
    already_proposed: &HashSet<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let mut neighbors: HashMap<usize, HashSet<usize>> = HashMap::new();
    for p in pairwise {
        if p.matches.is_empty() {
            continue;
        }
        neighbors.entry(p.image_i).or_default().insert(p.image_j);
        neighbors.entry(p.image_j).or_default().insert(p.image_i);
    }
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut seen = already_proposed.clone();
    for (&i, ni) in &neighbors {
        for &j in ni {
            // Partners of partners.
            let Some(nj) = neighbors.get(&j) else {
                continue;
            };
            for &k in nj {
                if k == i {
                    continue;
                }
                let key = if i < k { (i, k) } else { (k, i) };
                if seen.insert(key) {
                    out.push(key);
                }
            }
        }
    }
    out.sort_unstable();
    out
}
