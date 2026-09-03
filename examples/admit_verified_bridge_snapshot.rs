//! Preserve a frozen verified-pair base and admit only verified component bridges.
//!
//! The augmented feature bank must retain every base feature as an unchanged
//! per-image prefix.  This utility keeps the base snapshot's pair records
//! byte-for-byte in their original order, computes connected components over
//! rig frames in that base graph, then appends only augmented pairs whose
//! endpoints belong to different base components.  It therefore exposes new
//! verified connectivity without perturbing already-good component-internal
//! tracks with every extra correspondence.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use visloc_rs::verified_pair_snapshot::{merge_owned, read, write_atomic, PairRecord, Snapshot};

fn usage() -> ! {
    eprintln!(
        "usage: admit_verified_bridge_snapshot \
         --base-snapshot PATH --augmented-snapshot PATH \
         --base-features-dir DIR --rig-manifest PATH --output PATH \
         [--feature-suffix _features.txt] [--include-prefix-supplements] \
         [--repair-registered-components PATH] [--include-all-new-pairs] \
         [--frozen-track-extensions --structure-pair-prefix COUNT \
          --frozen-track-extension-cap COUNT]"
    );
    std::process::exit(2);
}

#[derive(Debug)]
struct FeatureDisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl FeatureDisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            size: vec![1; len],
        }
    }

    fn find(&mut self, mut value: usize) -> usize {
        while self.parent[value] != value {
            self.parent[value] = self.parent[self.parent[value]];
            value = self.parent[value];
        }
        value
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return;
        }
        if self.size[left] < self.size[right]
            || (self.size[left] == self.size[right] && left > right)
        {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        self.size[left] += self.size[right];
    }
}

#[derive(Debug, Clone, Copy)]
struct FrozenExtensionCandidate {
    pair_index: usize,
    match_index: usize,
    track_root: usize,
    singleton_node: usize,
    target_image: usize,
    pair_support: usize,
    e_inliers: u64,
    frame_gap: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrozenExtensionStats {
    structure_pairs: usize,
    deferred_pairs: usize,
    candidate_matches: usize,
    admitted_pairs: usize,
    admitted_matches: usize,
    duplicate_image_rejections: usize,
    track_cap_rejections: usize,
    singleton_reuse_rejections: usize,
}

fn next_path(args: &[String], index: &mut usize, flag: &str) -> Result<PathBuf, String> {
    let value = args
        .get(*index + 1)
        .ok_or_else(|| format!("{flag} requires PATH"))?;
    *index += 1;
    if value.trim().is_empty() {
        return Err(format!("{flag} requires a non-empty PATH"));
    }
    Ok(PathBuf::from(value))
}

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            size: vec![1; len],
        }
    }

    fn find(&mut self, mut value: usize) -> usize {
        while self.parent[value] != value {
            self.parent[value] = self.parent[self.parent[value]];
            value = self.parent[value];
        }
        value
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return;
        }
        if self.size[left] < self.size[right]
            || (self.size[left] == self.size[right] && left > right)
        {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        self.size[left] += self.size[right];
    }
}

fn parse_rig_frames(path: &Path, image_names: &[String]) -> Result<(Vec<usize>, usize), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read rig manifest {}: {error}", path.display()))?;
    let image_index = image_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut image_frames = vec![None; image_names.len()];
    let mut frame_ids = HashSet::new();
    for (zero_line, raw) in text.lines().enumerate() {
        let fields = raw.split_whitespace().collect::<Vec<_>>();
        if fields.first().copied() != Some("F") {
            continue;
        }
        if fields.len() != 4 {
            return Err(format!(
                "rig manifest {}:{} has malformed frame row",
                path.display(),
                zero_line + 1
            ));
        }
        let frame = fields[1].parse::<usize>().map_err(|error| {
            format!(
                "rig manifest {}:{} has invalid frame id: {error}",
                path.display(),
                zero_line + 1
            )
        })?;
        let index = *image_index.get(fields[2]).ok_or_else(|| {
            format!(
                "rig manifest {}:{} names unknown image {:?}",
                path.display(),
                zero_line + 1,
                fields[2]
            )
        })?;
        if image_frames[index].replace(frame).is_some() {
            return Err(format!("rig manifest assigns image {:?} twice", fields[2]));
        }
        frame_ids.insert(frame);
    }
    if image_frames.iter().any(Option::is_none) {
        return Err("rig manifest does not assign every snapshot image".into());
    }
    let frame_count = frame_ids.len();
    if frame_ids.iter().copied().collect::<HashSet<_>>() != (0..frame_count).collect() {
        return Err("rig frame ids must be contiguous from zero".into());
    }
    Ok((
        image_frames.into_iter().map(Option::unwrap).collect(),
        frame_count,
    ))
}

fn parse_registered_frames(
    path: &Path,
    image_names: &[String],
    image_frames: &[usize],
    frame_count: usize,
) -> Result<Vec<bool>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "read registered component manifest {}: {error}",
            path.display()
        )
    })?;
    let image_index = image_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut registered = vec![false; frame_count];
    let mut rows = 0usize;
    for (zero_line, raw) in text.lines().enumerate() {
        let fields = raw.split_whitespace().collect::<Vec<_>>();
        if fields.is_empty() || fields[0].starts_with('#') {
            continue;
        }
        if fields.first().copied() != Some("C") || fields.len() != 3 {
            return Err(format!(
                "registered component manifest {}:{} has malformed row",
                path.display(),
                zero_line + 1
            ));
        }
        let image = *image_index.get(fields[2]).ok_or_else(|| {
            format!(
                "registered component manifest {}:{} names unknown image {:?}",
                path.display(),
                zero_line + 1,
                fields[2]
            )
        })?;
        registered[image_frames[image]] = true;
        rows += 1;
    }
    if rows == 0 {
        return Err("registered component manifest contains no C rows".into());
    }
    Ok(registered)
}

fn feature_path(directory: &Path, image_name: &str, suffix: &str) -> Result<PathBuf, String> {
    let stem = Path::new(image_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("image name has no UTF-8 stem: {image_name:?}"))?;
    Ok(directory.join(format!("{stem}{suffix}")))
}

fn feature_count(path: &Path) -> Result<u64, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read base feature file {}: {error}", path.display()))?;
    Ok(text
        .lines()
        .filter(|line| {
            let line = line.trim();
            !line.is_empty() && !line.starts_with('#')
        })
        .count() as u64)
}

fn validate_envelopes(base: &Snapshot, augmented: &Snapshot) -> Result<(), String> {
    if base.image_names != augmented.image_names
        || base.image_manifest_hash != augmented.image_manifest_hash
    {
        return Err("base and augmented image manifests differ".into());
    }
    if base.width != augmented.width
        || base.height != augmented.height
        || base.intrinsics_bits != augmented.intrinsics_bits
    {
        return Err("base and augmented camera envelopes differ".into());
    }
    if base.verifier_config_hash != augmented.verifier_config_hash
        || base.verifier_config != augmented.verifier_config
    {
        return Err("base and augmented verifier configurations differ".into());
    }
    if base.feature_counts.len() != augmented.feature_counts.len() {
        return Err("base and augmented feature count vectors differ in length".into());
    }
    for (index, (&base_count, &augmented_count)) in base
        .feature_counts
        .iter()
        .zip(&augmented.feature_counts)
        .enumerate()
    {
        if base_count > augmented_count {
            return Err(format!(
                "augmented image {index} has {augmented_count} features, fewer than base prefix {base_count}"
            ));
        }
    }
    Ok(())
}

fn feature_offsets(feature_counts: &[u64]) -> Result<Vec<usize>, String> {
    let mut offsets = Vec::with_capacity(feature_counts.len() + 1);
    offsets.push(0usize);
    for &count in feature_counts {
        let count = usize::try_from(count)
            .map_err(|_| "snapshot feature count does not fit usize".to_owned())?;
        let next = offsets
            .last()
            .copied()
            .expect("offset vector is non-empty")
            .checked_add(count)
            .ok_or_else(|| "snapshot total feature count overflows usize".to_owned())?;
        offsets.push(next);
    }
    Ok(offsets)
}

fn feature_node(offsets: &[usize], image: u64, keypoint: u64) -> Result<(usize, usize), String> {
    let image = usize::try_from(image).map_err(|_| "image index does not fit usize".to_owned())?;
    let keypoint =
        usize::try_from(keypoint).map_err(|_| "keypoint index does not fit usize".to_owned())?;
    let start = *offsets
        .get(image)
        .ok_or_else(|| format!("image index {image} is outside feature offsets"))?;
    let end = *offsets
        .get(image + 1)
        .ok_or_else(|| format!("image index {image} is outside feature offsets"))?;
    if keypoint >= end - start {
        return Err(format!(
            "keypoint {keypoint} is outside image {image} feature count {}",
            end - start
        ));
    }
    Ok((image, start + keypoint))
}

fn retain_pair_matches(pair: &PairRecord, selected: &[usize]) -> PairRecord {
    let mut output = pair.clone();
    output.accepted_inlier_indices = selected
        .iter()
        .map(|&index| pair.accepted_inlier_indices[index])
        .collect();
    output.matches = selected.iter().map(|&index| pair.matches[index]).collect();
    let selected_matches = output.matches.iter().copied().collect::<HashSet<_>>();
    if let Some(essential) = pair.essential_matches.as_ref() {
        let retained = pair
            .essential_inlier_indices
            .iter()
            .copied()
            .zip(essential.iter().copied())
            .filter(|(_, matched)| selected_matches.contains(matched))
            .collect::<Vec<_>>();
        output.essential_inlier_indices = retained.iter().map(|entry| entry.0).collect();
        output.essential_matches = Some(retained.iter().map(|entry| entry.1).collect());
        output.e_inlier_count = retained.len() as u64;
    } else {
        output.essential_inlier_indices.clear();
        output.e_inlier_count = output.e_inlier_count.min(output.matches.len() as u64);
    }
    output.f_inlier_count = output.f_inlier_count.min(output.matches.len() as u64);
    output.h_inlier_count = output.h_inlier_count.min(output.matches.len() as u64);
    output
}

/// Insert only frozen-prefix track extensions before the base snapshot's
/// deferred tail. A candidate is eligible when exactly one endpoint belongs
/// to a non-trivial track in the frozen structure prefix and the other endpoint
/// is still a singleton in that same frozen graph. Existing tracks are never
/// merged, and a track never gains a second observation from the same image.
fn admit_frozen_track_extensions(
    mut base: Snapshot,
    augmented: &Snapshot,
    image_frames: &[usize],
    structure_pair_prefix: usize,
    max_extensions_per_track: usize,
) -> Result<(Snapshot, FrozenExtensionStats), String> {
    if structure_pair_prefix > base.pairs.len() {
        return Err(format!(
            "structure pair prefix {structure_pair_prefix} exceeds {} base pairs",
            base.pairs.len()
        ));
    }
    if max_extensions_per_track == 0 {
        return Err("frozen track extension cap must be positive".into());
    }
    let offsets = feature_offsets(&base.feature_counts)?;
    let total_features = *offsets.last().expect("feature offsets are non-empty");
    let mut tracks = FeatureDisjointSet::new(total_features);
    for pair in &base.pairs[..structure_pair_prefix] {
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (_, left) = feature_node(&offsets, pair.image_i, keypoint_i)?;
            let (_, right) = feature_node(&offsets, pair.image_j, keypoint_j)?;
            tracks.union(left, right);
        }
    }
    let mut roots = Vec::with_capacity(total_features);
    for node in 0..total_features {
        roots.push(tracks.find(node));
    }
    let mut occupied = HashSet::new();
    for image in 0..base.feature_counts.len() {
        for node in offsets[image]..offsets[image + 1] {
            let root = roots[node];
            if tracks.size[root] > 1 {
                occupied.insert((root, image));
            }
        }
    }
    let base_keys = base
        .pairs
        .iter()
        .map(|pair| {
            (
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
            )
        })
        .collect::<HashSet<_>>();
    let mut candidates = Vec::new();
    for (pair_index, pair) in augmented.pairs.iter().enumerate() {
        let key = (
            pair.image_i.min(pair.image_j),
            pair.image_i.max(pair.image_j),
        );
        if base_keys.contains(&key) {
            continue;
        }
        let image_i = usize::try_from(pair.image_i)
            .map_err(|_| "augmented image_i does not fit usize".to_owned())?;
        let image_j = usize::try_from(pair.image_j)
            .map_err(|_| "augmented image_j does not fit usize".to_owned())?;
        let frame_gap = image_frames
            .get(image_i)
            .ok_or_else(|| "augmented image_i is outside rig frames".to_owned())?
            .abs_diff(
                *image_frames
                    .get(image_j)
                    .ok_or_else(|| "augmented image_j is outside rig frames".to_owned())?,
            );
        for (match_index, &(keypoint_i, keypoint_j)) in pair.matches.iter().enumerate() {
            let (_, left) = feature_node(&offsets, pair.image_i, keypoint_i)?;
            let (_, right) = feature_node(&offsets, pair.image_j, keypoint_j)?;
            let left_root = roots[left];
            let right_root = roots[right];
            let left_established = tracks.size[left_root] > 1;
            let right_established = tracks.size[right_root] > 1;
            let (track_root, singleton_node, target_image) =
                match (left_established, right_established) {
                    (true, false) => (left_root, right, image_j),
                    (false, true) => (right_root, left, image_i),
                    _ => continue,
                };
            candidates.push(FrozenExtensionCandidate {
                pair_index,
                match_index,
                track_root,
                singleton_node,
                target_image,
                pair_support: pair.matches.len(),
                e_inliers: pair.e_inlier_count,
                frame_gap,
            });
        }
    }
    candidates.sort_unstable_by(|left, right| {
        right
            .pair_support
            .cmp(&left.pair_support)
            .then_with(|| right.e_inliers.cmp(&left.e_inliers))
            .then_with(|| right.frame_gap.cmp(&left.frame_gap))
            .then_with(|| left.pair_index.cmp(&right.pair_index))
            .then_with(|| left.match_index.cmp(&right.match_index))
    });
    let candidate_matches = candidates.len();
    let mut selected = vec![Vec::new(); augmented.pairs.len()];
    let mut used_singletons = HashSet::new();
    let mut extension_counts = HashMap::<usize, usize>::new();
    let mut duplicate_image_rejections = 0usize;
    let mut track_cap_rejections = 0usize;
    let mut singleton_reuse_rejections = 0usize;
    for candidate in candidates {
        if used_singletons.contains(&candidate.singleton_node) {
            singleton_reuse_rejections += 1;
            continue;
        }
        if occupied.contains(&(candidate.track_root, candidate.target_image)) {
            duplicate_image_rejections += 1;
            continue;
        }
        let count = extension_counts.entry(candidate.track_root).or_default();
        if *count >= max_extensions_per_track {
            track_cap_rejections += 1;
            continue;
        }
        *count += 1;
        used_singletons.insert(candidate.singleton_node);
        occupied.insert((candidate.track_root, candidate.target_image));
        selected[candidate.pair_index].push(candidate.match_index);
    }
    let mut admitted = Vec::new();
    let mut admitted_matches = 0usize;
    for (pair, selected) in augmented.pairs.iter().zip(&mut selected) {
        if selected.is_empty() {
            continue;
        }
        selected.sort_unstable();
        admitted_matches += selected.len();
        admitted.push(retain_pair_matches(pair, selected));
    }
    let admitted_pairs = admitted.len();
    let deferred = base.pairs.split_off(structure_pair_prefix);
    base.pairs.extend(admitted);
    base.pairs.extend(deferred);
    let output = merge_owned(vec![base])?;
    let deferred_pairs = output.pairs.len() - structure_pair_prefix - admitted_pairs;
    Ok((
        output,
        FrozenExtensionStats {
            structure_pairs: structure_pair_prefix + admitted_pairs,
            deferred_pairs,
            candidate_matches,
            admitted_pairs,
            admitted_matches,
            duplicate_image_rejections,
            track_cap_rejections,
            singleton_reuse_rejections,
        },
    ))
}

fn frame_pair(pair: &PairRecord, image_frames: &[usize]) -> Result<(usize, usize), String> {
    let left =
        usize::try_from(pair.image_i).map_err(|_| "pair image_i does not fit usize".to_owned())?;
    let right =
        usize::try_from(pair.image_j).map_err(|_| "pair image_j does not fit usize".to_owned())?;
    let left = *image_frames
        .get(left)
        .ok_or_else(|| "pair image_i is outside the image manifest".to_owned())?;
    let right = *image_frames
        .get(right)
        .ok_or_else(|| "pair image_j is outside the image manifest".to_owned())?;
    Ok((left, right))
}

fn append_prefix_supplements(
    base: &mut PairRecord,
    augmented: &PairRecord,
    base_feature_counts: &[u64],
) -> Result<usize, String> {
    if (base.image_i, base.image_j) != (augmented.image_i, augmented.image_j) {
        return Err("base and augmented duplicate pairs use different orientations".into());
    }
    let left = usize::try_from(base.image_i).map_err(|_| "image_i does not fit usize")?;
    let right = usize::try_from(base.image_j).map_err(|_| "image_j does not fit usize")?;
    let left_prefix = *base_feature_counts
        .get(left)
        .ok_or("image_i is outside base feature counts")?;
    let right_prefix = *base_feature_counts
        .get(right)
        .ok_or("image_j is outside base feature counts")?;
    let is_supplement = |&(keypoint_i, keypoint_j): &(u64, u64)| {
        keypoint_i >= left_prefix || keypoint_j >= right_prefix
    };

    let mut accepted = base.matches.iter().copied().collect::<HashSet<_>>();
    let mut raw_index = base
        .raw_matches
        .iter()
        .copied()
        .enumerate()
        .map(|(index, pair)| (pair, index as u64))
        .collect::<HashMap<_, _>>();
    let mut appended = 0usize;
    for (&source_index, &matched) in augmented
        .accepted_inlier_indices
        .iter()
        .zip(&augmented.matches)
    {
        if !is_supplement(&matched) || !accepted.insert(matched) {
            continue;
        }
        let source_index = usize::try_from(source_index)
            .map_err(|_| "augmented accepted raw index does not fit usize")?;
        if augmented.raw_matches.get(source_index) != Some(&matched) {
            return Err("augmented accepted match disagrees with its raw index".into());
        }
        let index = if let Some(&index) = raw_index.get(&matched) {
            index
        } else {
            let index = base.raw_matches.len() as u64;
            base.raw_matches.push(matched);
            raw_index.insert(matched, index);
            index
        };
        base.accepted_inlier_indices.push(index);
        base.matches.push(matched);
        appended += 1;
    }

    if let Some(source_essential) = augmented.essential_matches.as_ref() {
        let destination = base.essential_matches.get_or_insert_with(Vec::new);
        let mut present = destination.iter().copied().collect::<HashSet<_>>();
        for (&source_index, &matched) in augmented
            .essential_inlier_indices
            .iter()
            .zip(source_essential)
        {
            if !is_supplement(&matched) || !present.insert(matched) {
                continue;
            }
            let source_index = usize::try_from(source_index)
                .map_err(|_| "augmented essential raw index does not fit usize")?;
            if augmented.raw_matches.get(source_index) != Some(&matched) {
                return Err("augmented essential match disagrees with its raw index".into());
            }
            let index = if let Some(&index) = raw_index.get(&matched) {
                index
            } else {
                let index = base.raw_matches.len() as u64;
                base.raw_matches.push(matched);
                raw_index.insert(matched, index);
                index
            };
            base.essential_inlier_indices.push(index);
            destination.push(matched);
        }
    }
    base.raw_match_count = base.raw_matches.len() as u64;
    base.e_inlier_count = base
        .essential_matches
        .as_ref()
        .map_or(base.e_inlier_count, |matches| matches.len() as u64);
    Ok(appended)
}

fn admit_bridges(
    mut base: Snapshot,
    mut augmented: Snapshot,
    image_frames: &[usize],
    frame_count: usize,
    include_prefix_supplements: bool,
    registered_frames: Option<&[bool]>,
    include_all_new_pairs: bool,
) -> Result<(Snapshot, usize, usize, u64, usize, u64, usize, u64), String> {
    let mut components = DisjointSet::new(frame_count);
    let mut base_keys = HashMap::with_capacity(base.pairs.len());
    for (pair_index, pair) in base.pairs.iter().enumerate() {
        let (left, right) = frame_pair(pair, image_frames)?;
        components.union(left, right);
        let key = (
            pair.image_i.min(pair.image_j),
            pair.image_i.max(pair.image_j),
        );
        if base_keys.insert(key, pair_index).is_some() {
            return Err(format!("base snapshot repeats pair ({},{})", key.0, key.1));
        }
    }
    let mut roots = HashSet::new();
    for frame in 0..frame_count {
        roots.insert(components.find(frame));
    }
    let base_component_count = roots.len();

    let mut bridges = Vec::new();
    let mut bridge_matches = 0u64;
    let mut supplemented_pairs = 0usize;
    let mut supplemented_matches = 0u64;
    let mut repair_pairs = 0usize;
    let mut repair_matches = 0u64;
    for pair in augmented.pairs.drain(..) {
        let key = (
            pair.image_i.min(pair.image_j),
            pair.image_i.max(pair.image_j),
        );
        if let Some(&base_index) = base_keys.get(&key) {
            if include_prefix_supplements {
                let appended = append_prefix_supplements(
                    &mut base.pairs[base_index],
                    &pair,
                    &base.feature_counts,
                )?;
                if appended > 0 {
                    supplemented_pairs += 1;
                    supplemented_matches += appended as u64;
                }
            }
            continue;
        }
        let (left, right) = frame_pair(&pair, image_frames)?;
        let crosses_base_components = components.find(left) != components.find(right);
        let repairs_unregistered_frame = !crosses_base_components
            && registered_frames.is_some_and(|registered| !registered[left] || !registered[right]);
        if include_all_new_pairs || crosses_base_components || repairs_unregistered_frame {
            if repairs_unregistered_frame {
                repair_pairs += 1;
                repair_matches += pair.matches.len() as u64;
            }
            bridge_matches += pair.matches.len() as u64;
            bridges.push(pair);
        }
    }
    let bridge_count = bridges.len();
    augmented.pairs = base.pairs;
    augmented.pairs.extend(bridges);
    let output = merge_owned(vec![augmented])?;
    Ok((
        output,
        base_component_count,
        bridge_count,
        bridge_matches,
        supplemented_pairs,
        supplemented_matches,
        repair_pairs,
        repair_matches,
    ))
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        usage();
    }
    let mut base_path = None;
    let mut augmented_path = None;
    let mut base_features_dir = None;
    let mut rig_manifest = None;
    let mut output_path = None;
    let mut feature_suffix = "_features.txt".to_owned();
    let mut include_prefix_supplements = false;
    let mut include_all_new_pairs = false;
    let mut frozen_track_extensions = false;
    let mut structure_pair_prefix = None;
    let mut frozen_track_extension_cap = 1usize;
    let mut repair_registered_components = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--base-snapshot" => base_path = Some(next_path(&args, &mut index, "--base-snapshot")?),
            "--augmented-snapshot" => {
                augmented_path = Some(next_path(&args, &mut index, "--augmented-snapshot")?)
            }
            "--base-features-dir" => {
                base_features_dir = Some(next_path(&args, &mut index, "--base-features-dir")?)
            }
            "--rig-manifest" => {
                rig_manifest = Some(next_path(&args, &mut index, "--rig-manifest")?)
            }
            "--output" => output_path = Some(next_path(&args, &mut index, "--output")?),
            "--feature-suffix" => {
                feature_suffix = args
                    .get(index + 1)
                    .ok_or_else(|| "--feature-suffix requires VALUE".to_owned())?
                    .clone();
                index += 1;
                if feature_suffix.trim().is_empty() {
                    return Err("--feature-suffix requires a non-empty VALUE".into());
                }
            }
            "--include-prefix-supplements" => include_prefix_supplements = true,
            "--include-all-new-pairs" => include_all_new_pairs = true,
            "--frozen-track-extensions" => frozen_track_extensions = true,
            "--structure-pair-prefix" => {
                structure_pair_prefix = Some(
                    args.get(index + 1)
                        .ok_or_else(|| "--structure-pair-prefix requires COUNT".to_owned())?
                        .parse::<usize>()
                        .map_err(|error| {
                            format!("--structure-pair-prefix requires COUNT: {error}")
                        })?,
                );
                index += 1;
            }
            "--frozen-track-extension-cap" => {
                frozen_track_extension_cap = args
                    .get(index + 1)
                    .ok_or_else(|| "--frozen-track-extension-cap requires COUNT".to_owned())?
                    .parse::<usize>()
                    .map_err(|error| {
                        format!("--frozen-track-extension-cap requires COUNT: {error}")
                    })?;
                index += 1;
            }
            "--repair-registered-components" => {
                repair_registered_components = Some(next_path(
                    &args,
                    &mut index,
                    "--repair-registered-components",
                )?)
            }
            "-h" | "--help" => usage(),
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 1;
    }
    let base_path = base_path.ok_or_else(|| "--base-snapshot is required".to_owned())?;
    let augmented_path =
        augmented_path.ok_or_else(|| "--augmented-snapshot is required".to_owned())?;
    let base_features_dir =
        base_features_dir.ok_or_else(|| "--base-features-dir is required".to_owned())?;
    let rig_manifest = rig_manifest.ok_or_else(|| "--rig-manifest is required".to_owned())?;
    let output_path = output_path.ok_or_else(|| "--output is required".to_owned())?;
    if frozen_track_extensions
        && (include_all_new_pairs
            || include_prefix_supplements
            || repair_registered_components.is_some())
    {
        return Err(
            "--frozen-track-extensions cannot be combined with bridge/supplement admission modes"
                .into(),
        );
    }
    if frozen_track_extensions != structure_pair_prefix.is_some() {
        return Err(
            "--frozen-track-extensions and --structure-pair-prefix must be specified together"
                .into(),
        );
    }

    let base = read(&base_path)
        .map_err(|error| format!("invalid base snapshot {}: {error}", base_path.display()))?;
    let augmented = read(&augmented_path).map_err(|error| {
        format!(
            "invalid augmented snapshot {}: {error}",
            augmented_path.display()
        )
    })?;
    validate_envelopes(&base, &augmented)?;
    for (index, image_name) in base.image_names.iter().enumerate() {
        let path = feature_path(&base_features_dir, image_name, &feature_suffix)?;
        let actual = feature_count(&path)?;
        let expected = base.feature_counts[index];
        if actual != expected {
            return Err(format!(
                "base feature prefix count differs for {image_name:?}: snapshot={expected}, file={actual}"
            ));
        }
    }
    let (image_frames, frame_count) = parse_rig_frames(&rig_manifest, &base.image_names)?;
    if let Some(structure_pair_prefix) = structure_pair_prefix {
        let (output, stats) = admit_frozen_track_extensions(
            base,
            &augmented,
            &image_frames,
            structure_pair_prefix,
            frozen_track_extension_cap,
        )?;
        write_atomic(&output_path, &output)?;
        println!(
            "verified frozen-track extension admission: frames={frame_count} \
             structure_pairs={} deferred_pairs={} candidate_matches={} admitted_pairs={} \
             admitted_matches={} duplicate_image_rejections={} track_cap_rejections={} \
             singleton_reuse_rejections={} output_pairs={} output_matches={} -> {}",
            stats.structure_pairs,
            stats.deferred_pairs,
            stats.candidate_matches,
            stats.admitted_pairs,
            stats.admitted_matches,
            stats.duplicate_image_rejections,
            stats.track_cap_rejections,
            stats.singleton_reuse_rejections,
            output.pairs.len(),
            output.accepted_match_count,
            output_path.display(),
        );
        return Ok(());
    }
    let registered_frames = repair_registered_components
        .as_deref()
        .map(|path| parse_registered_frames(path, &base.image_names, &image_frames, frame_count))
        .transpose()?;
    let base_pairs = base.pairs.len();
    let base_matches = base.accepted_match_count;
    let (
        output,
        base_components,
        bridges,
        bridge_matches,
        supplemented_pairs,
        supplemented_matches,
        repair_pairs,
        repair_matches,
    ) = admit_bridges(
        base,
        augmented,
        &image_frames,
        frame_count,
        include_prefix_supplements,
        registered_frames.as_deref(),
        include_all_new_pairs,
    )?;
    write_atomic(&output_path, &output)?;
    println!(
        "verified bridge admission: frames={frame_count} base_components={base_components} \
         base_pairs={base_pairs} base_matches={base_matches} admitted_pairs={bridges} \
         admitted_matches={bridge_matches} supplemented_pairs={supplemented_pairs} \
         supplemented_matches={supplemented_matches} repair_pairs={repair_pairs} \
         repair_matches={repair_matches} output_pairs={} output_matches={} -> {}",
        output.pairs.len(),
        output.accepted_match_count,
        output_path.display(),
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(left: u64, right: u64, matches: usize) -> PairRecord {
        PairRecord {
            image_i: left,
            image_j: right,
            raw_match_count: matches as u64,
            raw_matches: (0..matches)
                .map(|index| (index as u64, index as u64))
                .collect(),
            accepted_inlier_indices: (0..matches as u64).collect(),
            essential_inlier_indices: Vec::new(),
            matches: (0..matches)
                .map(|index| (index as u64, index as u64))
                .collect(),
            essential_matches: None,
            config: 0,
            calibrated: true,
            e_inlier_count: matches as u64,
            f_inlier_count: matches as u64,
            h_inlier_count: 0,
            essential_matrix_bits: None,
            fundamental_matrix_bits: None,
            homography_matrix_bits: None,
            relative_rotation_bits: None,
            relative_translation_bits: None,
        }
    }

    fn snapshot(pairs: Vec<PairRecord>) -> Snapshot {
        let accepted_match_count = pairs.iter().map(|pair| pair.matches.len() as u64).sum();
        Snapshot {
            schema_version: 1,
            image_names: (0..4).map(|index| format!("image-{index}.png")).collect(),
            image_manifest_hash: 1,
            feature_manifest_hash: 2,
            feature_counts: vec![8; 4],
            width: 10,
            height: 10,
            intrinsics_bits: [0; 4],
            effective_config_hash: 3,
            effective_config: "effective".into(),
            verifier_config_hash: 4,
            verifier_config: "verifier".into(),
            pair_order_hash: 0,
            unordered_edge_hash: 0,
            accepted_match_count,
            pairs,
        }
    }

    #[test]
    fn inserts_only_frozen_one_sided_extensions_before_deferred_tail() {
        let base = snapshot(vec![pair(0, 1, 2), pair(2, 3, 2)]);
        let augmented = snapshot(vec![pair(0, 2, 2), pair(1, 3, 2), pair(0, 1, 4)]);
        let (output, stats) =
            admit_frozen_track_extensions(base, &augmented, &[0, 1, 2, 3], 1, 1).unwrap();

        assert_eq!(stats.structure_pairs, 2);
        assert_eq!(stats.deferred_pairs, 1);
        assert_eq!(stats.candidate_matches, 4);
        assert_eq!(stats.admitted_pairs, 1);
        assert_eq!(stats.admitted_matches, 2);
        assert_eq!(stats.track_cap_rejections, 2);
        assert_eq!(
            output
                .pairs
                .iter()
                .map(|pair| (pair.image_i, pair.image_j))
                .collect::<Vec<_>>(),
            vec![(0, 1), (0, 2), (2, 3)]
        );
        assert_eq!(output.pairs[1].matches.len(), 2);
    }

    #[test]
    fn keeps_base_order_and_only_static_cross_component_pairs() {
        let base = snapshot(vec![pair(0, 1, 2), pair(2, 3, 3)]);
        let augmented = snapshot(vec![
            pair(0, 1, 7),
            pair(0, 2, 5),
            pair(1, 3, 6),
            pair(2, 3, 8),
        ]);
        let (
            output,
            components,
            bridges,
            bridge_matches,
            supplemented_pairs,
            supplemented_matches,
            repair_pairs,
            repair_matches,
        ) = admit_bridges(base, augmented, &[0, 1, 2, 3], 4, false, None, false).unwrap();
        assert_eq!(components, 2);
        assert_eq!(bridges, 2);
        assert_eq!(bridge_matches, 11);
        assert_eq!(supplemented_pairs, 0);
        assert_eq!(supplemented_matches, 0);
        assert_eq!(repair_pairs, 0);
        assert_eq!(repair_matches, 0);
        assert_eq!(output.pairs.len(), 4);
        assert_eq!(output.pairs[0].matches.len(), 2);
        assert_eq!(output.pairs[1].matches.len(), 3);
        assert_eq!((output.pairs[2].image_i, output.pairs[2].image_j), (0, 2));
        assert_eq!((output.pairs[3].image_i, output.pairs[3].image_j), (1, 3));
        assert_eq!(output.accepted_match_count, 16);
    }

    #[test]
    fn appends_only_matches_outside_the_frozen_feature_prefix() {
        let mut base = pair(0, 1, 2);
        let augmented = pair(0, 1, 4);
        let mut augmented = augmented;
        augmented.matches = vec![(0, 0), (1, 1), (8, 2), (3, 9)];
        augmented.raw_matches = augmented.matches.clone();
        augmented.raw_match_count = 4;
        augmented.accepted_inlier_indices = (0..4).collect();

        let appended = append_prefix_supplements(&mut base, &augmented, &[8, 8]).unwrap();

        assert_eq!(appended, 2);
        assert_eq!(base.matches, vec![(0, 0), (1, 1), (8, 2), (3, 9)]);
        assert_eq!(base.accepted_inlier_indices.len(), base.matches.len());
        assert_eq!(base.raw_match_count, base.raw_matches.len() as u64);
    }

    #[test]
    fn admits_internal_new_pair_only_when_it_repairs_an_unregistered_frame() {
        let base = snapshot(vec![pair(0, 1, 2), pair(1, 2, 3)]);
        let augmented = snapshot(vec![
            pair(0, 1, 2),
            pair(1, 2, 3),
            pair(0, 2, 4),
            pair(2, 3, 5),
        ]);
        let registered = [false, true, true, true];

        let (output, components, bridges, _, _, _, repair_pairs, repair_matches) = admit_bridges(
            base,
            augmented,
            &[0, 1, 2, 3],
            4,
            false,
            Some(&registered),
            false,
        )
        .unwrap();

        assert_eq!(components, 2);
        assert_eq!(bridges, 2);
        assert_eq!(repair_pairs, 1);
        assert_eq!(repair_matches, 4);
        assert_eq!(output.pairs.len(), 4);
        assert_eq!((output.pairs[2].image_i, output.pairs[2].image_j), (0, 2));
        assert_eq!((output.pairs[3].image_i, output.pairs[3].image_j), (2, 3));
    }
}
