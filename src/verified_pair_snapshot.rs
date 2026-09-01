//! Lossless, versioned snapshots for an already verified pair stream.
//!
//! This module deliberately uses a small dependency-free binary codec.  The
//! codec stores integer and floating-point values as their exact little-endian
//! bit patterns, keeps vectors in source order, and checks the payload with a
//! stable FNV-1a checksum.  It is an interchange format for the example
//! executable, not a replacement for the historical human-readable imports.

use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;

const MAGIC: &[u8] = b"VISLOC-VERIFIED-PAIR-SNAPSHOT\0";
const MAX_VECTOR_ITEMS: usize = 50_000_000;

/// One accepted pair and the verifier information needed to replay it.
///
/// The vectors intentionally retain the order produced by the verifier.  The
/// accepted/essential inlier index fields refer to the raw matcher vector used
/// during the original verification pass; matches and essential_matches are
/// the feature-index subsets consumed by the mapper.
#[derive(Debug, Clone, PartialEq)]
pub struct PairRecord {
    pub image_i: u64,
    pub image_j: u64,
    pub raw_match_count: u64,
    /// The exact correspondence stream presented to the final verifier.  It
    /// is retained even though the mapper consumes only `matches`, so an
    /// imported snapshot can be audited without rerunning matching.
    pub raw_matches: Vec<(u64, u64)>,
    pub accepted_inlier_indices: Vec<u64>,
    pub essential_inlier_indices: Vec<u64>,
    pub matches: Vec<(u64, u64)>,
    pub essential_matches: Option<Vec<(u64, u64)>>,
    pub config: u8,
    pub calibrated: bool,
    pub e_inlier_count: u64,
    pub f_inlier_count: u64,
    pub h_inlier_count: u64,
    pub essential_matrix_bits: Option<[u64; 9]>,
    pub fundamental_matrix_bits: Option<[u64; 9]>,
    pub homography_matrix_bits: Option<[u64; 9]>,
    pub relative_rotation_bits: Option<[u64; 9]>,
    pub relative_translation_bits: Option<[u64; 3]>,
}

/// Complete snapshot envelope. Manifest/configuration hashes are checked on
/// import. The embedded configuration is deliberately path-independent so
/// the same inputs produce identical bytes in different resumable run roots;
/// full command paths remain in the runner log and shard index.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    pub schema_version: u32,
    pub image_names: Vec<String>,
    pub image_manifest_hash: u64,
    pub feature_manifest_hash: u64,
    pub feature_counts: Vec<u64>,
    pub width: u64,
    pub height: u64,
    pub intrinsics_bits: [u64; 4],
    pub effective_config_hash: u64,
    pub effective_config: String,
    pub verifier_config_hash: u64,
    pub verifier_config: String,
    pub pair_order_hash: u64,
    pub unordered_edge_hash: u64,
    pub accepted_match_count: u64,
    pub pairs: Vec<PairRecord>,
}

/// Merge complete, hash-validated shard payloads without changing the pair
/// order inside any shard.
///
/// A snapshot shard carries the complete image/feature/configuration envelope
/// even when it contains only a subset of candidate pairs.  This helper is
/// intentionally strict about that envelope and rejects overlapping pairs so
/// a resumed run cannot silently double-count a shard.  The returned snapshot
/// uses the argument order as its deterministic pair order and recomputes the
/// two stream-integrity hashes and accepted-match count.
pub fn merge(snapshots: &[Snapshot]) -> Result<Snapshot, String> {
    merge_owned(snapshots.to_vec())
}

/// Merge owned snapshot shards while moving pair payloads into the result.
///
/// This is the bounded-memory path for large shard sets: unlike [`merge`], it
/// does not clone every raw/inlier/mapping vector while the decoded inputs are
/// still resident. Envelope validation and output ordering are identical.
pub fn merge_owned(mut snapshots: Vec<Snapshot>) -> Result<Snapshot, String> {
    let first = snapshots
        .first()
        .ok_or_else(|| "cannot merge an empty verified-pair snapshot list".to_owned())?;
    for (index, snapshot) in snapshots.iter().enumerate() {
        if snapshot.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "snapshot shard {index} has unsupported schema {} (expected {SCHEMA_VERSION})",
                snapshot.schema_version
            ));
        }
        if snapshot.image_names != first.image_names {
            return Err(format!(
                "snapshot shard {index} image manifest differs from shard 0"
            ));
        }
        if snapshot.feature_manifest_hash != first.feature_manifest_hash
            || snapshot.feature_counts != first.feature_counts
            || snapshot.image_manifest_hash != first.image_manifest_hash
        {
            return Err(format!(
                "snapshot shard {index} feature/image manifest differs from shard 0"
            ));
        }
        if snapshot.width != first.width
            || snapshot.height != first.height
            || snapshot.intrinsics_bits != first.intrinsics_bits
        {
            return Err(format!(
                "snapshot shard {index} camera envelope differs from shard 0"
            ));
        }
        // Older shards may carry a complete CLI diagnostic in
        // `effective_config`, including candidate/output paths. Stream
        // compatibility is defined by the path-independent verifier config;
        // the merged envelope below canonicalizes old and new shards alike.
        if snapshot.verifier_config_hash != first.verifier_config_hash
            || snapshot.verifier_config != first.verifier_config
        {
            return Err(format!(
                "snapshot shard {index} matcher/verifier configuration differs from shard 0"
            ));
        }
    }

    let image_names = first.image_names.clone();
    let image_manifest_hash = first.image_manifest_hash;
    let feature_manifest_hash = first.feature_manifest_hash;
    let feature_counts = first.feature_counts.clone();
    let width = first.width;
    let height = first.height;
    let intrinsics_bits = first.intrinsics_bits;
    let verifier_config_hash = first.verifier_config_hash;
    let verifier_config = first.verifier_config.clone();
    let image_count = image_names.len();
    let pair_count = snapshots.iter().map(|snapshot| snapshot.pairs.len()).sum();
    let mut pairs = Vec::with_capacity(pair_count);
    let mut seen = std::collections::HashSet::new();
    for snapshot in &mut snapshots {
        for pair in &snapshot.pairs {
            let image_i = usize::try_from(pair.image_i)
                .map_err(|_| "snapshot pair image_i does not fit usize".to_owned())?;
            let image_j = usize::try_from(pair.image_j)
                .map_err(|_| "snapshot pair image_j does not fit usize".to_owned())?;
            if image_i >= image_count || image_j >= image_count || image_i == image_j {
                return Err(format!(
                    "snapshot pair has invalid image indices ({image_i},{image_j})"
                ));
            }
            let key = (image_i.min(image_j), image_i.max(image_j));
            if !seen.insert(key) {
                return Err(format!(
                    "snapshot shards overlap at image pair ({},{})",
                    key.0, key.1
                ));
            }
        }
        pairs.append(&mut snapshot.pairs);
    }

    let accepted_match_count = pairs
        .iter()
        .map(|pair| pair.matches.len() as u64)
        .sum::<u64>();
    let effective_config = format!("verified-pair-export-v1;{verifier_config}");
    let effective_config_hash = fnv1a64(effective_config.as_bytes());
    Ok(Snapshot {
        schema_version: SCHEMA_VERSION,
        image_names,
        image_manifest_hash,
        feature_manifest_hash,
        feature_counts,
        width,
        height,
        intrinsics_bits,
        effective_config_hash,
        effective_config,
        verifier_config_hash,
        verifier_config,
        pair_order_hash: ordered_pair_hash(&pairs),
        unordered_edge_hash: unordered_edge_hash(&pairs),
        accepted_match_count,
        pairs,
    })
}

/// Write a complete snapshot through a same-directory temporary file and an
/// atomic rename.  A caller may safely retry after an interrupted write; only
/// the final path is considered complete.
pub fn write_atomic(path: &Path, snapshot: &Snapshot) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("create snapshot directory {}: {error}", parent.display()))?;
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("snapshot path has no valid filename: {}", path.display()))?;
    let temporary = parent.join(format!(".{filename}.tmp-{}", std::process::id()));
    write(&temporary, snapshot)?;
    std::fs::rename(&temporary, path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        format!("install snapshot {}: {error}", path.display())
    })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hash_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn ordered_pair_hash(pairs: &[PairRecord]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    hash = hash_u64(hash, pairs.len() as u64);
    for pair in pairs {
        hash = hash_u64(hash, pair.image_i);
        hash = hash_u64(hash, pair.image_j);
        hash = hash_u64(hash, u64::from(pair.config));
        hash = hash_u64(hash, pair.matches.len() as u64);
        for &(left, right) in &pair.matches {
            hash = hash_u64(hash, left);
            hash = hash_u64(hash, right);
        }
        if let Some(matches) = &pair.essential_matches {
            hash = hash_u64(hash, 1);
            hash = hash_u64(hash, matches.len() as u64);
            for &(left, right) in matches {
                hash = hash_u64(hash, left);
                hash = hash_u64(hash, right);
            }
        } else {
            hash = hash_u64(hash, 0);
        }
        if let Some(matrix) = pair.essential_matrix_bits {
            hash = hash_u64(hash, 1);
            for value in matrix {
                hash = hash_u64(hash, value);
            }
        } else {
            hash = hash_u64(hash, 0);
        }
    }
    hash
}

fn unordered_edge_hash(pairs: &[PairRecord]) -> u64 {
    let mut pair_order = pairs
        .iter()
        .enumerate()
        .map(|(index, pair)| {
            (
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
                index,
            )
        })
        .collect::<Vec<_>>();
    pair_order.sort_unstable_by_key(|&(image_i, image_j, _)| (image_i, image_j));
    let mut hash = 0xcbf29ce484222325u64;
    let mut group_start = 0;
    while group_start < pair_order.len() {
        let (image_i, image_j, _) = pair_order[group_start];
        let mut group_end = group_start + 1;
        while group_end < pair_order.len()
            && pair_order[group_end].0 == image_i
            && pair_order[group_end].1 == image_j
        {
            group_end += 1;
        }
        let group_match_count = pair_order[group_start..group_end]
            .iter()
            .map(|&(_, _, index)| pairs[index].matches.len())
            .sum();
        let mut matches = Vec::with_capacity(group_match_count);
        for &(_, _, index) in &pair_order[group_start..group_end] {
            let pair = &pairs[index];
            let swapped = pair.image_i > pair.image_j;
            matches.extend(pair.matches.iter().map(|&(left, right)| {
                if swapped {
                    (right, left)
                } else {
                    (left, right)
                }
            }));
        }
        matches.sort_unstable();
        for (left, right) in matches {
            hash = hash_u64(hash, image_i);
            hash = hash_u64(hash, image_j);
            hash = hash_u64(hash, left);
            hash = hash_u64(hash, right);
        }
        group_start = group_end;
    }
    hash
}

fn put_u8<W: Write>(out: &mut W, value: u8) -> Result<(), String> {
    out.write_all(&[value]).map_err(|error| error.to_string())
}

fn put_u32<W: Write>(out: &mut W, value: u32) -> Result<(), String> {
    out.write_all(&value.to_le_bytes())
        .map_err(|error| error.to_string())
}

fn put_u64<W: Write>(out: &mut W, value: u64) -> Result<(), String> {
    out.write_all(&value.to_le_bytes())
        .map_err(|error| error.to_string())
}

fn put_bool<W: Write>(out: &mut W, value: bool) -> Result<(), String> {
    put_u8(out, u8::from(value))
}

fn put_bytes<W: Write>(out: &mut W, bytes: &[u8]) -> Result<(), String> {
    put_u64(out, bytes.len() as u64)?;
    out.write_all(bytes).map_err(|error| error.to_string())
}

fn put_string<W: Write>(out: &mut W, value: &str) -> Result<(), String> {
    put_bytes(out, value.as_bytes())
}

fn put_u64_vec<W: Write>(out: &mut W, values: &[u64]) -> Result<(), String> {
    put_u64(out, values.len() as u64)?;
    for value in values {
        put_u64(out, *value)?;
    }
    Ok(())
}

fn put_match_vec<W: Write>(out: &mut W, values: &[(u64, u64)]) -> Result<(), String> {
    put_u64(out, values.len() as u64)?;
    for &(left, right) in values {
        put_u64(out, left)?;
        put_u64(out, right)?;
    }
    Ok(())
}

fn put_matrix9<W: Write>(out: &mut W, value: &Option<[u64; 9]>) -> Result<(), String> {
    put_bool(out, value.is_some())?;
    if let Some(values) = value {
        for value in values {
            put_u64(out, *value)?;
        }
    }
    Ok(())
}

fn put_matrix3<W: Write>(out: &mut W, value: &Option<[u64; 3]>) -> Result<(), String> {
    put_bool(out, value.is_some())?;
    if let Some(values) = value {
        for value in values {
            put_u64(out, *value)?;
        }
    }
    Ok(())
}

fn encode_payload<W: Write>(snapshot: &Snapshot, out: &mut W) -> Result<(), String> {
    if snapshot.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported verified-pair snapshot schema {} (expected {})",
            snapshot.schema_version, SCHEMA_VERSION
        ));
    }
    if snapshot.image_names.len() != snapshot.feature_counts.len() {
        return Err(format!(
            "snapshot image/feature manifest length mismatch: {} names vs {} feature counts",
            snapshot.image_names.len(),
            snapshot.feature_counts.len()
        ));
    }
    put_u32(out, snapshot.schema_version)?;
    put_u64(out, snapshot.image_names.len() as u64)?;
    for name in &snapshot.image_names {
        put_string(out, name)?;
    }
    put_u64(out, snapshot.image_manifest_hash)?;
    put_u64(out, snapshot.feature_manifest_hash)?;
    put_u64_vec(out, &snapshot.feature_counts)?;
    put_u64(out, snapshot.width)?;
    put_u64(out, snapshot.height)?;
    for value in snapshot.intrinsics_bits {
        put_u64(out, value)?;
    }
    put_u64(out, snapshot.effective_config_hash)?;
    put_string(out, &snapshot.effective_config)?;
    put_u64(out, snapshot.verifier_config_hash)?;
    put_string(out, &snapshot.verifier_config)?;
    put_u64(out, snapshot.pair_order_hash)?;
    put_u64(out, snapshot.unordered_edge_hash)?;
    put_u64(out, snapshot.accepted_match_count)?;
    put_u64(out, snapshot.pairs.len() as u64)?;
    for pair in &snapshot.pairs {
        if pair.raw_match_count != pair.raw_matches.len() as u64 {
            return Err(format!(
                "snapshot raw match count {} does not match raw stream length {} for pair ({},{})",
                pair.raw_match_count,
                pair.raw_matches.len(),
                pair.image_i,
                pair.image_j,
            ));
        }
        put_u64(out, pair.image_i)?;
        put_u64(out, pair.image_j)?;
        put_u64(out, pair.raw_match_count)?;
        put_match_vec(out, &pair.raw_matches)?;
        put_u64_vec(out, &pair.accepted_inlier_indices)?;
        put_u64_vec(out, &pair.essential_inlier_indices)?;
        put_match_vec(out, &pair.matches)?;
        put_bool(out, pair.essential_matches.is_some())?;
        if let Some(matches) = &pair.essential_matches {
            put_match_vec(out, matches)?;
        }
        put_u8(out, pair.config)?;
        put_bool(out, pair.calibrated)?;
        put_u64(out, pair.e_inlier_count)?;
        put_u64(out, pair.f_inlier_count)?;
        put_u64(out, pair.h_inlier_count)?;
        put_matrix9(out, &pair.essential_matrix_bits)?;
        put_matrix9(out, &pair.fundamental_matrix_bits)?;
        put_matrix9(out, &pair.homography_matrix_bits)?;
        put_matrix9(out, &pair.relative_rotation_bits)?;
        put_matrix3(out, &pair.relative_translation_bits)?;
    }
    Ok(())
}

struct Reader<R> {
    inner: R,
    offset: u64,
    payload_len: u64,
}

impl<R: Read> Reader<R> {
    fn new(inner: R, payload_len: u64) -> Self {
        Self {
            inner,
            offset: 0,
            payload_len,
        }
    }

    fn read_exact(&mut self, bytes: &mut [u8]) -> Result<(), String> {
        let count = bytes.len() as u64;
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| "snapshot offset overflow".to_owned())?;
        if end > self.payload_len {
            return Err(format!(
                "truncated verified-pair snapshot at byte {} (need {} bytes)",
                self.offset, count
            ));
        }
        self.inner.read_exact(bytes).map_err(|error| {
            format!(
                "truncated verified-pair snapshot at byte {} (need {count} bytes): {error}",
                self.offset
            )
        })?;
        self.offset = end;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, String> {
        let mut bytes = [0; 1];
        self.read_exact(&mut bytes)?;
        Ok(bytes[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn bool(&mut self) -> Result<bool, String> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(format!("invalid snapshot boolean {value}")),
        }
    }

    fn len(&mut self, label: &str) -> Result<usize, String> {
        let value = self.u64()?;
        let length = usize::try_from(value)
            .map_err(|_| format!("snapshot {label} length {value} does not fit usize"))?;
        if length > MAX_VECTOR_ITEMS {
            return Err(format!(
                "snapshot {label} length {length} exceeds safety limit {MAX_VECTOR_ITEMS}"
            ));
        }
        Ok(length)
    }

    fn bytes(&mut self, label: &str) -> Result<Vec<u8>, String> {
        let length = self.len(label)?;
        let mut bytes = vec![0; length];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn string(&mut self, label: &str) -> Result<String, String> {
        String::from_utf8(self.bytes(label)?)
            .map_err(|error| format!("snapshot {label} is not valid UTF-8: {error}"))
    }

    fn u64_vec(&mut self, label: &str) -> Result<Vec<u64>, String> {
        let length = self.len(label)?;
        (0..length).map(|_| self.u64()).collect()
    }

    fn match_vec(&mut self, label: &str) -> Result<Vec<(u64, u64)>, String> {
        let length = self.len(label)?;
        (0..length)
            .map(|_| Ok((self.u64()?, self.u64()?)))
            .collect()
    }

    fn matrix9(&mut self, label: &str) -> Result<Option<[u64; 9]>, String> {
        if !self.bool()? {
            return Ok(None);
        }
        let mut values = [0; 9];
        for value in &mut values {
            *value = self.u64()?;
        }
        let _ = label;
        Ok(Some(values))
    }

    fn matrix3(&mut self, label: &str) -> Result<Option<[u64; 3]>, String> {
        if !self.bool()? {
            return Ok(None);
        }
        let mut values = [0; 3];
        for value in &mut values {
            *value = self.u64()?;
        }
        let _ = label;
        Ok(Some(values))
    }

    fn done(&self) -> bool {
        self.offset == self.payload_len
    }
}

fn decode_payload<R: Read>(
    input: R,
    payload_len: u64,
    retain_audit_streams: bool,
) -> Result<Snapshot, String> {
    let mut reader = Reader::new(input, payload_len);
    let schema_version = reader.u32()?;
    if schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported verified-pair snapshot schema {schema_version} (expected {SCHEMA_VERSION})"
        ));
    }
    let image_count = reader.len("image names")?;
    let image_names = (0..image_count)
        .map(|_| reader.string("image name"))
        .collect::<Result<Vec<_>, _>>()?;
    let image_manifest_hash = reader.u64()?;
    let feature_manifest_hash = reader.u64()?;
    let feature_counts = reader.u64_vec("feature counts")?;
    if feature_counts.len() != image_names.len() {
        return Err(format!(
            "snapshot feature counts {} do not match image names {}",
            feature_counts.len(),
            image_names.len()
        ));
    }
    let width = reader.u64()?;
    let height = reader.u64()?;
    let mut intrinsics_bits = [0; 4];
    for value in &mut intrinsics_bits {
        *value = reader.u64()?;
    }
    let effective_config_hash = reader.u64()?;
    let effective_config = reader.string("effective config")?;
    let verifier_config_hash = reader.u64()?;
    let verifier_config = reader.string("verifier config")?;
    let pair_order_hash = reader.u64()?;
    let unordered_edge_hash = reader.u64()?;
    let accepted_match_count = reader.u64()?;
    let pair_count = reader.len("pairs")?;
    let mut pairs = Vec::with_capacity(pair_count);
    for pair_number in 0..pair_count {
        let image_i = reader.u64()?;
        let image_j = reader.u64()?;
        let raw_match_count = reader.u64()?;
        let raw_matches = reader.match_vec("raw matches")?;
        if raw_match_count != raw_matches.len() as u64 {
            return Err(format!(
                "snapshot raw match count {raw_match_count} does not match raw stream length {}",
                raw_matches.len()
            ));
        }
        let accepted_inlier_indices = reader.u64_vec("accepted inlier indices")?;
        let essential_inlier_indices = reader.u64_vec("essential inlier indices")?;
        let matches = reader.match_vec("accepted matches")?;
        let essential_matches = if reader.bool()? {
            Some(reader.match_vec("essential matches")?)
        } else {
            None
        };
        if accepted_inlier_indices.len() != matches.len() {
            return Err(format!(
                "snapshot pair {pair_number} has {} accepted indices but {} accepted matches",
                accepted_inlier_indices.len(),
                matches.len()
            ));
        }
        for (position, (&raw_index, &accepted)) in
            accepted_inlier_indices.iter().zip(&matches).enumerate()
        {
            let raw_index = usize::try_from(raw_index).map_err(|_| {
                format!("snapshot pair {pair_number} accepted index does not fit usize")
            })?;
            if raw_matches.get(raw_index) != Some(&accepted) {
                return Err(format!(
                    "snapshot pair {pair_number} accepted match at position {position} disagrees with raw index {raw_index}"
                ));
            }
        }
        match &essential_matches {
            Some(essential) if essential_inlier_indices.len() == essential.len() => {
                for (position, (&raw_index, &accepted)) in
                    essential_inlier_indices.iter().zip(essential).enumerate()
                {
                    let raw_index = usize::try_from(raw_index).map_err(|_| {
                        format!("snapshot pair {pair_number} essential index does not fit usize")
                    })?;
                    if raw_matches.get(raw_index) != Some(&accepted) {
                        return Err(format!(
                            "snapshot pair {pair_number} essential match at position {position} disagrees with raw index {raw_index}"
                        ));
                    }
                }
            }
            Some(essential) => {
                return Err(format!(
                    "snapshot pair {pair_number} has {} essential indices but {} essential matches",
                    essential_inlier_indices.len(),
                    essential.len()
                ));
            }
            None if !essential_inlier_indices.is_empty() => {
                return Err(format!(
                    "snapshot pair {pair_number} has essential indices but no essential matches"
                ));
            }
            None => {}
        }
        let config = reader.u8()?;
        let calibrated = reader.bool()?;
        let e_inlier_count = reader.u64()?;
        let f_inlier_count = reader.u64()?;
        let h_inlier_count = reader.u64()?;
        let essential_matrix_bits = reader.matrix9("essential matrix")?;
        let fundamental_matrix_bits = reader.matrix9("fundamental matrix")?;
        let homography_matrix_bits = reader.matrix9("homography matrix")?;
        let relative_rotation_bits = reader.matrix9("relative rotation")?;
        let relative_translation_bits = reader.matrix3("relative translation")?;
        pairs.push(PairRecord {
            image_i,
            image_j,
            raw_match_count: if retain_audit_streams {
                raw_match_count
            } else {
                0
            },
            raw_matches: if retain_audit_streams {
                raw_matches
            } else {
                Vec::new()
            },
            accepted_inlier_indices: if retain_audit_streams {
                accepted_inlier_indices
            } else {
                Vec::new()
            },
            essential_inlier_indices: if retain_audit_streams {
                essential_inlier_indices
            } else {
                Vec::new()
            },
            matches,
            essential_matches,
            config,
            calibrated,
            e_inlier_count,
            f_inlier_count,
            h_inlier_count,
            essential_matrix_bits,
            fundamental_matrix_bits,
            homography_matrix_bits,
            relative_rotation_bits,
            relative_translation_bits,
        });
    }
    if !reader.done() {
        return Err(format!(
            "verified-pair snapshot has {} trailing bytes",
            reader.payload_len - reader.offset
        ));
    }
    Ok(Snapshot {
        schema_version,
        image_names,
        image_manifest_hash,
        feature_manifest_hash,
        feature_counts,
        width,
        height,
        intrinsics_bits,
        effective_config_hash,
        effective_config,
        verifier_config_hash,
        verifier_config,
        pair_order_hash,
        unordered_edge_hash,
        accepted_match_count,
        pairs,
    })
}

struct PayloadDigestWriter<W> {
    inner: W,
    checksum: u64,
    len: u64,
}

fn payload_len(snapshot: &Snapshot) -> Result<u64, String> {
    if snapshot.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported verified-pair snapshot schema {} (expected {})",
            snapshot.schema_version, SCHEMA_VERSION
        ));
    }
    if snapshot.image_names.len() != snapshot.feature_counts.len() {
        return Err(format!(
            "snapshot image/feature manifest length mismatch: {} names vs {} feature counts",
            snapshot.image_names.len(),
            snapshot.feature_counts.len()
        ));
    }
    let mut len = 4u64 + 8;
    let mut add = |value: u64| -> Result<(), String> {
        len = len
            .checked_add(value)
            .ok_or_else(|| "snapshot payload length overflow".to_owned())?;
        Ok(())
    };
    for name in &snapshot.image_names {
        add(8 + name.len() as u64)?;
    }
    add(8 + 8 + 8 + 8 * snapshot.feature_counts.len() as u64)?;
    add(8 + 8 + 8 * 4)?;
    add(8 + 8 + snapshot.effective_config.len() as u64)?;
    add(8 + 8 + snapshot.verifier_config.len() as u64)?;
    add(8 + 8 + 8 + 8)?;
    for pair in &snapshot.pairs {
        if pair.raw_match_count != pair.raw_matches.len() as u64 {
            return Err(format!(
                "snapshot raw match count {} does not match raw stream length {} for pair ({},{})",
                pair.raw_match_count,
                pair.raw_matches.len(),
                pair.image_i,
                pair.image_j,
            ));
        }
        add(8 * 3)?;
        add(8 + 16 * pair.raw_matches.len() as u64)?;
        add(8 + 8 * pair.accepted_inlier_indices.len() as u64)?;
        add(8 + 8 * pair.essential_inlier_indices.len() as u64)?;
        add(8 + 16 * pair.matches.len() as u64)?;
        add(1)?;
        if let Some(matches) = &pair.essential_matches {
            add(8 + 16 * matches.len() as u64)?;
        }
        add(1 + 1 + 8 * 3)?;
        for matrix in [
            &pair.essential_matrix_bits,
            &pair.fundamental_matrix_bits,
            &pair.homography_matrix_bits,
            &pair.relative_rotation_bits,
        ] {
            add(1 + if matrix.is_some() { 8 * 9 } else { 0 })?;
        }
        add(1 + if pair.relative_translation_bits.is_some() {
            8 * 3
        } else {
            0
        })?;
    }
    Ok(len)
}

impl<W> PayloadDigestWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            checksum: 0xcbf29ce484222325,
            len: 0,
        }
    }
}

impl<W: Write> Write for PayloadDigestWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        for byte in &bytes[..written] {
            self.checksum ^= u64::from(*byte);
            self.checksum = self.checksum.wrapping_mul(0x100000001b3);
        }
        self.len = self
            .len
            .checked_add(written as u64)
            .ok_or_else(|| std::io::Error::other("snapshot payload length overflow"))?;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Write a complete snapshot and append a checksum over its exact payload.
///
/// The payload length is computed from vector/string lengths, then the payload
/// is streamed directly to the destination while its checksum is accumulated.
/// This keeps peak memory independent of encoded size and visits each
/// correspondence only once while preserving the original byte-for-byte
/// format.
pub fn write(path: &Path, snapshot: &Snapshot) -> Result<(), String> {
    let payload_len = payload_len(snapshot)?;
    let file = std::fs::File::create(path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(MAGIC)
        .and_then(|()| writer.write_all(&SCHEMA_VERSION.to_le_bytes()))
        .and_then(|()| writer.write_all(&payload_len.to_le_bytes()))
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    let mut payload_writer = PayloadDigestWriter::new(&mut writer);
    encode_payload(snapshot, &mut payload_writer)
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    if payload_writer.len != payload_len {
        return Err(format!(
            "snapshot encoded payload length {} differs from computed length {payload_len}",
            payload_writer.len
        ));
    }
    let checksum = payload_writer.checksum;
    writer
        .write_all(&checksum.to_le_bytes())
        .and_then(|()| writer.flush())
        .map_err(|error| format!("write {}: {error}", path.display()))
}

fn read_with_retention(path: &Path, retain_audit_streams: bool) -> Result<Snapshot, String> {
    let header_len = MAGIC.len() + 4 + 8;
    let mut file =
        std::fs::File::open(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|error| format!("stat {}: {error}", path.display()))?
        .len();
    if file_len < (header_len + 8) as u64 {
        return Err(format!(
            "{} is not a verified-pair snapshot (bad magic or truncated header)",
            path.display()
        ));
    }
    let mut magic = vec![0; MAGIC.len()];
    file.read_exact(&mut magic)
        .map_err(|error| format!("read {} header: {error}", path.display()))?;
    if magic != MAGIC {
        return Err(format!(
            "{} is not a verified-pair snapshot (bad magic or truncated header)",
            path.display()
        ));
    }
    let mut u32_bytes = [0; 4];
    file.read_exact(&mut u32_bytes)
        .map_err(|error| format!("read {} schema: {error}", path.display()))?;
    let version = u32::from_le_bytes(u32_bytes);
    if version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported verified-pair snapshot header schema {version} (expected {SCHEMA_VERSION})"
        ));
    }
    let mut u64_bytes = [0; 8];
    file.read_exact(&mut u64_bytes)
        .map_err(|error| format!("read {} payload length: {error}", path.display()))?;
    let payload_len = u64::from_le_bytes(u64_bytes);
    let expected_file_len = (header_len as u64)
        .checked_add(payload_len)
        .and_then(|value| value.checked_add(8))
        .ok_or_else(|| "snapshot payload offset overflow".to_owned())?;
    if expected_file_len != file_len {
        return Err(format!(
            "snapshot payload length {payload_len} does not match file size {}",
            file_len
        ));
    }
    let mut remaining = payload_len;
    let mut actual = 0xcbf29ce484222325u64;
    let mut buffer = vec![0; 1024 * 1024];
    while remaining > 0 {
        let chunk_len = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| "snapshot checksum chunk does not fit usize".to_owned())?;
        file.read_exact(&mut buffer[..chunk_len])
            .map_err(|error| format!("read {} payload: {error}", path.display()))?;
        for byte in &buffer[..chunk_len] {
            actual ^= u64::from(*byte);
            actual = actual.wrapping_mul(0x100000001b3);
        }
        remaining -= chunk_len as u64;
    }
    file.read_exact(&mut u64_bytes)
        .map_err(|error| format!("read {} checksum: {error}", path.display()))?;
    let expected = u64::from_le_bytes(u64_bytes);
    if expected != actual {
        return Err(format!(
            "verified-pair snapshot checksum mismatch: expected {expected:016x}, computed {actual:016x}"
        ));
    }
    file.seek(SeekFrom::Start(header_len as u64))
        .map_err(|error| format!("seek {} payload: {error}", path.display()))?;
    decode_payload(
        BufReader::new(file.take(payload_len)),
        payload_len,
        retain_audit_streams,
    )
}

/// Read, checksum-validate, and losslessly decode a snapshot without reordering anything.
pub fn read(path: &Path) -> Result<Snapshot, String> {
    read_with_retention(path, true)
}

/// Read a snapshot for mapper replay while bounding transient memory per pair.
///
/// The complete file checksum and every raw-index-to-accepted-match relation
/// are validated as the payload streams past. Raw matcher vectors and their
/// inlier-index arrays are then released at each pair boundary; the returned
/// snapshot retains the exact mapper stream, two-view metadata, and envelope.
/// It is intentionally lossy and must not be re-exported as a full snapshot.
pub fn read_mapper_compact(path: &Path) -> Result<Snapshot, String> {
    read_with_retention(path, false)
}

#[cfg(test)]
mod tests {
    use super::{merge, read, read_mapper_compact, write, PairRecord, Snapshot, SCHEMA_VERSION};
    use std::path::PathBuf;

    fn sample() -> Snapshot {
        Snapshot {
            schema_version: SCHEMA_VERSION,
            image_names: vec!["a.png".into(), "b.png".into()],
            image_manifest_hash: 11,
            feature_manifest_hash: 22,
            feature_counts: vec![3, 4],
            width: 1600,
            height: 1066,
            intrinsics_bits: [
                1.25f64.to_bits(),
                2.5f64.to_bits(),
                3.75f64.to_bits(),
                4.0f64.to_bits(),
            ],
            effective_config_hash: 33,
            effective_config: "effective".into(),
            verifier_config_hash: 44,
            verifier_config: "verifier".into(),
            pair_order_hash: 55,
            unordered_edge_hash: 66,
            accepted_match_count: 2,
            pairs: vec![PairRecord {
                image_i: 1,
                image_j: 0,
                raw_match_count: 3,
                raw_matches: vec![(0, 0), (1, 1), (2, 2)],
                accepted_inlier_indices: vec![2, 0],
                essential_inlier_indices: vec![2],
                matches: vec![(2, 2), (0, 0)],
                essential_matches: Some(vec![(2, 2)]),
                config: 3,
                calibrated: true,
                e_inlier_count: 1,
                f_inlier_count: 2,
                h_inlier_count: 0,
                essential_matrix_bits: Some(std::array::from_fn(|i| (i as f64 + 0.25).to_bits())),
                fundamental_matrix_bits: Some(std::array::from_fn(|i| (i as f64 - 0.5).to_bits())),
                homography_matrix_bits: None,
                relative_rotation_bits: None,
                relative_translation_bits: Some([
                    0.1f64.to_bits(),
                    0.2f64.to_bits(),
                    0.3f64.to_bits(),
                ]),
            }],
        }
    }

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "visloc_verified_pair_snapshot_{tag}_{}",
            std::process::id()
        ))
    }

    #[test]
    fn round_trip_preserves_order_and_float_bits() {
        let path = temp_path("roundtrip");
        let snapshot = sample();
        write(&path, &snapshot).unwrap();
        assert_eq!(read(&path).unwrap(), snapshot);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_mapper_read_validates_then_drops_audit_streams() {
        let path = temp_path("compact");
        let snapshot = sample();
        write(&path, &snapshot).unwrap();
        let compact = read_mapper_compact(&path).unwrap();
        assert_eq!(compact.image_names, snapshot.image_names);
        assert_eq!(compact.accepted_match_count, snapshot.accepted_match_count);
        assert_eq!(compact.pairs[0].matches, snapshot.pairs[0].matches);
        assert_eq!(
            compact.pairs[0].essential_matches,
            snapshot.pairs[0].essential_matches
        );
        assert_eq!(
            compact.pairs[0].essential_matrix_bits,
            snapshot.pairs[0].essential_matrix_bits
        );
        assert_eq!(compact.pairs[0].raw_match_count, 0);
        assert!(compact.pairs[0].raw_matches.is_empty());
        assert!(compact.pairs[0].accepted_inlier_indices.is_empty());
        assert!(compact.pairs[0].essential_inlier_indices.is_empty());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_mapper_read_rejects_an_invalid_raw_index_relation() {
        let path = temp_path("compact-invalid-index");
        let mut snapshot = sample();
        snapshot.pairs[0].accepted_inlier_indices[0] = 99;
        write(&path, &snapshot).unwrap();

        let error = read_mapper_compact(&path).unwrap_err();
        assert!(error.contains("accepted match"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn grouped_unordered_edge_hash_matches_flat_reference() {
        let base = sample().pairs.remove(0);
        let mut pairs = Vec::new();
        for (image_i, image_j, matches) in [
            (3, 1, vec![(9, 2), (4, 7), (4, 1)]),
            (0, 2, vec![(8, 8), (1, 5)]),
            (1, 3, vec![(6, 6), (0, 3)]),
        ] {
            let mut pair = base.clone();
            pair.image_i = image_i;
            pair.image_j = image_j;
            pair.matches = matches;
            pairs.push(pair);
        }
        let mut edges = Vec::new();
        for pair in &pairs {
            let (image_i, image_j, swapped) = if pair.image_i <= pair.image_j {
                (pair.image_i, pair.image_j, false)
            } else {
                (pair.image_j, pair.image_i, true)
            };
            for &(left, right) in &pair.matches {
                let (left, right) = if swapped {
                    (right, left)
                } else {
                    (left, right)
                };
                edges.push((image_i, image_j, left, right));
            }
        }
        edges.sort_unstable();
        let reference = edges.into_iter().fold(
            0xcbf29ce484222325u64,
            |mut hash, (image_i, image_j, left, right)| {
                for value in [image_i, image_j, left, right] {
                    hash = super::hash_u64(hash, value);
                }
                hash
            },
        );
        assert_eq!(super::unordered_edge_hash(&pairs), reference);
    }

    #[test]
    fn checksum_and_version_rejection_are_explicit() {
        let path = temp_path("reject");
        write(&path, &sample()).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        assert!(read(&path).unwrap_err().contains("checksum"));

        write(&path, &sample()).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        // The header version starts immediately after MAGIC.
        let version_offset = b"VISLOC-VERIFIED-PAIR-SNAPSHOT\0".len();
        bytes[version_offset] = 2;
        std::fs::write(&path, &bytes).unwrap();
        assert!(read(&path).unwrap_err().contains("schema"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn merge_preserves_shard_order_recomputes_integrity_and_rejects_overlap() {
        let mut first = sample();
        first.pairs[0].image_i = 0;
        first.pairs[0].image_j = 1;
        first.pair_order_hash = super::ordered_pair_hash(&first.pairs);
        first.unordered_edge_hash = super::unordered_edge_hash(&first.pairs);

        let mut second = first.clone();
        second.pairs[0].image_i = 0;
        second.pairs[0].image_j = 0;
        // Use a distinct valid pair with a third image in a small synthetic
        // envelope.  The source sample has only two names, so extend the
        // image manifest consistently for this merge-only test.
        first.image_names.push("c.png".into());
        first.feature_counts.push(5);
        second.image_names = first.image_names.clone();
        second.feature_counts = first.feature_counts.clone();
        second.pairs[0].image_i = 1;
        second.pairs[0].image_j = 2;
        second.pair_order_hash = super::ordered_pair_hash(&second.pairs);
        second.unordered_edge_hash = super::unordered_edge_hash(&second.pairs);

        let merged = merge(&[first.clone(), second.clone()]).unwrap();
        assert_eq!(merged.pairs.len(), 2);
        assert_eq!((merged.pairs[0].image_i, merged.pairs[0].image_j), (0, 1));
        assert_eq!((merged.pairs[1].image_i, merged.pairs[1].image_j), (1, 2));
        assert_eq!(merged.accepted_match_count, 4);
        assert_eq!(
            merged.effective_config,
            format!("verified-pair-export-v1;{}", first.verifier_config)
        );
        assert_eq!(
            merged.effective_config_hash,
            super::fnv1a64(merged.effective_config.as_bytes())
        );
        assert_eq!(
            merged.pair_order_hash,
            super::ordered_pair_hash(&merged.pairs)
        );
        assert_eq!(
            merged.unordered_edge_hash,
            super::unordered_edge_hash(&merged.pairs)
        );

        let overlap = merge(&[first, second.clone(), second]).unwrap_err();
        assert!(overlap.contains("overlap"));
    }

    #[test]
    fn merge_accepts_empty_shards() {
        let populated = sample();
        let mut empty = populated.clone();
        empty.pairs.clear();
        empty.accepted_match_count = 0;
        empty.pair_order_hash = super::ordered_pair_hash(&empty.pairs);
        empty.unordered_edge_hash = super::unordered_edge_hash(&empty.pairs);

        let merged = merge(&[empty.clone(), populated.clone(), empty]).unwrap();
        assert_eq!(merged.pairs, populated.pairs);
        assert_eq!(merged.accepted_match_count, populated.accepted_match_count);
        assert_eq!(
            merged.pair_order_hash,
            super::ordered_pair_hash(&populated.pairs)
        );
        assert_eq!(
            merged.unordered_edge_hash,
            super::unordered_edge_hash(&populated.pairs)
        );
    }

    #[test]
    fn merge_rejects_configuration_mismatch() {
        let first = sample();
        let mut second = first.clone();
        second.verifier_config = "different".into();
        let error = merge(&[first, second]).unwrap_err();
        assert!(error.contains("configuration"));
    }
}
