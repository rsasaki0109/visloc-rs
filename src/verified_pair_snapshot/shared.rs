//! Content-addressed envelope storage. Legacy v1 payload encoding is unchanged.
use super::*;
use sha2::{Digest, Sha256};
use std::sync::Arc;

struct CachedEnvelope {
    path: PathBuf,
    digest: [u8; 32],
    bytes: Arc<[u8]>,
}

/// One-entry cache scoped to a single immutable-input merge pass. Revalidate
/// before eviction and before accepting the pass, never across merge passes.
#[derive(Default)]
pub(super) struct EnvelopeCache {
    entry: Option<CachedEnvelope>,
    pub(super) loads: usize,
}

impl EnvelopeCache {
    pub(super) fn finish(&mut self) -> Result<(), String> {
        if let Some(entry) = self.entry.take() {
            let mut file = std::fs::File::open(&entry.path)
                .map_err(|error| format!("revalidate shared envelope: {error}"))?;
            let mut hasher = Sha256::new();
            let mut buffer = vec![0u8; 64 * 1024];
            loop {
                let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
                if count == 0 {
                    break;
                }
                hasher.update(&buffer[..count]);
            }
            if hasher.finalize()[..] != entry.digest {
                return Err("shared envelope changed during cached merge pass".to_owned());
            }
        }
        Ok(())
    }

    fn get(&mut self, path: PathBuf, digest: [u8; 32]) -> Result<Arc<[u8]>, String> {
        if let Some(entry) = &self.entry {
            if entry.path == path && entry.digest == digest {
                return Ok(Arc::clone(&entry.bytes));
            }
        }
        self.finish()?;
        let bytes = load_envelope(&path, &digest)?;
        self.loads += 1;
        self.entry = Some(CachedEnvelope {
            path,
            digest,
            bytes: Arc::clone(&bytes),
        });
        Ok(bytes)
    }
}

fn load_envelope(path: &Path, digest: &[u8; 32]) -> Result<Arc<[u8]>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read shared envelope: {error}"))?;
    if Sha256::digest(&bytes)[..] != *digest {
        return Err("shared envelope SHA-256 mismatch".to_owned());
    }
    Ok(bytes.into())
}

pub(super) const MAGIC: &[u8] = b"VISLOC-PAIR-CHUNK-V1\0";

fn envelope_path(parent: &Path, digest: &[u8]) -> PathBuf {
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    parent.join(format!("envelope-{hex}.vpe"))
}

fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// Atomically publish a pair chunk referencing an immutable SHA-256 envelope.
/// Keep the `.vpe` files alongside chunks when moving/copying a run directory.
/// Existing legacy writers remain unchanged. This reduces serialized storage;
/// callers still need an envelope cache to avoid repeated envelope reads.
pub fn write_shared_atomic(path: &Path, snapshot: &Snapshot) -> Result<(), String> {
    let writer = SharedSnapshotWriter::new(parent(path), snapshot)?;
    writer.write_chunk(
        path,
        SharedPairChunk {
            pair_order_hash: snapshot.pair_order_hash,
            unordered_edge_hash: snapshot.unordered_edge_hash,
            accepted_match_count: snapshot.accepted_match_count,
            pairs: &snapshot.pairs,
        },
    )?;
    writer.finish()
}

/// Shard-local data referring to the immutable envelope supplied at preparation.
pub struct SharedPairChunk<'a> {
    pub pair_order_hash: u64,
    pub unordered_edge_hash: u64,
    pub accepted_match_count: u64,
    pub pairs: &'a [PairRecord],
}

/// Prepare and publish an envelope once, then stream shard-local records only.
/// Call `finish` before declaring the batch complete. The output directory must
/// remain immutable except for this writer's chunk publications.
pub struct SharedSnapshotWriter {
    directory: PathBuf,
    digest: [u8; 32],
    image_count: usize,
}

impl SharedSnapshotWriter {
    pub fn new(directory: &Path, snapshot: &Snapshot) -> Result<Self, String> {
        let mut prefix = Vec::new();
        encode_payload_prefix(
            snapshot,
            snapshot.pair_order_hash,
            snapshot.unordered_edge_hash,
            snapshot.accepted_match_count,
            snapshot.pairs.len() as u64,
            &mut prefix,
        )?;
        let split = prefix
            .len()
            .checked_sub(32)
            .ok_or("invalid snapshot prefix")?;
        let envelope = &prefix[..split];
        let digest = Sha256::digest(envelope);
        std::fs::create_dir_all(directory).map_err(|error| error.to_string())?;
        let envelope_destination = envelope_path(directory, &digest);
        if !envelope_destination.exists() {
            let (temporary, mut file) =
                create_merge_temporary_file(directory, "envelope", "shared")?;
            file.write_all(envelope)
                .and_then(|()| file.sync_all())
                .map_err(|error| error.to_string())?;
            // Atomic create-if-absent, including concurrent workers using the same envelope.
            match std::fs::hard_link(&temporary.path, &envelope_destination) {
                Ok(()) => (),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
                Err(error) => return Err(format!("publish envelope: {error}")),
            }
        }
        // Never replace a corrupt file at a content-addressed name.
        if std::fs::read(&envelope_destination).map_err(|error| error.to_string())? != envelope {
            return Err("existing shared envelope content mismatch".to_owned());
        }
        Ok(Self {
            directory: directory.to_path_buf(),
            digest: digest.into(),
            image_count: snapshot.image_names.len(),
        })
    }

    pub fn finish(&self) -> Result<(), String> {
        load_envelope(&envelope_path(&self.directory, &self.digest), &self.digest)?;
        Ok(())
    }

    pub fn write_chunk(&self, path: &Path, chunk: SharedPairChunk<'_>) -> Result<(), String> {
        if parent(path) != self.directory
            || path.file_name().is_none()
            || path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("envelope-"))
        {
            return Err(
                "shared chunk must be a non-envelope file in the prepared directory".to_owned(),
            );
        }
        let directory = &self.directory;
        let digest = &self.digest;
        let mut suffix_len = 32u64;
        for pair in chunk.pairs {
            validate_merge_pair(pair, self.image_count)?;
            suffix_len = suffix_len
                .checked_add(pair_payload_len(pair)?)
                .ok_or("shared chunk size overflow")?;
        }
        let (temporary, file) = create_merge_temporary_file(directory, "chunk", "shared")?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(MAGIC)
            .and_then(|()| writer.write_all(digest))
            .and_then(|()| writer.write_all(&suffix_len.to_le_bytes()))
            .map_err(|error| error.to_string())?;
        let mut payload = PayloadDigestWriter::new(&mut writer);
        for byte in digest {
            payload.checksum ^= u64::from(*byte);
            payload.checksum = payload.checksum.wrapping_mul(0x100000001b3);
        }
        for value in [
            chunk.pair_order_hash,
            chunk.unordered_edge_hash,
            chunk.accepted_match_count,
            chunk.pairs.len() as u64,
        ] {
            put_u64(&mut payload, value)?;
        }
        for pair in chunk.pairs {
            encode_pair(pair, &mut payload)?;
        }
        if payload.len != suffix_len {
            return Err("shared chunk length mismatch".to_owned());
        }
        let checksum = payload.checksum;
        writer
            .write_all(&checksum.to_le_bytes())
            .and_then(|()| writer.flush())
            .and_then(|()| writer.get_ref().sync_all())
            .map_err(|error| error.to_string())?;
        drop(writer);
        std::fs::rename(&temporary.path, path).map_err(|error| format!("publish chunk: {error}"))
    }
}

pub(super) fn read(
    path: &Path,
    retain: bool,
    cap: Option<usize>,
    cache: Option<&mut EnvelopeCache>,
) -> Result<Snapshot, String> {
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    let header_len = MAGIC.len() as u64 + 32 + 8;
    file.seek(SeekFrom::Start(MAGIC.len() as u64))
        .map_err(|error| error.to_string())?;
    let mut digest = [0u8; 32];
    let mut bytes = [0u8; 8];
    file.read_exact(&mut digest)
        .and_then(|()| file.read_exact(&mut bytes))
        .map_err(|error| error.to_string())?;
    let suffix_len = u64::from_le_bytes(bytes);
    if header_len
        .checked_add(suffix_len)
        .and_then(|size| size.checked_add(8))
        != Some(length)
    {
        return Err("shared chunk length mismatch".to_owned());
    }
    let envelope_path = envelope_path(parent(path), &digest);
    let envelope = match cache {
        Some(cache) => cache.get(envelope_path, digest)?,
        None => load_envelope(&envelope_path, &digest)?,
    };
    let mut remaining = suffix_len;
    let mut checksum = 0xcbf29ce484222325u64;
    for byte in &digest {
        checksum ^= u64::from(*byte);
        checksum = checksum.wrapping_mul(0x100000001b3);
    }
    let mut buffer = vec![0u8; 1024 * 1024];
    while remaining > 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..count])
            .map_err(|error| error.to_string())?;
        for byte in &buffer[..count] {
            checksum ^= u64::from(*byte);
            checksum = checksum.wrapping_mul(0x100000001b3);
        }
        remaining -= count as u64;
    }
    file.read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    if checksum != u64::from_le_bytes(bytes) {
        return Err("shared chunk checksum mismatch".to_owned());
    }
    file.seek(SeekFrom::Start(header_len))
        .map_err(|error| error.to_string())?;
    let payload_len = (envelope.len() as u64)
        .checked_add(suffix_len)
        .ok_or("shared payload size overflow")?;
    decode_payload(
        std::io::Cursor::new(envelope).chain(BufReader::new(file.take(suffix_len))),
        payload_len,
        retain,
        cap,
    )
}
