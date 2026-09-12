//! Restart-safe, bounded-memory storage for global image descriptors.
//!
//! The format binds descriptor rows to the model, ordered input manifest, and
//! preprocessing protocol. Rows are appended to a `.partial` sibling and the
//! complete file is installed with one same-directory rename.

use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"VLVPRD01";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 160;
const COMPLETED_OFFSET: u64 = 40;
const TAG_LEN: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalDescriptorBinding {
    pub row_count: u64,
    pub dimension: u32,
    pub resize_width: u32,
    pub resize_height: u32,
    pub model_sha256: [u8; 32],
    pub manifest_sha256: [u8; 32],
    pub preprocessing_sha256: [u8; 32],
}

impl GlobalDescriptorBinding {
    fn record_len(&self) -> Result<usize, String> {
        (self.dimension as usize)
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(TAG_LEN))
            .ok_or_else(|| "global-descriptor record size overflow".to_owned())
    }
}

pub struct GlobalDescriptorWriter {
    final_path: PathBuf,
    partial_path: PathBuf,
    file: File,
    binding: GlobalDescriptorBinding,
    completed: u64,
}

impl GlobalDescriptorWriter {
    pub fn begin_or_resume(
        final_path: impl AsRef<Path>,
        binding: GlobalDescriptorBinding,
    ) -> Result<Self, String> {
        let final_path = final_path.as_ref().to_path_buf();
        if final_path.exists() {
            return Err(format!(
                "refusing to overwrite complete descriptor store {}",
                final_path.display()
            ));
        }
        let parent = final_path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
        let partial_path = partial_path(&final_path)?;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(!partial_path.exists())
            .open(&partial_path)
            .map_err(|error| format!("open {}: {error}", partial_path.display()))?;

        let completed = if file.metadata().map_err(|e| e.to_string())?.len() == 0 {
            write_header(&mut file, &binding, 0)?;
            file.sync_all().map_err(|e| e.to_string())?;
            0
        } else {
            let (stored, _) = read_header(&mut file)?;
            if stored != binding {
                return Err(format!(
                    "partial descriptor binding differs: {}",
                    partial_path.display()
                ));
            }
            recover_valid_rows(&mut file, &binding)?
        };
        file.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        Ok(Self {
            final_path,
            partial_path,
            file,
            binding,
            completed,
        })
    }

    pub fn completed(&self) -> u64 {
        self.completed
    }

    pub fn append(&mut self, descriptor: &[f32]) -> Result<(), String> {
        if self.completed >= self.binding.row_count {
            return Err("descriptor store already contains every declared row".to_owned());
        }
        if descriptor.len() != self.binding.dimension as usize {
            return Err(format!(
                "descriptor dimension {} differs from declared {}",
                descriptor.len(),
                self.binding.dimension
            ));
        }
        let norm2 = descriptor.iter().try_fold(0.0_f64, |sum, value| {
            if value.is_finite() {
                Ok(sum + f64::from(*value) * f64::from(*value))
            } else {
                Err("descriptor contains a non-finite value".to_owned())
            }
        })?;
        if (norm2.sqrt() - 1.0).abs() > 1.0e-3 {
            return Err(format!(
                "descriptor is not L2-normalized (norm={:.9})",
                norm2.sqrt()
            ));
        }
        let mut bytes = Vec::with_capacity(self.binding.dimension as usize * 4);
        for value in descriptor {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let digest = Sha256::digest(&bytes);
        self.file.write_all(&bytes).map_err(|e| e.to_string())?;
        self.file
            .write_all(&digest[..TAG_LEN])
            .map_err(|e| e.to_string())?;
        self.completed += 1;
        Ok(())
    }

    pub fn checkpoint(&mut self) -> Result<(), String> {
        self.file.sync_data().map_err(|e| e.to_string())?;
        write_completed(&mut self.file, self.completed)?;
        self.file.sync_data().map_err(|e| e.to_string())?;
        self.file
            .seek(SeekFrom::End(0))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn finalize(mut self) -> Result<(), String> {
        if self.completed != self.binding.row_count {
            return Err(format!(
                "cannot finalize {} of {} descriptor rows",
                self.completed, self.binding.row_count
            ));
        }
        self.checkpoint()?;
        self.file.sync_all().map_err(|e| e.to_string())?;
        drop(self.file);
        std::fs::rename(&self.partial_path, &self.final_path).map_err(|error| {
            format!(
                "install {} from {}: {error}",
                self.final_path.display(),
                self.partial_path.display()
            )
        })?;
        if let Some(parent) = self.final_path.parent() {
            File::open(parent)
                .and_then(|directory| directory.sync_all())
                .map_err(|error| format!("sync {}: {error}", parent.display()))?;
        }
        Ok(())
    }
}

pub struct GlobalDescriptorStore {
    binding: GlobalDescriptorBinding,
    file: File,
    record_len: usize,
}

impl GlobalDescriptorStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let mut file = File::open(path)
            .map_err(|error| format!("open descriptor store {}: {error}", path.display()))?;
        let (binding, completed) = read_header(&mut file)?;
        if completed != binding.row_count {
            return Err(format!(
                "descriptor store is incomplete: {completed}/{} rows",
                binding.row_count
            ));
        }
        let record_len = binding.record_len()?;
        let expected = expected_len(&binding, binding.row_count)?;
        let actual = file.metadata().map_err(|e| e.to_string())?.len();
        if actual != expected {
            return Err(format!(
                "descriptor store length {actual} != expected {expected}"
            ));
        }
        Ok(Self {
            binding,
            file,
            record_len,
        })
    }

    pub fn binding(&self) -> &GlobalDescriptorBinding {
        &self.binding
    }

    /// Byte stride between descriptor rows, including the trailing checksum.
    pub fn record_len_bytes(&self) -> usize {
        self.record_len
    }

    /// Byte range of only the little-endian f32 payload for one row.
    pub fn descriptor_byte_range(&self, row: u64) -> Result<std::ops::Range<usize>, String> {
        if row >= self.binding.row_count {
            return Err(format!("descriptor row {row} is out of range"));
        }
        let start = HEADER_LEN + row as usize * self.record_len;
        Ok(start..start + self.binding.dimension as usize * 4)
    }

    pub fn descriptor(&self, row: u64) -> Result<Vec<f32>, String> {
        let range = self.descriptor_byte_range(row)?;
        let mut file = self.file.try_clone().map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(range.start as u64))
            .map_err(|e| e.to_string())?;
        let descriptor_len = self.binding.dimension as usize * 4;
        let mut record = vec![0_u8; self.record_len];
        file.read_exact(&mut record).map_err(|e| e.to_string())?;
        let digest = Sha256::digest(&record[..descriptor_len]);
        if digest[..TAG_LEN] != record[descriptor_len..] {
            return Err(format!("descriptor row {row} checksum mismatch"));
        }
        Ok(record[..descriptor_len]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .collect())
    }
}

fn partial_path(path: &Path) -> Result<PathBuf, String> {
    let name = path
        .file_name()
        .ok_or_else(|| format!("output has no filename: {}", path.display()))?;
    let mut partial = name.to_os_string();
    partial.push(".partial");
    Ok(path.with_file_name(partial))
}

fn expected_len(binding: &GlobalDescriptorBinding, rows: u64) -> Result<u64, String> {
    let record = u64::try_from(binding.record_len()?).map_err(|e| e.to_string())?;
    (HEADER_LEN as u64)
        .checked_add(
            rows.checked_mul(record)
                .ok_or("descriptor length overflow")?,
        )
        .ok_or_else(|| "descriptor length overflow".to_owned())
}

fn write_header(
    file: &mut File,
    binding: &GlobalDescriptorBinding,
    completed: u64,
) -> Result<(), String> {
    let mut bytes = [0_u8; HEADER_LEN];
    bytes[0..8].copy_from_slice(MAGIC);
    bytes[8..12].copy_from_slice(&VERSION.to_le_bytes());
    bytes[12..16].copy_from_slice(&(HEADER_LEN as u32).to_le_bytes());
    bytes[16..24].copy_from_slice(&binding.row_count.to_le_bytes());
    bytes[24..28].copy_from_slice(&binding.dimension.to_le_bytes());
    bytes[28..32].copy_from_slice(&binding.resize_width.to_le_bytes());
    bytes[32..36].copy_from_slice(&binding.resize_height.to_le_bytes());
    bytes[40..48].copy_from_slice(&completed.to_le_bytes());
    bytes[48..80].copy_from_slice(&binding.model_sha256);
    bytes[80..112].copy_from_slice(&binding.manifest_sha256);
    bytes[112..144].copy_from_slice(&binding.preprocessing_sha256);
    bytes[144..152].copy_from_slice(&(binding.record_len()? as u64).to_le_bytes());
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    file.write_all(&bytes).map_err(|e| e.to_string())
}

fn read_header(file: &mut File) -> Result<(GlobalDescriptorBinding, u64), String> {
    let mut bytes = [0_u8; HEADER_LEN];
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    if &bytes[0..8] != MAGIC {
        return Err("invalid global-descriptor store magic".to_owned());
    }
    if le_u32(&bytes[8..12]) != VERSION || le_u32(&bytes[12..16]) != HEADER_LEN as u32 {
        return Err("unsupported global-descriptor store version/header".to_owned());
    }
    let mut model = [0_u8; 32];
    let mut manifest = [0_u8; 32];
    let mut preprocessing = [0_u8; 32];
    model.copy_from_slice(&bytes[48..80]);
    manifest.copy_from_slice(&bytes[80..112]);
    preprocessing.copy_from_slice(&bytes[112..144]);
    let binding = GlobalDescriptorBinding {
        row_count: le_u64(&bytes[16..24]),
        dimension: le_u32(&bytes[24..28]),
        resize_width: le_u32(&bytes[28..32]),
        resize_height: le_u32(&bytes[32..36]),
        model_sha256: model,
        manifest_sha256: manifest,
        preprocessing_sha256: preprocessing,
    };
    if binding.row_count == 0 || binding.dimension == 0 {
        return Err("descriptor store declares zero rows or dimension".to_owned());
    }
    if le_u64(&bytes[144..152]) != binding.record_len()? as u64 {
        return Err("descriptor store record length is inconsistent".to_owned());
    }
    Ok((binding, le_u64(&bytes[40..48])))
}

fn recover_valid_rows(file: &mut File, binding: &GlobalDescriptorBinding) -> Result<u64, String> {
    let record_len = binding.record_len()? as u64;
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    if len < HEADER_LEN as u64 {
        return Err("partial descriptor store is shorter than its header".to_owned());
    }
    let candidate_rows = ((len - HEADER_LEN as u64) / record_len).min(binding.row_count);
    let mut record = vec![0_u8; record_len as usize];
    let mut valid = 0;
    file.seek(SeekFrom::Start(HEADER_LEN as u64))
        .map_err(|e| e.to_string())?;
    while valid < candidate_rows {
        file.read_exact(&mut record).map_err(|e| e.to_string())?;
        let split = record.len() - TAG_LEN;
        if Sha256::digest(&record[..split])[..TAG_LEN] != record[split..] {
            break;
        }
        valid += 1;
    }
    file.set_len(expected_len(binding, valid)?)
        .map_err(|e| e.to_string())?;
    write_completed(file, valid)?;
    file.sync_all().map_err(|e| e.to_string())?;
    Ok(valid)
}

fn write_completed(file: &mut File, completed: u64) -> Result<(), String> {
    file.seek(SeekFrom::Start(COMPLETED_OFFSET))
        .map_err(|e| e.to_string())?;
    file.write_all(&completed.to_le_bytes())
        .map_err(|e| e.to_string())
}

fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four bytes"))
}

fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("eight bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "visloc-vpr-store-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn binding(rows: u64) -> GlobalDescriptorBinding {
        GlobalDescriptorBinding {
            row_count: rows,
            dimension: 2,
            resize_width: 640,
            resize_height: 480,
            model_sha256: [1; 32],
            manifest_sha256: [2; 32],
            preprocessing_sha256: [3; 32],
        }
    }

    #[test]
    fn final_store_is_mmap_readable_and_bound() {
        let output = path("final");
        let mut writer = GlobalDescriptorWriter::begin_or_resume(&output, binding(2)).unwrap();
        writer.append(&[0.6, 0.8]).unwrap();
        writer.append(&[0.0, 1.0]).unwrap();
        writer.finalize().unwrap();
        assert!(!partial_path(&output).unwrap().exists());
        let store = GlobalDescriptorStore::open(&output).unwrap();
        assert_eq!(store.binding(), &binding(2));
        let first = store.descriptor(0).unwrap();
        let second = store.descriptor(1).unwrap();
        let dot: f32 = first.iter().zip(second).map(|(a, b)| a * b).sum();
        assert!((dot - 0.8).abs() < 1e-6);
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn resume_discards_a_torn_tail_and_keeps_valid_rows() {
        let output = path("resume");
        {
            let mut writer = GlobalDescriptorWriter::begin_or_resume(&output, binding(2)).unwrap();
            writer.append(&[0.6, 0.8]).unwrap();
            writer.checkpoint().unwrap();
        }
        let partial = partial_path(&output).unwrap();
        OpenOptions::new()
            .append(true)
            .open(&partial)
            .unwrap()
            .write_all(&[9, 8, 7])
            .unwrap();
        let mut resumed = GlobalDescriptorWriter::begin_or_resume(&output, binding(2)).unwrap();
        assert_eq!(resumed.completed(), 1);
        resumed.append(&[0.0, 1.0]).unwrap();
        resumed.finalize().unwrap();
        assert_eq!(
            GlobalDescriptorStore::open(&output)
                .unwrap()
                .binding()
                .row_count,
            2
        );
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn resume_rejects_a_different_binding() {
        let output = path("binding");
        let writer = GlobalDescriptorWriter::begin_or_resume(&output, binding(1)).unwrap();
        drop(writer);
        let mut changed = binding(1);
        changed.model_sha256 = [9; 32];
        assert!(GlobalDescriptorWriter::begin_or_resume(&output, changed).is_err());
        std::fs::remove_file(partial_path(&output).unwrap()).unwrap();
    }
}
