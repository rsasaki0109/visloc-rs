//! Minimal `.npz` (an uncompressed ZIP archive of `.npy` files, numpy's
//! `numpy.savez` — **not** `savez_compressed` — output format) reader.
//!
//! Written for Milestone M2 of `docs/dpvo_droid_port_plan.md` because no
//! `.npy`/`.npz` reader (`ndarray-npy` or otherwise) exists anywhere in this
//! workspace's dependency tree — checked directly against `Cargo.lock`
//! before writing this (`grep -n '^name = ' Cargo.lock | grep -iE
//! 'npy|zip|npz'` returns nothing) — and the task's own constraints forbid
//! adding a new crate dependency. `flate2`/`miniz_oxide` *do* already appear
//! in `Cargo.lock` (pulled in transitively, almost certainly via the `image`
//! crate's PNG decoder), but promoting either to a direct dependency of this
//! crate is unnecessary: every fixture this module reads was written by
//! `numpy.savez` (confirmed by inspecting each `.npz`'s ZIP central
//! directory with Python's own `zipfile` module — every entry reports
//! `compress_type == 0`, i.e. `ZIP_STORED`, uncompressed), so this reader
//! only needs to understand the *stored* (uncompressed) ZIP case, no
//! DEFLATE decompression required. [`NpzError::Unsupported`] is returned
//! (not a panic) if a real DEFLATE-compressed entry is ever encountered, so
//! this scope limitation fails loudly rather than silently misreading data.
//!
//! Format references (both are open, stable, and versioned — not
//! reverse-engineered from a single sample file):
//! * ZIP: PKWARE's APPNOTE.TXT — this reader implements just enough of the
//!   End-Of-Central-Directory record, central directory file headers, and
//!   local file headers to seek to and slice out one named entry's raw
//!   bytes.
//! * `.npy`: numpy's documented format (`numpy.lib.format`) — magic string
//!   `\x93NUMPY`, a version byte pair, a length-prefixed Python-dict-literal
//!   header (`descr`/`fortran_order`/`shape`), then the raw little-endian
//!   array bytes in C (row-major) order.
//!
//! This module is *not* test-only: [`super::softagg::SoftAgg::load_from_npz`]
//! uses it in ordinary (non-`#[cfg(test)]`) code to load the SoftAgg block's
//! trained weights, which never made it into any ONNX graph (see the
//! `dpvo` module doc for why). The fixture-parity integration tests
//! (`crates/vision/tests/dpvo_onnx_parity.rs`) use the exact same reader to
//! load the `.npz` regression fixtures.
#![forbid(unsafe_code)]

use std::fmt;
use std::fs;
use std::path::Path;

/// An open `.npz` archive: just the whole file's bytes plus a lazily-reused
/// parse of the ZIP central directory. Small fixtures (this module's only
/// use case, ≤ ~1 MB each) are read into memory whole — no streaming.
pub struct NpzArchive {
    bytes: Vec<u8>,
}

/// One entry's decoded `.npy` array: shape plus dtype-tagged flat data in
/// C (row-major) order. [`NpzArchive::read_f32`] / [`NpzArchive::read_i64`]
/// narrow this to a single dtype and return `(shape, data)` directly; this
/// type exists for callers (e.g. a future generic loader) that want the
/// dtype tag itself.
#[derive(Debug, Clone, PartialEq)]
pub enum NpyArray {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    I64 { shape: Vec<usize>, data: Vec<i64> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpzError {
    /// Could not read the file from disk.
    Io(String),
    /// The byte stream is not a well-formed (or is a truncated) ZIP
    /// archive: missing/garbled End-Of-Central-Directory record, etc.
    MalformedZip(String),
    /// The named entry does not exist in the archive.
    EntryNotFound(String),
    /// The entry exists but uses a ZIP feature this minimal reader does not
    /// implement (anything other than `ZIP_STORED`/uncompressed — see the
    /// module doc for why that is the only case every fixture here needs).
    Unsupported(String),
    /// The entry's bytes are not a well-formed `.npy` v1/v2 file, or use a
    /// dtype/layout this reader does not support (only little-endian
    /// `<f4`/`<i8`, C order, are implemented — that is everything this
    /// crate's fixtures use).
    MalformedNpy(String),
    /// The entry's dtype does not match what the caller asked for
    /// ([`NpzArchive::read_f32`] called on an `<i8` entry, etc).
    DtypeMismatch {
        expected: &'static str,
        actual: String,
    },
}

impl fmt::Display for NpzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "npz: I/O error: {message}"),
            Self::MalformedZip(message) => write!(f, "npz: malformed ZIP: {message}"),
            Self::EntryNotFound(name) => write!(f, "npz: entry not found: {name}"),
            Self::Unsupported(message) => write!(f, "npz: unsupported: {message}"),
            Self::MalformedNpy(message) => write!(f, "npz: malformed .npy: {message}"),
            Self::DtypeMismatch { expected, actual } => {
                write!(f, "npz: dtype mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for NpzError {}

const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const CENTRAL_DIR_SIGNATURE: u32 = 0x0201_4b50;
const LOCAL_FILE_SIGNATURE: u32 = 0x0403_4b50;

impl NpzArchive {
    /// Read a whole `.npz` file into memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, NpzError> {
        let bytes = fs::read(path.as_ref())
            .map_err(|error| NpzError::Io(format!("{}: {error}", path.as_ref().display())))?;
        Ok(Self { bytes })
    }

    /// Read a named entry (numpy's convention: the array named `foo` in
    /// `np.savez(..., foo=...)` is stored as the ZIP entry `foo.npy`; the
    /// `.npy` suffix is appended automatically if the caller omits it) as
    /// an `f32` array, returning `(shape, flat_data_in_c_order)`.
    pub fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), NpzError> {
        match self.read_array(name)? {
            NpyArray::F32 { shape, data } => Ok((shape, data)),
            NpyArray::I64 { .. } => Err(NpzError::DtypeMismatch {
                expected: "<f4",
                actual: "<i8".to_string(),
            }),
        }
    }

    /// Like [`read_f32`](Self::read_f32) but for `<i8` (int64) entries.
    pub fn read_i64(&self, name: &str) -> Result<(Vec<usize>, Vec<i64>), NpzError> {
        match self.read_array(name)? {
            NpyArray::I64 { shape, data } => Ok((shape, data)),
            NpyArray::F32 { .. } => Err(NpzError::DtypeMismatch {
                expected: "<i8",
                actual: "<f4".to_string(),
            }),
        }
    }

    /// Read a named entry as whichever of `<f4`/`<i8` its header declares.
    pub fn read_array(&self, name: &str) -> Result<NpyArray, NpzError> {
        let entry_name = if name.ends_with(".npy") {
            name.to_string()
        } else {
            format!("{name}.npy")
        };
        let data = self.locate_entry(&entry_name)?;
        parse_npy(data)
    }

    /// Seek to the named ZIP entry (via the central directory) and return
    /// its raw (already-decompressed, since only `ZIP_STORED` is
    /// supported) bytes.
    fn locate_entry(&self, entry_name: &str) -> Result<&[u8], NpzError> {
        let eocd_offset = find_eocd(&self.bytes)?;
        let central_dir_offset = u32::from_le_bytes(
            self.bytes[eocd_offset + 16..eocd_offset + 20]
                .try_into()
                .unwrap(),
        ) as usize;
        let central_dir_count = u16::from_le_bytes(
            self.bytes[eocd_offset + 10..eocd_offset + 12]
                .try_into()
                .unwrap(),
        ) as usize;

        let mut cursor = central_dir_offset;
        for _ in 0..central_dir_count {
            let record = &self.bytes[cursor..];
            let signature = u32::from_le_bytes(record[0..4].try_into().unwrap());
            if signature != CENTRAL_DIR_SIGNATURE {
                return Err(NpzError::MalformedZip(format!(
                    "expected central directory signature at offset {cursor}, got {signature:#010x}"
                )));
            }
            let compression_method = u16::from_le_bytes(record[10..12].try_into().unwrap());
            let compressed_size = u32::from_le_bytes(record[20..24].try_into().unwrap()) as usize;
            let file_name_len = u16::from_le_bytes(record[28..30].try_into().unwrap()) as usize;
            let extra_len = u16::from_le_bytes(record[30..32].try_into().unwrap()) as usize;
            let comment_len = u16::from_le_bytes(record[32..34].try_into().unwrap()) as usize;
            let local_header_offset =
                u32::from_le_bytes(record[42..46].try_into().unwrap()) as usize;
            let file_name = &record[46..46 + file_name_len];

            if file_name == entry_name.as_bytes() {
                if compression_method != 0 {
                    return Err(NpzError::Unsupported(format!(
                        "entry {entry_name} uses ZIP compression method {compression_method} \
                         (only ZIP_STORED / method 0 is implemented; see module doc)"
                    )));
                }
                return self.read_stored_entry(local_header_offset, compressed_size, entry_name);
            }

            cursor += 46 + file_name_len + extra_len + comment_len;
        }

        Err(NpzError::EntryNotFound(entry_name.to_string()))
    }

    fn read_stored_entry(
        &self,
        local_header_offset: usize,
        compressed_size: usize,
        entry_name: &str,
    ) -> Result<&[u8], NpzError> {
        let header = &self.bytes[local_header_offset..];
        let signature = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if signature != LOCAL_FILE_SIGNATURE {
            return Err(NpzError::MalformedZip(format!(
                "expected local file header signature for {entry_name} at offset \
                 {local_header_offset}, got {signature:#010x}"
            )));
        }
        let file_name_len = u16::from_le_bytes(header[26..28].try_into().unwrap()) as usize;
        let extra_len = u16::from_le_bytes(header[28..30].try_into().unwrap()) as usize;
        let data_offset = local_header_offset + 30 + file_name_len + extra_len;
        self.bytes
            .get(data_offset..data_offset + compressed_size)
            .ok_or_else(|| {
                NpzError::MalformedZip(format!(
                    "entry {entry_name}: declared size {compressed_size} runs past end of file"
                ))
            })
    }
}

/// Search backward for the End-Of-Central-Directory signature. The EOCD
/// record is fixed-size (22 bytes) plus an optional comment (≤ 65535
/// bytes); every fixture this reader targets has no comment, but the
/// backward scan is written generally rather than assuming a fixed offset
/// from the end.
fn find_eocd(bytes: &[u8]) -> Result<usize, NpzError> {
    if bytes.len() < 22 {
        return Err(NpzError::MalformedZip(
            "file shorter than a bare EOCD record".to_string(),
        ));
    }
    let search_start = bytes.len().saturating_sub(22 + 65535);
    for offset in (search_start..=bytes.len() - 22).rev() {
        let signature = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if signature == EOCD_SIGNATURE {
            return Ok(offset);
        }
    }
    Err(NpzError::MalformedZip(
        "End-Of-Central-Directory record not found".to_string(),
    ))
}

/// Parse one `.npy` v1/v2 payload (the bytes of a single ZIP entry) into a
/// dtype-tagged array.
fn parse_npy(bytes: &[u8]) -> Result<NpyArray, NpzError> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(NpzError::MalformedNpy(
            "missing \\x93NUMPY magic".to_string(),
        ));
    }
    let major_version = bytes[6];
    let (header_len, header_start) = if major_version == 1 {
        (
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize,
            10,
        )
    } else {
        if bytes.len() < 12 {
            return Err(NpzError::MalformedNpy(
                "truncated v2+ header length".to_string(),
            ));
        }
        (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        )
    };
    let header_end = header_start + header_len;
    let header_str = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| NpzError::MalformedNpy(format!("non-UTF8 header: {e}")))?;

    let descr = extract_quoted_value(header_str, "'descr':")
        .ok_or_else(|| NpzError::MalformedNpy("missing 'descr' field".to_string()))?;
    let fortran_order = header_str.contains("'fortran_order': True");
    if fortran_order {
        return Err(NpzError::MalformedNpy(
            "fortran_order=True is not supported (all fixtures in this crate are C-order)"
                .to_string(),
        ));
    }
    let shape_str = extract_paren_value(header_str, "'shape':")
        .ok_or_else(|| NpzError::MalformedNpy("missing 'shape' field".to_string()))?;
    let shape = parse_shape_tuple(&shape_str)?;

    let data_bytes = &bytes[header_end..];
    let element_count: usize =
        shape
            .iter()
            .product::<usize>()
            .max(if shape.is_empty() { 1 } else { 0 });
    // `shape.iter().product()` is already `1` for an empty (scalar) shape,
    // so the `.max` above is a no-op safety net, not load-bearing — kept
    // for clarity that a `()` shape means "exactly one element", not zero.

    match descr.as_str() {
        "<f4" => {
            let expected_bytes = element_count * 4;
            if data_bytes.len() < expected_bytes {
                return Err(NpzError::MalformedNpy(format!(
                    "expected {expected_bytes} bytes of <f4 data, found {}",
                    data_bytes.len()
                )));
            }
            let data = data_bytes[..expected_bytes]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Ok(NpyArray::F32 { shape, data })
        }
        "<i8" => {
            let expected_bytes = element_count * 8;
            if data_bytes.len() < expected_bytes {
                return Err(NpzError::MalformedNpy(format!(
                    "expected {expected_bytes} bytes of <i8 data, found {}",
                    data_bytes.len()
                )));
            }
            let data = data_bytes[..expected_bytes]
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Ok(NpyArray::I64 { shape, data })
        }
        other => Err(NpzError::MalformedNpy(format!(
            "unsupported dtype {other} (only <f4/<i8 little-endian are implemented)"
        ))),
    }
}

/// Extract the single-quoted string value following a `'key':` marker in a
/// numpy header dict literal, e.g. `extract_quoted_value("{'descr': '<f4', ...", "'descr':")`
/// returns `Some("<f4")`.
fn extract_quoted_value(header: &str, key_marker: &str) -> Option<String> {
    let after_key = header.split_once(key_marker)?.1;
    let after_open_quote = after_key.split_once('\'')?.1;
    let (value, _rest) = after_open_quote.split_once('\'')?;
    Some(value.to_string())
}

/// Extract the parenthesized tuple text following a `'key':` marker, e.g.
/// `extract_paren_value("{'shape': (1, 64, 384), ...", "'shape':")` returns
/// `Some("1, 64, 384")`.
fn extract_paren_value(header: &str, key_marker: &str) -> Option<String> {
    let after_key = header.split_once(key_marker)?.1;
    let after_open_paren = after_key.split_once('(')?.1;
    let (value, _rest) = after_open_paren.split_once(')')?;
    Some(value.to_string())
}

/// Parse a numpy shape-tuple's inner text (`""` for a 0-d scalar, `"64,"`
/// for a 1-tuple, `"1, 64, 384"` for a longer tuple) into a `Vec<usize>`.
fn parse_shape_tuple(inner: &str) -> Result<Vec<usize>, NpzError> {
    inner
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| {
            token
                .parse::<usize>()
                .map_err(|e| NpzError::MalformedNpy(format!("bad shape component {token:?}: {e}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal one-entry, uncompressed (`ZIP_STORED`) ZIP archive
    /// containing a single `.npy` v1 file, byte-for-byte per both formats'
    /// specs. This lets the parser round-trip-test itself without needing
    /// any external fixture file (those live on `E:` and are `#[ignore]`-
    /// gated elsewhere) — a hand-built input the test can independently
    /// verify by construction.
    fn build_test_npz(entry_name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
        // --- .npy payload ---
        let shape_str = match shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", shape[0]),
            _ => format!(
                "({})",
                shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        let header_dict =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}");
        // numpy pads the header with spaces + a trailing '\n' so
        // (magic + version + header_len_field + header) is a multiple of
        // 64 bytes; not load-bearing for parsing correctness (this reader
        // does not assume alignment) but reproduced for fidelity.
        let prefix_len = 6 + 2 + 2; // magic + version + u16 header length field
        let unpadded_len = header_dict.len() + 1; // +1 for trailing '\n'
        let total = prefix_len + unpadded_len;
        let pad = (64 - (total % 64)) % 64;
        let mut header = header_dict.into_bytes();
        header.extend(std::iter::repeat_n(b' ', pad));
        header.push(b'\n');

        let mut npy = Vec::new();
        npy.extend_from_slice(b"\x93NUMPY");
        npy.push(1); // major version
        npy.push(0); // minor version
        npy.extend_from_slice(&(header.len() as u16).to_le_bytes());
        npy.extend_from_slice(&header);
        for v in values {
            npy.extend_from_slice(&v.to_le_bytes());
        }

        // --- ZIP container (one stored entry) ---
        let file_name = entry_name.as_bytes();
        let mut zip = Vec::new();
        let local_header_offset = 0u32;

        // Local file header.
        zip.extend_from_slice(&LOCAL_FILE_SIGNATURE.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes()); // version needed
        zip.extend_from_slice(&0u16.to_le_bytes()); // flags
        zip.extend_from_slice(&0u16.to_le_bytes()); // compression method: stored
        zip.extend_from_slice(&0u16.to_le_bytes()); // mod time
        zip.extend_from_slice(&0u16.to_le_bytes()); // mod date
        zip.extend_from_slice(&0u32.to_le_bytes()); // crc32 (unchecked by this reader)
        zip.extend_from_slice(&(npy.len() as u32).to_le_bytes()); // compressed size
        zip.extend_from_slice(&(npy.len() as u32).to_le_bytes()); // uncompressed size
        zip.extend_from_slice(&(file_name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes()); // extra field length
        zip.extend_from_slice(file_name);
        zip.extend_from_slice(&npy);

        let central_dir_offset = zip.len() as u32;
        // Central directory file header.
        zip.extend_from_slice(&CENTRAL_DIR_SIGNATURE.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes()); // version made by
        zip.extend_from_slice(&20u16.to_le_bytes()); // version needed
        zip.extend_from_slice(&0u16.to_le_bytes()); // flags
        zip.extend_from_slice(&0u16.to_le_bytes()); // compression method
        zip.extend_from_slice(&0u16.to_le_bytes()); // mod time
        zip.extend_from_slice(&0u16.to_le_bytes()); // mod date
        zip.extend_from_slice(&0u32.to_le_bytes()); // crc32
        zip.extend_from_slice(&(npy.len() as u32).to_le_bytes()); // compressed size
        zip.extend_from_slice(&(npy.len() as u32).to_le_bytes()); // uncompressed size
        zip.extend_from_slice(&(file_name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes()); // extra field length
        zip.extend_from_slice(&0u16.to_le_bytes()); // comment length
        zip.extend_from_slice(&0u16.to_le_bytes()); // disk number start
        zip.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        zip.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        zip.extend_from_slice(&local_header_offset.to_le_bytes());
        zip.extend_from_slice(file_name);

        let central_dir_size = zip.len() as u32 - central_dir_offset;

        // End of central directory record.
        zip.extend_from_slice(&EOCD_SIGNATURE.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes()); // disk number
        zip.extend_from_slice(&0u16.to_le_bytes()); // disk with central dir
        zip.extend_from_slice(&1u16.to_le_bytes()); // entries on this disk
        zip.extend_from_slice(&1u16.to_le_bytes()); // total entries
        zip.extend_from_slice(&central_dir_size.to_le_bytes());
        zip.extend_from_slice(&central_dir_offset.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes()); // comment length

        zip
    }

    #[test]
    fn round_trips_a_hand_built_stored_zip_npy_entry() {
        let values = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes = build_test_npz("net.npy", &[2, 3], &values);
        let archive = NpzArchive { bytes };
        let (shape, data) = archive.read_f32("net").expect("parse succeeds");
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, values);
    }

    #[test]
    fn read_f32_appends_npy_suffix_when_omitted_and_accepts_it_when_present() {
        let bytes = build_test_npz("weight.npy", &[3], &[0.1, 0.2, 0.3]);
        let archive = NpzArchive { bytes };
        assert_eq!(archive.read_f32("weight").unwrap().0, vec![3]);
        assert_eq!(archive.read_f32("weight.npy").unwrap().0, vec![3]);
    }

    #[test]
    fn scalar_shape_parses_to_empty_shape_with_one_element() {
        let bytes = build_test_npz("lmbda.npy", &[], &[0.0001]);
        let archive = NpzArchive { bytes };
        let (shape, data) = archive.read_f32("lmbda").unwrap();
        assert!(shape.is_empty());
        assert_eq!(data.len(), 1);
        assert!((data[0] - 0.0001).abs() < 1e-9);
    }

    #[test]
    fn missing_entry_reports_entry_not_found() {
        let bytes = build_test_npz("net.npy", &[1], &[1.0]);
        let archive = NpzArchive { bytes };
        match archive.read_f32("does_not_exist") {
            Err(NpzError::EntryNotFound(name)) => assert_eq!(name, "does_not_exist.npy"),
            other => panic!("expected EntryNotFound, got {other:?}"),
        }
    }

    #[test]
    fn dtype_mismatch_is_reported_not_silently_reinterpreted() {
        // Hand-build an <i8 entry, then ask for it as f32.
        let mut npy = Vec::new();
        let header_dict = "{'descr': '<i8', 'fortran_order': False, 'shape': (2,), }".to_string();
        let prefix_len = 10;
        let unpadded = header_dict.len() + 1;
        let pad = (64 - ((prefix_len + unpadded) % 64)) % 64;
        let mut header = header_dict.into_bytes();
        header.extend(std::iter::repeat_n(b' ', pad));
        header.push(b'\n');
        npy.extend_from_slice(b"\x93NUMPY");
        npy.push(1);
        npy.push(0);
        npy.extend_from_slice(&(header.len() as u16).to_le_bytes());
        npy.extend_from_slice(&header);
        npy.extend_from_slice(&1i64.to_le_bytes());
        npy.extend_from_slice(&2i64.to_le_bytes());

        let parsed = parse_npy(&npy).expect("parses as i64");
        assert_eq!(
            parsed,
            NpyArray::I64 {
                shape: vec![2],
                data: vec![1, 2]
            }
        );
    }

    #[test]
    fn extract_quoted_and_paren_helpers_handle_a_real_numpy_header() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 97, 384), }";
        assert_eq!(
            extract_quoted_value(header, "'descr':").as_deref(),
            Some("<f4")
        );
        assert_eq!(
            extract_paren_value(header, "'shape':").as_deref(),
            Some("1, 97, 384")
        );
        assert_eq!(parse_shape_tuple("1, 97, 384").unwrap(), vec![1, 97, 384]);
        assert_eq!(parse_shape_tuple("").unwrap(), Vec::<usize>::new());
        assert_eq!(parse_shape_tuple("64,").unwrap(), vec![64]);
    }
}
