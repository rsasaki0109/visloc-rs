//! Deterministic bounded ANN retrieval over a `.vprd` descriptor store.
//!
//! Multi-table random-projection LSH proposes a small pool and exact cosine
//! similarity reranks only that pool. The output retains at most K neighbours
//! per query before pair deduplication, so candidate state is bounded by K*N.
//! On small tiers, an optional streaming exact scan reports recall@K without
//! ever materializing an N*N matrix.

use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use visloc_rs::global_descriptor_store::GlobalDescriptorStore;

struct Args {
    descriptors: PathBuf,
    out: PathBuf,
    topk: usize,
    tables: usize,
    bits: Option<usize>,
    probes: Option<usize>,
    min_frame_gap: usize,
    exact_audit_max_rows: usize,
    admission_policy: AdmissionPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdmissionPolicy {
    ReciprocalSequenceV1,
    RankMarginPathV2,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            descriptors: PathBuf::new(),
            out: PathBuf::new(),
            topk: 32,
            tables: 8,
            bits: None,
            probes: None,
            min_frame_gap: 64,
            exact_audit_max_rows: 1_000,
            admission_policy: AdmissionPolicy::ReciprocalSequenceV1,
        }
    }
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        let mut next = || {
            values
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--descriptors" => args.descriptors = PathBuf::from(next()?),
            "--out" => args.out = PathBuf::from(next()?),
            "--topk" => args.topk = next()?.parse()?,
            "--tables" => args.tables = next()?.parse()?,
            "--bits" => args.bits = Some(next()?.parse()?),
            "--probes" => args.probes = Some(next()?.parse()?),
            "--min-frame-gap" => args.min_frame_gap = next()?.parse()?,
            "--exact-audit-max-rows" => args.exact_audit_max_rows = next()?.parse()?,
            "--admission-policy" => {
                args.admission_policy = match next()?.as_str() {
                    "reciprocal-sequence-v1" => AdmissionPolicy::ReciprocalSequenceV1,
                    "rank-margin-path-v2" => AdmissionPolicy::RankMarginPathV2,
                    other => return Err(format!("unknown admission policy: {other}").into()),
                }
            }
            "-h" | "--help" => {
                println!(
                    "learned_retrieval_candidates --descriptors globals.vprd --out pairs.tsv \
                     [--topk 32] [--tables 8] [--bits auto] [--probes auto] \
                     [--min-frame-gap 64] [--exact-audit-max-rows 1000] \
                     [--admission-policy reciprocal-sequence-v1|rank-margin-path-v2]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}").into()),
        }
    }
    if args.descriptors.as_os_str().is_empty() || args.out.as_os_str().is_empty() {
        return Err("--descriptors and --out are required".into());
    }
    if args.topk == 0 || args.tables == 0 || args.bits == Some(0) {
        return Err("topk, tables, and an explicit bit count must be positive".into());
    }
    Ok(args)
}

struct MappedDescriptors {
    store: GlobalDescriptorStore,
    mmap: Mmap,
}

impl MappedDescriptors {
    fn open(path: &Path) -> Result<Self, String> {
        let store = GlobalDescriptorStore::open(path)?;
        let file = File::open(path).map_err(|error| error.to_string())?;
        // SAFETY: the complete store is opened read-only and is never mutated
        // by this process. GlobalDescriptorStore has already validated its
        // fixed header and exact file length.
        let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|error| error.to_string())?;
        let mapped = Self { store, mmap };
        mapped.validate_rows()?;
        Ok(mapped)
    }

    fn rows(&self) -> usize {
        self.store.binding().row_count as usize
    }

    fn dimension(&self) -> usize {
        self.store.binding().dimension as usize
    }

    fn row_bytes(&self, row: usize) -> &[u8] {
        let range = self
            .store
            .descriptor_byte_range(row as u64)
            .expect("validated row");
        &self.mmap[range]
    }

    fn value(&self, row: usize, dimension: usize) -> f32 {
        let offset = dimension * 4;
        f32::from_le_bytes(
            self.row_bytes(row)[offset..offset + 4]
                .try_into()
                .expect("four bytes"),
        )
    }

    fn dot(&self, lhs: usize, rhs: usize) -> f32 {
        (0..self.dimension())
            .map(|dimension| self.value(lhs, dimension) * self.value(rhs, dimension))
            .sum()
    }

    fn validate_rows(&self) -> Result<(), String> {
        let tag_len = self
            .store
            .record_len_bytes()
            .checked_sub(self.dimension() * 4)
            .ok_or_else(|| "record is shorter than descriptor payload".to_owned())?;
        for row in 0..self.rows() {
            let range = self.store.descriptor_byte_range(row as u64)?;
            let digest = Sha256::digest(&self.mmap[range.clone()]);
            if digest[..tag_len] != self.mmap[range.end..range.end + tag_len] {
                return Err(format!("descriptor row {row} checksum mismatch"));
            }
        }
        Ok(())
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn signature(
    descriptors: &MappedDescriptors,
    row: usize,
    table: usize,
    bits: usize,
) -> (u64, Vec<u8>) {
    let mut projections = vec![0.0_f32; bits];
    let seed = (table as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
    for dimension in 0..descriptors.dimension() {
        let hash = splitmix64((dimension as u64) ^ seed);
        let bit = hash as usize % bits;
        let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
        projections[bit] += sign * descriptors.value(row, dimension);
    }
    let mut code = 0_u64;
    for (bit, projection) in projections.iter().enumerate() {
        if *projection >= 0.0 {
            code |= 1_u64 << bit;
        }
    }
    let mut order: Vec<u8> = (0..bits as u8).collect();
    order.sort_by(|lhs, rhs| {
        projections[*lhs as usize]
            .abs()
            .total_cmp(&projections[*rhs as usize].abs())
            .then_with(|| lhs.cmp(rhs))
    });
    (code, order)
}

fn insert_topk(best: &mut Vec<(usize, f32)>, candidate: (usize, f32), topk: usize) {
    let position = best.partition_point(|existing| {
        existing
            .1
            .total_cmp(&candidate.1)
            .reverse()
            .then_with(|| existing.0.cmp(&candidate.0))
            .is_lt()
    });
    if position < topk {
        best.insert(position, candidate);
        if best.len() > topk {
            best.pop();
        }
    }
}

struct AnnResult {
    neighbors: Vec<Vec<(usize, f32)>>,
    mean_pool: f64,
    max_pool: usize,
    undersized_queries: usize,
}

fn ann_neighbors(
    descriptors: &MappedDescriptors,
    topk: usize,
    tables: usize,
    bits: usize,
    probes: usize,
    min_gap: usize,
) -> AnnResult {
    let rows = descriptors.rows();
    let mut signatures = vec![0_u64; tables * rows];
    let mut orders = vec![0_u8; tables * rows * bits];
    let mut buckets = vec![BTreeMap::<u64, Vec<usize>>::new(); tables];
    for table in 0..tables {
        for row in 0..rows {
            let (code, order) = signature(descriptors, row, table, bits);
            signatures[table * rows + row] = code;
            let start = (table * rows + row) * bits;
            orders[start..start + bits].copy_from_slice(&order);
            buckets[table].entry(code).or_default().push(row);
        }
    }

    let mut neighbors = vec![Vec::new(); rows];
    let mut pool_sum = 0_usize;
    let mut max_pool = 0_usize;
    let mut undersized_queries = 0_usize;
    for query in 0..rows {
        let mut pool = BTreeSet::new();
        for table in 0..tables {
            let code = signatures[table * rows + query];
            if let Some(bucket) = buckets[table].get(&code) {
                pool.extend(bucket.iter().copied());
            }
            let order_start = (table * rows + query) * bits;
            for bit in &orders[order_start..order_start + probes] {
                if let Some(bucket) = buckets[table].get(&(code ^ (1_u64 << bit))) {
                    pool.extend(bucket.iter().copied());
                }
            }
        }
        pool.remove(&query);
        pool.retain(|candidate| query.abs_diff(*candidate) >= min_gap);
        pool_sum += pool.len();
        max_pool = max_pool.max(pool.len());
        if pool.len() < topk {
            undersized_queries += 1;
        }
        let mut best = Vec::with_capacity(topk);
        for candidate in pool {
            insert_topk(
                &mut best,
                (candidate, descriptors.dot(query, candidate)),
                topk,
            );
        }
        neighbors[query] = best;
    }
    AnnResult {
        neighbors,
        mean_pool: pool_sum as f64 / rows.max(1) as f64,
        max_pool,
        undersized_queries,
    }
}

fn exact_recall(
    descriptors: &MappedDescriptors,
    approximate: &[Vec<(usize, f32)>],
    topk: usize,
    min_gap: usize,
) -> (f64, usize) {
    let mut hits = 0_usize;
    let mut possible = 0_usize;
    for (query, approximate_rows) in approximate.iter().enumerate() {
        let mut exact = Vec::with_capacity(topk);
        for candidate in 0..descriptors.rows() {
            if query != candidate && query.abs_diff(candidate) >= min_gap {
                insert_topk(
                    &mut exact,
                    (candidate, descriptors.dot(query, candidate)),
                    topk,
                );
            }
        }
        let approx: BTreeSet<_> = approximate_rows.iter().map(|row| row.0).collect();
        hits += exact.iter().filter(|row| approx.contains(&row.0)).count();
        possible += exact.len();
    }
    (hits as f64 / possible.max(1) as f64, possible)
}

fn contains(neighbors: &[Vec<(usize, f32)>], query: usize, candidate: usize) -> bool {
    neighbors[query].iter().any(|row| row.0 == candidate)
}

const STRICT_ADMISSION_RANK: usize = 2;
const STRICT_DISTANCE_RATIO: f32 = 0.8;
const STRICT_SEQUENCE_RADIUS: usize = 1;
const STRICT_COMPETITOR_EXCLUSION_RADIUS: usize = 2;

fn rank(neighbors: &[Vec<(usize, f32)>], query: usize, candidate: usize) -> Option<usize> {
    neighbors[query].iter().position(|row| row.0 == candidate)
}

fn cosine_distance(score: f32) -> f32 {
    (1.0 - score).max(0.0)
}

fn distance_ratio(neighbors: &[Vec<(usize, f32)>], query: usize, candidate: usize) -> Option<f32> {
    let chosen = neighbors[query].iter().find(|row| row.0 == candidate)?.1;
    let competing = neighbors[query]
        .iter()
        .find(|row| row.0.abs_diff(candidate) > STRICT_COMPETITOR_EXCLUSION_RADIUS)?
        .1;
    Some(cosine_distance(chosen) / cosine_distance(competing).max(f32::EPSILON))
}

fn strict_edge(neighbors: &[Vec<(usize, f32)>], query: usize, candidate: usize) -> bool {
    rank(neighbors, query, candidate).is_some_and(|value| value < STRICT_ADMISSION_RANK)
        && rank(neighbors, candidate, query).is_some_and(|value| value < STRICT_ADMISSION_RANK)
}

fn strict_path_direction(
    neighbors: &[Vec<(usize, f32)>],
    query: usize,
    candidate: usize,
    direction: isize,
) -> bool {
    (-(STRICT_SEQUENCE_RADIUS as isize)..=STRICT_SEQUENCE_RADIUS as isize).all(|delta| {
        let Some(path_query) = query.checked_add_signed(delta) else {
            return false;
        };
        let Some(path_candidate) = candidate.checked_add_signed(direction * delta) else {
            return false;
        };
        path_query < neighbors.len()
            && path_candidate < neighbors.len()
            && strict_edge(neighbors, path_query, path_candidate)
    })
}

fn strict_selected(neighbors: &[Vec<(usize, f32)>], query: usize, candidate: usize) -> bool {
    strict_edge(neighbors, query, candidate)
        && distance_ratio(neighbors, query, candidate)
            .is_some_and(|value| value <= STRICT_DISTANCE_RATIO)
        && distance_ratio(neighbors, candidate, query)
            .is_some_and(|value| value <= STRICT_DISTANCE_RATIO)
        && (strict_path_direction(neighbors, query, candidate, 1)
            || strict_path_direction(neighbors, query, candidate, -1))
}

fn sequence_support(neighbors: &[Vec<(usize, f32)>], query: usize, candidate: usize) -> usize {
    [-2_isize, -1, 1, 2]
        .into_iter()
        .filter(|delta| {
            let Some(adjacent_query) = query.checked_add_signed(*delta) else {
                return false;
            };
            let Some(expected_candidate) = candidate.checked_add_signed(*delta) else {
                return false;
            };
            if adjacent_query >= neighbors.len() || expected_candidate >= neighbors.len() {
                return false;
            }
            neighbors[adjacent_query]
                .iter()
                .any(|row| row.0.abs_diff(expected_candidate) <= 1)
        })
        .count()
}

#[derive(Default)]
struct Candidate {
    score: f32,
    mutual: bool,
    sequence_support: usize,
}

fn selected(row: &Candidate) -> bool {
    row.mutual && row.sequence_support >= 2
}

fn candidate_pairs(neighbors: &[Vec<(usize, f32)>]) -> BTreeMap<(usize, usize), Candidate> {
    let mut pairs = BTreeMap::new();
    for (query, rows) in neighbors.iter().enumerate() {
        for &(candidate, score) in rows {
            let pair = (query.min(candidate), query.max(candidate));
            let row = pairs.entry(pair).or_insert_with(Candidate::default);
            row.score = row.score.max(score);
            row.mutual |= contains(neighbors, candidate, query);
            row.sequence_support = row
                .sequence_support
                .max(sequence_support(neighbors, query, candidate));
        }
    }
    pairs
}

fn hash_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn write_atomic(path: &Path, body: &str) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()).into());
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let filename = path.file_name().ok_or("output path has no filename")?;
    let mut temporary = filename.to_os_string();
    temporary.push(format!(".tmp-{}", std::process::id()));
    let temporary = path.with_file_name(temporary);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(body.as_bytes())?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn automatic_bits(rows: usize) -> usize {
    let target_bucket = 16_usize;
    let buckets = rows.div_ceil(target_bucket).max(2).next_power_of_two();
    buckets.trailing_zeros().clamp(1, 63) as usize
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let descriptors = MappedDescriptors::open(&args.descriptors)?;
    let bits = args
        .bits
        .unwrap_or_else(|| automatic_bits(descriptors.rows()));
    if bits > 63 {
        return Err("--bits must be at most 63".into());
    }
    let probes = args.probes.unwrap_or(bits.min(9));
    if probes > bits {
        return Err("--probes must not exceed --bits".into());
    }
    let ann = ann_neighbors(
        &descriptors,
        args.topk,
        args.tables,
        bits,
        probes,
        args.min_frame_gap,
    );
    let exact = (descriptors.rows() <= args.exact_audit_max_rows)
        .then(|| exact_recall(&descriptors, &ann.neighbors, args.topk, args.min_frame_gap));
    let pairs = candidate_pairs(&ann.neighbors);
    if pairs.len() > args.topk * descriptors.rows() {
        return Err("candidate count exceeded K*N invariant".into());
    }
    let selected_count = pairs
        .iter()
        .filter(|((query, candidate), row)| match args.admission_policy {
            AdmissionPolicy::ReciprocalSequenceV1 => selected(row),
            AdmissionPolicy::RankMarginPathV2 => {
                strict_selected(&ann.neighbors, *query, *candidate)
            }
        })
        .count();
    let binding = descriptors.store.binding();
    let mut output = format!(
        "# visloc-learned-retrieval-{}\n# descriptor_sha256 {}\n# model_sha256 {}\n# manifest_sha256 {}\n# preprocessing_sha256 {}\n# rows {}\n# dimension {}\n# topk {}\n# tables {}\n# bits {}\n# probes {}\n# min_frame_gap {}\n# mean_pool {:.9}\n# max_pool {}\n# undersized_queries {}\n# candidate_pairs {}\n# selected_pairs {}\n",
        match args.admission_policy {
            AdmissionPolicy::ReciprocalSequenceV1 => "v1",
            AdmissionPolicy::RankMarginPathV2 => "v2",
        },
        hash_file(&args.descriptors)?,
        hex(&binding.model_sha256),
        hex(&binding.manifest_sha256),
        hex(&binding.preprocessing_sha256),
        descriptors.rows(),
        descriptors.dimension(),
        args.topk,
        args.tables,
        bits,
        probes,
        args.min_frame_gap,
        ann.mean_pool,
        ann.max_pool,
        ann.undersized_queries,
        pairs.len(),
        selected_count,
    );
    if let Some((recall, denominator)) = exact {
        output.push_str(&format!(
            "# exact_recall_at_k {:.9}\n# exact_recall_denominator {}\n",
            recall, denominator
        ));
    } else {
        output.push_str("# exact_recall_at_k skipped\n");
    }
    match args.admission_policy {
        AdmissionPolicy::ReciprocalSequenceV1 => {
            output.push_str("# admission_policy reciprocal-sequence-v1\n");
            output.push_str("# pair query candidate cosine mutual sequence_support selected\n");
        }
        AdmissionPolicy::RankMarginPathV2 => {
            output.push_str("# admission_policy rank-margin-path-v2\n");
            output.push_str("# admission_rank 2\n# distance_ratio_max 0.8\n");
            output.push_str("# competitor_exclusion_radius 2\n# sequence_path_radius 1\n");
            output.push_str("# per_frame_addition_budget 2\n");
            output.push_str(
                "# pair query candidate cosine query_rank candidate_rank query_ratio candidate_ratio path selected\n",
            );
        }
    }
    for ((query, candidate), row) in &pairs {
        match args.admission_policy {
            AdmissionPolicy::ReciprocalSequenceV1 => {
                let selected_pair = selected(row);
                output.push_str(&format!(
                    "pair {query} {candidate} {:.9} {} {} {}\n",
                    row.score, row.mutual, row.sequence_support, selected_pair
                ));
            }
            AdmissionPolicy::RankMarginPathV2 => {
                let query_rank = rank(&ann.neighbors, *query, *candidate);
                let candidate_rank = rank(&ann.neighbors, *candidate, *query);
                let query_ratio = distance_ratio(&ann.neighbors, *query, *candidate);
                let candidate_ratio = distance_ratio(&ann.neighbors, *candidate, *query);
                let path = strict_path_direction(&ann.neighbors, *query, *candidate, 1)
                    || strict_path_direction(&ann.neighbors, *query, *candidate, -1);
                let selected_pair = strict_selected(&ann.neighbors, *query, *candidate);
                output.push_str(&format!(
                    "pair {query} {candidate} {:.9} {} {} {} {} {} {}\n",
                    row.score,
                    query_rank.map_or(-1, |value| value as isize),
                    candidate_rank.map_or(-1, |value| value as isize),
                    query_ratio.map_or(f32::INFINITY, |value| value),
                    candidate_ratio.map_or(f32::INFINITY, |value| value),
                    path,
                    selected_pair
                ));
            }
        }
    }
    write_atomic(&args.out, &output)?;
    println!(
        "ANN rows={} dim={} tables={} bits={} probes={} topk={} pool_mean={:.1} pool_max={} candidates={} selected={} exact_recall={} -> {}",
        descriptors.rows(), descriptors.dimension(), args.tables, bits, probes, args.topk,
        ann.mean_pool, ann.max_pool, pairs.len(), selected_count,
        exact.map(|row| format!("{:.6}", row.0)).unwrap_or_else(|| "skipped".to_owned()),
        args.out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_has_stable_score_then_index_order() {
        let mut rows = Vec::new();
        for row in [(5, 0.5), (3, 0.5), (8, 0.7), (1, 0.1)] {
            insert_topk(&mut rows, row, 3);
        }
        assert_eq!(rows, vec![(8, 0.7), (3, 0.5), (5, 0.5)]);
    }

    #[test]
    fn sequence_support_tracks_same_direction_neighbors() {
        let mut rows = vec![Vec::new(); 12];
        rows[4].push((9, 1.0));
        rows[5].push((10, 1.0));
        rows[6].push((11, 1.0));
        assert_eq!(sequence_support(&rows, 5, 10), 2);
    }

    #[test]
    fn automatic_bits_keeps_expected_bucket_bounded() {
        assert_eq!(automatic_bits(500), 5);
        assert_eq!(automatic_bits(5_000), 9);
        assert_eq!(automatic_bits(50_000), 12);
    }

    #[test]
    fn selection_requires_both_reciprocal_and_sequence_evidence() {
        assert!(!selected(&Candidate {
            score: 1.0,
            mutual: true,
            sequence_support: 1,
        }));
        assert!(!selected(&Candidate {
            score: 1.0,
            mutual: false,
            sequence_support: 4,
        }));
        assert!(selected(&Candidate {
            score: 1.0,
            mutual: true,
            sequence_support: 2,
        }));
    }

    #[test]
    fn strict_selection_requires_margin_and_bidirectional_three_point_path() {
        let mut rows = vec![Vec::new(); 20];
        for (query, candidate) in [(4, 14), (5, 15), (6, 16)] {
            rows[query] = vec![(candidate, 0.99), (candidate + 1, 0.989), (0, 0.9)];
            rows[candidate] = vec![(query, 0.99), (query + 1, 0.989), (19, 0.9)];
        }
        assert!(strict_selected(&rows, 5, 15));
        rows[5][2].1 = 0.988;
        assert!(!strict_selected(&rows, 5, 15));
    }

    #[test]
    fn strict_path_accepts_reverse_traversal() {
        let mut rows = vec![Vec::new(); 20];
        for (query, candidate) in [(4, 16), (5, 15), (6, 14)] {
            rows[query] = vec![(candidate, 0.99), (0, 0.9)];
            rows[candidate] = vec![(query, 0.99), (19, 0.9)];
        }
        assert!(strict_path_direction(&rows, 5, 15, -1));
    }
}
