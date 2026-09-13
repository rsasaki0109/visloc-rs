//! Admit learned retrieval additions only when verified rotations form a
//! multi-sensor, contiguous rig-frame cycle.
//!
//! The gate consumes only the rig calibration, the pre-mapping attribution
//! ledger, and relative rotations already published by the frozen two-view
//! verifier. It never reads a reconstructed pose or ground truth. A rig pair
//! needs at least two sensor-pair rotations whose common-rig-frame dispersion
//! is at most three degrees. The representative rotations must then agree
//! within three degrees over a three-frame forward or reverse sequence path.
//! Only image-pair records carrying the admitted rotation evidence survive.

use nalgebra::{Matrix3, Quaternion, UnitQuaternion};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use visloc_rs::verified_pair_snapshot::{merge_owned, read, write_atomic, PairRecord, Snapshot};

const MIN_SENSOR_ROTATIONS: usize = 2;
const MAX_WITHIN_RIG_PAIR_DEG: f64 = 3.0;
const MAX_PATH_ROTATION_DEG: f64 = 3.0;
const PATH_RADIUS: isize = 1;

#[derive(Debug)]
struct Args {
    addition_snapshot: PathBuf,
    addition_ledger: PathBuf,
    rig_manifest: PathBuf,
    output_directory: PathBuf,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut addition_snapshot = None;
    let mut addition_ledger = None;
    let mut rig_manifest = None;
    let mut output_directory = None;
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        let mut next = || {
            values
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--addition-snapshot" => addition_snapshot = Some(PathBuf::from(next()?)),
            "--addition-ledger" => addition_ledger = Some(PathBuf::from(next()?)),
            "--rig-manifest" => rig_manifest = Some(PathBuf::from(next()?)),
            "--output-directory" => output_directory = Some(PathBuf::from(next()?)),
            "-h" | "--help" => {
                println!(
                    "admit_learned_rig_rotation_cycles --addition-snapshot ADD.vps \
                     --addition-ledger additions.tsv --rig-manifest rig.txt \
                     --output-directory NEW_DIR"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}").into()),
        }
    }
    Ok(Args {
        addition_snapshot: addition_snapshot.ok_or("--addition-snapshot is required")?,
        addition_ledger: addition_ledger.ok_or("--addition-ledger is required")?,
        rig_manifest: rig_manifest.ok_or("--rig-manifest is required")?,
        output_directory: output_directory.ok_or("--output-directory is required")?,
    })
}

#[derive(Clone)]
struct ImageAssignment {
    frame: usize,
    sensor: usize,
}

struct RigBinding {
    sensors: Vec<UnitQuaternion<f64>>,
    images: Vec<ImageAssignment>,
}

fn parse_rig(path: &Path, image_names: &[String]) -> Result<RigBinding, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read rig manifest {}: {error}", path.display()))?;
    let image_index = image_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut sensors = BTreeMap::new();
    let mut images = vec![None; image_names.len()];
    let mut frames = BTreeSet::new();
    for (zero_line, raw) in text.lines().enumerate() {
        let fields = raw.split_whitespace().collect::<Vec<_>>();
        match fields.first().copied() {
            Some("S") => {
                if fields.len() != 16 {
                    return Err(format!("rig line {} has malformed S row", zero_line + 1));
                }
                let sensor: usize = fields[1]
                    .parse()
                    .map_err(|error| format!("rig sensor index: {error}"))?;
                let quaternion: Quaternion<f64> = Quaternion::new(
                    fields[9]
                        .parse()
                        .map_err(|error| format!("rig qw: {error}"))?,
                    fields[10]
                        .parse()
                        .map_err(|error| format!("rig qx: {error}"))?,
                    fields[11]
                        .parse()
                        .map_err(|error| format!("rig qy: {error}"))?,
                    fields[12]
                        .parse()
                        .map_err(|error| format!("rig qz: {error}"))?,
                );
                if !quaternion.coords.iter().all(|value| value.is_finite())
                    || quaternion.norm() <= 1.0e-12
                {
                    return Err(format!("rig sensor {sensor} has invalid rotation"));
                }
                if sensors
                    .insert(sensor, UnitQuaternion::new_normalize(quaternion))
                    .is_some()
                {
                    return Err(format!("rig sensor {sensor} is repeated"));
                }
            }
            Some("F") => {
                if fields.len() != 4 {
                    return Err(format!("rig line {} has malformed F row", zero_line + 1));
                }
                let frame: usize = fields[1]
                    .parse()
                    .map_err(|error| format!("rig frame index: {error}"))?;
                let sensor: usize = fields[3]
                    .parse()
                    .map_err(|error| format!("rig frame sensor: {error}"))?;
                let image = *image_index
                    .get(fields[2])
                    .ok_or_else(|| format!("rig names unknown snapshot image {:?}", fields[2]))?;
                if images[image]
                    .replace(ImageAssignment { frame, sensor })
                    .is_some()
                {
                    return Err(format!("rig assigns image {:?} twice", fields[2]));
                }
                frames.insert(frame);
            }
            _ => {}
        }
    }
    if sensors.keys().copied().ne(0..sensors.len()) {
        return Err("rig sensor indices must be contiguous from zero".into());
    }
    if frames.iter().copied().ne(0..frames.len()) {
        return Err("rig frame indices must be contiguous from zero".into());
    }
    let sensors = sensors.into_values().collect::<Vec<_>>();
    let images = images
        .into_iter()
        .enumerate()
        .map(|(image, assignment)| {
            let assignment =
                assignment.ok_or_else(|| format!("rig omits snapshot image {image}"))?;
            if assignment.sensor >= sensors.len() {
                return Err(format!(
                    "image {image} references unknown sensor {}",
                    assignment.sensor
                ));
            }
            Ok(assignment)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(RigBinding { sensors, images })
}

type ImagePair = (usize, usize);
type RigPair = (usize, usize);

fn validate_snapshot_subset(
    snapshot_keys: &BTreeSet<ImagePair>,
    owners: &BTreeMap<ImagePair, RigPair>,
) -> Result<(), String> {
    let ledger_keys = owners.keys().copied().collect::<BTreeSet<_>>();
    if !snapshot_keys.is_subset(&ledger_keys) {
        return Err(
            "addition snapshot contains image pairs absent from the attribution ledger".into(),
        );
    }
    Ok(())
}

fn parse_ledger(path: &Path) -> Result<BTreeMap<ImagePair, RigPair>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read addition ledger {}: {error}", path.display()))?;
    if !text.lines().any(|line| {
        matches!(
            line,
            "# admission_policy rank-margin-path-v2"
                | "# admission_policy rank-path-cycle-v3"
                | "# admission_policy component-bridge-v4"
                | "# admission_policy multi-scale-component-bridge-v5"
        )
    }) {
        return Err("addition ledger is not bound to a cycle-gated retrieval policy".into());
    }
    let mut owners = BTreeMap::new();
    for (zero_line, line) in text.lines().enumerate() {
        if !line.starts_with("image_pair ")
            || line == "image_pair image_i image_j rig_query rig_candidate cosine sequence_support"
        {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 7 {
            return Err(format!(
                "addition ledger line {} is malformed",
                zero_line + 1
            ));
        }
        let image_i: usize = fields[1]
            .parse()
            .map_err(|error| format!("ledger image_i: {error}"))?;
        let image_j: usize = fields[2]
            .parse()
            .map_err(|error| format!("ledger image_j: {error}"))?;
        let query: usize = fields[3]
            .parse()
            .map_err(|error| format!("ledger rig_query: {error}"))?;
        let candidate: usize = fields[4]
            .parse()
            .map_err(|error| format!("ledger rig_candidate: {error}"))?;
        if image_i >= image_j || query >= candidate {
            return Err(format!(
                "addition ledger line {} is not canonical",
                zero_line + 1
            ));
        }
        if owners
            .insert((image_i, image_j), (query, candidate))
            .is_some()
        {
            return Err(format!(
                "addition ledger repeats pair ({image_i},{image_j})"
            ));
        }
    }
    if owners.is_empty() {
        return Err("addition ledger contains no image pairs".into());
    }
    Ok(owners)
}

fn rotation_from_bits(bits: [u64; 9]) -> Option<UnitQuaternion<f64>> {
    let matrix = Matrix3::from_column_slice(&bits.map(f64::from_bits));
    if !matrix.iter().all(|value| value.is_finite())
        || (matrix.transpose() * matrix - Matrix3::identity()).norm() > 1.0e-3
        || (matrix.determinant() - 1.0).abs() > 1.0e-3
    {
        return None;
    }
    Some(UnitQuaternion::from_matrix(&matrix))
}

fn rig_rotation(
    pair: &PairRecord,
    owner: RigPair,
    rig: &RigBinding,
) -> Result<Option<UnitQuaternion<f64>>, String> {
    let image_i = usize::try_from(pair.image_i).map_err(|_| "image_i does not fit usize")?;
    let image_j = usize::try_from(pair.image_j).map_err(|_| "image_j does not fit usize")?;
    let assignment_i = rig
        .images
        .get(image_i)
        .ok_or_else(|| format!("pair image {image_i} is outside rig"))?;
    let assignment_j = rig
        .images
        .get(image_j)
        .ok_or_else(|| format!("pair image {image_j} is outside rig"))?;
    let Some(sensor_rotation) = pair.relative_rotation_bits.and_then(rotation_from_bits) else {
        return Ok(None);
    };
    let mut rotation = rig.sensors[assignment_j.sensor].inverse()
        * sensor_rotation
        * rig.sensors[assignment_i.sensor];
    if (assignment_i.frame, assignment_j.frame) == owner {
        return Ok(Some(rotation));
    }
    if (assignment_j.frame, assignment_i.frame) == owner {
        rotation = rotation.inverse();
        return Ok(Some(rotation));
    }
    Err(format!(
        "image pair ({image_i},{image_j}) frames ({},{}) disagree with ledger owner {owner:?}",
        assignment_i.frame, assignment_j.frame
    ))
}

fn rotation_medoid(rotations: &[UnitQuaternion<f64>]) -> (usize, f64) {
    let index = (0..rotations.len())
        .min_by(|&left, &right| {
            let score = |index: usize| {
                rotations
                    .iter()
                    .map(|other| rotations[index].angle_to(other))
                    .sum::<f64>()
            };
            score(left)
                .total_cmp(&score(right))
                .then_with(|| left.cmp(&right))
        })
        .expect("non-empty rotations");
    let maximum = rotations
        .iter()
        .map(|other| rotations[index].angle_to(other).to_degrees())
        .fold(0.0_f64, f64::max);
    (index, maximum)
}

struct RigEvidence {
    rotation: UnitQuaternion<f64>,
    image_pairs: Vec<ImagePair>,
}

fn path_consistent(
    evidence: &BTreeMap<RigPair, RigEvidence>,
    owner: RigPair,
    direction: isize,
) -> bool {
    let center = &evidence[&owner].rotation;
    (-PATH_RADIUS..=PATH_RADIUS).all(|delta| {
        let Some(query) = owner.0.checked_add_signed(delta) else {
            return false;
        };
        let Some(candidate) = owner.1.checked_add_signed(direction * delta) else {
            return false;
        };
        evidence
            .get(&(query, candidate))
            .is_some_and(|row| row.rotation.angle_to(center).to_degrees() <= MAX_PATH_ROTATION_DEG)
    })
}

struct Admission {
    snapshot: Snapshot,
    ledger: String,
    rig_pairs: usize,
    image_pairs: usize,
}

fn admit(
    mut snapshot: Snapshot,
    owners: &BTreeMap<ImagePair, RigPair>,
    rig: &RigBinding,
) -> Result<Admission, String> {
    let snapshot_keys = snapshot
        .pairs
        .iter()
        .map(|pair| {
            let left = usize::try_from(pair.image_i).map_err(|_| "image_i does not fit usize")?;
            let right = usize::try_from(pair.image_j).map_err(|_| "image_j does not fit usize")?;
            Ok((left.min(right), left.max(right)))
        })
        .collect::<Result<BTreeSet<_>, String>>()?;
    validate_snapshot_subset(&snapshot_keys, owners)?;
    let mut grouped = BTreeMap::<RigPair, Vec<(ImagePair, UnitQuaternion<f64>)>>::new();
    for pair in &snapshot.pairs {
        let key = (pair.image_i as usize, pair.image_j as usize);
        let owner = owners[&key];
        if let Some(rotation) = rig_rotation(pair, owner, rig)? {
            grouped.entry(owner).or_default().push((key, rotation));
        }
    }
    let mut evidence = BTreeMap::new();
    let all_owners = owners.values().copied().collect::<BTreeSet<_>>();
    let mut diagnostic = BTreeMap::<RigPair, (usize, f64)>::new();
    for owner in &all_owners {
        let rotations = grouped.get(owner).map(Vec::as_slice).unwrap_or(&[]);
        if rotations.is_empty() {
            diagnostic.insert(*owner, (0, f64::INFINITY));
            continue;
        }
        let values = rotations.iter().map(|row| row.1).collect::<Vec<_>>();
        let (medoid, maximum) = rotation_medoid(&values);
        diagnostic.insert(*owner, (values.len(), maximum));
        if values.len() >= MIN_SENSOR_ROTATIONS && maximum <= MAX_WITHIN_RIG_PAIR_DEG {
            evidence.insert(
                *owner,
                RigEvidence {
                    rotation: values[medoid],
                    image_pairs: rotations.iter().map(|row| row.0).collect(),
                },
            );
        }
    }
    let admitted_owners = evidence
        .keys()
        .copied()
        .filter(|owner| {
            path_consistent(&evidence, *owner, 1) || path_consistent(&evidence, *owner, -1)
        })
        .collect::<BTreeSet<_>>();
    let admitted_images = admitted_owners
        .iter()
        .flat_map(|owner| evidence[owner].image_pairs.iter().copied())
        .collect::<BTreeSet<_>>();
    snapshot
        .pairs
        .retain(|pair| admitted_images.contains(&(pair.image_i as usize, pair.image_j as usize)));
    let image_pairs = snapshot.pairs.len();
    let snapshot = merge_owned(vec![snapshot])?;
    let mut ledger = format!(
        "# visloc-learned-rig-rotation-cycles-v1\n# min_sensor_rotations {MIN_SENSOR_ROTATIONS}\n# max_within_rig_pair_deg {MAX_WITHIN_RIG_PAIR_DEG}\n# max_path_rotation_deg {MAX_PATH_ROTATION_DEG}\n# path_radius {PATH_RADIUS}\n# rig_pair query candidate rotations max_dispersion_deg path admitted_image_pairs\n"
    );
    for owner in all_owners {
        let (count, dispersion) = diagnostic[&owner];
        let path = admitted_owners.contains(&owner);
        let admitted = evidence
            .get(&owner)
            .filter(|_| path)
            .map_or(0, |row| row.image_pairs.len());
        writeln!(
            ledger,
            "rig_pair {} {} {count} {dispersion} {path} {admitted}",
            owner.0, owner.1
        )
        .expect("write String");
    }
    Ok(Admission {
        snapshot,
        ledger,
        rig_pairs: admitted_owners.len(),
        image_pairs,
    })
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
    let snapshot = read(&args.addition_snapshot).map_err(std::io::Error::other)?;
    let rig =
        parse_rig(&args.rig_manifest, &snapshot.image_names).map_err(std::io::Error::other)?;
    let owners = parse_ledger(&args.addition_ledger).map_err(std::io::Error::other)?;
    let admission = admit(snapshot, &owners, &rig).map_err(std::io::Error::other)?;

    let name = args
        .output_directory
        .file_name()
        .ok_or("output directory has no filename")?;
    let mut temporary_name = name.to_os_string();
    temporary_name.push(format!(".tmp-{}", std::process::id()));
    let temporary = args.output_directory.with_file_name(temporary_name);
    std::fs::create_dir(&temporary)?;
    write_atomic(
        &temporary.join("verified-admitted.vps"),
        &admission.snapshot,
    )
    .map_err(std::io::Error::other)?;
    write_synced(&temporary.join("admission-ledger.tsv"), &admission.ledger)?;
    File::open(&temporary)?.sync_all()?;
    std::fs::rename(&temporary, &args.output_directory)?;
    if let Some(parent) = args.output_directory.parent() {
        File::open(parent)?.sync_all()?;
    }
    println!(
        "admitted rig_pairs={} image_pairs={} -> {}",
        admission.rig_pairs,
        admission.image_pairs,
        args.output_directory.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z_rotation(degrees: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), degrees.to_radians())
    }

    fn evidence(rotation: f64) -> RigEvidence {
        RigEvidence {
            rotation: z_rotation(rotation),
            image_pairs: Vec::new(),
        }
    }

    #[test]
    fn medoid_reports_rotation_dispersion() {
        let values = [z_rotation(1.0), z_rotation(2.0), z_rotation(8.0)];
        let (index, maximum) = rotation_medoid(&values);
        assert_eq!(index, 1);
        assert!((maximum - 6.0).abs() < 1.0e-9);
    }

    #[test]
    fn path_gate_accepts_forward_and_rejects_missing_neighbor() {
        let mut rows = BTreeMap::from([
            ((4, 14), evidence(2.0)),
            ((5, 15), evidence(2.5)),
            ((6, 16), evidence(3.0)),
        ]);
        assert!(path_consistent(&rows, (5, 15), 1));
        rows.remove(&(4, 14));
        assert!(!path_consistent(&rows, (5, 15), 1));
    }

    #[test]
    fn path_gate_accepts_reverse_direction() {
        let rows = BTreeMap::from([
            ((4, 16), evidence(2.0)),
            ((5, 15), evidence(2.5)),
            ((6, 14), evidence(3.0)),
        ]);
        assert!(path_consistent(&rows, (5, 15), -1));
    }

    #[test]
    fn verified_snapshot_may_be_a_ledger_subset_but_not_add_unknown_pairs() {
        let owners = BTreeMap::from([((1, 2), (3, 4)), ((5, 6), (7, 8))]);
        assert!(validate_snapshot_subset(&BTreeSet::from([(1, 2)]), &owners).is_ok());
        assert!(validate_snapshot_subset(&BTreeSet::from([(1, 3)]), &owners).is_err());
    }

    #[test]
    fn component_bridge_ledger_is_cycle_gate_compatible() {
        let path = std::env::temp_dir().join(format!(
            "visloc-component-cycle-ledger-{}",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "# visloc-learned-rig-additions-v1\n\
             # admission_policy component-bridge-v4\n\
             # image_pair image_i image_j rig_query rig_candidate cosine sequence_support\n\
             image_pair 1 2 3 4 0.9 3\n",
        )
        .unwrap();
        assert_eq!(
            parse_ledger(&path).unwrap(),
            BTreeMap::from([((1, 2), (3, 4))])
        );
        std::fs::remove_file(path).unwrap();
    }
}
