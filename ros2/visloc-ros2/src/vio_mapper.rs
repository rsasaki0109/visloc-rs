//! Opt-in online mapper (loop closure + global BA) for `visloc_vio_node`.
//!
//! Wires Basalt's [`OnlineNfrMapper`] to the node the way
//! `examples/basalt_euroc_online_slam_demo.rs` wires it to the EuRoC
//! reader: the VIO backend sends each mapper packet (`MargData` of the
//! marginalized keyframes, already-sent images stripped) through a bounded
//! channel to a mapper thread. Unlike the demo, a live node must not stall
//! on a slow mapper, so [`MapperSession::offer`] never blocks: when the
//! queue is full the packet is dropped and counted (see the README for the
//! trade-off).
//!
//! The live output is REP-105's `map -> odom` correction. After each packet
//! the mapper thread takes the newest keyframe the mapper holds an
//! *optimized* pose for and sets
//!
//! ```text
//! T_map_odom = T_mapper(k) * T_vio(k)^-1
//! ```
//!
//! with `T_vio(k)` the pose the VIO published for that frame. At shutdown
//! [`MapperSession::finish`] runs the mapper's final optimization and
//! applies its keyframe corrections to every VIO frame, interpolated
//! between keyframes as the demo's `trajectory_online.tum` is
//! ([`propagate_interpolated`]).
//!
//! Which mapper poses are optimized is tracked here
//! ([`OptimizedPoseTracker`]): the mapper stores each ingested keyframe's
//! VIO window estimate (in the VIO's world frame, `odom`) and overwrites
//! poses only when a background global BA merges. A keyframe that is still
//! in the VIO window can be re-ingested with its window estimate after a
//! merge, so "optimized" means "differs from the pose its latest packet
//! carried".

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use visloc_basalt::config::BasaltConfig;
use visloc_basalt::mapper::NfrMapperHeadlessConfig;
use visloc_basalt::mapper_correction::{
    keyframe_correction, propagate_interpolated, result_keyframe_poses,
};
use visloc_basalt::mapper_online::{OnlineMapperConfig, OnlineNfrMapper, SentImageFilter};
use visloc_basalt::vio::MargData;
use visloc_basalt::BasaltCalibration;
use visloc_core::geometry::SE3;

use crate::msgs::{
    Header, PoseStamped, TfMessage, Time, Transform, TransformStamped, Vector3 as RosVector3,
};
use crate::params::NodeArgs;
use crate::vio::ros_pose;

/// Default bound on queued mapper packets. Each queued packet holds the raw
/// images of its new keyframes (about 1.4 MB per EuRoC-sized stereo
/// keyframe), so this caps the backlog near 100 MB; the online demo, which
/// may block its VIO, uses 256.
pub const DEFAULT_QUEUE_CAPACITY: usize = 64;

/// The mapper's settings from the node's parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct MapperParams {
    /// `map_frame`: the frame the corrected outputs are expressed in.
    pub map_frame: String,
    /// `mapper_queue`: packets waiting for the mapper before new ones drop.
    pub queue_capacity: usize,
    /// `mapper_optimize_every_k`: new keyframes between background global
    /// BA passes (a loop closure can start one earlier).
    pub optimize_every_k: usize,
    /// `mapper_trajectory_out`: TUM file for the final corrected trajectory.
    pub trajectory_out: Option<PathBuf>,
}

impl MapperParams {
    /// Reads the mapper parameters; `None` unless `mapper` is true. All of
    /// them are read either way, so they never count as unknown.
    pub fn from_args(args: &NodeArgs) -> Result<Option<Self>, String> {
        let enabled = args.flag("mapper", false)?;
        let map_frame = args.string_or("map_frame", "map");
        let queue_capacity = args.parsed_or("mapper_queue", DEFAULT_QUEUE_CAPACITY)?;
        let optimize_every_k = args.parsed_or(
            "mapper_optimize_every_k",
            OnlineMapperConfig::default().optimize_every_k,
        )?;
        let trajectory_out = args
            .string("mapper_trajectory_out")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);
        if !enabled {
            return Ok(None);
        }
        if map_frame.is_empty() {
            return Err("parameter `map_frame` must not be empty".into());
        }
        if queue_capacity == 0 {
            return Err("parameter `mapper_queue` must be positive".into());
        }
        if optimize_every_k == 0 {
            return Err("parameter `mapper_optimize_every_k` must be positive".into());
        }
        Ok(Some(Self {
            map_frame,
            queue_capacity,
            optimize_every_k,
            trajectory_out,
        }))
    }
}

/// Builds the online mapper from the VIO's calibration and Basalt config,
/// with the online demo's default settings (`--num-opt-iter 10`, the
/// [`OnlineMapperConfig`] defaults) except `optimize_every_k`.
pub fn new_online_mapper(
    calibration: &BasaltCalibration,
    config: &BasaltConfig,
    params: &MapperParams,
) -> Result<OnlineNfrMapper, String> {
    let mapper_config = config
        .mapper_config()
        .map_err(|error| format!("mapper config: {error:?}"))?;
    let feature_config = config
        .offline_mapper_config()
        .map_err(|error| format!("mapper feature config: {error:?}"))?;
    let optimize_config = config
        .mapper_global_ba_config()
        .map_err(|error| format!("mapper global BA config: {error:?}"))?;
    Ok(OnlineNfrMapper::new(
        mapper_config,
        calibration.clone(),
        feature_config,
        optimize_config,
        OnlineMapperConfig {
            optimize_every_k: params.optimize_every_k,
            headless: NfrMapperHeadlessConfig {
                num_opt_iter: 10,
                ..NfrMapperHeadlessConfig::default()
            },
            ..OnlineMapperConfig::default()
        },
    ))
}

/// The frame ids whose poses a packet carries (and the mapper stores when
/// it ingests it): its frame poses and frame states.
pub fn packet_pose_ids(packet: &MargData) -> BTreeSet<u64> {
    packet
        .frame_poses
        .iter()
        .map(|pose| pose.frame_id)
        .chain(packet.frame_states.iter().map(|state| state.frame_id))
        .collect()
}

/// Tells optimized mapper poses from the VIO window estimates the mapper
/// stores on ingestion (see the module docs).
#[derive(Debug, Default)]
pub struct OptimizedPoseTracker {
    as_ingested: BTreeMap<u64, SE3>,
}

impl OptimizedPoseTracker {
    /// Call after a packet carrying `ids` was ingested: the mapper's poses
    /// for them are now the packet's own.
    pub fn record_ingested(&mut self, ids: &BTreeSet<u64>, mapper_poses: &BTreeMap<u64, SE3>) {
        for id in ids {
            if let Some(pose) = mapper_poses.get(id) {
                self.as_ingested.insert(*id, pose.clone());
            }
        }
    }

    /// Whether the mapper's `pose` for `id` came from an optimization.
    pub fn is_optimized(&self, id: u64, pose: &SE3) -> bool {
        self.as_ingested
            .get(&id)
            .is_some_and(|ingested| ingested != pose)
    }

    /// `(k, T_map_odom)` from the newest optimized keyframe `k` that the
    /// VIO trajectory also has.
    pub fn newest_correction(
        &self,
        mapper_poses: &BTreeMap<u64, SE3>,
        vio_trajectory: &BTreeMap<u64, (i64, SE3)>,
    ) -> Option<(u64, SE3)> {
        mapper_poses
            .iter()
            .rev()
            .filter(|(id, pose)| self.is_optimized(**id, pose))
            .find_map(|(id, mapper_pose)| {
                vio_trajectory
                    .get(id)
                    .map(|(_, vio_pose)| (*id, keyframe_correction(mapper_pose, vio_pose)))
            })
    }
}

/// The mapper's live state, for the publisher and the status line.
#[derive(Clone, Debug)]
pub struct MapperStatus {
    /// `T_map_odom`; identity until the first optimized keyframe.
    pub map_from_odom: SE3,
    /// The keyframe `map_from_odom` was computed from.
    pub correction_keyframe: Option<u64>,
    pub packets_sent: u64,
    /// Packets dropped because the queue was full (or the mapper stopped).
    pub packets_dropped: u64,
    pub packets_ingested: u64,
    pub ingest_errors: u64,
    /// Background global BA results merged into the map.
    pub optimize_merges: u64,
    pub accepted_loops: usize,
    pub keyframes: usize,
}

impl Default for MapperStatus {
    fn default() -> Self {
        Self {
            map_from_odom: SE3::identity(),
            correction_keyframe: None,
            packets_sent: 0,
            packets_dropped: 0,
            packets_ingested: 0,
            ingest_errors: 0,
            optimize_merges: 0,
            accepted_loops: 0,
            keyframes: 0,
        }
    }
}

impl MapperStatus {
    /// The mapper's part of the node's status line.
    pub fn summary(&self) -> String {
        format!(
            "mapper_sent={} mapper_dropped={} mapper_ingested={} mapper_errors={} \
             keyframes={} merges={} loops={} correction_kf={}",
            self.packets_sent,
            self.packets_dropped,
            self.packets_ingested,
            self.ingest_errors,
            self.keyframes,
            self.optimize_merges,
            self.accepted_loops,
            self.correction_keyframe
                .map_or_else(|| "none".to_owned(), |id| id.to_string()),
        )
    }
}

struct SessionShared {
    status: Mutex<MapperStatus>,
    /// Every VIO frame since the session started: `frame_id -> (stamp, T_odom_imu)`.
    vio_trajectory: Mutex<BTreeMap<u64, (i64, SE3)>>,
    /// Packets sent and not yet received by the mapper thread.
    queued: AtomicUsize,
    /// Set when the session is dropped without [`MapperSession::finish`].
    cancel: AtomicBool,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// The final result of a mapper session.
#[derive(Debug)]
pub struct FinishedMap {
    /// Keyframes in the final optimization.
    pub keyframe_count: usize,
    /// Every VIO frame of the session, `(header stamp ns, T_map_imu)`.
    pub trajectory: Vec<(i64, SE3)>,
}

/// One online mapper run, from the VIO's (re)start to shutdown.
pub struct MapperSession {
    sender: Option<SyncSender<MargData>>,
    sent_images: SentImageFilter,
    shared: Arc<SessionShared>,
    capacity: usize,
    thread: Option<JoinHandle<OnlineNfrMapper>>,
}

impl MapperSession {
    /// Starts the mapper thread.
    pub fn spawn(mapper: OnlineNfrMapper, capacity: usize) -> Result<Self, String> {
        let capacity = capacity.max(1);
        let shared = Arc::new(SessionShared {
            status: Mutex::new(MapperStatus::default()),
            vio_trajectory: Mutex::new(BTreeMap::new()),
            queued: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
        });
        let (sender, receiver) = mpsc::sync_channel::<MargData>(capacity);
        let thread = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("vio-mapper".into())
                .spawn(move || run_mapper(mapper, &receiver, &shared))
                .map_err(|error| format!("spawning the mapper thread: {error}"))?
        };
        Ok(Self {
            sender: Some(sender),
            sent_images: SentImageFilter::new(),
            shared,
            capacity,
            thread: Some(thread),
        })
    }

    /// Records the VIO's estimate of a frame (`T_odom_imu`), stamped with
    /// its header stamp. Call it for every frame, before
    /// [`Self::offer`]ing the frame's packet.
    pub fn record_vio_pose(&self, frame_id: u64, stamp_ns: i64, imu_to_odom: &SE3) {
        lock(&self.shared.vio_trajectory).insert(frame_id, (stamp_ns, imu_to_odom.clone()));
    }

    /// Queues a mapper packet without blocking; returns whether it was
    /// queued. A full queue drops the packet: its keyframes never reach the
    /// map, but the VIO keeps its rate.
    pub fn offer(&mut self, mut packet: MargData) -> bool {
        let queued = match &self.sender {
            Some(_) if self.shared.queued.load(Ordering::SeqCst) >= self.capacity => false,
            Some(sender) => {
                // Strip only packets that are sent: the images of a dropped
                // packet then travel with the next one.
                self.sent_images.strip_sent(&mut packet);
                // Counted before sending: the mapper thread decrements on
                // receipt, so the count never undercounts the channel and a
                // send after the check above has room.
                self.shared.queued.fetch_add(1, Ordering::SeqCst);
                match sender.try_send(packet) {
                    Ok(()) => true,
                    Err(error) => {
                        self.shared.queued.fetch_sub(1, Ordering::SeqCst);
                        if matches!(error, TrySendError::Disconnected(_)) {
                            eprintln!("visloc_vio_node: the mapper thread stopped");
                            self.sender = None;
                        }
                        false
                    }
                }
            }
            None => false,
        };
        let mut status = lock(&self.shared.status);
        if queued {
            status.packets_sent += 1;
        } else {
            status.packets_dropped += 1;
        }
        queued
    }

    /// `T_map_odom` as of the latest ingested packet.
    pub fn map_from_odom(&self) -> SE3 {
        lock(&self.shared.status).map_from_odom.clone()
    }

    pub fn status(&self) -> MapperStatus {
        lock(&self.shared.status).clone()
    }

    /// Lets the mapper ingest the packets still queued, runs its final
    /// global optimization and applies the keyframe corrections to every
    /// VIO frame of the session.
    pub fn finish(mut self) -> Result<FinishedMap, String> {
        drop(self.sender.take());
        let mut mapper = self
            .thread
            .take()
            .ok_or("the mapper thread is gone")?
            .join()
            .map_err(|_| "the mapper thread panicked".to_owned())?;
        if mapper.inner().frame_poses.is_empty() {
            return Err("the mapper received no keyframes".into());
        }
        let report = mapper
            .finalize()
            .map_err(|error| format!("final mapper optimization: {error}"))?;
        let keyframes = result_keyframe_poses(&report.result);
        let vio_trajectory = std::mem::take(&mut *lock(&self.shared.vio_trajectory));
        let trajectory = propagate_interpolated(&vio_trajectory, &keyframes)
            .map_err(|error| format!("propagating the mapper's corrections: {error}"))?;
        Ok(FinishedMap {
            keyframe_count: keyframes.len(),
            trajectory,
        })
    }
}

impl Drop for MapperSession {
    /// A session dropped without [`Self::finish`] (a VIO reset) stops its
    /// thread after the packet in progress; the thread is not joined.
    fn drop(&mut self) {
        self.shared.cancel.store(true, Ordering::SeqCst);
        drop(self.sender.take());
        drop(self.thread.take());
    }
}

/// Mapper-thread body: ingests packets until the channel closes (or the
/// session is cancelled) and keeps the shared correction current.
fn run_mapper(
    mut mapper: OnlineNfrMapper,
    receiver: &Receiver<MargData>,
    shared: &SessionShared,
) -> OnlineNfrMapper {
    // Same scheduler hint as `run_mapper_thread`: the mapper must not win
    // CPU contention against the real-time frontend and estimator threads.
    visloc_basalt::rt_priority::set_current_thread_priority_below_normal_best_effort();
    let mut tracker = OptimizedPoseTracker::default();
    loop {
        let mut packet = match receiver.recv_timeout(Duration::from_millis(500)) {
            Ok(packet) => packet,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if shared.cancel.load(Ordering::SeqCst) {
                    break;
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        shared.queued.fetch_sub(1, Ordering::SeqCst);
        if shared.cancel.load(Ordering::SeqCst) {
            break;
        }
        let ids = packet_pose_ids(&packet);
        // `None`: the production (wall-clock) RANSAC seed, as in the demo.
        let ingested = mapper.ingest_packet(&mut packet, None);
        drop(packet);
        match ingested {
            Ok(report) => {
                let poses = &mapper.inner().frame_poses;
                tracker.record_ingested(&ids, poses);
                let correction = tracker.newest_correction(poses, &lock(&shared.vio_trajectory));
                let mut status = lock(&shared.status);
                status.packets_ingested += 1;
                if report.optimize_merge.is_some() {
                    status.optimize_merges += 1;
                }
                status.accepted_loops = mapper.total_accepted_loops();
                status.keyframes = poses.len();
                if let Some((keyframe, map_from_odom)) = correction {
                    status.map_from_odom = map_from_odom;
                    status.correction_keyframe = Some(keyframe);
                }
            }
            Err(error) => {
                let errors = {
                    let mut status = lock(&shared.status);
                    status.ingest_errors += 1;
                    status.ingest_errors
                };
                if errors <= 5 || errors.is_power_of_two() {
                    eprintln!("visloc_vio_node: mapper packet rejected ({errors} so far): {error}");
                }
            }
        }
    }
    mapper
}

/// `/tf` transform `map_frame -> odom_frame`.
pub fn map_to_odom_transform(
    map_from_odom: &SE3,
    stamp_ns: i64,
    map_frame: &str,
    odom_frame: &str,
) -> TransformStamped {
    let pose = ros_pose(map_from_odom);
    TransformStamped {
        header: Header::new(Time::from_nanos(stamp_ns), map_frame),
        child_frame_id: odom_frame.to_owned(),
        transform: Transform {
            translation: RosVector3 {
                x: pose.position.x,
                y: pose.position.y,
                z: pose.position.z,
            },
            rotation: pose.orientation,
        },
    }
}

/// `/tf` message carrying only [`map_to_odom_transform`].
pub fn map_to_odom_tf(
    map_from_odom: &SE3,
    stamp_ns: i64,
    map_frame: &str,
    odom_frame: &str,
) -> TfMessage {
    TfMessage {
        transforms: vec![map_to_odom_transform(
            map_from_odom,
            stamp_ns,
            map_frame,
            odom_frame,
        )],
    }
}

/// A pose in the odometry frame, moved into the map frame.
pub fn map_pose_msg(
    map_from_odom: &SE3,
    pose_in_odom: &SE3,
    stamp_ns: i64,
    map_frame: &str,
) -> PoseStamped {
    PoseStamped {
        header: Header::new(Time::from_nanos(stamp_ns), map_frame),
        pose: ros_pose(&map_from_odom.compose(pose_in_odom)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    fn se3(x: f64, y: f64, yaw_deg: f64) -> SE3 {
        SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, yaw_deg.to_radians()),
            Vector3::new(x, y, 0.0),
        )
    }

    fn close(a: &SE3, b: &SE3) -> bool {
        (a.translation - b.translation).norm() < 1e-12 && a.rotation.angle_to(&b.rotation) < 1e-12
    }

    #[test]
    fn mapper_params_default_off_and_parse() {
        let args = NodeArgs::parse(["--calibration", "c.json"]).unwrap();
        assert_eq!(MapperParams::from_args(&args).unwrap(), None);
        // Read even when off, so they are not reported as unknown.
        let args = NodeArgs::parse(["--mapper-queue", "8", "--map-frame", "world"]).unwrap();
        assert_eq!(MapperParams::from_args(&args).unwrap(), None);
        assert!(args.finish().is_ok());

        let args = NodeArgs::parse(["--ros-args", "-p", "mapper:=true"]).unwrap();
        assert_eq!(
            MapperParams::from_args(&args).unwrap(),
            Some(MapperParams {
                map_frame: "map".into(),
                queue_capacity: DEFAULT_QUEUE_CAPACITY,
                optimize_every_k: OnlineMapperConfig::default().optimize_every_k,
                trajectory_out: None,
            })
        );
        let args = NodeArgs::parse([
            "--ros-args",
            "-p",
            "mapper:=true",
            "-p",
            "map_frame:=world",
            "-p",
            "mapper_queue:=16",
            "-p",
            "mapper_optimize_every_k:=20",
            "-p",
            "mapper_trajectory_out:=/tmp/t.tum",
        ])
        .unwrap();
        let params = MapperParams::from_args(&args).unwrap().unwrap();
        assert_eq!(params.map_frame, "world");
        assert_eq!(params.queue_capacity, 16);
        assert_eq!(params.optimize_every_k, 20);
        assert_eq!(params.trajectory_out, Some(PathBuf::from("/tmp/t.tum")));
        assert!(args.finish().is_ok());

        let bad: [&[&str]; 4] = [
            &["--mapper", "--mapper-queue", "0"],
            &["--mapper", "--mapper-optimize-every-k", "0"],
            &["--mapper", "--map-frame", ""],
            &["--mapper", "maybe"],
        ];
        for bad in bad {
            let args = NodeArgs::parse(bad.iter().copied()).unwrap();
            assert!(MapperParams::from_args(&args).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn correction_moves_odom_poses_onto_the_map() {
        // The VIO drifted: it puts keyframe 7 at (10, 0) heading +x, the
        // optimized map at (10, 1) heading +10 deg.
        let vio_k = se3(10.0, 0.0, 0.0);
        let map_k = se3(10.0, 1.0, 10.0);
        let mapper = BTreeMap::from([(3, se3(3.0, 0.0, 0.0)), (7, map_k.clone())]);
        let vio = BTreeMap::from([(3, (30, se3(3.0, 0.0, 0.0))), (7, (70, vio_k.clone()))]);
        let mut tracker = OptimizedPoseTracker::default();
        // As ingested, nothing is optimized yet.
        tracker.record_ingested(
            &BTreeSet::from([3, 7]),
            &BTreeMap::from([(3, se3(3.0, 0.0, 0.0)), (7, vio_k.clone())]),
        );
        assert!(!tracker.is_optimized(3, &mapper[&3]));
        assert!(tracker.is_optimized(7, &mapper[&7]));
        let (keyframe, map_from_odom) = tracker.newest_correction(&mapper, &vio).unwrap();
        assert_eq!(keyframe, 7);
        assert!(close(&map_from_odom.compose(&vio_k), &map_k));
        // A later odom pose keeps its VIO motion relative to keyframe 7.
        let later = vio_k.compose(&se3(1.0, 0.0, 0.0));
        let msg = map_pose_msg(&map_from_odom, &later, 80, "map");
        assert_eq!(msg.header.frame_id, "map");
        let expected = map_k.compose(&se3(1.0, 0.0, 0.0));
        assert!((msg.pose.position.x - expected.translation.x).abs() < 1e-12);
        assert!((msg.pose.position.y - expected.translation.y).abs() < 1e-12);
        let tf = map_to_odom_tf(&map_from_odom, 80, "map", "odom");
        assert_eq!(tf.transforms[0].header.frame_id, "map");
        assert_eq!(tf.transforms[0].child_frame_id, "odom");
        assert_eq!(tf.transforms[0].header.stamp, Time::from_nanos(80));
    }

    #[test]
    fn re_ingested_and_unknown_poses_do_not_count_as_optimized() {
        let ingested = BTreeMap::from([
            (1, se3(1.0, 0.0, 0.0)),
            (2, se3(2.0, 0.0, 0.0)),
            (3, se3(3.0, 0.0, 0.0)),
        ]);
        let mut tracker = OptimizedPoseTracker::default();
        tracker.record_ingested(&BTreeSet::from([1, 2, 3]), &ingested);
        // A global BA moved 1 and 2; then a packet re-ingested 2's window
        // estimate. 4 was never recorded.
        let mut mapper = ingested.clone();
        mapper.insert(1, se3(1.0, 0.5, 0.0));
        mapper.insert(2, se3(2.0, 0.5, 0.0));
        mapper.insert(4, se3(4.0, 0.5, 0.0));
        let reingested = BTreeMap::from([(2, se3(2.1, 0.0, 0.0))]);
        mapper.insert(2, reingested[&2].clone());
        tracker.record_ingested(&BTreeSet::from([2, 99]), &mapper);
        let vio: BTreeMap<u64, (i64, SE3)> = (1..=4)
            .map(|id| (id, (id as i64, se3(id as f64, 0.0, 0.0))))
            .collect();
        let (keyframe, map_from_odom) = tracker.newest_correction(&mapper, &vio).unwrap();
        assert_eq!(keyframe, 1);
        assert!(close(&map_from_odom, &se3(0.0, 0.5, 0.0)));
        // Without the VIO pose of the optimized keyframe there is none.
        assert!(tracker
            .newest_correction(&mapper, &BTreeMap::from([(2, vio[&2].clone())]))
            .is_none());
    }

    #[test]
    fn status_summary_names_the_counters() {
        let status = MapperStatus {
            packets_dropped: 2,
            correction_keyframe: Some(42),
            ..MapperStatus::default()
        };
        let summary = status.summary();
        assert!(summary.contains("mapper_dropped=2"), "{summary}");
        assert!(summary.contains("correction_kf=42"), "{summary}");
        assert!(MapperStatus::default()
            .summary()
            .contains("correction_kf=none"));
    }
}
