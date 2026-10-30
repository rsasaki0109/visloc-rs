//! Thin DDS shell over `ros2-client` (RustDDS): node creation, QoS
//! profiles, typed publishers/subscriptions and receive threads.
//!
//! All algorithmic behavior lives in the transport-agnostic modules; this
//! file only moves messages between DDS and those cores.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use futures::StreamExt;
use ros2_client::{
    qos::{Durability, History, WhenFull},
    Context, ContextOptions, MessageTypeName, Name, Node, NodeName, NodeOptions, Publisher,
    QosProfile, Subscription,
};

use crate::msgs::{self, RosMessageType};
use crate::tf_tree::StaticTransforms;

macro_rules! impl_message {
    ($($ty:ty),* $(,)?) => {
        $(impl ros2_client::Message for $ty {})*
    };
}

impl_message!(
    msgs::Image,
    msgs::PointCloud2,
    msgs::CameraInfo,
    msgs::Imu,
    msgs::PoseStamped,
    msgs::PoseWithCovarianceStamped,
    msgs::Odometry,
    msgs::Path,
    msgs::TfMessage,
    msgs::DiagnosticArray,
    msgs::Int32,
);

/// Sensor-input QoS: best-effort, volatile, keep-last `depth`.
///
/// A best-effort reader matches both best-effort (`SensorDataQoS`, most
/// camera drivers) and reliable publishers, so this is the most compatible
/// choice for inputs.
pub fn sensor_qos(depth: usize) -> QosProfile {
    QosProfile::subscription_default()
        .reliability_best_effort()
        .durability(Durability::Volatile)
        .history(History::KeepLast {
            depth: depth.max(1),
        })
}

/// Output QoS: reliable, volatile, keep-last `depth` (the rclcpp default
/// `QoS(10)`). A reliable writer matches reliable and best-effort readers,
/// e.g. RViz with either setting.
pub fn output_qos(depth: usize) -> QosProfile {
    QosProfile::publisher_default()
        .reliability_reliable(WhenFull::Wait(std::time::Duration::from_millis(20)))
        .durability(Durability::Volatile)
        .history(History::KeepLast {
            depth: depth.max(1),
        })
}

/// `/tf_static` subscription QoS, as `tf2_ros::StaticListenerQoS`: reliable,
/// transient-local, keep-last 100. Transient-local durability is what
/// delivers the transforms a `robot_state_publisher` or
/// `static_transform_publisher` sent before this node started.
pub fn static_tf_qos() -> QosProfile {
    QosProfile::subscription_default()
        .reliability_reliable(WhenFull::Wait(std::time::Duration::from_millis(20)))
        .durability(Durability::TransientLocal)
        .history(History::KeepLast { depth: 100 })
}

/// Latched output QoS: reliable, transient-local, keep-last `depth`. A late
/// subscriber that asks for transient-local durability (RViz can) still
/// receives what was published before it joined, e.g. a static map cloud.
pub fn latched_qos(depth: usize) -> QosProfile {
    QosProfile::publisher_default()
        .reliability_reliable(WhenFull::Wait(std::time::Duration::from_millis(20)))
        .durability(Durability::TransientLocal)
        .history(History::KeepLast {
            depth: depth.max(1),
        })
}

/// `/ns/name` for logging (`/name` in the root namespace).
pub fn qualified_name(namespace: &str, name: &str) -> String {
    format!("{}/{name}", namespace.trim_end_matches('/'))
}

/// The absolute name of topic `name` for a node in `namespace`: absolute
/// names are kept, relative ones are put under the namespace (`odom` is
/// `/odom` in the root namespace, `/robot1/odom` in `/robot1`).
///
/// `ros2-client` 0.11 resolves a relative name by prefixing the namespace
/// and a separator, which in the root namespace yields the DDS topic
/// `rt//odom` instead of ROS 2's `rt/odom`; such a topic matches only other
/// `ros2-client` nodes. Names are therefore resolved here and passed on
/// absolute.
pub fn resolve_topic_name(namespace: &str, name: &str) -> String {
    if name.starts_with('/') {
        name.to_owned()
    } else {
        qualified_name(namespace, name)
    }
}

/// A ROS 2 node over RustDDS with a background spinner thread.
pub struct RosNode {
    pub node: Node,
    pub context: Context,
    namespace: String,
    _spinner: Option<JoinHandle<()>>,
}

impl RosNode {
    /// Creates a context on `domain_id` and a node `namespace/name`.
    pub fn new(domain_id: u16, namespace: &str, name: &str) -> Result<Self, String> {
        let context = Context::with_options(ContextOptions::new().domain_id(domain_id))
            .map_err(|error| format!("DDS context (domain {domain_id}): {error:?}"))?;
        let namespace = if namespace.is_empty() { "/" } else { namespace };
        let node_name = NodeName::new(namespace, name)
            .map_err(|error| format!("node name {namespace}/{name}: {error:?}"))?;
        let mut node = context
            .new_node(node_name, NodeOptions::new().enable_rosout(true))
            .map_err(|error| format!("create node: {error:?}"))?;
        let spinner = node
            .spinner()
            .map_err(|error| format!("node spinner: {error:?}"))?;
        let handle = std::thread::Builder::new()
            .name("ros2-spinner".into())
            .spawn(move || {
                if let Err(error) = futures::executor::block_on(spinner.spin()) {
                    eprintln!("ros2 spinner stopped: {error:?}");
                }
            })
            .map_err(|error| format!("spawn spinner: {error}"))?;
        Ok(Self {
            node,
            context,
            namespace: namespace.to_owned(),
            _spinner: Some(handle),
        })
    }

    fn topic<M: RosMessageType>(
        &mut self,
        topic_name: &str,
        qos: &QosProfile,
    ) -> Result<ros2_client::dds::rustdds::Topic, String> {
        let resolved = resolve_topic_name(&self.namespace, topic_name);
        let name =
            Name::parse(&resolved).map_err(|error| format!("topic `{resolved}`: {error:?}"))?;
        self.node
            .create_topic(&name, MessageTypeName::new(M::PACKAGE, M::TYPE), qos)
            .map_err(|error| format!("topic `{topic_name}`: {error:?}"))
    }

    pub fn publisher<M>(
        &mut self,
        topic_name: &str,
        qos: QosProfile,
    ) -> Result<Publisher<M>, String>
    where
        M: RosMessageType + ros2_client::Message,
    {
        let topic = self.topic::<M>(topic_name, &qos)?;
        self.node
            .create_publisher(&topic, Some(qos))
            .map_err(|error| format!("publisher `{topic_name}`: {error:?}"))
    }

    pub fn subscription<M>(
        &mut self,
        topic_name: &str,
        qos: QosProfile,
    ) -> Result<Subscription<M>, String>
    where
        M: RosMessageType + ros2_client::Message + 'static,
    {
        let topic = self.topic::<M>(topic_name, &qos)?;
        self.node
            .create_subscription(&topic, Some(qos))
            .map_err(|error| format!("subscription `{topic_name}`: {error:?}"))
    }
}

/// Spawns a thread that drains `subscription` and calls `on_message` for
/// every received sample (receive errors are logged and skipped).
pub fn spawn_receiver<M, F>(
    thread_name: &str,
    subscription: Subscription<M>,
    mut on_message: F,
) -> Result<JoinHandle<()>, String>
where
    M: ros2_client::Message + Send + 'static,
    F: FnMut(M) + Send + 'static,
{
    let name = thread_name.to_owned();
    std::thread::Builder::new()
        .name(name.clone())
        .spawn(move || {
            futures::executor::block_on(async {
                let stream = subscription.async_stream();
                futures::pin_mut!(stream);
                while let Some(item) = stream.next().await {
                    match item {
                        Ok((message, _info)) => on_message(message),
                        Err(error) => eprintln!("{name}: receive error: {error:?}"),
                    }
                }
            });
        })
        .map_err(|error| format!("spawn {thread_name}: {error}"))
}

/// Publishes, logging (not propagating) failures: a slow or vanished
/// subscriber must never stop the estimator.
pub fn publish_logged<M>(publisher: &Publisher<M>, message: M, what: &str)
where
    M: ros2_client::Message + std::fmt::Debug,
{
    if let Err(error) = publisher.publish(message) {
        eprintln!("publish {what} failed: {error:?}");
    }
}

/// Receives `/tf_static` into shared [`StaticTransforms`], and subscribes
/// again while a transform the node needs is missing.
///
/// A RustDDS reader sometimes never receives a transient-local writer's
/// samples: in a test where `static_transform_publisher`s and readers all
/// restarted every few seconds, 4 of 200 readers had nothing after 30 s,
/// while the others had both transforms within 1.6 s. A new reader in the
/// same node receives them; with one 5 s after the first, all 300 readers of
/// a second test had both.
pub struct StaticTfListener {
    topic: String,
    transforms: Arc<Mutex<StaticTransforms>>,
    retries: RetrySchedule,
}

impl StaticTfListener {
    pub fn new(ros: &mut RosNode, topic: &str) -> Result<Self, String> {
        let listener = Self {
            topic: topic.to_owned(),
            transforms: Arc::new(Mutex::new(StaticTransforms::new())),
            retries: RetrySchedule::new(Instant::now()),
        };
        listener.subscribe(ros)?;
        Ok(listener)
    }

    /// The transforms received so far, shared with the receive threads.
    pub fn transforms(&self) -> &Arc<Mutex<StaticTransforms>> {
        &self.transforms
    }

    fn subscribe(&self, ros: &mut RosNode) -> Result<(), String> {
        let transforms = Arc::clone(&self.transforms);
        let subscription = ros.subscription::<msgs::TfMessage>(&self.topic, static_tf_qos())?;
        spawn_receiver("tf-static-rx", subscription, move |msg: msgs::TfMessage| {
            transforms
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(&msg);
        })?;
        Ok(())
    }

    /// Call when a transform the node needs is missing: subscribes again
    /// when [`RetrySchedule`] says so, and returns whether it did. Each
    /// subscription keeps its reader and thread, so the retries are few.
    pub fn missing(&mut self, ros: &mut RosNode) -> Result<bool, String> {
        if !self.retries.due(Instant::now()) {
            return Ok(false);
        }
        self.subscribe(ros)?;
        Ok(true)
    }
}

/// When to subscribe again: 5 s after the first subscription, then after
/// twice the previous wait, at most six times (5 s to 5 min after the start).
#[derive(Clone, Debug)]
pub struct RetrySchedule {
    last: Instant,
    wait: Duration,
    left: u32,
}

impl RetrySchedule {
    pub const FIRST_WAIT: Duration = Duration::from_secs(5);
    pub const MAX_RETRIES: u32 = 6;

    pub const fn new(subscribed: Instant) -> Self {
        Self {
            last: subscribed,
            wait: Self::FIRST_WAIT,
            left: Self::MAX_RETRIES,
        }
    }

    /// Whether to subscribe again at `now`; if so, the next wait doubles.
    pub fn due(&mut self, now: Instant) -> bool {
        if self.left == 0 || now.saturating_duration_since(self.last) < self.wait {
            return false;
        }
        self.left -= 1;
        self.last = now;
        self.wait *= 2;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retries_back_off_and_stop() {
        let start = Instant::now();
        let mut schedule = RetrySchedule::new(start);
        let at = |s: u64| start + Duration::from_secs(s);
        assert!(!schedule.due(at(4)));
        assert!(schedule.due(at(5)));
        assert!(!schedule.due(at(14)));
        assert!(schedule.due(at(15)));
        let fired: Vec<u64> = (16..400).filter(|&s| schedule.due(at(s))).collect();
        assert_eq!(fired, [35, 75, 155, 315]);
        assert!(!schedule.due(at(10_000)));
    }

    #[test]
    fn relative_topics_resolve_under_the_namespace() {
        assert_eq!(resolve_topic_name("/", "odom"), "/odom");
        assert_eq!(resolve_topic_name("/", "left/image_raw"), "/left/image_raw");
        assert_eq!(resolve_topic_name("/robot1", "odom"), "/robot1/odom");
        assert_eq!(resolve_topic_name("/robot1/", "odom"), "/robot1/odom");
        assert_eq!(resolve_topic_name("/robot1", "/tf"), "/tf");
        assert_eq!(
            resolve_topic_name("/", "/cam0/image_raw"),
            "/cam0/image_raw"
        );
    }

    #[test]
    fn resolved_names_map_to_ros2_dds_topic_names() {
        let node = NodeName::new("/", "n").unwrap();
        let robot = NodeName::new("/robot1", "n").unwrap();
        let dds = |namespace: &str, node: &NodeName, name: &str| {
            Name::parse(&resolve_topic_name(namespace, name))
                .unwrap()
                .to_dds_name("rt", node, "")
        };
        assert_eq!(dds("/", &node, "odom"), "rt/odom");
        assert_eq!(dds("/", &node, "/tf"), "rt/tf");
        assert_eq!(dds("/robot1", &robot, "odom"), "rt/robot1/odom");
        // What ros2-client 0.11 makes of the relative name on its own.
        assert_eq!(
            Name::parse("odom").unwrap().to_dds_name("rt", &node, ""),
            "rt//odom"
        );
    }
}
