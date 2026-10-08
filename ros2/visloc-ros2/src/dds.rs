//! Thin DDS shell over `ros2-client` (RustDDS): node creation, QoS
//! profiles, typed publishers/subscriptions and receive threads.
//!
//! All algorithmic behavior lives in the transport-agnostic modules; this
//! file only moves messages between DDS and those cores.

use std::thread::JoinHandle;

use futures::StreamExt;
use ros2_client::{
    qos::{Durability, History, WhenFull},
    Context, ContextOptions, MessageTypeName, Name, Node, NodeName, NodeOptions, Publisher,
    QosProfile, Subscription,
};

use crate::msgs::{self, RosMessageType};

macro_rules! impl_message {
    ($($ty:ty),* $(,)?) => {
        $(impl ros2_client::Message for $ty {})*
    };
}

impl_message!(
    msgs::Image,
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

/// `/ns/name` for logging (`/name` in the root namespace).
pub fn qualified_name(namespace: &str, name: &str) -> String {
    format!("{}/{name}", namespace.trim_end_matches('/'))
}

/// A ROS 2 node over RustDDS with a background spinner thread.
pub struct RosNode {
    pub node: Node,
    pub context: Context,
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
            _spinner: Some(handle),
        })
    }

    fn topic<M: RosMessageType>(
        &mut self,
        topic_name: &str,
        qos: &QosProfile,
    ) -> Result<ros2_client::dds::rustdds::Topic, String> {
        let name =
            Name::parse(topic_name).map_err(|error| format!("topic `{topic_name}`: {error:?}"))?;
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
