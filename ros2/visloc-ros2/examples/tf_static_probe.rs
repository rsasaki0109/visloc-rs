//! Subscribes to `/tf_static` with the nodes' QoS and reports what arrives:
//! `tf_static_probe [timeout_s] [expected transforms] [retry: none|reader|node] [retry_s]`,
//! exit 0 once the expected number of transforms arrived, 1 on timeout.
use std::collections::BTreeSet;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use visloc_ros2::dds::{spawn_receiver, static_tf_qos, RosNode};
use visloc_ros2::msgs::TfMessage;

fn subscribe(ros: &mut RosNode, tx: mpsc::Sender<TfMessage>) {
    let subscription = ros
        .subscription::<TfMessage>("/tf_static", static_tf_qos())
        .expect("subscription");
    spawn_receiver("tf-static-rx", subscription, move |msg: TfMessage| {
        let _ = tx.send(msg);
    })
    .expect("receiver");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let timeout_s: f64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(10.0);
    let expected: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(2);
    let retry = args.next().unwrap_or_else(|| "none".into());
    let retry_s: f64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(5.0);
    let domain_id: u16 = std::env::var("ROS_DOMAIN_ID")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let start = Instant::now();
    let (tx, rx) = mpsc::channel();
    let mut nodes = vec![RosNode::new(domain_id, "/", "tf_static_probe").expect("node")];
    subscribe(nodes.last_mut().unwrap(), tx.clone());
    let mut frames = BTreeSet::new();
    let mut last_try = Instant::now();
    let mut retries = 0;
    while start.elapsed() < Duration::from_secs_f64(timeout_s) && frames.len() < expected {
        if let Ok(msg) = rx.recv_timeout(Duration::from_millis(20)) {
            for t in &msg.transforms {
                frames.insert(format!("{} -> {}", t.header.frame_id, t.child_frame_id));
            }
            println!("{:.3} s: {:?}", start.elapsed().as_secs_f64(), frames);
        }
        if retry != "none"
            && last_try.elapsed() > Duration::from_secs_f64(retry_s)
            && frames.len() < expected
        {
            retries += 1;
            last_try = Instant::now();
            println!(
                "{:.3} s: retry {retries} ({retry})",
                start.elapsed().as_secs_f64()
            );
            if retry == "node" {
                nodes.push(RosNode::new(domain_id, "/", "tf_static_probe").expect("node"));
            }
            subscribe(nodes.last_mut().unwrap(), tx.clone());
        }
    }
    let ok = frames.len() >= expected;
    println!(
        "{} {} of {expected} after {:.3} s, {retries} retries",
        if ok { "OK" } else { "MISSING" },
        frames.len(),
        start.elapsed().as_secs_f64()
    );
    std::process::exit(if ok { 0 } else { 1 });
}
