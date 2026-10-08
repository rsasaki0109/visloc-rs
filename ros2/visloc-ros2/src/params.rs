//! Command-line parameters with ROS 2-style syntax.
//!
//! Both spellings are accepted and can be mixed:
//!
//! ```text
//! visloc_vio_node --calibration calib.json --config config.json \
//!     --remap left/image_raw=/cam0/image_raw
//! visloc_vio_node --ros-args -p calibration:=calib.json -p config:=config.json \
//!     -r left/image_raw:=/cam0/image_raw -r __ns:=/robot -r __node:=vio
//! ```
//!
//! `--some-name` and `-p some_name:=` address the same parameter (dashes
//! become underscores). A bare `--flag` means `true`. Unknown parameters are
//! an error so typos do not go unnoticed. `ROS_DOMAIN_ID` is read from the
//! environment, as in ROS 2, unless `--domain-id` is given.

use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeArgs {
    params: BTreeMap<String, String>,
    remaps: BTreeMap<String, String>,
    pub node_name: Option<String>,
    pub namespace: Option<String>,
    pub help: bool,
    consumed: std::cell::RefCell<BTreeSet<String>>,
}

fn normalize_key(key: &str) -> String {
    key.trim_start_matches('-').replace('-', "_")
}

fn normalize_topic(topic: &str) -> String {
    topic.trim().to_owned()
}

impl NodeArgs {
    pub fn parse<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let tokens: Vec<String> = args.into_iter().map(Into::into).collect();
        let mut out = Self::default();
        let mut index = 0;
        let mut in_ros_args = false;
        while index < tokens.len() {
            let token = tokens[index].as_str();
            index += 1;
            match token {
                "--ros-args" => {
                    in_ros_args = true;
                    continue;
                }
                "--" => {
                    in_ros_args = false;
                    continue;
                }
                "-h" | "--help" => {
                    out.help = true;
                    continue;
                }
                _ => {}
            }
            if in_ros_args && (token == "-p" || token == "--param") {
                let assignment = tokens
                    .get(index)
                    .ok_or_else(|| format!("{token} needs name:=value"))?;
                index += 1;
                let (key, value) = assignment.split_once(":=").ok_or_else(|| {
                    format!("expected name:=value after {token}, got `{assignment}`")
                })?;
                out.params.insert(normalize_key(key), value.to_owned());
                continue;
            }
            if in_ros_args && (token == "-r" || token == "--remap") {
                let rule = tokens
                    .get(index)
                    .ok_or_else(|| format!("{token} needs from:=to"))?;
                index += 1;
                let (from, to) = rule
                    .split_once(":=")
                    .ok_or_else(|| format!("expected from:=to after {token}, got `{rule}`"))?;
                out.add_remap(from, to);
                continue;
            }
            if in_ros_args {
                return Err(format!("unsupported ROS argument `{token}` (use -p / -r)"));
            }
            let Some(stripped) = token.strip_prefix("--") else {
                return Err(format!("unexpected argument `{token}`"));
            };
            let (key, inline_value) = match stripped.split_once('=') {
                Some((key, value)) => (key, Some(value.to_owned())),
                None => (stripped, None),
            };
            if key == "remap" {
                let rule = match inline_value {
                    Some(value) => value,
                    None => {
                        let value = tokens.get(index).ok_or("--remap needs from=to")?.clone();
                        index += 1;
                        value
                    }
                };
                let (from, to) = rule
                    .split_once(":=")
                    .or_else(|| rule.split_once('='))
                    .ok_or_else(|| format!("expected from=to after --remap, got `{rule}`"))?;
                out.add_remap(from, to);
                continue;
            }
            let value = match inline_value {
                Some(value) => value,
                None => match tokens.get(index) {
                    Some(next) if !next.starts_with("--") => {
                        index += 1;
                        next.clone()
                    }
                    _ => "true".to_owned(),
                },
            };
            out.params.insert(normalize_key(key), value);
        }
        Ok(out)
    }

    fn add_remap(&mut self, from: &str, to: &str) {
        match from {
            "__node" | "__name" => self.node_name = Some(to.to_owned()),
            "__ns" => self.namespace = Some(to.to_owned()),
            _ => {
                self.remaps
                    .insert(normalize_topic(from), normalize_topic(to));
            }
        }
    }

    fn raw(&self, key: &str) -> Option<&str> {
        self.consumed.borrow_mut().insert(key.to_owned());
        self.params.get(key).map(String::as_str)
    }

    pub fn string(&self, key: &str) -> Option<String> {
        self.raw(key).map(str::to_owned)
    }

    pub fn string_or(&self, key: &str, default: &str) -> String {
        self.string(key).unwrap_or_else(|| default.to_owned())
    }

    pub fn required(&self, key: &str) -> Result<String, String> {
        self.string(key).ok_or_else(|| {
            format!(
                "missing required parameter `{key}` (--{})",
                key.replace('_', "-")
            )
        })
    }

    pub fn parsed<T: FromStr>(&self, key: &str) -> Result<Option<T>, String>
    where
        T::Err: std::fmt::Display,
    {
        self.raw(key)
            .map(|value| {
                value
                    .parse::<T>()
                    .map_err(|error| format!("parameter `{key}`=`{value}`: {error}"))
            })
            .transpose()
    }

    pub fn parsed_or<T: FromStr>(&self, key: &str, default: T) -> Result<T, String>
    where
        T::Err: std::fmt::Display,
    {
        Ok(self.parsed(key)?.unwrap_or(default))
    }

    pub fn flag(&self, key: &str, default: bool) -> Result<bool, String> {
        match self.raw(key) {
            None => Ok(default),
            Some("true" | "1" | "yes" | "on" | "True") => Ok(true),
            Some("false" | "0" | "no" | "off" | "False") => Ok(false),
            Some(other) => Err(format!(
                "parameter `{key}` expects a boolean, got `{other}`"
            )),
        }
    }

    /// Applies `from:=to` remapping to a topic name. Matches the name as
    /// written by the node (e.g. `left/image_raw`) with or without a
    /// leading slash.
    pub fn topic(&self, default_name: &str) -> String {
        let bare = default_name.trim_start_matches('/');
        self.remaps
            .get(default_name)
            .or_else(|| self.remaps.get(bare))
            .or_else(|| self.remaps.get(&format!("/{bare}")))
            .cloned()
            .unwrap_or_else(|| default_name.to_owned())
    }

    /// Parameters that were given but never read: typos.
    pub fn unknown_params(&self) -> Vec<String> {
        let consumed = self.consumed.borrow();
        self.params
            .keys()
            .filter(|key| !consumed.contains(*key))
            .cloned()
            .collect()
    }

    /// Errors out on unknown parameters (call after reading all of them).
    pub fn finish(&self) -> Result<(), String> {
        let unknown = self.unknown_params();
        if unknown.is_empty() {
            Ok(())
        } else {
            Err(format!("unknown parameter(s): {}", unknown.join(", ")))
        }
    }

    /// ROS domain id: `--domain-id`, else `ROS_DOMAIN_ID`, else 0.
    pub fn domain_id(&self) -> Result<u16, String> {
        if let Some(id) = self.parsed::<u16>("domain_id")? {
            return Ok(id);
        }
        match std::env::var("ROS_DOMAIN_ID") {
            Ok(value) if !value.trim().is_empty() => value
                .trim()
                .parse()
                .map_err(|error| format!("ROS_DOMAIN_ID=`{value}`: {error}")),
            _ => Ok(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_and_ros_syntax_mix() {
        let args = NodeArgs::parse([
            "--calibration",
            "c.json",
            "--publish-tf",
            "--max-frames=5",
            "--remap",
            "imu=/imu0",
            "--ros-args",
            "-p",
            "config:=cfg.json",
            "-r",
            "left/image_raw:=/cam0/image_raw",
            "-r",
            "__ns:=/robot",
            "-r",
            "__node:=vio",
        ])
        .unwrap();
        assert_eq!(args.string("calibration").as_deref(), Some("c.json"));
        assert_eq!(args.string("config").as_deref(), Some("cfg.json"));
        assert!(args.flag("publish_tf", false).unwrap());
        assert_eq!(args.parsed::<usize>("max_frames").unwrap(), Some(5));
        assert_eq!(args.topic("left/image_raw"), "/cam0/image_raw");
        assert_eq!(args.topic("imu"), "/imu0");
        assert_eq!(args.topic("right/image_raw"), "right/image_raw");
        assert_eq!(args.namespace.as_deref(), Some("/robot"));
        assert_eq!(args.node_name.as_deref(), Some("vio"));
        assert!(args.finish().is_ok());
    }

    #[test]
    fn unknown_and_malformed_params_are_errors() {
        let args = NodeArgs::parse(["--calibraton", "x"]).unwrap();
        assert!(args.string("calibration").is_none());
        assert!(args.finish().unwrap_err().contains("calibraton"));
        assert!(NodeArgs::parse(["--ros-args", "-p", "novalue"]).is_err());
        assert!(NodeArgs::parse(["positional"]).is_err());
        let args = NodeArgs::parse(["--n", "abc"]).unwrap();
        assert!(args.parsed::<u32>("n").is_err());
        let args = NodeArgs::parse(["--b", "maybe"]).unwrap();
        assert!(args.flag("b", false).is_err());
    }

    #[test]
    fn domain_id_flag_wins() {
        let args = NodeArgs::parse(["--domain-id", "42"]).unwrap();
        assert_eq!(args.domain_id().unwrap(), 42);
    }
}
