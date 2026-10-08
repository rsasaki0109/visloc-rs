//! Hand-written ROS 2 message definitions (serde, CDR-compatible).
//!
//! `ros2-client` ships no `sensor_msgs` / `geometry_msgs` / `nav_msgs`
//! bindings (it has no IDL code generator); applications define the structs
//! they need with `serde` derives and DDS matches them by type name. The
//! structs here mirror the ROS 2 IDL **field order and field types exactly**
//! because CDR is a positional encoding: field names are irrelevant on the
//! wire, but every field, its width, its signedness, fixed-array lengths and
//! sequence-vs-array distinctions must match `rosidl`'s generated types.
//!
//! The layouts match ROS 2 Humble, Iron, Jazzy and Kilted (none of these
//! message definitions changed across those distributions).
//!
//! Large byte buffers (`sensor_msgs/Image::data`) use [`serde_bytes_vec`] so
//! the CDR serializer writes them as one `u32` length prefix plus a
//! contiguous `memcpy` instead of one serde call per pixel. The bytes on the
//! wire are identical either way (`sequence<uint8>`).

use serde::{Deserialize, Serialize};

/// The ROS 2 type name of a message, split into package and type.
pub trait RosMessageType {
    /// ROS package, e.g. `sensor_msgs`.
    const PACKAGE: &'static str;
    /// ROS type name inside the package's `msg` namespace, e.g. `Image`.
    const TYPE: &'static str;
}

macro_rules! ros_type {
    ($ty:ty, $package:literal, $name:literal) => {
        impl RosMessageType for $ty {
            const PACKAGE: &'static str = $package;
            const TYPE: &'static str = $name;
        }
    };
}

/// `builtin_interfaces/Time`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Time {
    pub sec: i32,
    pub nanosec: u32,
}

impl Time {
    /// Converts a signed nanosecond count (since the Unix epoch, or since
    /// any other clock origin) into the ROS wire form, whose fractional part
    /// is always in `[0, 1e9)`. Seconds saturate at the `i32` range.
    pub fn from_nanos(nanos: i64) -> Self {
        let sec = nanos.div_euclid(1_000_000_000);
        let nanosec = nanos.rem_euclid(1_000_000_000) as u32;
        Self {
            sec: sec.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
            nanosec,
        }
    }

    /// Signed nanosecond count represented by this stamp.
    pub fn to_nanos(self) -> i64 {
        i64::from(self.sec) * 1_000_000_000 + i64::from(self.nanosec)
    }
}

/// `std_msgs/Header`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    pub stamp: Time,
    pub frame_id: String,
}
ros_type!(Header, "std_msgs", "Header");

impl Header {
    pub fn new(stamp: Time, frame_id: impl Into<String>) -> Self {
        Self {
            stamp,
            frame_id: frame_id.into(),
        }
    }
}

/// `std_msgs/Int32`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Int32 {
    pub data: i32,
}
ros_type!(Int32, "std_msgs", "Int32");

/// `sensor_msgs/Image`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Image {
    pub header: Header,
    pub height: u32,
    pub width: u32,
    pub encoding: String,
    pub is_bigendian: u8,
    pub step: u32,
    #[serde(with = "serde_bytes_vec")]
    pub data: Vec<u8>,
}
ros_type!(Image, "sensor_msgs", "Image");

/// `sensor_msgs/RegionOfInterest`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionOfInterest {
    pub x_offset: u32,
    pub y_offset: u32,
    pub height: u32,
    pub width: u32,
    pub do_rectify: bool,
}

/// `sensor_msgs/CameraInfo` (ROS 2 lower-case field names `d`, `k`, `r`, `p`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CameraInfo {
    pub header: Header,
    pub height: u32,
    pub width: u32,
    pub distortion_model: String,
    pub d: Vec<f64>,
    pub k: [f64; 9],
    pub r: [f64; 9],
    pub p: [f64; 12],
    pub binning_x: u32,
    pub binning_y: u32,
    pub roi: RegionOfInterest,
}
ros_type!(CameraInfo, "sensor_msgs", "CameraInfo");

/// `geometry_msgs/Vector3`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// `geometry_msgs/Point`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// `geometry_msgs/Quaternion` (ROS default is the identity `w = 1`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

/// `sensor_msgs/Imu`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Imu {
    pub header: Header,
    pub orientation: Quaternion,
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: Vector3,
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: Vector3,
    pub linear_acceleration_covariance: [f64; 9],
}
ros_type!(Imu, "sensor_msgs", "Imu");

/// `geometry_msgs/Pose`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub position: Point,
    pub orientation: Quaternion,
}

/// `geometry_msgs/PoseStamped`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PoseStamped {
    pub header: Header,
    pub pose: Pose,
}
ros_type!(PoseStamped, "geometry_msgs", "PoseStamped");

/// A `float64[36]` row-major 6x6 covariance (x, y, z, rot x, rot y, rot z).
///
/// `serde` only derives fixed arrays up to 32 elements, so this newtype
/// (de)serializes itself as a 36-element tuple: CDR writes fixed arrays with
/// no length prefix, exactly like `rosidl`'s `double[36]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Covariance6(pub [f64; 36]);

impl Default for Covariance6 {
    fn default() -> Self {
        Self([0.0; 36])
    }
}

impl Covariance6 {
    /// Diagonal covariance from three positional and three rotational
    /// variances.
    pub fn diagonal(position_variance: f64, rotation_variance: f64) -> Self {
        let mut values = [0.0; 36];
        for axis in 0..3 {
            values[axis * 7] = position_variance;
            values[(axis + 3) * 7] = rotation_variance;
        }
        Self(values)
    }
}

impl Serialize for Covariance6 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let mut tuple = serializer.serialize_tuple(36)?;
        for value in &self.0 {
            tuple.serialize_element(value)?;
        }
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for Covariance6 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct CovarianceVisitor;
        impl<'de> serde::de::Visitor<'de> for CovarianceVisitor {
            type Value = Covariance6;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("36 float64 values")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = [0.0; 36];
                for (index, slot) in values.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(index, &self))?;
                }
                Ok(Covariance6(values))
            }
        }
        deserializer.deserialize_tuple(36, CovarianceVisitor)
    }
}

/// `geometry_msgs/PoseWithCovariance`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PoseWithCovariance {
    pub pose: Pose,
    pub covariance: Covariance6,
}

/// `geometry_msgs/PoseWithCovarianceStamped`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PoseWithCovarianceStamped {
    pub header: Header,
    pub pose: PoseWithCovariance,
}
ros_type!(
    PoseWithCovarianceStamped,
    "geometry_msgs",
    "PoseWithCovarianceStamped"
);

/// `geometry_msgs/Twist`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Twist {
    pub linear: Vector3,
    pub angular: Vector3,
}

/// `geometry_msgs/TwistWithCovariance`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TwistWithCovariance {
    pub twist: Twist,
    pub covariance: Covariance6,
}

/// `geometry_msgs/Transform`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vector3,
    pub rotation: Quaternion,
}

/// `geometry_msgs/TransformStamped`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TransformStamped {
    pub header: Header,
    pub child_frame_id: String,
    pub transform: Transform,
}

/// `tf2_msgs/TFMessage`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TfMessage {
    pub transforms: Vec<TransformStamped>,
}
ros_type!(TfMessage, "tf2_msgs", "TFMessage");

/// `nav_msgs/Odometry`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Odometry {
    pub header: Header,
    pub child_frame_id: String,
    pub pose: PoseWithCovariance,
    pub twist: TwistWithCovariance,
}
ros_type!(Odometry, "nav_msgs", "Odometry");

/// `nav_msgs/Path`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Path {
    pub header: Header,
    pub poses: Vec<PoseStamped>,
}
ros_type!(Path, "nav_msgs", "Path");

/// `diagnostic_msgs/KeyValue`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}

/// `diagnostic_msgs/DiagnosticStatus`. `level` is IDL `byte` (one octet).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticStatus {
    pub level: u8,
    pub name: String,
    pub message: String,
    pub hardware_id: String,
    pub values: Vec<KeyValue>,
}

impl DiagnosticStatus {
    pub const OK: u8 = 0;
    pub const WARN: u8 = 1;
    pub const ERROR: u8 = 2;
    pub const STALE: u8 = 3;
}

/// `diagnostic_msgs/DiagnosticArray`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticArray {
    pub header: Header,
    pub status: Vec<DiagnosticStatus>,
}
ros_type!(DiagnosticArray, "diagnostic_msgs", "DiagnosticArray");

/// `#[serde(with = ...)]` helper for `sequence<uint8>` fields.
///
/// Serializes through `serialize_bytes` (one length prefix + one bulk copy
/// in the CDR serializer) and accepts either a byte buffer or a generic
/// sequence when deserializing, so it also works with self-describing
/// formats used in tests.
pub mod serde_bytes_vec {
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(bytes)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        struct BytesVisitor;
        impl<'de> serde::de::Visitor<'de> for BytesVisitor {
            type Value = Vec<u8>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a byte sequence")
            }
            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(v.to_vec())
            }
            fn visit_byte_buf<E: serde::de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                Ok(v)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(byte) = seq.next_element::<u8>()? {
                    out.push(byte);
                }
                Ok(out)
            }
        }
        deserializer.deserialize_byte_buf(BytesVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros2_client::dds::rustdds::serialization::{
        deserialize_from_cdr_with_rep_id, to_writer_with_rep_id, RepresentationIdentifier,
    };

    /// CDR little-endian payload exactly as RustDDS's `CDRSerializerAdapter`
    /// produces it for `ros2-client` publishers (the bytes following RTPS's
    /// 4-byte `CDR_LE` encapsulation header).
    mod cdr {
        use super::*;
        pub fn to_vec<T: serde::Serialize>(value: &T) -> Vec<u8> {
            let mut buffer = Vec::new();
            to_writer_with_rep_id(&mut buffer, value, RepresentationIdentifier::CDR_LE)
                .expect("serialize");
            buffer
        }
        pub fn from_bytes<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> (T, usize) {
            deserialize_from_cdr_with_rep_id(bytes, RepresentationIdentifier::CDR_LE)
                .expect("deserialize")
        }
    }

    fn roundtrip<T>(value: &T) -> Vec<u8>
    where
        T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let bytes = cdr::to_vec(value);
        let (decoded, consumed): (T, usize) = cdr::from_bytes(&bytes);
        assert_eq!(&decoded, value);
        assert_eq!(consumed, bytes.len());
        bytes
    }

    fn header() -> Header {
        Header::new(Time::from_nanos(1_403_636_579_763_555_584), "cam0")
    }

    #[test]
    fn time_from_nanos_normalizes_negative_fraction() {
        let t = Time::from_nanos(-1_500_000_000);
        assert_eq!(
            t,
            Time {
                sec: -2,
                nanosec: 500_000_000
            }
        );
        assert_eq!(t.to_nanos(), -1_500_000_000);
        let t = Time::from_nanos(1_403_636_579_763_555_584);
        assert_eq!(t.sec, 1_403_636_579);
        assert_eq!(t.nanosec, 763_555_584);
        assert_eq!(t.to_nanos(), 1_403_636_579_763_555_584);
    }

    #[test]
    fn header_matches_reference_cdr_bytes() {
        // Hand-assembled from the IDL: int32 sec, uint32 nanosec,
        // string frame_id = u32 length (incl. NUL) + bytes + NUL.
        let header = Header::new(Time { sec: 1, nanosec: 2 }, "map");
        let bytes = roundtrip(&header);
        assert_eq!(
            bytes,
            vec![1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, b'm', b'a', b'p', 0]
        );
    }

    #[test]
    fn image_matches_reference_cdr_layout() {
        let image = Image {
            header: Header::new(Time { sec: 7, nanosec: 9 }, "c"),
            height: 1,
            width: 3,
            encoding: "mono8".into(),
            is_bigendian: 0,
            step: 3,
            data: vec![10, 20, 30],
        };
        let bytes = roundtrip(&image);
        let mut expected = vec![7, 0, 0, 0, 9, 0, 0, 0, 2, 0, 0, 0, b'c', 0];
        expected.extend_from_slice(&[0, 0]); // pad to 4 for height
        expected.extend_from_slice(&1u32.to_le_bytes());
        expected.extend_from_slice(&3u32.to_le_bytes());
        expected.extend_from_slice(&6u32.to_le_bytes());
        expected.extend_from_slice(b"mono8\0");
        expected.push(0); // is_bigendian
        expected.push(0); // pad to 4 for step
        expected.extend_from_slice(&3u32.to_le_bytes());
        expected.extend_from_slice(&3u32.to_le_bytes()); // data length
        expected.extend_from_slice(&[10, 20, 30]);
        assert_eq!(bytes, expected);
    }

    #[test]
    fn byte_sequence_matches_generic_sequence_encoding() {
        // `serialize_bytes` must produce the same wire bytes as a generic
        // `Vec<u8>` sequence (what rosidl emits for `uint8[]`).
        #[derive(Serialize)]
        struct Generic {
            data: Vec<u8>,
        }
        #[derive(Serialize)]
        struct Fast {
            #[serde(with = "serde_bytes_vec")]
            data: Vec<u8>,
        }
        let data: Vec<u8> = (0..=255).collect();
        assert_eq!(
            cdr::to_vec(&Generic { data: data.clone() }),
            cdr::to_vec(&Fast { data })
        );
    }

    #[test]
    fn imu_roundtrip_and_size() {
        let imu = Imu {
            header: header(),
            orientation: Quaternion::default(),
            orientation_covariance: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            angular_velocity: Vector3 {
                x: 0.1,
                y: -0.2,
                z: 0.3,
            },
            angular_velocity_covariance: [0.01; 9],
            linear_acceleration: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 9.81,
            },
            linear_acceleration_covariance: [0.02; 9],
        };
        let bytes = roundtrip(&imu);
        // header: 8 (stamp) + 4 (len) + 5 ("cam0\0") = 17, padded to 24 for
        // the first float64; then 4 + 9 + 3 + 9 + 3 + 9 = 37 float64s.
        assert_eq!(bytes.len(), 24 + 37 * 8);
    }

    #[test]
    fn camera_info_roundtrip() {
        let info = CameraInfo {
            header: header(),
            height: 480,
            width: 752,
            distortion_model: "plumb_bob".into(),
            d: vec![-0.28, 0.07, 0.0002, 0.00002, 0.0],
            k: [458.6, 0.0, 367.2, 0.0, 457.3, 248.4, 0.0, 0.0, 1.0],
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            p: [
                458.6, 0.0, 367.2, 0.0, 0.0, 457.3, 248.4, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            binning_x: 0,
            binning_y: 0,
            roi: RegionOfInterest {
                x_offset: 1,
                y_offset: 2,
                height: 3,
                width: 4,
                do_rectify: true,
            },
        };
        roundtrip(&info);
    }

    #[test]
    fn covariance_is_a_fixed_array_without_length_prefix() {
        let covariance = Covariance6::diagonal(0.25, 0.5);
        let bytes = roundtrip(&covariance);
        assert_eq!(bytes.len(), 36 * 8);
        assert_eq!(&bytes[..8], &0.25f64.to_le_bytes());
        assert_eq!(&bytes[3 * 7 * 8..3 * 7 * 8 + 8], &0.5f64.to_le_bytes());
    }

    #[test]
    fn output_messages_roundtrip() {
        let pose = Pose {
            position: Point {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: std::f64::consts::FRAC_1_SQRT_2,
                w: std::f64::consts::FRAC_1_SQRT_2,
            },
        };
        let stamped = PoseStamped {
            header: header(),
            pose,
        };
        roundtrip(&stamped);
        roundtrip(&PoseWithCovarianceStamped {
            header: header(),
            pose: PoseWithCovariance {
                pose,
                covariance: Covariance6::diagonal(0.1, 0.2),
            },
        });
        roundtrip(&Odometry {
            header: header(),
            child_frame_id: "imu".into(),
            pose: PoseWithCovariance {
                pose,
                covariance: Covariance6::diagonal(0.1, 0.2),
            },
            twist: TwistWithCovariance {
                twist: Twist {
                    linear: Vector3 {
                        x: 0.5,
                        y: 0.0,
                        z: 0.0,
                    },
                    angular: Vector3::default(),
                },
                covariance: Covariance6::default(),
            },
        });
        roundtrip(&Path {
            header: header(),
            poses: vec![stamped.clone(), stamped],
        });
        roundtrip(&TfMessage {
            transforms: vec![TransformStamped {
                header: header(),
                child_frame_id: "imu".into(),
                transform: Transform {
                    translation: Vector3 {
                        x: 1.0,
                        y: 2.0,
                        z: 3.0,
                    },
                    rotation: Quaternion::default(),
                },
            }],
        });
        roundtrip(&DiagnosticArray {
            header: header(),
            status: vec![DiagnosticStatus {
                level: DiagnosticStatus::WARN,
                name: "visloc_localize_node: localization".into(),
                message: "few inliers".into(),
                hardware_id: String::new(),
                values: vec![KeyValue {
                    key: "inliers".into(),
                    value: "7".into(),
                }],
            }],
        });
        roundtrip(&Int32 { data: -42 });
    }

    #[test]
    fn type_names_match_ros_interfaces() {
        assert_eq!((Image::PACKAGE, Image::TYPE), ("sensor_msgs", "Image"));
        assert_eq!(
            (TfMessage::PACKAGE, TfMessage::TYPE),
            ("tf2_msgs", "TFMessage")
        );
        assert_eq!(
            (
                PoseWithCovarianceStamped::PACKAGE,
                PoseWithCovarianceStamped::TYPE
            ),
            ("geometry_msgs", "PoseWithCovarianceStamped")
        );
    }
}
