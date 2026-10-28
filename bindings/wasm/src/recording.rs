//! A recorded session (`.vrec`): the rig, grey frames and IMU samples the
//! VIO session was (or would be) fed, in one file, so a phone capture from
//! the demo page can be replayed natively (`examples/replay.rs`) and a native
//! replay can be fed to the wasm build (`web/node_replay.mjs`).
//!
//! Layout (little-endian): the magic `VLOCREC2`, a `u32` header length, the
//! header as JSON (`{ rig, width, height, frames: [t_ns, ...], imu: count,
//! ... }`, the rig as `Vio`'s constructor takes it), then the IMU samples as
//! `f64`s packed `[t_ns, gx, gy, gz, ax, ay, az]`, then every frame's
//! `width x height` grey bytes. With the frames last, a recording cut short
//! (an upload from a phone that failed half-way) still replays up to its
//! last whole frame. `VLOCREC1` files (frames before the IMU samples) are
//! read too.

use crate::vio::PhoneRig;
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use visloc_basalt::ImuSample;
use visloc_core::geometry::SE3;

pub const MAGIC: &[u8; 8] = b"VLOCREC2";
/// The first layout: frames, then the IMU samples.
pub const MAGIC_V1: &[u8; 8] = b"VLOCREC1";

/// A decoded recording.
pub struct Recording {
    pub header: serde_json::Value,
    pub rig: PhoneRig,
    pub frame_times_ns: Vec<i64>,
    /// `frame_times_ns.len()` frames of `rig.width x rig.height` bytes.
    pub frames: Vec<u8>,
    pub imu: Vec<ImuSample>,
}

impl Recording {
    pub fn frame(&self, i: usize) -> &[u8] {
        let n = (self.rig.width * self.rig.height) as usize;
        &self.frames[i * n..(i + 1) * n]
    }
}

/// The rig as the JSON object `Vio`'s constructor takes.
pub fn rig_json(rig: &PhoneRig) -> serde_json::Value {
    let q = rig.imu_from_camera.rotation.quaternion();
    let t = rig.imu_from_camera.translation;
    serde_json::json!({
        "width": rig.width,
        "height": rig.height,
        "fx": rig.fx,
        "fy": rig.fy,
        "cx": rig.cx,
        "cy": rig.cy,
        "distortion": rig.distortion,
        "imuFromCamera": [q.w, q.i, q.j, q.k, t.x, t.y, t.z],
        "imuRateHz": rig.imu_rate_hz,
        "accelNoise": rig.accel_noise,
        "gyroNoise": rig.gyro_noise,
        "accelBiasWalk": rig.accel_bias_walk,
        "gyroBiasWalk": rig.gyro_bias_walk,
        "frameRateHz": rig.frame_rate_hz,
    })
}

/// [`rig_json`]'s inverse.
pub fn rig_from_json(v: &serde_json::Value) -> Result<PhoneRig, String> {
    let num = |key: &str| {
        v.get(key)
            .and_then(serde_json::Value::as_f64)
            .ok_or(format!("rig.{key} missing"))
    };
    let pose: Vec<f64> = v
        .get("imuFromCamera")
        .and_then(serde_json::Value::as_array)
        .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
        .unwrap_or_default();
    if pose.len() != 7 {
        return Err("rig.imuFromCamera needs 7 numbers".into());
    }
    let mut distortion = [0.0; 4];
    if let Some(d) = v.get("distortion").and_then(serde_json::Value::as_array) {
        for (k, x) in distortion.iter_mut().zip(d) {
            *k = x.as_f64().unwrap_or(0.0);
        }
    }
    Ok(PhoneRig {
        width: num("width")? as u32,
        height: num("height")? as u32,
        fx: num("fx")?,
        fy: num("fy")?,
        cx: num("cx")?,
        cy: num("cy")?,
        distortion,
        imu_from_camera: SE3::new(
            UnitQuaternion::from_quaternion(Quaternion::new(pose[0], pose[1], pose[2], pose[3])),
            Vector3::new(pose[4], pose[5], pose[6]),
        ),
        imu_rate_hz: num("imuRateHz")?,
        accel_noise: num("accelNoise")?,
        gyro_noise: num("gyroNoise")?,
        accel_bias_walk: num("accelBiasWalk")?,
        gyro_bias_walk: num("gyroBiasWalk")?,
        // Recordings from before the field tracked at 30 Hz.
        frame_rate_hz: num("frameRateHz").unwrap_or(30.0),
    })
}

/// Encodes a recording; `extra` keys are merged into the header.
pub fn encode(
    rig: &PhoneRig,
    frame_times_ns: &[i64],
    frames: &[u8],
    imu: &[ImuSample],
    extra: serde_json::Value,
) -> Vec<u8> {
    let mut header = serde_json::json!({
        "rig": rig_json(rig),
        "width": rig.width,
        "height": rig.height,
        "frames": frame_times_ns,
        "imu": imu.len(),
    });
    if let (Some(h), Some(e)) = (header.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            h.insert(k.clone(), v.clone());
        }
    }
    let json = header.to_string();
    let mut out = Vec::with_capacity(12 + json.len() + frames.len() + imu.len() * 56);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    for s in imu {
        for v in [
            s.timestamp_ns as f64,
            s.gyro_rad_s.x,
            s.gyro_rad_s.y,
            s.gyro_rad_s.z,
            s.accel_m_s2.x,
            s.accel_m_s2.y,
            s.accel_m_s2.z,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out.extend_from_slice(frames);
    out
}

/// Reads either layout; a `VLOCREC2` file cut short keeps its whole frames.
pub fn decode(bytes: &[u8]) -> Result<Recording, String> {
    let v1 = match bytes.get(..8) {
        Some(m) if m == MAGIC => false,
        Some(m) if m == MAGIC_V1 => true,
        _ => return Err("not a .vrec recording".into()),
    };
    let len = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(bytes.get(12..12 + len).ok_or("truncated header")?)
            .map_err(|e| format!("header: {e}"))?;
    let rig = rig_from_json(header.get("rig").ok_or("header.rig missing")?)?;
    let mut frame_times_ns: Vec<i64> = header
        .get("frames")
        .and_then(serde_json::Value::as_array)
        .ok_or("header.frames missing")?
        .iter()
        .map(|t| t.as_f64().unwrap_or(0.0).round() as i64)
        .collect();
    let imu_count = header
        .get("imu")
        .and_then(serde_json::Value::as_u64)
        .ok_or("header.imu missing")? as usize;
    let frame_size = (rig.width * rig.height) as usize;
    let (imu_start, frames_start) = if v1 {
        let frames_start = 12 + len;
        (
            frames_start + frame_times_ns.len() * frame_size,
            frames_start,
        )
    } else {
        (12 + len, 12 + len + imu_count * 56)
    };
    if bytes.len() < imu_start + imu_count * 56 {
        return Err("truncated recording (IMU samples missing)".into());
    }
    if !v1 {
        let whole = bytes.len().saturating_sub(frames_start) / frame_size.max(1);
        frame_times_ns.truncate(whole);
    }
    let frames_end = frames_start + frame_times_ns.len() * frame_size;
    if bytes.len() < frames_end {
        return Err("truncated recording (frames missing)".into());
    }
    let f = |o: usize| f64::from_le_bytes(bytes[o..o + 8].try_into().expect("8 bytes"));
    let imu = (0..imu_count)
        .map(|k| {
            let o = imu_start + k * 56;
            ImuSample::new(
                f(o).round() as i64,
                Vector3::new(f(o + 8), f(o + 16), f(o + 24)),
                Vector3::new(f(o + 32), f(o + 40), f(o + 48)),
            )
        })
        .collect();
    Ok(Recording {
        header,
        rig,
        frame_times_ns,
        frames: bytes[frames_start..frames_end].to_vec(),
        imu,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rig() -> PhoneRig {
        PhoneRig {
            width: 4,
            height: 2,
            fx: 3.0,
            fy: 3.0,
            cx: 1.5,
            cy: 0.5,
            distortion: [0.0; 4],
            imu_from_camera: PhoneRig::rear_portrait_camera_to_imu(Vector3::zeros()),
            imu_rate_hz: 60.0,
            accel_noise: 0.01,
            gyro_noise: 0.001,
            accel_bias_walk: 0.001,
            gyro_bias_walk: 0.0001,
            frame_rate_hz: 30.0,
        }
    }

    #[test]
    fn round_trip_and_truncation() {
        let frames: Vec<u8> = (0..24).collect(); // 3 frames of 4 x 2
        let imu: Vec<ImuSample> = (0..5)
            .map(|k| ImuSample::new(k * 10, Vector3::repeat(0.1), Vector3::new(0.0, 9.8, 0.0)))
            .collect();
        let bytes = encode(
            &rig(),
            &[0, 33, 66],
            &frames,
            &imu,
            serde_json::json!({ "note": 1 }),
        );
        let rec = decode(&bytes).expect("whole file");
        assert_eq!(rec.frame_times_ns, [0, 33, 66]);
        assert_eq!(rec.frame(2), &frames[16..24]);
        assert_eq!(rec.imu.len(), 5);
        assert_eq!(rec.imu[4].timestamp_ns, 40);
        assert_eq!(rec.header["note"], 1);
        // Cut inside the last frame: the two whole ones remain.
        let cut = decode(&bytes[..bytes.len() - 3]).expect("cut in a frame");
        assert_eq!(cut.frame_times_ns, [0, 33]);
        assert_eq!(cut.frame(1), &frames[8..16]);
        // Cut inside the IMU samples: nothing to replay.
        assert!(decode(&bytes[..bytes.len() - 24 - 10]).is_err());
    }
}
