//! The JavaScript API: a [`Vio`] session driven from a Web Worker
//! (`web/worker.js`). The rig comes in as a plain object, IMU samples as a
//! packed `Float64Array`, frames as 8-bit grey `Uint8Array`s, and each
//! processed frame comes back as a plain object of numbers and typed arrays.
//! Timestamps are nanoseconds as JavaScript numbers (exact up to 2^53 ns,
//! about 104 days of `performance.now()`).

use js_sys::{Float32Array, Float64Array, Object, Reflect, Uint32Array};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use visloc_basalt::ImuSample;
use visloc_core::geometry::SE3;
use wasm_bindgen::prelude::*;

use crate::gpu_flow::GpuContext;
use crate::vio::{FrameUpdate, PhoneRig, Session, Status};

// Exports `initThreadPool` to JavaScript (the lint cannot see wasm-bindgen
// exports through this private module).
#[cfg(feature = "threads")]
#[allow(unused_imports)]
pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

/// Threads rayon runs on (1 without the `threads` build).
#[wasm_bindgen]
pub fn threads() -> usize {
    rayon::current_num_threads()
}

/// Whether this build was compiled with wasm threads.
#[wasm_bindgen(js_name = threadedBuild)]
pub fn threaded_build() -> bool {
    cfg!(feature = "threads")
}

fn get(obj: &JsValue, key: &str) -> JsValue {
    Reflect::get(obj, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

fn num(obj: &JsValue, key: &str) -> Result<f64, JsError> {
    get(obj, key)
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| JsError::new(&format!("rig.{key} must be a finite number")))
}

fn set(obj: &Object, key: &str, value: impl Into<JsValue>) {
    Reflect::set(obj, &JsValue::from_str(key), &value.into()).expect("plain object");
}

/// `[qw, qx, qy, qz, tx, ty, tz]`.
fn pose_array(pose: &SE3) -> Float64Array {
    let q = pose.rotation.quaternion();
    let t = pose.translation;
    Float64Array::from([q.w, q.i, q.j, q.k, t.x, t.y, t.z].as_slice())
}

fn vec3(v: &Vector3<f64>) -> Float64Array {
    Float64Array::from(v.as_slice())
}

/// A live VIO session.
#[wasm_bindgen]
pub struct Vio {
    session: Session,
}

#[wasm_bindgen]
impl Vio {
    /// `rig`: `{ width, height, fx, fy, cx, cy, distortion?: [k1, k2, p1,
    /// p2], imuFromCamera: [qw, qx, qy, qz, tx, ty, tz], imuRateHz,
    /// accelNoise, gyroNoise, accelBiasWalk, gyroBiasWalk, frameRateHz? }`
    /// (`frameRateHz`: frames tracked per second, 30 if absent). `overrides`: an object of Basalt `"config.*"` keys
    /// replacing the pinned EuRoC configuration's values (may be undefined).
    #[wasm_bindgen(constructor)]
    pub fn new(rig: JsValue, overrides: JsValue) -> Result<Vio, JsError> {
        let pose: Vec<f64> = get(&rig, "imuFromCamera")
            .dyn_into::<js_sys::Array>()
            .map_err(|_| JsError::new("rig.imuFromCamera must be an array of 7 numbers"))?
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();
        if pose.len() != 7 {
            return Err(JsError::new(
                "rig.imuFromCamera must be an array of 7 numbers",
            ));
        }
        let rig = PhoneRig {
            width: num(&rig, "width")? as u32,
            height: num(&rig, "height")? as u32,
            fx: num(&rig, "fx")?,
            fy: num(&rig, "fy")?,
            cx: num(&rig, "cx")?,
            cy: num(&rig, "cy")?,
            distortion: {
                let d = get(&rig, "distortion");
                let mut k = [0.0; 4];
                if let Ok(array) = d.dyn_into::<js_sys::Array>() {
                    for (i, v) in array.iter().take(4).enumerate() {
                        k[i] = v.as_f64().unwrap_or(0.0);
                    }
                }
                k
            },
            imu_from_camera: SE3::new(
                UnitQuaternion::from_quaternion(Quaternion::new(
                    pose[0], pose[1], pose[2], pose[3],
                )),
                Vector3::new(pose[4], pose[5], pose[6]),
            ),
            imu_rate_hz: num(&rig, "imuRateHz")?,
            accel_noise: num(&rig, "accelNoise")?,
            gyro_noise: num(&rig, "gyroNoise")?,
            accel_bias_walk: num(&rig, "accelBiasWalk")?,
            gyro_bias_walk: num(&rig, "gyroBiasWalk")?,
            frame_rate_hz: num(&rig, "frameRateHz").unwrap_or(30.0),
        };
        let mut config = Vec::new();
        if overrides.is_object() {
            for entry in Object::entries(overrides.unchecked_ref()).iter() {
                let entry: js_sys::Array = entry.unchecked_into();
                let key = entry.get(0).as_string().unwrap_or_default();
                let value = entry.get(1);
                let json = if let Some(b) = value.as_bool() {
                    serde_json::Value::Bool(b)
                } else if let Some(n) = value.as_f64() {
                    // JS has one number type; integer keys reject 36.0, and
                    // float keys take 36.
                    if n.fract() == 0.0 && n.abs() <= 9_007_199_254_740_991.0 {
                        serde_json::Value::from(n as i64)
                    } else {
                        serde_json::Number::from_f64(n)
                            .map(serde_json::Value::Number)
                            .ok_or_else(|| JsError::new(&format!("{key}: not a finite number")))?
                    }
                } else if let Some(s) = value.as_string() {
                    serde_json::Value::String(s)
                } else {
                    return Err(JsError::new(&format!(
                        "{key}: expected a number, bool or string"
                    )));
                };
                config.push((key, json));
            }
        }
        let session = Session::new(rig, config).map_err(|e| JsError::new(&e))?;
        Ok(Self { session })
    }

    /// Queues IMU samples packed as `[t_ns, gx, gy, gz, ax, ay, az, ...]`
    /// (rad/s; m/s² of specific force, +g up at rest, in the device frame).
    /// Returns how many were accepted (out-of-order ones are dropped).
    #[wasm_bindgen(js_name = pushImu)]
    pub fn push_imu(&mut self, samples: &[f64]) -> u32 {
        samples
            .chunks_exact(7)
            .filter(|s| {
                self.session.push_imu(ImuSample::new(
                    s[0].round() as i64,
                    Vector3::new(s[1], s[2], s[3]),
                    Vector3::new(s[4], s[5], s[6]),
                ))
            })
            .count() as u32
    }

    /// Whether a frame at `t_ns` can be processed (the IMU has reached it).
    pub fn ready(&self, t_ns: f64) -> bool {
        self.session.ready(t_ns.round() as i64)
    }

    /// Tracks one 8-bit grey frame. Returns `{ frameId, t, camera, imu,
    /// velocity, accelBias, gyroBias, keyframe, trackIds, trackXY,
    /// connected, imuSamples, status, resets }`: `camera` / `imu` are
    /// `T_w_c` / `T_w_i` as `[qw, qx, qy, qz, tx, ty, tz]` in a
    /// gravity-aligned world (+z up); `status` is `"wait"`, `"init"`,
    /// `"track"`, `"coast"` (no landmark connected for two frames after some
    /// were: the pose is the IMU's alone) or `"reset:<reason>"` (the watchdog
    /// restarted the estimator after this frame; the next poses continue
    /// from the last good one).
    pub fn process(&mut self, t_ns: f64, gray: &[u8]) -> Result<Object, JsError> {
        let u = self
            .session
            .process(t_ns.round() as i64, gray)
            .map_err(|e| JsError::new(&e))?;
        Ok(update_object(&u))
    }

    /// Tracks on the GPU from now on (consumes `gpu`); returns the adapter's
    /// description. Frames then go through [`Vio::submit`] and
    /// [`Vio::finish_gpu`] instead of [`Vio::process`].
    #[wasm_bindgen(js_name = attachGpu)]
    pub fn attach_gpu(&mut self, gpu: &mut Gpu) -> Result<String, JsError> {
        let ctx = gpu
            .ctx
            .take()
            .ok_or_else(|| JsError::new("this GPU is already attached"))?;
        let description = ctx.description.clone();
        self.session.attach_gpu(ctx).map_err(|e| JsError::new(&e))?;
        Ok(description)
    }

    /// Runs the estimator on every `n`th frame (the others get the last
    /// pose carried forward by the IMU); 1 by default.
    #[wasm_bindgen(js_name = setEstimateEvery)]
    pub fn set_estimate_every(&mut self, n: u32) {
        self.session.estimate_every = n.max(1);
    }

    /// Whether frames are tracked on the GPU.
    #[wasm_bindgen(js_name = hasGpu)]
    pub fn has_gpu(&self) -> bool {
        self.session.gpu().is_some()
    }

    /// Starts tracking a frame on the GPU (as [`Vio::process`] takes it);
    /// the promise resolves when [`Vio::finish_gpu`] can complete it.
    pub fn submit(&mut self, t_ns: f64, gray: &[u8]) -> Result<js_sys::Promise, JsError> {
        let mut settle = None;
        let promise = js_sys::Promise::new(&mut |resolve, reject| settle = Some((resolve, reject)));
        let (resolve, reject) = settle.expect("the executor runs at once");
        self.session
            .submit_gpu(t_ns.round() as i64, gray, move |result| {
                let _ = match result {
                    Ok(()) => resolve.call0(&JsValue::NULL),
                    Err(e) => reject.call1(&JsValue::NULL, &JsValue::from_str(&e.to_string())),
                };
            })
            .map_err(|e| JsError::new(&e))?;
        Ok(promise)
    }

    /// Completes the frame [`Vio::submit`] started; the same object as
    /// [`Vio::process`].
    #[wasm_bindgen(js_name = finishGpu)]
    pub fn finish_gpu(&mut self) -> Result<Object, JsError> {
        let u = self.session.finish_gpu().map_err(|e| JsError::new(&e))?;
        Ok(update_object(&u))
    }

    /// Collects the tracks of the frame on the GPU once [`Vio::submit`]'s
    /// promise resolved; false if a restart cancelled it. The next frame can
    /// then be submitted before [`Vio::estimate_gpu`] runs the estimator.
    #[wasm_bindgen(js_name = takeGpu)]
    pub fn take_gpu(&mut self) -> Result<bool, JsError> {
        self.session.take_gpu().map_err(|e| JsError::new(&e))
    }

    /// Runs the estimator on the tracks [`Vio::take_gpu`] collected: the
    /// same object as [`Vio::process`], or undefined without tracks.
    #[wasm_bindgen(js_name = estimateGpu)]
    pub fn estimate_gpu(&mut self) -> Result<Option<Object>, JsError> {
        let u = self.session.estimate_gpu().map_err(|e| JsError::new(&e))?;
        Ok(u.as_ref().map(update_object))
    }
}

/// A WebGPU device for [`Vio::attach_gpu`].
#[wasm_bindgen]
pub struct Gpu {
    ctx: Option<GpuContext>,
}

#[wasm_bindgen]
impl Gpu {
    /// Requests an adapter and a device; rejects where WebGPU is missing.
    pub async fn create() -> Result<Gpu, JsError> {
        let ctx = GpuContext::new().await.map_err(|e| JsError::new(&e))?;
        Ok(Gpu { ctx: Some(ctx) })
    }
}

fn update_object(u: &FrameUpdate) -> Object {
    {
        let obj = Object::new();
        set(&obj, "frameId", u.frame_id as f64);
        set(&obj, "t", u.timestamp_ns as f64);
        set(&obj, "camera", pose_array(&u.camera_to_world));
        set(&obj, "imu", pose_array(&u.state.imu_to_world));
        set(&obj, "velocity", vec3(&u.state.velocity_world_m_s));
        set(&obj, "accelBias", vec3(&u.state.accel_bias_m_s2));
        set(&obj, "gyroBias", vec3(&u.state.gyro_bias_rad_s));
        set(&obj, "keyframe", u.is_keyframe);
        let ids: Vec<u32> = u.tracks.iter().map(|t| t.0 as u32).collect();
        let xy: Vec<f32> = u.tracks.iter().flat_map(|t| [t.1, t.2]).collect();
        set(&obj, "trackIds", Uint32Array::from(ids.as_slice()));
        set(&obj, "trackXY", Float32Array::from(xy.as_slice()));
        set(&obj, "connected", u.connected as u32);
        set(&obj, "imuSamples", u.imu_samples as u32);
        let status = match u.status {
            Status::Waiting => "wait".to_string(),
            Status::Initializing => "init".to_string(),
            Status::Tracking => "track".to_string(),
            Status::Coasting => "coast".to_string(),
            Status::Reset(reason) => format!("reset:{}", reason.name()),
        };
        set(&obj, "status", status);
        set(&obj, "resets", u.resets);
        obj
    }
}

#[wasm_bindgen]
impl Vio {
    /// World positions of the landmarks in the window, xyz triples.
    pub fn landmarks(&self) -> Float32Array {
        let points: Vec<f32> = self.session.landmarks().into_iter().flatten().collect();
        Float32Array::from(points.as_slice())
    }

    /// Watchdog settings, any of `{ enabled, maxSpeed, maxGyroBias,
    /// maxAccelBias, maxUnconnectedSeconds, stillStart }` (`stillStart`:
    /// wait for the phone to be held still before each (re)start, while the
    /// status reads `"wait"`).
    #[wasm_bindgen(js_name = setWatchdog)]
    pub fn set_watchdog(&mut self, options: JsValue) {
        let w = &mut self.session.watchdog;
        if let Some(v) = get(&options, "enabled").as_bool() {
            w.enabled = v;
        }
        if let Some(v) = get(&options, "maxSpeed").as_f64() {
            w.max_speed_m_s = v;
        }
        if let Some(v) = get(&options, "maxGyroBias").as_f64() {
            w.max_gyro_bias_rad_s = v;
        }
        if let Some(v) = get(&options, "maxAccelBias").as_f64() {
            w.max_accel_bias_m_s2 = v;
        }
        if let Some(v) = get(&options, "maxUnconnectedSeconds").as_f64() {
            w.max_unconnected_s = v;
        }
        if let Some(v) = get(&options, "stillStart").as_bool() {
            self.session.still_start.enabled = v;
        }
    }

    /// The residual camera-IMU time offset seen over the last seconds of
    /// motion: `{ offsetMs, correlation, correlationAssumed, imuFromCamera,
    /// rotationErrorDeg, axisSpread, focalScale, intervals, rateRadS }` (add
    /// `offsetMs` to the frame timestamps; multiply the focal length by
    /// `focalScale`; `imuFromCamera` is the measured camera-IMU rotation
    /// `[qw, qx, qy, qz]`, `rotationErrorDeg` its angle from the rig's), or
    /// `null` until there is enough rotation.
    #[wasm_bindgen(js_name = timeSync)]
    pub fn time_sync(&self) -> JsValue {
        let Some(e) = self.session.time_sync() else {
            return JsValue::NULL;
        };
        let obj = Object::new();
        set(&obj, "offsetMs", e.offset_ms);
        set(&obj, "correlation", e.correlation);
        set(&obj, "correlationAssumed", e.correlation_assumed);
        let q = e.imu_from_camera.quaternion();
        set(
            &obj,
            "imuFromCamera",
            Float64Array::from([q.w, q.i, q.j, q.k].as_slice()),
        );
        set(&obj, "rotationErrorDeg", e.rotation_error_deg);
        set(&obj, "axisSpread", e.axis_spread);
        set(&obj, "focalScale", e.focal_scale);
        set(&obj, "intervals", e.intervals as u32);
        set(&obj, "rateRadS", e.rate_rad_s);
        obj.into()
    }

    /// Starts over from the next frame, in a new world.
    pub fn reset(&mut self) -> Result<(), JsError> {
        self.session.reset().map_err(|e| JsError::new(&e))
    }

    /// Frames processed since the start or the last reset.
    pub fn frames(&self) -> f64 {
        self.session.frames() as f64
    }

    /// Diagnostics: `{ imuQueued, imuFirst, imuLast, lastFrame, frames,
    /// resets }`, timestamps in ns (`null` where there is none). A frame is
    /// processed once `imuLast` reaches it and it is after `lastFrame`.
    pub fn debug(&self) -> Object {
        let ns = |t: Option<i64>| t.map_or(JsValue::NULL, |t| JsValue::from_f64(t as f64));
        let (queued, first, last) = self.session.imu_queue();
        let obj = Object::new();
        set(&obj, "imuQueued", queued as u32);
        set(&obj, "imuFirst", ns(first));
        set(&obj, "imuLast", ns(last));
        set(&obj, "lastFrame", ns(self.session.last_frame_ns()));
        set(&obj, "frames", self.session.frames() as f64);
        set(&obj, "resets", self.session.resets());
        obj
    }
}
