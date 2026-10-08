use crate::mapper::{GlobalBaConfig, MapperConfig, OfflineMapperConfig};
use crate::stream::{MultiCameraFlowOptions, StereoMatchingGuess};
use crate::vio::aom::LmConfig;
use crate::vio::estimator::{EstimatorConfig, UrgentKeyframePolicy};
use crate::vio::margdata::WindowPolicy;
use crate::vio::scalar::ScalarMode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use thiserror::Error;

const KEYS: &[&str] = &[
    "config.optical_flow_type",
    "config.optical_flow_detection_grid_size",
    "config.optical_flow_max_recovered_dist2",
    "config.optical_flow_pattern",
    "config.optical_flow_max_iterations",
    "config.optical_flow_epipolar_error",
    "config.optical_flow_levels",
    "config.optical_flow_skip_frames",
    "config.vio_linearization_type",
    "config.vio_sqrt_marg",
    "config.vio_max_states",
    "config.vio_max_kfs",
    "config.vio_min_frames_after_kf",
    "config.vio_new_kf_keypoints_thresh",
    "config.vio_debug",
    "config.vio_extended_logging",
    "config.vio_obs_std_dev",
    "config.vio_obs_huber_thresh",
    "config.vio_min_triangulation_dist",
    "config.vio_outlier_threshold",
    "config.vio_filter_iteration",
    "config.vio_max_iterations",
    "config.vio_enforce_realtime",
    "config.vio_use_lm",
    "config.vio_lm_lambda_initial",
    "config.vio_lm_lambda_min",
    "config.vio_lm_lambda_max",
    "config.vio_lm_landmark_damping_variant",
    "config.vio_lm_pose_damping_variant",
    "config.vio_scale_jacobian",
    "config.vio_init_pose_weight",
    "config.vio_init_ba_weight",
    "config.vio_init_bg_weight",
    "config.vio_marg_lost_landmarks",
    "config.vio_kf_marg_feature_ratio",
    "config.mapper_obs_std_dev",
    "config.mapper_obs_huber_thresh",
    "config.mapper_detection_num_points",
    "config.mapper_num_frames_to_match",
    "config.mapper_frames_to_match_threshold",
    "config.mapper_min_matches",
    "config.mapper_ransac_threshold",
    "config.mapper_min_track_length",
    "config.mapper_max_hamming_distance",
    "config.mapper_second_best_test_ratio",
    "config.mapper_bow_num_bits",
    "config.mapper_min_triangulation_dist",
    "config.mapper_no_factor_weights",
    "config.mapper_use_factors",
    "config.mapper_use_lm",
    "config.mapper_lm_lambda_min",
    "config.mapper_lm_lambda_max",
];
/// Opt-in keys that are not part of the upstream config schema.
const OPTIONAL_KEYS: &[&str] = &[
    "config.vio_urgent_kf_keypoints_thresh",
    "config.vio_urgent_min_frames_after_kf",
    // Raw-gyro rotation seed for temporal cam0 KLT (default false).  It was
    // already read by `adapter::direct_klt_config`, but only programmatic
    // callers could set it because this list rejected it in a JSON file.
    "config.optical_flow_imu_seed_rotation",
    // Multi-camera extensions for divergent rigs (all default off); see
    // `docs/lamaria_multicam.md`.
    "config.optical_flow_matching_guess_type",
    "config.optical_flow_matching_default_depth",
    "config.optical_flow_detect_all_cameras",
];
#[derive(Debug, Error, PartialEq)]
pub enum ConfigError {
    #[error("missing config key: {0}")]
    Missing(String),
    #[error("unknown config key: {0}")]
    Unknown(String),
    #[error("invalid config wrapper")]
    Wrapper,
    #[error("invalid config value: {0}")]
    Value(String),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BasaltConfig {
    pub values: BTreeMap<String, Value>,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompatProfile {
    pub enforce_realtime: bool,
    pub velocity_seed: bool,
    pub temporal_fb_threshold: Option<f64>,
    pub rebootstrap: bool,
}
impl BasaltConfig {
    pub fn from_json(s: &str) -> Result<Self, ConfigError> {
        let root: Value = serde_json::from_str(s).map_err(|e| ConfigError::Value(e.to_string()))?;
        let obj = root
            .get("value0")
            .and_then(Value::as_object)
            .ok_or(ConfigError::Wrapper)?;
        let mut values = BTreeMap::new();
        for k in obj.keys() {
            if !KEYS.contains(&k.as_str()) && !OPTIONAL_KEYS.contains(&k.as_str()) {
                return Err(ConfigError::Unknown(k.clone()));
            }
            values.insert(k.clone(), obj[k].clone());
        }
        for k in KEYS {
            if !values.contains_key(*k) {
                return Err(ConfigError::Missing((*k).into()));
            }
        }
        Ok(Self { values })
    }
    pub fn value<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<T, ConfigError> {
        self.values
            .get(key)
            .ok_or_else(|| ConfigError::Missing(key.into()))
            .and_then(|v| {
                serde_json::from_value(v.clone())
                    .map_err(|e| ConfigError::Value(format!("{key}: {e}")))
            })
    }
    fn urgent_kf_policy(&self) -> Result<Option<UrgentKeyframePolicy>, ConfigError> {
        let threshold = "config.vio_urgent_kf_keypoints_thresh";
        let spacing = "config.vio_urgent_min_frames_after_kf";
        match (
            self.values.contains_key(threshold),
            self.values.contains_key(spacing),
        ) {
            (false, false) => Ok(None),
            (true, true) => Ok(Some(UrgentKeyframePolicy {
                threshold: self.value(threshold)?,
                min_frames_after_kf: self.value(spacing)?,
            })),
            _ => Err(ConfigError::Value(format!(
                "{threshold} and {spacing} must be set together"
            ))),
        }
    }
    pub fn estimator_config(&self) -> Result<EstimatorConfig, ConfigError> {
        Ok(EstimatorConfig {
            scalar_mode: ScalarMode::UpstreamF32,
            window: WindowPolicy {
                max_states: self.value("config.vio_max_states")?,
                max_kfs: self.value("config.vio_max_kfs")?,
                min_feature_ratio: self.value("config.vio_kf_marg_feature_ratio")?,
            },
            min_frames_after_kf: self.value("config.vio_min_frames_after_kf")?,
            new_kf_keypoints_threshold: self.value("config.vio_new_kf_keypoints_thresh")?,
            urgent_kf: self.urgent_kf_policy()?,
            solver: LmConfig {
                lambda_initial: self.value("config.vio_lm_lambda_initial")?,
                lambda_min: self.value("config.vio_lm_lambda_min")?,
                lambda_max: self.value("config.vio_lm_lambda_max")?,
                max_iterations: self.value("config.vio_max_iterations")?,
                // Pinned upstream terminates when the infinity norm of the
                // accepted increment falls below 1e-4.
                convergence_step: 1e-4,
            },
            initial_pose_weight: self.value("config.vio_init_pose_weight")?,
            initial_accel_bias_weight: self.value("config.vio_init_ba_weight")?,
            initial_gyro_bias_weight: self.value("config.vio_init_bg_weight")?,
        })
    }
    /// An optional boolean key; absent means `false`.
    pub fn optional_bool(&self, key: &str) -> Result<bool, ConfigError> {
        if self.values.contains_key(key) {
            self.value(key)
        } else {
            Ok(false)
        }
    }
    /// The opt-in multi-camera optical-flow options.  With none of the keys
    /// present this is [`MultiCameraFlowOptions::default`], the pinned
    /// cam0-centric frontend.
    ///
    /// * `config.optical_flow_matching_guess_type`: `"SAME_PIXEL"` (default)
    ///   or `"REPROJ_FIX_DEPTH"`, as in newer upstream Basalt.
    ///   `"REPROJ_AVG_DEPTH"` is not implemented and is rejected.
    /// * `config.optical_flow_matching_default_depth`: metres along the cam0
    ///   ray for `REPROJ_FIX_DEPTH` (default 2.0, upstream's default).
    /// * `config.optical_flow_detect_all_cameras`: also replenish cam1.
    pub fn multi_camera_flow_options(&self) -> Result<MultiCameraFlowOptions, ConfigError> {
        let guess_key = "config.optical_flow_matching_guess_type";
        let depth_key = "config.optical_flow_matching_default_depth";
        let depth_m = if self.values.contains_key(depth_key) {
            let depth: f64 = self.value(depth_key)?;
            if !depth.is_finite() || depth <= 0.0 {
                return Err(ConfigError::Value(format!(
                    "{depth_key} must be finite and positive"
                )));
            }
            depth
        } else {
            2.0
        };
        let stereo_guess = if self.values.contains_key(guess_key) {
            let guess: String = self.value(guess_key)?;
            match guess.as_str() {
                "SAME_PIXEL" => StereoMatchingGuess::SamePixel,
                "REPROJ_FIX_DEPTH" => StereoMatchingGuess::ReprojectFixedDepth { depth_m },
                other => {
                    return Err(ConfigError::Value(format!(
                        "{guess_key}: unsupported value {other:?} (expected \"SAME_PIXEL\" or \"REPROJ_FIX_DEPTH\")"
                    )))
                }
            }
        } else {
            StereoMatchingGuess::SamePixel
        };
        Ok(MultiCameraFlowOptions {
            stereo_guess,
            detect_all_cameras: self.optional_bool("config.optical_flow_detect_all_cameras")?,
        })
    }
    pub fn compat_profile(&self) -> Result<CompatProfile, ConfigError> {
        Ok(CompatProfile {
            enforce_realtime: self.value("config.vio_enforce_realtime")?,
            velocity_seed: false,
            temporal_fb_threshold: None,
            rebootstrap: false,
        })
    }
    pub fn mapper_config(&self) -> Result<MapperConfig, ConfigError> {
        if self.value::<bool>("config.mapper_no_factor_weights")? {
            return Err(ConfigError::Value(
                "config.mapper_no_factor_weights=true is outside the pinned weighted-factor profile"
                    .into(),
            ));
        }
        Ok(MapperConfig::default())
    }
    pub fn offline_mapper_config(&self) -> Result<OfflineMapperConfig, ConfigError> {
        Ok(OfflineMapperConfig {
            max_points: self.value("config.mapper_detection_num_points")?,
            max_hamming: self.value("config.mapper_max_hamming_distance")?,
            second_best_ratio: self.value("config.mapper_second_best_test_ratio")?,
            bow_bits: self.value("config.mapper_bow_num_bits")?,
            match_window: self.value("config.mapper_num_frames_to_match")?,
            frames_to_match_threshold: self.value("config.mapper_frames_to_match_threshold")?,
            min_matches: self.value("config.mapper_min_matches")?,
            ransac_threshold: self.value("config.mapper_ransac_threshold")?,
            min_track_length: self.value("config.mapper_min_track_length")?,
            min_triangulation_distance: self.value("config.mapper_min_triangulation_dist")?,
        })
    }
    pub fn mapper_global_ba_config(&self) -> Result<GlobalBaConfig, ConfigError> {
        Ok(GlobalBaConfig {
            enabled: true,
            use_lm: self.value("config.mapper_use_lm")?,
            lambda_initial: self.value("config.mapper_lm_lambda_min")?,
            lambda_min: self.value("config.mapper_lm_lambda_min")?,
            lambda_max: self.value("config.mapper_lm_lambda_max")?,
            max_iterations: GlobalBaConfig::default().max_iterations,
            huber_delta: self.value("config.mapper_obs_huber_thresh")?,
            observation_std_dev: self.value("config.mapper_obs_std_dev")?,
            use_factors: self.value("config.mapper_use_factors")?,
        })
    }
    pub fn consumed_keys(&self) -> usize {
        self.values.len()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    const FIX: &str = include_str!("../../../configs/basalt/euroc_config.json");
    #[test]
    fn upstream_fixture_semantic_roundtrip_and_all_keys_consumed() {
        let c = BasaltConfig::from_json(FIX).unwrap();
        assert_eq!(c.consumed_keys(), KEYS.len());
        let e = c.estimator_config().unwrap();
        assert_eq!(e.window.max_states, 3);
        assert_eq!(e.window.max_kfs, 7);
        assert_eq!(e.window.min_feature_ratio, 0.1);
        assert_eq!(e.min_frames_after_kf, 5);
        assert_eq!(e.new_kf_keypoints_threshold, 0.7);
        assert_eq!(e.urgent_kf, None);
        assert_eq!(e.solver.max_iterations, 7);
        assert_eq!(e.initial_pose_weight, 1.0e8);
        assert_eq!(e.initial_accel_bias_weight, 1.0e1);
        assert_eq!(e.initial_gyro_bias_weight, 1.0e2);
        assert_eq!(c.mapper_config().unwrap(), MapperConfig::default());
        assert_eq!(
            c.offline_mapper_config().unwrap(),
            OfflineMapperConfig::default()
        );
        assert_eq!(
            c.mapper_global_ba_config().unwrap(),
            GlobalBaConfig::default()
        );
        assert_eq!(
            serde_json::to_string(&c).unwrap(),
            serde_json::to_string(&c).unwrap()
        );
    }
    #[test]
    fn unknown_and_missing_keys_rejected() {
        let mut v: Value = serde_json::from_str(FIX).unwrap();
        v["value0"]["config.unknown"] = Value::Bool(true);
        assert!(matches!(
            BasaltConfig::from_json(&v.to_string()),
            Err(ConfigError::Unknown(_))
        ));
        let mut v: Value = serde_json::from_str(FIX).unwrap();
        v["value0"]
            .as_object_mut()
            .unwrap()
            .remove("config.vio_max_states");
        assert!(matches!(
            BasaltConfig::from_json(&v.to_string()),
            Err(ConfigError::Missing(_))
        ));
    }
    #[test]
    fn urgent_keyframe_keys_are_optional_and_paired() {
        let mut v: Value = serde_json::from_str(FIX).unwrap();
        v["value0"]["config.vio_urgent_kf_keypoints_thresh"] = serde_json::json!(0.3);
        let partial = BasaltConfig::from_json(&v.to_string()).unwrap();
        assert!(matches!(
            partial.estimator_config(),
            Err(ConfigError::Value(_))
        ));
        v["value0"]["config.vio_urgent_min_frames_after_kf"] = serde_json::json!(2);
        let e = BasaltConfig::from_json(&v.to_string())
            .unwrap()
            .estimator_config()
            .unwrap();
        assert_eq!(
            e.urgent_kf,
            Some(UrgentKeyframePolicy {
                threshold: 0.3,
                min_frames_after_kf: 2,
            })
        );
    }
    #[test]
    fn compat_profile_has_no_out_of_json_extensions() {
        let c = BasaltConfig::from_json(FIX).unwrap();
        let p = c.compat_profile().unwrap();
        assert!(!p.velocity_seed);
        assert!(p.temporal_fb_threshold.is_none());
        assert!(!p.rebootstrap);
        assert!(!p.enforce_realtime);
    }
}
