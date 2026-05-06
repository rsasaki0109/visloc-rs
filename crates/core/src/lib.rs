#![forbid(unsafe_code)]
//! Core geometry and data model types for `visloc-rs`.
//!
//! This crate intentionally stays free of pipeline state. It provides camera,
//! frame, landmark, observation, map, pose, and validation types that can be
//! reused by localization, tracking, mapping, and future sensor-fusion layers.

pub mod geometry;
pub mod optimizer;
pub mod types;
