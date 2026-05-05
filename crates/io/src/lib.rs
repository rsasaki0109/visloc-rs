#![forbid(unsafe_code)]
//! Input/output helpers for visual maps and descriptors.
//!
//! The initial IO surface focuses on COLMAP text/binary sparse models and a
//! simple landmark/query descriptor text formats. Loaded maps can be exposed
//! through provider traits in `visloc-localization`.

pub mod colmap;
pub mod descriptors;
pub mod query_features;
