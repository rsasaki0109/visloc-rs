#![forbid(unsafe_code)]
//! Vision building blocks for visual localization.
//!
//! This crate contains feature-extractor traits/adapters, descriptor matching,
//! PnP pose estimation, RANSAC, and pose refinement. Components are trait-based
//! so applications can replace the default brute-force matcher or DLT PnP with
//! OpenCV, learned features, or domain-specific implementations.

pub mod distortion;
pub mod features;
pub mod matching;
pub mod pnp;
pub mod ransac;
pub mod stereo;
pub mod stereo_bootstrap;
pub mod stereo_vo;
pub mod two_view;
