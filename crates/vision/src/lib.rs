#![forbid(unsafe_code)]
// The DPVO-port and verification modules document internal (crate-private)
// items extensively; intra-doc links to them are intentional and resolve
// under --document-private-items. Tolerated repo-wide until that doc pass.
#![allow(rustdoc::private_intra_doc_links, rustdoc::broken_intra_doc_links)]
//! Vision building blocks for visual localization.
//!
//! This crate contains feature-extractor traits/adapters, descriptor matching,
//! PnP pose estimation, RANSAC, and pose refinement. Components are trait-based
//! so applications can replace the default brute-force matcher or DLT PnP with
//! OpenCV, learned features, or domain-specific implementations.

pub mod dense_stereo;
pub mod distortion;
// Whole-module feature gate lives inside `dpvo/mod.rs` itself
// (`#![cfg(feature = "onnx-inference")]`) — see that module's doc comment
// for why this differs from `features::superpoint_onnx`'s always-visible
// stub pattern.
pub mod dpvo;
pub mod features;
pub mod matching;
pub mod place_recognition;
pub mod pnp;
pub mod ransac;
pub mod stereo;
pub mod stereo_bootstrap;
pub mod stereo_vo;
pub mod two_view;
pub mod vocab_tree;
