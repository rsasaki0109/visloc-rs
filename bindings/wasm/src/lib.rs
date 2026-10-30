//! visloc-rs in the browser: live monocular-inertial VIO on a phone.
//!
//! [`vio`] is the whole computation, target-agnostic (it also builds
//! natively: `examples/advio.rs` replays an ADVIO iPhone recording through
//! it, `examples/replay.rs` a [`recording`] captured by the demo page);
//! `web` wraps it for JavaScript with wasm-bindgen when compiled for
//! `wasm32`. `web/` holds the demo page.

#![forbid(unsafe_code)]

pub mod gpu_flow;
pub mod recording;
pub mod sync;
pub mod vio;
#[cfg(target_arch = "wasm32")]
mod web;
