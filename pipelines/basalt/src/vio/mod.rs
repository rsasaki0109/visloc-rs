// Scoped workaround for rustc 1.94's dead_code diagnostic ICE in aom;
// see target/m7im15-r15-basalt-build.log. Runtime arithmetic is unchanged.
#[allow(dead_code)]
pub mod aom;
pub mod estimator;
pub mod landmarks;
pub mod margdata;
pub mod scalar;
pub mod static_hold;
pub mod window;
pub use aom::*;
pub use estimator::*;
pub use landmarks::*;
pub use margdata::*;
pub use scalar::*;
pub use static_hold::*;
pub use window::*;
