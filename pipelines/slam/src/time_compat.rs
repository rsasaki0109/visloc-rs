//! `std::time::Instant` where the platform has a clock. On
//! `wasm32-unknown-unknown`, where `Instant::now` panics ("time not
//! implemented on this platform"), an instant that never advances: the
//! timings the mapper takes only go to its logs and reports, so they read
//! zero there instead of stopping the run.

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) use std::time::Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) use frozen::Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod frozen {
    use std::time::Duration;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(crate) struct Instant;

    impl Instant {
        pub(crate) fn now() -> Self {
            Instant
        }

        pub(crate) fn elapsed(&self) -> Duration {
            Duration::ZERO
        }
    }

    impl std::ops::Sub for Instant {
        type Output = Duration;

        fn sub(self, _earlier: Instant) -> Duration {
            Duration::ZERO
        }
    }
}
