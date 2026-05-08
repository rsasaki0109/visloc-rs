## Summary

Describe the smallest user-visible change in this PR.

## Scope

- [ ] Visual localization
- [ ] Tracking
- [ ] Local mapping
- [ ] Online SLAM
- [ ] Sensor fusion / GNSS / IMU
- [ ] COLMAP / SfM IO
- [ ] Demo / documentation
- [ ] CI / packaging

## Checks

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace --all-targets`
- [ ] `scripts/run_examples.sh`
- [ ] `sh scripts/check_gnss_demo_outputs.sh`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- [ ] `scripts/package_check.sh`

## Notes

Mention API changes, compatibility concerns, dataset provenance, or intentionally deferred SLAM/fusion work.
