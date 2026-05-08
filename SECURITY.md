# Security Policy

`visloc-rs` is an early-stage robotics localization library. It is not safety-certified and should not be used as the sole localization source for safety-critical control.

## Supported Versions

Security and correctness reports should target the latest `main` branch or the latest published crate version. The project is pre-1.0, so APIs may still evolve.

## Reporting A Vulnerability

Please open a private security advisory on GitHub if the report involves:

- Memory safety or dependency vulnerabilities.
- Crashes caused by untrusted map, descriptor, trajectory, or image metadata inputs.
- Parser behavior that could enable denial of service with malformed COLMAP/SfM files.

For ordinary localization failures, parser errors, or incorrect results, use the bug report issue template instead.

## Safety-Critical Use

This project currently provides visual localization, tracking scaffolds, local mapping scaffolds, online SLAM MVP composition, and loose sensor-prior interfaces. It does not provide production-grade SLAM, loop closure, global bundle adjustment, dense mapping, GNSS/INS fusion, or certified autonomy behavior.
