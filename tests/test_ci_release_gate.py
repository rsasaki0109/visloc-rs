import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def ci_job(name: str) -> str:
    ci = read(".github/workflows/ci.yml")
    match = re.search(rf"(?ms)^  {re.escape(name)}:.*?(?=^  [A-Za-z0-9_-]+:|\Z)", ci)
    if match is None:
        raise AssertionError(f"missing CI job: {name}")
    return match.group(0)


class CiReleaseGateTests(unittest.TestCase):
    def test_rust_ci_job_covers_release_gate_scripts(self):
        rust = ci_job("rust")
        for needle in [
            "cargo fmt --all -- --check",
            "cargo clippy --workspace --all-targets -- -D warnings",
            "cargo test --workspace --all-targets",
            "cargo test --test api_stability -- --nocapture",
            "python3 -m unittest discover -s tests -p 'test_*.py'",
            "scripts/benchmark_registry.py validate",
            "scripts/benchmark_registry.py check-generated",
            "scripts/check_docs_links.sh",
            "scripts/check_release_metadata.sh",
            "scripts/run_examples.sh",
            "scripts/check_trajectory_evaluation.sh",
            "scripts/check_gnss_demo_outputs.sh",
            "scripts/check_timestamped_gnss_image_demo_outputs.sh",
            "scripts/check_kitti_image_sequence_demo_outputs.sh",
            "cargo doc --workspace --no-deps",
            "scripts/package_check.sh",
        ]:
            self.assertIn(needle, rust)

    def test_ci_keeps_dnf_and_artifact_visibility_gates(self):
        rust = ci_job("rust")
        for artifact in [
            "gnss-demo-outputs",
            "timestamped-gnss-image-demo-outputs",
            "kitti-image-sequence-demo-outputs",
        ]:
            self.assertIn(f"name: {artifact}", rust)
            self.assertIn("if-no-files-found: error", rust)

        self.assertIn("benchmarks/registry/runs", rust)
        self.assertIn("docs/generated/registered_runs.md", read("docs/release_checklist.md"))

    def test_feature_matrix_ci_keeps_linux_and_windows_tier_one(self):
        matrix = ci_job("feature-matrix")
        self.assertRegex(matrix, r"os:\s*\[ubuntu-latest,\s*windows-latest\]")
        for command in [
            "cargo check --workspace --all-targets --no-default-features",
            "cargo check --workspace --all-targets",
            "cargo check --workspace --all-targets --features image-io",
        ]:
            self.assertIn(command, matrix)

        self.assertNotIn("onnx-inference", matrix)
        self.assertNotIn("onnx-cuda", matrix)


if __name__ == "__main__":
    unittest.main()
