import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class FeatureMatrixTests(unittest.TestCase):
    def test_root_cargo_features_match_documented_matrix(self):
        manifest = tomllib.loads(read("Cargo.toml"))
        self.assertEqual(
            set(manifest["features"].keys()),
            {"default", "image-io", "onnx-inference", "onnx-cuda"},
        )

        doc = read("docs/feature_matrix.md")
        documented = set()
        for line in doc.splitlines():
            if not line.startswith("| "):
                continue
            first_cell = line.split("|", 2)[1].strip().strip("`")
            if first_cell in {"Feature set", "---"}:
                continue
            documented.add(first_cell)

        self.assertEqual(
            documented,
            {
                "--no-default-features",
                "default",
                "image-io",
                "onnx-inference",
                "onnx-cuda",
            },
        )

    def test_feature_matrix_msrv_matches_root_manifest_and_ci(self):
        manifest = tomllib.loads(read("Cargo.toml"))
        msrv = manifest["package"]["rust-version"]
        self.assertEqual(msrv, "1.82")

        doc = read("docs/feature_matrix.md")
        ci = read(".github/workflows/ci.yml")
        check_msrv = read("scripts/check_msrv.sh")

        self.assertIn(f"Rust {msrv}", doc)
        self.assertIn(f"rust-toolchain@{msrv}.0", ci)
        self.assertRegex(
            check_msrv,
            rf"(rustup run {re.escape(msrv)}\.0|cargo \+{re.escape(msrv)}\.0)",
        )

    def test_check_script_covers_documented_feature_tiers(self):
        script = read("scripts/check_feature_matrix.sh")

        for command in [
            "cargo check --workspace --all-targets --no-default-features",
            "cargo check --workspace --all-targets",
            "cargo check --workspace --all-targets --features image-io",
            "cargo check --workspace --all-targets --features image-io,onnx-inference",
            "cargo check --workspace --all-targets --features image-io,onnx-cuda",
        ]:
            self.assertIn(command, script)

        self.assertIn("VISLOC_CHECK_ONNX", script)
        self.assertIn("VISLOC_CHECK_ONNX_CUDA", script)

    def test_ci_feature_matrix_matches_documented_tier_one(self):
        ci = read(".github/workflows/ci.yml")

        self.assertIn("feature-matrix:", ci)
        self.assertRegex(ci, r"os:\s*\[ubuntu-latest,\s*windows-latest\]")

        expected_entries = {
            "no-default": "cargo check --workspace --all-targets --no-default-features",
            "default": "cargo check --workspace --all-targets",
            "image-io": "cargo check --workspace --all-targets --features image-io",
        }
        for name, command in expected_entries.items():
            self.assertIn(f"name: {name}", ci)
            self.assertIn(f"command: {command}", ci)

        feature_matrix_job = re.search(
            r"(?ms)^  feature-matrix:.*?(?=^  [A-Za-z0-9_-]+:|\Z)", ci
        )
        self.assertIsNotNone(feature_matrix_job)
        self.assertNotIn("onnx-inference", feature_matrix_job.group(0))
        self.assertNotIn("onnx-cuda", feature_matrix_job.group(0))


if __name__ == "__main__":
    unittest.main()
