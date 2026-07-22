import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_external_ssfm_setup.py"
SPEC = importlib.util.spec_from_file_location("prepare_external_ssfm_setup", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


REVISION = "a" * 40


def protocol() -> dict:
    return {
        "engines": {
            "gluemap": {"revision": REVISION},
            "instantsfm": {
                "published_repository": "https://github.com/cre185/InstantSfM.git",
                "observed_browser_redirect": "https://github.com/flqcsvqqvw/InstantSfM",
            },
        }
    }


def attempt(command=None, returncode=0, stdout="", stderr="") -> dict:
    return {
        "command": command or ["fixture"],
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


class ExternalSetupTests(unittest.TestCase):
    def test_sha256sum_parser_binds_expected_path(self) -> None:
        digest = "b" * 64
        self.assertEqual(MODULE.parse_sha256sum(f"{digest}  /src/model.bin\n", "/src/model.bin"), digest)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            MODULE.parse_sha256sum(f"{digest}  /other/model.bin\n", "/src/model.bin")

    def test_ready_gluemap_audit_binds_revision_submodules_models_and_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wrapper = root / "wrapper.py"
            inner = root / "inner.py"
            wrapper.write_text("wrapper\n", encoding="utf-8")
            inner.write_text("inner\n", encoding="utf-8")
            replies = [
                attempt(stdout=REVISION + "\n"),
                attempt(stdout=""),
                attempt(stdout=" abc123 thirdparty/pi3\n"),
            ]
            for relative in MODULE.GLUEMAP_CHECKPOINTS:
                replies.append(attempt(stdout=f"{'c' * 64}  /src/{relative}\n"))
            replies.extend(
                [
                    attempt(stdout="/env/pygluemap.so\nTrue\n"),
                    attempt(stdout="usage: gluemap-demo\n"),
                ]
            )
            with patch.object(MODULE, "run_capture", side_effect=replies):
                result = MODULE.audit_gluemap(
                    protocol=protocol(),
                    distro="Ubuntu-22.04",
                    source_dir="/src",
                    python="/env/bin/python",
                    demo="/env/bin/gluemap-demo",
                    wrapper=wrapper,
                    inner_adapter=inner,
                )
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["source_revision"], REVISION)
            self.assertEqual(len(result["checkpoint_sha256"]), 4)
            self.assertEqual(len(result["source_tree_sha256"]), 64)
            self.assertEqual(len(result["adapter"]["dependencies"]), 1)
            self.assertIn("{timestamps_path}", result["adapter"]["command_template"])

    def test_dirty_gluemap_worktree_is_install_failed_not_ready(self) -> None:
        replies = [attempt(stdout=REVISION + "\n"), attempt(stdout=" M gluemap/cli.py\n")]
        with patch.object(MODULE, "run_capture", side_effect=replies):
            result = MODULE.audit_gluemap(
                protocol=protocol(),
                distro="Ubuntu-22.04",
                source_dir="/src",
                python="python",
                demo="gluemap-demo",
                wrapper=SCRIPT_PATH,
                inner_adapter=SCRIPT_PATH,
            )
        self.assertEqual(result["status"], "install_failed")
        self.assertIn("not clean", result["reason"])

    def test_instantsfm_outage_requires_both_fresh_failures(self) -> None:
        failures = [attempt(returncode=128, stderr="not found"), attempt(returncode=128, stderr="not found")]
        with patch.object(MODULE, "run_capture", side_effect=failures):
            result = MODULE.audit_instantsfm(protocol())
        self.assertEqual(result["status"], "source_unavailable")
        self.assertEqual(len(result["retrieval_attempts"]), 2)
        with patch.object(
            MODULE,
            "run_capture",
            side_effect=[attempt(returncode=0, stdout="deadbeef HEAD\n"), failures[1]],
        ):
            with self.assertRaisesRegex(RuntimeError, "source is available"):
                MODULE.audit_instantsfm(protocol())


if __name__ == "__main__":
    unittest.main()
