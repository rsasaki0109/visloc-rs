import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stress_scale_io.py"
SPEC = importlib.util.spec_from_file_location("stress_scale_io", SCRIPT)
stress = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stress)


class ScaleIoStressTests(unittest.TestCase):
    def test_pair_count_matches_stream(self):
        for images, neighbors in [(2, 32), (17, 4), (100, 8)]:
            self.assertEqual(
                stress.expected_pair_count(images, neighbors),
                sum(1 for _ in stress.iter_pairs(images, neighbors)),
            )

    def test_resume_matches_clean_run_and_verify(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            interrupted = base / "interrupted"
            clean = base / "clean"
            common = ["--images", "31", "--neighbors", "5", "--pairs-per-shard", "11"]
            self.assertEqual(
                stress.main(
                    ["--artifact-root", str(interrupted), *common, "--inject-stop-after-shards", "3"]
                ),
                stress.INTERRUPTED,
            )
            self.assertEqual(
                stress.main(["--artifact-root", str(interrupted), *common, "--resume"]), 0
            )
            self.assertEqual(stress.main(["--artifact-root", str(clean), *common]), 0)
            self.assertEqual(
                (interrupted / "index.json").read_bytes(), (clean / "index.json").read_bytes()
            )
            self.assertEqual(
                stress.main(["--artifact-root", str(interrupted), *common, "--verify-only"]), 0
            )

    def test_same_size_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = ["--artifact-root", str(root), "--images", "20", "--neighbors", "3"]
            self.assertEqual(stress.main(common), 0)
            index = json.loads((root / "index.json").read_text())
            shard = root / index["shards"][0]["path"]
            value = bytearray(shard.read_bytes())
            value[0] ^= 1
            shard.write_bytes(value)
            self.assertEqual(stress.main([*common, "--verify-only"]), 1)

    def test_resume_adopts_complete_unindexed_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = [
                "--artifact-root",
                str(root),
                "--images",
                "41",
                "--neighbors",
                "6",
                "--pairs-per-shard",
                "13",
            ]
            self.assertEqual(stress.main(common), 0)
            canonical = (root / "index.json").read_bytes()
            index = json.loads(canonical)
            index["complete"] = False
            index["shards"] = index["shards"][:2]
            (root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
            self.assertEqual(stress.main([*common, "--resume"]), 0)
            self.assertEqual((root / "index.json").read_bytes(), canonical)

    def test_malformed_index_and_unbounded_schedule_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "index.json").write_text("[]\n")
            self.assertEqual(
                stress.main(["--artifact-root", str(root), "--images", "10"]), 1
            )
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                stress.main(
                    [
                        "--artifact-root",
                        temporary,
                        "--images",
                        "1000",
                        "--neighbors",
                        str(stress.MAX_NEIGHBORS + 1),
                    ]
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
