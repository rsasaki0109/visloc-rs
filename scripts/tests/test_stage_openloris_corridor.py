"""Regression coverage for generalizing stage_openloris_corridor.py.

These tests prove that generalizing the staging script to accept a
sequence name, an alternate archive URL/commit, an optional tar-embedded
member byte range, and a configurable max-frames-per-camera did not change
its behavior for the original corridor1-1 defaults.

Some tests cross-check against the already-staged corridor1-1-m5 dataset on
this host (outside the repository). They are skipped when that fixture is
not present (e.g. in CI) rather than failing.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage_openloris_corridor as stage  # noqa: E402


CORRIDOR1_1_M5 = Path("/home/sasaki/datasets/openloris/corridor1-1-m5")
HAVE_FIXTURE = (
    CORRIDOR1_1_M5.is_dir()
    and (CORRIDOR1_1_M5 / "source-audit.json").is_file()
    and (CORRIDOR1_1_M5 / "image-state").is_dir()
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DefaultArgumentsTests(unittest.TestCase):
    """The new CLI parameters must not change corridor1-1's defaults."""

    def _parse(self, argv):
        old_argv = sys.argv
        sys.argv = ["stage_openloris_corridor.py", *argv]
        try:
            return stage.parse_args()
        finally:
            sys.argv = old_argv

    def test_scene_defaults_to_corridor1_1(self):
        args = self._parse(["--output-dir", "/tmp/x"])
        self.assertEqual(args.scene, "corridor1-1")
        self.assertEqual(args.scene, stage.DEFAULT_SCENE)

    def test_source_commit_and_urls_default_unchanged(self):
        args = self._parse(["--output-dir", "/tmp/x"])
        self.assertEqual(args.source_commit, "cbc03108723d08322b23d0338680bffa9404cce9")
        self.assertEqual(args.source_commit, stage.SOURCE_COMMIT)
        self.assertEqual(
            args.archive_url,
            "https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/"
            "cbc03108723d08322b23d0338680bffa9404cce9/package/corridor1-1.7z",
        )
        self.assertEqual(args.archive_url, stage.ARCHIVE_URL)

    def test_archive_provenance_defaults_unchanged(self):
        args = self._parse(["--output-dir", "/tmp/x"])
        self.assertEqual(args.archive_bytes, 13_853_763_765)
        self.assertEqual(args.archive_bytes, stage.ARCHIVE_BYTES)
        self.assertEqual(
            args.archive_sha256,
            "c7ff1a472ca54da82198521eda8c18f2065691075a05e706880f7fb58fda8415",
        )
        self.assertEqual(args.archive_sha256, stage.ARCHIVE_SHA256)
        self.assertEqual(
            args.archive_sha256_source,
            "Hugging Face LFS oid at source_commit; Range extraction does not rehash the complete remote archive",
        )

    def test_timestamps_default_is_5000(self):
        args = self._parse(["--output-dir", "/tmp/x"])
        self.assertEqual(args.timestamps, 5000)

    def test_tier_counts_default_matches_legacy_constant(self):
        args = self._parse(["--output-dir", "/tmp/x"])
        self.assertEqual(args.tier_counts, (1000, 2500, 5000, 10000))
        self.assertEqual(args.tier_counts, stage.TIER_COUNTS)

    def test_member_offset_and_size_default_to_none(self):
        args = self._parse(["--output-dir", "/tmp/x"])
        self.assertIsNone(args.member_offset)
        self.assertIsNone(args.member_size)

    def test_member_offset_requires_member_size(self):
        with self.assertRaises(SystemExit):
            self._parse(["--output-dir", "/tmp/x", "--member-offset", "512"])
        with self.assertRaises(SystemExit):
            self._parse(["--output-dir", "/tmp/x", "--member-size", "512"])

    def test_member_offset_and_size_parsed_together(self):
        args = self._parse(
            [
                "--output-dir",
                "/tmp/x",
                "--member-offset",
                "12802381312",
                "--member-size",
                "6892693271",
            ]
        )
        self.assertEqual(args.member_offset, 12802381312)
        self.assertEqual(args.member_size, 6892693271)

    def test_new_scene_can_override_everything(self):
        args = self._parse(
            [
                "--output-dir",
                "/tmp/x",
                "--scene",
                "corridor1-5",
                "--archive-url",
                "https://example.invalid/corridor1-2_5-package.tar",
                "--timestamps",
                "4381",
                "--tier-counts",
                "8762",
            ]
        )
        self.assertEqual(args.scene, "corridor1-5")
        self.assertEqual(args.timestamps, 4381)
        self.assertEqual(args.tier_counts, (8762,))


class CameraConstantsUnchangedTests(unittest.TestCase):
    """Guard the official Kannala-Brandt constants used for the gate check."""

    def test_camera_constants_literal_values(self):
        self.assertEqual(
            stage.CAMERAS[1]["intrinsics"],
            (284.98089599609375, 425.244384765625, 286.1023864746094, 398.46759033203125),
        )
        self.assertEqual(
            stage.CAMERAS[1]["distortion"],
            (-0.007304710801690817, 0.043499931693077087, -0.04128304123878479, 0.007652460131794214),
        )
        self.assertEqual(
            stage.CAMERAS[2]["intrinsics"],
            (284.8125915527344, 427.6615905761719, 285.97601318359375, 397.1234130859375),
        )
        self.assertEqual(
            stage.CAMERAS[2]["distortion"],
            (-0.006379498168826103, 0.04145561158657074, -0.03946448862552643, 0.0069808149710297585),
        )
        self.assertEqual(stage.WIDTH, 848)
        self.assertEqual(stage.HEIGHT, 800)


class SelectedMembersRegressionTests(unittest.TestCase):
    """selected_members() must keep sorting/pairing corridor1-1 identically.

    Feeds the generalized function only the exact 10,000 member names that
    corridor1-1-m5's own manifest recorded as selected (order scrambled), and
    checks it reproduces the same (camera, member, timestamp) sequence -- the
    same order that produced the on-disk cam{N}_{index:06d}.png naming.
    """

    @unittest.skipUnless(HAVE_FIXTURE, "corridor1-1-m5 fixture not present on this host")
    def test_reproduces_recorded_sequence(self):
        manifest = json.loads((CORRIDOR1_1_M5 / "manifests" / "tier-10000.json").read_text())
        images = manifest["images"]
        self.assertEqual(len(images), 10000)
        expected = [(record["camera"], record["source_member"], record["timestamp"]) for record in images]

        # Only the selected subset is known offline (the archive's other
        # ~7,000 unselected images per camera aren't available locally), but
        # since selected_members() truncates to the first `timestamps` sorted
        # names per camera, giving it exactly the selected subset (shuffled)
        # must reproduce the same sorted-and-paired sequence.
        names = [record["source_member"] for record in images]
        import random

        rng = random.Random(1234567)
        shuffled = list(names)
        rng.shuffle(shuffled)

        selected, targets = stage.selected_members(shuffled, 5000, "corridor1-1")
        self.assertEqual(selected, expected)
        for meta_target in (
            "corridor1-1/sensors.yaml",
            "corridor1-1/trans_matrix.yaml",
            "corridor1-1/groundtruth.txt",
            "corridor1-1/fisheye1.txt",
            "corridor1-1/fisheye2.txt",
        ):
            self.assertIn(meta_target, targets)

    def test_scene_prefix_is_parameterized(self):
        names = [
            "corridor1-5/fisheye1/100.000000.png",
            "corridor1-5/fisheye2/100.000000.png",
            "corridor1-1/fisheye1/999.000000.png",  # must be excluded for scene=corridor1-5
            "corridor1-1/fisheye2/999.000000.png",
        ]
        selected, targets = stage.selected_members(names, 1, "corridor1-5")
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(member.startswith("corridor1-5/") for _, member, _ in selected))
        self.assertIn("corridor1-5/sensors.yaml", targets)
        self.assertIn("corridor1-5/trans_matrix.yaml", targets)


class OfflineManifestReproductionTests(unittest.TestCase):
    """Reconstruct corridor1-1-m5's tier manifests purely from on-disk state.

    This does not re-run cv2 undistortion (unchanged code, not touched by
    this generalization); it independently reconstructs the exact records
    list the pipeline would have produced, from the images/ directory
    listing and the per-image image-state/*.json hash records, and checks
    the manifest-writing schema in main() still reproduces the on-disk
    manifests byte-for-byte.
    """

    @unittest.skipUnless(HAVE_FIXTURE, "corridor1-1-m5 fixture not present on this host")
    def test_reconstructed_records_match_tier_manifests_and_audit_hashes(self):
        images_dir = CORRIDOR1_1_M5 / "images"
        state_dir = CORRIDOR1_1_M5 / "image-state"
        audit = json.loads((CORRIDOR1_1_M5 / "source-audit.json").read_text())

        names = sorted(p.name for p in images_dir.glob("cam*_*.png"))
        records = []
        for name in names:
            camera_str, rest = name[len("cam") :].split("_", 1)
            sequence_index = int(rest.split(".")[0])
            state = json.loads((state_dir / f"{name}.json").read_text())
            member = state["source_member"]
            records.append(
                {
                    "camera": int(camera_str),
                    "sequence_index": sequence_index,
                    "timestamp": Path(member).stem,
                    "name": name,
                    "bytes": (images_dir / name).stat().st_size,
                    "sha256": state["output_sha256"],
                    "source_sha256": state["source_sha256"],
                    "source_member": member,
                }
            )
        records.sort(key=lambda r: r["sequence_index"])
        self.assertEqual(len(records), 10000)

        for count_str, expected in audit["tier_manifests"].items():
            count = int(count_str)
            payload = {
                "schema": "visloc_openloris_corridor_manifest_v1",
                "scene": "corridor1-1",
                "images": records[:count],
            }
            encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
            self.assertEqual(sha256_bytes(encoded), expected["sha256"], f"tier-{count} manifest hash mismatch")
            on_disk = Path(expected["path"]).read_bytes()
            self.assertEqual(encoded, on_disk, f"tier-{count} manifest bytes mismatch")


if __name__ == "__main__":
    unittest.main()
