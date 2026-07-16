from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_official_baselines", ROOT / "scripts" / "run_official_baselines.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OfficialBaselineRunnerTests(unittest.TestCase):
    def test_orb_command_never_receives_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in ("vocab", "settings", "timestamps"):
                paths[name] = root / name
                paths[name].write_text(name, encoding="utf-8")
            sequence = root / "sequence"
            sequence.mkdir()
            args = MODULE.parse_args(
                [
                    "orb-slam3",
                    "--executable",
                    sys.executable,
                    "--vocabulary",
                    str(paths["vocab"]),
                    "--settings",
                    str(paths["settings"]),
                    "--sequence-dir",
                    str(sequence),
                    "--timestamps",
                    str(paths["timestamps"]),
                    "--sequence",
                    "MH_01_easy",
                    "--source-revision",
                    "official-sha",
                    "--out-root",
                    str(root / "out"),
                    "--dry-run",
                ]
            )
            command = MODULE.orb_command(args, "label")
            self.assertEqual(command[-1], "label")
            self.assertFalse(any("ground" in argument.lower() for argument in command))

    def test_colmap_plan_copies_database_and_runs_global_mapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "database.db"
            database.write_bytes(b"sqlite fixture")
            images = root / "images"
            images.mkdir()
            (images / "a.png").write_bytes(b"image fixture")
            out = root / "out"
            return_code = MODULE.main(
                [
                    "colmap-global",
                    "--executable",
                    sys.executable,
                    "--database",
                    str(database),
                    "--images",
                    str(images),
                    "--sequence",
                    "fixture",
                    "--source-revision",
                    "official-sha",
                    "--out-root",
                    str(out),
                    "--repetitions",
                    "2",
                    "--dry-run",
                ]
            )
            self.assertEqual(return_code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["engine"], "colmap-global-mapper")
            self.assertFalse(manifest["protocol"]["ground_truth_available_to_engine"])
            self.assertTrue(manifest["protocol"]["same_database_copied_per_repetition"])
            self.assertEqual(len(manifest["runs"]), 2)
            commands = manifest["runs"][0]["commands"]
            self.assertEqual(commands[0][1], "view_graph_calibrator")
            self.assertEqual(commands[1][1], "global_mapper")
            self.assertIn(str(out / "run_01" / "database.db"), commands[1])

    def test_parses_current_colmap_model_analyzer_output(self) -> None:
        metrics = MODULE.parse_model_analyzer(
            """
            Registered images: 128
            Points: 12345
            Observations: 50000
            Mean track length: 4.050
            Mean observations per image: 390.625
            Mean reprojection error: 0.718px
            """
        )
        self.assertEqual(metrics["registered_images"], 128)
        self.assertEqual(metrics["points3d"], 12345)
        self.assertAlmostEqual(metrics["mean_reprojection_error_px"], 0.718)

    def test_rejects_non_key_value_colmap_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "NAME=VALUE"):
            MODULE.option_argv(["Mapper.ba_global_max_num_iterations"])

    def test_directory_identity_changes_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            image.write_bytes(b"first")
            first = MODULE.directory_identity(root)
            image.write_bytes(b"other")
            second = MODULE.directory_identity(root)
            self.assertEqual(first["file_count"], 1)
            self.assertEqual(first["total_bytes"], second["total_bytes"])
            self.assertNotEqual(first["tree_sha256"], second["tree_sha256"])

    def test_executes_orb_shim_and_captures_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shim = root / "fake_orb.py"
            shim.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "label = sys.argv[-1]\n"
                "Path(f'f_{label}.txt').write_text('1.0 0 0 0 0 0 0 1\\n')\n"
                "Path(f'kf_{label}.txt').write_text('1.0 0 0 0 0 0 0 1\\n')\n",
                encoding="utf-8",
            )
            files = {}
            for name in ("vocabulary", "settings", "timestamps"):
                files[name] = root / name
                files[name].write_text(name, encoding="utf-8")
            sequence = root / "sequence"
            sequence.mkdir()
            out = root / "out"
            return_code = MODULE.main(
                [
                    "orb-slam3",
                    "--executable",
                    sys.executable,
                    "--executable-arg",
                    str(shim),
                    "--vocabulary",
                    str(files["vocabulary"]),
                    "--settings",
                    str(files["settings"]),
                    "--sequence-dir",
                    str(sequence),
                    "--timestamps",
                    str(files["timestamps"]),
                    "--sequence",
                    "MH_01_easy",
                    "--source-revision",
                    "fixture",
                    "--out-root",
                    str(out),
                ]
            )
            self.assertEqual(return_code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            run = manifest["runs"][0]
            self.assertEqual(run["status"], "success")
            self.assertEqual(run["metrics"]["frame_trajectory_poses"], 1)
            self.assertEqual(run["metrics"]["keyframe_trajectory_poses"], 1)

    def test_executes_colmap_shim_and_parses_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shim = root / "fake_colmap.py"
            shim.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "command = sys.argv[1]\n"
                "if command == 'global_mapper':\n"
                "    out = Path(sys.argv[sys.argv.index('--output_path') + 1]) / '0'\n"
                "    out.mkdir(parents=True)\n"
                "    (out / 'cameras.bin').write_bytes(b'model')\n"
                "elif command == 'model_analyzer':\n"
                "    print('Registered images: 12', file=sys.stderr)\n"
                "    print('Points: 34', file=sys.stderr)\n"
                "    print('Mean reprojection error: 0.5px', file=sys.stderr)\n",
                encoding="utf-8",
            )
            database = root / "database.db"
            database.write_bytes(b"database")
            images = root / "images"
            images.mkdir()
            (images / "a.png").write_bytes(b"image")
            out = root / "out"
            return_code = MODULE.main(
                [
                    "colmap-global",
                    "--executable",
                    sys.executable,
                    "--executable-arg",
                    str(shim),
                    "--database",
                    str(database),
                    "--images",
                    str(images),
                    "--sequence",
                    "fixture",
                    "--source-revision",
                    "fixture",
                    "--out-root",
                    str(out),
                ]
            )
            self.assertEqual(return_code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            run = manifest["runs"][0]
            self.assertEqual(run["status"], "success")
            self.assertEqual(run["metrics"]["registered_images"], 12)
            self.assertEqual(run["metrics"]["points3d"], 34)
            self.assertAlmostEqual(run["metrics"]["mean_reprojection_error_px"], 0.5)

    def test_colmap_run_fails_when_model_analyzer_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shim = root / "fake_colmap.py"
            shim.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "command = sys.argv[1]\n"
                "if command == 'global_mapper':\n"
                "    out = Path(sys.argv[sys.argv.index('--output_path') + 1]) / '0'\n"
                "    out.mkdir(parents=True)\n"
                "    (out / 'cameras.bin').write_bytes(b'model')\n"
                "elif command == 'model_analyzer':\n"
                "    raise SystemExit(4)\n",
                encoding="utf-8",
            )
            database = root / "database.db"
            database.write_bytes(b"database")
            images = root / "images"
            images.mkdir()
            (images / "a.png").write_bytes(b"image")
            out = root / "out"
            return_code = MODULE.main(
                [
                    "colmap-global",
                    "--executable",
                    sys.executable,
                    "--executable-arg",
                    str(shim),
                    "--database",
                    str(database),
                    "--images",
                    str(images),
                    "--sequence",
                    "fixture",
                    "--source-revision",
                    "fixture",
                    "--out-root",
                    str(out),
                ]
            )
            self.assertEqual(return_code, 1)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runs"][0]["status"], "failure")
            self.assertEqual(
                manifest["runs"][0]["model_analyzer_process"]["return_code"], 4
            )


if __name__ == "__main__":
    unittest.main()
