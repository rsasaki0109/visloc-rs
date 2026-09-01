"""Focused tests for the resumable ETH3D electro runner."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_electro", ROOT / "scripts" / "benchmark_electro.py"
)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class ElectroBenchmarkTests(unittest.TestCase):
    def _candidate(self, root: Path) -> tuple[Path, list[str], list[tuple[int, int]]]:
        names = [f"{index:06d}.png" for index in range(5)]
        pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4)]
        path = root / "candidates.txt"
        benchmark.write_candidate_manifest(path, names, pairs)
        return path, names, pairs

    def _feature_manifest(self, root: Path, names: list[str]) -> Path:
        features = root / "features"
        features.mkdir()
        for name in names:
            (features / name.replace(".png", "_features.txt")).write_text(
                "0 0 1 2\n", encoding="utf-8"
            )
        path = root / "features.json"
        benchmark.write_feature_manifest(path, benchmark.feature_manifest(features))
        benchmark.validate_feature_manifest(path, features)
        return path

    @staticmethod
    def _gnu_time_report(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "User time (seconds): 0.01",
                    "System time (seconds): 0.00",
                    "Maximum resident set size (kbytes): 2048",
                    "Major (requiring I/O) page faults: 0",
                    "Minor (reclaiming a frame) page faults: 1",
                    "File system inputs: 0",
                    "File system outputs: 8",
                    "Exit status: 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_candidate_shards_are_contiguous_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, names, pairs = self._candidate(root)
            index = benchmark.split_candidate_manifest(source, root / "candidates", 2)
            self.assertEqual(index["pair_count"], len(pairs))
            validated = benchmark.validate_candidate_shards(root / "candidates" / "index.json")
            self.assertEqual(validated["image_names"], names)
            self.assertEqual(len(validated["shards"]), 3)
            self.assertEqual(
                [entry["start"] for entry in validated["shards"]], [0, 2, 4]
            )

            shard = root / "candidates" / "candidate-000001.txt"
            shard.write_text(shard.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "hash mismatch"):
                benchmark.validate_candidate_shards(root / "candidates" / "index.json")

    def test_measured_command_records_peak_rss_and_exit_status(self) -> None:
        if not benchmark.GNU_TIME.is_file():
            self.skipTest("GNU time is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timing = root / "true.time.txt"
            elapsed = benchmark._run_command(
                ["/bin/true"], root / "true.log", cwd=root, timing_path=timing
            )
            measurement = benchmark.measured_phase(timing, elapsed)
            self.assertEqual(measurement["exit_status"], 0)
            self.assertGreaterEqual(measurement["peak_rss_kib"], 0)
            self.assertGreaterEqual(measurement["elapsed_s"], 0.0)

    def test_phase_ledger_keeps_completed_prior_phases(self) -> None:
        current = {
            "candidate_generation": None,
            "candidate_sharding": None,
            "merge": None,
            "mapping": {"elapsed_s": 3.0},
        }
        previous = {
            "candidate_generation": {"elapsed_s": 1.0},
            "candidate_sharding": {"elapsed_s": 0.1},
            "merge": {"elapsed_s": 2.0},
            "mapping": None,
        }
        self.assertEqual(
            benchmark.carry_forward_phase_ledger(current, previous),
            {
                "candidate_generation": {"elapsed_s": 1.0},
                "candidate_sharding": {"elapsed_s": 0.1},
                "merge": {"elapsed_s": 2.0},
                "mapping": {"elapsed_s": 3.0},
            },
        )

    def test_interrupted_match_shard_is_rerun_and_corruption_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self._candidate(root)
            benchmark.split_candidate_manifest(source, root / "candidates", 10)
            candidate_index = root / "candidates" / "index.json"
            benchmark.prepare_match_index(candidate_index, root / "matches")
            match_index = root / "matches" / "index.json"

            def interrupted(command, _log_path, **_kwargs):
                snapshot = Path(command[command.index("--export-verified-pairs-snapshot") + 1])
                snapshot.write_bytes(b"partial")
                raise benchmark.ValidationError("injected interruption")

            with mock.patch.object(benchmark, "_run_command", side_effect=interrupted):
                with self.assertRaisesRegex(benchmark.ValidationError, "injected"):
                    benchmark.run_match_shards(
                        candidate_index,
                        match_index,
                        binary=Path("/bin/visloc"),
                        features_dir=root,
                        calibration_dir=root,
                    )
            failed = json.loads(match_index.read_text(encoding="utf-8"))
            self.assertEqual(failed["shards"][0]["status"], "failed")

            def completed(command, _log_path, *, timing_path=None, **_kwargs):
                snapshot = Path(command[command.index("--export-verified-pairs-snapshot") + 1])
                snapshot.write_bytes(b"complete")
                assert timing_path is not None
                timing_path.parent.mkdir(parents=True, exist_ok=True)
                timing_path.write_text(
                    "\n".join(
                        [
                            "User time (seconds): 0.01",
                            "System time (seconds): 0.00",
                            "Maximum resident set size (kbytes): 1024",
                            "Major (requiring I/O) page faults: 0",
                            "Minor (reclaiming a frame) page faults: 1",
                            "File system inputs: 0",
                            "File system outputs: 8",
                            "Exit status: 0",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return 0.01

            with mock.patch.object(benchmark, "_run_command", side_effect=completed):
                benchmark.run_match_shards(
                    candidate_index,
                    match_index,
                    binary=Path("/bin/visloc"),
                    features_dir=root,
                    calibration_dir=root,
                )
            complete = json.loads(match_index.read_text(encoding="utf-8"))
            self.assertEqual(complete["shards"][0]["status"], "complete")
            self.assertEqual(complete["shards"][0]["measurement"]["peak_rss_kib"], 1024)

            snapshot = root / "matches" / complete["shards"][0]["snapshot_path"]
            payload = bytearray(snapshot.read_bytes())
            payload[0] ^= 1
            snapshot.write_bytes(payload)
            with self.assertRaisesRegex(benchmark.ValidationError, "hash mismatch"):
                benchmark.validate_match_index(match_index, candidate_index)

    def test_candidate_metadata_survives_sharding_and_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["cam4_100.png", "cam5_100.png", "cam4_101.png"]
            source = root / "candidates.txt"
            metadata = {
                "candidate_policy": "vlad-union-v1",
                "cross_camera_rule": "same-timestamp",
                "local_grouping": "rig-prefix-timestamp-v1",
            }
            benchmark.write_candidate_manifest(source, names, [(0, 1), (0, 2)], metadata=metadata)
            index = benchmark.split_candidate_manifest(
                source,
                root / "candidates",
                1,
                local_grouping="rig-prefix-timestamp-v1",
            )
            self.assertEqual(index["candidate_manifest_metadata"], metadata)
            parsed = benchmark.parse_candidate_manifest_with_metadata(source)
            self.assertEqual(parsed[2], metadata)
            benchmark.validate_candidate_shards(root / "candidates" / "index.json")
            shard = root / "candidates" / "candidate-000000.txt"
            shard.write_text(
                shard.read_text(encoding="utf-8").replace("same-timestamp", "other"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(benchmark.ValidationError, "hash mismatch"):
                benchmark.validate_candidate_shards(root / "candidates" / "index.json")

    def test_complete_verification_rejects_pending_match_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self._candidate(root)
            benchmark.split_candidate_manifest(source, root / "candidates", 10)
            candidate_index = root / "candidates" / "index.json"
            benchmark.prepare_match_index(candidate_index, root / "matches")
            with self.assertRaisesRegex(benchmark.ValidationError, "not complete"):
                benchmark.validate_match_index(
                    root / "matches" / "index.json",
                    candidate_index,
                    require_complete=True,
                )

    def test_resume_rewrites_corrupt_candidate_shard_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self._candidate(root)
            first = benchmark.split_candidate_manifest(
                source,
                root / "candidates",
                3,
                retrieval_topk=32,
                local_stem_window=3,
                candidate_budget=5,
            )
            shard = root / "candidates" / "candidate-000000.txt"
            shard.write_text("broken\n", encoding="utf-8")
            second = benchmark.split_candidate_manifest(source, root / "candidates", 3)
            self.assertEqual(first["shards"][0]["sha256"], second["shards"][0]["sha256"])
            self.assertEqual(
                first["candidate_policy"],
                {"retrieval_topk": 32, "local_stem_window": 3, "candidate_budget": 5},
            )
            benchmark.validate_candidate_shards(root / "candidates" / "index.json")

    def test_match_plan_binds_candidate_index_hash_and_commands_are_gt_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self._candidate(root)
            benchmark.split_candidate_manifest(source, root / "candidates", 2)
            candidate_index = root / "candidates" / "index.json"
            match_index = benchmark.prepare_match_index(candidate_index, root / "matches")
            self.assertEqual(match_index["schema"], benchmark.MATCH_INDEX_SCHEMA)
            self.assertEqual(
                match_index["candidate_index_sha256"], benchmark.sha256_file(candidate_index)
            )
            benchmark.validate_match_index(root / "matches" / "index.json", candidate_index)

            candidate_command = benchmark.build_candidate_command(
                Path("/bin/visloc"),
                features_dir=Path("/input/features"),
                calibration_dir=Path("/input/calibration"),
                candidate_manifest=Path("/run/candidates.txt"),
            )
            match_command = benchmark.build_match_command(
                Path("/bin/visloc"),
                features_dir=Path("/input/features"),
                calibration_dir=Path("/input/calibration"),
                candidate_shard=Path("/run/candidate.txt"),
                snapshot=Path("/run/matches.vps"),
            )
            map_command = benchmark.build_mapping_command(
                Path("/bin/visloc"),
                features_dir=Path("/input/features"),
                calibration_dir=Path("/input/calibration"),
                merged_snapshot=Path("/run/merged.vps"),
                output_model=Path("/run/model"),
                max_mapper_matches_per_pair=128,
                snapshot_keypoints_only=True,
                periodic_ba_min_registered_images=1201,
                seed_trials=1,
                seed_pair="355,5305",
                ba_linear_solver="sparse",
                ba_max_iterations=8,
                post_refinement_registration=True,
                final_iterative_refinement=True,
                global_ba_max_refinements=0,
                confidence_ordered_tracks=True,
            )
            for command in (candidate_command, match_command, map_command):
                self.assertNotIn("--gt", command)
                self.assertNotIn("points3D.txt", command)
            self.assertIn("--export-verified-pairs-only", match_command)
            self.assertNotIn("--max-mapper-matches-per-pair", match_command)
            self.assertIn("--max-mapper-matches-per-pair", map_command)
            self.assertIn("--snapshot-keypoints-only", map_command)
            self.assertIn("--post-refinement-registration", map_command)
            self.assertIn("--confidence-ordered-tracks", map_command)
            self.assertEqual(map_command[map_command.index("--seed-pair") + 1], "355,5305")
            self.assertIn("--final-iterative-refinement", map_command)
            self.assertEqual(map_command[map_command.index("--ba-max-iterations") + 1], "8")
            self.assertEqual(
                map_command[map_command.index("--global-ba-max-refinements") + 1], "0"
            )
            no_ba_command = benchmark.build_mapping_command(
                Path("/bin/visloc"),
                features_dir=Path("/input/features"),
                calibration_dir=Path("/input/calibration"),
                merged_snapshot=Path("/run/merged.vps"),
                output_model=Path("/run/model"),
                final_ba=False,
            )
            self.assertIn("--no-final-ba", no_ba_command)
            with self.assertRaisesRegex(benchmark.ValidationError, "ba_linear_solver"):
                benchmark.build_mapping_command(
                    Path("/bin/visloc"),
                    features_dir=Path("/input/features"),
                    calibration_dir=Path("/input/calibration"),
                    merged_snapshot=Path("/run/merged.vps"),
                    output_model=Path("/run/model"),
                    ba_linear_solver="invalid",
                )
            with self.assertRaisesRegex(benchmark.ValidationError, "seed_pair"):
                benchmark.build_mapping_command(
                    Path("/bin/visloc"),
                    features_dir=Path("/input/features"),
                    calibration_dir=Path("/input/calibration"),
                    merged_snapshot=Path("/run/merged.vps"),
                    output_model=Path("/run/model"),
                    seed_pair="355",
                )
            with self.assertRaisesRegex(benchmark.ValidationError, "seed_pair image indices"):
                benchmark.build_mapping_command(
                    Path("/bin/visloc"),
                    features_dir=Path("/input/features"),
                    calibration_dir=Path("/input/calibration"),
                    merged_snapshot=Path("/run/merged.vps"),
                    output_model=Path("/run/model"),
                    seed_pair="01,1",
                )

            rig_candidate_command = benchmark.build_candidate_command(
                Path("/bin/visloc"),
                features_dir=Path("/input/features"),
                calibration_dir=Path("/input/calibration"),
                candidate_manifest=Path("/run/candidates.txt"),
                rig_local_grouping=True,
            )
            self.assertIn("--rig-local-grouping", rig_candidate_command)

            temporal_candidate_command = benchmark.build_candidate_command(
                Path("/bin/visloc"),
                features_dir=Path("/input/features"),
                calibration_dir=Path("/input/calibration"),
                candidate_manifest=Path("/run/temporal-candidates.txt"),
                pair_source="temporal-pyramid",
                temporal_pyramid_max_offset=64,
                candidate_budget=12000,
            )
            self.assertIn("--pair-source", temporal_candidate_command)
            self.assertIn("temporal-pyramid", temporal_candidate_command)
            self.assertIn("--temporal-pyramid-max-offset", temporal_candidate_command)
            self.assertIn("64", temporal_candidate_command)
            self.assertNotIn("--local-stem-window", temporal_candidate_command)
            self.assertNotIn("--rig-local-grouping", temporal_candidate_command)

            ann_candidate_command = benchmark.build_candidate_command(
                Path("/bin/visloc"),
                features_dir=Path("/input/features"),
                calibration_dir=Path("/input/calibration"),
                candidate_manifest=Path("/run/ann-candidates.txt"),
                pair_source="temporal-pyramid",
                stream_candidate_features=True,
                retrieval_backend="lsh",
                ann_tables=8,
                ann_bits=6,
                ann_probes=6,
            )
            self.assertIn("--stream-candidate-features", ann_candidate_command)
            self.assertEqual(
                ann_candidate_command[ann_candidate_command.index("--retrieval-backend") + 1],
                "lsh",
            )
            self.assertEqual(
                ann_candidate_command[ann_candidate_command.index("--ann-tables") + 1], "8"
            )
            self.assertEqual(
                ann_candidate_command[ann_candidate_command.index("--ann-bits") + 1], "6"
            )
            with self.assertRaisesRegex(benchmark.ValidationError, "requires stream"):
                benchmark.build_candidate_command(
                    Path("/bin/visloc"),
                    features_dir=Path("/input/features"),
                    calibration_dir=Path("/input/calibration"),
                    candidate_manifest=Path("/run/invalid-ann.txt"),
                    retrieval_backend="lsh",
                )

    def test_persistent_plan_binds_pending_shards_and_rejects_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, names, _ = self._candidate(root)
            benchmark.split_candidate_manifest(source, root / "candidates", 2)
            candidate_index = root / "candidates" / "index.json"
            benchmark.prepare_match_index(candidate_index, root / "matches")
            match_index = root / "matches" / "index.json"
            feature_manifest = self._feature_manifest(root, names)

            plan_path = benchmark.write_persistent_match_worker_plan(
                candidate_index, match_index, feature_manifest
            )
            self.assertIsNotNone(plan_path)
            assert plan_path is not None
            parsed = benchmark.parse_persistent_match_worker_plan(plan_path)
            validation = benchmark.validate_persistent_match_worker_plan(
                plan_path, candidate_index, match_index, feature_manifest
            )
            self.assertEqual(parsed["image_names"], names)
            self.assertEqual(
                [shard["id"] for shard in parsed["shards"]], [0, 1, 2]
            )
            self.assertEqual(validation["plan_path"], str(plan_path.resolve()))

            lines = plan_path.read_text(encoding="utf-8").splitlines()
            shard_line = next(index for index, line in enumerate(lines) if line.startswith("shard "))
            fields = lines[shard_line].split()
            fields[2] = "../escape.txt"
            unsafe = root / "unsafe.plan"
            unsafe.write_text(
                "\n".join(lines[:shard_line] + [" ".join(fields)] + lines[shard_line + 1 :])
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(benchmark.ValidationError, "relative path"):
                benchmark.parse_persistent_match_worker_plan(unsafe)

            duplicate = root / "duplicate.plan"
            duplicate_lines = plan_path.read_text(encoding="utf-8").splitlines()
            first = next(index for index, line in enumerate(duplicate_lines) if line.startswith("shard "))
            second = first + 1
            second_fields = duplicate_lines[second].split()
            first_fields = duplicate_lines[first].split()
            second_fields[3] = first_fields[3]
            duplicate_lines[second] = " ".join(second_fields)
            duplicate.write_text("\n".join(duplicate_lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "repeats"):
                benchmark.parse_persistent_match_worker_plan(duplicate)

            command = benchmark.build_persistent_match_command(
                Path("/bin/visloc"),
                features_dir=root / "features",
                calibration_dir=root / "calibration",
                plan=plan_path,
            )
            self.assertIn("--persistent-match-worker-plan", command)
            self.assertIn("--verification-mode", command)
            self.assertIn("full", command)
            self.assertIn("--matcher", command)
            self.assertIn("nn", command)
            self.assertIn("--mapper", command)
            self.assertIn("incremental", command)
            self.assertNotIn("--gt", command)
            self.assertNotIn("points3D.txt", command)

    def test_persistent_completion_parser_rejects_corruption_and_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "persistent-match.log"
            hash_value = "a" * 64
            line = (
                "persistent-match-complete shard_id=0 candidate_path=candidates/a.txt "
                "snapshot_path=matches/a.vps candidate_sha256="
                f"{hash_value} candidate_pairs=2 pairs=1 accepted=3 "
                "ordered_edge_fnv1a64=0123456789abcdef "
                "unordered_edge_fnv1a64=fedcba9876543210 elapsed_s=1.25"
            )
            log.write_text(line + "\n", encoding="utf-8")
            parsed = benchmark.parse_persistent_match_completions(log)
            self.assertEqual(parsed[0]["shard_id"], 0)
            self.assertEqual(parsed[0]["accepted"], 3)
            self.assertEqual(parsed[0]["ordered_edge_fnv1a64"], "0123456789abcdef")

            log.write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "repeats"):
                benchmark.parse_persistent_match_completions(log)
            log.write_text(line.replace("candidate_path=candidates/a.txt", "candidate_path=../a.txt") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "relative path"):
                benchmark.parse_persistent_match_completions(log)

    def test_persistent_matcher_recovers_flushed_prefix_and_skips_it_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, names, _ = self._candidate(root)
            benchmark.split_candidate_manifest(source, root / "candidates", 2)
            candidate_index = root / "candidates" / "index.json"
            benchmark.prepare_match_index(candidate_index, root / "matches")
            match_index = root / "matches" / "index.json"
            feature_manifest = self._feature_manifest(root, names)

            def completion_line(shard: dict[str, object], pair_count: int) -> str:
                return (
                    "persistent-match-complete "
                    f"shard_id={shard['id']} candidate_path={shard['candidate_path']} "
                    f"snapshot_path={shard['snapshot_path']} "
                    f"candidate_sha256={shard['candidate_sha256']} "
                    f"candidate_pairs={pair_count} pairs=1 accepted=1 "
                    "ordered_edge_fnv1a64=0123456789abcdef "
                    "unordered_edge_fnv1a64=fedcba9876543210 elapsed_s=1.25"
                )

            calls: list[list[str]] = []

            def killed(command, log_path, *, timing_path=None, **_kwargs):
                calls.append(command)
                plan = benchmark.parse_persistent_match_worker_plan(
                    Path(command[command.index("--persistent-match-worker-plan") + 1])
                )
                shard = plan["shards"][0]
                snapshot = root / shard["snapshot_path"]
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(b"snapshot-0")
                log_path.write_text(
                    "persistent-match-plan "
                    f"candidate_index_sha256={plan['candidate_index_sha256']} "
                    f"feature_manifest_sha256={plan['feature_manifest_sha256']}\n"
                    + completion_line(shard, 2)
                    + "\n",
                    encoding="utf-8",
                )
                raise benchmark.ValidationError("injected SIGKILL")

            with mock.patch.object(benchmark, "_run_command", side_effect=killed):
                with self.assertRaisesRegex(benchmark.ValidationError, "injected SIGKILL"):
                    benchmark.run_persistent_matcher(
                        candidate_index,
                        match_index,
                        binary=Path("/bin/visloc"),
                        features_dir=root / "features",
                        calibration_dir=root,
                        feature_manifest_path=feature_manifest,
                    )
            partial = json.loads(match_index.read_text(encoding="utf-8"))
            self.assertEqual([entry["status"] for entry in partial["shards"]], ["complete", "failed", "failed"])
            first_snapshot = root / "matches" / partial["shards"][0]["snapshot_path"]
            first_bytes = first_snapshot.read_bytes()

            def completed(command, log_path, *, timing_path=None, **_kwargs):
                calls.append(command)
                plan = benchmark.parse_persistent_match_worker_plan(
                    Path(command[command.index("--persistent-match-worker-plan") + 1])
                )
                records = [
                    "persistent-match-plan "
                    f"candidate_index_sha256={plan['candidate_index_sha256']} "
                    f"feature_manifest_sha256={plan['feature_manifest_sha256']}"
                ]
                for shard in plan["shards"]:
                    snapshot = root / shard["snapshot_path"]
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    snapshot.write_bytes(f"snapshot-{shard['id']}".encode())
                    records.append(completion_line(shard, 2 if shard["id"] < 2 else 1))
                log_path.write_text("\n".join(records) + "\n", encoding="utf-8")
                assert timing_path is not None
                self._gnu_time_report(timing_path)
                return 2.5

            with mock.patch.object(benchmark, "_run_command", side_effect=completed):
                result = benchmark.run_persistent_matcher(
                    candidate_index,
                    match_index,
                    binary=Path("/bin/visloc"),
                    features_dir=root / "features",
                    calibration_dir=root,
                    feature_manifest_path=feature_manifest,
                )
            self.assertEqual(len(calls), 2)
            resumed_plan = benchmark.parse_persistent_match_worker_plan(
                Path(calls[1][calls[1].index("--persistent-match-worker-plan") + 1])
            )
            self.assertEqual([shard["id"] for shard in resumed_plan["shards"]], [1, 2])
            self.assertEqual(first_snapshot.read_bytes(), first_bytes)
            self.assertEqual(result["persistent_worker"]["shards"], 2)
            complete = json.loads(match_index.read_text(encoding="utf-8"))
            self.assertEqual([entry["status"] for entry in complete["shards"]], ["complete"] * 3)

            with mock.patch.object(benchmark, "_run_command", side_effect=AssertionError("rerun")):
                benchmark.run_persistent_matcher(
                    candidate_index,
                    match_index,
                    binary=Path("/bin/visloc"),
                    features_dir=root / "features",
                    calibration_dir=root,
                    feature_manifest_path=feature_manifest,
                )
            first_snapshot.write_bytes(b"corrupt")
            with self.assertRaisesRegex(benchmark.ValidationError, "hash mismatch"):
                benchmark.validate_match_index(match_index, candidate_index, require_complete=True)

    def test_temporal_candidate_policy_is_hash_bound_in_shard_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self._candidate(root)
            index = benchmark.split_candidate_manifest(
                source,
                root / "temporal-candidates",
                2,
                retrieval_topk=32,
                candidate_budget=12000,
                pair_source="temporal-pyramid",
                temporal_pyramid_max_offset=32,
                local_grouping="rig-prefix-timestamp-v1",
            )
            self.assertEqual(
                index["candidate_policy"],
                {
                    "retrieval_topk": 32,
                    "local_stem_window": None,
                    "candidate_budget": 12000,
                    "pair_source": "temporal-pyramid",
                    "temporal_pyramid_max_offset": 32,
                    "local_grouping": "rig-prefix-timestamp-v1",
                },
            )
            benchmark.validate_candidate_shards(root / "temporal-candidates" / "index.json")

    def test_feature_manifest_rejects_changed_bytes_even_with_same_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features"
            features.mkdir()
            feature = features / "000000_features.txt"
            feature.write_text("0 0 1 2\n", encoding="utf-8")
            manifest = benchmark.feature_manifest(features)
            manifest_path = root / "features.json"
            benchmark.write_feature_manifest(manifest_path, manifest)
            benchmark.validate_feature_manifest(manifest_path, features)
            feature.write_text("0 0 1 3\n", encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "hash mismatch"):
                benchmark.validate_feature_manifest(manifest_path, features)

    def test_verify_mode_does_not_require_rust_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features"
            features.mkdir()
            (features / "000000_features.txt").write_text("0 0 1 2\n", encoding="utf-8")
            (features / "000001_features.txt").write_text("0 0 1 2\n", encoding="utf-8")
            feature_manifest = benchmark.feature_manifest(features)
            benchmark.write_feature_manifest(root / "features.json", feature_manifest)
            source = root / "source.txt"
            benchmark.write_candidate_manifest(source, ["000000.png", "000001.png"], [(0, 1)])
            benchmark.split_candidate_manifest(source, root / "candidates", 1)
            self.assertEqual(
                benchmark.main(
                    [
                        "--verify-only",
                        "--features-dir",
                        str(features),
                        "--calibration-dir",
                        str(root),
                        "--artifact-root",
                        str(root),
                    ]
                ),
                0,
            )
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["ground_truth_used_for_selection_or_mapping"])

    def test_verify_mode_rejects_corrupt_complete_match_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features"
            features.mkdir()
            for image_id in range(2):
                (features / f"{image_id:06d}_features.txt").write_text(
                    "0 0 1 2\n", encoding="utf-8"
                )
            feature_manifest = benchmark.feature_manifest(features)
            benchmark.write_feature_manifest(root / "features.json", feature_manifest)
            source = root / "source.txt"
            benchmark.write_candidate_manifest(
                source, ["000000.png", "000001.png"], [(0, 1)]
            )
            benchmark.split_candidate_manifest(source, root / "candidates", 1)
            candidate_index = root / "candidates" / "index.json"
            benchmark.prepare_match_index(candidate_index, root / "matches")
            match_index = root / "matches" / "index.json"
            index = json.loads(match_index.read_text(encoding="utf-8"))
            snapshot = root / "matches" / index["shards"][0]["snapshot_path"]
            snapshot.write_bytes(b"complete")
            index["shards"][0].update(
                {
                    "status": "complete",
                    "snapshot_sha256": benchmark.sha256_file(snapshot),
                }
            )
            benchmark.atomic_json(match_index, index)
            snapshot.write_bytes(b"Xomplete")

            self.assertEqual(
                benchmark.main(
                    [
                        "--verify-only",
                        "--features-dir",
                        str(features),
                        "--calibration-dir",
                        str(root),
                        "--artifact-root",
                        str(root),
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
