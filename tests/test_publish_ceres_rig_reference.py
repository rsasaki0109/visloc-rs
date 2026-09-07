"""Small identity/transaction tests for the frozen Ceres model publisher."""

import importlib.util
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "publish_ceres_rig_reference",
    Path(__file__).resolve().parents[1]
    / "tools"
    / "ceres_rig_reference"
    / "publish_model.py",
)
PUBLISH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PUBLISH
SPEC.loader.exec_module(PUBLISH)
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_colmap_pinhole_model",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_colmap_pinhole_model.py",
)
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = AUDIT
AUDIT_SPEC.loader.exec_module(AUDIT)


def bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def qz(angle):
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def compose(q_sensor, t_sensor, q_rig, t_rig):
    return (
        PUBLISH._qnormalize(PUBLISH._qmul(q_sensor, q_rig)),
        PUBLISH._vadd(PUBLISH._qrotate(q_sensor, t_rig), t_sensor),
    )


class PublisherFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.fixture = self.root / "input.fixture"
        self.state = self.root / "solve.state"
        self.manifest = self.root / "rig-manifest.txt"
        self.out = self.root / "published"
        self._write_fixture()

    def _write_fixture(self):
        camera_text = (
            "# camera bytes must survive publication\n"
            "10 PINHOLE 640 480 100 100 320 240\n"
            "20 PINHOLE 640 480 100 100 320 240\n"
        )
        self.model.joinpath("cameras.txt").write_text(camera_text)
        sensor0 = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        sensor1 = (qz(0.25), (0.4, 0.0, 0.02))
        self.sensors = {0: sensor0, 1: sensor1}
        manifest_text = (
            "# generalized-rig-manifest-v1\n"
            "S 0 10 640 480 100 100 320 240 1 0 0 0 0 0 0\n"
            "S 1 20 640 480 100 100 320 240 "
            + " ".join(f"{x:.17g}" for x in (*sensor1[0], *sensor1[1]))
            + "\n"
            "F 0 f0-s0.png 0\nF 0 f0-s1.png 1\n"
            "F 1 f1-s0.png 0\nF 1 f1-s1.png 1\n"
        )
        self.manifest.write_text(manifest_text)
        true_poses = {
            0: ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            1: (qz(0.03), (0.1, 0.02, 0.01)),
        }
        initial_poses = {
            0: true_poses[0],
            1: ((1.0, 0.0, 0.0, 0.0), (0.21, 0.02, 0.01)),
        }
        true_points = {1: (0.0, 0.1, 4.0), 2: (0.6, -0.2, 5.0)}
        initial_points = {1: (0.03, 0.08, 3.9), 2: (0.62, -0.18, 4.9)}
        image_specs = [
            (1, 0, 0, "f0-s0.png"),
            (2, 0, 1, "f0-s1.png"),
            (3, 1, 0, "f1-s0.png"),
            (4, 1, 1, "f1-s1.png"),
        ]
        image_text = ["# image header\n", "# POINTS2D\n"]
        fixture_observations = []
        initial_cost = 0.0
        for image_id, frame, sensor_index, name in image_specs:
            q_image, t_image = compose(*self.sensors[sensor_index], *initial_poses[frame])
            camera_id = 10 if sensor_index == 0 else 20
            pairs = []
            for keypoint, point_id in enumerate((1, 2)):
                q_obs, t_obs = self.sensors[sensor_index]
                xy = PUBLISH._project(
                    PUBLISH.Camera(camera_id, "PINHOLE", 640, 480, (100.0, 100.0, 320.0, 240.0)),
                    *true_poses[frame],
                    true_points[point_id],
                    q_obs,
                    t_obs,
                )
                pairs.append((xy[0], xy[1], point_id))
                fixture_observations.append((frame, point_id, xy[0], xy[1], camera_id, *self.sensors[sensor_index]))
                old = PUBLISH._project(
                    PUBLISH.Camera(camera_id, "PINHOLE", 640, 480, (100.0, 100.0, 320.0, 240.0)),
                    *initial_poses[frame],
                    initial_points[point_id],
                    q_obs,
                    t_obs,
                )
                initial_cost += (old[0] - xy[0]) ** 2 + (old[1] - xy[1]) ** 2
            image_text.append(
                f"{image_id} {' '.join(f'{x:.17g}' for x in (*q_image, *t_image))} {camera_id} {name}\n"
            )
            image_text.append(" ".join(f"{x:.17g} {y:.17g} {point_id}" for x, y, point_id in pairs) + "\n")
        fixture_observations.sort(key=lambda row: (row[1], row[0], row[4]))
        self.model.joinpath("images.txt").write_text("".join(image_text))
        point_lines = []
        for point_id in (1, 2):
            track = []
            for image_id, _, _, name in image_specs:
                keypoint = 0 if point_id == 1 else 1
                track.extend((str(image_id), str(keypoint)))
            point_lines.append(
                f"{point_id} {' '.join(f'{x:.17g}' for x in initial_points[point_id])} 10 20 30 0 "
                + " ".join(track)
                + "\n"
            )
        self.model.joinpath("points3D.txt").write_text("".join(point_lines))
        model = PUBLISH.parse_model(self.model)
        fixture_hashes = {
            "SOURCE_SHA256_CAMERAS": PUBLISH._sha256(self.model / "cameras.txt"),
            "SOURCE_SHA256_IMAGES": PUBLISH._sha256(self.model / "images.txt"),
            "SOURCE_SHA256_POINTS": PUBLISH._sha256(self.model / "points3D.txt"),
            "SOURCE_SHA256_MANIFEST": PUBLISH._sha256(self.manifest),
        }
        fixture_hashes["SOURCE_SHA256"] = PUBLISH._combined_sha(fixture_hashes)
        fixture_lines = [
            "VISLOC_BA_ORACLE_FIXTURE 1",
            *[f"{key} {fixture_hashes[key]}" for key in PUBLISH.SHA_FIELDS],
            f"INITIAL_COST {initial_cost:.17g}",
            f"INITIAL_COST_BITS {bits(initial_cost)}",
            "CAMERA_COUNT 2",
            "POSE_COUNT 2",
            "LANDMARK_COUNT 2",
            "OBSERVATION_COUNT 8",
            "CAMERA 10 PINHOLE 640 480 4 100 100 320 240",
            "CAMERA 20 PINHOLE 640 480 4 100 100 320 240",
        ]
        for frame in (0, 1):
            fixture_lines.append("POSE %d %s" % (frame, " ".join(f"{x:.17g}" for x in (*initial_poses[frame][0], *initial_poses[frame][1]))))
        fixture_lines.append("FIXED_POSE 0")
        for point_id in (1, 2):
            fixture_lines.append("LANDMARK %d %s" % (point_id, " ".join(f"{x:.17g}" for x in initial_points[point_id])))
        for row in fixture_observations:
            frame, point_id, x, y, camera_id, q, t = row
            fixture_lines.append("RIG_OBSERVATION %d %d %.17g %.17g %d %s" % (frame, point_id, x, y, camera_id, " ".join(f"{v:.17g}" for v in (*q, *t))))
        fixture_lines.append("END")
        self.fixture.write_text("\n".join(fixture_lines) + "\n")
        final_cost = 0.0
        state_lines = [
            "VISLOC_BA_CERES_SOLVE_STATE 1",
            *[f"{key} {fixture_hashes[key]}" for key in PUBLISH.SHA_FIELDS],
            "CERES_VERSION 2.2.0",
            "FIXED_POSE 0",
            "CAMERA_COUNT 2",
            "POSE_COUNT 2",
            "LANDMARK_COUNT 2",
            "OBSERVATION_COUNT 8",
            f"DECLARED_INITIAL_COST {initial_cost:.17g}",
            f"DECLARED_INITIAL_COST_BITS {bits(initial_cost)}",
            "SOLVER_MODE CERES_STANDALONE_REFERENCE",
            "COST_CONVENTION CERES_HALF_SQUARED_INTERNAL_FULL_SQUARED_REPORTED",
            "OPTIONS_BEGIN",
            "OPTION_MAX_NUM_ITERATIONS 20",
            "OPTIONS_END",
            f"INITIAL_FULL_SQUARED_COST {initial_cost:.17g}",
            f"INITIAL_FULL_SQUARED_COST_BITS {bits(initial_cost)}",
            "INITIAL_OBSERVATIONS 8",
            "INITIAL_ALL_OBSERVATIONS_VALID 1",
            "INITIAL_POSITIVE_DEPTH 8",
            "INITIAL_MIN_DEPTH 3.0",
            "INITIAL_MAX_DEPTH 5.0",
            "INITIAL_CERES_EIGEN_MAX_RESIDUAL_ABS_DIFF 0",
            "INITIAL_CERES_EIGEN_MAX_DEPTH_ABS_DIFF 0",
            "FINAL_FULL_SQUARED_COST 0",
            "FINAL_FULL_SQUARED_COST_BITS 0",
            "FINAL_OBSERVATIONS 8",
            "FINAL_ALL_OBSERVATIONS_VALID 1",
            "FINAL_POSITIVE_DEPTH 8",
            "FINAL_MIN_DEPTH 3.0",
            "FINAL_MAX_DEPTH 5.0",
            "FINAL_CERES_EIGEN_MAX_RESIDUAL_ABS_DIFF 0",
            "FINAL_CERES_EIGEN_MAX_DEPTH_ABS_DIFF 0",
            "CERES_SUMMARY_INITIAL_HALF_COST %.17g" % (initial_cost / 2),
            "CERES_SUMMARY_FINAL_HALF_COST 0",
            "CERES_SUMMARY_TERMINATION_TYPE 1",
            "CERES_SUMMARY_IS_SOLUTION_USABLE 1",
            "CERES_SUMMARY_NUM_SUCCESSFUL_STEPS 1",
            "CERES_SUMMARY_NUM_UNSUCCESSFUL_STEPS 0",
            "CERES_SUMMARY_TOTAL_TIME_SECONDS 0",
            "CERES_SUMMARY_MESSAGE_BEGIN",
            "synthetic",
            "CERES_SUMMARY_MESSAGE_END",
            "CERES_SUMMARY_FULL_REPORT_BEGIN",
            "synthetic",
            "CERES_SUMMARY_FULL_REPORT_END",
            "ITERATION_COUNT 0",
            "POSE_STATE_BEGIN",
        ]
        for frame in (0, 1):
            state_lines.append("POSE %d %s" % (frame, " ".join(f"{x:.17g}" for x in (*true_poses[frame][0], *true_poses[frame][1]))))
        state_lines += ["POSE_STATE_END", "LANDMARK_STATE_BEGIN"]
        for point_id in (1, 2):
            state_lines.append("LANDMARK %d %s" % (point_id, " ".join(f"{x:.17g}" for x in true_points[point_id])))
        state_lines += ["LANDMARK_STATE_END", "END"]
        self.state.write_text("\n".join(state_lines) + "\n")
        self.initial_cost = initial_cost
        self.true_poses = true_poses
        self.true_points = true_points
        self.model_before = model

    def test_publish_preserves_identity_and_composes_sensor_pose(self):
        PUBLISH.publish_model(
            fixture_path=self.fixture,
            state_path=self.state,
            model_path=self.model,
            rig_manifest_path=self.manifest,
            out_dir=self.out,
        )
        output = PUBLISH.parse_model(self.out)
        audit = AUDIT.audit(self.out)
        self.assertEqual(audit["observations"], 8)
        self.assertEqual(audit["nonpositive_depth_observations"], 0)
        self.assertAlmostEqual(audit["mean_reprojection_px"], 0.0, places=12)
        self.assertEqual((self.out / "cameras.txt").read_bytes(), (self.model / "cameras.txt").read_bytes())
        self.assertEqual([(point.point_id, point.rgb, point.track) for point in output.points], [(point.point_id, point.rgb, point.track) for point in self.model_before.points])
        by_name = {image.name: image for image in output.images}
        for name, (frame, sensor) in PUBLISH.parse_manifest(self.manifest).assignments.items():
            image = by_name[name]
            expected = compose(*self.sensors[sensor], *self.true_poses[frame])
            self.assertTrue(PUBLISH._quat_close(image.q, expected[0], 1e-12, 1e-12))
            self.assertTrue(PUBLISH._vec_close(image.t, expected[1], 1e-12, 1e-12))
        for point in output.points:
            self.assertEqual(point.xyz, self.true_points[point.point_id])
            self.assertAlmostEqual(point.error, 0.0, places=12)

    def test_bad_sha_anchor_depth_duplicate_and_output_collisions_are_rejected(self):
        original_state = self.state.read_text()
        self.state.write_text(original_state.replace("SOURCE_SHA256 ", "SOURCE_SHA256 " + "0" * 64 + " #", 1))
        with self.assertRaises(PUBLISH.PublisherError):
            PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=self.out)
        self.assertFalse(self.out.exists())
        self.state.write_text(original_state)
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "sentinel").write_text("keep")
        with self.assertRaises(PUBLISH.PublisherError):
            PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=existing)
        self.assertEqual((existing / "sentinel").read_text(), "keep")
        symlink = self.root / "link-out"
        symlink.symlink_to(existing, target_is_directory=True)
        with self.assertRaises(PUBLISH.PublisherError):
            PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=symlink)
        self.assertTrue(symlink.is_symlink())

    def test_duplicate_state_pose_and_bad_final_depth_are_rejected_without_output(self):
        text = self.state.read_text()
        duplicate = text.replace("POSE_STATE_END", "POSE 0 1 0 0 0 0 0 0\nPOSE_STATE_END", 1)
        self.state.write_text(duplicate)
        with self.assertRaises(PUBLISH.PublisherError):
            PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=self.out)
        self.assertFalse(self.out.exists())
        self.state.write_text(text.replace("LANDMARK 1 0 0.10000000000000001 4", "LANDMARK 1 0 0.10000000000000001 -4", 1))
        with self.assertRaises(PUBLISH.PublisherError):
            PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=self.out)
        self.assertFalse(self.out.exists())

    def test_state_anchor_cost_and_id_contracts_are_strict(self):
        original = self.state.read_text()
        changed_anchor = original.replace("POSE 0 1 0 0 0 0 0 0", "POSE 0 0.999999 0 0 0.001 0 0 0", 1)
        self.state.write_text(changed_anchor)
        with self.assertRaises(PUBLISH.PublisherError):
            PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=self.out)
        self.assertFalse(self.out.exists())

        changed_cost = original.replace("FINAL_FULL_SQUARED_COST 0\nFINAL_FULL_SQUARED_COST_BITS 0", f"FINAL_FULL_SQUARED_COST 1\nFINAL_FULL_SQUARED_COST_BITS {bits(1.0)}", 1).replace("CERES_SUMMARY_FINAL_HALF_COST 0", "CERES_SUMMARY_FINAL_HALF_COST 0.5", 1)
        self.state.write_text(changed_cost)
        with self.assertRaises(PUBLISH.PublisherError):
            PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=self.out)
        self.assertFalse(self.out.exists())

        missing = original.replace("POSE 1 0.99988750210935917 0 0 0.01499943750632809 0.10000000000000001 0.02 0.01\n", "", 1)
        self.state.write_text(missing)
        with self.assertRaises(PUBLISH.PublisherError):
            PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=self.out)
        self.assertFalse(self.out.exists())

        extra = original.replace("POSE_STATE_END", "POSE 2 1 0 0 0 0 0 0\nPOSE_STATE_END", 1)
        self.state.write_text(extra)
        with self.assertRaises(PUBLISH.PublisherError):
            PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=self.out)
        self.assertFalse(self.out.exists())

    def test_owned_staging_is_removed_after_validation_failure(self):
        original_validator = PUBLISH._validate_staged_model

        def fail_after_staging(*_args, **_kwargs):
            raise PUBLISH.PublisherError("synthetic staged validation failure")

        PUBLISH._validate_staged_model = fail_after_staging
        try:
            with self.assertRaises(PUBLISH.PublisherError):
                PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=self.out)
        finally:
            PUBLISH._validate_staged_model = original_validator
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob(".published.staging-*")), [])

    def test_link_failure_does_not_delete_unowned_concurrent_file(self):
        original_link = PUBLISH.os.link
        calls = 0

        def link_with_unowned_file(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                destination.parent.joinpath("concurrent-sentinel").write_text("keep")
                raise OSError("synthetic concurrent link failure")
            return original_link(source, destination)

        PUBLISH.os.link = link_with_unowned_file
        try:
            with self.assertRaises(OSError):
                PUBLISH.publish_model(fixture_path=self.fixture, state_path=self.state, model_path=self.model, rig_manifest_path=self.manifest, out_dir=self.out)
        finally:
            PUBLISH.os.link = original_link
        self.assertTrue((self.out / "concurrent-sentinel").is_file())
        (self.out / "concurrent-sentinel").unlink()
        self.out.rmdir()


if __name__ == "__main__":
    unittest.main()
