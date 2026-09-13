import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score_openloris_model as om  # noqa: E402
import score_openloris_rpe as rpe  # noqa: E402


IDENTITY_TRANSFORM_YAML = (
    "parent_frame: base_link\n"
    "child_frame: t265_fisheye1_optical_frame\n"
    "matrix:\n"
    "  data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]\n"
    "parent_frame: t265_fisheye1_optical_frame\n"
    "child_frame: t265_fisheye2_optical_frame\n"
    "matrix:\n"
    "  data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]\n"
)


def _write_manifest(path: Path, images) -> None:
    payload = {
        "schema": "visloc_openloris_corridor_manifest_v1",
        "images": [
            {"name": name, "camera": camera, "timestamp": timestamp}
            for name, camera, timestamp in images
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_ground_truth(path: Path, samples) -> None:
    lines = [f"{t} {x} {y} {z} 0 0 0 1" for t, (x, y, z) in samples]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_transform(path: Path) -> None:
    path.write_text(IDENTITY_TRANSFORM_YAML, encoding="utf-8")


def _write_images_txt(path: Path, rows) -> None:
    """rows: iterable of (image_id, (cx, cy, cz), name); rotation is fixed to
    identity, so translation = -center exactly reproduces the given center
    through score_openloris_model.load_model_centres' -R^T @ T convention."""
    lines = []
    for image_id, (cx, cy, cz), name in rows:
        lines.append(f"{image_id} 1 0 0 0 {-cx} {-cy} {-cz} 1 {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _buffered_gt_samples(ts, position_fn):
    """GT samples for ts plus one extra point before/after so every ts query
    lands strictly inside the GT range (exact match => fraction=1.0, i.e. no
    approximation from interpolation, regardless of position_fn's shape)."""
    all_ts = [ts[0] - 1] + list(ts) + [ts[-1] + 1]
    return [(float(t), position_fn(t)) for t in all_ts]


class RpeScorerTests(unittest.TestCase):
    def test_perfect_model_zero_rmse_full_coverage(self):
        ts = list(range(0, 21))

        def position(t):
            return (float(t), 0.3 * t, -0.2 * t)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, gt_path, transform_path, model_path = (
                root / "manifest.json",
                root / "gt.txt",
                root / "transform.yaml",
                root / "model.txt",
            )
            _write_manifest(manifest_path, [(f"c{t}.png", 1, float(t)) for t in ts])
            _write_ground_truth(gt_path, _buffered_gt_samples(ts, position))
            _write_transform(transform_path)
            _write_images_txt(model_path, [(t + 1, position(t), f"c{t}.png") for t in ts])

            result = rpe.score(
                [model_path], manifest_path, gt_path, transform_path, max_gap_seconds=1.5
            )

        self.assertEqual(result["cam1_gt_interpolated_images"], 21)
        windows = {w["window_seconds"]: w for w in result["windows"]}
        self.assertEqual(windows[1.0]["reference_pairs"], 20)
        self.assertEqual(windows[1.0]["evaluable_pairs"], 20)
        self.assertAlmostEqual(windows[1.0]["coverage"], 1.0)
        self.assertAlmostEqual(windows[1.0]["rmse_m"], 0.0, places=9)
        self.assertAlmostEqual(windows[1.0]["median_m"], 0.0, places=9)
        self.assertAlmostEqual(windows[1.0]["p95_m"], 0.0, places=9)

        self.assertEqual(windows[10.0]["reference_pairs"], 11)
        self.assertEqual(windows[10.0]["evaluable_pairs"], 11)
        self.assertAlmostEqual(windows[10.0]["coverage"], 1.0)
        self.assertAlmostEqual(windows[10.0]["rmse_m"], 0.0, places=9)

    def test_two_components_coverage_drops_by_cross_boundary_pairs(self):
        ts = list(range(0, 21))

        def position(t):
            return (float(t), 0.3 * t, -0.2 * t)

        rotation_b = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        scale_b, translation_b = 2.0, np.asarray([5.0, 5.0, 5.0])

        def raw_b(t):
            centre = np.asarray(position(t))
            return rotation_b.T @ ((centre - translation_b) / scale_b)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, gt_path, transform_path = (
                root / "manifest.json",
                root / "gt.txt",
                root / "transform.yaml",
            )
            path_a, path_b = root / "a.txt", root / "b.txt"
            ts_a, ts_b = ts[:10], ts[10:]

            _write_manifest(manifest_path, [(f"c{t}.png", 1, float(t)) for t in ts])
            _write_ground_truth(gt_path, _buffered_gt_samples(ts, position))
            _write_transform(transform_path)
            _write_images_txt(path_a, [(t + 1, position(t), f"c{t}.png") for t in ts_a])
            _write_images_txt(
                path_b, [(t + 1, tuple(raw_b(t)), f"c{t}.png") for t in ts_b]
            )

            result = rpe.score(
                [path_a, path_b], manifest_path, gt_path, transform_path, max_gap_seconds=1.5
            )

        # Both components are individually perfect up to their own Sim(3).
        for component in result["components_ate_sanity"]:
            self.assertAlmostEqual(component["ate_rmse_m"], 0.0, places=6)

        windows = {w["window_seconds"]: w for w in result["windows"]}

        # Window=1s: only the (t=9 -> t=10) pair crosses the A/B boundary.
        self.assertEqual(windows[1.0]["reference_pairs"], 20)
        self.assertEqual(windows[1.0]["evaluable_pairs"], 19)
        self.assertAlmostEqual(windows[1.0]["coverage"], 19 / 20)
        self.assertAlmostEqual(windows[1.0]["rmse_m"], 0.0, places=6)

        # Window=10s: i in {0..9} cross into B, only i=10 (both ends in B) stays.
        self.assertEqual(windows[10.0]["reference_pairs"], 11)
        self.assertEqual(windows[10.0]["evaluable_pairs"], 1)
        self.assertAlmostEqual(windows[10.0]["coverage"], 1 / 11)
        self.assertAlmostEqual(windows[10.0]["rmse_m"], 0.0, places=6)

    def test_uniform_scale_drift_matches_analytic_value(self):
        n, k = 11, 1.1  # t = 0..10, cam1 raw positions scaled by k relative to GT
        ts = list(range(0, n))

        def true_position(t):
            return (float(t), 0.0, 0.0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, gt_path, transform_path, model_path = (
                root / "manifest.json",
                root / "gt.txt",
                root / "transform.yaml",
                root / "model.txt",
            )
            _write_manifest(
                manifest_path,
                [(f"cam1_{t}.png", 1, float(t)) for t in ts]
                + [(f"cam2_{t}.png", 2, float(t)) for t in ts],
            )
            _write_ground_truth(gt_path, _buffered_gt_samples(ts, true_position))
            _write_transform(transform_path)
            rows = [(t + 1, (k * t, 0.0, 0.0), f"cam1_{t}.png") for t in ts] + [
                (n + t + 1, (float(t), 0.0, 0.0), f"cam2_{t}.png") for t in ts
            ]
            _write_images_txt(model_path, rows)

            result = rpe.score(
                [model_path], manifest_path, gt_path, transform_path, max_gap_seconds=1.5
            )

        # Independent closed-form least-squares scale for the mixed point
        # cloud {(t, t)} (cam2, true) union {(k*t, t)} (cam1, drifted):
        # s = (1+k)*V / (V*(1+k^2) + (Tsum^2/(2N))*(1-k)^2), V = Tsq - Tsum^2/N.
        Tsum = sum(ts)
        Tsq = sum(t * t for t in ts)
        V = Tsq - Tsum**2 / n
        analytic_scale = ((1 + k) * V) / (V * (1 + k**2) + (Tsum**2 / (2 * n)) * (1 - k) ** 2)

        reported_scale = result["components_ate_sanity"][0]["sim3_scale"]
        self.assertAlmostEqual(reported_scale, analytic_scale, places=9)

        # A pure scale-and-translation fit with rotation forced to identity
        # (all points lie on the x axis) leaves a constant residual velocity
        # error of |s*k - 1| per second, applied uniformly to every cam1 pair.
        analytic_error_per_second = abs(analytic_scale * k - 1.0)

        windows = {w["window_seconds"]: w for w in result["windows"]}
        self.assertEqual(windows[1.0]["reference_pairs"], n - 1)
        self.assertEqual(windows[1.0]["evaluable_pairs"], n - 1)
        for metric in ("rmse_m", "median_m", "p95_m"):
            self.assertAlmostEqual(windows[1.0][metric], analytic_error_per_second, places=9)

        self.assertEqual(windows[10.0]["reference_pairs"], 1)
        self.assertEqual(windows[10.0]["evaluable_pairs"], 1)
        for metric in ("rmse_m", "median_m", "p95_m"):
            self.assertAlmostEqual(
                windows[10.0][metric], analytic_error_per_second * 10.0, places=8
            )

        self.assertGreater(analytic_error_per_second, 1e-6)  # sanity: genuinely nonzero


class ReferencePairBuilderTests(unittest.TestCase):
    def test_closest_within_tolerance_chosen_once_per_i(self):
        times = {"a": 0.0, "b": 0.94, "c": 1.02, "d": 1.2, "e": 2.5}
        pairs = rpe.build_reference_pairs(times, 1.0)
        self.assertEqual(pairs, [("a", "c")])  # b is outside [0.95,1.05] tol of a+1


if __name__ == "__main__":
    unittest.main()
