from pathlib import Path
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scale_stereo_baseline import baseline_m, main, scale_baseline


def calibration():
    camera = {"qx": 0.0055, "qy": 0.0014, "qz": 0.9999806, "qw": 0.0025}
    return {
        "value0": {
            "T_imu_cam": [
                {"px": 0.0107, "py": -0.00005, "pz": -0.00012, **camera},
                {"px": -0.0533, "py": 0.00011, "pz": 0.00019, **camera},
            ],
            "intrinsics": [{"camera_type": "ds", "intrinsics": {"fx": 284.8}}],
            "accel_noise_std": [0.016, 0.016, 0.016],
        }
    }


class ScaleStereoBaselineTests(unittest.TestCase):
    def test_moves_the_second_camera_toward_the_first(self):
        original = calibration()
        scaled = scale_baseline(original, 0.986)
        self.assertAlmostEqual(baseline_m(scaled), 0.986 * baseline_m(original), places=12)
        first, second = scaled["value0"]["T_imu_cam"]
        self.assertEqual(first, original["value0"]["T_imu_cam"][0])
        for key in ("qx", "qy", "qz", "qw"):
            self.assertEqual(second[key], original["value0"]["T_imu_cam"][1][key])
        self.assertEqual(scaled["value0"]["intrinsics"], original["value0"]["intrinsics"])
        self.assertEqual(scaled["value0"]["accel_noise_std"], [0.016, 0.016, 0.016])
        # The input is left alone.
        self.assertEqual(original, calibration())

    def test_reads_a_calibration_without_value0(self):
        flat = calibration()["value0"]
        self.assertAlmostEqual(baseline_m(scale_baseline(flat, 1.02)), 1.02 * baseline_m(flat), places=12)

    def test_rejects_bad_input(self):
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                scale_baseline(calibration(), scale)
        mono = calibration()
        mono["value0"]["T_imu_cam"] = mono["value0"]["T_imu_cam"][:1]
        with self.assertRaises(ValueError):
            scale_baseline(mono, 0.99)

    def test_main_writes_the_scaled_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "calibration.json", Path(tmp) / "scaled.json"
            source.write_text(json.dumps(calibration()))
            printed = io.StringIO()
            with redirect_stdout(printed):
                main(["--calibration", str(source), "--scale", "0.98", "--out", str(out)])
            written = json.loads(out.read_text())
        self.assertAlmostEqual(baseline_m(written), 0.98 * baseline_m(calibration()), places=12)
        self.assertIn("64.00 mm -> 62.72 mm", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
