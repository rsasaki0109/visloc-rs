import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_sfm_temporal_support.py"
SPEC = importlib.util.spec_from_file_location("diagnose_sfm_temporal_support", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_analyze_component_reports_temporal_support_and_exact_residual(tmp_path):
    (tmp_path / "cameras.txt").write_text(
        "1 SIMPLE_PINHOLE 640 480 1 0 0\n", encoding="utf-8"
    )
    (tmp_path / "images.txt").write_text(
        "# image records\n"
        "1 1 0 0 0 0 0 0 1 cam1_000000.png\n"
        "0 0 1\n"
        "2 1 0 0 0 -1 0 0 1 cam1_000002.png\n"
        "-0.2 0 1\n",
        encoding="utf-8",
    )
    (tmp_path / "points3D.txt").write_text(
        "1 0 0 5 255 255 255 0 1 0 2 0\n", encoding="utf-8"
    )

    result = MODULE.analyze_component(tmp_path, 500)

    assert result["registered_images"] == 2
    assert result["points"] == 1
    assert result["invalid_projections"] == 0
    assert result["reprojection_px"]["count"] == 2
    assert result["reprojection_px"]["mean"] == 0.0
    assert result["reprojection_by_track_span"]["1-7"]["count"] == 2
    assert result["bins"][0]["tracks_anchored"] == 1
    assert result["bins"][0]["track_observations"]["median"] == 2.0
    assert result["bins"][0]["track_span_frames"]["median"] == 1.0


def test_percentile_interpolates_and_handles_empty_values():
    assert MODULE.percentile([], 0.5) is None
    assert MODULE.percentile([0.0, 10.0], 0.95) == 9.5


def test_track_span_classes_match_mapper_diagnostic_boundaries():
    assert [MODULE.track_span_class(value) for value in (0, 1, 7, 8, 15, 16, 31, 32, 127, 128)] == [
        "same-frame",
        "1-7",
        "1-7",
        "8-15",
        "8-15",
        "16-31",
        "16-31",
        "32-127",
        "32-127",
        "128+",
    ]
