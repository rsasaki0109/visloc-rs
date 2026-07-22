#!/usr/bin/env python3
"""Run a same-input monocular Sequential SfM head-to-head on EuRoC.

Both engines consume the same rectified cam0 frames. Ground truth is only read
after both engines exit. Benchmark artifacts must live on E: because C: is not
large enough for generated image/features/models.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from benchmark_process_metrics import run_monitored
except ModuleNotFoundError:  # Imported as `scripts.run_mono_sfm_headtohead` in tests.
    from scripts.benchmark_process_metrics import run_monitored


REPO = Path(__file__).resolve().parents[1]


def run_logged(argv: list[str], log: Path, *, cwd: Path = REPO) -> float:
    return run_logged_measured(argv, log, cwd=cwd, sample_rss=False)["wall_seconds"]


def run_logged_measured(
    argv: list[str],
    log: Path,
    *,
    cwd: Path = REPO,
    sample_rss: bool = True,
    poll_seconds: float = 0.5,
) -> dict:
    if not sample_rss:
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with log.open("w", encoding="utf-8") as stream:
            stream.write("COMMAND: " + subprocess.list2cmdline(argv) + "\n\n")
            stream.flush()
            completed = subprocess.run(
                argv,
                cwd=cwd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        measured = {
            "command": argv,
            "returncode": completed.returncode,
            "wall_seconds": time.perf_counter() - start,
            "peak_process_tree_rss_bytes": None,
            "idle_gpu_memory_mib": None,
            "peak_global_gpu_memory_mib": None,
            "resource_poll_seconds": None,
        }
    else:
        measured = run_monitored(argv, log, cwd=cwd, poll_seconds=poll_seconds)
    if measured["returncode"] != 0:
        raise RuntimeError(
            f"command failed ({measured['returncode']}); see {log}: "
            f"{subprocess.list2cmdline(argv)}"
        )
    return measured


def is_unsupported_caspar_failure(requested_backend: str, log: Path) -> bool:
    return (
        requested_backend == "CASPAR"
        and log.is_file()
        and "ba_global_backend != BundleAdjustmentBackend::CASPAR"
        in log.read_text(encoding="utf-8", errors="replace")
    )


def parse_pinhole(calib: Path) -> tuple[float, float, float, float]:
    line = next(line for line in calib.read_text(encoding="utf-8").splitlines() if line.startswith("P0:"))
    values = [float(value) for value in line.split()[1:]]
    return values[0], values[5], values[2], values[6]


def registered_images(images_txt: Path) -> int:
    count = 0
    for line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 10:
            continue
        try:
            int(fields[0])
            [float(value) for value in fields[1:8]]
            int(fields[8])
        except ValueError:
            continue
        count += 1
    return count


def point_statistics(points_txt: Path) -> tuple[int, float | None]:
    errors = []
    for line in points_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 8:
            errors.append(float(fields[7]))
    return len(errors), (sum(errors) / len(errors) if errors else None)


def reported_mean_reprojection(log: Path) -> float | None:
    """Read the reconstruction's aggregate reprojection error from its log.

    visloc's COLMAP text writer currently emits zero in the per-point ERROR
    field, so averaging points3D.txt would silently report a misleading 0 px.
    The reconstruction summary is the authoritative aggregate until the writer
    persists real per-point residuals.
    """
    match = re.search(
        r"mean reproj\s+([0-9]+(?:\.[0-9]+)?)\s+px",
        log.read_text(encoding="utf-8", errors="replace"),
    )
    return float(match.group(1)) if match else None


def select_colmap_model(
    colmap: Path, sparse: Path, text_root: Path, log_root: Path
) -> Path:
    candidates = sorted(path for path in sparse.iterdir() if path.is_dir())
    if not candidates:
        raise RuntimeError(f"COLMAP produced no sparse model under {sparse}")
    scored = []
    for candidate in candidates:
        text_model = text_root / candidate.name
        text_model.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                str(colmap),
                "model_converter",
                "--input_path",
                str(candidate),
                "--output_path",
                str(text_model),
                "--output_type",
                "TXT",
            ],
            log_root / f"colmap_model_converter_{candidate.name}.log",
        )
        count = registered_images(text_model / "images.txt")
        scored.append((count, text_model))
    return max(scored, key=lambda item: item[0])[1]


def capture_registry(
    python: Path,
    registry_dir: Path,
    sequence: str,
    mav0: Path,
    frames: int,
    summary_path: Path,
    summary: dict,
    visloc_model: Path,
    colmap_model: Path,
) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    common = [
        str(python),
        str(REPO / "scripts" / "benchmark_registry.py"),
        "capture",
        "--benchmark-id",
        "mono-sequential-sfm-vs-colmap",
        "--benchmark-name",
        "Monocular Sequential SfM vs COLMAP 4.1",
        "--script",
        "scripts/run_mono_sfm_headtohead.py",
        "--protocol",
        "Same rectified EuRoC cam0 frames, mono-to-mono, Sim(3) camera trajectory against EuRoC ground truth; GT read only after both engines exit",
        "--docs",
        "docs/visual_slam_sequential_sfm_plan.md",
        "--dataset-name",
        "EuRoC MAV",
        "--dataset-sequence",
        sequence,
        "--dataset-version",
        f"first {frames} synchronized cam0 frames",
        "--dataset-path",
        str(mav0),
        "--claim-scope",
        "exploratory",
        "--status",
        "success",
        "--command",
        subprocess.list2cmdline(sys.argv),
        "--artifact",
        f"summary={summary_path}",
        "--artifact",
        f"evaluation={summary_path.parent / 'evaluation.json'}",
    ]
    registry_dir.mkdir(parents=True, exist_ok=True)
    for engine, model, result_kind in (
        ("visloc", visloc_model, "visloc_run"),
        ("colmap", colmap_model, "external_rerun"),
    ):
        result = summary["engines"][engine]
        evaluation = result["evaluation"]
        argv = common + [
            "--out",
            str(registry_dir / f"mono-sfm-{engine}-{sequence}-{stamp}.json"),
            "--run-id",
            f"mono-sfm-{engine}-{sequence}-{stamp}",
            "--result-kind",
            result_kind,
            "--config",
            f"engine={json.dumps(engine)}",
            "--config",
            f"frames={frames}",
            "--config",
            'sensor_mode="monocular"',
            "--config",
            "resource_poll_seconds="
            f"{summary['resource_sampling']['poll_seconds']}",
            "--metric",
            f"wall_clock_s={result['wall_seconds']}:s",
            "--metric",
            f"registered_frames={result['registered']}:count",
            "--metric",
            f"registration_rate={result['registered'] / frames}:ratio",
            "--metric",
            f"ate_sim3_m={evaluation['ate_translation_sim3_m']['rmse']}:m",
            "--metric",
            f"sim3_scale={evaluation['sim3_scale']}:ratio",
            "--metric",
            f"mean_reprojection_px={result['mean_reprojection_px']}:px",
            "--metric",
            f"points3d={result['points3d']}:count",
            "--primary-metric",
            "ate_sim3_m",
            "--artifact",
            f"images={model / 'images.txt'}",
            "--artifact",
            f"points3d={model / 'points3D.txt'}",
            "--notes",
            "Same-input monocular head-to-head; shared rectification and Rust build time excluded from engine wall time.",
        ]
        if result["peak_process_tree_rss_bytes"] is not None:
            argv.extend(
                [
                    "--metric",
                    "peak_process_tree_rss_bytes="
                    f"{result['peak_process_tree_rss_bytes']}:bytes",
                ]
            )
        if result["peak_global_gpu_memory_mib"] is not None:
            argv.extend(
                [
                    "--metric",
                    "peak_global_gpu_memory_mib="
                    f"{result['peak_global_gpu_memory_mib']}:MiB",
                ]
            )
        if engine == "visloc":
            argv.extend(
                [
                    "--config",
                    f"next_image_policy={json.dumps(result['next_image_policy'])}",
                    "--config",
                    f"skip_offsets={json.dumps(result['skip_offsets'])}",
                    "--config",
                    f"skip_stride={result['skip_stride']}",
                    "--config",
                    f"pose_graph_offsets={json.dumps(result['pose_graph_offsets'])}",
                    "--config",
                    f"pose_graph_stride={result['pose_graph_stride']}",
                    "--config",
                    "wide_hypothesis=" + json.dumps(result["wide_hypothesis"]),
                    "--config",
                    "post_refinement_registration="
                    + json.dumps(result["post_refinement_registration"]),
                    "--config",
                    "geometry_conflict_recovery="
                    + json.dumps(result["geometry_conflict_recovery"]),
                    "--config",
                    "structureless_registration="
                    + json.dumps(result["structureless_registration"]),
                ]
            )
        else:
            argv.extend(
                [
                    "--config",
                    "ba_global_backend_requested="
                    + json.dumps(result["ba_global_backend_requested"]),
                    "--config",
                    "ba_global_backend=" + json.dumps(result["ba_global_backend"]),
                    "--config",
                    "caspar_dnf_unsupported="
                    + json.dumps(result["caspar_dnf_unsupported"]),
                ]
            )
        subprocess.run(argv, cwd=REPO, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mav0", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--cargo",
        type=Path,
        default=Path.home() / ".cargo" / "bin" / "cargo.exe",
    )
    parser.add_argument(
        "--colmap",
        type=Path,
        default=Path("E:/tools/colmap/bin/colmap.exe"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--skip-offsets",
        default="",
        help="comma-separated ordered-view-graph skip offsets",
    )
    parser.add_argument("--skip-stride", type=int, default=1)
    parser.add_argument(
        "--pose-graph-offsets",
        default="",
        help="comma-separated offsets matched separately for a candidate hypothesis",
    )
    parser.add_argument("--pose-graph-stride", type=int, default=1)
    parser.add_argument("--wide-hypothesis", action="store_true")
    parser.add_argument("--post-refinement-registration", action="store_true")
    parser.add_argument(
        "--geometry-conflict-recovery",
        action="store_true",
        help="enable guarded geometry-guided recovery of conflicted tracks",
    )
    parser.add_argument(
        "--structureless-registration",
        action="store_true",
        help="enable guarded multi-neighbour relative-pose registration",
    )
    parser.add_argument(
        "--visloc-next-image-policy",
        choices=("visibility", "count"),
        default="visibility",
    )
    parser.add_argument("--colmap-feature", choices=("SIFT", "ALIKED"), default="SIFT")
    parser.add_argument("--colmap-backend", choices=("CERES", "CASPAR"), default="CERES")
    parser.add_argument(
        "--resource-poll-seconds",
        type=float,
        default=0.5,
        help="process-tree working-set sampling interval",
    )
    parser.add_argument("--capture-registry", action="store_true")
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=REPO / "benchmarks" / "registry" / "runs" / "euroc",
    )
    args = parser.parse_args()

    if args.frames < 3:
        parser.error("--frames must be at least 3")
    if args.skip_stride < 1:
        parser.error("--skip-stride must be at least 1")
    if args.pose_graph_stride < 1:
        parser.error("--pose-graph-stride must be at least 1")
    if not 0.1 <= args.resource_poll_seconds <= 10.0:
        parser.error("--resource-poll-seconds must be in [0.1, 10.0]")
    if args.wide_hypothesis and not args.pose_graph_offsets:
        parser.error("--wide-hypothesis requires --pose-graph-offsets")
    if args.out_dir.drive.upper() != "E:":
        parser.error("--out-dir must be on E: (benchmark outputs must not use C:)")
    if args.out_dir.exists():
        launcher_files = {"runner.stdout.log", "runner.stderr.log", "runner.pid"}
        unexpected = [path for path in args.out_dir.iterdir() if path.name not in launcher_files]
        if unexpected:
            parser.error("--out-dir must be fresh; unexpected existing path: " + str(unexpected[0]))
    for executable in (args.python, args.cargo, args.colmap):
        if not executable.is_file():
            parser.error(f"executable not found: {executable}")

    out = args.out_dir.resolve()
    logs = out / "logs"
    rect = out / "rect"
    features = out / "visloc_features"
    visloc_model = out / "visloc_model"
    colmap_root = out / "colmap"
    sparse = colmap_root / "sparse"
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir()

    stage_seconds: dict[str, float] = {}
    stage_peak_process_tree_rss_bytes: dict[str, int | None] = {}
    stage_resource_metrics: dict[str, dict] = {}

    def run_measured_stage(key: str, command: list[str], log: Path) -> None:
        measured = run_logged_measured(
            command,
            log,
            poll_seconds=args.resource_poll_seconds,
        )
        stage_seconds[key] = measured["wall_seconds"]
        stage_peak_process_tree_rss_bytes[key] = measured[
            "peak_process_tree_rss_bytes"
        ]
        stage_resource_metrics[key] = measured
    stage_seconds["shared_rectify"] = run_logged(
        [
            str(args.python),
            str(REPO / "scripts" / "rectify_euroc_stereo.py"),
            "--mav0",
            str(args.mav0),
            "--out-dir",
            str(rect),
            "--frames",
            str(args.frames),
            "--left-only",
        ],
        logs / "rectify.log",
    )
    images = rect / "image_0"
    if len(list(images.glob("*.png"))) != args.frames:
        raise RuntimeError("rectification did not produce the requested frame count")
    fx, fy, cx, cy = parse_pinhole(rect / "calib.txt")

    run_measured_stage(
        "visloc_feature_extraction",
        [
            str(args.python),
            str(REPO / "scripts" / "export_superpoint_lightglue.py"),
            "--mono-dir",
            str(images),
            "--out-dir",
            str(features),
            "--frames",
            str(args.frames),
            "--device",
            args.device,
            "--max-keypoints",
            str(args.max_keypoints),
        ],
        logs / "visloc_feature_extraction.log",
    )
    run_logged(
        [
            str(args.cargo),
            "build",
            "--release",
            "--example",
            "sequential_sfm_demo",
        ],
        logs / "visloc_build.log",
    )
    visloc_exe = REPO / "target" / "release" / "examples" / "sequential_sfm_demo.exe"
    visloc_mapping_command = [
            str(visloc_exe),
            "--features-dir",
            str(features),
            "--out-colmap",
            str(visloc_model),
            "--width",
            "752",
            "--height",
            "480",
            "--fx",
            str(fx),
            "--fy",
            str(fy),
            "--cx",
            str(cx),
            "--cy",
            str(cy),
            "--window",
            str(args.window),
            "--min-matches",
            "30",
            "--colmap-style",
            "--next-image-policy",
            args.visloc_next_image_policy,
        ]
    if args.skip_offsets:
        visloc_mapping_command.extend(
            ["--skip-offsets", args.skip_offsets, "--skip-stride", str(args.skip_stride)]
        )
    if args.pose_graph_offsets:
        visloc_mapping_command.extend(
            [
                "--pose-graph-offsets",
                args.pose_graph_offsets,
                "--pose-graph-stride",
                str(args.pose_graph_stride),
            ]
        )
    if args.wide_hypothesis:
        visloc_mapping_command.append("--wide-hypothesis")
    if args.post_refinement_registration:
        visloc_mapping_command.append("--post-refinement-registration")
    if args.geometry_conflict_recovery:
        visloc_mapping_command.append("--geometry-conflict-recovery")
    if args.structureless_registration:
        visloc_mapping_command.append("--structureless-registration")
    run_measured_stage(
        "visloc_mapping",
        visloc_mapping_command,
        logs / "visloc_mapping.log",
    )

    colmap_root.mkdir()
    sparse.mkdir()
    database = colmap_root / "database.db"
    run_measured_stage(
        "colmap_feature_extraction",
        [
            str(args.colmap),
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.camera_params",
            f"{fx},{fy},{cx},{cy}",
            "--FeatureExtraction.type",
            args.colmap_feature,
            "--FeatureExtraction.use_gpu",
            "1",
        ],
        logs / "colmap_feature_extraction.log",
    )
    match_type = "SIFT_BRUTEFORCE" if args.colmap_feature == "SIFT" else "ALIKED_LIGHTGLUE"
    run_measured_stage(
        "colmap_matching",
        [
            str(args.colmap),
            "sequential_matcher",
            "--database_path",
            str(database),
            "--FeatureMatching.type",
            match_type,
            "--FeatureMatching.use_gpu",
            "1",
        ],
        logs / "colmap_matching.log",
    )
    colmap_backend_requested = args.colmap_backend
    colmap_backend_effective = args.colmap_backend
    colmap_caspar_dnf = False

    def mapper_command(output_path: Path, backend: str) -> list[str]:
        return [
            str(args.colmap),
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(output_path),
            "--Mapper.random_seed",
            "0",
            "--Mapper.ba_local_backend",
            # COLMAP 4.1 explicitly rejects CASPAR for local BA. Caspar is the
            # global backend; local refinement remains Ceres.
            "CERES",
            "--Mapper.ba_global_backend",
            backend,
            "--Mapper.ba_use_gpu",
            "1" if backend == "CASPAR" else "0",
        ]

    try:
        run_measured_stage(
            "colmap_mapping",
            mapper_command(sparse, args.colmap_backend),
            logs / "colmap_mapping.log",
        )
    except RuntimeError:
        caspar_log = logs / "colmap_mapping.log"
        unsupported_caspar = is_unsupported_caspar_failure(
            args.colmap_backend, caspar_log
        )
        if not unsupported_caspar:
            raise
        colmap_caspar_dnf = True
        colmap_backend_effective = "CERES"
        sparse = colmap_root / "sparse_ceres"
        sparse.mkdir()
        run_measured_stage(
            "colmap_mapping",
            mapper_command(sparse, colmap_backend_effective),
            logs / "colmap_mapping_ceres_fallback.log",
        )
    colmap_model = select_colmap_model(
        args.colmap, sparse, colmap_root / "text_models", logs
    )

    visloc_tum = out / "visloc.tum"
    colmap_tum = out / "colmap.tum"
    for model, trajectory, name in (
        (visloc_model, visloc_tum, "visloc"),
        (colmap_model, colmap_tum, "colmap"),
    ):
        run_logged(
            [
                str(args.python),
                str(REPO / "scripts" / "colmap_images_to_tum.py"),
                str(model / "images.txt"),
                str(rect / "timestamps.txt"),
                str(trajectory),
            ],
            logs / f"{name}_trajectory_conversion.log",
        )

    evaluation = out / "evaluation.json"
    run_logged(
        [
            str(args.python),
            str(REPO / "scripts" / "evaluate_euroc_trajectory.py"),
            "--ground-truth-csv",
            str(args.mav0 / "state_groundtruth_estimate0" / "data.csv"),
            "--trajectory",
            str(visloc_tum),
            "--trajectory",
            str(colmap_tum),
            "--tum-time-unit",
            "s",
            "--common-estimate-timestamps",
            "--out-json",
            str(evaluation),
        ],
        logs / "evaluation.log",
    )
    evaluated = json.loads(evaluation.read_text(encoding="utf-8"))["runs"]
    visloc_points, _ = point_statistics(visloc_model / "points3D.txt")
    visloc_reproj = reported_mean_reprojection(logs / "visloc_mapping.log")
    colmap_points, colmap_reproj = point_statistics(colmap_model / "points3D.txt")

    def max_stage_metric(stage_keys: tuple[str, ...], metric: str) -> int | None:
        values = [
            stage_resource_metrics[key].get(metric)
            for key in stage_keys
            if stage_resource_metrics[key].get(metric) is not None
        ]
        return max(values) if values else None

    visloc_stages = ("visloc_feature_extraction", "visloc_mapping")
    colmap_stages = (
        "colmap_feature_extraction",
        "colmap_matching",
        "colmap_mapping",
    )
    summary = {
        "protocol": "same rectified EuRoC cam0 frames; mono-to-mono; GT read after engine exit",
        "frames": args.frames,
        "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": 752, "height": 480},
        "stage_seconds": stage_seconds,
        "resource_sampling": {
            "metric": "process_tree_working_set_bytes",
            "poll_seconds": args.resource_poll_seconds,
            "platform_supported": os.name == "nt",
        },
        "stage_peak_process_tree_rss_bytes": stage_peak_process_tree_rss_bytes,
        "stage_resource_metrics": stage_resource_metrics,
        "engines": {
            "visloc": {
                "next_image_policy": args.visloc_next_image_policy,
                "skip_offsets": args.skip_offsets,
                "skip_stride": args.skip_stride,
                "pose_graph_offsets": args.pose_graph_offsets,
                "pose_graph_stride": args.pose_graph_stride,
                "wide_hypothesis": args.wide_hypothesis,
                "post_refinement_registration": args.post_refinement_registration,
                "geometry_conflict_recovery": args.geometry_conflict_recovery,
                "structureless_registration": args.structureless_registration,
                "registered": registered_images(visloc_model / "images.txt"),
                "points3d": visloc_points,
                "mean_reprojection_px": visloc_reproj,
                "wall_seconds": stage_seconds["visloc_feature_extraction"] + stage_seconds["visloc_mapping"],
                "peak_process_tree_rss_bytes": max_stage_metric(
                    visloc_stages, "peak_process_tree_rss_bytes"
                ),
                "peak_global_gpu_memory_mib": max_stage_metric(
                    visloc_stages, "peak_global_gpu_memory_mib"
                ),
                "peak_global_gpu_memory_delta_mib": max_stage_metric(
                    visloc_stages, "peak_global_gpu_memory_delta_mib"
                ),
                "evaluation": evaluated[0],
            },
            "colmap": {
                "version": "4.1.0 fa8e3b3",
                "feature": args.colmap_feature,
                "ba_local_backend": "CERES",
                "ba_global_backend_requested": colmap_backend_requested,
                "ba_global_backend": colmap_backend_effective,
                "caspar_dnf_unsupported": colmap_caspar_dnf,
                "registered": registered_images(colmap_model / "images.txt"),
                "points3d": colmap_points,
                "mean_reprojection_px": colmap_reproj,
                "wall_seconds": stage_seconds["colmap_feature_extraction"] + stage_seconds["colmap_matching"] + stage_seconds["colmap_mapping"],
                "peak_process_tree_rss_bytes": max_stage_metric(
                    colmap_stages, "peak_process_tree_rss_bytes"
                ),
                "peak_global_gpu_memory_mib": max_stage_metric(
                    colmap_stages, "peak_global_gpu_memory_mib"
                ),
                "peak_global_gpu_memory_delta_mib": max_stage_metric(
                    colmap_stages, "peak_global_gpu_memory_delta_mib"
                ),
                "evaluation": evaluated[1],
            },
        },
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.capture_registry:
        capture_registry(
            args.python,
            args.registry_dir,
            args.mav0.parent.name,
            args.mav0,
            args.frames,
            summary_path,
            summary,
            visloc_model,
            colmap_model,
        )
    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
