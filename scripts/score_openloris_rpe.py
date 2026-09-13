#!/usr/bin/env python3
"""Relative-pose-error (RPE) scoring for OpenLORIS COLMAP text models.

This is a companion to ``score_openloris_model.py`` (the "ATE" scorer): it
reuses that module's manifest/ground-truth loading, timestamp interpolation,
image-to-timestamp mapping, camera-center convention, and per-component
Sim(3) fit *by import* -- it does not reimplement or modify any of that
logic.  What it adds is a translation relative-pose-error (RPE) metric over
fixed time windows, computed only on camera-1 (one camera of the stereo
rig) manifest images so that stereo pairs are not double counted.

Metric definition (pre-registered; see task record for the exact text):

* Let ``cam1_gt`` be the set of manifest images with ``camera == 1`` that
  are GT-interpolable (i.e. present in the dict returned by
  ``interpolate_camera_centres``).
* For a window ``W`` seconds, and tolerance ``tol = 0.1 * W``, the
  reference pair set ``P_W`` contains, for every ``i`` in ``cam1_gt``
  (ordered by timestamp), at most one pair ``(i, j)``: the ``j`` in
  ``cam1_gt`` whose timestamp is closest to ``t_i + W`` among all
  candidates with ``t_j - t_i`` in ``[W - tol/2, W + tol/2]``.  ``P_W`` is
  fixed by the manifest and ground truth alone (independent of the model
  being scored).
* A pair ``(i, j)`` in ``P_W`` is *evaluable* for a given model iff both
  ``i`` and ``j`` are registered images of the model and land in the same
  connected component (the same COLMAP text sub-model / same Sim(3)
  gauge).
* For an evaluable pair, using that component's Sim(3) fit
  ``(scale, rotation)`` from ``score_openloris_model.umeyama`` (translation
  cancels in a displacement), the translation RPE is::

      e = || scale * rotation @ (c_j - c_i)_est - (c_j - c_i)_gt ||

  where ``c_*`` are raw (unaligned) model camera centres for ``_est`` and
  GT-interpolated camera centres for ``_gt``.
* Rotation RPE is *not* computed: neither ``load_model_centres`` nor
  ``interpolate_camera_centres`` in ``score_openloris_model.py`` exposes
  orientation (only camera-center translation), so there is no orientation
  to reuse without reimplementing scorer internals. This is reported
  explicitly in the output (``rotation_rpe: null`` plus an explanatory
  note) rather than silently omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("score_openloris_rpe.py requires numpy") from exc

import score_openloris_model as om

ScoreError = om.ScoreError

DEFAULT_WINDOWS_SECONDS = (1.0, 10.0)


def parse_windows_seconds(text: str) -> list[float]:
    windows = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0:
            raise ScoreError(f"window seconds must be positive: {value!r}")
        windows.append(value)
    if not windows:
        raise ScoreError("no windows-seconds values supplied")
    return windows


def build_reference_pairs(cam1_times: dict[str, float], window_seconds: float) -> list[tuple[str, str]]:
    """P_W: for each i (by timestamp), the closest-to-(t_i+W) j within 10% of W."""
    tolerance = 0.1 * window_seconds
    low_offset, high_offset = window_seconds - 0.5 * tolerance, window_seconds + 0.5 * tolerance
    names_sorted = sorted(cam1_times, key=lambda name: (cam1_times[name], name))
    times = np.asarray([cam1_times[name] for name in names_sorted], dtype=float)
    pairs: list[tuple[str, str]] = []
    for index, name_i in enumerate(names_sorted):
        t_i = times[index]
        left = int(np.searchsorted(times, t_i + low_offset, side="left"))
        right = int(np.searchsorted(times, t_i + high_offset, side="right"))
        candidates = [k for k in range(left, right) if k != index]
        if not candidates:
            continue
        target = t_i + window_seconds
        best = min(candidates, key=lambda k: abs(times[k] - target))
        pairs.append((name_i, names_sorted[best]))
    return pairs


def _summarize(errors: list[float], reference_pairs: int) -> dict[str, Any]:
    evaluable = len(errors)
    coverage = float(evaluable) / reference_pairs if reference_pairs > 0 else None
    if evaluable > 0:
        arr = np.asarray(errors, dtype=float)
        rmse = float(np.sqrt(np.mean(arr**2)))
        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
    else:
        rmse = median = p95 = None
    return {
        "evaluable_pairs": evaluable,
        "reference_pairs": reference_pairs,
        "coverage": coverage,
        "rmse_m": rmse,
        "median_m": median,
        "p95_m": p95,
    }


def _load_components(
    model_paths: list[Path],
    reference: dict[str, np.ndarray],
    aliases: dict[str, str] | None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, np.ndarray]]:
    """Fit each component's Sim(3) exactly as score_openloris_model.score_component does.

    Returns (components, name -> component index, name -> raw (unaligned) model
    camera centre in that component's own frame).
    """
    components: list[dict[str, Any]] = []
    name_to_component: dict[str, int] = {}
    raw_positions: dict[str, np.ndarray] = {}
    registered_all: set[str] = set()
    for index, path in enumerate(model_paths):
        # Reuse score_component verbatim for the ATE-side numbers (and as a
        # sanity cross-check against the recorded score.json aggregate).
        ate_component, ate_errors, ate_common = om.score_component(path, reference, aliases)

        query_raw = om.load_model_centres(path)
        query = {
            (aliases.get(name, name) if aliases else name): centre
            for name, centre in query_raw.items()
        }
        if len(query) != len(query_raw):
            raise ScoreError(f"image aliases collapse multiple model names in {path}")
        duplicates = registered_all & set(query)
        if duplicates:
            raise ScoreError(f"images occur in multiple models; first duplicate={min(duplicates)!r}")
        registered_all.update(query)

        common = sorted(set(query) & set(reference))
        if common != ate_common:
            raise ScoreError(f"internal inconsistency recomputing common images for {path}")
        source = np.asarray([query[name] for name in common])
        destination = np.asarray([reference[name] for name in common])
        scale, rotation, translation = om.umeyama(source, destination)
        if abs(scale - ate_component["sim3_scale"]) > 1e-9 * max(1.0, abs(scale)):
            raise ScoreError(f"internal inconsistency recomputing Sim(3) scale for {path}")

        components.append(
            {
                "index": index,
                "images_txt": ate_component["images_txt"],
                "images_txt_sha256": ate_component["images_txt_sha256"],
                "registered": ate_component["registered"],
                "gt_scored": ate_component["gt_scored"],
                "sim3_scale": scale,
                "rotation_align": rotation,
                "translation_align": translation,
                "ate_rmse_m": ate_component["rmse_m"],
                "ate_median_m": ate_component["median_m"],
                "ate_p95_m": ate_component["p95_m"],
            }
        )
        for name in query:
            name_to_component[name] = index
            raw_positions[name] = query[name]
    return components, name_to_component, raw_positions


def score(
    model_paths: list[Path],
    manifest_path: Path,
    ground_truth_path: Path,
    transform_path: Path,
    *,
    alias_path: Path | None = None,
    max_gap_seconds: float = 0.1,
    windows_seconds: list[float] = list(DEFAULT_WINDOWS_SECONDS),
) -> dict[str, Any]:
    if max_gap_seconds <= 0:
        raise ScoreError("max interpolation gap must be positive")
    if not windows_seconds:
        raise ScoreError("no windows-seconds values supplied")

    manifest = om.load_manifest(manifest_path)
    ground_truth = om.load_ground_truth(ground_truth_path)
    extrinsics = om.load_camera_extrinsics(transform_path)
    # Reference camera centres for ALL manifest images (both cameras), exactly
    # as score_openloris_model.score() computes them -- this is what the
    # per-component Sim(3) fit aligns against, so RPE reuses the identical
    # alignment.
    reference = om.interpolate_camera_centres(
        manifest, ground_truth, extrinsics, max_gap_seconds=max_gap_seconds
    )
    aliases = om.load_colmap_aliases(alias_path) if alias_path else None

    cam1_gt_interp = sorted(
        name for name, (camera, _timestamp) in manifest.items() if camera == 1 and name in reference
    )
    cam1_times = {name: manifest[name][1] for name in cam1_gt_interp}

    components, name_to_component, raw_positions = _load_components(model_paths, reference, aliases)
    if not components:
        raise ScoreError("no COLMAP text models were supplied")

    windows_out = []
    for window_seconds in windows_seconds:
        pairs = build_reference_pairs(cam1_times, window_seconds)
        reference_pairs = len(pairs)

        overall_errors: list[float] = []
        per_component_ref_count: dict[int, int] = defaultdict(int)
        per_component_errors: dict[int, list[float]] = defaultdict(list)

        for name_i, name_j in pairs:
            component_i = name_to_component.get(name_i)
            if component_i is not None:
                per_component_ref_count[component_i] += 1
            component_j = name_to_component.get(name_j)
            if component_i is None or component_j is None or component_i != component_j:
                continue
            component = components[component_i]
            displacement_est = raw_positions[name_j] - raw_positions[name_i]
            aligned = component["sim3_scale"] * (component["rotation_align"] @ displacement_est)
            displacement_gt = reference[name_j] - reference[name_i]
            error = float(np.linalg.norm(aligned - displacement_gt))
            overall_errors.append(error)
            per_component_errors[component_i].append(error)

        per_component_out = []
        for component in components:
            index = component["index"]
            per_component_out.append(
                {
                    "component_index": index,
                    "images_txt": component["images_txt"],
                    **_summarize(per_component_errors.get(index, []), per_component_ref_count.get(index, 0)),
                }
            )

        windows_out.append(
            {
                "window_seconds": window_seconds,
                "tolerance_seconds": 0.1 * window_seconds,
                **_summarize(overall_errors, reference_pairs),
                "components": per_component_out,
            }
        )

    return {
        "schema": "visloc_openloris_rpe_score_v1",
        "scorer_sha256": {
            "score_openloris_rpe.py": om.sha256_file(Path(__file__).resolve()),
            "score_openloris_model.py": om.sha256_file(Path(om.__file__).resolve()),
        },
        "rpe_convention": (
            "translation RPE only, cam1-only rig-frame pairs, per-component Sim(3) "
            "alignment reused from score_openloris_model.score_component/umeyama"
        ),
        "rotation_rpe": None,
        "rotation_rpe_note": (
            "omitted: score_openloris_model.py exposes only camera-center translations "
            "(load_model_centres and interpolate_camera_centres return positions, not "
            "orientation), so no orientation is available to reuse without reimplementing "
            "scorer internals"
        ),
        "max_interpolation_gap_seconds": max_gap_seconds,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": om.sha256_file(manifest_path),
        "ground_truth": str(ground_truth_path.resolve()),
        "ground_truth_sha256": om.sha256_file(ground_truth_path),
        "transform_matrix": str(transform_path.resolve()),
        "transform_matrix_sha256": om.sha256_file(transform_path),
        "image_aliases": str(alias_path.resolve()) if alias_path else None,
        "image_aliases_sha256": om.sha256_file(alias_path) if alias_path else None,
        "model_images_sha256": [component["images_txt_sha256"] for component in components],
        "manifest_images": len(manifest),
        "cam1_manifest_images": sum(1 for _n, (camera, _t) in manifest.items() if camera == 1),
        "cam1_gt_interpolated_images": len(cam1_gt_interp),
        "registered_images": len(name_to_component),
        "models": len(components),
        "components_ate_sanity": [
            {
                "component_index": component["index"],
                "images_txt": component["images_txt"],
                "registered": component["registered"],
                "gt_scored": component["gt_scored"],
                "sim3_scale": component["sim3_scale"],
                "ate_rmse_m": component["ate_rmse_m"],
                "ate_median_m": component["ate_median_m"],
                "ate_p95_m": component["ate_p95_m"],
            }
            for component in components
        ],
        "windows": windows_out,
        "ground_truth_used_only_for_post_mapping_score": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-images", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--transform-matrix", type=Path, required=True)
    parser.add_argument("--image-aliases", type=Path)
    parser.add_argument("--max-gap-seconds", type=float, default=0.1)
    parser.add_argument("--windows-seconds", type=str, default="1,10")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        windows_seconds = parse_windows_seconds(args.windows_seconds)
        result = score(
            [path.resolve() for path in args.model_images],
            args.manifest.resolve(),
            args.ground_truth.resolve(),
            args.transform_matrix.resolve(),
            alias_path=args.image_aliases.resolve() if args.image_aliases else None,
            max_gap_seconds=args.max_gap_seconds,
            windows_seconds=windows_seconds,
        )
        payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (OSError, ValueError, ScoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
