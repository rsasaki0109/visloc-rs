from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_kitti_adaptive_depth_gate_smoke import (  # noqa: E402
    load_failure_runs,
    load_latest_runs,
    missing_expected_runs,
    render,
)


def manifest(
    *,
    run_id: str,
    created_utc: str,
    depth_gate: str,
    max_frames: int = 2,
    status: str = "success",
    failure_reason: str | None = None,
) -> dict:
    metrics = [
        {"name": "frames", "value": max_frames},
        {"name": "candidates_mean", "value": 219},
        {"name": "accepted_mean", "value": 206.5},
        {"name": "effective_min_depth_m_mean", "value": 3},
        {"name": "vo_ate_rmse_m", "value": 16.4839},
    ]
    if depth_gate == "adaptive" and status == "success":
        metrics.append({"name": "depth_quantile_m_mean", "value": 14.380565})
    if status != "success":
        metrics = [
            {"name": "frames_requested", "value": max_frames},
            {"name": "frame_pairs_completed", "value": 1},
            {"name": "failed_pair_index", "value": 1},
            {"name": "kabsch_correspondence_count", "value": 81},
            {"name": "kabsch_min_inliers", "value": 8},
        ]
    return {
        "run_id": run_id,
        "created_utc": created_utc,
        "status": status,
        "failure_reason": failure_reason,
        "benchmark": {"id": "kitti-adaptive-depth-gate-smoke"},
        "dataset": {
            "sequence": "00",
            "checksum": "dataset-sha",
            "checksum_method": "sha256_tree_v1",
        },
        "config": {
            "params": {
                "depth_gate": depth_gate,
                "max_frames": max_frames,
                "frontend": "deep",
            }
        },
        "metrics": metrics,
        "artifacts": [
            {
                "kind": "depth_gate_diagnostics",
                "path": "diagnostics.csv",
                "exists": status == "success",
            }
        ],
    }


class KittiAdaptiveDepthGateSmokeSummaryTests(unittest.TestCase):
    def test_success_ab_and_failure_rows_render_from_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "adaptive-old.json").write_text(
                json.dumps(
                    manifest(
                        run_id="adaptive-old",
                        created_utc="2026-06-20T00:00:00Z",
                        depth_gate="adaptive",
                    )
                ),
                encoding="utf-8",
            )
            (registry / "adaptive-new.json").write_text(
                json.dumps(
                    manifest(
                        run_id="adaptive-new",
                        created_utc="2026-06-20T01:00:00Z",
                        depth_gate="adaptive",
                    )
                ),
                encoding="utf-8",
            )
            (registry / "fixed.json").write_text(
                json.dumps(
                    manifest(
                        run_id="fixed3",
                        created_utc="2026-06-20T02:00:00Z",
                        depth_gate="fixed",
                    )
                ),
                encoding="utf-8",
            )
            (registry / "adaptive-failure.json").write_text(
                json.dumps(
                    manifest(
                        run_id="adaptive-failure",
                        created_utc="2026-06-20T03:00:00Z",
                        depth_gate="adaptive",
                        max_frames=6,
                        status="failure",
                        failure_reason=(
                            "KabschFailed { pair_index: 1, "
                            "correspondence_count: 81, min_inliers: 8 }"
                        ),
                    )
                ),
                encoding="utf-8",
            )
            (registry / "wrong-sequence.json").write_text(
                json.dumps(
                    {
                        **manifest(
                            run_id="wrong-sequence",
                            created_utc="2026-06-20T04:00:00Z",
                            depth_gate="adaptive",
                        ),
                        "dataset": {"sequence": "02"},
                    }
                ),
                encoding="utf-8",
            )

            args = Namespace(
                registry_dir=registry,
                sequence="00",
                max_frames=2,
                variant=["adaptive", "fixed"],
            )
            runs = load_latest_runs(args)
            failures = load_failure_runs(args)

            self.assertEqual(runs["adaptive"]["run_id"], "adaptive-new")
            self.assertEqual(runs["fixed"]["run_id"], "fixed3")
            self.assertEqual([row["run_id"] for row in failures], ["adaptive-failure"])
            self.assertEqual(missing_expected_runs(args, runs), [])

            table = render(args, runs, failures)
            self.assertIn("Dataset checksum: `sha256_tree_v1 dataset-sha`.", table)
            self.assertIn("| adaptive | 2 | 3 | 219 | 206.5 | 14.381 | 16.4839 | yes | adaptive-new |", table)
            self.assertIn("| fixed | 2 | 3 | 219 | 206.5 |  | 16.4839 | yes | fixed3 |", table)
            self.assertIn("## Recorded Failures", table)
            self.assertIn("| adaptive | 6 | failure | KabschFailed", table)
            self.assertIn("adaptive-failure", table)
            self.assertNotIn("wrong-sequence", table)

    def test_missing_expected_runs_reports_absent_fixed_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "adaptive.json").write_text(
                json.dumps(
                    manifest(
                        run_id="adaptive",
                        created_utc="2026-06-20T00:00:00Z",
                        depth_gate="adaptive",
                    )
                ),
                encoding="utf-8",
            )
            args = Namespace(
                registry_dir=registry,
                sequence="00",
                max_frames=2,
                variant=["adaptive", "fixed"],
            )

            self.assertEqual(missing_expected_runs(args, load_latest_runs(args)), ["fixed"])


if __name__ == "__main__":
    unittest.main()
