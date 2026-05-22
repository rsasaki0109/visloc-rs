from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kitti_revisit_smoke_lib import (  # noqa: E402
    README_HEADLINE_EXPECTATIONS,
    RevisitExpectations,
    row,
    strongest_candidate,
    validate_expectations,
)


def headline_rows() -> list[dict[str, str]]:
    rows = [
        row(
            score=float(index),
            matched_keyframe_id=index,
            query_frame_id=4500 + index,
            inliers=12,
            inlier_ratio=0.4,
        )
        for index in range(40)
    ]
    rows.append(
        row(
            score=16083.0719,
            matched_keyframe_id=49,
            query_frame_id=4501,
            inliers=57,
            inlier_ratio=0.6,
        )
    )
    return rows


class KittiRevisitSmokeLibTest(unittest.TestCase):
    def test_strongest_candidate_uses_score(self) -> None:
        strongest = strongest_candidate(
            [
                row(score=1.0, matched_keyframe_id=1, query_frame_id=4501, inliers=20, inlier_ratio=0.5),
                row(score=3.5, matched_keyframe_id=2, query_frame_id=4502, inliers=18, inlier_ratio=0.7),
                row(score=2.0, matched_keyframe_id=3, query_frame_id=4503, inliers=22, inlier_ratio=0.6),
            ]
        )

        self.assertEqual(strongest["matched_keyframe_id"], "2")

    def test_readme_headline_expectations_pass(self) -> None:
        validate_expectations(headline_rows(), README_HEADLINE_EXPECTATIONS)

    def test_expectation_failures_are_collected(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected at least 41 candidates") as raised:
            validate_expectations(
                [
                    row(
                        score=100.0,
                        matched_keyframe_id=48,
                        query_frame_id=4502,
                        inliers=56,
                        inlier_ratio=0.599,
                    )
                ],
                RevisitExpectations(
                    min_candidates=41,
                    strongest_from=49,
                    strongest_to=4501,
                    min_strongest_inliers=57,
                    min_strongest_ratio=0.6,
                ),
            )

        message = str(raised.exception)
        self.assertIn("expected strongest_from=49, got 48", message)
        self.assertIn("expected strongest_to=4501, got 4502", message)
        self.assertIn("expected strongest inliers >= 57, got 56", message)
        self.assertIn("expected strongest ratio >= 0.6, got 0.599000", message)

    def test_empty_candidates_fail_before_expectation_checks(self) -> None:
        with self.assertRaisesRegex(ValueError, "no accepted candidates"):
            validate_expectations([], RevisitExpectations(min_candidates=1))


if __name__ == "__main__":
    unittest.main()
