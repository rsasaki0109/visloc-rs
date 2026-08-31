import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_electro_decision_trace.py"
SPEC = importlib.util.spec_from_file_location("summarize_electro_decision_trace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ElectroDecisionTraceTests(unittest.TestCase):
    def test_parse_trace_captures_mapper_decisions_and_omits_timing(self) -> None:
        text = """\
sfm-debug: PnP attempt #1 on image 7 succeeded (25 corrs -> 19 inliers, ratio=0.760)
sfm-debug: PnP attempt #1 on image 9 failed (12 corrs -> 3 inliers, need >=8)
sfm-debug:   image 9: trials exhausted (1/1, 31 corrs available)
sfm-timing-seed-trial: index=2 pair=(0, 4) reach=8 elapsed=0.125s
sfm-timing-seed-summary: candidates=20 attempted=3 zero_reach=2 successful=1 winner_reach=8 elapsed=0.250s
sfm-timing-ba: registered=8 landmarks=50 observations=120 warm_start=0.000s assemble=0.003s solve=0.100s writeback=0.001s total=0.104s iterations=4 accepted=2
sfm-debug: post-refinement registered image 9 (31 corrs, 20 inliers)
sfm-timing: total=1.000s track_build=0.100s seed_growth=0.250s final_refinement=0.600s geometry_recovery=0.000s structureless=0.000s assembly=0.050s
"""
        trace = MODULE.parse_trace(text, include_timing=False)

        self.assertEqual(
            trace["growth_pnp"],
            [
                {
                    "image": 7,
                    "attempt": 1,
                    "correspondences": 25,
                    "inliers": 19,
                    "accepted": True,
                },
                {
                    "image": 9,
                    "attempt": 1,
                    "correspondences": 12,
                    "inliers": 3,
                    "accepted": False,
                },
            ],
        )
        self.assertEqual(trace["exhausted_images"][0]["available_correspondences"], 31)
        self.assertTrue(trace["post_refinement_pnp"][0]["accepted"])
        self.assertNotIn("elapsed_s", trace["seed_trials"][0])
        self.assertNotIn("timing_s", trace["refinement_rounds"][0])
        self.assertEqual(
            trace["summary"],
            {
                "growth_attempts": 2,
                "growth_accepted": 1,
                "post_attempts": 1,
                "post_accepted": 1,
                "exhausted": 1,
                "refinement_rounds": 1,
            },
        )


    def test_parse_trace_records_failed_post_refinement_pnp(self) -> None:
        trace = MODULE.parse_trace(
            "sfm-debug: post-refinement PnP on image 12 failed "
            "(44 corrs -> none inliers, need >=8)\n"
        )
        self.assertEqual(
            trace["post_refinement_pnp"],
            [
                {
                    "image": 12,
                    "correspondences": 44,
                    "inliers": None,
                    "accepted": False,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
