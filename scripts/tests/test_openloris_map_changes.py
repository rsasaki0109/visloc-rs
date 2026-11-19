import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openloris_map_changes import classify, visibility
from rgbd_map_cloud import keyframe_poses


class MapChangesTest(unittest.TestCase):
    def test_visibility_evidence_handles_occlusion_and_missing_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points = np.array([[-0.6, 0, 2], [-0.2, 0, 2], [0.2, 0, 2], [0.6, 0, 2]])
            intrinsics = (10, 10, 8, 8)
            depth = np.full((17, 17), 3000, dtype=np.uint16)
            depth[7:10, 4:7] = 2000
            depth[8, 9] = 1000
            depth[8, 11] = 0
            path = root / "depth.png"
            Image.fromarray(depth).save(path)
            hits, free = visibility(points, [(np.eye(4), path)], intrinsics, 0.1)
            np.testing.assert_array_equal(hits, [1, 0, 0, 0])
            np.testing.assert_array_equal(free, [0, 0, 0, 0])
            depth[7:10, 6:9] = 3000
            Image.fromarray(depth).save(path)
            _, free = visibility(points, [(np.eye(4), path)], intrinsics, 0.1)
            self.assertEqual(free[1], 1)
            masks = root / "masks"
            masks.mkdir()
            mask = np.zeros_like(depth, dtype=np.uint8)
            mask[8, 7] = 255
            Image.fromarray(mask).save(masks / path.name)
            _, free = visibility(points, [(np.eye(4), path)], intrinsics, 0.1, masks)
            self.assertEqual(free[1], 0)

    def test_classification_requires_cross_session_support_and_keeps_unknown(self):
        hits = np.array([[4, 4, 3, 5], [4, 0, 0, 0]])
        free = np.array([[0, 0, 3, 0], [0, 4, 0, 0]])
        np.testing.assert_array_equal(classify(hits, free, 2), [1, 2, 3, 0])

    def test_session_pose_reader_keeps_empty_colmap_observation_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sparse").mkdir()
            (root / "images.csv").write_text("image_id,frame_id,cam_id,timestamp_ns,session\n1,1,0,1000000000,0\n2,2,0,2000000000,1\n")
            (root / "sparse" / "images.txt").write_text(
                "# poses\n1 1 0 0 0 0 0 0 1 first\n\n2 1 0 0 0 -1 0 0 1 second\n0 0 1\n"
            )
            [(timestamp, pose)] = keyframe_poses(root, session=1)
            self.assertEqual(timestamp, 2)
            np.testing.assert_allclose(pose[:3, 3], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
