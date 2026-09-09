from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_native_candidates import sha
from run_atlas_backtrack_trial import validate_outputs


class BacktrackTrialTests(unittest.TestCase):
    def test_only_post_ba_geometry_may_change(self):
        for changed in ('model/images.txt', 'model/points3D.txt',
                        'pre-ba/images.txt', 'pre-ba/points3D.txt',
                        'pre-ba/cameras.txt', 'model/cameras.txt'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected = {}
                for phase in ('pre-ba', 'model'):
                    (root / phase).mkdir()
                    for name in ('cameras', 'images', 'points3D'):
                        path = root / phase / (name + '.txt')
                        path.write_text('reference')
                        expected[str(path.relative_to(root))] = sha(path)
                (root / changed).write_text('changed geometry')
                if changed in ('model/images.txt', 'model/points3D.txt'):
                    actual = validate_outputs(root, expected)
                    self.assertNotEqual(actual[changed], expected[changed])
                else:
                    with self.assertRaises(ValueError):
                        validate_outputs(root, expected)

    def test_incomplete_reference_contract_rejected(self):
        with self.assertRaises(ValueError):
            validate_outputs(Path('/unused'), {'model/images.txt': 'bad'})


if __name__ == '__main__':
    unittest.main()
