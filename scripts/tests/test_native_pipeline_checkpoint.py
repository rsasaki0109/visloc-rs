from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_pipeline_checkpoint import snapshot, stage_checkpoint, verify_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_modified_added_and_removed_members_are_rejected(self):
        for mutation in ('modify', 'add', 'remove'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                member = root / 'data'
                member.write_text('original')
                saved = {str(root): snapshot(root)}
                verify_checkpoint(saved)
                if mutation == 'modify':
                    member.write_text('changed')
                elif mutation == 'add':
                    (root / 'extra').write_text('extra')
                else:
                    member.unlink()
                with self.assertRaises(ValueError):
                    verify_checkpoint(saved)

    def test_file_link_target_bytes_and_directory_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target'
            target.write_text('original')
            link = root / 'link'
            link.symlink_to(target)
            saved = {str(link): snapshot(link)}
            target.write_text('changed')
            with self.assertRaises(ValueError):
                verify_checkpoint(saved)
            link.unlink()
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                snapshot(root)

    def test_stage_includes_output_capture_payload_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('output', 'capture', 'payload', 'log'):
                (root / name).write_text(name)
            stage = {'argv': ['tool', '--output', str(root / 'output')],
                     'capture': str(root / 'capture'), 'payload': {'path': str(root / 'payload')}}
            saved = stage_checkpoint(stage, root / 'log')
            self.assertEqual(set(saved), {str(root / name) for name in ('output', 'capture', 'payload', 'log')})
            verify_checkpoint(saved)


if __name__ == '__main__':
    unittest.main()
