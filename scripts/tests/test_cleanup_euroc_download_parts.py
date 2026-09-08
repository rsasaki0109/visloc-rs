import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'cleanup_euroc_download_parts.py'


class CleanupPartsTest(unittest.TestCase):
    def run_case(self, *, corrupt=False, remove=False, archive_corrupt=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'ranges').mkdir()
            content = b'0123456789'
            (root / 'machine_hall.zip').write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            (root / 'machine_hall.zip.sha256').write_text(
                ('0' * 64 if archive_corrupt else digest) + '  machine_hall.zip\n')
            part = root / 'ranges' / '2-5.part'
            part.write_bytes(b'xxxx' if corrupt else content[2:6])
            header = root / 'ranges' / '2-5.part.headers'
            header.write_text('retained')
            command = [sys.executable, str(SCRIPT), '--root', str(root)]
            if remove:
                command.append('--remove-verified-parts')
            result = subprocess.run(command, capture_output=True, timeout=10)
            self.assertEqual(result.returncode == 0, not (corrupt or archive_corrupt))
            self.assertEqual(part.exists(), not (remove and not corrupt and not archive_corrupt))
            self.assertTrue(header.exists())
            self.assertEqual((root / 'machine_hall.zip').read_bytes(), content)

    def test_audit_retains(self):
        self.run_case()

    def test_remove_verified(self):
        self.run_case(remove=True)

    def test_bad_part_retained(self):
        self.run_case(corrupt=True, remove=True)

    def test_bad_archive_retained(self):
        self.run_case(archive_corrupt=True, remove=True)
