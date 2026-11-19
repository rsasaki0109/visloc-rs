import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persistent_clearance import ClearanceState, load_clearance, save_clearance


class PersistentClearanceTest(unittest.TestCase):
    def test_reload_and_retry_do_not_count_as_an_independent_visit(self):
        voxel = (1, 2, 3)
        initial = ClearanceState(frozenset({voxel}))
        pending = initial.observe("A", 1, {voxel}, set())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            save_clearance(path, pending, 0.07)
            pending = load_clearance(path, 0.07)
        self.assertEqual(pending.observe("A", 1, {voxel}, set()), pending)
        with self.assertRaises(ValueError):
            pending.observe("A", 2, {voxel}, set())
        with self.assertRaises(ValueError):
            pending.observe("B", 0, {voxel}, set())
        confirmed = pending.observe("B", 2, {voxel}, set())
        self.assertEqual(confirmed.occupied, frozenset())
        self.assertEqual(confirmed.pending, frozenset())

    def test_unknown_keeps_pending_and_occupied_resets_it(self):
        first, second = (1, 2, 3), (1, 2, 4)
        initial = ClearanceState(frozenset({first, second}))
        pending = initial.observe("A", 1, {first, second}, set())
        reset = pending.observe("B", 2, set(), {second})
        self.assertEqual(reset.pending, {first})
        confirmed = reset.observe("C", 3, {first, second}, set())
        self.assertEqual(confirmed.occupied, {second})
        self.assertEqual(confirmed.pending, {second})


if __name__ == "__main__":
    unittest.main()
