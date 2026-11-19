"""Persist occupied voxels and pending clearance across independent revisits."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


Voxel = tuple[int, int, int]


@dataclass(frozen=True)
class ClearanceState:
    occupied: frozenset[Voxel]
    pending: frozenset[Voxel] = frozenset()
    visit_ids: tuple[str, ...] = ()
    last_timestamp_ns: int = -1

    def observe(self, visit_id, timestamp_ns, cleared, reoccupied):
        """Delete only after two distinct visits report clearance without occupied evidence."""
        if not visit_id:
            raise ValueError("expected a stable visit identifier")
        if self.visit_ids and (visit_id, timestamp_ns) == (self.visit_ids[-1], self.last_timestamp_ns):
            return self
        if visit_id in self.visit_ids or timestamp_ns <= self.last_timestamp_ns:
            raise ValueError("clearance visits must be independent and chronological")
        cleared = (frozenset(cleared) & self.occupied) - reoccupied
        pending = self.pending - reoccupied
        removed = pending & cleared
        return ClearanceState(
            occupied=self.occupied - removed,
            pending=(pending | cleared) - removed,
            visit_ids=(*self.visit_ids, visit_id),
            last_timestamp_ns=timestamp_ns,
        )


def save_clearance(path: Path, state: ClearanceState, voxel_m: float):
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary, version=2, voxel_m=voxel_m,
            occupied=np.asarray(sorted(state.occupied), dtype=np.int32).reshape(-1, 3),
            pending=np.asarray(sorted(state.pending), dtype=np.int32).reshape(-1, 3),
            visit_ids=np.asarray(state.visit_ids, dtype=str), last_timestamp_ns=state.last_timestamp_ns,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_clearance(path: Path, voxel_m: float):
    with np.load(path, allow_pickle=False) as data:
        if data["version"].item() != 2 or data["voxel_m"].item() != voxel_m:
            raise ValueError("unexpected clearance snapshot schema")

        def keys(name):
            rows = data[name]
            if rows.ndim != 2 or rows.shape[1] != 3 or rows.dtype.kind not in "iu":
                raise ValueError(f"invalid clearance field {name}")
            return frozenset(map(tuple, rows.tolist()))

        occupied, pending = keys("occupied"), keys("pending")
        visits, timestamp = tuple(data["visit_ids"].tolist()), int(data["last_timestamp_ns"].item())
        if not pending <= occupied or len(set(visits)) != len(visits):
            raise ValueError("inconsistent pending evidence")
        return ClearanceState(occupied, pending, visits, timestamp)
