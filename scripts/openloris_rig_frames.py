#!/usr/bin/env python3
"""Deterministically group OpenLORIS stereo exposures into rig frames."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Mapping


class RigFrameGroupingError(ValueError):
    pass


def canonicalize_rig_timestamps(
    rows: Iterable[Mapping[str, object]], *, tolerance_seconds: float = 0.001
) -> tuple[dict[str, str], int]:
    """Return image -> frame timestamp and the number of repaired stereo pairs.

    Exact timestamp groups are preferred. Two singleton groups may be joined
    only when they contain complementary cameras, are adjacent in timestamp
    order, and differ by at most ``tolerance_seconds``. This covers the known
    sub-millisecond OpenLORIS sensor timestamp skew without guessing across a
    missing exposure or an ambiguous cluster.
    """
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0.0:
        raise RigFrameGroupingError("rig timestamp tolerance must be finite and non-negative")
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        try:
            name = str(row["name"])
            camera = int(row["camera"])
            timestamp = str(row["timestamp"])
            numeric_timestamp = float(timestamp)
        except (KeyError, TypeError, ValueError) as exc:
            raise RigFrameGroupingError("malformed rig timestamp row") from exc
        if (
            not name
            or name in seen
            or camera not in (1, 2)
            or not math.isfinite(numeric_timestamp)
        ):
            raise RigFrameGroupingError(f"invalid rig timestamp row for {name!r}")
        seen.add(name)
        groups[timestamp].append((name, camera))

    canonical = {
        name: timestamp
        for timestamp, members in groups.items()
        if {camera for _, camera in members} == {1, 2} and len(members) == 2
        for name, _ in members
    }
    incomplete = []
    for timestamp, members in groups.items():
        if all(name in canonical for name, _ in members):
            continue
        if len(members) != 1:
            raise RigFrameGroupingError(f"invalid rig frame at timestamp {timestamp}")
        incomplete.append((float(timestamp), timestamp, members[0]))
    incomplete.sort()

    repaired = 0
    index = 0
    while index < len(incomplete):
        if index + 1 >= len(incomplete):
            raise RigFrameGroupingError(
                f"incomplete rig frame at timestamp {incomplete[index][1]}"
            )
        left, right = incomplete[index], incomplete[index + 1]
        delta = right[0] - left[0]
        if left[2][1] == right[2][1] or delta > tolerance_seconds:
            raise RigFrameGroupingError(f"incomplete rig frame at timestamp {left[1]}")
        # Reject a three-way cluster instead of selecting an arbitrary pairing.
        if index + 2 < len(incomplete) and incomplete[index + 2][0] - right[0] <= tolerance_seconds:
            raise RigFrameGroupingError(f"ambiguous rig timestamps near {left[1]}")
        canonical[left[2][0]] = left[1]
        canonical[right[2][0]] = left[1]
        repaired += 1
        index += 2

    if len(canonical) != len(seen):
        raise RigFrameGroupingError("not every rig image was assigned to a complete frame")
    return canonical, repaired
