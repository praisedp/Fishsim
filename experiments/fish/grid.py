"""Spatial hash grid for fast neighbour lookups.

Naively, every fish checking every other fish is O(n²). We bucket fish into
square cells of side ``cell_size`` (>= perception radius), so a fish's neighbours
can only live in its own cell plus the 8 adjacent ones. Lookup becomes ~O(n).

The grid is cheap to rebuild from scratch every frame, so we do exactly that —
no incremental bookkeeping to get wrong.
"""

from __future__ import annotations

from boid import Boid


class SpatialGrid:
    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], list[Boid]] = {}

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def build(self, fish: list[Boid]) -> None:
        self.cells = {}
        for f in fish:
            self.cells.setdefault(self._key(f.pos.x, f.pos.y), []).append(f)

    def neighbours(self, boid: Boid, radius: float) -> list[Boid]:
        """All fish (excluding ``boid``) within ``radius`` of ``boid``."""
        col, row = self._key(boid.pos.x, boid.pos.y)
        radius_sq = radius * radius
        found: list[Boid] = []
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                bucket = self.cells.get((col + dc, row + dr))
                if not bucket:
                    continue
                for other in bucket:
                    if other is boid:
                        continue
                    if boid.pos.distance_squared_to(other.pos) <= radius_sq:
                        found.append(other)
        return found
