"""Struct-of-arrays containers for the two populations.

Instead of one Python object per animal, each population is a handful of numpy
buffers preallocated at its hard cap, with the first ``n`` rows alive. Row order
is meaningful (predation claiming and the reproduction caps are first-come,
first-served), so:

- **births** append at the end (:meth:`_SoABase.append`), and
- **deaths** compact the live prefix *preserving order* (:meth:`_SoABase.compact`).

Because births check the population caps before appending, the buffers can never
overflow — ``append`` asserts it anyway.

The ``pos`` / ``vel`` / ... properties are zero-copy **views** of the live prefix:
writing through them (e.g. ``fish.pos += fish.vel``) mutates the underlying
buffers directly.
"""

from __future__ import annotations

import numpy as np

from brain import N_SENSES
from genome import SHARK_SPEC


class _SoABase:
    """Preallocated struct-of-arrays with a live prefix of ``n`` rows."""

    def __init__(self, capacity: int, fields: dict[str, tuple[tuple[int, ...], type]]):
        self.capacity = capacity
        self.n = 0
        self._fields = {
            name: np.zeros((capacity, *shape), dtype=dtype)
            for name, (shape, dtype) in fields.items()
        }

    def __len__(self) -> int:
        return self.n

    def _view(self, name: str) -> np.ndarray:
        return self._fields[name][: self.n]

    def append(self, **rows: np.ndarray) -> None:
        """Add ``k`` newborn rows (every field required, each shaped ``(k, ...)``).

        ``prev_pos`` — the render-interpolation anchor — may be omitted and
        defaults to ``pos``: a newborn is drawn at its spawn point, not lerped
        in from somewhere else."""
        if "prev_pos" in self._fields:
            rows.setdefault("prev_pos", rows["pos"])
        assert rows.keys() == self._fields.keys(), "append() must supply every field"
        k = len(next(iter(rows.values())))
        assert self.n + k <= self.capacity, "population cap should prevent overflow"
        for name, values in rows.items():
            self._fields[name][self.n : self.n + k] = values
        self.n += k

    def ensure_capacity(self, capacity: int) -> None:
        """Grow the buffers to hold ``capacity`` rows (no-op if they already do).

        Lets population caps be raised while the sim runs: the live prefix is
        copied into fresh, larger buffers. Never shrinks — lowering a cap just
        leaves the extra rows idle."""
        if capacity <= self.capacity:
            return
        for name, buf in self._fields.items():
            grown = np.zeros((capacity, *buf.shape[1:]), dtype=buf.dtype)
            grown[: self.n] = buf[: self.n]
            self._fields[name] = grown
        self.capacity = capacity

    def compact(self, keep: np.ndarray) -> int:
        """Drop live rows where ``keep`` is False, preserving the order of the rest.

        Returns how many were removed. Safe to do in place: the fancy-indexed
        right-hand side materialises before the assignment writes.
        """
        k = int(keep.sum())
        if k < self.n:
            for buf in self._fields.values():
                buf[:k] = buf[: self.n][keep]
        removed = self.n - k
        self.n = k
        return removed


class FishState(_SoABase):
    """The fish population: motion plus a cosmetic per-fish colour — rule-based
    boids carry no genes or breeding bookkeeping."""

    def __init__(self, capacity: int):
        super().__init__(capacity, {
            "pos": ((2,), np.float64),
            "prev_pos": ((2,), np.float64),
            "vel": ((2,), np.float64),
            "color": ((3,), np.uint8),  # cosmetic RGB, assigned at spawn
        })

    @property
    def pos(self) -> np.ndarray:
        return self._view("pos")

    @property
    def prev_pos(self) -> np.ndarray:
        return self._view("prev_pos")

    @property
    def vel(self) -> np.ndarray:
        return self._view("vel")

    @property
    def color(self) -> np.ndarray:
        return self._view("color")


class SharkState(_SoABase):
    """The shark population: like fish, plus the energy reserve that predation
    tops up and metabolism drains, each shark's neural-net weights, and its
    short-term memory (the last ``memory_frames`` sense blocks, newest first —
    the frame stack fed to the brain alongside the current frame)."""

    def __init__(self, capacity: int, weight_count: int, memory_frames: int):
        super().__init__(capacity, {
            "pos": ((2,), np.float64),
            "prev_pos": ((2,), np.float64),
            "vel": ((2,), np.float64),
            "genes": ((SHARK_SPEC.count,), np.float64),
            "brain": ((weight_count,), np.float64),
            "memory": ((memory_frames, N_SENSES), np.float64),
            "generation": ((), np.int64),
            "age": ((), np.int64),
            "energy": ((), np.float64),
        })

    @property
    def pos(self) -> np.ndarray:
        return self._view("pos")

    @property
    def prev_pos(self) -> np.ndarray:
        return self._view("prev_pos")

    @property
    def vel(self) -> np.ndarray:
        return self._view("vel")

    @property
    def genes(self) -> np.ndarray:
        return self._view("genes")

    @property
    def brain(self) -> np.ndarray:
        return self._view("brain")

    @property
    def memory(self) -> np.ndarray:
        return self._view("memory")

    @property
    def generation(self) -> np.ndarray:
        return self._view("generation")

    @property
    def age(self) -> np.ndarray:
        return self._view("age")

    @property
    def energy(self) -> np.ndarray:
        return self._view("energy")
