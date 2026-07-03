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

from brain import WEIGHT_COUNT
from genome import FISH_SPEC, SHARK_SPEC


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
        """Add ``k`` newborn rows (every field required, each shaped ``(k, ...)``)."""
        assert rows.keys() == self._fields.keys(), "append() must supply every field"
        k = len(next(iter(rows.values())))
        assert self.n + k <= self.capacity, "population cap should prevent overflow"
        for name, values in rows.items():
            self._fields[name][self.n : self.n + k] = values
        self.n += k

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
    """The fish population: motion, genes, and breeding bookkeeping."""

    def __init__(self, capacity: int):
        super().__init__(capacity, {
            "pos": ((2,), np.float64),
            "vel": ((2,), np.float64),
            "genes": ((FISH_SPEC.count,), np.float64),
            "generation": ((), np.int64),
            "age": ((), np.int64),
        })

    @property
    def pos(self) -> np.ndarray:
        return self._view("pos")

    @property
    def vel(self) -> np.ndarray:
        return self._view("vel")

    @property
    def genes(self) -> np.ndarray:
        return self._view("genes")

    @property
    def generation(self) -> np.ndarray:
        return self._view("generation")

    @property
    def age(self) -> np.ndarray:
        return self._view("age")


class SharkState(_SoABase):
    """The shark population: like fish, plus the energy reserve that predation
    tops up and metabolism drains, and each shark's neural-net weights."""

    def __init__(self, capacity: int):
        super().__init__(capacity, {
            "pos": ((2,), np.float64),
            "vel": ((2,), np.float64),
            "genes": ((SHARK_SPEC.count,), np.float64),
            "brain": ((WEIGHT_COUNT,), np.float64),
            "generation": ((), np.int64),
            "age": ((), np.int64),
            "energy": ((), np.float64),
        })

    @property
    def pos(self) -> np.ndarray:
        return self._view("pos")

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
    def generation(self) -> np.ndarray:
        return self._view("generation")

    @property
    def age(self) -> np.ndarray:
        return self._view("age")

    @property
    def energy(self) -> np.ndarray:
        return self._view("energy")
