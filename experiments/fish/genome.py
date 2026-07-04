"""Heritable traits for the one evolving species: sharks.

(Fish are plain rule-based boids — their shared behaviour knobs live in
``config.py``, and nothing about them evolves.)

Genes live in numpy arrays, one **row per animal, one column per gene** — the
struct-of-arrays layout the whole engine uses. A species is described by a
:class:`GeneSpec` (its gene names and per-gene bounds); the operations below
(``random_genes`` / ``mutate_genes`` / ``gene_fractions``) work on whole
populations at once.

Reproduction copies gene rows with small random tweaks (:func:`mutate_genes`);
predation and starvation cull the ones that behave badly, so the gene values
drift toward whatever survives. Two-parent breeding exists as
:func:`crossover_genes` (uniform crossover); the sim uses it when reseeding
sharks from the hall-of-fame.

``SHARK_GENE_RANGES`` remains the single source of truth for the gene bounds.
"""

from __future__ import annotations

import numpy as np

# --- Gene declarations (single source of truth) -------------------------------

# The shark's *body*; its behaviour lives in an evolved neural net (brain.py)
# whose throttle output picks a speed between the two speed genes each tick.
# The bounds here are the hard caps evolution works within. Real trade-offs:
# actual speed and sight both cost energy per tick (see the metabolism formula
# in sim.py), so a wide speed band is only worth its price if the brain uses it.
SHARK_GENE_RANGES: dict[str, tuple[float, float]] = {
    "max_speed": (3.0, 6.0),      # top speed; only just faster than fish (top out at 5)
    "min_speed": (0.5, 3.0),      # cruise floor — the cheapest pace it can hold
    "perception": (80.0, 400.0),  # how far it can spot prey
}

# Order genes appear in the UI / averages. Same as dict order (Python 3.7+).
SHARK_GENE_NAMES: tuple[str, ...] = tuple(SHARK_GENE_RANGES.keys())


class GeneSpec:
    """One species' gene set: names plus per-gene bounds as aligned arrays.

    The column order of every ``(N, G)`` genes array in the sim is this spec's
    ``names`` order. ``lo`` / ``hi`` / ``span`` are ``(G,)`` arrays so the GA
    operations below can broadcast over whole populations.
    """

    __slots__ = ("names", "lo", "hi", "span", "_cols")

    def __init__(self, ranges: dict[str, tuple[float, float]]):
        self.names = tuple(ranges.keys())
        self.lo = np.array([lo for lo, _hi in ranges.values()])
        self.hi = np.array([hi for _lo, hi in ranges.values()])
        self.span = self.hi - self.lo
        self._cols = {name: c for c, name in enumerate(self.names)}

    @property
    def count(self) -> int:
        return len(self.names)

    def col(self, name: str) -> int:
        """Column index of ``name`` in this species' genes array."""
        return self._cols[name]

    def mid(self) -> np.ndarray:
        """Every gene at its mid-range value, shape ``(G,)`` (handy in tests)."""
        return (self.lo + self.hi) / 2


SHARK_SPEC = GeneSpec(SHARK_GENE_RANGES)

# Column indices, precomputed once so hot code reads like prose instead of
# doing dict lookups: genes[:, S_MAXSPD] is a whole population's top speeds.
S_MAXSPD = SHARK_SPEC.col("max_speed")
S_MINSPD = SHARK_SPEC.col("min_speed")
S_PERC = SHARK_SPEC.col("perception")


# --- Population-wide GA operations --------------------------------------------


def random_genes(rng: np.random.Generator, spec: GeneSpec, n: int) -> np.ndarray:
    """``(n, G)`` fresh genomes, each gene drawn uniformly from its range."""
    return rng.uniform(spec.lo, spec.hi, (n, spec.count))


def mutate_genes(
    rng: np.random.Generator,
    spec: GeneSpec,
    parents: np.ndarray,
    rate: float,
    sigma: float,
) -> np.ndarray:
    """Child copies of ``parents`` ``(k, G)``: each gene has a ``rate`` chance of a
    Gaussian nudge.

    ``sigma`` is a *fraction of each gene's span*, so the same setting tweaks a
    wide-range gene proportionally harder than a narrow one. Results are clamped
    back into range. (Lineage depth is tracked outside the genes array — callers
    set the children's ``generation`` to the parents' plus one.)
    """
    mutated = rng.random(parents.shape) < rate
    noise = rng.normal(0.0, sigma * spec.span, parents.shape)
    return np.clip(parents + mutated * noise, spec.lo, spec.hi)


def crossover_genes(
    rng: np.random.Generator, mothers: np.ndarray, fathers: np.ndarray
) -> np.ndarray:
    """Uniform crossover of two aligned parent stacks ``(k, G)``: each child
    gene comes from its mother or its father with equal odds. Both parents'
    genes are already in range, so no clamping is needed; callers usually
    follow with :func:`mutate_genes` for fresh variation."""
    return np.where(rng.random(mothers.shape) < 0.5, mothers, fathers)


def gene_fractions(genes: np.ndarray, spec: GeneSpec, name: str) -> np.ndarray:
    """Where each animal's ``name`` gene sits in its range, as 0.0–1.0 ``(n,)``
    (for colour mapping)."""
    c = spec.col(name)
    if spec.span[c] == 0:
        return np.zeros(len(genes))
    return np.clip((genes[:, c] - spec.lo[c]) / spec.span[c], 0.0, 1.0)
