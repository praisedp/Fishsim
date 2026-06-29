"""Heritable traits for the two evolving species: fish (``Genome``) and sharks
(``SharkGenome``).

Each animal carries its own genome — the numbers below fully determine how it
behaves. Reproduction copies a genome with small random tweaks (:meth:`mutate`);
predation and starvation cull the ones that behave badly, so the gene values drift
toward whatever survives. Fish evolve to evade; sharks evolve to hunt.

The two genomes share all their generic GA machinery (``random`` / ``mutate`` /
``crossover`` / ``gene_fraction``) via ``GenomeBase`` — each subclass just declares
its own ``RANGES`` dict, the single source of truth for that species' gene bounds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


class GenomeBase:
    """Generic genetic-algorithm behaviour shared by every species' genome.

    Subclasses are dataclasses that (a) set the class attribute ``RANGES`` to an
    ordered ``{gene_name: (min, max)}`` dict and (b) carry one float field per gene
    plus an ``int`` ``generation`` field. Everything below is written against
    ``RANGES`` so it works for any gene set.
    """

    RANGES: ClassVar[dict[str, tuple[float, float]]] = {}
    generation: int

    @classmethod
    def random(cls, rng: random.Random):
        """A fresh genome with each gene drawn uniformly from its range."""
        return cls(**{
            name: rng.uniform(lo, hi) for name, (lo, hi) in cls.RANGES.items()
        })

    def mutate(self, rng: random.Random, rate: float, sigma: float):
        """A child copy: each gene has a ``rate`` chance of a Gaussian nudge.

        ``sigma`` is a *fraction of each gene's span*, so the same setting tweaks a
        wide-range gene proportionally harder than a narrow one. Results are clamped
        back into range and ``generation`` advances by one.
        """
        child: dict[str, float] = {}
        for name, (lo, hi) in self.RANGES.items():
            value = getattr(self, name)
            if rng.random() < rate:
                value = _clamp(value + rng.gauss(0.0, sigma * (hi - lo)), lo, hi)
            child[name] = value
        return type(self)(generation=self.generation + 1, **child)

    def crossover(self, other, rng: random.Random):
        """Sexual reproduction: each gene picked from one parent at random.

        Not used by the default (asexual) reproduction path, but kept ready so the
        sim can switch to two-parent breeding without touching this module.
        """
        child = {name: getattr(rng.choice((self, other)), name) for name in self.RANGES}
        gen = max(self.generation, other.generation) + 1
        return type(self)(generation=gen, **child)

    def gene_fraction(self, name: str) -> float:
        """Where this gene sits in its range, as 0.0–1.0 (for colour mapping)."""
        lo, hi = self.RANGES[name]
        if hi == lo:
            return 0.0
        return _clamp((getattr(self, name) - lo) / (hi - lo), 0.0, 1.0)


# --- Fish -------------------------------------------------------------------

# Inclusive (min, max) bounds per gene. Keys MUST match the field names below.
GENE_RANGES: dict[str, tuple[float, float]] = {
    "w_sep": (0.0, 3.0),         # separation weight
    "w_ali": (0.0, 3.0),         # alignment weight
    "w_coh": (0.0, 3.0),         # cohesion weight
    "w_flee": (0.0, 6.0),        # flee-from-shark weight
    "perception": (20.0, 80.0),  # how far it sees neighbours
    "max_speed": (2.0, 5.0),     # top speed
    "fear": (0.5, 2.0),          # multiplies the base flee radius
}

# Order genes appear in the UI / averages. Same as dict order (Python 3.7+).
GENE_NAMES: tuple[str, ...] = tuple(GENE_RANGES.keys())


@dataclass
class Genome(GenomeBase):
    """Seven heritable fish traits plus a lineage depth (``generation``)."""

    RANGES: ClassVar[dict[str, tuple[float, float]]] = GENE_RANGES

    w_sep: float
    w_ali: float
    w_coh: float
    w_flee: float
    perception: float
    max_speed: float
    fear: float
    generation: int = 0


# --- Shark ------------------------------------------------------------------

# What makes a good hunter. Each gene has a real trade-off so evolution must
# balance rather than max everything: speed and sight both cost energy per tick
# (see the metabolism formula in sim.py), and lead just converges to the best
# interception aim.
SHARK_GENE_RANGES: dict[str, tuple[float, float]] = {
    "max_speed": (3.0, 6.0),      # chase speed; only just faster than fish (top out at 5)
    "perception": (80.0, 400.0),  # how far it can spot prey
    "lead": (0.0, 2.0),           # predictive aim-ahead (intercept) factor
}

SHARK_GENE_NAMES: tuple[str, ...] = tuple(SHARK_GENE_RANGES.keys())


@dataclass
class SharkGenome(GenomeBase):
    """Three heritable hunting traits plus a lineage depth (``generation``)."""

    RANGES: ClassVar[dict[str, tuple[float, float]]] = SHARK_GENE_RANGES

    max_speed: float
    perception: float
    lead: float
    generation: int = 0
