"""Evolvable shark brains: one MLP per shark, weights bred by the GA.

Each shark carries a flat weight vector (a row of the population's ``(S,
weight_count)`` array, stored beside its genes in :mod:`state`). Every tick the
sim builds an egocentric observation per shark — a sector retina of the fish it
can see plus a few body senses, for the current frame *and* the last few
(frame stacking, the net's only memory) — and :func:`forward` maps all of them
to a ``(turn, throttle)`` pair in one batched pass. There is no learning rule:
weights change only at birth (:func:`mutate_brains`), and hunting quality is
selected the same way as any other gene — sharks that steer badly starve.

Like :mod:`steering`, this module is pure array math (no config, no rng state,
no sim imports) so every piece can be oracle-tested headlessly.

The architecture is *derived from config* at sim construction: a
:class:`BrainSpec` turns ``(shark_memory_frames, shark_hidden)`` into input
counts and the flat-weight reshape layout, and sizes the ``SharkState`` weight
buffer. Only the retina resolution stays a compile-time constant.
"""

from __future__ import annotations

import numpy as np

# --- Architecture ---------------------------------------------------------------

N_SECTORS = 8  # egocentric retina resolution around the shark's heading
# One frame's sense block: retina (closeness + count per sector) then body
# senses — energy, speed, obstacle proximity. The constant bias input is NOT
# part of the block (it would be redundant per frame); it's a single extra
# slot at the end of the observation. See sim._shark_observe for the layout.
N_SENSES = 2 * N_SECTORS + 3
N_OUTPUTS = 2  # turn in [-1, 1], throttle in [0, 1]


class BrainSpec:
    """Shapes of the shark MLP, derived from config.

    ``memory_frames`` past sense blocks are stacked after the current one, and
    ``hidden`` sizes the tanh layers. Flat weight layout: each layer's matrix
    in order. Every layer after the first gets a bias via a constant 1.0
    appended to its incoming activations (the input layer's bias is the
    constant observation slot), hence the ``+1`` row. One spec is shared by
    every brain array in a sim — it sizes the ``SharkState`` weight buffer and
    the reshape layout in :func:`forward`.
    """

    __slots__ = ("memory_frames", "hidden", "n_inputs", "weight_count", "layers")

    def __init__(self, memory_frames: int, hidden: tuple[int, ...]):
        assert memory_frames >= 0 and len(hidden) >= 1
        self.memory_frames = memory_frames
        self.hidden = tuple(hidden)
        self.n_inputs = N_SENSES * (1 + memory_frames) + 1  # +1 = bias slot
        sizes = (self.n_inputs, *self.hidden, N_OUTPUTS)
        self.layers: list[tuple[int, int, int]] = []  # (offset, rows, cols)
        offset = 0
        for i in range(len(sizes) - 1):
            rows = sizes[i] + (1 if i > 0 else 0)
            self.layers.append((offset, rows, sizes[i + 1]))
            offset += rows * sizes[i + 1]
        self.weight_count = offset


# --- Breeding weights (the GA operations, mirroring genome.py) ------------------


def random_brains(
    rng: np.random.Generator, n: int, init_scale: float, weight_count: int
) -> np.ndarray:
    """``(n, weight_count)`` founder brains, weights uniform in ``±init_scale``."""
    return rng.uniform(-init_scale, init_scale, (n, weight_count))


def mutate_brains(
    rng: np.random.Generator,
    parents: np.ndarray,
    rate: float,
    sigma: float,
    cap: float,
) -> np.ndarray:
    """Child copies of ``parents`` ``(k, weight_count)``: each weight has a
    ``rate`` chance of a Gaussian nudge, clipped to ``±cap``.

    Unlike :func:`genome.mutate_genes`, ``sigma`` is in absolute weight units —
    weights share one uniform scale, so a span-relative sigma would just couple
    the nudge size to the clip cap for no reason.
    """
    mutated = rng.random(parents.shape) < rate
    noise = rng.normal(0.0, sigma, parents.shape)
    return np.clip(parents + mutated * noise, -cap, cap)


def crossover_brains(
    rng: np.random.Generator, mothers: np.ndarray, fathers: np.ndarray
) -> np.ndarray:
    """Uniform per-weight crossover of two aligned parent stacks
    ``(k, weight_count)``: each child weight comes from either parent with
    equal odds.

    Mixing weights of two independently evolved nets can break co-adapted
    structure (the classic competing-conventions problem), so this is used
    sparingly — only for hall-of-fame rescues, where the elites usually share
    recent ancestry and the follow-up mutation restores variation anyway.
    """
    return np.where(rng.random(mothers.shape) < 0.5, mothers, fathers)


# --- Perception and thinking, batched over the whole population ----------------


def sector_retina(
    delta: np.ndarray,
    d2: np.ndarray,
    headings: np.ndarray,
    perception: np.ndarray,
    count_norm: float,
) -> np.ndarray:
    """The retina block of every shark's observation, ``(S, 2 * N_SECTORS)``.

    The full circle around each shark's heading is cut into ``N_SECTORS`` equal
    wedges. For every fish within that shark's ``perception`` gene, its wedge
    records how close the *nearest* such fish is (``1 - dist/perception``, so 0
    = barely visible, ~1 = touching) and how *many* sit there (saturating at
    ``count_norm``). Sharks that see nothing return all zeros.

    ``delta`` ``(S, F, 2)`` points shark→fish (already minimum-imaged when the
    world wraps) and ``d2`` ``(S, F)`` is its squared length, both exactly as the
    sim's predation pass builds them.
    """
    s = len(headings)
    retina = np.zeros((s, 2 * N_SECTORS))
    visible = d2 <= perception[:, None] ** 2
    if not visible.any():
        return retina

    # Bearing of each fish relative to the shark's facing, folded to [0, 2π)
    # and bucketed. The scale puts dead-ahead on the seam between the two
    # front sectors; the minimum() guards the exact-2π float edge.
    facing = np.arctan2(headings[:, 1], headings[:, 0])
    bearing = np.arctan2(delta[:, :, 1], delta[:, :, 0]) - facing[:, None]
    sector = np.mod(bearing + np.pi, 2 * np.pi) * (N_SECTORS / (2 * np.pi))
    sector = np.minimum(sector.astype(np.int64), N_SECTORS - 1)

    # One flat bucket per (shark, sector); reduce the visible pairs into them.
    flat = (np.arange(s)[:, None] * N_SECTORS + sector)[visible]
    nearest = np.full(s * N_SECTORS, np.inf)
    np.minimum.at(nearest, flat, np.sqrt(d2[visible]))
    count = np.bincount(flat, minlength=s * N_SECTORS).astype(np.float64)

    closeness = 1.0 - nearest / np.repeat(perception, N_SECTORS)
    closeness = np.where(np.isfinite(nearest), np.clip(closeness, 0.0, 1.0), 0.0)
    retina[:, :N_SECTORS] = closeness.reshape(s, N_SECTORS)
    retina[:, N_SECTORS:] = np.minimum(count / count_norm, 1.0).reshape(s, N_SECTORS)
    return retina


def forward(
    spec: BrainSpec, brains: np.ndarray, obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run every shark's own network on its own observation, in one batch.

    ``brains`` ``(S, spec.weight_count)``, ``obs`` ``(S, spec.n_inputs)`` →
    ``(turn, throttle)``, each ``(S,)``: ``turn`` in ``[-1, 1]`` (tanh) is a
    fraction of the max turn rate, ``throttle`` in ``[0, 1]`` (sigmoid) picks a
    speed between the shark's min- and max-speed genes. Pure and deterministic;
    the caller applies both scalings.
    """
    s = len(brains)
    x = obs
    last = len(spec.layers) - 1
    for i, (offset, rows, cols) in enumerate(spec.layers):
        if i > 0:  # the bias constant for every layer past the input
            x = np.concatenate([x, np.ones((s, 1))], axis=1)
        w = brains[:, offset : offset + rows * cols].reshape(s, rows, cols)
        x = np.einsum("si,sio->so", x, w)
        if i < last:
            x = np.tanh(x)
    turn = np.tanh(x[:, 0])
    throttle = 1.0 / (1.0 + np.exp(-x[:, 1]))
    return turn, throttle
